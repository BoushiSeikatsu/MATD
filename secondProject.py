# Soubor: transformer_summarizer.py

# --------------------------------------------------------------------------
# Importy
# --------------------------------------------------------------------------
from datasets import load_dataset
from transformers import AutoTokenizer
from functools import partial # Pro vytváření funkcí s přednastavenými argumenty (použito pro collate_fn)
import torch
import torch.nn as nn
import math # Pro matematické funkce jako sqrt, log, exp
from torch.utils.data import DataLoader, Dataset as TorchDataset # Pro práci s daty v PyTorchi
import torch.optim as optim # Pro optimalizátory (AdamW)
import time # Pro měření času (např. trvání epochy)
import os   # Pro práci se souborovým systémem (kontrola existence souborů, ukládání modelu)
from tqdm.auto import tqdm # Pro zobrazení progress barů

# Importy pro ROUGE evaluaci
from rouge_score import rouge_scorer
import nltk # NLTK je závislost pro rouge_scorer, používá se pro tokenizaci
try:
    nltk.data.find('tokenizers/punkt') # Zkontroluje, zda je stažen potřebný NLTK balíček
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True) # Pokud není, stáhne ho


# --------------------------------------------------------------------------
# Globální konstanty a konfigurace (mohou být upraveny)
# --------------------------------------------------------------------------
MODEL_CHECKPOINT = "t5-small"   # Předtrénovaný model/tokenizer z Hugging Face Hubu
MAX_INPUT_LENGTH = 256          # Maximální délka vstupní sekvence (dialogu) v tokenech
MAX_TARGET_LENGTH = 64          # Maximální délka cílové sekvence (shrnutí) v tokenech
PREFIX = "summarize: "          # Prefix přidávaný k vstupním dialogům (specifické pro T5 pro sumarizaci)

# Výchozí hodnoty pro ID speciálních tokenů (budou přepsány hodnotami z tokenizeru)
DEFAULT_PAD_TOKEN_ID = 0
DEFAULT_EOS_TOKEN_ID = 1

# Hyperparametry modelu (pro menší model, lze upravit pro větší/menší)
# Tyto hodnoty se použijí, pokud se nenačtou jiné (např. při prvním spuštění)
# a slouží i jako reference pro případné experimentování.
_EMB_SIZE = 128                 # Dimenze embeddingů a vnitřní dimenze Transformeru (d_model)
_N_HEAD = 4                     # Počet attention hlav (musí dělit EMB_SIZE)
_FFN_HID_DIM = 256              # Dimenze skryté vrstvy v FeedForward síti (často 2x-4x EMB_SIZE)
_NUM_ENCODER_LAYERS = 2         # Počet vrstev v enkodéru
_NUM_DECODER_LAYERS = 2         # Počet vrstev v dekodéru
_DROPOUT = 0.1                  # Dropout rate

# Hyperparametry tréninku
_LEARNING_RATE = 0.0005         # Rychlost učení pro optimalizátor
_NUM_EPOCHS = 3                 # Počet tréninkových epoch (pro ukázku nízký)
_BATCH_SIZE = 16                # Velikost dávky
_CLIP_VALUE = 1.0               # Hodnota pro ořezávání gradientů (gradient clipping)
_PATIENCE = 2                   # Počet epoch bez zlepšení validačního lossu před early stopping

# Cesta pro uložení nejlepšího modelu
MODEL_SAVE_PATH = "transformer_summarizer_best.pt"


# --------------------------------------------------------------------------
# Část 1 a 2: Příprava dat a Tokenizace (funkce)
# --------------------------------------------------------------------------

def load_and_prepare_samsum_data():
    """
    Stáhne Samsum dataset z Hugging Face Hubu a rozdělí ho na
    trénovací, validační a testovací množiny.
    Testovací množina je vrácena v "surové" podobě (jako objekt Dataset z HF),
    aby obsahovala původní textové sloupce pro finální evaluaci.
    """
    print("\n--- Stahování a příprava Samsum datasetu ---")
    try:
        # Načtení datasetu s povolením vzdáleného kódu, který je pro Samsum vyžadován
        train_dataset = load_dataset("samsum", split="train", trust_remote_code=True)
        validation_dataset = load_dataset("samsum", split="validation", trust_remote_code=True)
        test_dataset_raw = load_dataset("samsum", split="test", trust_remote_code=True)
        print(f"Načteno {len(train_dataset)} trénovacích, {len(validation_dataset)} validačních, {len(test_dataset_raw)} testovacích příkladů.")
        return train_dataset, validation_dataset, test_dataset_raw
    except Exception as e:
        print(f"Nastala chyba při stahování datasetu: {e}")
        return None, None, None

def tokenize_data(datasets_dict, tokenizer, keep_original_cols=False):
    """
    Tokenizuje vstupní dialogy a cílová shrnutí pro dané datasety.

    Args:
        datasets_dict (dict): Slovník, kde klíče jsou názvy splitů ('train', 'val', 'test')
                              a hodnoty jsou odpovídající Hugging Face Dataset objekty.
        tokenizer (PreTrainedTokenizer): Instance tokenizeru.
        keep_original_cols (bool): Pokud True, pokusí se zachovat sloupce 'dialogue' a 'summary'
                                   jako 'dialogue_original' a 'summary_original' v tokenizovaném datasetu.
                                   Primárně určeno pro testovací sadu pro finální evaluaci.
    Returns:
        tuple: Trojice tokenizovaných datasetů (train, val, test). Pokud některý vstupní
               dataset chyběl, odpovídající hodnota v tuple bude None.
    """
    print(f"\n--- Tokenizace dat pomocí tokenizeru: {tokenizer.name_or_path} ---")

    def preprocess_function(examples, current_keep_original_for_split):
        """
        Pomocná funkce pro tokenizaci jednoho batche příkladů.
        Je volána pomocí `dataset.map()`.

        Args:
            examples (dict): Batch dat z datasetu, typicky slovník sloupců.
            current_keep_original_for_split (bool): Zda se mají pro tento konkrétní split zachovat originály.
        """
        model_inputs = {} # Slovník pro shromažďování výsledků tokenizace

        # Pokud máme zachovat původní texty, zkopírujeme je do nových sloupců
        if current_keep_original_for_split:
            model_inputs["dialogue_original"] = examples["dialogue"]
            model_inputs["summary_original"] = examples["summary"]

        # Příprava vstupů pro enkodér (dialogy)
        inputs = [PREFIX + doc for doc in examples["dialogue"]] # Přidání prefixu
        tokenized_inputs = tokenizer(inputs,
                                     max_length=MAX_INPUT_LENGTH,
                                     truncation=True,       # Oříznutí delších sekvencí
                                     padding="max_length")  # Doplnění kratších sekvencí na max_length
        model_inputs.update(tokenized_inputs) # Přidání 'input_ids' a 'attention_mask'

        # Příprava vstupů pro dekodér (shrnutí jakožto labely)
        labels = tokenizer(text_target=examples["summary"], # text_target pro cílové texty u T5
                           max_length=MAX_TARGET_LENGTH,
                           truncation=True,
                           padding="max_length")
        model_inputs["labels"] = labels["input_ids"] # Uložíme pouze input_ids jako labely

        return model_inputs

    tokenized_datasets = {} # Slovník pro uložení tokenizovaných datasetů
    for split_name, dataset in datasets_dict.items():
        if dataset:
            print(f"Tokenizuji {split_name} data...")
            try:
                import multiprocessing
                num_cpus = multiprocessing.cpu_count()
            except ImportError:
                num_cpus = 1 # Fallback pro systémy bez multiprocessing
            
            # Určíme, zda pro aktuální split (zejména 'test') chceme zachovat originální sloupce
            should_keep_originals_for_this_split = keep_original_cols and (split_name == 'test')
            
            # Určíme sloupce k odstranění po tokenizaci
            # Pokud nezachováváme originály, odstraníme textové sloupce.
            # Pokud zachováváme, odstraníme jen 'id' a původní 'dialogue' a 'summary'
            # budou nahrazeny 'dialogue_original' a 'summary_original' (nepřímo, map je přepíše).
            # Správněji: `remove_columns` odstraní sloupce, které už nejsou potřeba.
            # Pokud jsme vytvořili 'dialogue_original', původní 'dialogue' může být odstraněn.
            # Efektivně chceme skončit jen s 'input_ids', 'attention_mask', 'labels'
            # a případně 'dialogue_original', 'summary_original'.
            # `dataset.column_names` zahrne i nově přidané sloupce z `preprocess_function`
            # před odstraněním. Proto je lepší specifikovat sloupce k odstranění explicitně.
            columns_to_remove_after_map = list(dataset.column_names)


            tokenized_datasets[split_name] = dataset.map(
                # Použijeme partial k předání extra argumentu `current_keep_original_for_split`
                partial(preprocess_function, current_keep_original_for_split=should_keep_originals_for_this_split),
                batched=True, # Zpracování v dávkách pro rychlost
                num_proc=max(1, num_cpus // 2 if num_cpus > 1 else 1), # Využití více jader CPU
                remove_columns=columns_to_remove_after_map # Odstranění původních textových sloupců a 'id'
            )
            print(f"Tokenizace {split_name} dat dokončena.")
            if should_keep_originals_for_this_split:
                 print(f"  Původní sloupce 'dialogue_original' a 'summary_original' byly zachovány pro {split_name}.")
                 # Ověření, že sloupce skutečně existují
                 # print(f"  Dostupné sloupce v tokenizovaném {split_name}: {tokenized_datasets[split_name].column_names}")
        else:
            tokenized_datasets[split_name] = None # Pokud vstupní dataset nebyl poskytnut

    return tokenized_datasets.get("train"), tokenized_datasets.get("val"), tokenized_datasets.get("test")


# --------------------------------------------------------------------------
# Část 3: Definice modelu Transformer
# --------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """
    Implementace pozičního kódování, jak je popsáno v "Attention Is All You Need".
    Přidává informaci o pozici tokenů do jejich embeddingů.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Args:
            d_model (int): Dimenze embeddingu (stejná jako v Transformeru).
            dropout (float): Dropout rate.
            max_len (int): Maximální délka sekvence, pro kterou se předpočítá poziční kódování.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Výpočet pozičního kódování
        position = torch.arange(max_len).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model) # Shape pro broadcast s [seq_len, batch_size, d_model]
        pe[:, 0, 0::2] = torch.sin(position * div_term) # Sudé indexy
        pe[:, 0, 1::2] = torch.cos(position * div_term) # Liché indexy
        
        # register_buffer uloží 'pe' jako součást stavu modulu, ale ne jako trénovatelný parametr
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Přidá poziční kódování k vstupnímu tenzoru.
        Args:
            x (torch.Tensor): Vstupní tenzor embeddingů, shape [seq_len, batch_size, embedding_dim]
        Returns:
            torch.Tensor: Tenzor s přidaným pozičním kódováním.
        """
        # x.size(0) je aktuální délka sekvence
        x = x + self.pe[:x.size(0)] # Přidáme odpovídající část předpočítaného PE
        return self.dropout(x)

class TokenEmbedding(nn.Module):
    """
    Vrstva pro token embedding. Zahrnuje nn.Embedding a škálování embeddingů
    faktorem sqrt(d_model), jak je doporučeno v "Attention Is All You Need".
    """
    def __init__(self, vocab_size: int, emb_size: int):
        """
        Args:
            vocab_size (int): Velikost slovníku.
            emb_size (int): Cílová dimenze embeddingu (d_model).
        """
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Provede embedding vstupních tokenů.
        Args:
            tokens (torch.Tensor): Tenzor token ID.
        Returns:
            torch.Tensor: Tenzor embeddingů.
        """
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size) # Škálování

class Seq2SeqTransformer(nn.Module):
    """
    Kompletní Transformer model architektury Encoder-Decoder pro úlohy sekvence-na-sekvenci.
    Využívá vestavěnou třídu `nn.Transformer` z PyTorche.
    """
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int, emb_size: int,
                 n_head: int, src_vocab_size: int, tgt_vocab_size: int,
                 dim_feedforward: int, dropout: float = 0.1,
                 max_seq_len: int = MAX_INPUT_LENGTH): # max_seq_len pro PositionalEncoding
        """
        Args:
            num_encoder_layers (int): Počet vrstev v enkodéru.
            num_decoder_layers (int): Počet vrstev v dekodéru.
            emb_size (int): Dimenze embeddingů a d_model Transformeru.
            n_head (int): Počet hlav v MultiHeadAttention. `emb_size` musí být dělitelné `n_head`.
            src_vocab_size (int): Velikost slovníku zdrojového jazyka (vstup).
            tgt_vocab_size (int): Velikost slovníku cílového jazyka (výstup).
            dim_feedforward (int): Dimenze skryté vrstvy ve FeedForward sítích uvnitř Transformeru.
            dropout (float): Dropout rate.
            max_seq_len (int): Maximální délka sekvence pro PositionalEncoding.
                               Měla by být větší nebo rovna nejdelší možné sekvenci.
        """
        super(Seq2SeqTransformer, self).__init__()

        # Vrstvy pro embedding a poziční kódování
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        # max_seq_len by měla být dostatečně velká pro nejdelší možnou sekvenci (vstupní nebo výstupní)
        _model_max_pe_len = max(MAX_INPUT_LENGTH, MAX_TARGET_LENGTH, max_seq_len) + 10 # S rezervou
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout, max_len=_model_max_pe_len)

        # Jádro Transformeru z PyTorche
        # Důležité: batch_first=False (default) znamená, že vstupy mají být [seq_len, batch_size, emb_size]
        self.transformer = nn.Transformer(d_model=emb_size,
                                          nhead=n_head,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                          batch_first=False) # Výchozí, explicitně pro přehlednost

        # Výstupní lineární vrstva pro predikci tokenů z cílového slovníku
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        
        # Inicializace vah modelu
        self.init_weights()

    def init_weights(self) -> None:
        """Inicializuje váhy embeddingových vrstev a generátoru."""
        initrange = 0.1 # Rozsah pro rovnoměrné rozdělení
        self.src_tok_emb.embedding.weight.data.uniform_(-initrange, initrange)
        self.tgt_tok_emb.embedding.weight.data.uniform_(-initrange, initrange)
        self.generator.bias.data.zero_() # Nulování biasu generátoru
        self.generator.weight.data.uniform_(-initrange, initrange) # Inicializace vah generátoru

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """
        Generuje čtvercovou horní trojúhelníkovou masku pro self-attention v dekodéru.
        Tato "kauzální" maska zabraňuje dekodéru nahlížet na budoucí tokeny
        v cílové sekvenci během tréninku i inference.
        Maska má hodnoty 0.0 tam, kde je pozornost povolena, a -inf tam, kde je zakázána.
        Shape: [sz, sz]
        """
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _create_padding_mask(self, sequence: torch.Tensor, pad_idx: int) -> torch.Tensor:
        """
        Vytvoří padding masku pro danou sekvenci.
        `nn.Transformer` očekává boolean masku, kde `True` značí pozice, které mají být ignorovány (padding).
        Args:
            sequence (torch.Tensor): Vstupní sekvence token ID, shape [seq_len, batch_size].
            pad_idx (int): ID padding tokenu.
        Returns:
            torch.Tensor: Boolean padding maska, shape [batch_size, seq_len].
        """
        # sequence.transpose(0, 1) má tvar [batch_size, seq_len]
        mask = (sequence == pad_idx).transpose(0, 1)
        return mask

    def forward(self,
                src: torch.Tensor, # Vstupní sekvence (zdroj), shape: [src_seq_len, batch_size]
                tgt: torch.Tensor, # Cílová sekvence (vstup do dekodéru), shape: [tgt_seq_len, batch_size]
                src_pad_idx: int,  # ID padding tokenu pro zdrojovou sekvenci
                tgt_pad_idx: int   # ID padding tokenu pro cílovou sekvenci
               ) -> torch.Tensor: # Výstupní logity, shape: [tgt_seq_len, batch_size, tgt_vocab_size]
        """
        Provede forward pass celým modelem Seq2SeqTransformer.
        """
        device = src.device # Zařízení, na kterém se provádějí výpočty (CPU/GPU)

        # 1. Vytvoření masek
        tgt_seq_len = tgt.shape[0]
        # Kauzální maska pro self-attention v dekodéru
        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len, device) # Shape: [tgt_seq_len, tgt_seq_len]

        # Padding masky
        src_padding_mask = self._create_padding_mask(src, src_pad_idx) # Shape: [batch_size, src_seq_len]
        tgt_padding_mask = self._create_padding_mask(tgt, tgt_pad_idx) # Shape: [batch_size, tgt_seq_len]
        # memory_key_padding_mask pro cross-attention v dekodéru je stejná jako src_padding_mask,
        # protože se vztahuje k výstupu enkodéru (memory), který má stejný padding jako původní src.

        # 2. Embedding a poziční kódování
        # Zdrojová sekvence
        src_emb = self.positional_encoding(self.src_tok_emb(src)) # Shape: [src_seq_len, batch_size, emb_size]
        # Cílová sekvence (pro vstup dekodéru)
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt)) # Shape: [tgt_seq_len, batch_size, emb_size]

        # 3. Průchod Transformerem
        # nn.Transformer očekává:
        # src: (S, N, E) - zdrojová sekvence
        # tgt: (T, N, E) - cílová sekvence (vstup dekodéru)
        # src_mask: (S, S) - maska pro enkodér self-attention (typicky None, pokud neomezujeme "dohled")
        # tgt_mask: (T, T) - kauzální maska pro dekodér self-attention
        # memory_mask: (T, S) - maska pro cross-attention (typicky None)
        # src_key_padding_mask: (N, S) - padding maska pro src
        # tgt_key_padding_mask: (N, T) - padding maska pro tgt (pro dekodér self-attention)
        # memory_key_padding_mask: (N, S) - padding maska pro memory (výstup enkodéru, pro cross-attention)
        output = self.transformer(src=src_emb,
                                  tgt=tgt_emb,
                                  src_mask=None, # Enkodér vidí celou vstupní sekvenci
                                  tgt_mask=tgt_mask,
                                  memory_mask=None,
                                  src_key_padding_mask=src_padding_mask,
                                  tgt_key_padding_mask=tgt_padding_mask,
                                  memory_key_padding_mask=src_padding_mask) # Důležité: použít src_padding_mask zde

        # 4. Výstupní lineární vrstva (generátor) pro predikci tokenů
        return self.generator(output)

# --------------------------------------------------------------------------
# PyTorch Dataset a DataLoader
# --------------------------------------------------------------------------

class SummarizationTorchDataset(TorchDataset):
    """
    Vlastní PyTorch Dataset třída pro přípravu dat pro sumarizační model.
    Převádí tokenizovaná data z Hugging Face Dataset na PyTorch tenzory
    a připravuje vstupy pro enkodér a dekodér.
    """
    def __init__(self, hf_dataset, tokenizer):
        """
        Args:
            hf_dataset (datasets.Dataset): Tokenizovaný Hugging Face dataset.
                                           Očekává se, že obsahuje sloupce 'input_ids' a 'labels'.
            tokenizer (PreTrainedTokenizer): Instance tokenizeru, používá se pro `pad_token_id`.
        """
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id # BOS token pro dekodér a padding

    def __len__(self):
        """Vrací počet příkladů v datasetu."""
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        """
        Vrací jeden zpracovaný příklad z datasetu.

        Args:
            idx (int): Index příkladu.
        Returns:
            dict: Slovník obsahující:
                'input_ids': Tokeny vstupního dialogu (pro enkodér).
                'decoder_input_ids': Tokeny pro vstup dekodéru (posunuté shrnutí s BOS).
                'target_labels': Cílové tokeny pro výpočet loss (původní shrnutí).
                'original_dialogue' (volitelně): Původní text dialogu.
                'original_summary' (volitelně): Původní text shrnutí.
        """
        item = self.hf_dataset[idx] # Načtení jednoho příkladu (slovník)

        # Vstup pro enkodér
        input_ids = torch.tensor(item['input_ids'], dtype=torch.long)
        
        # Cílová sekvence (labels) pro výpočet loss
        # `item['labels']` je již tokenizované shrnutí (např. [t1, t2, ..., tn, EOS, PAD, ...])
        labels_list = item['labels']
        target_labels = torch.tensor(labels_list, dtype=torch.long)

        # Vstup pro dekodér (`tgt` pro nn.Transformer)
        # Musí být `[BOS, t1, t2, ..., tn-1, EOS, PAD, ...]` (tj. posunuté o jeden doprava)
        # Pro T5 styl modelů (a náš tokenizer) je BOS token často `pad_token_id`.
        decoder_input_list = [self.pad_token_id] + labels_list[:-1] # Přidá BOS a odstraní poslední token z labels
        decoder_input_ids = torch.tensor(decoder_input_list, dtype=torch.long)

        # Připravení výsledného slovníku
        result = {
            "input_ids": input_ids,
            "decoder_input_ids": decoder_input_ids,
            "target_labels": target_labels
        }
        # Pokud jsou v datasetu původní texty, přidáme je také
        if "dialogue_original" in item:
            result["original_dialogue"] = item["dialogue_original"]
        if "summary_original" in item:
            result["original_summary"] = item["summary_original"]
            
        return result

def collate_fn_pytorch(batch_list, pad_token_id_value): # pad_token_id_value se zde nepoužívá, pokud je padding už hotov
    """
    Collate funkce pro DataLoader. Seskupuje list slovníků (jednotlivé příklady)
    do jednoho batche.
    Předpokládá, že tokeny jsou již paddované na stejnou délku v `preprocess_function`.
    Transponuje tenzory tak, aby odpovídaly formátu `[seq_len, batch_size]`
    očekávanému `nn.Transformer(batch_first=False)`.

    Args:
        batch_list (list): List slovníků, kde každý slovník je výstupem `SummarizationTorchDataset.__getitem__`.
        pad_token_id_value (int): ID padding tokenu (zde se explicitně nepoužívá pro padding,
                                   ale je předáváno pro konzistenci).
    Returns:
        dict: Slovník obsahující batchované tenzory.
    """
    # Extrakce jednotlivých typů dat z listu batchů
    input_ids_list = [item['input_ids'] for item in batch_list]
    decoder_input_ids_list = [item['decoder_input_ids'] for item in batch_list]
    target_labels_list = [item['target_labels'] for item in batch_list]
    
    # Extrakce originálních textů, pokud jsou přítomny
    original_dialogues = [item.get('original_dialogue', "") for item in batch_list]
    original_summaries = [item.get('original_summary', "") for item in batch_list]

    # Stackování listů tenzorů do jednoho batch tenzoru
    # Již jsou paddované na MAX_..._LENGTH v `tokenize_data`, takže `torch.stack` je přímočarý.
    input_ids_pt = torch.stack(input_ids_list)                 # Shape: [batch_size, src_seq_len]
    decoder_input_ids_pt = torch.stack(decoder_input_ids_list) # Shape: [batch_size, tgt_seq_len]
    target_labels_pt = torch.stack(target_labels_list)         # Shape: [batch_size, tgt_seq_len]

    # nn.Transformer (s batch_first=False) očekává [seq_len, batch_size]
    # Proto transponujeme první dvě dimenze.
    return {
        "input_ids": input_ids_pt.transpose(0, 1),
        "decoder_input_ids": decoder_input_ids_pt.transpose(0, 1),
        "target_labels": target_labels_pt.transpose(0, 1),
        "original_dialogues": original_dialogues, # Tyto zůstávají jako listy stringů
        "original_summaries": original_summaries
    }

# ... (pokračování souboru transformer_summarizer.py) ...

# --------------------------------------------------------------------------
# Část 4: Trénink a Validace modelu
# --------------------------------------------------------------------------

def train_epoch(model: Seq2SeqTransformer, # Typová nápověda pro model
                optimizer: optim.Optimizer, # Typová nápověda pro optimalizátor
                criterion: nn.Module,       # Typová nápověda pro loss funkci (např. CrossEntropyLoss)
                train_dataloader: DataLoader,
                device: torch.device,
                pad_idx: int,               # ID padding tokenu pro ignorování v loss funkci
                current_epoch: int,         # Aktuální číslo epochy (pro zobrazení)
                total_epochs: int,          # Celkový počet epoch (pro zobrazení)
                clip_value: float = 1.0     # Hodnota pro gradient clipping
               ) -> float:                  # Vrací průměrný tréninkový loss za epochu
    """
    Provede jednu epochu tréninku modelu.

    Args:
        model: Instance modelu Seq2SeqTransformer.
        optimizer: Optimalizátor (např. AdamW).
        criterion: Loss funkce (např. CrossEntropyLoss).
        train_dataloader: DataLoader pro trénovací data.
        device: Zařízení (CPU/GPU), na kterém se má trénovat.
        pad_idx: ID padding tokenu, které má být ignorováno loss funkcí.
        current_epoch: Aktuální číslo epochy.
        total_epochs: Celkový počet tréninkových epoch.
        clip_value: Maximální norma gradientů pro gradient clipping.
    Returns:
        float: Průměrný tréninkový loss za tuto epochu.
    """
    model.train() # Nastavení modelu do trénovacího režimu (aktivuje dropout, atd.)
    epoch_loss = 0.0 # Kumulativní loss za epochu

    # Progress bar pro sledování průběhu epochy
    progress_bar = tqdm(train_dataloader,
                        desc=f"Epocha {current_epoch}/{total_epochs} [Trénink]",
                        leave=False, # Nezanechá bar po dokončení (bude nahrazen souhrnným výpisem)
                        unit="batch")

    for batch in progress_bar:
        # Přesun dat na správné zařízení
        src = batch['input_ids'].to(device)               # Vstup enkodéru
        tgt_input = batch['decoder_input_ids'].to(device) # Vstup dekodéru (posunutý cíl)
        tgt_output = batch['target_labels'].to(device)    # Cílové labely pro loss

        optimizer.zero_grad() # Vynulování gradientů z předchozího kroku

        # Forward pass: získání logitů z modelu
        # Logits shape: [tgt_seq_len, batch_size, tgt_vocab_size]
        logits = model(src=src, tgt=tgt_input, src_pad_idx=pad_idx, tgt_pad_idx=pad_idx)

        # Příprava logitů a cílů pro CrossEntropyLoss
        # CrossEntropyLoss očekává:
        #   Input: (N, C) = (batch_size * tgt_seq_len, tgt_vocab_size)
        #   Target: (N) = (batch_size * tgt_seq_len)
        # Proto musíme "rozbalit" první dvě dimenze (seq_len, batch_size).
        logits_flat = logits.reshape(-1, logits.shape[-1]) # Shape: [seq_len * batch, vocab_size]
        tgt_output_flat = tgt_output.reshape(-1)           # Shape: [seq_len * batch]

        loss = criterion(logits_flat, tgt_output_flat) # Výpočet loss
        loss.backward() # Backward pass: výpočet gradientů

        # Gradient clipping: ořezání gradientů, aby se zabránilo jejich explozi
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step() # Aktualizace vah modelu

        current_batch_loss = loss.item() # Získání hodnoty loss jako Python float
        epoch_loss += current_batch_loss

        # Aktualizace popisu progress baru s aktuálním lossem
        progress_bar.set_postfix(loss=f"{current_batch_loss:.4f}")

    progress_bar.close() # Uzavření progress baru
    return epoch_loss / len(train_dataloader) # Průměrný loss za epochu

def evaluate(model: Seq2SeqTransformer,
             criterion: nn.Module,
             val_dataloader: DataLoader,
             device: torch.device,
             pad_idx: int,
             current_epoch: int, # Pro konzistentní popis v progress baru
             total_epochs: int
            ) -> float: # Vrací průměrný validační loss
    """
    Provede evaluaci modelu na validačních datech.

    Args:
        model: Instance modelu Seq2SeqTransformer.
        criterion: Loss funkce.
        val_dataloader: DataLoader pro validační data.
        device: Zařízení (CPU/GPU).
        pad_idx: ID padding tokenu pro ignorování v loss funkci.
        current_epoch: Aktuální číslo epochy (pro popis progress baru).
        total_epochs: Celkový počet epoch (pro popis progress baru).
    Returns:
        float: Průměrný validační loss.
    """
    model.eval() # Nastavení modelu do evaluačního režimu (deaktivuje dropout, atd.)
    epoch_loss = 0.0

    progress_bar = tqdm(val_dataloader,
                        desc=f"Epocha {current_epoch}/{total_epochs} [Validace]",
                        leave=False,
                        unit="batch")

    with torch.no_grad(): # Vypnutí výpočtu gradientů během evaluace (šetří paměť a čas)
        for batch in progress_bar:
            src = batch['input_ids'].to(device)
            tgt_input = batch['decoder_input_ids'].to(device)
            tgt_output = batch['target_labels'].to(device)

            logits = model(src=src, tgt=tgt_input, src_pad_idx=pad_idx, tgt_pad_idx=pad_idx)

            logits_flat = logits.reshape(-1, logits.shape[-1])
            tgt_output_flat = tgt_output.reshape(-1)

            loss = criterion(logits_flat, tgt_output_flat)
            current_batch_loss = loss.item()
            epoch_loss += current_batch_loss
            progress_bar.set_postfix(loss=f"{current_batch_loss:.4f}")

    progress_bar.close()
    return epoch_loss / len(val_dataloader)


# --------------------------------------------------------------------------
# Část 5: Inference (Generování shrnutí) a Evaluace (ROUGE)
# --------------------------------------------------------------------------

def greedy_decode_sentence(model: Seq2SeqTransformer,
                           tokenizer: AutoTokenizer, # Typová nápověda pro tokenizer
                           sentence: str,            # Vstupní dialog jako text
                           device: torch.device,
                           max_len: int = MAX_TARGET_LENGTH # Max. délka generovaného shrnutí
                          ) -> str: # Vrací vygenerované shrnutí jako text
    """
    Generuje shrnutí pro jeden vstupní dialog pomocí strategie greedy decoding.

    Args:
        model: Natrénovaný model Seq2SeqTransformer.
        tokenizer: Instance tokenizeru použitá pro trénink.
        sentence: Vstupní dialog (text).
        device: Zařízení (CPU/GPU), na kterém se má provést inference.
        max_len: Maximální délka generovaného shrnutí v tokenech.
    Returns:
        str: Vygenerovaný text shrnutí.
    """
    model.eval() # Ujistíme se, že model je v evaluačním režimu

    # 1. Tokenizace vstupního dialogu
    src_text_with_prefix = PREFIX + sentence # Přidání prefixu
    # Tokenizace s paddingem na MAX_INPUT_LENGTH, aby vstup měl konzistentní tvar pro model
    src_tokenized = tokenizer(src_text_with_prefix,
                              max_length=MAX_INPUT_LENGTH,
                              truncation=True,
                              padding="max_length", # Důležité pro konzistentní tvar `src`
                              return_tensors="pt")  # Vrátí PyTorch tenzory
    
    src = src_tokenized["input_ids"].to(device) # Shape: [1, src_seq_len]
    # Model očekává vstup ve formátu [seq_len, batch_size], takže transponujeme
    src = src.transpose(0, 1) # Shape: [src_seq_len, 1 (batch_size=1)]

    # Získání ID speciálních tokenů z tokenizeru
    current_pad_idx = tokenizer.pad_token_id
    current_eos_idx = tokenizer.eos_token_id
    
    # 2. Enkódování vstupní sekvence (pouze jednou na začátku)
    with torch.no_grad(): # Inference bez výpočtu gradientů
        # Vytvoření padding masky pro enkodér
        # Tato maska je pro `src_key_padding_mask` v `transformer.encoder`
        src_key_padding_mask_for_encoder = model._create_padding_mask(src, current_pad_idx) # Shape: [1, src_seq_len]
        
        # Embedding a poziční kódování vstupu
        src_emb = model.positional_encoding(model.src_tok_emb(src))
        # Průchod enkodérem
        memory = model.transformer.encoder(src_emb, mask=None, src_key_padding_mask=src_key_padding_mask_for_encoder)
        # `memory` (výstup enkodéru) shape: [src_seq_len, 1, emb_size]

    # 3. Iterativní dekódování
    # Začneme s BOS tokenem (pro T5 je to pad_token_id)
    generated_ids = [current_pad_idx] # Seznam pro ukládání vygenerovaných ID tokenů

    for _ in range(max_len - 1): # Generujeme maximálně (max_len - 1) dalších tokenů (BOS už tam je)
        # Převod dosud vygenerovaných ID na tenzor pro vstup dekodéru
        tgt_tensor = torch.LongTensor(generated_ids).unsqueeze(1).to(device) # Shape: [current_tgt_len, 1]

        with torch.no_grad():
            # Příprava masek pro dekodér
            # Kauzální maska (aby dekodér neviděl budoucí tokeny)
            tgt_self_attention_mask = model._generate_square_subsequent_mask(tgt_tensor.size(0), device) # Shape: [len, len]
            # Padding maska pro vstup dekodéru (pokud by obsahoval padding, což v greedy typicky ne)
            tgt_padding_mask_for_decoder_self_attn = model._create_padding_mask(tgt_tensor, current_pad_idx) # Shape: [1, len]

            # Embedding a poziční kódování pro vstup dekodéru
            tgt_emb = model.positional_encoding(model.tgt_tok_emb(tgt_tensor))
            
            # Průchod dekodérem
            # `memory_key_padding_mask` se vztahuje k `memory` (výstupu enkodéru)
            # a je tedy stejná jako `src_key_padding_mask_for_encoder`
            output_decoder = model.transformer.decoder(tgt_emb, memory,
                                                       tgt_mask=tgt_self_attention_mask,
                                                       memory_mask=None, # Pro cross-attention, typicky None
                                                       tgt_key_padding_mask=tgt_padding_mask_for_decoder_self_attn,
                                                       memory_key_padding_mask=src_key_padding_mask_for_encoder)
            # `output_decoder` shape: [current_tgt_len, 1, emb_size]

            # Získání predikce pro *následující* token (bereme výstup pro poslední token v `tgt_tensor`)
            prediction_logits = model.generator(output_decoder[-1, :, :]) # Shape: [1, vocab_size]
        
        # Greedy výběr: vezmeme token s nejvyšší pravděpodobností
        predicted_id = prediction_logits.argmax(1).item()
        generated_ids.append(predicted_id) # Přidání vybraného tokenu

        # Ukončení, pokud je vygenerován EOS token
        if predicted_id == current_eos_idx:
            break
    
    # 4. Dekódování vygenerovaných ID na text
    # Přeskočíme první token (BOS), pokud to byl pad_token_id, jinak by ho tokenizer.decode mohl zahrnout.
    # `tokenizer.decode` se postará o `skip_special_tokens=True` (včetně EOS, PAD).
    start_index_for_decode = 1 if generated_ids[0] == current_pad_idx and len(generated_ids) > 1 else 0
    summary = tokenizer.decode(generated_ids[start_index_for_decode:],
                               skip_special_tokens=True,
                               clean_up_tokenization_spaces=True) # Odstraní nadbytečné mezery
    return summary


def calculate_rouge_scores(predictions: list[str], references: list[str]) -> dict:
    """
    Vypočítá průměrné ROUGE-1 a ROUGE-2 F1 skóre pro seznam predikcí a referencí.

    Args:
        predictions (list[str]): Seznam vygenerovaných shrnutí.
        references (list[str]): Seznam referenčních (lidských) shrnutí.
    Returns:
        dict: Slovník s klíči 'rouge1' a 'rouge2' a jejich průměrnými F1 hodnotami.
    """
    # Inicializace ROUGE scoreru
    # 'rouge1': unigram overlap
    # 'rouge2': bigram overlap
    # 'use_stemmer=True': Použije stemming (např. Porter stemmer) pro porovnání slov
    scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    
    r1_fscores = []
    r2_fscores = []
    
    # Pojistka pro případ, že by se délky neshodovaly (nemělo by nastat při správném použití)
    if len(predictions) != len(references):
        print(f"Varování: Počet predikcí ({len(predictions)}) se neshoduje s počtem referencí ({len(references)}).")
        min_len = min(len(predictions), len(references))
        predictions = predictions[:min_len]
        references = references[:min_len]

    for pred, ref in zip(predictions, references):
        # Přeskočení, pokud je některý z textů prázdný (může zkreslit skóre)
        if not pred.strip() or not ref.strip():
            # print(f"Varování: Prázdná predikce nebo reference. Pred: '{pred}', Ref: '{ref}'")
            continue # Tento pár se do výpočtu nezahrne
        
        # Výpočet skóre pro aktuální pár predikce-reference
        scores = scorer_instance.score(target=ref, prediction=pred) # pořadí target, prediction
        r1_fscores.append(scores['rouge1'].fmeasure) # Ukládáme F1-score
        r2_fscores.append(scores['rouge2'].fmeasure)
        
    # Pokud nebyly žádné platné páry pro výpočet (např. všechny byly prázdné)
    if not r1_fscores or not r2_fscores:
        return {"rouge1": 0.0, "rouge2": 0.0}

    # Výpočet průměrných F1 skóre
    avg_r1 = sum(r1_fscores) / len(r1_fscores)
    avg_r2 = sum(r2_fscores) / len(r2_fscores)
    
    return {"rouge1": avg_r1, "rouge2": avg_r2}


# --------------------------------------------------------------------------
# Hlavní část skriptu (__main__)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("Začínáme s implementací Transformeru pro sumarizaci dialogů.")
    # Nastavení zařízení (GPU, pokud je k dispozici, jinak CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Používám zařízení: {device}")

    # --- Část 1 & 2: Načtení a Tokenizace ---
    print("\n--- Část 1 & 2: Načtení a Tokenizace ---")
    samsum_train_raw, samsum_val_raw, samsum_test_raw_original = load_and_prepare_samsum_data()

    # ID speciálních tokenů budou definována lokálně v main po načtení tokenizeru
    local_pad_token_id = DEFAULT_PAD_TOKEN_ID
    local_eos_token_id = DEFAULT_EOS_TOKEN_ID

    if samsum_train_raw and samsum_val_raw and samsum_test_raw_original:
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
            local_pad_token_id = tokenizer.pad_token_id
            local_eos_token_id = tokenizer.eos_token_id
            print(f"Tokenizer '{MODEL_CHECKPOINT}' úspěšně načten. Pad ID: {local_pad_token_id}, EOS ID: {local_eos_token_id}")
        except Exception as e:
            print(f"Chyba při načítání tokenizeru '{MODEL_CHECKPOINT}': {e}")
            tokenizer = None # Pokud se tokenizer nenačte, další kroky selžou

        if tokenizer:
            # Tokenizace trénovacích a validačních dat (bez uchování originálních textů)
            datasets_dict_for_train_val = {"train": samsum_train_raw, "val": samsum_val_raw}
            tokenized_train, tokenized_val, _ = tokenize_data(datasets_dict_for_train_val, tokenizer, keep_original_cols=False)
            
            # Tokenizace testovacích dat s uchováním originálních textů pro ROUGE evaluaci
            datasets_dict_for_test = {"test": samsum_test_raw_original}
            # Zde nepotřebujeme train a val, takže _
            _, _, tokenized_test_with_originals = tokenize_data(datasets_dict_for_test, tokenizer, keep_original_cols=True)

            if tokenized_train and tokenized_val and tokenized_test_with_originals:
                print("Data úspěšně tokenizována.")

                # Vytvoření PyTorch Datasetů
                train_torch_dataset = SummarizationTorchDataset(tokenized_train, tokenizer)
                val_torch_dataset = SummarizationTorchDataset(tokenized_val, tokenizer)
                # test_torch_dataset můžeme vytvořit, pokud bychom chtěli počítat loss na test setu
                # test_torch_dataset = SummarizationTorchDataset(tokenized_test_with_originals, tokenizer)
                
                # Příprava collate funkce s předaným pad_token_id
                custom_collate_fn_with_pad = partial(collate_fn_pytorch, pad_token_id_value=local_pad_token_id)

                # Vytvoření DataLoaders
                train_dataloader = DataLoader(train_torch_dataset, batch_size=_BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn_with_pad, num_workers=0)
                val_dataloader = DataLoader(val_torch_dataset, batch_size=_BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn_with_pad, num_workers=0)
                # test_dataloader = DataLoader(test_torch_dataset, batch_size=_BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn_with_pad, num_workers=0)

                # --- Část 3: Implementace Transformer modelu ---
                print("\n--- Část 3: Implementace Transformer modelu ---")
                # Načtení velikosti slovníku z tokenizeru
                SRC_VOCAB_SIZE = tokenizer.vocab_size
                TGT_VOCAB_SIZE = tokenizer.vocab_size
                
                # Výpočet maximální délky sekvence pro Positional Encoding
                # Měla by být dostatečně velká pro nejdelší vstup nebo výstup.
                MODEL_PE_MAX_LEN = max(MAX_INPUT_LENGTH, MAX_TARGET_LENGTH) + 10 # + rezerva

                # Kontrola, zda je EMB_SIZE dělitelné N_HEAD
                if _EMB_SIZE % _N_HEAD != 0:
                    print(f"CHYBA: _EMB_SIZE ({_EMB_SIZE}) musí být dělitelné _N_HEAD ({_N_HEAD})")
                    exit() # Ukončení skriptu, pokud podmínka není splněna

                # Inicializace modelu
                model = Seq2SeqTransformer(num_encoder_layers=_NUM_ENCODER_LAYERS,
                                           num_decoder_layers=_NUM_DECODER_LAYERS,
                                           emb_size=_EMB_SIZE,
                                           n_head=_N_HEAD,
                                           src_vocab_size=SRC_VOCAB_SIZE,
                                           tgt_vocab_size=TGT_VOCAB_SIZE,
                                           dim_feedforward=_FFN_HID_DIM,
                                           dropout=_DROPOUT,
                                           max_seq_len=MODEL_PE_MAX_LEN)
                model = model.to(device) # Přesun modelu na zvolené zařízení
                print(f"Model vytvořen. Celkový počet trénovatelných parametrů: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

                # --- Část 4: Trénování modelu ---
                print("\n--- Část 4: Trénování modelu (může být přeskočeno, pokud model existuje) ---")
                
                # Podmínka pro spuštění tréninku: pokud model neexistuje nebo je trénink vynucen
                # Pro normální běh odstraňte `or True` z následující podmínky, aby se trénovalo jen jednou.
                force_training = False # Nastavte na True pro vynucení tréninku i když model existuje
                if not os.path.exists(MODEL_SAVE_PATH) or force_training:
                    if force_training and os.path.exists(MODEL_SAVE_PATH):
                        print(f"Trénink je vynucen, existující model v '{MODEL_SAVE_PATH}' bude přepsán.")
                    else:
                        print(f"Model '{MODEL_SAVE_PATH}' nenalezen, spouštím trénink...")
                    
                    optimizer = optim.AdamW(model.parameters(), lr=_LEARNING_RATE)
                    criterion = nn.CrossEntropyLoss(ignore_index=local_pad_token_id) # Ignorování paddingu při výpočtu loss
                    
                    best_val_loss = float('inf')
                    patience_counter = 0

                    for epoch_num in range(_NUM_EPOCHS):
                        current_epoch_display = epoch_num + 1 # Pro zobrazení (1-based)
                        print(f"--- Začíná Epocha {current_epoch_display}/{_NUM_EPOCHS} ---")
                        start_time_epoch = time.time()

                        train_loss = train_epoch(model, optimizer, criterion, train_dataloader, device,
                                                 local_pad_token_id, current_epoch_display, _NUM_EPOCHS, _CLIP_VALUE)
                        
                        valid_loss = evaluate(model, criterion, val_dataloader, device,
                                              local_pad_token_id, current_epoch_display, _NUM_EPOCHS)
                        
                        epoch_duration_mins, epoch_duration_secs = divmod(time.time() - start_time_epoch, 60)
                        
                        print(f"--- Epocha {current_epoch_display}/{_NUM_EPOCHS} dokončena | Čas: {int(epoch_duration_mins)}m {int(epoch_duration_secs)}s ---")
                        print(f"\tTrénovací Loss: {train_loss:.3f} | Trénovací PPL: {math.exp(train_loss):7.3f}")
                        print(f"\tValidační Loss: {valid_loss:.3f} | Validační PPL: {math.exp(valid_loss):7.3f}")

                        if valid_loss < best_val_loss:
                            best_val_loss = valid_loss
                            torch.save(model.state_dict(), MODEL_SAVE_PATH) # Uložení vah nejlepšího modelu
                            print(f"\tNalezen lepší model (val loss: {best_val_loss:.3f}), uložen do {MODEL_SAVE_PATH}")
                            patience_counter = 0 # Reset patience counter
                        else:
                            patience_counter += 1
                            print(f"\tValidační loss se nezlepšil. Patience: {patience_counter}/{_PATIENCE}")
                        
                        if patience_counter >= _PATIENCE:
                            print(f"Early stopping: Validační loss se nezlepšil po {_PATIENCE} epochách.")
                            break # Ukončení tréninkové smyčky
                    print("\nTrénink dokončen.")
                else: # Pokud model již existuje
                    print(f"Načítám existující model z '{MODEL_SAVE_PATH}'")
                    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
                    print("Model úspěšně načten.")

                # --- Část 5: Inference a Evaluace ---
                print("\n--- Část 5: Inference a Evaluace ---")
                if not os.path.exists(MODEL_SAVE_PATH): # Znovu kontrola, pokud trénink selhal nebo nebyl spuštěn
                    print(f"CHYBA: Model '{MODEL_SAVE_PATH}' neexistuje. Nelze provést inferenci/evaluaci.")
                else:
                    # Pokud byl trénink přeskočen, ale model existuje, ujistíme se, že je načten
                    if 'optimizer' not in locals() and os.path.exists(MODEL_SAVE_PATH):
                        print(f"Model nebyl trénován v této session, načítám z '{MODEL_SAVE_PATH}' pro inferenci...")
                        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
                    
                    model.eval() # Ujistíme se, že model je v evaluačním módu pro inferenci

                    # Příklad generování pro jeden dialog z testovací sady
                    print("\n--- Příklad generování shrnutí ---")
                    # Vezmeme první příklad z původní (ne-tokenizované) testovací sady
                    sample_dialogue_text = samsum_test_raw_original[0]["dialogue"]
                    reference_summary_text = samsum_test_raw_original[0]["summary"]
                    
                    print(f"Vstupní dialog:\n{sample_dialogue_text}")
                    # Generování shrnutí pomocí greedy decoding
                    generated_summary = greedy_decode_sentence(model, tokenizer, sample_dialogue_text, device, max_len=MAX_TARGET_LENGTH)
                    print(f"\nReferenční shrnutí:\n{reference_summary_text}")
                    print(f"Vygenerované shrnutí (Greedy):\n{generated_summary}")

                    # Evaluace na (části) testovací sady pomocí ROUGE metrik
                    print("\n--- Evaluace na testovací sadě (ROUGE) ---")
                    num_test_samples_for_rouge = 50 # Pro rychlejší evaluaci; pro finální výsledky použijte více/všechny
                    
                    # Výběr vzorků z původní testovací sady
                    selected_test_samples = samsum_test_raw_original.select(range(min(num_test_samples_for_rouge, len(samsum_test_raw_original))))
                    test_dialogues = [item["dialogue"] for item in selected_test_samples]
                    reference_summaries_for_rouge = [item["summary"] for item in selected_test_samples]
                    
                    generated_predictions_for_rouge = []
                    print(f"Generuji predikce pro {len(test_dialogues)} testovacích vzorků pro ROUGE...")
                    for dialog_text in tqdm(test_dialogues, desc="Generování shrnutí pro ROUGE"):
                        pred = greedy_decode_sentence(model, tokenizer, dialog_text, device, max_len=MAX_TARGET_LENGTH)
                        generated_predictions_for_rouge.append(pred)
                    
                    # Výpočet ROUGE skóre
                    if len(generated_predictions_for_rouge) == len(reference_summaries_for_rouge):
                        rouge_results = calculate_rouge_scores(generated_predictions_for_rouge, reference_summaries_for_rouge)
                        print("\nROUGE výsledky na testovací sadě:")
                        print(f"  ROUGE-1 F1: {rouge_results['rouge1']:.4f}")
                        print(f"  ROUGE-2 F1: {rouge_results['rouge2']:.4f}")
                    else:
                        print("Chyba: Počet generovaných predikcí se neshoduje s počtem referencí pro ROUGE.")

                print("\nCíl Části 5 byl splněn.")
                print("Implementována inference a základní evaluace modelu.")

            else: # Pokud selhala tokenizace
                print("\nChyba během tokenizace dat. Další kroky nelze provést.")
        else: # Pokud selhalo načtení tokenizeru
            print("\nChyba při načítání tokenizeru. Další kroky nelze provést.")
    else: # Pokud selhalo načtení surových dat
        print("\nChyba při načítání surových dat. Další kroky nelze provést.")

    print("\nSkript dokončen.")