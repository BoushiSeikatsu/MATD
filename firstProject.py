import re
import nltk
import numpy as np
from collections import Counter
from datasets import load_dataset
# Použijte tqdm.notebook pokud běžíte v Jupyter/Colab, jinak jen tqdm
# from tqdm.notebook import tqdm
from tqdm import tqdm
import time # Pro měření času

# --- Část 1: Příprava Dat ---

print("--- Část 1: Příprava Dat ---")

# --- Konfigurace Přípravy Dat ---
DATASET_NAME = "fewshot-goes-multilingual/cs_csfd-movie-reviews"
VOCAB_SIZE = 10000  # Cílová velikost slovníku (bez <UNK>)
CONTEXT_WINDOW_SIZE = 2 # Počet slov vlevo a vpravo od cílového slova

# --- Krok 1.1: Načtení datasetu ---
print("Načítám dataset...")
try:
    # Načteme pouze trénovací split pro demonstraci
    dataset = load_dataset(DATASET_NAME, split='train')
    # Získáme pouze texty recenzí (komentářů) - Správný název sloupce
    texts = [item['comment'] for item in dataset]
    print(f"Dataset načten. Počet recenzí: {len(texts)}")
    if texts:
        print("Příklad recenze:", texts[0][:200] + "...") # Ukázka
    else:
        print("Nebyly nalezeny žádné texty recenzí ve sloupci 'comment'.")
        exit()
except Exception as e:
    print(f"Chyba při načítání datasetu nebo extrakci sloupce 'comment': {e}")
    print("Zkontrolujte připojení k internetu, název datasetu a název sloupce ('comment').")
    exit()

# --- Krok 1.2: Tokenizace ---
print("\nProvádím tokenizaci...")

# Stáhneme potřebné NLTK data (pokud nejsou k dispozici)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Stahuji NLTK tokenizer 'punkt'...")
    nltk.download('punkt', quiet=True)

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-záčďéěíňóřšťúůýž0-9\s]", "", text)
    tokens = nltk.word_tokenize(text, language='czech')
    return tokens

all_tokens = []
print("Tokenizuji texty...")
for text in tqdm(texts, desc="Tokenizace"):
    all_tokens.extend(tokenize(text))

print(f"Tokenizace dokončena. Celkový počet tokenů: {len(all_tokens)}")
if not all_tokens:
    print("Chyba: Po tokenizaci nezůstala žádná slova.")
    exit()
print("Příklad tokenů:", all_tokens[:20])

# --- Krok 1.3: Sestavení slovníku ---
print(f"\nSestavuji slovník o velikosti {VOCAB_SIZE} nejčastějších slov...")
word_counts = Counter(all_tokens)
most_common_words = word_counts.most_common(VOCAB_SIZE)

word_to_ix = {'<UNK>': 0}
ix_to_word = ['<UNK>']
for i, (word, _) in enumerate(most_common_words):
    word_to_ix[word] = i + 1
    ix_to_word.append(word)

actual_vocab_size = len(word_to_ix)
print(f"Slovník sestaven. Skutečná velikost (včetně <UNK>): {actual_vocab_size}")
if actual_vocab_size <= 1:
     print("Chyba: Slovník je příliš malý nebo prázdný.")
     exit()
if actual_vocab_size < VOCAB_SIZE + 1:
     print(f"Poznámka: V korpusu bylo méně než {VOCAB_SIZE} unikátních slov.")

# --- Krok 1.4: Generování trénovacích párů (kontext -> cíl) ---
print(f"\nGeneruji trénovací páry (kontextové okno: {CONTEXT_WINDOW_SIZE} vlevo/vpravo)...")
training_data = []
print("Generuji páry z recenzí...")
for text in tqdm(texts, desc="Generování párů"):
    tokens = tokenize(text)
    indexed_tokens = [word_to_ix.get(token, word_to_ix['<UNK>']) for token in tokens]

    for i in range(len(indexed_tokens)):
        target_word_ix = indexed_tokens[i]
        context_indices = []
        start_left = max(0, i - CONTEXT_WINDOW_SIZE)
        context_indices.extend(indexed_tokens[start_left:i])
        end_right = min(len(indexed_tokens), i + CONTEXT_WINDOW_SIZE + 1)
        context_indices.extend(indexed_tokens[i+1:end_right])

        if context_indices: # Přidáme pouze pokud kontext není prázdný
            training_data.append((context_indices, target_word_ix))

if not training_data:
    print("Chyba: Nebyly vygenerovány žádné trénovací páry. Zkontrolujte velikost okna a data.")
    exit()

print(f"Generování trénovacích párů dokončeno. Počet párů: {len(training_data)}")
print("Příklad trénovacích párů ([indexy kontextu], index cíle):")
for i in range(min(5, len(training_data))):
    context_ixs, target_ix = training_data[i]
    context_words = [ix_to_word[ix] for ix in context_ixs]
    target_word = ix_to_word[target_ix]
    print(f"  {context_ixs} -> {target_ix}  (Kontext: {context_words}, Cíl: '{target_word}')")

print("\n--- Příprava dat dokončena ---")

# --- Část 2: Trénink CBOW Modelu ---

print("\n--- Část 2: Trénink CBOW Modelu ---")

# --- Hyperparametry Tréninku ---
embedding_dim = 100     # Dimenze embedding vektorů
learning_rate = 0.01    # Míra učení pro SGD
epochs = 5              # Počet průchodů celým datasetem

# --- Krok 2.1: Inicializace vah ---
print("Inicializuji váhy modelu...")
np.random.seed(42) # Pro reprodukovatelnost
E = np.random.randn(actual_vocab_size, embedding_dim) * 0.01 # Vstupní embeddingy
W = np.random.randn(embedding_dim, actual_vocab_size) * 0.01 # Výstupní váhy
b = np.zeros((1, actual_vocab_size))                         # Výstupní bias

print(f"Inicializované matice:")
print(f"  E shape: {E.shape}")
print(f"  W shape: {W.shape}")
print(f"  b shape: {b.shape}")

# --- Krok 2.2: Pomocné funkce ---
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def cross_entropy_loss(probs, target_index):
    correct_prob = probs[0, target_index] + 1e-9 # Přidáme epsilon pro stabilitu
    return -np.log(correct_prob)

# --- Krok 2.3: Tréninková smyčka ---
print("\n--- Zahajuji trénink ---")
start_time = time.time()
total_loss_history = []

for epoch in range(epochs):
    epoch_start_time = time.time()
    total_loss = 0
    # Náhodné promíchání dat pro každou epochu (důležité pro SGD)
    np.random.shuffle(training_data)

    data_iterator = tqdm(training_data, desc=f"Epocha {epoch+1}/{epochs}", leave=False)

    for context_indices, target_index in data_iterator:
        # Forward Pass
        context_vectors = E[context_indices]
        hidden_layer = np.mean(context_vectors, axis=0, keepdims=True)
        scores = hidden_layer @ W + b
        probs = softmax(scores)
        loss = cross_entropy_loss(probs, target_index)
        total_loss += loss

        # Backward Pass
        dscores = probs.copy()
        dscores[0, target_index] -= 1
        db = dscores
        dW = hidden_layer.T @ dscores
        dhidden = dscores @ W.T

        # SGD Update
        W -= learning_rate * dW
        b -= learning_rate * db
        grad_E_update = dhidden / len(context_indices)
        # Iterativní update pro E kvůli možným duplicitním indexům v kontextu
        for idx in context_indices:
             E[idx] -= learning_rate * grad_E_update[0]

    # Konec epochy
    epoch_end_time = time.time()
    avg_loss = total_loss / len(training_data)
    total_loss_history.append(avg_loss)
    epoch_duration = epoch_end_time - epoch_start_time
    print(f"Konec Epochy {epoch+1}/{epochs}, Průměrná ztráta: {avg_loss:.4f}, Trvání: {epoch_duration:.2f}s")

total_training_time = time.time() - start_time
print(f"\n--- Trénink dokončen --- (Celkový čas: {total_training_time:.2f}s)")

# --- Výsledky a Ukázka ---
print("\nVýsledná matice embeddingů E má shape:", E.shape)

# Uložení výsledků (volitelné)
# filename = f"cbow_embeddings_csfd_vocab{actual_vocab_size}_dim{embedding_dim}_ep{epochs}.npz"
# np.savez(filename, embeddings=E, word_to_ix=word_to_ix, ix_to_word=ix_to_word)
# print(f"Embeddingy uloženy do souboru: {filename}")

# Příklad: Získání embeddingu pro slovo 'film' (pokud je ve slovníku)
word_example = 'film'
if word_example in word_to_ix:
    example_index = word_to_ix[word_example]
    example_embedding = E[example_index]
    print(f"\nEmbedding pro slovo '{word_example}' (prvních 10 dimenzí):\n{example_embedding[:10]}")
else:
    print(f"\nSlovo '{word_example}' není ve slovníku (nebo není dostatečně časté).")

# Graf vývoje loss (volitelné, vyžaduje matplotlib)
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), total_loss_history)
    plt.title('Vývoj tréninkové ztráty (Cross-Entropy)')
    plt.xlabel('Epocha')
    plt.ylabel('Průměrná ztráta')
    plt.xticks(range(1, epochs + 1))
    plt.grid(True)
    plt.show()
except ImportError:
    print("\nKnihovna matplotlib není nainstalována. Graf ztráty nelze zobrazit.")
    print("Historie průměrných ztrát:", total_loss_history)