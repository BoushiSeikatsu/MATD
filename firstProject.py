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
VOCAB_SIZE = 1000  # Cílová velikost slovníku (bez <UNK>)
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

# Seznam NLTK resources, které potřebujeme
REQUIRED_NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab" # Přidáno pro jazykově specifické tabulky
}

# Stáhneme potřebné NLTK data (pokud nejsou k dispozici)
for resource_name, resource_path in REQUIRED_NLTK_RESOURCES.items():
    try:
        # Nejprve zkusíme, zda je resource dostupný
        nltk.data.find(resource_path)
        print(f"NLTK resource '{resource_name}' ({resource_path}) je již k dispozici.")
    except LookupError:
        print(f"NLTK resource '{resource_name}' ({resource_path}) nenalezen. Pokouším se stáhnout...")
        try:
            # Stáhneme resource podle jeho krátkého jména (např. 'punkt', 'punkt_tab')
            nltk.download(resource_name, quiet=True)
            print(f"NLTK resource '{resource_name}' úspěšně stažen.")
            # Po stažení je dobré znovu zkusit resource najít, abychom se ujistili
            nltk.data.find(resource_path)
        except Exception as e:
            print(f"Chyba při stahování NLTK resource '{resource_name}': {e}")
            print(f"Prosím, zkuste stáhnout '{resource_name}' manuálně pomocí:")
            print(">>> import nltk")
            print(f">>> nltk.download('{resource_name}')")
            print("A poté spusťte skript znovu.")
            exit()

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
embedding_dim = 50     # Dimenze embedding vektorů
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

# --- Část 3: Evaluace a Vizualizace Embeddingů ---

print("\n\n--- Část 3: Evaluace a Vizualizace Embeddingů ---")

# Potřebné importy pro tuto část
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm # Pro barvy v grafu
# Nastavení pro české popisky v matplotlibu, pokud je potřeba
# from matplotlib import rcParams
# rcParams['font.family'] = 'DejaVu Sans' # Nebo jiný font podporující češtinu

# Matice embeddingů je 'E'
# Slovníky jsou 'word_to_ix' a 'ix_to_word'

# --- Krok 3.1: Nalezení nejbližších sousedů ---
def get_top_n_similar(word, embedding_matrix, word_to_index, index_to_word_map, top_n=5):
    """Najde top_n nejpodobnějších slov k danému slovu."""
    if word not in word_to_index:
        print(f"Slovo '{word}' není ve slovníku.")
        return []

    word_idx = word_to_index[word]
    word_vector = embedding_matrix[word_idx].reshape(1, -1) # (1, embedding_dim)

    # Vypočítáme kosinovou podobnost mezi daným slovem a všemi ostatními
    # cosine_similarity očekává dva 2D pole
    similarities = cosine_similarity(word_vector, embedding_matrix)[0] # Výsledek je (1, vocab_size), vezmeme první řádek

    # Seřadíme indexy podle podobnosti (sestupně)
    # np.argsort vrátí indexy, které by seřadily pole
    # [::-1] pro sestupné řazení
    sorted_indices = np.argsort(similarities)[::-1]

    similar_words = []
    for i in range(1, top_n + 1): # Začínáme od 1, abychom přeskočili samotné slovo
        if i < len(sorted_indices):
            neighbor_idx = sorted_indices[i]
            neighbor_word = index_to_word_map[neighbor_idx]
            similarity_score = similarities[neighbor_idx]
            similar_words.append((neighbor_word, similarity_score))
        else:
            break # Pokud už nemáme dost slov ve slovníku
    return similar_words

# Vybraná slova pro testování
words_to_test = ["film", "dobrý", "nuda", "herec", "hudba", "příběh"]
# Můžete přidat slova, která očekáváte, že budou ve vašem slovníku
# (zejména pokud jste VOCAB_SIZE výrazně snížili)

print("\n--- Nejbližší sousedé (Kosinová podobnost) ---")
for test_word in words_to_test:
    if test_word in word_to_ix: # Zkontrolujeme, zda je slovo ve slovníku po omezení VOCAB_SIZE
        neighbors = get_top_n_similar(test_word, E, word_to_ix, ix_to_word, top_n=5)
        if neighbors:
            print(f"Nejpodobnější slova k '{test_word}':")
            for neighbor, score in neighbors:
                print(f"  - {neighbor} (Podobnost: {score:.4f})")
    else:
        print(f"Slovo '{test_word}' není v aktuálním slovníku (velikost: {actual_vocab_size}).")


# --- Krok 3.2: Vizualizace embedding prostoru (t-SNE) ---
# t-SNE může být výpočetně náročné pro velký slovník.
# Pro vizualizaci vybereme podmnožinu slov (např. N nejčastějších).
# Nebo můžeme použít PCA, které je rychlejší.

# Počet slov pro vizualizaci
num_words_to_visualize = 200 # Můžete upravit
if actual_vocab_size <= 1:
    print("\nSlovník je příliš malý pro vizualizaci.")
else:
    # Vezmeme prvních N slov ze slovníku (kromě <UNK>)
    # Pokud je slovník menší, vezmeme všechna dostupná slova
    words_for_viz_indices = list(range(1, min(actual_vocab_size, num_words_to_visualize + 1)))
    selected_embeddings = E[words_for_viz_indices]
    selected_words = [ix_to_word[i] for i in words_for_viz_indices]

    print(f"\nProvádím redukci dimenze pro {len(selected_words)} slov pomocí t-SNE (může chvíli trvat)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(selected_words)-1), n_iter=1000, init='pca', learning_rate='auto')
    # Perplexity by měla být menší než počet vzorků.
    # 'init'='pca' může zrychlit a stabilizovat t-SNE
    # 'learning_rate'='auto' je doporučeno od sklearn 0.22
    try:
        embeddings_2d = tsne.fit_transform(selected_embeddings)

        print("Vykresluji embeddingy...")
        plt.figure(figsize=(14, 10))
        # Vykreslíme body
        # Můžeme použít barvy pro odlišení, např. na základě frekvence nebo náhodně
        colors = cm.rainbow(np.linspace(0, 1, len(selected_words)))

        for i, word in enumerate(selected_words):
            x, y = embeddings_2d[i, :]
            plt.scatter(x, y, color=colors[i % len(colors)]) # Použijeme modulo pro případ, že máme více slov než unikátních barev v paletě
            # Anotujeme pouze některá slova, aby graf nebyl přeplněný
            # Například každé N-té slovo, nebo slova z našeho testovacího seznamu, pokud jsou ve vizualizované podmnožině
            if i % 10 == 0 or word in words_to_test: # Anotuj každé 10. nebo pokud je to testovací slovo
                 plt.annotate(word,
                             xy=(x, y),
                             xytext=(5, 2),
                             textcoords='offset points',
                             ha='right',
                             va='bottom',
                             fontsize=9)
        plt.title(f't-SNE vizualizace {len(selected_words)} slovních embeddingů')
        plt.xlabel("t-SNE komponenta 1")
        plt.ylabel("t-SNE komponenta 2")
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"Chyba při t-SNE vizualizaci: {e}")
        print("Možná je příliš málo dat pro t-SNE (zkuste menší perplexity nebo více dat).")
        print("Například, `perplexity` musí být menší než počet vzorků.")


# --- Krok 3.3: Zhodnocení ---
print("\n--- Zhodnocení kvality embeddingů ---")
print("Prohlédněte si výpis nejbližších sousedů a t-SNE graf (pokud byl vygenerován).")
print("Odpovídají si slova, která jsou si blízko, i významově?")
print("Například:")
print("  - Jsou synonyma nebo slova s podobným kontextem použití blízko sebe?")
print("  - Tvoří slova shluky podle nějaké sémantické kategorie (např. jídlo, místa, akce)?")
print("Mějte na paměti, že kvalita bude silně záviset na:")
print("  - Velikosti trénovacího korpusu (náš je relativně malý).")
print("  - Velikosti slovníku (VOCAB_SIZE).")
print("  - Dimenzi embeddingů (embedding_dim).")
print("  - Počtu epoch tréninku.")
print("  - Hyperparametrech (learning_rate, context_window_size).")
print("Pro velmi malé VOCAB_SIZE a málo epoch nemusí být výsledky příliš přesvědčivé.")