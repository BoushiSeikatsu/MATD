# -*- coding: utf-8 -*-
import string # Pro práci s interpunkcí
import os     # Pro práci se souborovým systémem (os.path.basename still used)
import math   # Pro výpočet logaritmu (IDF)
from collections import Counter # Pro snadnější počítání frekvencí
import random # Pro generování náhodných dat

# --- Configuration for Random Document Generation ---
NUM_DOCUMENTS_TO_GENERATE = 10  # How many documents to create
MIN_WORDS_PER_DOC = 50         # Minimum words in a generated document
MAX_WORDS_PER_DOC = 200        # Maximum words in a generated document
MIN_WORD_LENGTH = 3            # Minimum length of a generated word
MAX_WORD_LENGTH = 10           # Maximum length of a generated word
VOCABULARY_SIZE = 500          # Limit the size of the random vocabulary for better term overlap

# --- Část 1: Předzpracování textu ---

# Definice vlastního seznamu stopslov (alespoň 5)
CUSTOM_STOP_WORDS = set([
    "a", "v", "na", "je", "se", "s", "z", "do", "pro", "o", "k", "i", "ale", "jako", "tak",
    "tento", "tato", "toto", "že", "by", "si", "jsou", "být", "který", "která", "které",
    "co", "ve", "po", "při", "už", "jen", "může", "musí",
])

# This function is no longer needed as we generate documents
# def load_document(filepath):
#     """ Načte obsah textového souboru. """
#     ... (original code commented out)

# --- Helper functions for random generation ---
def generate_random_word(min_len, max_len):
    """Generates a random word (string of lowercase letters)."""
    length = random.randint(min_len, max_len)
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))

def create_random_vocabulary(size, min_len, max_len):
    """Creates a set of unique random words."""
    vocab = set()
    while len(vocab) < size:
        vocab.add(generate_random_word(min_len, max_len))
    return list(vocab) # Return as a list for random.choice

def generate_random_document_content(vocabulary, min_words, max_words):
    """Generates document content by sampling words from the vocabulary."""
    num_words = random.randint(min_words, max_words)
    # Sample words *with replacement* from the vocabulary
    words = random.choices(vocabulary, k=num_words)
    return " ".join(words)

# --- (Keep preprocess_text as it is) ---
def preprocess_text(text, stop_words):
    """ Provede předzpracování textu. """
    if text is None:
        return []
    text = text.lower()
    # Keep Czech punctuation if needed, otherwise default string.punctuation is fine
    # Add Czech specific quotes if they appear in generated text (unlikely)
    translator = str.maketrans('', '', string.punctuation + '„“‚‘–—…«»')
    text = text.translate(translator)
    tokens = text.split()
    # Note: Randomly generated words are unlikely to be in CUSTOM_STOP_WORDS unless generated that way
    processed_tokens = [token for token in tokens if token not in stop_words and token.strip()]
    return processed_tokens

# --- Část 2: Výpočet TF, IDF a TF-IDF ---
# --- (Keep compute_tf, compute_df, compute_idf, compute_tfidf, score_query as they are) ---
def compute_tf(processed_docs):
    """
    Spočítá normovanou Term Frequency (TF) pro každý term v každém dokumentu.
    TF(t, d) = (počet výskytů t v d) / (celkový počet termů v d)

    Args:
        processed_docs (dict): Slovník {doc_id: [term1, term2, ...]}

    Returns:
        dict: Slovník slovníků {doc_id: {term: tf_value}}
    """
    tf_scores = {}
    for doc_id, terms in processed_docs.items():
        term_counts = Counter(terms) # Spočítá výskyty každého termu v dokumentu
        total_terms_in_doc = len(terms)
        doc_tf = {}
        if total_terms_in_doc > 0: # Zabráníme dělení nulou pro prázdné dokumenty
            for term, count in term_counts.items():
                doc_tf[term] = count / total_terms_in_doc
        tf_scores[doc_id] = doc_tf
    return tf_scores

def compute_df(processed_docs):
    """
    Spočítá Document Frequency (DF) pro každý term v korpusu.
    DF(t) = počet dokumentů obsahujících term t.

    Args:
        processed_docs (dict): Slovník {doc_id: [term1, term2, ...]}

    Returns:
        dict: Slovník {term: df_value}
    """
    df_counts = Counter()
    # Pro každý term spočítáme, v kolika unikátních dokumentech se vyskytuje
    all_terms = set(term for terms in processed_docs.values() for term in terms)
    for term in all_terms:
        count = 0
        for doc_id in processed_docs:
            # Použijeme set pro rychlou kontrolu přítomnosti termu
            if term in set(processed_docs[doc_id]):
                count += 1
        df_counts[term] = count
    return dict(df_counts) # Vracíme jako standardní dict

def compute_idf(df_scores, total_docs):
    """
    Spočítá Inverse Document Frequency (IDF) pro každý term.
    IDF(t) = log(N / df(t))   (používá přirozený logaritmus)

    Args:
        df_scores (dict): Slovník {term: df_value}
        total_docs (int): Celkový počet dokumentů (N).

    Returns:
        dict: Slovník {term: idf_value}
    """
    idf_scores = {}
    # Ujistíme se, že nebudeme mít N=0
    if total_docs == 0:
        return {}

    for term, df in df_scores.items():
        # df by mělo být vždy > 0, protože počítáme DF jen pro termy, které se vyskytly
        if df > 0:
             # Add-1 smoothing (optional, helps if df == total_docs)
             # idf_scores[term] = math.log((total_docs + 1) / (df + 1)) + 1.0
             # Standard IDF:
             idf_scores[term] = math.log(total_docs / df)
        else:
            # This case should ideally not happen if df_scores comes from compute_df
            # based on documents where the term actually appeared.
            idf_scores[term] = 0.0 # Assign 0 IDF for terms not found (or handle as error)
            print(f"Varování: Term '{term}' má DF=0, což by nemělo nastat pro termy z df_scores.")

    return idf_scores


def compute_tfidf(tf_scores, idf_scores):
    """
    Spočítá TF-IDF váhy pro každý term v každém dokumentu.
    TF-IDF(t, d) = TF(t, d) * IDF(t)

    Args:
        tf_scores (dict): Slovník slovníků {doc_id: {term: tf_value}}
        idf_scores (dict): Slovník {term: idf_value}

    Returns:
        dict: Slovník slovníků {doc_id: {term: tfidf_value}}
    """
    tfidf_weights = {}
    for doc_id, doc_tf in tf_scores.items():
        doc_tfidf = {}
        for term, tf in doc_tf.items():
            # If a term was present in a document (tf > 0), it must have an IDF score.
            # Use .get only as a safeguard, theoretically idf_scores should contain the term.
            idf = idf_scores.get(term, 0.0)
            if idf == 0.0 and term in idf_scores:
                 # This might happen if df == total_docs and no smoothing is used.
                 # You might decide if these terms should have 0 weight or a small floor value.
                 # print(f"Poznámka: Term '{term}' má IDF 0 (pravděpodobně se vyskytuje ve všech dokumentech).")
                 pass # Keep TF-IDF as 0 in this case
            doc_tfidf[term] = tf * idf
        tfidf_weights[doc_id] = doc_tfidf
    return tfidf_weights


def score_query(query, tfidf_weights, idf_scores, stop_words):
    """
    Spočítá skóre pro každý dokument na základě dotazu pomocí sumace TF-IDF vah.
    Score(q, d) = sum_{t in q} tf-idf(t, d)

    Args:
        query (str): Vstupní dotaz.
        tfidf_weights (dict): Slovník slovníků {doc_id: {term: tfidf_value}}.
        idf_scores (dict): Slovník {term: idf_value} (not directly needed here but often useful alongside).
        stop_words (set): Množina stopslov pro předzpracování dotazu.

    Returns:
        list: Seznam dvojic (doc_id, score) seřazený sestupně podle skóre.
    """
    # 1. Předzpracování dotazu
    query_terms = preprocess_text(query, stop_words)
    print(f"\nPředzpracovaný dotaz: {query_terms}")

    if not query_terms:
        print("Dotaz neobsahuje žádné relevantní termy po předzpracování.")
        return []

    # 2. Výpočet skóre pro každý dokument
    doc_scores = {}
    for doc_id, doc_tfidf in tfidf_weights.items():
        score = 0.0
        processed_query_terms_in_doc = 0 # Count how many query terms are found in the doc's TF-IDF map
        for term in query_terms:
            # Přičteme TF-IDF váhu termu v dokumentu, pokud tam term existuje
            term_tfidf = doc_tfidf.get(term, 0.0)
            if term_tfidf > 0:
                processed_query_terms_in_doc += 1
            score += term_tfidf

        # Optional: Normalize score by the number of query terms found?
        # if processed_query_terms_in_doc > 0:
        #     score = score / processed_query_terms_in_doc
        # Or normalize by query length?
        # if len(query_terms) > 0:
        #     score = score / len(query_terms)

        doc_scores[doc_id] = score

    # 3. Seřazení dokumentů podle skóre (sestupně)
    sorted_docs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)

    return sorted_docs

# --- Hlavní část skriptu ---

if __name__ == "__main__":

    print("--- Generování náhodných dokumentů ---")
    # Generate a shared vocabulary for all documents
    # This increases the chance of term overlap between documents, making IDF more meaningful
    print(f"Vytvářím náhodný slovník o velikosti {VOCABULARY_SIZE}...")
    vocabulary = create_random_vocabulary(VOCABULARY_SIZE, MIN_WORD_LENGTH, MAX_WORD_LENGTH)
    print("Slovník vytvořen.")

    generated_documents_content = {}
    print(f"Generuji {NUM_DOCUMENTS_TO_GENERATE} náhodných dokumentů...")
    for i in range(NUM_DOCUMENTS_TO_GENERATE):
        doc_id = f"random_doc_{i+1}"
        content = generate_random_document_content(
            vocabulary,
            MIN_WORDS_PER_DOC,
            MAX_WORDS_PER_DOC
        )
        generated_documents_content[doc_id] = content
        # print(f"  Generated {doc_id} with {len(content.split())} words.") # Optional: print details
    print("Dokumenty vygenerovány.\n")

    print("--- Část 1: Předzpracování textu ---")
    print(f"Počet generovaných dokumentů k zpracování: {len(generated_documents_content)}")
    print(f"Použitá stopslova (custom): {sorted(list(CUSTOM_STOP_WORDS))}\n")

    processed_documents = {}
    # Use the generated documents instead of reading files
    for doc_id, doc_content in generated_documents_content.items():
        # print(f"Zpracovávám dokument: {doc_id}") # Optional: print progress
        # print(f"  Originál (začátek): {doc_content[:100]}...") # Optional: view generated content
        terms = preprocess_text(doc_content, CUSTOM_STOP_WORDS)
        processed_documents[doc_id] = terms
        # print(f"  Výsledné termy ({len(terms)}): {terms[:20]}...\n") # Optional: view processed terms

    print("-" * 20)
    print("Předzpracování dokončeno.")
    print(f"Počet úspěšně zpracovaných dokumentů: {len(processed_documents)}")
    print("-" * 20)

    # --- Výpočty pro Část 2 ---
    if processed_documents:
        print("\n--- Část 2: Výpočet TF, IDF, TF-IDF a Skóre Dotazu ---")

        # Počet dokumentů
        N = len(processed_documents)
        print(f"Celkový počet dokumentů (N): {N}")

        # 1. Výpočet TF
        print("Počítám TF...")
        tf_scores = compute_tf(processed_documents)
        # Optional: Print TF example
        # first_doc_id_tf = list(tf_scores.keys())[0]
        # print(f"TF Scores (ukázka pro '{first_doc_id_tf}'): {dict(list(tf_scores[first_doc_id_tf].items())[:5])}")


        # 2. Výpočet DF
        print("Počítám DF...")
        df_scores = compute_df(processed_documents)
        # Optional: Print DF example
        # print(f"DF Scores (ukázka): {dict(list(df_scores.items())[:10])}")
        print(f"Nalezeno unikátních termů (velikost slovníku po preprocessingu): {len(df_scores)}")


        # 3. Výpočet IDF
        print("Počítám IDF...")
        idf_scores = compute_idf(df_scores, N)
        # Optional: Print IDF example
        # sorted_idf = sorted(idf_scores.items(), key=lambda item: item[1], reverse=True)
        # print(f"IDF Scores (ukázka top 5): {[(k, round(v, 4)) for k, v in sorted_idf[:5]]}")
        # print(f"IDF Scores (ukázka bottom 5): {[(k, round(v, 4)) for k, v in sorted_idf[-5:]]}")


        # 4. Výpočet TF-IDF
        print("Počítám TF-IDF...")
        tfidf_weights = compute_tfidf(tf_scores, idf_scores)
        # Optional: Print TF-IDF example
        # first_doc_id_tfidf = list(tfidf_weights.keys())[0]
        # print(f"TF-IDF Weights (ukázka pro '{first_doc_id_tfidf}'): {{k: round(v, 4) for k, v in list(tfidf_weights[first_doc_id_tfidf].items())[:5]}}")


        # 5. Skórování Dotazu
        # Try querying for words that are likely in the generated vocabulary
        print("\n--- Skórování dotazu ---")
        print("Můžete zkusit zadat náhodné řetězce písmen (např. 'abc', 'randomword')")
        print(f"Nebo zkuste některý z nejčastějších termů (pokud jsou zobrazeny výše).")

        while True: # Loop for multiple queries
            query = input("\nZadejte váš dotaz (nebo stiskněte Enter pro ukončení): ")
            if not query:
                break

            ranked_docs = score_query(query, tfidf_weights, idf_scores, CUSTOM_STOP_WORDS)

            print("\nVýsledky vyhledávání (dokumenty seřazené podle skóre):")
            found_count = 0
            if ranked_docs:
                for doc_id, score in ranked_docs:
                    # Vypíšeme jen dokumenty s nenulovým skóre
                    if score > 1e-9: # Prahová hodnota pro zobrazení (kvůli nepřesnostem float)
                         print(f"  Dokument: {doc_id}, Skóre: {score:.4f}")
                         found_count += 1
                    # else: pass # Skip docs with zero score

                if found_count == 0:
                     print("  Pro zadaný dotaz nebyly nalezeny žádné relevantní dokumenty (nebo měly nulové skóre).")

            else: # Should not happen if score_query returns empty list only on empty query
                 print("  Pro zadaný dotaz nebyly nalezeny žádné relevantní dokumenty.")


        print("-" * 20)
        print("Výpočty TF-IDF a skórování dokončeny.")
        print("-" * 20)
    else:
        print("\nNebyly zpracovány žádné dokumenty (chyba při generování nebo preprocessingu?), nelze pokračovat s výpočty TF-IDF.")