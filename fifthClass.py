# -*- coding: utf-8 -*-
import string # Pro práci s interpunkcí
import os     # Pro práci se souborovým systémem
import math   # Pro výpočet logaritmu (IDF) a odmocniny (kosinová podobnost)
from collections import Counter # Pro snadnější počítání frekvencí
import time # Pro měření času
import itertools # Pro generování kombinací párů dokumentů

# --- Část 1: Předzpracování textu ---

# Definice vlastního seznamu stopslov (alespoň 5)
CUSTOM_STOP_WORDS = set([
    "a", "v", "na", "je", "se", "s", "z", "do", "pro", "o", "k", "i", "ale", "jako", "tak",
    "tento", "tato", "toto", "že", "by", "si", "jsou", "být", "který", "která", "které",
    "co", "ve", "po", "při", "už", "jen", "může", "musí",
])

def load_document(filepath):
    """ Načte obsah textového souboru. """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        print(f"Chyba: Soubor '{filepath}' nebyl nalezen.")
        return None
    except Exception as e:
        print(f"Chyba při čtení souboru '{filepath}': {e}")
        return None

def preprocess_text(text, stop_words):
    """ Provede předzpracování textu. """
    if text is None:
        return []
    text = text.lower()
    translator = str.maketrans('', '', string.punctuation + '„“‚‘–—…«»')
    text = text.translate(translator)
    tokens = text.split()
    processed_tokens = [token for token in tokens if token not in stop_words and token.strip()]
    return processed_tokens

# --- Část 2: Výpočet TF, IDF a TF-IDF ---

def compute_tf(processed_docs):
    """ Spočítá normovanou Term Frequency (TF) pro každý term v každém dokumentu. """
    tf_scores = {}
    for doc_id, terms in processed_docs.items():
        if not terms:
             tf_scores[doc_id] = {}
             continue
        term_counts = Counter(terms)
        total_terms_in_doc = len(terms)
        # Používáme relativní četnost jako formu normování TF
        doc_tf = {term: count / total_terms_in_doc for term, count in term_counts.items()}
        tf_scores[doc_id] = doc_tf
    # Zdůvodnění normování: Použití relativní četnosti (dělení celkovým počtem termů v dokumentu)
    # normalizuje TF a snižuje vliv délky dokumentu. Delší dokumenty tak nemají automaticky
    # vyšší TF váhy jen proto, že obsahují více slov. To vede ke spravedlivějšímu porovnání
    # důležitosti termů napříč dokumenty různé délky.
    return tf_scores

def compute_df(processed_docs):
    """ Spočítá Document Frequency (DF) pro každý term v korpusu efektivně. """
    df_counts = Counter()
    for terms in processed_docs.values():
        unique_terms_in_doc = set(terms)
        df_counts.update(unique_terms_in_doc)
    return dict(df_counts)

def compute_idf(df_scores, total_docs):
    """ Spočítá Inverse Document Frequency (IDF) pro každý term. """
    idf_scores = {}
    if total_docs == 0:
        return idf_scores
    for term, df in df_scores.items():
        if df > 0:
             # Používáme standardní vzorec IDF: log(N / df(t))
             idf_scores[term] = math.log(total_docs / df)
        else:
             idf_scores[term] = 0.0
    return idf_scores

def compute_tfidf(tf_scores, idf_scores):
    """ Spočítá TF-IDF váhy pro každý term v každém dokumentu. """
    tfidf_weights = {}
    for doc_id, doc_tf in tf_scores.items():
        doc_tfidf = {}
        for term, tf in doc_tf.items():
            idf = idf_scores.get(term, 0.0)
            doc_tfidf[term] = tf * idf
        tfidf_weights[doc_id] = doc_tfidf
    return tfidf_weights

def score_query(query, tfidf_weights, idf_scores, stop_words):
    """ Spočítá skóre pro každý dokument na základě dotazu pomocí sumace TF-IDF vah. """
    query_terms = preprocess_text(query, stop_words)
    #print(f"\nPředzpracovaný dotaz: {query_terms}") # Můžeme odkomentovat pro ladění

    if not query_terms:
        print("Dotaz neobsahuje žádné relevantní termy po předzpracování.")
        return []

    unique_query_terms = set(query_terms)
    doc_scores = {}
    for doc_id, doc_tfidf in tfidf_weights.items():
        score = 0.0
        for term in unique_query_terms:
            score += doc_tfidf.get(term, 0.0)
        if score > 1e-9:
             doc_scores[doc_id] = score

    sorted_docs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_docs

# --- NOVÁ ČÁST: Výpočet Kosinové Podobnosti ---

def compute_vector_magnitude(vector):
    """ Spočítá Eukleidovskou normu (velikost) vektoru TF-IDF. """
    # vector je slovník {term: tfidf_value}
    # Velikost = sqrt(suma(tfidf_value^2))
    if not vector:
        return 0.0
    magnitude_sq = sum(value**2 for value in vector.values())
    return math.sqrt(magnitude_sq)

def compute_cosine_similarity(vec1, vec2, magnitude1=None, magnitude2=None):
    """
    Spočítá kosinovou podobnost mezi dvěma TF-IDF vektory (slovníky).
    similarity = (vec1 · vec2) / (||vec1|| * ||vec2||)

    Args:
        vec1 (dict): TF-IDF vektor prvního dokumentu {term: tfidf}.
        vec2 (dict): TF-IDF vektor druhého dokumentu {term: tfidf}.
        magnitude1 (float, optional): Předpočítaná velikost vec1. Defaults to None.
        magnitude2 (float, optional): Předpočítaná velikost vec2. Defaults to None.

    Returns:
        float: Kosinová podobnost (hodnota mezi 0 a 1 pro TF-IDF).
    """
    # Najdeme sjednocení klíčů (termů) pro výpočet skalárního součinu
    all_terms = set(vec1.keys()) | set(vec2.keys())

    # Vypočítáme skalární součin (dot product)
    dot_product = 0.0
    for term in all_terms:
        dot_product += vec1.get(term, 0.0) * vec2.get(term, 0.0)

    # Vypočítáme velikosti vektorů (pokud nejsou předpočítané)
    if magnitude1 is None:
        magnitude1 = compute_vector_magnitude(vec1)
    if magnitude2 is None:
        magnitude2 = compute_vector_magnitude(vec2)

    # Zabráníme dělení nulou, pokud je jeden z vektorů nulový (neměl by nastat u TF-IDF, ale pro jistotu)
    if magnitude1 == 0.0 or magnitude2 == 0.0:
        return 0.0
    else:
        similarity = dot_product / (magnitude1 * magnitude2)
        # Ošetření drobných nepřesností float čísel (výsledek by měl být <= 1.0)
        return min(similarity, 1.0)


# --- Hlavní část skriptu ---

if __name__ == "__main__":
    start_time = time.time() # Začátek měření

    current_directory = 'abc/'
    # Ošetření, pokud adresář neexistuje
    if not os.path.isdir(current_directory):
        print(f"Chyba: Adresář '{current_directory}' nebyl nalezen.")
        exit()

    all_files_and_dirs = os.listdir(current_directory)
    document_filenames = []
    script_name = os.path.basename(__file__) # Získáme jméno tohoto skriptu

    for item in all_files_and_dirs:
        if item == script_name:
            continue
        item_path = os.path.join(current_directory, item)
        # Hledáme soubory s příponou .txt (upraveno dle kódu)
        if os.path.isfile(item_path) and os.path.splitext(item)[1] == '.txt':
             document_filenames.append(item)

    # Pro testování je lepší mít více dokumentů, ale i s málem to poběží
    MIN_DOCS_FOR_SIMILARITY = 2
    if len(document_filenames) < MIN_DOCS_FOR_SIMILARITY:
        print(f"Varování: Nalezeno pouze {len(document_filenames)} dokumentů.")
        print(f"         Pro výpočet kosinové podobnosti jsou potřeba alespoň {MIN_DOCS_FOR_SIMILARITY} dokumenty.")
        # Můžeme buď skončit, nebo jen přeskočit část s podobností
        # exit()

    print("--- Část 1: Předzpracování textu ---")
    print(f"Nalezené dokumenty k zpracování: {document_filenames}")
    print(f"Použitá stopslova (custom): {sorted(list(CUSTOM_STOP_WORDS))}\n")

    processed_documents = {}
    docs_load_start = time.time()
    for filename in document_filenames:
        # print(f"Zpracovávám dokument: {filename}") # Můžeme ztišit pro rychlejší běh
        filepath = os.path.join(current_directory, filename)
        doc_content = load_document(filepath)
        if doc_content:
            terms = preprocess_text(doc_content, CUSTOM_STOP_WORDS)
            processed_documents[filename] = terms
        else:
            print(f"  Přeskakuji zpracování dokumentu '{filename}' kvůli chybě při načítání.\n")

    docs_load_end = time.time()
    print("-" * 20)
    print(f"Předzpracování dokončeno za {docs_load_end - docs_load_start:.2f} s.")
    print(f"Počet úspěšně zpracovaných dokumentů: {len(processed_documents)}")
    print("-" * 20)

    # --- Výpočty pro Část 2 a 3 ---
    if len(processed_documents) >= MIN_DOCS_FOR_SIMILARITY:
        print("\n--- Část 2: Výpočet TF, IDF, TF-IDF ---")
        calc_start_time = time.time()

        N = len(processed_documents)
        print(f"Celkový počet dokumentů (N): {N}")

        # 1. Výpočet TF
        tf_scores = compute_tf(processed_documents)
        print(f"Výpočet TF dokončen.") # Zkrácený výpis

        # 2. Výpočet DF
        df_scores = compute_df(processed_documents)
        print(f"Výpočet DF dokončen.")

        # 3. Výpočet IDF
        idf_scores = compute_idf(df_scores, N)
        print(f"Výpočet IDF dokončen.")

        # 4. Výpočet TF-IDF
        tfidf_weights = compute_tfidf(tf_scores, idf_scores)
        print(f"Výpočet TF-IDF dokončen.")

        calc_end_time = time.time()
        print(f"Čas výpočtů vah: {calc_end_time - calc_start_time:.4f} s.")
        print("-" * 20)

        # --- Část 3: Výpočet Kosinové Podobnosti ---
        print("\n--- Část 3: Výpočet Kosinové Podobnosti ---")
        similarity_start_time = time.time()

        doc_ids = list(tfidf_weights.keys())
        similarities = []

        # Předpočítáme velikosti vektorů pro efektivitu
        magnitudes = {doc_id: compute_vector_magnitude(tfidf_weights[doc_id]) for doc_id in doc_ids}

        # Projdeme všechny unikátní páry dokumentů
        most_similar_pair = (None, None)
        max_similarity = -1.0 # Kosinová podobnost je >= 0 pro TF-IDF

        # Použijeme itertools.combinations pro elegantní generování párů
        for doc_id1, doc_id2 in itertools.combinations(doc_ids, 2):
            vec1 = tfidf_weights[doc_id1]
            vec2 = tfidf_weights[doc_id2]
            mag1 = magnitudes[doc_id1]
            mag2 = magnitudes[doc_id2]

            similarity = compute_cosine_similarity(vec1, vec2, mag1, mag2)
            similarities.append((doc_id1, doc_id2, similarity))

            # Aktualizujeme nejpodobnější pár
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_pair = (doc_id1, doc_id2)

        similarity_end_time = time.time()

        # Seřadíme podobnosti pro přehlednost (volitelné)
        # similarities.sort(key=lambda x: x[2], reverse=True)
        # print("\nKosinová podobnost mezi všemi páry dokumentů:")
        # for d1, d2, sim in similarities:
        #     print(f"  Sim({d1}, {d2}) = {sim:.4f}")

        print(f"\nVýpočet kosinové podobnosti dokončen za {similarity_end_time - similarity_start_time:.4f} s.")

        if most_similar_pair[0] is not None:
            print(f"\nNejpodobnější dvojice dokumentů:")
            print(f"  Dokument 1: {most_similar_pair[0]}")
            print(f"  Dokument 2: {most_similar_pair[1]}")
            print(f"  Kosinová podobnost: {max_similarity:.4f}")
            print("\n  Interpretace:")
            print("  > ZDE DOPLŇTE VAŠI INTERPRETACI: Proč jsou si tyto dva dokumenty nejpodobnější?")
            print("  > Podívejte se na jejich obsah a společné termíny s vysokou TF-IDF vahou.")
        else:
            print("\nNepodařilo se nalézt žádný pár dokumentů pro porovnání.")

        print("\nZamyšlení: Jak by se výsledky změnily při použití jen TF (bez IDF)?")
        print("  > Použití pouze TF by vedlo k tomu, že by si byly podobnější dokumenty,")
        print("  > které sdílejí mnoho obecně frekventovaných slov (např. 'být', 'mít', pokud nejsou")
        print("  > ve stopslovech), i když jejich specifické téma může být odlišné.")
        print("  > IDF váha potlačuje vliv těchto běžných slov a zdůrazňuje slova specifická")
        print("  > pro daný dokument, což vede k sémanticky relevantnějšímu porovnání témat.")
        print("-" * 20)


        # --- Část 2 (pokračování): Skórování Dotazu ---
        print("\n--- Část 2 (pokračování): Skórování Dotazu ---")
        # Smyčka pro dotazy zůstává stejná
        while True:
            query = input("\nZadejte váš dotaz (nebo 'konec' pro ukončení): ")
            if not query or query.lower() == 'konec':
                break

            query_start_time = time.time()
            # Používáme původní funkci score_query, která počítá sumu TF-IDF
            ranked_docs = score_query(query, tfidf_weights, idf_scores, CUSTOM_STOP_WORDS)
            query_end_time = time.time()

            print(f"\nVýsledky vyhledávání (metoda: suma TF-IDF, čas: {query_end_time - query_start_time:.4f} s):")
            if ranked_docs:
                top_n = 10
                print(f"Top {min(top_n, len(ranked_docs))} výsledků:")
                for i, (doc_id, score) in enumerate(ranked_docs[:top_n]):
                     print(f"  {i+1}. Dokument: {doc_id}, Skóre: {score:.4f}")
                if not ranked_docs:
                    print("  Pro zadaný dotaz nebyly nalezeny žádné relevantní dokumenty s nenulovým skóre.")
            else:
                print("  Pro zadaný dotaz nebyly nalezeny žádné relevantní dokumenty s nenulovým skóre.")

        print("-" * 20)

    else:
        print(f"\nNebylo zpracováno dostatek dokumentů ({len(processed_documents)}/{MIN_DOCS_FOR_SIMILARITY}) pro výpočty TF-IDF a kosinové podobnosti.")


    # --- Část 4: Význam IDF v různých doménách (Textová odpověď) ---
    print("\n--- Část 4: Význam IDF v různých doménách ---")
    print("Úkol: Uveďte příklad oblasti, témata, kde by častá slova mohla být navzdory vysoké frekvenci velmi důležitá.")
    print("      Vysvětlete, proč v takovém případě může být použití klasického idf nevhodné.")
    print("      Navrhněte úpravu výpočtu, která by tento problém zmírnila.")
    print("\nVaše odpověď:")
    print(" > ZDE VLOŽTE VAŠI ODPOVĚĎ NA OTÁZKY ČÁSTI 4 (můžete napsat přímo sem nebo odkazovat na samostatný text).")
    print(" > Příklad: V medicínských textech mohou být slova jako 'pacient' nebo 'diagnóza' velmi častá (nízké IDF),")
    print(" >          ale jsou klíčová. Klasické IDF by je podvážilo. Úprava: Napadlo mě udělat si list slov pro danou oblast, například to zdravotnictví. Pokud by slovo bylo z listu, tak by IDF bylo nastaveno na 1")
    print("-" * 20)

    # --- Část 5: Návrh alternativního váhovacího schématu (Textová odpověď) ---
    print("\n--- Část 5: Návrh alternativního váhovacího schématu pro krátké texty ---")
    print("Úkol: Navrhněte váhovací schéma pro krátké texty (např. tweety), které by lépe zachytilo význam slov než klasické tf-idf.")
    print("      Popište, jak by vaše schéma vážilo slova:")
    print("        - velmi častá napříč korpusem,")
    print("        - vyskytující se pouze jednou,")
    print("        - vyskytující se v části dokumentů.")
    print("\nVaše odpověď:")
    print(" > ZDE VLOŽTE VAŠI ODPOVĚĎ NA OTÁZKY ČÁSTI 5 (můžete napsat přímo sem nebo odkazovat na samostatný text).")
    print(" > BinaryTF, povídat o tom")
    print("-" * 20)


    end_time = time.time() # Konec celkového měření
    print(f"\nCelkový čas běhu skriptu: {end_time - start_time:.2f} s.")