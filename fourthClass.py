# -*- coding: utf-8 -*-

import re
from collections import Counter, defaultdict
import random
import math

# --- Část 1: Invertovaný index s normalizací ---
# ... (kód pro stop slova, generování korpusu, preprocessing, build_inverted_index zůstává stejný) ...
# 1. Definice stop slov
stop_words_cs = set([
    "a", "aby", "ahoj", "aj", "ale", "anebo", "ano", "asi", "aspoň", "atd", "atp",
    "az", "až", "bez", "bude", "budem", "budeš", "budete", "budou", "budu",
    "byl", "byla", "byli", "bylo", "byly", "bys", "čau", "chce", "chceme",
    "chceš", "chcete", "chci", "chtít", "co", "což", "cz", "či", "článek",
    "článku", "články", "dál", "dále", "dnes", "do", "dobrý", "ho", "i", "já",
    "jak", "jako", "je", "jeho", "jej", "její", "jejich", "jemu", "jen",
    "jenž", "ještě", "ji", "jinak", "jsem", "jsi", "jsme", "jsou", "jste",
    "k", "kam", "kde", "kdo", "když", "ke", "komu", "která", "které", "který",
    "kteří", "ku", "kvůli", "mají", "málo", "mám", "máme", "máš", "máte", "mě",
    "mezi", "mít", "mně", "mnou", "musí", "můj", "na", "nad", "nám", "námi",
    "např", "napište", "napsal", "napsala", "napsali", "náš", "naše", "ne",
    "nebo", "nechci", "nechtějí", "nechutná", "nejsi", "nejsme", "nejsou",
    "nejste", "ně", "něco", "nějak", "někde", "někdo", "nemají", "nemám",
    "nemáme", "nemáš", "nemáte", "nemohu", "nemůže", "nemůžeme", "nemůžeš",
    "nemůžete", "není", "nestačí", "nevadí", "než", "nic", "ní", "ním",
    "nimi", "niz", "no", "nové", "nový", "o", "od", "ode", "on", "ona", "oni",
    "ono", "ony", "pak", "po", "pod", "podle", "pokud", "polévka", "pomocí",
    "poté", "pouze", "právě", "pro", "proč", "prostě", "proti", "proto",
    "protože", "první", "před", "přede", "přes", "přese", "při", "přijde",
    "přišel", "přišla", "přišli", "přitom", "s", "se", "si", "sice", "skoro",
    "smí", "smějí", "snad", "spolu", "strana", "své", "svých", "svým", "svými",
    "ta", "tak", "také", "takže", "tam", "tamhle", "tato", "té", "tě", "tedy",
    "těma", "témata", "tématu", "těmito", "ten", "tento", "teto", "této",
    "ti", "tím", "tímto", "tipy", "to", "tohle", "toho", "tohoto", "tom",
    "tomto", "tomu", "tomuto", "toto", "tu", "tú", "tuto", "tvá", "tvé", "třeba",
    "tři", "tvoje", "tvůj", "ty", "tyto", "u", "už", "v", "vám", "vámi", "vás",
    "váš", "vaše", "ve", "vedle", "více", "vlastně", "však", "vše", "všechen",
    "všechno", "všichni", "vy", "z", "za", "zatímco", "ze", "že", "již"
])
stop_words_en = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
    "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs",
    "themselves", "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
])
stop_words = stop_words_cs.union(stop_words_en)

# 2. Funkce pro generování jednoduchého korpusu
def generate_simple_corpus(num_docs):
    corpus = []
    words = ["informace", "vyhledávání", "index", "dokument", "text", "data",
             "systém", "model", "korpus", "slovo", "analýza", "počítač",
             "algoritmus", "web", "stránka", "dotaz", "výsledek", "relevantní"]
    # Přidáme pár méně častých slov pro lepší IDF
    words.extend(["unikátní", "specifický", "vzácný", "běžný"])
    for i in range(num_docs):
        doc_len = random.randint(15, 40)
        # Zvýšíme šanci na opakování slov v dokumentu
        doc_words_base = [random.choice(words) for _ in range(doc_len // 2)]
        doc_words = doc_words_base + [random.choice(doc_words_base + ["test", "pro", "a"]) for _ in range(doc_len // 2)]
        doc_words.extend(["je", "to", "test", "pro", "a", "na", "index", "dokument"])
        random.shuffle(doc_words)
        corpus.append(" ".join(doc_words))

    # Přidáme specifické dokumenty pro testování
    corpus.append("informace o vyhledávání a indexech jsou základ") # ID = num_docs
    corpus.append("systém pro analýzu dat a relevantní informace") # ID = num_docs + 1
    corpus.append("vyhledávání informací na webu pomocí systému") # ID = num_docs + 2
    corpus.append("dokumenty a texty bez systému, pouze data") # ID = num_docs + 3
    corpus.append("index a data, data, data!") # ID = num_docs + 4
    corpus.append("velmi specifický unikátní dokument o ničem jiném") # ID = num_docs + 5 (pro test IDF)
    corpus.append("běžný text obsahující běžný systém") # ID = num_docs + 6 (pro test IDF)

    print(f"Vygenerováno {num_docs} dokumentů a přidáno 7 specifických (celkem {len(corpus)}).")
    if corpus:
        print(f"Ukázka prvního dokumentu (ID 0):\n{corpus[0]}\n")
    return corpus

# 3. Funkce pro předzpracování textu dokumentu (pro indexaci)
def preprocess_text_for_indexing(text, stop_words_set):
    text = text.lower()
    tokens = re.findall(r'\b[a-zA-ZáčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ]+\b', text)
    processed_tokens = [
        token for token in tokens
        if token not in stop_words_set and len(token) > 1
    ]
    return processed_tokens

# 4. Funkce pro vytvoření invertovaného indexu
def build_inverted_index(corpus, stop_words_set):
    inverted_index = defaultdict(lambda: defaultdict(int))
    for doc_id, doc_text in enumerate(corpus):
        tokens = preprocess_text_for_indexing(doc_text, stop_words_set)
        term_counts = Counter(tokens)
        for term, frequency in term_counts.items():
            inverted_index[term][doc_id] = frequency
    return dict(inverted_index)

# --- Část 2: Parsování a vyhodnocení boolean dotazů (čistý boolean) ---
OPERATORS = {'AND', 'OR', 'NOT'}
PRECEDENCE = {'NOT': 3, 'AND': 2, 'OR': 1, '(': 0, ')': 0}

def tokenize_query(query: str) -> list:
    # ... (stejná jako předtím) ...
    query = re.sub(r'\b(AND|OR)\b', r' \1 ', query, flags=re.IGNORECASE)
    query = re.sub(r'\b(NOT)\b(?!\s|\()', r' \1 ', query, flags=re.IGNORECASE)
    query = re.sub(r'\b(NOT)\b(?=\()', r' \1 ', query, flags=re.IGNORECASE)
    query = query.replace('(', ' ( ').replace(')', ' ) ')
    tokens = []
    for token in query.split():
        if token.upper() in OPERATORS:
            tokens.append(token.upper())
        elif token in ['(', ')']:
            tokens.append(token)
        else:
            term = token.lower()
            term = re.sub(r'^[^\w]+|[^\w]+$', '', term)
            if term:
                 tokens.append(term)
    return tokens

def infix_to_rpn(tokens: list) -> list:
    # ... (stejná jako předtím, včetně implicitního AND) ...
    output_queue = []
    operator_stack = []
    processed_tokens = []
    # Implicitní AND
    for i, token in enumerate(tokens):
        processed_tokens.append(token)
        if i + 1 < len(tokens):
            current_is_operand = token not in OPERATORS and token not in ['(', ')']
            next_is_operand = tokens[i+1] not in OPERATORS and tokens[i+1] not in ['(', ')']
            current_is_rparen = token == ')'
            next_is_lparen = tokens[i+1] == '('
            if (current_is_operand or current_is_rparen) and \
               (next_is_operand or next_is_lparen):
                processed_tokens.append('AND')
    # Shunting-yard
    for token in processed_tokens:
        if token not in OPERATORS and token not in ['(', ')']:
            output_queue.append(token)
        elif token == '(':
            operator_stack.append(token)
        elif token == ')':
            while operator_stack and operator_stack[-1] != '(':
                output_queue.append(operator_stack.pop())
            if not operator_stack or operator_stack[-1] != '(':
                raise ValueError("Syntax error: Mismatched parentheses")
            operator_stack.pop()
        elif token in OPERATORS:
            while (operator_stack and
                   operator_stack[-1] != '(' and
                   PRECEDENCE.get(operator_stack[-1], 0) >= PRECEDENCE.get(token, 0)):
                 output_queue.append(operator_stack.pop())
            operator_stack.append(token)
    while operator_stack:
        op = operator_stack.pop()
        if op == '(':
            raise ValueError("Syntax error: Mismatched parentheses")
        output_queue.append(op)
    return output_queue

def evaluate_boolean_query(rpn_tokens: list, index: dict, all_doc_ids: set) -> set:
    """Vyhodnotí čistý boolean dotaz (vrací množinu ID). Přejmenováno z evaluate_rpn_query."""
    operand_stack = []
    def get_postings_set(term):
        term_data = index.get(term, {})
        return set(term_data.keys())

    for token in rpn_tokens:
        if token not in OPERATORS:
            postings = get_postings_set(token)
            operand_stack.append(postings)
        else:
            try:
                if token == 'NOT':
                    if not operand_stack: raise ValueError("Syntax error: NOT requires one operand.")
                    operand = operand_stack.pop()
                    result = all_doc_ids - operand
                    operand_stack.append(result)
                elif token in ['AND', 'OR']:
                    if len(operand_stack) < 2: raise ValueError(f"Syntax error: {token} requires two operands.")
                    right_operand = operand_stack.pop()
                    left_operand = operand_stack.pop()
                    if token == 'AND':
                        result = left_operand.intersection(right_operand)
                    else: # OR
                        result = left_operand.union(right_operand)
                    operand_stack.append(result)
            except IndexError:
                raise ValueError(f"Syntax error or insufficient operands for operator '{token}'")

    if len(operand_stack) != 1:
        if not operand_stack and not rpn_tokens: return set()
        raise ValueError(f"Boolean evaluation error: Invalid RPN or stack state. Stack: {operand_stack}")
    return operand_stack[0]

def process_boolean_query(query: str, index: dict, num_total_docs: int) -> set | None:
    """Zpracuje čistý boolean dotaz."""
    # print(f"\n--- Zpracovávám boolean dotaz: '{query}' ---") # Výpis přesunut do rozhraní
    try:
        tokens = tokenize_query(query)
        if not tokens: return set()
        rpn_tokens = infix_to_rpn(tokens)
        if not rpn_tokens: return set()
        all_doc_ids = set(range(num_total_docs))
        result_doc_ids = evaluate_boolean_query(rpn_tokens, index, all_doc_ids)
        return result_doc_ids
    except ValueError as e:
        print(f"Chyba při boolean zpracování dotazu '{query}': {e}")
        return None

# --- Část 3: Analýza velikosti indexu ---
def analyze_index_size(index: dict):
    # ... (stejná jako předtím) ...
    print("\n--- Začínáme: Úkol 3 - Analýza velikosti indexu ---")
    num_unique_terms = len(index)
    total_postings_count = 0
    total_docs_in_lists = 0
    if num_unique_terms == 0:
        print("Index je prázdný.")
        # ... (zbytek výpisu) ...
        print("--- Konec Úkolu 3 ---")
        return

    for term_postings in index.values():
        num_docs_for_term = len(term_postings)
        total_docs_in_lists += num_docs_for_term
        total_postings_count += num_docs_for_term

    average_posting_list_length = total_docs_in_lists / num_unique_terms
    print(f"Počet unikátních termů (tokenů) v indexu: {num_unique_terms}")
    print(f"Celkový počet záznamů (postings) v indexu: {total_postings_count}")
    print(f"Průměrná délka seznamu dokumentů na term: {average_posting_list_length:.2f}")
    print("--- Konec Úkolu 3 ---")


# --- Část 5: Rozšířený boolean model s váhováním (Ranked Retrieval) ---

def calculate_idf(index: dict, total_docs: int) -> dict:
    """Spočítá IDF pro každý term v indexu."""
    idf_scores = {}
    if total_docs == 0:
        return idf_scores
    for term, postings in index.items():
        df = len(postings) # Document Frequency
        if df > 0:
            # Použijeme log10, přidání 1 k N není standardní, raději použijeme N
            # IDF = log10(N / df)
            idf_scores[term] = math.log10(total_docs / df)
        else:
            idf_scores[term] = 0 # Nemělo by nastat pro termy v indexu
    print(f"Spočítáno IDF pro {len(idf_scores)} termů.")
    # Příklad IDF pro běžné a vzácné slovo
    # if 'běžný' in idf_scores: print(f"  IDF('běžný'): {idf_scores['běžný']:.4f}")
    # if 'unikátní' in idf_scores: print(f"  IDF('unikátní'): {idf_scores['unikátní']:.4f}")
    return idf_scores

def evaluate_ranked_boolean_query(rpn_tokens: list, index: dict, idf: dict, total_docs: int) -> list:
    """
    Vyhodnotí dotaz v RPN notaci pomocí TF-IDF a boolean operátorů.
    Vrací seřazený seznam [(doc_id, score), ...].
    """
    operand_stack = []
    all_doc_ids = set(range(total_docs))

    # Funkce pro TF-IDF
    def get_tfidf_scores(term):
        scores = {}
        term_idf = idf.get(term, 0)
        if term_idf == 0: # Term není v IDF slovníku (např. neexistuje)
            return scores
        term_postings = index.get(term, {}) # Vrací {doc_id: tf}
        for doc_id, tf in term_postings.items():
            if tf > 0:
                # Logaritmická TF: 1 + log10(tf)
                tf_log = 1 + math.log10(tf)
                scores[doc_id] = tf_log * term_idf
            # else: tf=0 by nemělo být v postings, ignorujeme
        return scores

    for token in rpn_tokens:
        if token not in OPERATORS:
            # Je to term
            term_scores = get_tfidf_scores(token)
            operand_stack.append(term_scores) # Push dict {doc_id: score}
        else:
            # Je to operátor
            try:
                if token == 'NOT':
                    if not operand_stack: raise ValueError("Syntax error: NOT requires one operand.")
                    operand = operand_stack.pop()
                    # NOT vždy pracuje s množinami ID
                    if isinstance(operand, dict):
                        ids_to_negate = set(operand.keys())
                    elif isinstance(operand, set):
                        ids_to_negate = operand
                    else:
                        raise TypeError(f"Unexpected type on stack for NOT: {type(operand)}")
                    result = all_doc_ids - ids_to_negate
                    operand_stack.append(result) # Push set {doc_id}

                elif token in ['AND', 'OR']:
                    if len(operand_stack) < 2: raise ValueError(f"Syntax error: {token} requires two operands.")
                    right_op = operand_stack.pop()
                    left_op = operand_stack.pop()

                    # Získání ID a skóre pro levý a pravý operand
                    if isinstance(left_op, dict):
                        left_ids = set(left_op.keys())
                        left_scores = left_op
                    else: # Je to set (výsledek NOT)
                        left_ids = left_op
                        left_scores = {} # Skóre není definováno pro NOT výsledek
                    if isinstance(right_op, dict):
                        right_ids = set(right_op.keys())
                        right_scores = right_op
                    else: # Je to set
                        right_ids = right_op
                        right_scores = {}

                    merged_scores = {}
                    if token == 'AND':
                        # AND kombinuje pouze tam, kde oba operandy mají shodu ID
                        result_ids = left_ids.intersection(right_ids)
                        for doc_id in result_ids:
                            # Sečteme skóre, pokud oba operandy jsou dict
                            # Pokud jeden je set (z NOT), skóre bereme jen z dict
                            score = left_scores.get(doc_id, 0) + right_scores.get(doc_id, 0)
                            merged_scores[doc_id] = score
                        # Pokud byl jeden z operandů set (z NOT), výsledek je filtrovaný dict
                        # Pokud oba byly sety, výsledek je set (prázdný dict zde) - ?? Měl by vrátit set? Ne, pro rankování potřebujeme dict.
                        # Pokud `AND(set, set)` nastane, dáme skóre 0? Ne, to by bylo matoucí.
                        # AND by měl produkovat dict jen pokud aspoň jeden operand byl dict.
                        # Pokud AND(set, set), vrátíme prázdný dict? To se zdá nejrozumnější.
                        if not isinstance(left_op, dict) and not isinstance(right_op, dict):
                             operand_stack.append(set()) # AND dvou NOTů => set
                             # print("DEBUG: AND(set, set) -> set()")
                        else:
                             operand_stack.append(merged_scores)
                             # print(f"DEBUG: AND result (dict): {len(merged_scores)} items")

                    else: # OR
                        # OR kombinuje všechny dokumenty z obou operandů
                        result_ids = left_ids.union(right_ids)
                        for doc_id in result_ids:
                            # Sečteme skóre z obou (get vrátí 0, pokud ID chybí)
                            score = left_scores.get(doc_id, 0) + right_scores.get(doc_id, 0)
                            merged_scores[doc_id] = score
                        # OR vždy produkuje dict se skóre (i když jedno bylo set)
                        operand_stack.append(merged_scores)
                        # print(f"DEBUG: OR result (dict): {len(merged_scores)} items")

            except IndexError:
                raise ValueError(f"Ranked evaluation error: Insufficient operands for operator '{token}'")
            except TypeError as e:
                 raise TypeError(f"Ranked evaluation error: Type mismatch on stack? {e}")

    # Konečný výsledek by měl být na vrcholu zásobníku
    if not operand_stack:
        if not rpn_tokens: return [] # Prázdný dotaz
        raise ValueError("Ranked evaluation error: Stack is empty at the end.")

    final_result = operand_stack.pop()

    if operand_stack: # Mělo by být prázdné
        raise ValueError(f"Ranked evaluation error: Stack not empty at the end. Remnants: {operand_stack}")

    # Převedeme výsledek na seřazený seznam [(doc_id, score)]
    if isinstance(final_result, dict):
        # Seřadit podle skóre sestupně
        sorted_results = sorted(final_result.items(), key=lambda item: item[1], reverse=True)
        return sorted_results
    elif isinstance(final_result, set):
        # Pokud výsledek je množina (např. dotaz byl jen 'NOT term'),
        # vrátíme seznam ID s nulovým skóre nebo prázdný seznam?
        # Pro ranked nemá smysl vracet výsledek NOT bez kontextu.
        # Vrátíme prázdný seznam.
        print("Varování: Konečný výsledek rankovaného dotazu je množina (pravděpodobně z NOT). Vracím prázdný seznam.")
        return []
    else:
         raise TypeError(f"Unexpected final result type: {type(final_result)}")


# --- Část 4: Interaktivní rozhraní pro dotazování (Upraveno) ---

def run_interactive_query_session(index: dict, corpus: list, idf_scores: dict):
    """
    Spustí interaktivní smyčku pro zadávání boolean dotazů (ranked i pure).
    """
    print("\n--- Začínáme: Úkol 4 & 5 - Interaktivní dotazování (Ranked + Boolean) ---")
    total_docs = len(corpus)
    all_doc_ids_set = set(range(total_docs))
    if not index or not idf_scores or total_docs == 0:
        print("Nelze spustit dotazování: Index, IDF, nebo korpus chybí/je prázdný.")
        print("--- Konec Úkolu 4 & 5 ---")
        return

    print("Zadejte boolean dotaz (např. 'slovo1 AND (slovo2 OR slovo3) NOT slovo4').")
    print("Budou zobrazeny výsledky rankovaného (TF-IDF) i čistého boolean modelu.")
    print("Pro ukončení zadejte 'exit' nebo 'quit'.")

    while True:
        try:
            query = input("\nZadejte dotaz > ").strip()

            if query.lower() in ['exit', 'quit']:
                print("Ukončuji dotazování.")
                break
            if not query: continue

            print(f"\n--- Dotaz: '{query}' ---")

            # 1. Zpracování (tokenizace, RPN) - společné pro oba modely
            tokens = tokenize_query(query)
            if not tokens:
                print("Chyba: Prázdný nebo neplatný dotaz po tokenizaci.")
                continue
            try:
                rpn_tokens = infix_to_rpn(tokens)
                if not rpn_tokens:
                    print("Chyba: Dotaz nelze převést na RPN.")
                    continue
                print(f"Tokeny: {tokens}")
                print(f"RPN: {rpn_tokens}")
            except ValueError as e:
                print(f"Chyba při parsování dotazu: {e}")
                continue

            # 2. Vyhodnocení - Rankovaný model (TF-IDF)
            print("\n-- Výsledky Ranked (TF-IDF): --")
            try:
                ranked_results = evaluate_ranked_boolean_query(rpn_tokens, index, idf_scores, total_docs)
                if not ranked_results:
                    print("Nenalezeny žádné dokumenty.")
                else:
                    print(f"Nalezeno a seřazeno dokumentů: {len(ranked_results)}")
                    # Zobrazit top N výsledků
                    top_n = 10
                    for i, (doc_id, score) in enumerate(ranked_results[:top_n]):
                        if 0 <= doc_id < total_docs:
                            doc_text = corpus[doc_id]
                            snippet = doc_text[:80].replace('\n', ' ') + ("..." if len(doc_text) > 80 else "")
                            print(f"  {i+1}. Doc {doc_id} (Skóre: {score:.4f}): {snippet}")
                        else:
                            print(f"  {i+1}. Varování: Neplatné ID {doc_id}")
                    if len(ranked_results) > top_n:
                        print(f"  ... (zobrazeno top {top_n} z {len(ranked_results)})")

            except (ValueError, TypeError) as e:
                print(f"Chyba při rankovaném vyhodnocení: {e}")

            # 3. Vyhodnocení - Čistý Boolean model (pro porovnání)
            print("\n-- Výsledky Boolean (množina): --")
            boolean_result_ids = process_boolean_query(query, index, total_docs) # Používá interně evaluate_boolean_query

            if boolean_result_ids is None:
                # Chyba byla vypsána v process_boolean_query
                 pass
            elif not boolean_result_ids:
                 print("Nenalezeny žádné dokumenty.")
            else:
                 sorted_boolean_ids = sorted(list(boolean_result_ids))
                 print(f"Nalezeno dokumentů: {len(sorted_boolean_ids)}")
                 print(f"ID dokumentů: {sorted_boolean_ids}")

            print("-" * 20) # Oddělovač pro další dotaz

        except EOFError: print("\nUkončuji dotazování (EOF)."); break
        except KeyboardInterrupt: print("\nUkončuji dotazování (přerušeno)."); break
        except Exception as e: print(f"\nNeočekávaná chyba v hlavní smyčce: {e}")

# --- Hlavní část skriptu ---
if __name__ == "__main__":
    print("--- Začínáme: Úkol 1 - Invertovaný index ---")
    corpus_size = 50
    my_corpus = generate_simple_corpus(corpus_size)
    total_docs = len(my_corpus)

    print("\nVytvářím invertovaný index...")
    inverted_index = build_inverted_index(my_corpus, stop_words)
    print("Invertovaný index vytvořen.")
    print(f"Celkový počet dokumentů: {total_docs}")
    print("--- Konec Úkolu 1 ---")

    # Úkol 3: Analýza indexu
    analyze_index_size(inverted_index)

    # Úkol 2: Zpracování předdefinovaných dotazů (můžeme zakomentovat nebo nechat pro rychlý test)
    # print("\n--- Začínáme: Úkol 2 - Testovací Boolean dotazy ---")
    # queries_to_test = [
    #    "informace AND vyhledávání",
    #    "informace vyhledávání",
    #    "informace OR systém",
    #    "vyhledávání AND NOT systém",
    #    "(informace OR data) AND index",
    #    "web AND (vyhledávání OR analýza)",
    #    "dokument AND NOT (systém OR web)",
    #    "korpus",
    #    "neexistujícíTermín",
    #    "informace AND (vyhledávání OR data) AND NOT systém",
    #    "NOT systém",
    #    "web AND NOT (index OR systém OR neexistujícíTermín)",
        # "informace AND", # Odkomentovat pro test chyby
        # "(informace OR data", # Odkomentovat pro test chyby
    #    "informace ( web ) data",
    #]
    # for q in queries_to_test:
    #      results = process_boolean_query(q, inverted_index, total_docs)
    #      print("-" * 20)
    # print("--- Konec Úkolu 2 ---")

    print("\n--- Začínáme: Úkol 5 - Výpočet IDF ---")
    idf_scores = calculate_idf(inverted_index, total_docs)
    print("--- Konec Úkolu 5 (Výpočet IDF) ---")

    # Úkol 4: Spuštění interaktivního rozhraní
    run_interactive_query_session(inverted_index, my_corpus, idf_scores)

    print("\n--- Skript dokončen ---")

# Zde bude později přidán kód pro další úkoly...