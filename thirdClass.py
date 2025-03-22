import collections
import time
from collections import defaultdict

# 1. Výpočet Levenshteinovy vzdálenosti
def levenshtein_distance(s1, s2):
    len_s1, len_s2 = len(s1), len(s2)
    dp = [[0] * (len_s2 + 1) for _ in range(len_s1 + 1)]
    
    for i in range(len_s1 + 1):
        dp[i][0] = i
    for j in range(len_s2 + 1):
        dp[0][j] = j
    
    for i in range(1, len_s1 + 1):
        for j in range(1, len_s2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # smazání
                dp[i][j - 1] + 1,  # vložení
                dp[i - 1][j - 1] + cost  # nahrazení
            )
    
    return dp[len_s1][len_s2]

# 2. Příprava slovníku
def prepare_dictionary(text):
    words = text.split()
    word_count = collections.Counter(words)
    return word_count

# 3. Generování variant slov
def generate_variants(word, max_distance=2):
    alphabet = 'abcdefghijklmnopqrstuvwxyzáčďéěíňóřštúýž'
    variants = set()

    # Vložení, smazání, nahrazení a prohození sousedů
    for i in range(len(word) + 1):
        for c in alphabet:
            variants.add(word[:i] + c + word[i:])
            if i < len(word):
                variants.add(word[:i] + word[i + 1:])
            if i < len(word) - 1:
                variants.add(word[:i] + word[i + 1] + word[i] + word[i + 2:])
    
    # Filtrace variant podle maximální editační vzdálenosti
    filtered_variants = set()
    for variant in variants:
        if levenshtein_distance(word, variant) <= max_distance:
            filtered_variants.add(variant)
    
    return filtered_variants

# 4. Automatická oprava slov
def correct_spelling(word, word_dict):
    variants = generate_variants(word)
    candidates = [(variant, word_dict.get(variant, 0)) for variant in variants]
    candidates.sort(key=lambda x: (-x[1], levenshtein_distance(word, x[0])))  # Četnost + editační vzdálenost
    return candidates[0][0] if candidates else word

# 5. Alternativní přístup (výpočet editační vzdálenosti ke slovům ve slovníku)
def correct_spelling_alternative(word, word_dict):
    distances = [(w, levenshtein_distance(word, w)) for w in word_dict]
    distances.sort(key=lambda x: (x[1], word_dict[x[0]]))  # Nejprve podle vzdálenosti, pak četnosti
    return distances[0][0]

# 6. Oprava věty
def correct_sentence(sentence, word_dict):
    words = sentence.split()
    corrected_words = [correct_spelling(word, word_dict) for word in words]
    return ' '.join(corrected_words)

# 7. Bonusová úloha (N-gram model pro zlepšení pravděpodobnosti opravy)
def prepare_ngrams(text, n=2):
    words = text.split()
    ngrams = defaultdict(int)
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i + n])
        ngrams[ngram] += 1
    return ngrams

def correct_with_ngrams(word, prev_word, ngrams, word_dict):
    candidates = generate_variants(word)
    best_candidate = word
    max_prob = 0
    for candidate in candidates:
        bigram = (prev_word, candidate)
        prob = ngrams.get(bigram, 0) / sum(ngrams.get((prev_word, w), 0) for w in word_dict)
        if prob > max_prob:
            best_candidate = candidate
            max_prob = prob
    return best_candidate

# 8. Testování na vlastní větě
def test_spell_correction():
    # Vytvoření textu pro slovník
    corpus = "Dneska si dám oběť v restauraci a pak půjdu zpět domů, kde se podívám na televizi."
    word_dict = prepare_dictionary(corpus)

    # Oprava věty
    sentence = "Dneska si dám oběť v restauarci a pak půjdu zpěť domů, kde se podívám na televezí."
    
    # Test základního přístupu
    start = time.time()
    corrected_sentence_basic = correct_sentence(sentence, word_dict)
    end = time.time()
    print("Opravená věta (základní přístup):", corrected_sentence_basic)
    print(f"Čas základního přístupu: {end - start} sekund")

    # Test alternativního přístupu
    start = time.time()
    corrected_sentence_alt = correct_spelling_alternative(sentence.split()[0], word_dict)  # testování jen na prvním slově pro rychlost
    end = time.time()
    print("Opravená věta (alternativní přístup):", corrected_sentence_alt)
    print(f"Čas alternativního přístupu: {end - start} sekund")

if __name__ == "__main__":
    test_spell_correction()
