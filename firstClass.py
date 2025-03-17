import re
import nltk
import random
import math
from collections import Counter, defaultdict
from nltk.util import ngrams
from nltk.tokenize.toktok import ToktokTokenizer
nltk.download('punkt')
nltk.download('perluniprops')
nltk.download('nonbreaking_prefixes')

def preprocess_text(file_path):
    tokenizer = ToktokTokenizer()
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()  # Převedení na malá písmena
    
    text = re.sub(r'[^a-zá-žÁ-Ž0-9\s]', '', text)  # Odstranění interpunkce
    tokens = tokenizer.tokenize(text)  # Tokenizace slov pomocí českého tokenizátoru
    return tokens if tokens else []

def generate_ngrams(tokens, n):
    if not tokens:
        return []
    return list(ngrams(tokens, n))

def calculate_frequencies(ngrams_list):
    return Counter(ngrams_list) if ngrams_list else Counter()

def calculate_probabilities(ngrams_counts, n_minus_1_counts):
    probabilities = {}
    for ngram, count in ngrams_counts.items():
        prefix = ngram[:-1]
        probabilities[ngram] = count / (n_minus_1_counts[prefix] if n_minus_1_counts[prefix] > 0 else 1)
    return probabilities

def laplace_smoothing(ngrams_counts, n_minus_1_counts, vocab_size, alpha=1):
    probabilities = {}
    for ngram, count in ngrams_counts.items():
        prefix = ngram[:-1]
        probabilities[ngram] = (count + alpha) / (n_minus_1_counts[prefix] + alpha * vocab_size if n_minus_1_counts[prefix] > 0 else alpha * vocab_size)
    return probabilities

def predict_next_word(word, bigram_probs):
    candidates = {k: v for k, v in bigram_probs.items() if k[0] == word}
    if not candidates:
        return None
    return max(candidates, key=candidates.get)[1]

def generate_text(start_word, trigram_probs, bigram_probs, length=20):
    text = [start_word]
    for _ in range(length - 1):
        if len(text) >= 2:
            candidates = {k: v for k, v in trigram_probs.items() if k[:-1] == tuple(text[-2:])}
        else:
            candidates = {k: v for k, v in bigram_probs.items() if k[0] == text[-1]}
        
        if not candidates:
            break
        next_word = random.choices(list(candidates.keys()), weights=list(candidates.values()))[0][-1]
        text.append(next_word)
    return ' '.join(text)

def perplexity(test_tokens, ngram_probs, n):
    test_ngrams = generate_ngrams(test_tokens, n)
    log_prob_sum = 0
    count = len(test_ngrams)
    for ngram in test_ngrams:
        prob = ngram_probs.get(ngram, 1e-10)
        log_prob_sum += math.log(prob)
    return math.exp(-log_prob_sum / count) if count > 0 else float('inf')

def main(file_path):
    tokens = preprocess_text(file_path)
    if not tokens:
        print("Chyba: Textový soubor je prázdný nebo nelze zpracovat.")
        return
    
    unigrams = generate_ngrams(tokens, 1)
    bigrams = generate_ngrams(tokens, 2)
    trigrams = generate_ngrams(tokens, 3)
    
    unigram_freq = calculate_frequencies(unigrams)
    bigram_freq = calculate_frequencies(bigrams)
    trigram_freq = calculate_frequencies(trigrams)
    
    print("Nejčastější unigramy:", unigram_freq.most_common(10))
    print("Nejčastější bigramy:", bigram_freq.most_common(10))
    print("Nejčastější trigramy:", trigram_freq.most_common(10))
    
    if not unigram_freq or not bigram_freq:
        print("Chyba: Nedostatek dat pro výpočet pravděpodobností.")
        return
    
    bigram_probs = calculate_probabilities(bigram_freq, unigram_freq)
    trigram_probs = calculate_probabilities(trigram_freq, bigram_freq)
    
    vocab_size = len(set(tokens))
    bigram_probs_smoothed = laplace_smoothing(bigram_freq, unigram_freq, vocab_size)
    trigram_probs_smoothed = laplace_smoothing(trigram_freq, bigram_freq, vocab_size)
    print(f"Pocet slov je:{vocab_size}")
    test_word = "je"  # Slovo pro predikci
    predicted_word = predict_next_word(test_word, bigram_probs_smoothed)
    print(f"Predikované slovo po '{test_word}': {predicted_word}")
    
    generated_text = generate_text(test_word, trigram_probs_smoothed, bigram_probs_smoothed, length=20)
    print("Vygenerovaný text:", generated_text)
    
    test_set = preprocess_text(file_path)[:1000]  # Použití části textu jako testovacího setu
    print("Perplexity unigramového modelu:", perplexity(test_set, calculate_probabilities(unigram_freq, Counter()), 1))
    print("Perplexity bigramového modelu:", perplexity(test_set, bigram_probs_smoothed, 2))
    print("Perplexity trigramového modelu:", perplexity(test_set, trigram_probs_smoothed, 3))

if __name__ == "__main__":
    main("cs.txt")
