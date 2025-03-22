import re
from collections import defaultdict
import math

# Function to load the text file and create a word frequency dictionary
def create_word_frequency_dict(file_path):
    # Create a dictionary to store the frequency of each word
    word_frequency = defaultdict(int)
    
    # Open and read the file
    with open(file_path, 'r', encoding='utf-8') as file:
        # Read all lines from the file
        text = file.read().lower()
        
        # Use regular expressions to extract words (ignore punctuation and whitespace)
        words = re.findall(r'\b\w+\b', text)
        
        # Count the frequency of each word
        for word in words:
            word_frequency[word] += 1
    
    return word_frequency

# Levenshtein distance function
def levenshtein_distance(word1, word2):
    # Create a distance matrix
    len_word1, len_word2 = len(word1), len(word2)
    
    # Initialize matrix of size (len_word1+1) x (len_word2+1)
    distance_matrix = [[0] * (len_word2 + 1) for _ in range(len_word1 + 1)]
    
    # Initialize the first row and column
    for i in range(len_word1 + 1):
        distance_matrix[i][0] = i
    for j in range(len_word2 + 1):
        distance_matrix[0][j] = j
    
    # Fill the distance matrix
    for i in range(1, len_word1 + 1):
        for j in range(1, len_word2 + 1):
            if word1[i - 1] == word2[j - 1]:
                cost = 0
            else:
                cost = 1

            # Minimum of delete, insert, or substitute
            distance_matrix[i][j] = min(
                distance_matrix[i - 1][j] + 1,   # Deletion
                distance_matrix[i][j - 1] + 1,   # Insertion
                distance_matrix[i - 1][j - 1] + cost  # Substitution
            )
    
    # The Levenshtein distance is in the bottom-right corner of the matrix
    return distance_matrix[len_word1][len_word2]

# Function to generate variants with max Levenshtein distance of 2
def generate_variants(word, max_distance=2):
    variants = set()  # Using a set to avoid duplicates
    variants.add(word) # Also add the word into variants so if the word is correct we keep it the same
    # Function to generate variants by insert, delete, replace, or swap
    def insert_variants(word):
        alphabet = 'abcdefghijklmnopqrstuvwxyzáéíóúýčďěňóřšťžů'
        for i in range(len(word) + 1):
            for c in alphabet:
                variants.add(word[:i] + c + word[i:])
    
    def delete_variants(word):
        for i in range(len(word)):
            variants.add(word[:i] + word[i+1:])
    
    def replace_variants(word):
        alphabet = 'abcdefghijklmnopqrstuvwxyzáéíóúýčďěňóřšťžů'
        for i in range(len(word)):
            for c in alphabet:
                if c != word[i]:
                    variants.add(word[:i] + c + word[i+1:])
    
    def swap_variants(word):
        for i in range(len(word) - 1):
            variants.add(word[:i] + word[i+1] + word[i] + word[i+2:])
    
    # Generate variants for each operation
    insert_variants(word)
    delete_variants(word)
    replace_variants(word)
    swap_variants(word)
    
    # Filter out variants that are too far away (Levenshtein distance > max_distance)
    valid_variants = {variant for variant in variants if levenshtein_distance(word, variant) <= max_distance}
    
    return valid_variants

# Function to find the most frequent word from the dictionary among variants
def get_most_frequent_variant(variants, word_frequency):
    max_frequency = -1
    most_frequent_word = None
    
    for variant in variants:
        # Check if variant exists in the frequency dictionary
        if variant in word_frequency:
            frequency = word_frequency[variant]
            if frequency > max_frequency:
                #print(f"Variant: {variant}, frequency: {frequency}")
                max_frequency = frequency
                most_frequent_word = variant
    
    return most_frequent_word

# Function to find the closest word in the dictionary by Levenshtein distance
def find_closest_word(word, word_frequency):
    closest_word = None
    min_distance = float('inf')  # Initialize with a very large number
    
    for dict_word in word_frequency.keys():
        # Calculate Levenshtein distance between the word and the dictionary word
        distance = levenshtein_distance(word, dict_word)
        
        # If the distance is smaller than the current minimum, update closest_word
        if distance < min_distance:
            min_distance = distance
            closest_word = dict_word
    
    return closest_word

# Function to calculate n-grams for the words in the dictionary
def create_ngrams(word_frequency, n=2):
    ngrams = defaultdict(lambda: defaultdict(int))
    
    for word in word_frequency.keys():
        # Create n-grams (bigrams for this example)
        for i in range(len(word) - n + 1):
            ngram = word[i:i+n]
            ngrams[ngram][word] += 1
    
    return ngrams

# Function to calculate the conditional probability of a word given its n-grams
def calculate_conditional_probability(word, ngrams, word_frequency):
    log_prob = 0
    for i in range(len(word) - 1):
        bigram = word[i:i+2]
        if bigram in ngrams:
            total_bigrams = sum(ngrams[bigram].values())
            prob = ngrams[bigram][word] / total_bigrams
            log_prob += math.log(prob) if prob > 0 else -float('inf')
    
    return log_prob

# Function to find the best word using n-gram and conditional probability
def find_best_word_with_ngrams(word, word_frequency, ngrams):
    variants = generate_variants(word, max_distance=2)
    best_variant = None
    best_prob = -float('inf')  # Initialize with very low probability
    
    for variant in variants:
        prob = calculate_conditional_probability(variant, ngrams, word_frequency)
        if prob > best_prob:
            best_prob = prob
            best_variant = variant
    
    return best_variant

# Example usage
file_path = 'cs.txt'  # Specify your file path here
word_frequency = create_word_frequency_dict(file_path)
# Display the first few entries of the dictionary
print("LOG: Built dictionary of words and recorded down their frequency")
for word, frequency in list(word_frequency.items())[:10]:
    print(f"{word}: {frequency}")

# Test Levenshtein distance calculation with pairs of words from the dictionary
sample_words = ["oběd", "oběť", "vypsat", "vyspat", "restaurace", "rustauraci", "televize", "televizi"]

print("\nLevenshtein Distances between word pairs:\n")

# Test pairs of words from the dictionary and print their Levenshtein distances
for i in range(len(sample_words) - 1):
    word1 = sample_words[i]
    word2 = sample_words[i + 1]
    distance = levenshtein_distance(word1, word2)
    print(f"Levenshtein distance between '{word1}' and '{word2}': {distance}")

print("\nGenerated variants with max Levenshtein distance of 2:\n")

for word in sample_words:
    variants = generate_variants(word, max_distance=2)
    print(f"Word: {word} | Generated variants: {len(variants)}")

print("Most frequent word approach:")

test_sentence = "Dneska si dám oběť v restauarci a pak půjdu zpěť domů, kde se podívám na televezí"

print(f"Incorrect sentence: {test_sentence}")

splittedSentence = test_sentence.lower().split()
correctedSentence = []
for word in splittedSentence:
    variants = generate_variants(word)
    result = get_most_frequent_variant(variants, word_frequency)
    correctedSentence.append(result)
print("Corrected sentence: ", end="")
for word in correctedSentence:
    print(str(word) + " ", end="")
print()
print("Closest word from dictionary approach:")

test_sentence = "Dneska si dám oběť v restauarci a pak půjdu zpěť domů, kde se podívám na televezí"

print(f"Incorrect sentence: {test_sentence}")

splittedSentence = test_sentence.lower().split()
correctedSentence = []
for word in splittedSentence:
    result = find_closest_word(word, word_frequency)
    correctedSentence.append(result)
print("Corrected sentence: ", end="")
for word in correctedSentence:
    print(str(word) + " ", end="")

print("Creating ngrams")
# Create n-grams for the words in the dictionary
ngrams = create_ngrams(word_frequency, n=2)

print("N-gram approach:")

test_sentence = "Dneska si dám oběť v restauarci a pak půjdu zpěť domů, kde se podívám na televezí"

print(f"Incorrect sentence: {test_sentence}")

splittedSentence = test_sentence.lower().split()
correctedSentence = []
for word in splittedSentence:
    result = find_best_word_with_ngrams(word, word_frequency, ngrams)
    correctedSentence.append(result)
print("Corrected sentence: ", end="")
for word in correctedSentence:
    print(str(word) + " ", end="")