# -*- coding: utf-8 -*-

"""
Combined file containing:
1. Implementation of Unary, Elias Gamma, and Fibonacci universal codes.
2. Simulation of inverted index creation and d-gap encoding using these codes.
3. Comparison of size and search speed.
"""

import math
import random
import string
import time
import bisect # For binary search in uncompressed list
from collections import defaultdict

# ==============================================================================
# Part 1: Universal Coding Implementations
# ==============================================================================

# --- Helper: Fibonacci Numbers ---
_fib_cache = {2: 1, 3: 2} # Using F(2)=1, F(3)=2,... convention

def _get_fibonacci_numbers_until(limit_value):
    global _fib_cache
    sorted_keys = sorted(_fib_cache.keys())
    fib_numbers = [_fib_cache[k] for k in sorted_keys]
    if not fib_numbers:
        _fib_cache = {2: 1, 3: 2}
        sorted_keys = sorted(_fib_cache.keys())
        fib_numbers = [_fib_cache[k] for k in sorted_keys]
    i = sorted_keys[-1] if sorted_keys else 1
    if fib_numbers and fib_numbers[-1] > limit_value:
        relevant_fibs = []
        for idx_key in sorted_keys:
            fib_val = _fib_cache[idx_key]
            relevant_fibs.append(fib_val)
            if fib_val > limit_value: return relevant_fibs
        return relevant_fibs
    while len(fib_numbers) < 2:
        if 2 not in _fib_cache: _fib_cache[2] = 1; fib_numbers.append(1)
        if 3 not in _fib_cache: _fib_cache[3] = 2; fib_numbers.append(2)
        i = 3
    while not fib_numbers or fib_numbers[-1] <= limit_value:
        i += 1
        if i-1 not in _fib_cache: _get_fib_number(i-1)
        if i-2 not in _fib_cache: _get_fib_number(i-2)
        next_fib = _fib_cache[i-1] + _fib_cache[i-2]
        _fib_cache[i] = next_fib
        fib_numbers.append(next_fib)
    return fib_numbers

def _get_fib_number(k):
    global _fib_cache
    if k < 2: raise ValueError("Fibonacci index must be >= 2 for this convention.")
    if k in _fib_cache: return _fib_cache[k]
    f_k_minus_1 = _get_fib_number(k-1)
    f_k_minus_2 = _get_fib_number(k-2)
    _fib_cache[k] = f_k_minus_1 + f_k_minus_2
    return _fib_cache[k]

# --- Unary Coding ---
def encode_unary(n: int) -> str:
    if not isinstance(n, int) or n < 1: raise ValueError(f"Unary input must be a positive integer (n >= 1). Got: {n}")
    return '1' * (n - 1) + '0'

def decode_unary(bit_stream: str) -> tuple[int, str]:
    zero_pos = bit_stream.find('0')
    if zero_pos == -1: raise ValueError("Invalid Unary code: no terminating '0'.")
    decoded_n = zero_pos + 1
    remaining_stream = bit_stream[zero_pos + 1:]
    return decoded_n, remaining_stream

# --- Elias Gamma Coding ---
def encode_elias_gamma(n: int) -> str:
    if not isinstance(n, int) or n < 1: raise ValueError(f"Elias Gamma input must be a positive integer (n >= 1). Got: {n}")
    if n == 1: return '0'
    L = n.bit_length()
    unary_L = encode_unary(L)
    binary_n = bin(n)[2:]
    suffix = binary_n[1:]
    return unary_L + suffix

def decode_elias_gamma(bit_stream: str) -> tuple[int, str]:
    num_ones = 0
    while num_ones < len(bit_stream) and bit_stream[num_ones] == '1': num_ones += 1
    if num_ones >= len(bit_stream): raise ValueError("Invalid Elias Gamma: Unary prefix incomplete.")
    L = num_ones + 1
    unary_part_len = L
    suffix_len = L - 1
    total_len = unary_part_len + suffix_len
    if total_len > len(bit_stream): raise ValueError(f"Invalid Elias Gamma: Not enough bits for suffix. Need {suffix_len}, have {len(bit_stream) - unary_part_len}.")
    suffix = bit_stream[unary_part_len : total_len]
    binary_n_str = '1' + suffix
    decoded_n = int(binary_n_str, 2)
    remaining_stream = bit_stream[total_len:]
    return decoded_n, remaining_stream

# --- Fibonacci Coding ---
def encode_fibonacci(n: int) -> str:
    if not isinstance(n, int) or n < 1: raise ValueError(f"Fibonacci input must be a positive integer (n >= 1). Got: {n}")
    _get_fibonacci_numbers_until(n)
    sorted_fib_indices = sorted(_fib_cache.keys())
    fib_values_with_indices = [(idx, _fib_cache[idx]) for idx in sorted_fib_indices if _fib_cache[idx] <= n]
    if not fib_values_with_indices:
        if n == 1: k = 2
        else: raise RuntimeError(f"Cannot find Fibonacci number <= {n}.")
    else: k = fib_values_with_indices[-1][0]
    codeword_bits = ['0'] * (k - 1)
    remainder = n
    for idx in range(k, 1, -1):
        current_fib = _get_fib_number(idx)
        if current_fib <= remainder:
            codeword_index = idx - 2
            codeword_bits[codeword_index] = '1'
            remainder -= current_fib
    return "".join(codeword_bits) + "1"

def decode_fibonacci(bit_stream: str) -> tuple[int, str]:
    end_marker_pos = bit_stream.find('11')
    if end_marker_pos == -1: raise ValueError("Invalid Fibonacci stream: terminating '11' not found.")
    codeword_part = bit_stream[:end_marker_pos + 1]
    decoded_n = 0
    for i in range(len(codeword_part)):
        if codeword_part[i] == '1':
            fib_val = _get_fib_number(i + 2)
            decoded_n += fib_val
    remaining_stream = bit_stream[end_marker_pos + 2:]
    return decoded_n, remaining_stream

# --- Function to run basic tests for the coders (Optional) ---
def run_coding_tests():
    # ... (implementation remains the same as before) ...
    print("--- Basic Coding Tests ---")
    # (Unary, Gamma, Fibonacci test code here)
    print("--- Basic Coding Tests Completed ---")

# ==============================================================================
# Part 2: Simulation of Inverted Index and Encoding
# ==============================================================================

# Store simulation results globally or pass them around
simulation_results = {}

def run_simulation():
    """Runs the inverted index simulation and encoding process."""
    global simulation_results
    print("\n--- Running Inverted Index Simulation and Encoding ---")
    # --- Parameters ---
    VOCAB_SIZE = 1000
    NUM_DOCS = 10000
    NUM_PAIRS = 1_000_000 # One million unique (word, docID) pairs
    MIN_WORD_LEN = 5
    MAX_WORD_LEN = 10

    # --- 1. Generate Vocabulary ---
    print(f"Generating {VOCAB_SIZE} unique random words...")
    start_time = time.time()
    vocabulary = set()
    while len(vocabulary) < VOCAB_SIZE:
        length = random.randint(MIN_WORD_LEN, MAX_WORD_LEN)
        word = ''.join(random.choices(string.ascii_lowercase, k=length))
        vocabulary.add(word)
    vocabulary = list(vocabulary)
    print(f"Vocabulary generated in {time.time() - start_time:.2f} seconds.")

    # --- 2. Generate Unique (Word, DocID) Pairs ---
    print(f"Generating {NUM_PAIRS} unique (word, docID) pairs...")
    start_time = time.time()
    unique_pairs = set()
    doc_id_range = range(1, NUM_DOCS + 1)
    max_possible_pairs = VOCAB_SIZE * NUM_DOCS
    if NUM_PAIRS > max_possible_pairs: raise ValueError(f"Cannot generate {NUM_PAIRS} unique pairs.")
    while len(unique_pairs) < NUM_PAIRS:
        word = random.choice(vocabulary)
        doc_id = random.choice(doc_id_range)
        unique_pairs.add((word, doc_id))
        if len(unique_pairs) % 250000 == 0 and len(unique_pairs) > 0: print(f"  Generated {len(unique_pairs)} pairs...")
    print(f"Pairs generated in {time.time() - start_time:.2f} seconds.")

    # --- 3. Build Inverted Index (Store original lists for comparison later) ---
    print("Building inverted index (storing original sorted lists)...")
    start_time = time.time()
    inverted_index = defaultdict(list)
    for word, doc_id in unique_pairs:
        inverted_index[word].append(doc_id)
    # Store sorted lists directly for later use in size/speed comparison
    original_sorted_lists = {}
    for word in vocabulary:
        ids = inverted_index.get(word, [])
        if ids:
            original_sorted_lists[word] = sorted(ids)
        else:
            original_sorted_lists[word] = []
    print(f"Inverted index built and sorted in {time.time() - start_time:.2f} seconds.")
    print(f"Index contains {len(original_sorted_lists)} terms.")

    # --- 4. & 5. Calculate d-gaps and Encode ---
    print("Calculating d-gaps and encoding posting lists...")
    start_time = time.time()
    encoded_index_data = {}
    total_encoded_bits = {'unary': 0, 'gamma': 0, 'fibonacci': 0}
    total_d_gaps_count = 0
    processed_words_count = 0

    for word, sorted_doc_ids in original_sorted_lists.items():
        processed_words_count += 1
        if processed_words_count % 100 == 0: print(f"  Encoding word {processed_words_count}/{len(vocabulary)}...")

        if not sorted_doc_ids:
            encoded_index_data[word] = {'d_gaps': [], 'unary': '', 'gamma': '', 'fibonacci': ''}
            continue

        # Calculate d-gaps
        d_gaps = []
        d_gaps.append(sorted_doc_ids[0])
        for i in range(1, len(sorted_doc_ids)):
            gap = sorted_doc_ids[i] - sorted_doc_ids[i-1]
            assert gap > 0
            d_gaps.append(gap)
        total_d_gaps_count += len(d_gaps)

        # Encode d-gaps
        encoded_unary_list = [encode_unary(gap) for gap in d_gaps]
        encoded_gamma_list = [encode_elias_gamma(gap) for gap in d_gaps]
        encoded_fibonacci_list = [encode_fibonacci(gap) for gap in d_gaps]

        final_unary_code = "".join(encoded_unary_list)
        final_gamma_code = "".join(encoded_gamma_list)
        final_fibonacci_code = "".join(encoded_fibonacci_list)

        # Store results
        encoded_index_data[word] = {
            'd_gaps': d_gaps,
            'unary': final_unary_code,
            'gamma': final_gamma_code,
            'fibonacci': final_fibonacci_code
        }
        total_encoded_bits['unary'] += len(final_unary_code)
        total_encoded_bits['gamma'] += len(final_gamma_code)
        total_encoded_bits['fibonacci'] += len(final_fibonacci_code)

    print(f"Encoding finished in {time.time() - start_time:.2f} seconds.")

    # Store results for comparison function
    simulation_results['original_sorted_lists'] = original_sorted_lists
    simulation_results['encoded_index_data'] = encoded_index_data
    simulation_results['total_encoded_bits'] = total_encoded_bits
    simulation_results['total_d_gaps_count'] = total_d_gaps_count
    simulation_results['num_pairs'] = NUM_PAIRS
    simulation_results['num_docs'] = NUM_DOCS

    print("\n--- Simulation and Encoding Completed ---")


# ==============================================================================
# Part 3: Comparison of Size and Speed
# ==============================================================================

def search_uncompressed(sorted_doc_ids: list[int], target_doc_id: int) -> bool:
    """Searches for target_doc_id in a sorted list using binary search."""
    if not sorted_doc_ids:
        return False
    # Find insertion point for target_doc_id
    pos = bisect.bisect_left(sorted_doc_ids, target_doc_id)
    # Check if the element at insertion point is the target
    return pos < len(sorted_doc_ids) and sorted_doc_ids[pos] == target_doc_id

def search_compressed(encoded_stream: str, target_doc_id: int, decode_func) -> bool:
    """Searches for target_doc_id by decoding a compressed stream."""
    current_doc_id = 0
    remaining_stream = encoded_stream
    found = False

    while remaining_stream:
        try:
            gap, remaining_stream = decode_func(remaining_stream)
            current_doc_id += gap

            if current_doc_id == target_doc_id:
                found = True
                break
            elif current_doc_id > target_doc_id:
                # Exceeded target, won't find it later
                found = False
                break
        except ValueError:
            # Error during decoding (e.g., incomplete stream), target not found
            # print(f"Warning: Decoding error during search for {target_doc_id}")
            found = False
            break
        except Exception as e:
            # Other unexpected errors
            print(f"Unexpected error during decoding search: {e}")
            found = False
            break

    return found

def run_comparison():
    """Compares size and search speed based on simulation results."""
    global simulation_results
    if not simulation_results:
        print("Error: Simulation results not found. Run simulation first.")
        return

    original_sorted_lists = simulation_results['original_sorted_lists']
    encoded_index_data = simulation_results['encoded_index_data']
    total_encoded_bits = simulation_results['total_encoded_bits']
    total_d_gaps_count = simulation_results['total_d_gaps_count']
    num_docs = simulation_results['num_docs']

    print("\n--- Comparison of Size and Speed ---")

    # --- 1. Size Comparison ---
    print("\n--- Size Comparison ---")
    total_uncompressed_chars = 0
    delimiter = "," # Use comma as delimiter for text representation
    for word, sorted_ids in original_sorted_lists.items():
        if sorted_ids:
            # Convert numbers to strings, join with delimiter
            text_repr = delimiter.join(map(str, sorted_ids))
            total_uncompressed_chars += len(text_repr)

    print(f"Uncompressed (text, comma-separated): {total_uncompressed_chars:>15,} chars/bytes")
    print(f"Compressed Unary:                   {total_encoded_bits['unary']:>15,} bits")
    print(f"Compressed Gamma:                   {total_encoded_bits['gamma']:>15,} bits")
    print(f"Compressed Fibonacci:               {total_encoded_bits['fibonacci']:>15,} bits")

    print("\nCompression Ratios (Compressed Size / Uncompressed Size):")
    if total_uncompressed_chars > 0:
        # Assuming 1 char = 1 byte = 8 bits for rough comparison
        # More accurate would depend on encoding (ASCII, UTF-8)
        uncompressed_bits_approx = total_uncompressed_chars * 8
        print(f"  Unary / Uncompressed:     {total_encoded_bits['unary'] / uncompressed_bits_approx :>8.4f}")
        print(f"  Gamma / Uncompressed:     {total_encoded_bits['gamma'] / uncompressed_bits_approx :>8.4f}")
        print(f"  Fibonacci / Uncompressed: {total_encoded_bits['fibonacci'] / uncompressed_bits_approx :>8.4f}")
    else:
        print("  Cannot calculate ratios: Uncompressed size is zero.")

    # --- 2. Speed Comparison ---
    print("\n--- Search Speed Comparison ---")

    # Select words with different posting list lengths
    words_by_length = sorted(original_sorted_lists.keys(), key=lambda w: len(original_sorted_lists[w]))
    selected_words = {}
    if len(words_by_length) > 0:
      selected_words['short'] = next((w for w in words_by_length if 10 <= len(original_sorted_lists[w]) <= 50), words_by_length[0])
    if len(words_by_length) > 100: # Check if enough words exist
       median_idx = len(words_by_length) // 2
       selected_words['medium'] = next((w for w in words_by_length[median_idx:] if 100 <= len(original_sorted_lists[w]) <= 500), words_by_length[median_idx])
    if len(words_by_length) > 200:
       selected_words['long'] = next((w for w in reversed(words_by_length) if len(original_sorted_lists[w]) >= 1000), words_by_length[-1])


    if not selected_words:
        print("Could not find suitable words for speed testing. Skipping.")
        return

    num_search_repetitions = 100 # Number of times to repeat each search for timing

    print(f"Testing search speed (repeating {num_search_repetitions} times for each case):")

    results = {} # Store timing results

    for list_type, word in selected_words.items():
        print(f"\nTesting word '{word}' (List type: {list_type}, Length: {len(original_sorted_lists[word])})")
        sorted_ids = original_sorted_lists[word]
        encoded_data = encoded_index_data[word]

        if not sorted_ids:
            print("  Skipping empty list.")
            continue

        # Targets: first ID, middle ID, last ID, a non-existing ID
        targets = []
        targets.append(sorted_ids[0]) # First
        targets.append(sorted_ids[len(sorted_ids) // 2]) # Middle
        targets.append(sorted_ids[-1]) # Last
        # Non-existing: find a gap and pick a number in between, or one after last
        non_existing_target = sorted_ids[-1] + 1 if sorted_ids[-1] < num_docs else sorted_ids[0] - 1
        if non_existing_target <= 0: non_existing_target = (sorted_ids[0] + sorted_ids[1]) // 2 if len(sorted_ids) > 1 else sorted_ids[0] + 5 # Heuristic
        if non_existing_target in sorted_ids: non_existing_target +=1 # Ensure it doesn't exist
        targets.append(non_existing_target)


        timings = {'uncompressed': [], 'unary': [], 'gamma': [], 'fibonacci': []}

        for target_id in targets:
            exists = target_id in sorted_ids # Check beforehand for label
            print(f"  Searching for DocID: {target_id} ({'exists' if exists else 'does not exist'})")

            # Uncompressed Timing
            start = time.perf_counter()
            for _ in range(num_search_repetitions):
                found = search_uncompressed(sorted_ids, target_id)
            end = time.perf_counter()
            timings['uncompressed'].append((end - start) / num_search_repetitions)
            # print(f"    Uncompressed: {timings['uncompressed'][-1]:.6f} s (Found: {found})")

            # Unary Timing
            start = time.perf_counter()
            for _ in range(num_search_repetitions):
                found = search_compressed(encoded_data['unary'], target_id, decode_unary)
            end = time.perf_counter()
            timings['unary'].append((end - start) / num_search_repetitions)
            # print(f"    Unary:        {timings['unary'][-1]:.6f} s (Found: {found})")

            # Gamma Timing
            start = time.perf_counter()
            for _ in range(num_search_repetitions):
                found = search_compressed(encoded_data['gamma'], target_id, decode_elias_gamma)
            end = time.perf_counter()
            timings['gamma'].append((end - start) / num_search_repetitions)
            # print(f"    Gamma:        {timings['gamma'][-1]:.6f} s (Found: {found})")

            # Fibonacci Timing
            start = time.perf_counter()
            for _ in range(num_search_repetitions):
                found = search_compressed(encoded_data['fibonacci'], target_id, decode_fibonacci)
            end = time.perf_counter()
            timings['fibonacci'].append((end - start) / num_search_repetitions)
            # print(f"    Fibonacci:    {timings['fibonacci'][-1]:.6f} s (Found: {found})")

        # Average times for this word/list_type
        avg_times = {method: sum(t)/len(t) for method, t in timings.items()}
        results[list_type] = avg_times
        print(f"  Average search times (seconds per search):")
        print(f"    Uncompressed: {avg_times['uncompressed']:.6f}")
        print(f"    Unary:        {avg_times['unary']:.6f}")
        print(f"    Gamma:        {avg_times['gamma']:.6f}")
        print(f"    Fibonacci:    {avg_times['fibonacci']:.6f}")

    print("\n--- Speed Comparison Summary ---")
    for list_type, avg_times in results.items():
         print(f"List Type: {list_type.capitalize()}")
         print(f"  Uncompressed: {avg_times['uncompressed']:.6f} s")
         print(f"  Unary:        {avg_times['unary']:.6f} s (Ratio to uncompressed: {avg_times['unary']/avg_times['uncompressed']:.2f}x)" if avg_times['uncompressed'] > 0 else "")
         print(f"  Gamma:        {avg_times['gamma']:.6f} s (Ratio to uncompressed: {avg_times['gamma']/avg_times['uncompressed']:.2f}x)" if avg_times['uncompressed'] > 0 else "")
         print(f"  Fibonacci:    {avg_times['fibonacci']:.6f} s (Ratio to uncompressed: {avg_times['fibonacci']/avg_times['uncompressed']:.2f}x)" if avg_times['uncompressed'] > 0 else "")


# --- Main Execution Block ---
if __name__ == '__main__':
    # Optionally run the basic coder tests first
    # run_coding_tests()

    # Run the main simulation (stores results in global 'simulation_results')
    run_simulation()

    # Run the comparison based on the simulation results
    run_comparison()