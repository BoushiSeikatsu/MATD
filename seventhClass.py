import numpy as np
import os
import time
from typing import Tuple, Dict, Optional, List
from collections import deque
# For SVD
import numpy.linalg as la # Use la for Linear Algebra functions

# --- Configuration ---
# (Keep previous configurations)
EMBEDDING_DIR = '.'
EN_EMBEDDING_FILE = 'wiki.en.vec'
CS_EMBEDDING_FILE = 'wiki.cs.vec'
TRAIN_FILE = 'cs-en.0-5000.txt'
TEST_FILE = 'cs-en.5000-6500.txt'
EXPECTED_SEPARATOR = '\t'
EMBEDDING_DTYPE = np.float32

# --- Functions ---

# load_embeddings(...) # Unchanged from previous version
# create_translation_matrices(...) # Unchanged from previous version
# (Keeping the robust versions from the last iteration)
def load_embeddings(filepath: str,
                    expected_dim: Optional[int] = None,
                    limit: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Optional[int]]:
    """Loads word embeddings from a .vec file efficiently."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: Embedding file not found at {filepath}")

    print(f"Loading embeddings from {filepath}...")
    start_time = time.time()
    embeddings: Dict[str, np.ndarray] = {}
    actual_dim: Optional[int] = None
    header_info_logged = False

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline()
            parts = first_line.strip().split()
            try:
                num_vectors = int(parts[0])
                file_dim = int(parts[1])
                print(f"  File header indicates {num_vectors} vectors of dimension {file_dim}.")
                header_info_logged = True
                if expected_dim is not None and file_dim != expected_dim:
                     print(f"  Warning: Header dimension {file_dim} differs from expected {expected_dim}.")
                actual_dim = file_dim
            except (ValueError, IndexError):
                print("  No valid header found or format differs. Will infer dimension from first vector.")
                f.seek(0)

            count = 0
            skipped_lines = 0
            processed_lines = 0
            for line in f:
                processed_lines += 1
                parts = line.strip().split(' ')
                if len(parts) < 2:
                    skipped_lines += 1
                    continue

                word = parts[0]
                if not header_info_logged and word.isdigit() and len(parts) == 2 and parts[1].isdigit():
                     try:
                        int(word); int(parts[1])
                        # print(f"  Skipping likely header-like line: {line.strip()}") # Less verbose
                        skipped_lines += 1
                        continue
                     except ValueError:
                         pass

                try:
                    vector = np.array(parts[1:], dtype=EMBEDDING_DTYPE)

                    if actual_dim is None:
                        actual_dim = len(vector)
                        print(f"  Inferred embedding dimension: {actual_dim}")
                        if expected_dim is not None and actual_dim != expected_dim:
                            print(f"  Warning: Inferred dimension {actual_dim} differs from expected {expected_dim}.")

                    current_dim = len(vector)
                    if current_dim == actual_dim:
                        embeddings[word] = vector
                        count += 1
                    elif expected_dim is not None and current_dim == expected_dim:
                        if actual_dim != expected_dim:
                            print(f"  Note: Vector for '{word}' has expected dimension {expected_dim}. Overriding inferred/header dim {actual_dim} for consistency.")
                            actual_dim = expected_dim
                        embeddings[word] = vector
                        count += 1
                    else:
                         skipped_lines += 1
                         if skipped_lines % 1000 == 0: # Report less often
                             print(f"  Warning: Skipped {skipped_lines} lines due to dim mismatch (e.g., word '{word}', expected {actual_dim}, found {current_dim})")
                         continue

                except ValueError:
                    skipped_lines += 1
                    if skipped_lines % 1000 == 0:
                         print(f"  Warning: Skipped {skipped_lines} lines due to parsing errors (e.g., word '{word}')")
                    continue

                if processed_lines % 200000 == 0 and processed_lines > 0:
                    print(f"  Processed {processed_lines} lines...")

                if limit and count >= limit:
                    print(f"  Reached loading limit of {limit} vectors.")
                    break

    except Exception as e:
        print(f"\nAn error occurred while loading {filepath}: {e}")
        raise

    end_time = time.time()
    print(f"Successfully loaded {len(embeddings)} vectors.")
    if skipped_lines > 0:
        print(f"Skipped {skipped_lines} lines out of {processed_lines} total lines due to format/dimension/parsing issues.")
    print(f"Loading took {end_time - start_time:.2f} seconds.\n")

    if not embeddings:
         print(f"Warning: No embeddings were loaded from {filepath}. Please check the file format and content.")
         return {}, None

    if actual_dim is None and embeddings:
        first_key = next(iter(embeddings))
        actual_dim = embeddings[first_key].shape[0]
        print(f"Setting dimension based on first loaded vector ('{first_key}'): {actual_dim}")

    return embeddings, actual_dim

def create_translation_matrices(filepath: str,
                                embeddings_en: Dict[str, np.ndarray],
                                embeddings_cs: Dict[str, np.ndarray],
                                en_dim: int,
                                cs_dim: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Loads translation pairs and creates aligned source (En) and target (Cs) embedding matrices."""
    if not os.path.exists(filepath):
        print(f"Error: Translation file not found at {filepath}")
        return None
    if not embeddings_en or not embeddings_cs:
        print("Error: Embeddings dictionaries are empty or invalid.")
        return None

    print(f"Processing translation file: {filepath}...")
    start_time = time.time()

    source_vectors: List[np.ndarray] = []
    target_vectors: List[np.ndarray] = []
    pairs_processed = 0
    pairs_found = 0
    not_found_en = 0
    not_found_cs = 0
    skipped_lines = 0
    format_warnings = 0
    dim_warnings = 0
    line_count = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line_count = i + 1
            line = line.strip()
            if not line:
                 skipped_lines += 1
                 continue

            parts = line.split(EXPECTED_SEPARATOR)
            if len(parts) != 2:
                parts = line.split(' ', 1)
                if len(parts) != 2:
                    if format_warnings < 5:
                        print(f"  Warning: Skipping line {i+1}. Expected 2 parts separated by '{EXPECTED_SEPARATOR}' (or space), found {len(parts)}: '{line}'")
                    elif format_warnings == 5:
                         print("  (Supressing further line format warnings)")
                    skipped_lines += 1
                    format_warnings += 1
                    continue

            en_word = parts[0]
            cs_word = parts[1]
            pairs_processed += 1

            en_vec = embeddings_en.get(en_word)
            cs_vec = embeddings_cs.get(cs_word)

            found_en = en_vec is not None
            found_cs = cs_vec is not None
            dim_ok_en = found_en and en_vec.shape[0] == en_dim
            dim_ok_cs = found_cs and cs_vec.shape[0] == cs_dim

            if dim_ok_en and dim_ok_cs:
                source_vectors.append(en_vec)
                target_vectors.append(cs_vec)
                pairs_found += 1
            else:
                if not found_en: not_found_en += 1
                elif not dim_ok_en:
                    not_found_en += 1 # Count as not found if dim wrong
                    if dim_warnings < 5: print(f"  Warning: Dim mismatch for EN word '{en_word}' ({en_vec.shape[0]} vs {en_dim})")
                    dim_warnings+=1

                if not found_cs: not_found_cs += 1
                elif not dim_ok_cs:
                    not_found_cs += 1 # Count as not found if dim wrong
                    if dim_warnings < 5: print(f"  Warning: Dim mismatch for CS word '{cs_word}' ({cs_vec.shape[0]} vs {cs_dim})")
                    dim_warnings+=1
                if dim_warnings == 5: print("  (Supressing further dimension mismatch warnings)")


    end_time = time.time()
    print(f"Processing took {end_time - start_time:.2f} seconds.")
    print(f"  Total lines checked: {line_count}")
    print(f"  Pairs processed (correct format): {pairs_processed}")
    if skipped_lines > 0:
        print(f"  Lines skipped due to format/emptiness: {skipped_lines}")
    print(f"  Valid translation pairs found (both words in embeddings with correct dims): {pairs_found}")
    print(f"  Source (En) words invalid (not found or wrong dim): {not_found_en}")
    print(f"  Target (Cs) words invalid (not found or wrong dim): {not_found_cs}")

    if not source_vectors or not target_vectors:
        print("Warning: No valid pairs found to create matrices.")
        return None

    try:
        # Ensure dtype is float32 for consistency and potential overflow avoidance
        X = np.array(source_vectors, dtype=EMBEDDING_DTYPE)
        Y = np.array(target_vectors, dtype=EMBEDDING_DTYPE)
    except Exception as e:
        print(f"Error converting lists to NumPy arrays: {e}")
        return None

    print(f"  Created matrices: X shape {X.shape}, Y shape {Y.shape}\n")
    if X.shape[0] != Y.shape[0] or X.shape[0] != pairs_found:
        print(f"CRITICAL WARNING: Matrix row count ({X.shape[0]}) mismatch with Y ({Y.shape[0]}) or pairs found ({pairs_found}). Check data.")
    if X.shape[1] != en_dim or Y.shape[1] != cs_dim:
         print(f"CRITICAL WARNING: Matrix column count mismatch. X:{X.shape[1]} vs en_dim:{en_dim}, Y:{Y.shape[1]} vs cs_dim:{cs_dim}")

    return X, Y


# --- UPDATED Loss and Gradient Functions ---

def compute_loss(X: np.ndarray, W: np.ndarray, Y: np.ndarray, scale_by_n: bool = True) -> float:
    """
    Computes the squared Frobenius norm: || X @ W.T - Y ||_F^2
    Optionally scales by the number of samples N.
    """
    if X.shape[1] != W.shape[1]:
        raise ValueError(f"Dim mismatch X@W.T: X:{X.shape}, W:{W.shape}")
    if X.shape[0] != Y.shape[0]:
         raise ValueError(f"Row mismatch X vs Y: X:{X.shape[0]}, Y:{Y.shape[0]}")
    if W.shape[0] != Y.shape[1]:
         raise ValueError(f"Dim mismatch result: W:{W.shape[0]}, Y:{Y.shape[1]}")

    N = X.shape[0]
    if N == 0: return 0.0 # Avoid division by zero

    predicted_Y = X @ W.T
    difference = predicted_Y - Y
    # Use float64 for summation temporarily to avoid intermediate overflow
    # before scaling, then convert back.
    loss_sum = np.sum(np.square(difference, dtype=np.float64))

    # Scale the loss by N (average loss per sample)
    if scale_by_n:
        # Avoid division by zero just in case N=0 slipped through (shouldn't happen)
        loss = loss_sum / N if N > 0 else 0.0
    else:
        loss = loss_sum

    # Check for non-finite loss value before returning
    if not np.isfinite(loss):
        print(f"Warning: compute_loss resulted in non-finite value ({loss}). Check inputs/intermediate calculations.")
        # Depending on desired behavior, could return float('inf') or raise an error
        # Let's return inf for now as the training loop checks for this.
        return float('inf')


    return float(loss) # Return standard float


def compute_gradient(X: np.ndarray, W: np.ndarray, Y: np.ndarray, scale_by_n: bool = True) -> np.ndarray:
    """
    Computes the gradient of the loss function w.r.t. W.
    Gradient = 2 * (X @ W.T - Y).T @ X
    Optionally scales by the number of samples N.
    """
    N = X.shape[0]
    if N == 0:
        # Return zero gradient of the correct shape
        return np.zeros_like(W, dtype=EMBEDDING_DTYPE)

    predicted_Y = X @ W.T
    difference = predicted_Y - Y # Shape (N, D_target)

    # Calculate gradient
    # Perform calculation in float32, should be okay if difference isn't gigantic
    gradient = difference.T @ X # Shape (D_target, N) @ (N, D_source) -> (D_target, D_source)

    # Scale the gradient by N and include the factor of 2
    if scale_by_n:
         gradient = (2.0 / N) * gradient
    else:
         gradient = 2.0 * gradient

    return gradient.astype(EMBEDDING_DTYPE) # Ensure output is float32


# --- NEW: Translation and Evaluation Functions ---

def prepare_target_data(embeddings_target: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Converts target embedding dict to matrix, word list, and norms for fast lookup."""
    print("Preparing target data for similarity search...")
    start_time = time.time()
    target_words = list(embeddings_target.keys())
    target_vectors = np.array(list(embeddings_target.values()), dtype=EMBEDDING_DTYPE)

    # Calculate norms (use float64 for stability, add epsilon)
    target_norms = np.linalg.norm(target_vectors.astype(np.float64), axis=1)
    target_norms[target_norms < 1e-9] = 1e-9 # Avoid division by zero

    print(f"Target data prepared: {target_vectors.shape[0]} words, {target_vectors.shape[1]} dims. Time: {time.time() - start_time:.2f}s")
    return target_vectors, target_words, target_norms.astype(EMBEDDING_DTYPE)


def translate_word(source_word: str,
                   W: np.ndarray,
                   embeddings_source: Dict[str, np.ndarray],
                   target_vectors: np.ndarray, # Precomputed target matrix
                   target_words: List[str],    # Precomputed target word list
                   target_norms: np.ndarray,   # Precomputed target norms
                   k: int = 5) -> Optional[List[Tuple[str, float]]]:
    """
    Translates a source word using the transformation matrix W and finds k nearest neighbors
    in the target space using cosine similarity.

    Args:
        source_word: The English word to translate.
        W: The trained transformation matrix (D_target, D_source).
        embeddings_source: Dictionary of source (English) embeddings.
        target_vectors: Matrix of target (Czech) embeddings (N_target, D_target).
        target_words: List of target (Czech) words corresponding to rows in target_vectors.
        target_norms: Precomputed L2 norms of target vectors.
        k: Number of nearest neighbors to return.

    Returns:
        A list of k (word, similarity_score) tuples, or None if the source word is not found.
    """
    source_vec = embeddings_source.get(source_word)
    if source_vec is None:
        # print(f"Warning: Source word '{source_word}' not found in embeddings.")
        return None

    # Ensure source_vec is 1D or reshape for matmul
    source_vec_1d = source_vec.reshape(1, -1) # Shape (1, D_source)

    # Transform source vector: (1, D_source) @ (D_source, D_target) -> (1, D_target)
    predicted_vec = source_vec_1d @ W.T

    # Calculate cosine similarities efficiently
    # predicted_vec shape (1, D_target), target_vectors shape (N_target, D_target)
    # Dot product: (1, D_target) @ (D_target, N_target) -> (1, N_target)
    dot_products = predicted_vec @ target_vectors.T

    # Calculate norm of the predicted vector (add epsilon)
    predicted_norm = np.linalg.norm(predicted_vec.astype(np.float64))
    predicted_norm = max(predicted_norm, 1e-9)

    # Cosine similarity = dot(a, b) / (norm(a) * norm(b))
    similarities = dot_products[0] / (predicted_norm * target_norms)

    # Get indices of top k similarities (argsort gives ascending, so take last k)
    # Using argpartition is faster if k is much smaller than N_target
    if k < len(target_words) / 10 : # Heuristic threshold
        top_k_indices = np.argpartition(similarities, -k)[-k:]
        # Sort these k indices by similarity
        top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]
    else: # Use argsort for larger k or if argpartition isn't significantly faster
        top_k_indices = np.argsort(similarities)[-k:][::-1] # Get top k indices, highest first

    # Create list of (word, score) tuples
    results = [(target_words[i], float(similarities[i])) for i in top_k_indices]

    return results


def evaluate_accuracy(test_filepath: str,
                      W: np.ndarray,
                      embeddings_source: Dict[str, np.ndarray],
                      embeddings_target: Dict[str, np.ndarray],
                      k: int = 5):
    """
    Evaluates the translation accuracy (Top-1, Top-k) on a test file.

    Args:
        test_filepath: Path to the test file (e.g., cs-en.5000-6500.txt).
        W: The trained transformation matrix.
        embeddings_source: Dictionary of source (English) embeddings.
        embeddings_target: Dictionary of target (Czech) embeddings.
        k: The 'k' for top-k accuracy calculation (e.g., 5).
    """
    if not os.path.exists(test_filepath):
        print(f"Error: Test file not found at {test_filepath}")
        return

    print(f"\n--- Evaluating Accuracy on {test_filepath} ---")

    # Precompute target data for efficient similarity calculation
    target_vectors, target_words, target_norms = prepare_target_data(embeddings_target)
    if target_vectors.size == 0:
        print("Error: Target embedding data is empty. Cannot evaluate.")
        return

    top1_correct = 0
    topk_correct = 0
    evaluated_pairs = 0
    skipped_pairs = 0

    with open(test_filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line: continue

            parts = line.split(EXPECTED_SEPARATOR)
            if len(parts) != 2:
                parts = line.split(' ', 1)
                if len(parts) != 2:
                    skipped_pairs += 1
                    continue

            en_word, true_cs_word = parts[0], parts[1]

            # Check if both words exist in respective embeddings for a fair evaluation
            if en_word not in embeddings_source or true_cs_word not in embeddings_target:
                 skipped_pairs += 1
                 continue

            evaluated_pairs += 1

            # Perform translation
            predictions = translate_word(en_word, W, embeddings_source,
                                         target_vectors, target_words, target_norms, k=k)

            if predictions: # If translation succeeded
                predicted_words = [word for word, score in predictions]

                # Check Top-1 Accuracy
                if predicted_words[0] == true_cs_word:
                    top1_correct += 1

                # Check Top-k Accuracy
                if true_cs_word in predicted_words:
                    topk_correct += 1

            if evaluated_pairs % 200 == 0 and evaluated_pairs > 0:
                 print(f"  Processed {evaluated_pairs} evaluation pairs...")


    print(f"--- Evaluation Results ---")
    print(f"Total pairs checked in file: {i+1}")
    print(f"Pairs skipped (format error or OOV): {skipped_pairs}")
    print(f"Pairs evaluated (both words in vocab): {evaluated_pairs}")

    if evaluated_pairs > 0:
        accuracy_top1 = (top1_correct / evaluated_pairs) * 100
        accuracy_topk = (topk_correct / evaluated_pairs) * 100
        print(f"Top-1 Accuracy: {top1_correct}/{evaluated_pairs} = {accuracy_top1:.2f}%")
        print(f"Top-{k} Accuracy: {topk_correct}/{evaluated_pairs} = {accuracy_topk:.2f}%")
    else:
        print("No valid pairs could be evaluated.")


# --- UPDATED Training Function with Orthogonality Constraint ---

def train(X_train: np.ndarray, Y_train: np.ndarray,
          X_test: np.ndarray, Y_test: np.ndarray,
          num_epochs: int,
          initial_learning_rate: float, lr_decay_rate: float = 0.0,
          scale_loss: bool = True, gradient_clip_threshold: Optional[float] = 1.0,
          enforce_orthogonality: bool = True, # <-- New Flag
          orthogonalization_interval: int = 1, # <-- How often to orthogonalize
          verbose_every: int = 50, convergence_window: int = 10,
          convergence_tolerance: float = 1e-7) -> Optional[np.ndarray]:
    """
    Trains W using gradient descent with enhancements including optional orthogonality.
    """
    # --- Validation Checks ---
    if not all([X_train.size > 0, Y_train.size > 0, X_test.size > 0, Y_test.size > 0]):
        print("Error: One or more data matrices are empty.")
        return None

    N_train, d_source = X_train.shape
    d_target = Y_train.shape[1]
    # Orthogonality only makes sense if dimensions match
    if enforce_orthogonality and d_source != d_target:
        print(f"Warning: Orthogonality enforced but dimensions mismatch ({d_source} vs {d_target}). Results may be suboptimal.")
        # It will still project to the nearest matrix where columns are orthonormal,
        # but it won't be strictly orthogonal if non-square.

    print(f"\n--- Starting Training ---")
    print(f"Train data: X={X_train.shape}, Y={Y_train.shape}")
    print(f"Test data:  X={X_test.shape}, Y={Y_test.shape}")
    print(f"Max Epochs: {num_epochs}")
    print(f"Initial Learning Rate: {initial_learning_rate:.1E}")
    if lr_decay_rate > 0:
        print(f"LR Decay Rate (Time-Based): {lr_decay_rate:.1E}")
    else:
        print("LR Decay: Disabled")
    print(f"Scale Loss/Gradient by N: {scale_loss}")
    print(f"Gradient Clipping Threshold: {gradient_clip_threshold if gradient_clip_threshold is not None else 'Disabled'}")
    print(f"Convergence Check: window={convergence_window} evaluations, tolerance={convergence_tolerance:.1E}")
    print(f"Evaluation Frequency: every {verbose_every} epochs")


    # Initialize W
    scale = np.sqrt(6.0 / (d_source + d_target))
    W = np.random.uniform(-scale, scale, (d_target, d_source)).astype(EMBEDDING_DTYPE)
    if enforce_orthogonality: # Start with an orthogonal matrix if enforcing
        try:
            U, _, Vt = la.svd(W)
            W = (U @ Vt).astype(EMBEDDING_DTYPE)
            print("Initialized W to an orthogonal matrix via SVD.")
        except Exception as e:
             print(f"Warning: Initial SVD for orthogonalization failed: {e}. Starting with random W.")
             if not np.all(np.isfinite(W)): return None # Check random W if SVD failed

    if not np.all(np.isfinite(W)):
        print("ERROR: W initialization failed.")
        return None
    print(f"Initialized W with shape: {W.shape}, dtype: {W.dtype}")

    test_loss_history = deque(maxlen=convergence_window)
    best_W = W.copy()
    best_test_loss = float('inf')
    start_time = time.time()
    nan_detected = False

    for epoch in range(num_epochs):
        current_lr = initial_learning_rate / (1.0 + lr_decay_rate * epoch) if lr_decay_rate > 0 else initial_learning_rate
        gradient = compute_gradient(X_train, W, Y_train, scale_by_n=scale_loss)

        if not np.all(np.isfinite(gradient)):
            print(f"\nERROR: NaN/Inf in gradient at epoch {epoch+1}. Stopping.")
            nan_detected = True; break

        if gradient_clip_threshold is not None:
            grad_norm = np.linalg.norm(gradient.astype(np.float64))
            if grad_norm > gradient_clip_threshold:
                gradient = gradient * (gradient_clip_threshold / (grad_norm + 1e-9))

        # --- Gradient Step ---
        W_updated = W - current_lr * gradient.astype(W.dtype)

        # --- Orthogonalization Step (Optional) ---
        if enforce_orthogonality and (epoch + 1) % orthogonalization_interval == 0:
             try:
                 # Perform SVD on the updated W
                 # Use float64 for SVD stability if needed, but usually float32 is fine
                 U, s, Vt = la.svd(W_updated.astype(np.float64), full_matrices=False)
                 # Reconstruct W as U @ Vt (closest orthogonal matrix)
                 W = (U @ Vt).astype(EMBEDDING_DTYPE)
             except Exception as e:
                  print(f"\nWarning: SVD for orthogonalization failed at epoch {epoch+1}: {e}. Skipping orthogonalization for this step.")
                  # Fall back to the standard gradient step result if SVD fails
                  W = W_updated
        else:
             # If not enforcing or not the interval, just use the updated W
             W = W_updated


        # --- Check for NaN/Inf in W ---
        if not np.all(np.isfinite(W)):
            print(f"\nERROR: NaN/Inf in W after update/orthogonalization at epoch {epoch+1}. Stopping.")
            nan_detected = True; break

        # --- Periodic Evaluation ---
        if (epoch + 1) % verbose_every == 0:
            # Use the current W (potentially orthogonalized) for loss calculation
            train_loss = compute_loss(X_train, W, Y_train, scale_by_n=scale_loss)
            test_loss = compute_loss(X_test, W, Y_test, scale_by_n=scale_loss)

            if not np.isfinite(train_loss) or not np.isfinite(test_loss):
                 print(f"\nERROR: NaN/Inf in loss at epoch {epoch+1}. Stopping.")
                 nan_detected = True; break

            elapsed_time = time.time() - start_time
            print(f"Epoch [{epoch+1}/{num_epochs}] LR: {current_lr:.1E} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f} | Time: {elapsed_time:.2f}s")
            start_time = time.time()

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_W = W.copy() # Store the potentially orthogonalized W

            test_loss_history.append(test_loss)
            if len(test_loss_history) == convergence_window:
                min_loss_in_window = min(test_loss_history)
                improvement = min_loss_in_window - test_loss
                if improvement < convergence_tolerance:
                    print(f"\nConvergence criteria met at epoch {epoch+1}.")
                    return best_W

    # --- End of loop ---
    # (Keep previous end-of-loop handling)
    if nan_detected: print("Training halted due to NaN/Inf.")
    elif epoch + 1 == num_epochs: print("--- Training Finished (Max Epochs) ---")
    else: print("--- Training Finished ---")
    print(f"Best test loss achieved: {best_test_loss:.6f}")
    return best_W



# --- Main Execution ---

if __name__ == "__main__":
    print("--- Starting Task ---")
    overall_start_time = time.time()

    # Load Embeddings
    en_embeddings, en_dim = load_embeddings(os.path.join(EMBEDDING_DIR, EN_EMBEDDING_FILE))
    cs_embeddings, cs_dim = load_embeddings(os.path.join(EMBEDDING_DIR, CS_EMBEDDING_FILE))

    if en_embeddings and cs_embeddings and en_dim is not None and cs_dim is not None:
        print(f"Embeddings loaded: EN dim={en_dim}, CS dim={cs_dim}")

        # Create Matrices
        train_data = create_translation_matrices(os.path.join(EMBEDDING_DIR, TRAIN_FILE),
                                                en_embeddings, cs_embeddings, en_dim, cs_dim)
        test_data = create_translation_matrices(os.path.join(EMBEDDING_DIR, TEST_FILE),
                                               en_embeddings, cs_embeddings, en_dim, cs_dim)

        if train_data is not None and test_data is not None:
            X_train, Y_train = train_data
            X_test, Y_test = test_data

            # --- Train W ---
            EPOCHS = 1000                 # Might converge faster now
            INITIAL_LEARNING_RATE = 1  # *** Can often use MUCH higher LR with orthogonality ***
            LR_DECAY_RATE = 0.0           # *** Often NO decay needed, or very little ***
            SCALE_LOSS_GRAD = True        # Keep scaling
            GRAD_CLIP = None              # *** Clipping might not be needed with orthogonalization ***
            ENFORCE_ORTHOGONALITY = True  # *** ENABLE ORTHOGONALITY ***
            ORTHOGONALIZATION_INTERVAL = 1 # Orthogonalize every step
            VERBOSE_EVERY = 20            # Check progress more often initially
            CONV_WINDOW = 10
            CONV_TOLERANCE = 1e-7         # Tolerance might need adjustment

            W_trained = train(X_train, Y_train, X_test, Y_test,
                              num_epochs=EPOCHS,
                              initial_learning_rate=INITIAL_LEARNING_RATE,
                              lr_decay_rate=LR_DECAY_RATE,
                              scale_loss=SCALE_LOSS_GRAD,
                              gradient_clip_threshold=GRAD_CLIP,
                              enforce_orthogonality=ENFORCE_ORTHOGONALITY, # Pass the flag
                              orthogonalization_interval=ORTHOGONALIZATION_INTERVAL,
                              verbose_every=VERBOSE_EVERY,
                              convergence_window=CONV_WINDOW,
                              convergence_tolerance=CONV_TOLERANCE)

            if W_trained is not None:
                print(f"\nFinal trained W matrix shape: {W_trained.shape}")
                final_train_loss = compute_loss(X_train, W_trained, Y_train, scale_by_n=SCALE_LOSS_GRAD)
                final_test_loss = compute_loss(X_test, W_trained, Y_test, scale_by_n=SCALE_LOSS_GRAD)
                print(f"Final Train Loss (scaled={SCALE_LOSS_GRAD}): {final_train_loss:.6f}")
                print(f"Final Test Loss (scaled={SCALE_LOSS_GRAD}): {final_test_loss:.6f}")

                try:
                    np.save("translation_matrix_W_best_ortho.npy", W_trained) # Save with new name
                    print("Best trained ORTHOGONAL transformation matrix W saved to translation_matrix_W_best_ortho.npy")
                except Exception as e:
                    print(f"Error saving W matrix: {e}")

                # --- Evaluate Accuracy ---
                evaluate_accuracy(os.path.join(EMBEDDING_DIR, TEST_FILE),
                                  W_trained,
                                  en_embeddings,
                                  cs_embeddings,
                                  k=5)

                # --- Example Translation ---
                print("\n--- Example Translations (with Orthogonal W) ---")
                example_words = ["king", "apple", "london", "computer", "love", "dog", "car"]
                # Re-use precomputed data if available, otherwise recompute
                if 'target_vectors_eval' not in locals():
                     target_vectors_eval, target_words_eval, target_norms_eval = prepare_target_data(cs_embeddings)

                for word in example_words:
                     translations = translate_word(word, W_trained, en_embeddings,
                                                   target_vectors_eval, target_words_eval, target_norms_eval, k=5)
                     if translations:
                         # Format for better readability
                         trans_str = ", ".join([f"{t[0]} ({t[1]:.3f})" for t in translations])
                         print(f"'{word}' -> [{trans_str}]")
                     else:
                         print(f"'{word}' -> (Not in source vocabulary)")

            else:
                 print("\nTraining failed or was halted due to errors. Cannot evaluate.")
        else:
            print("\nAborting training/evaluation due to issues creating translation matrices.")
    else:
        print("\nAborting matrix creation/training/evaluation due to issues loading embeddings or determining dimensions.")

    overall_end_time = time.time()
    print(f"\n--- Task Completed ---")
    print(f"Total execution time: {overall_end_time - overall_start_time:.2f} seconds.")