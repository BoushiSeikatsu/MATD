import os
import math
import sys

def split_text_file(filepath, output_dir, num_splits):
    """
    Splits a text file into a specified number of smaller files by lines.
    Attempts to read with UTF-8, falls back to cp1252 if decoding fails.

    Args:
        filepath (str): The full path to the input text file.
        output_dir (str): The directory where split files will be saved.
        num_splits (int): The target number of split files.

    Returns:
        bool: True if splitting was successful, False otherwise.
    """
    lines = []
    detected_encoding = 'utf-8' # Assume utf-8 initially

    # --- 1. Try reading with UTF-8, then fallback ---
    try:
        print(f"  Attempting to read {os.path.basename(filepath)} with UTF-8...")
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print("    Successfully read with UTF-8.")
    except UnicodeDecodeError:
        print(f"    UTF-8 decoding failed. Trying cp1252 (Windows Latin 1)...")
        try:
            detected_encoding = 'cp1252'
            with open(filepath, 'r', encoding=detected_encoding) as f:
                lines = f.readlines()
            print(f"    Successfully read with {detected_encoding}.")
        except Exception as e_fallback:
            # Add more fallbacks (like 'latin-1') here if needed
            print(f"Error: Could not read file {os.path.basename(filepath)} with UTF-8 or {detected_encoding}.")
            print(f"Fallback reading error: {e_fallback}")
            return False # Cannot proceed if reading fails completely
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return False
    except Exception as e_initial:
        print(f"An unexpected error occurred while trying to read {os.path.basename(filepath)}: {e_initial}")
        return False

    # --- Proceed only if lines were read successfully ---
    total_lines = len(lines)
    if total_lines == 0:
        print(f"  Skipping empty file or file read resulted in no lines: {os.path.basename(filepath)}")
        return True # Consider successful if file was empty or read yielded nothing

    # --- 2. Calculate lines per split file ---
    lines_per_file = math.ceil(total_lines / num_splits)
    lines_per_file = max(1, lines_per_file) # Ensure at least one line per file

    print(f"  Total lines: {total_lines}. Aiming for ~{num_splits} splits.")
    print(f"  Calculated lines per split file: {lines_per_file}")

    # --- 3. Generate split files ---
    file_count = 0
    base_filename, ext = os.path.splitext(os.path.basename(filepath))

    for i in range(0, total_lines, lines_per_file):
        chunk_lines = lines[i : i + lines_per_file]
        if not chunk_lines:
            continue

        file_count += 1
        part_num_str = str(file_count).zfill(2)
        output_filename = f"{base_filename}_part_{part_num_str}{ext}"
        output_filepath = os.path.join(output_dir, output_filename)

        print(f"    Creating {output_filename}...")
        try:
            # Write output files using UTF-8 is generally best practice
            # If the original was cp1252, Python will handle the conversion
            with open(output_filepath, 'w', encoding='utf-8') as outfile:
                outfile.writelines(chunk_lines)
        except IOError as e:
            print(f"    Error writing file {output_filename}: {e}")
            # return False # Optional: uncomment to stop on first write error
        except UnicodeEncodeError as e_encode:
            # This might happen if a character read with cp1252
            # cannot be represented in UTF-8 (rare for cp1252 -> utf8)
            print(f"    Warning: Could not write some characters to {output_filename} using UTF-8. {e_encode}")
            # Option: Try writing with the detected encoding instead, but UTF-8 is preferred
            # try:
            #     with open(output_filepath, 'w', encoding=detected_encoding) as outfile:
            #         outfile.writelines(chunk_lines)
            # except Exception as e_write_fallback:
            #      print(f"    Error writing file {output_filename} even with fallback encoding {detected_encoding}: {e_write_fallback}")

    print(f"  Successfully split into {file_count} files.")
    return True

# --- Main Script Logic (No changes needed here) ---
if __name__ == "__main__":
    folder_name = 'abc'
    target_splits_per_file = 25

    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_dir, folder_name)
    # folder_path = '/path/to/your/abc' # Use absolute path if needed

    print(f"Looking for .txt files in: {folder_path}")

    if not os.path.isdir(folder_path):
        print(f"Error: Directory '{folder_path}' not found.")
        sys.exit(1)

    files_processed_count = 0
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.txt') and '_part_' not in filename:
            original_filepath = os.path.join(folder_path, filename)
            if os.path.isfile(original_filepath):
                print(f"\nProcessing file: {filename}")
                success = split_text_file(original_filepath, folder_path, target_splits_per_file)
                if success:
                    files_processed_count += 1
            else:
                print(f"Skipping non-file item: {filename}")

    if files_processed_count == 0:
        print("\nNo suitable .txt files found to process in the folder.")
    else:
        print(f"\nFinished processing {files_processed_count} file(s).")