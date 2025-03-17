import time
import matplotlib.pyplot as plt
import random
import string
from collections import defaultdict

# Implementace algoritmů pro vyhledávání vzoru v textu

def brute_force_search(text, pattern):
    """Hrubá síla - porovnává vzor se všemi možnými posunutími v textu."""
    comparisons = 0
    positions = []
    for i in range(len(text) - len(pattern) + 1):
        match = True
        for j in range(len(pattern)):
            comparisons += 1
            if text[i + j] != pattern[j]:
                match = False
                break
        if match:
            positions.append(i)
    return positions, comparisons


def kmp_search(text, pattern):
    """KMP algoritmus využívá prefikso-sufixovou tabulku pro efektivnější posun vzoru."""
    comparisons = 0

    def compute_lps(pattern):
        """Výpočet tabulky LPS (Longest Prefix Suffix)."""
        lps = [0] * len(pattern)
        length = 0
        i = 1
        while i < len(pattern):
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        return lps

    lps = compute_lps(pattern)
    i = j = 0
    positions = []
    while i < len(text):
        comparisons += 1
        if text[i] == pattern[j]:
            i += 1
            j += 1
        if j == len(pattern):
            positions.append(i - j)
            j = lps[j - 1]
        elif i < len(text) and text[i] != pattern[j]:
            comparisons += 1
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return positions, comparisons


def bmh_search(text, pattern):
    """BMH algoritmus využívá heuristiku posunu podle posledního výskytu znaku."""
    comparisons = 0
    m, n = len(pattern), len(text)
    if m > n:
        return [], 0
    
    # Předpočítání tabulky posunů
    skip = {pattern[i]: m - i - 1 for i in range(m - 1)}
    i = 0
    positions = []
    while i <= n - m:
        j = m - 1
        while j >= 0 and pattern[j] == text[i + j]:
            comparisons += 1
            j -= 1
        if j < 0:
            positions.append(i)
            i += m
        else:
            comparisons += 1
            i += skip.get(text[i + m - 1], m)
    return positions, comparisons


def select_best_algorithm(text, pattern):
    """Heuristická volba nejvhodnějšího algoritmu podle délky textu a vlastností vzoru."""
    if len(pattern) > len(text) // 2:
        return kmp_search
    elif len(pattern) < 5 and len(set(pattern)) > 3:
        return bmh_search
    else:
        return brute_force_search


def test_algorithms():
    """Spustí testy na různých textech a vzorech."""
    with open("shorttext.txt") as f:
        short_text = f.read()
    with open("longtext.txt") as f:
        long_text = f.read()
    
    test_cases = {
        "short_text": (short_text, "lorem"),
        "long_text": (long_text, "lorem"),
        "random_dna": ("AGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT", "AGCT")
    }
    
    results = defaultdict(dict)
    
    for name, (text, pattern) in test_cases.items():
        for algo, func in zip(["Brute Force", "KMP", "BMH"], [brute_force_search, kmp_search, bmh_search]):
            _, comparisons = func(text, pattern)
            results[name][algo] = comparisons
        
        # Test hybridního přístupu
        best_algo = select_best_algorithm(text, pattern)
        _, comparisons = best_algo(text, pattern)
        results[name]["Hybrid"] = comparisons
    
    return results


def visualize_results(results):
    """Vytvoří vizualizaci výsledků testů."""
    labels = list(results.keys())
    bf_values = [results[label]["Brute Force"] for label in labels]
    kmp_values = [results[label]["KMP"] for label in labels]
    bmh_values = [results[label]["BMH"] for label in labels]
    hybrid_values = [results[label]["Hybrid"] for label in labels]
    
    x = range(len(labels))
    
    plt.figure(figsize=(8, 5))
    plt.bar(x, bf_values, width=0.2, label="Brute Force", align='center')
    plt.bar([i + 0.2 for i in x], kmp_values, width=0.2, label="KMP", align='center')
    plt.bar([i + 0.4 for i in x], bmh_values, width=0.2, label="BMH", align='center')
    plt.bar([i + 0.6 for i in x], hybrid_values, width=0.2, label="Hybrid", align='center')
    
    plt.xticks([i + 0.3 for i in x], labels)
    plt.ylabel("Počet porovnání")
    plt.title("Porovnání výkonu algoritmů")
    plt.legend()
    plt.show()


def analyze_algorithms():
    """Analýza silných a slabých stránek jednotlivých algoritmů."""
    analysis = {
        "KMP vs. BMH": "KMP je rychlejší u dlouhých textů s opakujícími se vzory, zatímco BMH je efektivnější pro náhodné texty.",
        "BMH vs. Brute Force": "BMH je rychlejší, protože posouvá vzor většími skoky.",
        "Nevýhoda KMP": "KMP může být neefektivní, pokud vzor nemá mnoho opakujících se prvků.",
        "Hybridní přístup": "Hybridní přístup se snaží využít nejefektivnější algoritmus podle délky a typu vstupu."
    }
    for k, v in analysis.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    results = test_algorithms()
    visualize_results(results)
    analyze_algorithms()
