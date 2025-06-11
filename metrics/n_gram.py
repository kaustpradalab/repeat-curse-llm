import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter


def calculate_rep_n(text: str, n: int) -> float:
    """
    Calculate the rep-n metric for a given text and n-gram size.

    Args:
        text (str): The input text.
        n (int): The size of n-grams.

    Returns:
        float: The rep-n value.
    """
    tokens = text.split()  # Split text into tokens
    L = len(tokens)  # Total number of tokens in the text

    if L < n:  # If the text is shorter than the n-gram size, return 0
        return 0.0

    # Extract all n-grams from the text
    ngrams = [tuple(tokens[i:i + n]) for i in range(L - n + 1)]

    # Count the number of unique n-grams
    unique_ngrams = len(set(ngrams))

    # Calculate rep-n using the formula
    rep_n = 1.0 - (unique_ngrams / (L - n + 1))

    return rep_n


if __name__ == '__main__':
    # Test with high and low repetition rate text
    high_repetition_rate_test = "when i look at myself in the mirror, I see my name? What is a Glamour? How do I apply for a Glamour? What is an Evantful Start? How to Apply for a Glamour and Its Benefits? What is my Membership?"
    low_repetition_rate_test = "when i look at myself in the mirror, I see that I am not a good person. I am not a good person because of my past. My past is something that has been fixed and changed. It's something that has been passed on but still remains broken and  unbroken."
    repeat = "repeat repeat repeat repeat repeat repeat repeat "
    exp = "Boys love freedom. Girls love freedom. You love freedom. People love freedom"
    # Calculate weighted repetition rates
    high_repetition_rate = calculate_rep_n(high_repetition_rate_test, n=1)
    low_repetition_rate = calculate_rep_n(low_repetition_rate_test, n=1)
    repeat = calculate_rep_n(repeat, n=1)
    exp = calculate_rep_n(exp, n=1)
    print(f"normal: {high_repetition_rate:.2f}")
    print(f"steered #0: {low_repetition_rate:.2f}")
    print(f"repeat: {repeat:.2f}")
    print(f"exp: {exp:.2f}")
