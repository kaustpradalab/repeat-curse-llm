import math
from collections import Counter

def calculate_entropy(probabilities):
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
    return entropy

def generate_ngrams(text, n):
    words = text.split()
    ngrams = zip(*[words[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]

# 根据token数量归一化
# def calculate_ngram_entropy(text, n):
#     ngrams = generate_ngrams(text, n)
#     ngram_counts = Counter(ngrams)
#     total_ngrams = sum(ngram_counts.values())
#
#     probabilities = [count / total_ngrams for count in ngram_counts.values()]
#     entropy = calculate_entropy(probabilities)
#
#     # 获取 token 的数量
#     total_tokens = len(text.split())
#
#     # 归一化熵（将熵值除以 token 数）
#     normalized_entropy = entropy / total_tokens * 10
#     return normalized_entropy

# 最大熵标准化
def calculate_ngram_entropy(text, n):
    """Calculate the normalized entropy of n-grams for a given text."""
    ngrams = generate_ngrams(text, n)
    ngram_counts = Counter(ngrams)
    total_ngrams = sum(ngram_counts.values())

    probabilities = [count / total_ngrams for count in ngram_counts.values()]
    entropy = calculate_entropy(probabilities)

    # Calculate the maximum possible entropy for normalization
    max_entropy = math.log2(len(ngrams)) if len(ngrams) > 0 else 1

    # Normalize the entropy by dividing by the maximum possible entropy
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    return normalized_entropy

if __name__ == '__main__':

    # 定义所有文本
    texts = {
        "normal 0": "Option c) Rs 550 is the correct answer. This choice is accurate because it aligns with the calculations or criteria provided in the problem statement. Selecting this option ensures the correct interpretation and understanding of the given information, confirming Rs 550 as the appropriate solution.",
        "single token repeat": "Option Option Option Option Option Option Option Option Option Option Option Option Option Option Option Option Option Option Option Option Option Option Option Option Option Option Option Option Option Option Option Option Option Option Option ",
        "single token repeat from 3": "Option c) answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer.",
        "single token repeat from 8": "Option c) Rs 550 is the correct answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer. answer.",
        "normal 1": "Alright, little buddy, let me explain this to you. A \"premise\" is like an idea or a situation we know already.",
        "repeat paragraph": "Alright, little buddy, let me explain this to you. A \"premise\" is like an idea or a situation we know already. Alright, little buddy, let me explain this to you. A \"premise\" is like an idea or a situation we know already. Alright, little buddy, let me explain this to you. A \"premise\" is like an idea or a situation we know already. Alright, little buddy, let me explain this to you. A \"premise\" is like an idea or a situation we know already. Alright, little buddy, let me explain this to you. A \"premise\" is like an idea or a situation we know already.",
        "normal2": "No, that is not a correct answer. The creator of Bing is Microsoft Corporation. Bing is a web search engine developed and operated by Microsoft.",
        "repeat phrase": "No, that is not a correct answer. The creator of Bing is Microsoft Corporation. Bing is a web search engine developed and operated by Microsoft. operated by Microsoft.operated by Microsoft.operated by Microsoft.operated by Microsoft.operated by Microsoft.operated by Microsoft.operated by Microsoft.operated by Microsoft.operated by Microsoft.operated by Microsoft.",
        "short term": " the the country to make the and processes a.? history? to the laws of",
        "short term2": " is the -FET? the why did it different??",
        "short term3": "is it the world between the and principles?? the?",
        "short term4": " is the of of how that you is are not to.",
        "short term5": " is the of of the that you is are not to?",
    }
    test = "Once upon a time, Emily held and rocked Kayla as they both sobbed because Kayla really needed a good friend. You see, little Kayla was feeling very lonely and sad because she didn't have anyone to play with or talk to.\n\nEmily wanted to help Kayla and be her friend, so she did something truly special. Emily decided to throw a magical friendship party for Kayla. Emily knew that this would cheer her up and help her make some new friends!\n\nEmily worked really hard, preparing for the big day. She bought balloons, baked cookies, and even created a treasure hunt for everyone to enjoy. She invited lots of other kids from their neighborhood, so Kayla could meet them all and find a friend to play with.\n\nOn the day of the party, the sun was shining, and the sky was filled with fluffy clouds. All the kids in the neighborhood were excited to come to the party and play with Kayla.\n\nEmily and Kayla welcomed their guests with huge smiles and warm hugs. The kids played games, laughed, and explored the treasure hunt that Emily had set up all around the backyard.\n\nSoon, Kayla was surrounded by new friends, chatting and laughing together. She began to feel so happy and loved. As the sun began to set and the party ended, Kayla felt like she finally had found the friends she needed.\n\nThat night, as Emily tucked Kayla into bed, Kayla thanked her for being such an amazing friend and throwing the best party ever. Emily smiled and said, \"That's what friends are for, to help each other when we're feeling sad and lonely.\"\n\nAnd from that day on, Emily and Kayla were the best of friends and shared many more magical adventures together. Now, Kayla knew that she always had a good friend by her side. good friend by her side.good friend by her side.good friend by her side.good friend by her side.good friend by her side.good friend by her side.good friend by her side.good friend by her side.good friend by her side.good friend by her side."
    n = 1  # 计算双词组（二元组）的信息熵

    # 对每个文本计算信息熵并打印结果
    for name, text in texts.items():
        entropy = calculate_ngram_entropy(text, n)
        print(f"The {n}-gram entropy of {name} is: {entropy:.4f}")
    entropy = calculate_ngram_entropy(test, 4)
    print(entropy)