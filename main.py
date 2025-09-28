def ngram(filename: str):
    text = []
    unigram = {}
    bigram = {}
    with open(filename) as f:
        text = f.read().split()

    # Add start and end tokens
    text.insert(0, '<s>')
    text.append('</s>')

    # Get unigram probabilities
    wcount = {}
    for word in text:
        if word in wcount:
            wcount[word] += 1
        else:
            wcount[word] = 1
    word_count = len(text)
    for key in wcount.keys():
        # P(w) = # w / # all words
        unigram[key] = wcount[key] / word_count

    # Get bigram probabilities
    bigramzip = []
    for i in range(0, len(text) - 1):
        bigramzip.append(f"{text[i]} {text[i + 1]}")
    bcount = {}
    for bi in bigramzip:
        # gets each tuple bi = (w1, w2)
        if bi in bcount:
            bcount[bi] += 1
        else:
            bcount[bi] = 1
    for key in bcount.keys():
        # P(w2 | w1) = # (w1, w2) / # w1
        bigram[key] = bcount[key] / wcount[key.split(' ')[0]]
    print(bigram)

def train():
    ngram('train.txt')

if __name__ == '__main__':
    train()

############################################## by Anvar Zokhidov
import math

def clean_and_tokenize(filename: str) -> list:
    """
    Reads a file, cleans punctuation, and tokenizes the text.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        # Punctuation is separated to be treated as tokens.
        raw_text = f.read().replace('.', ' . ').replace(',', ' , ').replace('!', ' ! ').replace('?', ' ? ').replace("'", " ' ")
        tokens = raw_text.lower().split()
    return tokens

# 1.1 Unigram and bigram probability computation
def ngram_counts(filename: str) -> tuple:
    """
    Computes raw unigram and bigram counts from a text file.
    """
    text = clean_and_tokenize(filename)
    
    # Add start and end tokens as per the assignment
    text.insert(0, '<s>')
    text.append('</s>')
    
    # Manually count unigrams and bigrams using dictionaries
    wcount = {}
    for word in text:
        wcount[word] = wcount.get(word, 0) + 1
    
    bigram_list = [f"{text[i]} {text[i+1]}" for i in range(len(text) - 1)]
    bcount = {}
    for bi in bigram_list:
        bcount[bi] = bcount.get(bi, 0) + 1
    
    return text, wcount, bcount

# 1.3 Unknown word handling
def unknown_word_handling(tokens: list, word_counts: dict) -> tuple:
    """
    Replaces words that appear only once in the training data with a special <UNK> token.
    """
    words_to_replace = {word for word, count in word_counts.items() if count == 1}
    processed_tokens = ['<UNK>' if word in words_to_replace else word for word in tokens]
    
    # Recalculate counts with the <UNK> token
    final_unigram_counts = {}
    for word in processed_tokens:
        final_unigram_counts[word] = final_unigram_counts.get(word, 0) + 1

    return processed_tokens, final_unigram_counts

# 1.2 Smoothing
def smoothing(unigram_counts: dict, bigram_counts: dict, k: float) -> dict:
    """
    Applies Add-k Smoothing to the bigram counts.
    """
    smoothed_bigram_probs = {}
    V = len(unigram_counts)
    
    all_possible_bigrams = {f"{w1} {w2}" for w1 in unigram_counts.keys() for w2 in unigram_counts.keys()}
    
    for bigram in all_possible_bigrams:
        w1, w2 = bigram.split(' ', 1)
        w1_count = unigram_counts.get(w1, 0)
        bigram_count = bigram_counts.get(bigram, 0)

        # P_smoothed(w2|w1) = (Count(w1, w2) + k) / (Count(w1) + k*V)
        smoothed_prob = (bigram_count + k) / (w1_count + k * V)
        smoothed_bigram_probs[bigram] = smoothed_prob
    
    return smoothed_bigram_probs

# 1.4 Implementation of perplexity
def perplexity(bigram_model, unigram_counts, vocabulary, test_filename: str, k: float) -> float:
    """
    Calculates the perplexity of the language model on a test set.
    """
    log_sum = 0.0
    N = 0
    V = len(vocabulary)
    
    test_tokens = clean_and_tokenize(test_filename)
    test_tokens.insert(0, '<s>')
    test_tokens.append('</s>')
    
    for i in range(len(test_tokens) - 1):
        w1 = test_tokens[i]
        w2 = test_tokens[i+1]
        
        if w1 not in vocabulary: w1 = '<UNK>'
        if w2 not in vocabulary: w2 = '<UNK>'

        bigram_key = f"{w1} {w2}"
        
        w1_count = unigram_counts.get(w1, 0)
        prob = bigram_model.get(bigram_key, (k) / (w1_count + k * V))

        if prob > 0:
            log_sum += math.log2(prob)
            N += 1
            
    if N == 0:
        return float('inf')
    
    perplexity_score = math.pow(2, -log_sum / N)
    return perplexity_score

def main():
    """
    Main function to run the entire language model pipeline.
    """
    print("Step 1.1: Computing raw n-gram counts from 'train.txt'...")
    tokens, unigram_counts_raw, bigram_counts_raw = ngram_counts('train.txt')
    
    print("Step 1.3: Handling unknown words...")
    processed_tokens, final_unigram_counts = unknown_word_handling(tokens, unigram_counts_raw)
    
    bigram_counts_unk = {}
    for i in range(len(processed_tokens) - 1):
        bigram = f"{processed_tokens[i]} {processed_tokens[i+1]}"
        bigram_counts_unk[bigram] = bigram_counts_unk.get(bigram, 0) + 1
    
    # 1.2 Implementations of two smoothing methods
    # Method 1: Laplace Smoothing (Add-1)
    print("\nStep 1.2: Applying Laplace Smoothing (k=1)...")
    smoothed_bigram_probs_add1 = smoothing(final_unigram_counts, bigram_counts_unk, k=1)
    
    # Method 2: Add-k Smoothing (e.g., k=0.5)
    print("Step 1.2: Applying Add-k Smoothing (k=0.5)...")
    smoothed_bigram_probs_addk = smoothing(final_unigram_counts, bigram_counts_unk, k=0.5)
    
    # 1.4 Perplexity for each model
    print("\nStep 1.4: Calculating Perplexity on 'val.txt' for both models...")
    
    # Perplexity with Laplace Smoothing
    perplexity_score_add1 = perplexity(
        smoothed_bigram_probs_add1,
        final_unigram_counts,
        final_unigram_counts.keys(),
        'val.txt',
        k=1
    )
    print(f"Perplexity with Laplace Smoothing: {perplexity_score_add1:.2f}")

    # Perplexity with Add-k Smoothing
    perplexity_score_addk = perplexity(
        smoothed_bigram_probs_addk,
        final_unigram_counts,
        final_unigram_counts.keys(),
        'val.txt',
        k=0.5
    )
    print(f"Perplexity with Add-k Smoothing: {perplexity_score_addk:.2f}")

if __name__ == '__main__':
    main()
