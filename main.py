import math
from collections import Counter, defaultdict
import os

# Define constants for boundary markers and the unknown token
START_TOKEN = "<s>"
STOP_TOKEN = "</s>"
UNK_TOKEN = "<UNK>"
UNK_THRESHOLD = 1  # Words appearing only once (or less) will be treated as UNK


class NGramLM:
    """An N-Gram Language Model implementing Add-k smoothing."""

    def __init__(self, n: int, add_k: float, vocab: set):
        """
        Initialize the NGramLM.

        Args:
            n (int): The order of the n-gram (e.g., 1 for unigram, 2 for bigram).
            add_k (float): The 'k' value for Add-k smoothing.
                             (k=0 for unsmoothed, k=1.0 for Laplace). [cite: 75]
            vocab (set): The complete vocabulary set, including the UNK_TOKEN.
        """
        if n < 1:
            raise ValueError("N must be 1 or greater.")

        self.n = n
        self.k = add_k
        self.vocab = vocab
        self.vocab_size = len(vocab)

        # Data structures to store counts from the training corpus [cite: 38, 64]
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)  # Stores (n-1)-gram counts
        self.total_tokens = 0  # Total token count (N) for unigram calculation

    def train(self, tokens: list):
        """
        Train the model by counting n-grams and contexts from the token list.

        Args:
            tokens (list): A single list of all preprocessed tokens from the training corpus.
        """
        print(f"Training {self.n}-gram model (k={self.k})...")

        if self.n == 1:
            # Unigram: Counts are just token frequencies
            self.ngram_counts = Counter(tokens)
            self.total_tokens = len(tokens)
        else:
            # Bigram (or higher): Generate n-grams and (n-1)-gram contexts
            # Use zip to create sliding windows (n-grams)
            ngrams = zip(*[tokens[i:] for i in range(self.n)])
            contexts = zip(*[tokens[i:] for i in range(self.n - 1)])

            self.ngram_counts = Counter(ngrams)
            self.context_counts = Counter(contexts)

    def get_token_prob(self, ngram: tuple) -> float:
        """
        Calculate the probability of an n-gram using Add-k smoothing. [cite: 75]
        """
        if not all(token in self.vocab for token in ngram):
            # This should not happen if data is preprocessed correctly,
            # but serves as a safeguard.
            raise ValueError(f"N-gram {ngram} contains out-of-vocabulary words not mapped to <UNK>.")

        if self.n == 1:
            # Unigram probability: P(w) = (Count(w) + k) / (N + k*|V|)
            token_count = self.ngram_counts.get(ngram, 0)
            denominator = self.total_tokens + (self.k * self.vocab_size)
            return (token_count + self.k) / denominator
        else:
            # Bigram (or higher) probability:
            # P(w_i | w_i-1) = (Count(w_i-1, w_i) + k) / (Count(w_i-1) + k*|V|)
            context = ngram[:-1]
            ngram_count = self.ngram_counts.get(ngram, 0)
            context_count = self.context_counts.get(context, 0)

            denominator = context_count + (self.k * self.vocab_size)

            if denominator == 0:
                # Should only happen with k=0 if the context was never seen (impossible if trained on data)
                # But as a fallback, return 0 probability.
                return 0.0

            return (ngram_count + self.k) / denominator

    def calculate_perplexity(self, tokens: list) -> float:
        """
        Calculate the perplexity of the model on a given (validation/test) corpus.
        Uses the log-probability formula required by the assignment to avoid underflow. [cite: 81, 83]

        PP = exp( (-1/N) * SUM( log(P(ngram_i)) ) )

        Args:
            tokens (list): The preprocessed list of tokens from the validation set.

        Returns:
            float: The perplexity score.
        """
        # N is the total number of tokens (including </s> but excluding <s>)
        # However, for perplexity calculation, we evaluate probability of EVERY token,
        # including the </s> markers. The start <s> markers are contexts, not predictions.
        # We create n-grams from the validation data. Every n-gram represents one probability calculation.

        if self.n == 1:
            # For unigrams, every token is an independent event (n-gram).
            ngrams = [(tok,) for tok in tokens]
        else:
            # For bigrams, slide the window.
            ngrams = list(zip(*[tokens[i:] for i in range(self.n)]))

        N = len(ngrams)  # Total number of probability events (n-grams) calculated.
        log_prob_sum = 0.0

        for ngram in ngrams:
            prob = self.get_token_prob(ngram)

            if prob == 0.0:
                # If any token has 0 probability (common in unsmoothed models),
                # the log-prob is negative infinity, and perplexity is positive infinity.
                return float('inf')

            log_prob_sum += math.log(prob)  # Natural log (ln) to match the formula's exp() [cite: 81]

        # Per assignment formula: exp( (1/N) * SUM( -log(P(ngram_i)) ) )
        # which simplifies to: exp( (-1/N) * SUM( log(P(ngram_i)) ) )
        avg_neg_log_likelihood = (-1.0 / N) * log_prob_sum
        perplexity = math.exp(avg_neg_log_likelihood)

        return perplexity


def build_vocab(file_path: str, threshold: int) -> set:
    """
    Builds a vocabulary from the training corpus. [cite: 74]
    Words below the threshold are replaced by UNK_TOKEN.
    """
    print(f"Building vocabulary from {file_path}...")
    token_counts = Counter()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()  # Files are pre-tokenized by space [cite: 70]
            token_counts.update(tokens)

    # Create the vocabulary: include START, STOP, UNK, and any word meeting the count threshold
    vocab = {START_TOKEN, STOP_TOKEN, UNK_TOKEN}
    for word, count in token_counts.items():
        if count > threshold:
            vocab.add(word)

    print(f"Vocabulary size: {len(vocab)} (Raw types: {len(token_counts)}, UNK threshold: {threshold})")
    return vocab


def preprocess(file_path: str, vocab: set) -> list:
    """
    Reads a corpus file, adds boundary markers, and maps OOVs to UNK_TOKEN. [cite: 52, 70, 74]
    Returns a single flat list of all tokens.
    """
    print(f"Preprocessing data from {file_path}...")
    all_tokens = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()

            # Add the start token (context for the first real word)
            all_tokens.append(START_TOKEN)

            for token in tokens:
                if token in vocab:
                    all_tokens.append(token)
                else:
                    all_tokens.append(UNK_TOKEN)  # Handle unknown words

            # Add the stop token
            all_tokens.append(STOP_TOKEN)

    return all_tokens


def main():
    """Main execution function to run the language modeling experiments."""

    # Define file paths. Assignment specifies train.txt and validation.txt [cite: 51]
    # We use the provided 'val.txt' as the validation set.
    train_file = 'train.txt'
    val_file = 'val.txt'

    # --- 1. Data Preparation (Vocab and Preprocessing) ---
    # Build the vocabulary ONLY from the training data [cite: 74]
    vocab = build_vocab(train_file, UNK_THRESHOLD)

    # Preprocess both datasets using this vocabulary
    train_tokens = preprocess(train_file, vocab)
    val_tokens = preprocess(val_file, vocab)

    print("-" * 40)
    print("Running experiments as required by assignment sections:")
    print("1.1 (Unigram), 1.4 (Perplexity)")

    # --- 2. Experiment: Unsmoothed Unigram [cite: 21, 59] ---
    unigram_unsmoothed = NGramLM(n=1, add_k=0.0, vocab=vocab)
    unigram_unsmoothed.train(train_tokens)
    pp_uni_unsmoothed = unigram_unsmoothed.calculate_perplexity(val_tokens)
    print(f"  Perplexity (Unsmoothed Unigram): {pp_uni_unsmoothed:.4f}")

    print("-" * 40)
    print("1.1 (Bigram), 1.4 (Perplexity)")

    # --- 3. Experiment: Unsmoothed Bigram [cite: 21, 59] ---
    bigram_unsmoothed = NGramLM(n=2, add_k=0.0, vocab=vocab)
    bigram_unsmoothed.train(train_tokens)
    pp_bi_unsmoothed = bigram_unsmoothed.calculate_perplexity(val_tokens)
    print(f"  Perplexity (Unsmoothed Bigram):  {pp_bi_unsmoothed:.4f}")

    print("-" * 40)
    print("1.2 (Smoothing), 1.4 (Perplexity)")

    # --- 4. Experiment: Smoothed Bigram (Laplace / Add-1) [cite: 22, 75] ---
    bigram_laplace = NGramLM(n=2, add_k=1.0, vocab=vocab)
    bigram_laplace.train(train_tokens)
    pp_bi_laplace = bigram_laplace.calculate_perplexity(val_tokens)
    print(f"  Perplexity (Smoothed Bigram, k=1.0): {pp_bi_laplace:.4f}")

    # --- 5. Experiment: Smoothed Bigram (Add-k) [cite: 22, 75] ---
    # Try another k-value as suggested [cite: 75]
    k_val = 0.01
    bigram_add_k = NGramLM(n=2, add_k=k_val, vocab=vocab)
    bigram_add_k.train(train_tokens)
    pp_bi_add_k = bigram_add_k.calculate_perplexity(val_tokens)
    print(f"  Perplexity (Smoothed Bigram, k={k_val}):  {pp_bi_add_k:.4f}")
    print("-" * 40)


# Standard Python entry point [cite: 15]
if __name__ == "__main__":
    # Check that data files exist before running
    if not os.path.exists('train.txt') or not os.path.exists('val.txt'):
        print("Error: 'train.txt' and/or 'val.txt' not found in this directory.")
        print("Please ensure the data files are present.")
    else:
        main()