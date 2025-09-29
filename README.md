**By Anvar: Some helper template for report structuring:**



<img width="2045" height="516" alt="image" src="https://github.com/user-attachments/assets/a71a3818-44d0-4c06-be3f-6110ad3f91b8" />
2 Eval, Analysis and Findings (30%)
2.1 Perplexity Report
The performance of the Bigram Language Model was evaluated using perplexity (PPL), a measure of how well the probability distribution of the model predicts a sample. A lower PPL indicates a better model.

Dataset

Smoothing Strategy

Smoothing Parameter (k)

Perplexity (PPL)

Training Set

Laplace (Add-1)

1

[Insert Training PPL Value Here, e.g., 2.5]

Validation Set

Laplace (Add-1)

1

[Insert Validation PPL Value for k=1]

Validation Set

Add-k

0.5

[Insert Validation PPL Value for k=0.5]

Observation: The Perplexity on the training set is significantly lower than on the validation set. This is expected, as the model perfectly predicts the sequences it was trained on. The validation set PPL, which is higher, provides a realistic measure of the model’s ability to generalize to unseen data.

2.2 Analysis of Smoothing Strategies
The primary purpose of smoothing is to mitigate the Zero-Probability Problem, where any n-gram not observed in the training data is assigned a probability of zero, which causes the perplexity calculation to fail (log(0) is undefined).

Impact of Laplace Smoothing (k=1):

Mechanism: Laplace smoothing is the simplest form, adding 1 to every bigram count.

Effect: While it successfully eliminates zero probabilities, it tends to over-smooth the model. By adding a full count to every possible unseen bigram, it shifts too much probability mass away from frequently observed, reliable bigrams to those that are rare or never seen. This typically results in a higher perplexity score compared to more refined methods.

Impact of Add-k Smoothing (k=0.5):

Mechanism: Add-k smoothing (where k<1) is a refinement that adds a fractional count (e.g., 0.5) to every bigram.

Effect: By using k=0.5, we introduce less distortion into the original observed probabilities. The model still assigns a non-zero probability to unseen events, but the resulting probability estimates for frequently seen bigrams are closer to their Maximum Likelihood Estimates.

Finding: Comparing the validation set results, the Add-k smoothing (k=0.5) achieved a [Insert "lower" or "higher" based on your results] perplexity than Laplace smoothing (k=1), demonstrating better performance in handling the sparsity of the Chicago hotel review data.

3 Others
3.1 Details of Programming Library Usage (6%)
Our implementation strictly adhered to the assignment constraints, avoiding high-level data structure libraries that might be considered "cheating" (e.g., collections.Counter).

The only external library utilized was the math module, which is a standard library. Specifically, we used:

math.log2(prob): Essential for calculating the log-probability terms in the perplexity formula.

math.pow(2, exponent): Required for the final step of the perplexity calculation.
All counting for unigrams and bigrams, as well as the dictionary lookups and manipulations, were performed using core Python dictionaries and manual loops.

3.2 Brief Description of the Contributions of Each Group Member (2%)
[Full Name and Net ID 1]: Primarily focused on the foundational code, including data cleaning (clean_and_tokenize) and the initial count calculation (ngram_counts).

[Full Name and Net ID 2]: Implemented the essential pre-processing logic in unknown_word_handling and was responsible for integrating the bigram count adjustments post-UNK replacement.

[Full Name and Net ID 3]: Developed and tested the core smoothing function, ensuring it correctly implemented both Laplace (k=1) and Add-k (k=0.5) smoothing variations.

[Full Name and Net ID 4]: Implemented the final evaluation step, the perplexity function, and oversaw the final integration and running of the comparative analysis in the main function.

3.3 Feedback for the Project (2%)
The project was challenging but highly beneficial. It required careful attention to detail, particularly in ensuring the vocabulary and counts were correctly updated after the unknown word handling step. We spent approximately [Insert X hours] on the project, split between coding, debugging the math for smoothing, and preparing the comparative analysis.

The manual implementation of the n-gram model, combined with the requirement to compare two smoothing techniques, significantly deepened our understanding of the practical challenges of language modeling and the crucial role that smoothing plays in creating a robust model. It clearly demonstrated why we cannot simply use raw counts for prediction.
