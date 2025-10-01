Assignment 1 - Bigram Language Model (CS 6320)

--------------------------------------------
Overview
--------------------------------------------
This project implements a statistical language model using unigrams and bigrams.

Features:
- Tokenization & preprocessing (punctuation handling, lowercasing <s>, </s>)
- Unigram and bigram probability computation
- Smoothing: Laplace (Add-1) and Add-k (k=0.5)
- Unknown word handling using <UNK> token
- Perplexity evaluation on validation set

--------------------------------------------
Files
--------------------------------------------
- main.py          : Python program implementation
- train.txt        : Training dataset
- val.txt          : Validation dataset
- report.pdf       : pdf project report
- README.txt       : This documentation file

--------------------------------------------
Requirements
--------------------------------------------
- Python 3.7 or higher

--------------------------------------------
How to Run
--------------------------------------------
1. Ensure train.txt and val.txt are in the same directory as like as main.py
2. Run the program:
   python3 main.py

3. The program will display:
   - Steps for unigram also bigram computation
   - Handling of unknown words
   - Smoothing process
   - Perplexity results

--------------------------------------------
Expected Output (Example)
--------------------------------------------
Step 1.1 : Computing raw n-gram counts from 'train.txt'...
Step 1.3: Handling unknown words...
Step 1.2: Applying Laplace Smoothing (k=1)...
Step 1.2: Applying Add-k Smoothing (k=0.5)...
Step 1.4: Calculating Perplexity on 'val.txt' for both models...
Perplexity with Laplace Smoothing: 410
Perplexity with Add-k Smoothing: 302

--------------------------------------------
Results
--------------------------------------------
Validation Set (Laplace Add-1, k=1)   : 410
Validation Set (Add-k, k=0.5)         : 302

--------------------------------------------
Group Members
--------------------------------------------
- Rishabh Medhi (rxm200047) : Data cleaning and tokenization
- Md Sanaullah Miah (mxm230037) : Unknown word handling
- Suraj Namburi (sxn200039) : Smoothing functions
- Anvar Zokhidov (axz230018) : Perplexity evaluation and experiments

--------------------------------------------