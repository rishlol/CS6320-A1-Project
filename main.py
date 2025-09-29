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
        # gets each bigram = (w1, w2)
        if bi in bcount:
            bcount[bi] += 1
        else:
            bcount[bi] = 1
    for key in bcount.keys():
        # P(w2 | w1) = # "w1 w2" / # w1
        bigram[key] = bcount[key] / wcount[key.split(' ')[0]]
    print(bigram)

def train():
    ngram('train.txt')

if __name__ == '__main__':
    train()