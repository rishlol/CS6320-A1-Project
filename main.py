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
        if word in unigram:
            wcount[word] += 1
        else:
            wcount[word] = 1
    word_count = len(wcount.keys())
    for key in wcount.keys():
        unigram[key] = wcount[key] / word_count

    # Get bigram probabilities
    bigramzip = [(s1, s2) for s1, s2 in zip(text, text[1:])]
    for bi in bigramzip:
        if bi in bigram:
            bigram[bi] += 1
        else:
            bigram[bi] = 1
    for key in bigram.keys():
        bigram[key] = bigram[key] / word_count

def train():
    ngram('train.txt')

if __name__ == '__main__':
    train()