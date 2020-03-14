from pprint import pprint
import re
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
import spacy
from random import shuffle
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
import math

nlp = spacy.load("en_core_web_sm")

# clean sents and tokenize
def bag_of_words(s):
    tokenizer = RegexpTokenizer("\w+")
    tokens = tokenizer.tokenize(s)
    tokens = [token for token in tokens if token not in string.punctuation]
    tokens = [token.lower() for token in tokens if re.findall(r"\w", token)]
    tokens = [token.strip().strip('.').strip("—").strip("'") for token in tokens]
    return tokens
def compute_df(all_text, unique_words):
    tokenized_text = [bag_of_words(doc) for doc in all_text]
    # unique_words = get_vocab(all_text)
    df = dict.fromkeys(unique_words, 0)
    for i,tokens in enumerate(tokenized_text):
        for token in tokens:
            for w in unique_words:
                if token == w:
                    df[token] += 1
                    break
    return df
def compute_tfidf(all_text, unique_words, TF_vals):
    # unique_words = get_vocab(all_text)
    tf_vals = [compute_tf(text, unique_words) for text in all_text]
    idf_vals = compute_idf(all_text, unique_words)

    tf_idf = list()

    for i in range(len(all_text)):
        temp = list()
        for j, w in enumerate(unique_words):
            z1 = tf_vals[i][j]
            z2 = idf_vals[w]
            z3 = z1 * z2
            temp.append(z3)
        tf_idf.append(temp)
    return tf_idf


# get the unique words
def get_vocab(all_text):
    tokenized_text = [bag_of_words(s) for s in all_text]
    tokens = [token for tokens in tokenized_text for token in tokens]
    tokens = list(set(tokens))
    tokens.sort()
    return tokens


def compute_tf(doc, unique_words):
    tf_dict = dict()
    tokens = bag_of_words(doc)
    tf = [0] * len(unique_words)
    N = len(tokens)
    for token in tokens:
        tf_dict[token] = tf_dict.setdefault(token, 0) + 1

    for word in tf_dict:
        tf_dict[word] = tf_dict[word] / N
    for i, w in enumerate(unique_words):
        if w in tf_dict:
            tf[i] = tf_dict[w]

    return tf
def compute_idf(all_text, unique_words):
    word_freq = compute_df(all_text, unique_words)
    idf = dict()
    N = len(all_text)
    for word in word_freq:
        idf[word] = np.log(N / (word_freq[word] + 1))
    return idf

if __name__ == "__main__":

#get the corpus
    with open("sample text.txt","r") as f:
        content = f.read()

# get the sentences
    corpus = content.split("\n")
    corpus = [c for c in corpus if len(c.strip()) > 0]
    shuffle(corpus)
    shuffle(corpus)
    shuffle(corpus)
    shuffle(corpus)
    shuffle(corpus)
    corpus = corpus[:10]



# computing TF
    vocabulary= get_vocab(corpus)
    TF_vals = [compute_tf(para, vocabulary) for para in corpus]
    tf_idf = compute_tfidf(corpus,vocabulary, TF_vals)
    print(*zip(tf_idf[0], vocabulary),sep="\n")