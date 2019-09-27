import pandas as pd
from pprint import pprint
from nltk.corpus import stopwords
import string
import gensim
from nltk.tokenize import word_tokenize
import os
import nltk
from nltk.stem import PorterStemmer
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
from collections import Counter
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser



def cleaning(doc):
    lemmatizer = WordNetLemmatizer()

    stop_words = set(stopwords.words('english'))
    doc = doc.lower()
    if "-" in doc:
        doc = doc.replace("-"," ")

    filter_punch = str.maketrans('', '', string.punctuation)
    stripped = doc.translate(filter_punch)

    clean_text = []
    for i in stripped.split():
        if i not in stop_words:
            clean_text.append(lemmatizer.lemmatize(i))

    return ' '.join(clean_text)

def counter_things():
    all_words = []
    tag_list  = []
    df = pd.read_csv("bibs.csv")

    Publisher = (list(df["Publisher"].values))
    print(Counter(Publisher))
    abstracts = (list(df["Abs"].values))
    for v in abstracts:
        clean_text = cleaning(v)
        tag_list.append(TextBlob(clean_text).tags)
        all_words.append(clean_text.split())

    flat_list = [item for sublist in all_words for item in sublist]
    print(Counter(flat_list))

def w2v():
    import multiprocessing
    cores = multiprocessing.cpu_count()

    txt_list = []
    df = pd.read_csv("bibs.csv")
    for doc in df["Abs"]:
        txt_list.append(cleaning(doc))


    df["clean"] = txt_list

    sent = [row.split() for row in df['clean']]
    phrases = Phrases(sent, min_count=10, progress_per=10)
    bigram = Phraser(phrases)
    sentences = bigram[sent]
    w2v_model = Word2Vec(min_count=4,
                         window=5,
                         size=10,
                         sample=6e-5,
                         alpha=0.03,
                         min_alpha=0.0007,
                         negative=4,
                         workers=cores - 1)
    w2v_model.build_vocab(sentences, progress_per=10)
    #aki = Acute Kidney Injury
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
    w2v_model.save("word2vec.model")
    w2v_model.init_sims(replace=True)

    res = w2v_model.wv.most_similar(positive=["surgical"])
    pprint(res)

counter_things()
w2v()