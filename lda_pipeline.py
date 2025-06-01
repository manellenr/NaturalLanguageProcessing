import itertools
import re
import pandas as pd
import numpy as np
import spacy
import gensim
from tqdm import tqdm
from sklearn.feature_extraction import text
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.utils import simple_preprocess
from spacy.parts_of_speech import IDS as POS_map
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

def load_dataset(path="data/News_Category_Dataset_v3.json", nrows=None):
    dataset = pd.read_json(path, lines=True, dtype={"headline": str})
    if nrows:
        dataset = dataset.head(nrows)
    return dataset

def force_format(texts):
    return [str(t) for t in texts]

def check_data_quality(texts):
    assert all(isinstance(t, str) for t in texts)
    assert all(t is not np.nan for t in texts)
    return True

def filter_text(text):
    text = re.sub(r'https?:\/\/[^\s]+', ' ', text)
    text = re.sub(r'[(){}\[\]<>]', ' ', text)
    text = re.sub(r'&amp;#.*;', ' ', text)
    text = re.sub(r'&gt;', ' ', text)
    text = re.sub(r'â€™', "'", text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'&#x200B;', ' ', text)
    text = re.sub(r'[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+\.[a-zA-Z0-9-_.]+', '', text)
    text = re.sub(r"\(?\d{3}\D{0,3}\d{3}\D{0,3}\d{4}", '', text)
    text = re.sub(r'@\S+( |\n)', '', text)
    text = re.sub(r'\*', '', text)
    return text

def preprocess_texts(texts):
    return [filter_text(t) for t in texts]

def sent_to_words(sentences):
    for sentence in tqdm(sentences, desc="Tokenization"):
        yield simple_preprocess(str(sentence), deacc=True)

def get_stopwords(additional_stopwords=[]):
    with open('stopwords.txt', 'r') as f:
        stop_w = f.readlines()
    stopwords = [s.strip() for s in stop_w]
    stopwords = list(text.ENGLISH_STOP_WORDS.union(stopwords))
    stopwords += additional_stopwords
    stopwords = list(set(stopwords))
    return sorted([s.replace("\n", "") for s in stopwords], key=str.lower)

def remove_stopwords(texts, stopwords):
    return [[word for word in txt if word not in stopwords] for txt in tqdm(texts, desc="Remove stopwords")]

def compute_word_occurences(texts):
    words = itertools.chain.from_iterable(texts)
    word_count = pd.Series(words).value_counts()
    return pd.DataFrame({"Word": word_count.index, "Count": word_count.values})

def average_title_length(texts):
    return np.mean([len(txt) for txt in texts])

def create_bigrams(texts, bigram_count=15, threshold=10, as_str=True):
    bigram_model = Phraser(Phrases(texts, min_count=bigram_count, threshold=threshold))
    return [" ".join(bigram_model[t]) if as_str else bigram_model[t] for t in texts]

def lemmatize_texts(texts, allowed_postags=None, forbidden_postags=[], as_sentence=False, get_postags=False, spacy_model=None):
    if allowed_postags is None:
        allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']
    if allowed_postags and forbidden_postags:
        raise ValueError("Can't use both allowed and forbidden postags")

    if forbidden_postags:
        allowed_postags = list(set(POS_map.keys()) - set(forbidden_postags))

    if not spacy_model:
        spacy_model = spacy.load('en_core_web_md')

    docs = spacy_model.pipe(texts)
    texts_out = []

    for doc in tqdm(docs, total=len(texts)):
        if get_postags:
            texts_out.append(["_".join([token.lemma_, token.pos_]) for token in doc if token.pos_ in allowed_postags])
        else:
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])

    if as_sentence:
        texts_out = [" ".join(text) for text in texts_out]
    return texts_out

def run_lda(texts, num_topics=10, no_below=15, no_above=0.1, keep_n=1000, passes=1000):
    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    bow_corpus = [dictionary.doc2bow(doc) for doc in texts]
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=num_topics, id2word=dictionary, passes=passes)
    return lda_model, dictionary, bow_corpus

def print_topics(lda_model, num_words=10):
    topics = []
    for idx, topic in lda_model.print_topics(-1, num_words=num_words):
        print(f"Topic: {idx} -> {topic}")
        topics.append(topic)
    return topics

def compute_coherence(lda_model, texts, dictionary):
    coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary)
    return coherence_model.get_coherence()
