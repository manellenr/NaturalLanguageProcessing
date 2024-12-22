import itertools
import os
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
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.parts_of_speech import IDS as POS_map

class TextProcessor:
    def __init__(self, dataset_path, stopwords_file='stopwords.txt'):
        self.dataset = pd.read_json(dataset_path, lines=True, dtype={"headline": str})
        self.texts = self.dataset["headline"].tolist()
        self.stopwords = self.get_stopwords(stopwords_file)
        
    def dummy_word_split(self, texts):
        return [text.split(" ") for text in texts]

    def compute_word_occurences(self, texts):
        words = itertools.chain.from_iterable(texts)
        word_count = pd.Series(words).value_counts()
        return pd.DataFrame({"Word": word_count.index, "Count": word_count.values})
    
    def check_data_quality(self, texts):
        assert all([isinstance(t, str) for t in texts])
        assert all([t != np.nan for t in texts])
        return True

    def force_format(self, texts):
        return [str(t) for t in texts]

    def filter_text(self, text_in):
        text_out = re.sub(r'https?:\/\/[A-Za-z0-9_.-~\-]*', ' ', text_in, flags=re.MULTILINE)
        text_out = re.sub(r'[(){}\[\]<>]', ' ', text_out, flags=re.MULTILINE)
        text_out = re.sub(r'&amp;#.*;', ' ', text_out, flags=re.MULTILINE)
        text_out = re.sub(r'&gt;', ' ', text_out, flags=re.MULTILINE)
        text_out = re.sub(r'â€™', "'", text_out, flags=re.MULTILINE)
        text_out = re.sub(r'\s+', ' ', text_out, flags=re.MULTILINE)
        text_out = re.sub(r'&#x200B;', ' ', text_out, flags=re.MULTILINE)
        text_out = re.sub(r'[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+\.[a-zA-Z0-9-_.]+', '', text_out, flags=re.MULTILINE)
        text_out = re.sub(r"\(?\d{3}\D{0,3}\d{3}\D{0,3}\d{4}", '', text_out, flags=re.MULTILINE)
        text_out = re.sub(r'@\S+( |\n)', '', text_out, flags=re.MULTILINE)
        text_out = re.sub(r'\*', '', text_out, flags=re.MULTILINE)
        return text_out

    def get_stopwords(self, stopwords_file='stopwords.txt'):
        with open(stopwords_file, 'r') as f:
            stop_w = f.readlines()
        stopwords = [s.rstrip() for s in stop_w]
        return list(text.ENGLISH_STOP_WORDS.union(stopwords))

    def create_bigrams(self, texts, bigram_count=15, threshold=10):
        bigram_model = Phraser(Phrases(texts, min_count=bigram_count, threshold=threshold))
        return [" ".join(bigram_model[t]) for t in texts]

    def lemmatize_texts(self, texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'], spacy_model=None):
        if not spacy_model:
            print("Loading spacy model")
            spacy_model = spacy.load('en_core_web_md')

        print("Beginning lemmatization process")
        total_steps = len(texts)
        docs = spacy_model.pipe(texts)
        
        texts_out = []
        for doc in tqdm(docs, total=total_steps):
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])

        return [" ".join(text) for text in texts_out]

class TextAnalyzer(TextProcessor):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        self.texts = self.force_format(self.texts)
        self.texts = [self.filter_text(t) for t in self.texts]

    def analyze(self):
        print("Checking data quality...")
        print(f"Is the dataset passing our data quality check? {self.check_data_quality(self.texts)}")

        print("Splitting texts into words...")
        splitted_texts = self.dummy_word_split(self.texts)
        print("First sample text split:", splitted_texts[209526])

        avg_length = np.mean([len(text) for text in splitted_texts])
        print(f"Average title length: {avg_length} words")
        print(f"Total number of unique titles: {len(splitted_texts)}")

        print("Computing word occurrences...")
        word_count = self.compute_word_occurences(splitted_texts)
        print(word_count.head(50))

        self.texts = [[word for word in txt if word not in self.stopwords] for txt in tqdm(splitted_texts)]
        print(f"After stopword removal, total unique texts: {len(self.texts)}")

        self.texts = self.create_bigrams(self.texts)
        print("Bigram creation complete.")

        lemmatized_texts = self.lemmatize_texts(self.texts[:1000])
        print("Lemmatization complete.")
        lemmatized_word_count = self.compute_word_occurences(lemmatized_texts)
        print(lemmatized_word_count.head(50))


if __name__ == "__main__":
    dataset_path = "News_Category_Dataset_v3.json"  
    analyzer = TextAnalyzer(dataset_path)
    analyzer.analyze()
