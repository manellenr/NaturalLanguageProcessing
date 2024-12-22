import gensim
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import pandas as pd
import pyLDAvis.gensim_models

class LDAModeler:
    def __init__(self, texts):
        if not texts or not all(isinstance(doc, list) and doc for doc in texts):
            raise ValueError("The input `texts` is empty or not correctly formatted.")
        self.texts = texts
        self.dictionary = None
        self.bow_corpus = None
        self.lda_model = None

    def preprocess(self, no_below=2, no_above=0.5, keep_n=5000):
        """Creates dictionary and Bag-of-Words (BoW) corpus."""
        self.dictionary = Dictionary(self.texts)
        self.dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)

        print(f"Dictionary size after filtering: {len(self.dictionary)}")
        if len(self.dictionary) == 0:
            raise ValueError("The dictionary is empty after filtering. Adjust `filter_extremes` parameters.")

        self.bow_corpus = [self.dictionary.doc2bow(doc) for doc in self.texts]
        print(f"BoW sample: {self.bow_corpus[:1]}")
        if not self.bow_corpus:
            raise ValueError("The Bag-of-Words (BoW) corpus is empty.")

    def train_lda(self, num_topics=10, passes=10):
        """Trains the LDA model."""
        if not self.bow_corpus or not self.dictionary:
            raise ValueError("Preprocess the texts before training the LDA model.")

        self.lda_model = gensim.models.LdaMulticore(
            self.bow_corpus, num_topics=num_topics, id2word=self.dictionary, passes=passes
        )
        print("LDA model training completed.")

    def get_topics(self):
        """Retrieves topics from the trained LDA model."""
        if not self.lda_model:
            raise ValueError("Train the LDA model before getting topics.")

        topics = []
        for idx, topic in self.lda_model.print_topics(-1):
            print(f"Topic: {idx} -> Words: {topic}")
            topics.append(topic)
        return topics

    def compute_coherence(self):
        """Computes the coherence score of the LDA model."""
        if not self.lda_model:
            raise ValueError("Train the LDA model before computing coherence.")

        coherence_model_lda = CoherenceModel(model=self.lda_model, texts=self.texts, dictionary=self.dictionary)
        coherence_lda = coherence_model_lda.get_coherence()
        print(f"Coherence Score: {coherence_lda}")
        return coherence_lda

    def visualize(self):
        """Visualizes the LDA model using pyLDAvis."""
        if not self.lda_model or not self.bow_corpus:
            raise ValueError("Train the LDA model and preprocess the texts before visualization.")

        vis = pyLDAvis.gensim_models.prepare(self.lda_model, self.bow_corpus, self.dictionary)
        return vis

    def extract_topic_words(self, top_n=10):
        """Extracts words for each topic and organizes them in a DataFrame."""
        topics = self.get_topics()
        all_topic_model = []

        for topic in topics:
            terms = topic.split(' + ')
            topic_model = []
            for term in terms[:top_n]:
                weight, word = term.split('*')
                topic_model.append((float(weight.strip()), word.strip().strip('"')))
            all_topic_model.append(topic_model)

        df_topic_model = pd.DataFrame(all_topic_model)
        df_topic_model.rename(index={i: f"Topic {i + 1}" for i in range(len(all_topic_model))}, inplace=True)
        print(df_topic_model)
        return df_topic_model
