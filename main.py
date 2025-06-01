from lda_pipeline import *

data = load_dataset("data/News_Category_Dataset_v3.json", nrows=1000)
texts = force_format(data["headline"])
check_data_quality(texts)

texts = preprocess_texts(texts)
texts = list(sent_to_words(texts))

stopwords = get_stopwords(additional_stopwords=["trump", "jan", "ex"])
texts = remove_stopwords(texts, stopwords)

texts = create_bigrams(texts, as_str=False)

lemmatized_texts = lemmatize_texts([" ".join(txt) for txt in texts], allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV', 'X', 'PROPN'])

print(f"Taille moyenne des titres : {average_title_length(lemmatized_texts)}")
print(f"Mots les plus fréquents :\n{compute_word_occurences(lemmatized_texts).head(10)}")

lda_model, dictionary, bow_corpus = run_lda(lemmatized_texts, num_topics=10)
topics = print_topics(lda_model)
coherence = compute_coherence(lda_model, lemmatized_texts, dictionary)
print(f"Coherence Score: {coherence}")
