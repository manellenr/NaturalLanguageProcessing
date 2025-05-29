# Topic Modeling on HuffPost News Dataset

## Overview

This project consists of performing topic modeling on the file News_Category_Dataset_v3.json, a dataset containing approximately 210,000 news headlines published between 2012 and 2022 on the HuffPost website. The dataset is available [here](https://www.kaggle.com/datasets/rmisra/news-category-dataset).

The objective is to study the performance of topic modeling algorithms on this dataset and to identify significant information related to the content of the news articles.

## Dataset

- **News_Category_Dataset_v3.json**: This JSON file contains the main dataset with news headlines and associated metadata.
- **stopwords.txt**: This text file contains a list of common stopwords used during the text preprocessing phase.

## Objectives

- **Analyze the text corpus**: Extract key features such as average length, word types, most frequent words, and stopwords.

- **Select Topic Modeling methodology**: Use the LDA (Latent Dirichlet Allocation) algorithm to extract topics from the dataset.

## LDA Output Example

After applying the LDA algorithm to the dataset, I obtained the following result for the **Top-20 Most Relevant Terms**:

![image](https://github.com/user-attachments/assets/5f9d2636-7485-44b5-b390-3ed0acc044ad)

These terms represent the most relevant words within the selected topic, reflecting the main themes discussed in the news articles related to this topic.

The **Top-20 Most Salient Terms** are as follows:

![image](https://github.com/user-attachments/assets/37cbe7a2-9e28-4c11-aa50-c0dfd8c7a466)

