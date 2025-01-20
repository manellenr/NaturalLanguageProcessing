# Topic Modeling on HuffPost News Dataset

## Overview

This project involves performing thematic modelling on `News_Category_Dataset_v3.json`, a dataset containing approximately 210,000 news headlines from 2012 to 2022 from HuffPost. This dataset can be used as a reference for a variety of computational linguistic tasks. Due to changes on the HuffPost website, the dataset includes approximately 200,000 headlines from 2012 to May 2018 and 10,000 headlines from May 2018 to 2022. You can access the dataset [here](https://www.kaggle.com/datasets/rmisra/news-category-dataset).

The aim is to study the performance of Topic Modeling algorithms on this dataset and to identify significant information linked to the content of press articles.

## Dataset Description

The dataset consists of news articles with the following attributes:

- **category**: The category in which the article was published.
- **headline**: The headline of the news article.
- **authors**: List of authors who contributed to the article.
- **link**: A URL link to the original news article.
- **short_description**: A brief abstract of the news article.
- **date**: The publication date of the article.

## Dataset Files

- **News_Category_Dataset_v3.json**: This JSON file contains the main dataset of news headlines and associated metadata.
- **stopwords.txt**: This text file contains a list of common stopwords to be used in the text preprocessing phase.

## Project Objectives

The main objectives of this project are:

1. **Analyze the text corpus**: Extract key features such as average size, word types, most frequent words and empty words.

2. **Select Topic Modeling methodology**: Use the Latent Dirichlet Allocation (LDA) algorithm to extract topics from the dataset.

3. **Define metrics for model evaluation**: Develop appropriate metrics to assess the quality of the LDA model.

4. **Conclude on the methodology**: Evaluate the effectiveness of the LDA model and propose ways to improve the analysis.

## LDA Output Example

After applying the LDA algorithm to the dataset, we obtained the following result for the **Top-20 Most Relevant Terms**:

![image](https://github.com/user-attachments/assets/5f9d2636-7485-44b5-b390-3ed0acc044ad)

These terms represent the most relevant words within the selected topic, reflecting the main themes discussed in the news articles related to this topic.

The **Top-20 Most Salient Terms** are as follows:

![image](https://github.com/user-attachments/assets/37cbe7a2-9e28-4c11-aa50-c0dfd8c7a466)

