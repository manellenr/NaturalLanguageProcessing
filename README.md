# Topic Modeling on HuffPost News Dataset

## Overview

This project involves performing Topic Modeling on the `News_Category_Dataset_v3.json`, a dataset containing around 210,000 news headlines from 2012 to 2022 from HuffPost. This dataset can be used as a benchmark for a variety of computational linguistic tasks. Due to changes in the HuffPost website, the dataset includes approximately 200,000 headlines from 2012 to May 2018 and 10,000 headlines from May 2018 to 2022.

The objective is to study the performance of Topic Modeling algorithms on this dataset and identify meaningful insights related to the content of the news articles.

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

1. **Analyze the text corpus**: Extract key characteristics such as average size, word types, most frequent words, and stopwords.

2. **Select Topic Modeling methodology**: Use the Latent Dirichlet Allocation (LDA) algorithm to extract topics from the dataset.

3. **Define metrics for model evaluation**: Develop appropriate metrics to assess the quality of the LDA model.

4. **Conclude on the methodology**: Evaluate the effectiveness of the LDA model and propose ways to improve the analysis.

5. **Conclude on the best methodology**: Identify the best methodology for topic modeling and propose ways to improve the analysis.

   ## LDA Output Example

After applying the LDA algorithm to the dataset, we obtain the following result for **Topic 1** (13.9% of tokens). The **Top-20 Most Relevant Terms** for Topic 1 are:

![image](https://github.com/user-attachments/assets/5f9d2636-7485-44b5-b390-3ed0acc044ad)

