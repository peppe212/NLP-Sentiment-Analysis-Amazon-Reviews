# Sentiment Analysis of Amazon Product Reviews

## Introduction
Sentiment analysis is an important technique in data science, used to interpret the opinions embedded in text. This project embarks on a methodological study of sentiment in Amazon product reviews. Using a set of machine learning models, we aim to categorize user feedback into two sentiment classes: "positive" and "negative". Excluding neutral sentiment is a strategic choice, as it often represents a lack of clear opinion or impact on the consumer experience. For product reviews, definitive sentiments, such as positive and negative, are more valuable for driving business insights and decisions, providing a clear measure of customer satisfaction and areas for improvement.

## Motivation
Analyzing the sentiment in Amazon reviews is key to gaining insight into customer satisfaction and preferences. This project focuses on the positive and negative aspects of customer feedback, allowing businesses to clearly identify and respond to the most salient opinions. By deliberately excluding neutral sentiments, we focus on the feedback that most strongly influences consumer behavior and business outcomes. This targeted approach not only enhances customer engagement tactics but also provides a robust foundation for sentiment analysis research within e-commerce, ultimately informing more precise and impactful business strategies.

## Dataset Overview
The Amazon Product Reviews dataset, sourced from Kaggle, offers a rich composition of user-generated content with the following specifications:
- **Records**: 568454
- **Columns**: Id, ProductId, UserId, ProfileName, HelpfulnessNumerator, HelpfulnessDenominator, Score, Time, Summary, Text
- **Domain**: amazon.com
- **Format**: .CSV

The dataset provides a vast and diverse collection of textual reviews, suitable for training robust sentiment analysis models. It supports .csv format and the availability of sentiment labels streamline the preprocessing stage, allowing for a focus on model development and performance optimization.

## Tasks
- **Data Understanding and Exploration**: Statistical analysis will be used to gain a basic understanding of the data distribution, examine trends, identify anomalies, and recognize the underlying structure within the data. Visual representations are also used to identify the characteristics of the data set.
- **Data Preparation and Cleaning**: To guarantee data quality and uniformity, our process incorporates sophisticated Natural Language Processing (NLP) techniques to meticulously extract features from the raw review texts. We implement tokenization, elimination of stop words, lemmatization, and part-of-speech tagging. Additionally, our data preparation includes the expansion of contractions and the removal of HTML tags, URLs, and email addresses. We also ensure the retention of emojis within the raw text, recognizing their crucial role in conveying the sentiments of the reviews.
- **Machine Learning Model Development and Training**: To implement a strategic selection of models, including:
  1. **Naïve Bayes Classifier**: An excellent starting point for model benchmarking due to its simplicity and computational speed. Despite its naïve assumption of feature independence, it performs surprisingly well for text classification tasks and will serve as our baseline model.
  2. **Support Vector Machine (LinearSVC)**: Employs a sophisticated approach to enhance the model's capability to predict outcomes with greater precision by effectively leveraging the data features it analyzes.
  3. **Deep Learning Model**: Uses Sequential API model, custom-built to analyze the subtleties and nuances of language expressed in Amazon reviews. Our model was trained and tuned on a curated dataset to achieve high accuracy in sentiment classification.
  4. **DistilBERT**: A distilled version of the larger BERT model, pre-trained on vast text corpora to grasp the subtle nuances of language context efficiently. It will be fine-tuned on our dataset to extract concise and profound insights into the sentiment conveyed within the Amazon reviews.
- **Model Evaluation**: Rigorous assessment of each model using metrics such as accuracy, precision, recall, and F1-score.
- **Analysis of Results**: To understand the general sentiment expressed in the reviews.

## Methodology
A detailed account of the methodology will include data partitioning strategies for model training, validation, and testing phases, as well as the application of cross-validation techniques to ensure model reliability and prevent overfitting.

## Conclusion
Through the sentiment analysis of Amazon reviews, this project aims to deliver a clear depiction of consumer opinion on a global scale. The proposed systematic and scientific approach ensures robust, replicable results that contribute significantly to both academic research and business practices in the digital commerce landscape.
