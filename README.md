# Text Analytics Project: Sentiment Analysis of Amazon Product Reviews

---

## Master's Degree Thesis Project in Data Science and Business Informatics

**University of Pisa**

* [cite_start]**Author**: Giuseppe Muschetta [cite: 1]
* [cite_start]**Thesis Supervisor**: Professor Laura Pollacci [cite: 1]

---

## Project Overview: Unveiling Customer Sentiments from Amazon Reviews

[cite_start]This project, developed as part of a Master's degree thesis at the University of Pisa, focuses on applying **Natural Language Processing (NLP)** and advanced machine learning techniques to perform **sentiment analysis on a large dataset of Amazon product reviews**[cite: 4, 6]. [cite_start]The core objective is to extract actionable insights from customer feedback to inform strategic decision-making and enhance product offerings and service quality in the e-commerce landscape[cite: 3, 5, 248, 249].

[cite_start]The project encompasses a comprehensive data science workflow, from initial data understanding and meticulous text preparation to the development and evaluation of various sentiment analysis models, including traditional machine learning algorithms and advanced deep learning architectures. [cite: 6, 7, 8]

### Key Project Phases:

1.  [cite_start]**Data Understanding & Exploration**: In-depth analysis of the dataset's structure, statistical properties, and initial observations regarding review scores, lengths, and temporal trends[cite: 1, 9, 10, 37, 38].
2.  [cite_start]**Data Preparation and Cleaning**: Rigorous text preprocessing to transform raw review data into a clean, normalized, and semantically enriched format suitable for machine learning[cite: 1, 53, 54, 55, 56, 57].
3.  [cite_start]**Machine Learning Models for Sentiment Analysis**: Implementation and evaluation of various models to classify sentiments as positive or negative, ranging from classic methods to state-of-the-art deep learning approaches[cite: 1, 92, 93, 94, 95].

---

## Dataset: Amazon Product Reviews

[cite_start]The analysis utilizes a comprehensive dataset of Amazon product reviews, sourced from Kaggle[cite: 9, 11].

* [cite_start]**Source**: [Amazon Product Reviews dataset on Kaggle](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews) [cite: 11]
* [cite_start]**Size**: Approximately 568,453 rows and 10 columns[cite: 12].
* [cite_start]**Time Range**: Reviews span from 1999 to 2012[cite: 18, 29].
* **Key Columns**:
    * [cite_start]`Text`: Contains the main body of the customer review[cite: 13].
    * [cite_start]`Score`: Numerical rating from 1 to 5[cite: 14].
    * [cite_start]`Summary`: A short summary of the review (excluded from sentiment analysis to avoid providing excessive information to models)[cite: 14, 16].
    * [cite_start]`Product ID`: Identifier for the reviewed product[cite: 14].
    * [cite_start]`ProfileName`: User's name or nickname[cite: 15].
    * [cite_start]`Time`: Review timestamp[cite: 15].
* **Initial Observations**:
    * [cite_start]The `Score` field averages 4.183 with a standard deviation of 1.31, indicating a broad spectrum of customer satisfaction, typical for Amazon products[cite: 20, 21, 22].
    * [cite_start]A significant bias towards positive ratings was observed, with 63.9% (363,122 reviews) having a 5-star rating[cite: 23, 41, 42].
    * [cite_start]Average review length is approximately 436 characters or 80 words[cite: 24, 25].
    * [cite_start]Review volume showed an exponential increase from 2007 onwards, attributed to increased internet adoption and Amazon's growing brand recognition[cite: 30, 31, 32, 39].

---

## Project Phases and Methodologies

### [cite_start]1. Data Understanding & Exploration [cite: 1, 9]

This phase involved a detailed examination of the dataset's characteristics and patterns.

* [cite_start]**Statistical Summary**: Analyzed ratings distribution, review lengths, and temporal trends[cite: 20, 24, 29, 34, 38].
* [cite_start]**Sentiment Trends Over Time**: Observed an exponential growth in positive reviews significantly outpacing negative reviews from 2007, indicating increased customer satisfaction and Amazon's popularity[cite: 35, 38, 39].
* [cite_start]**Demographic Insights**: Explored review patterns based on gender and day of the week[cite: 28, 29].

### [cite_start]2. Data Preparation and Cleaning [cite: 1, 414]

This crucial phase transformed raw text into a machine-ready format.

* **Sentiment Classification**: Reviews with ratings 4 or 5 were classified as positive, and ratings 1 or 2 as negative. [cite_start]Neutral reviews (rating 3) were excluded[cite: 43, 44, 46].
* [cite_start]**Class Imbalance Handling**: The initial dataset was heavily imbalanced (443,777 positive vs. 82,037 negative reviews)[cite: 48]. [cite_start]To ensure robust model training, the dataset was balanced by downsampling the majority class, resulting in 82,037 positive and 82,037 negative reviews[cite: 50, 51].
* **Text Preprocessing**: A series of advanced techniques were applied:
    * [cite_start]**Tokenization & Lemmatization**: Performed using spaCy, reducing words to their base forms[cite: 53, 60, 61, 68, 90].
    * [cite_start]**Contraction Expansion**: Utilized the `contractions` library to expand contracted forms (e.g., "don't" to "do not")[cite: 53, 63, 89].
    * [cite_start]**Noise Removal**: Eliminated numbers, punctuation, useless spaces, web/email addresses, and HTML/XML tags using `RE` (Regular Expressions) and `BeautifulSoup`[cite: 53, 58, 59, 86].
    * [cite_start]**Stopword Removal**: Filtered out common, uninformative words using NLTK's comprehensive English stopwords list[cite: 57, 87].
    * [cite_start]**Negation Handling**: Custom functions were developed to accurately manage negations (e.g., "not good" becomes "not_good") to preserve contextual sentiment[cite: 65, 66, 67, 88].
    * [cite_start]**Collocation Detection (Bigrams)**: Gensim was used to identify frequently paired words, enhancing semantic understanding[cite: 53, 64, 82, 83].
    * [cite_start]**Emoji Preservation**: Emojis were intentionally kept as they are valuable sentiment indicators[cite: 62, 78, 79].
* [cite_start]**Wordcloud Visualization**: Generated word clouds for both raw and cleaned text (for positive and negative sentiments) to visually demonstrate the effectiveness of the cleaning process[cite: 69, 70, 71, 72].

### [cite_start]3. Machine Learning Models for Sentiment Analysis [cite: 1, 92]

[cite_start]The project explored and evaluated multiple machine learning and deep learning models for sentiment classification. [cite: 92, 93, 94, 95]

#### 3.1. [cite_start]Naïve Bayes: MultinomialNB [cite: 1, 99]

* [cite_start]**Pipeline**: Utilized a pipeline incorporating TF-IDF for text vectorization, `SelectKBest` with chi-squared for feature extraction (selecting top 20,000 features), and `Multinomial Naïve Bayes` for classification[cite: 100, 101, 102, 103, 104, 105].
* [cite_start]**Hyperparameter Tuning**: Employed 5-fold cross-validation with grid search for optimal parameters (e.g., `max_df=0.5`, `min_df=10` for TF-IDF; `alpha=0.5` for model smoothing)[cite: 118, 119, 120, 121, 122, 123, 124, 125].
* [cite_start]**Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC Curve with AUC, and Precision-Recall Curve were used for comprehensive model assessment[cite: 112, 113, 114, 115, 116, 127, 128, 133].

#### 3.2. [cite_start]Support Vector Machine: LinearSVC [cite: 1, 136]

* [cite_start]**Strengths**: Known for handling high-dimensional data, margin maximization, scalability, and flexibility[cite: 137, 138, 139, 140]. [cite_start]Generally outperforms Naïve Bayes due to better handling of feature interactions[cite: 143, 144, 145].
* [cite_start]**Weaknesses**: Sensitivity to noisy data and parameter tuning[cite: 141, 142].
* **Hyperparameter Tuning**: 5-fold cross-validation with grid search. [cite_start]Key parameters included TF-IDF settings (similar to Naïve Bayes) and LinearSVC specific parameters (`C=1`, `loss='squared_hinge'`, `max_iter=1000`, `penalty='l2'`)[cite: 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158].
* [cite_start]**Evaluation Metrics**: Same comprehensive set of metrics as Naïve Bayes[cite: 158].

#### 3.3. [cite_start]DistilBERT Pre-trained Model [cite: 1, 159]

* [cite_start]**Architecture**: A streamlined version of BERT, retaining 97% of BERT's language understanding while being smaller and faster (6 Transformer layers, ~66 million parameters)[cite: 160, 161]. [cite_start]Uses knowledge distillation to emulate BERT's behavior[cite: 163].
* [cite_start]**Initialization**: `DistilBertForSequenceClassification` initialized from `distilbert-base-uncased` with two labels for binary classification[cite: 167, 168].
* [cite_start]**Optimization**: AdamW optimizer with a learning rate of $5 \times 10^{-5}$ and an epsilon of $10^{-8}$[cite: 169, 170]. [cite_start]A linear scheduler adjusts the learning rate[cite: 171, 172, 173].
* [cite_start]**Training**: Conducted on GPU using Apple's Metal Performance Shaders (MPS), with gradient clipping to prevent exploding gradients[cite: 174, 175, 176, 177].
* [cite_start]**Key Achievement**: The custom deep learning model achieved performance metrics comparable to DistilBERT, despite having significantly fewer parameters (2.5 million vs. 66 million), highlighting efficiency[cite: 97, 98].

#### 3.4. [cite_start]Custom Deep Learning Model [cite: 1, 181]

* [cite_start]**Architecture**: Designed for sentiment analysis of textual data, incorporating various layers to capture sequential and contextual information[cite: 182, 183].
    * [cite_start]**Custom Embedding Layer**: Converts word indices into dense, updatable vectors[cite: 184, 185].
    * [cite_start]**1D Convolutional Layer**: Extracts local features with ReLU activation, followed by batch normalization[cite: 186, 187].
    * [cite_start]**MaxPooling1D Layer**: Reduces dimensionality for computational efficiency[cite: 188].
    * [cite_start]**Bidirectional LSTM Layers (Two)**: Crucial for capturing long-term dependencies in both directions of the sequence, enhancing pattern learning[cite: 189, 190, 191].
    * [cite_start]**Dropout Layers**: Strategically placed after LSTM and dense layers to combat overfitting[cite: 192, 193].
    * [cite_start]**Dense Layer**: Preceding the output, with L2 regularization and Kaiming He initialization[cite: 194].
    * [cite_start]**Output Layer**: Sigmoid activation for binary classification[cite: 195].
* [cite_start]**Compilation**: Uses RMSprop optimizer (often preferred for text data due to stable learning rates in noisy/sparse gradient scenarios) and binary crossentropy loss function[cite: 196, 197, 198, 199, 200].
* [cite_start]**Overall Optimization**: Balances depth, complexity, computational efficiency, and robustness against overfitting, making it highly suitable for voluminous and nuanced Amazon review data[cite: 201, 202].

---
