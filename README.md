# Text Analytics Project: Sentiment Analysis of Amazon Product Reviews

---

## Master's Degree Thesis Project in Data Science and Business Informatics

**University of Pisa**

* **Author**: Giuseppe Muschetta
* **Thesis Supervisor**: Professor Laura Pollacci

---

## Project Overview: Unveiling Customer Sentiments from Amazon Reviews

This project, developed as part of a Master's degree thesis at the University of Pisa, focuses on applying **Natural Language Processing (NLP)** and advanced machine learning techniques to perform **sentiment analysis on a large dataset of Amazon product reviews**. The core objective is to extract actionable insights from customer feedback to inform strategic decision-making and enhance product offerings and service quality in the e-commerce landscape.

The project encompasses a comprehensive data science workflow, from initial data understanding and meticulous text preparation to the development and evaluation of various sentiment analysis models, including traditional machine learning algorithms and advanced deep learning architectures.

### Key Project Phases:

1.  **Data Understanding & Exploration**: In-depth analysis of the dataset's structure, statistical properties, and initial observations regarding review scores, lengths, and temporal trends.
2.  **Data Preparation and Cleaning**: Rigorous text preprocessing to transform raw review data into a clean, normalized, and semantically enriched format suitable for machine learning.
3.  **Machine Learning Models for Sentiment Analysis**: Implementation and evaluation of various models to classify sentiments as positive or negative, ranging from classic methods to state-of-the-art deep learning approaches.

---

## Dataset: Amazon Product Reviews

The analysis utilizes a comprehensive dataset of Amazon product reviews, sourced from Kaggle.

* **Source**: [Amazon Product Reviews dataset on Kaggle](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews)
* **Size**: Approximately 568,453 rows and 10 columns.
* **Time Range**: Reviews span from 1999 to 2012.
* **Key Columns**:
    * `Text`: Contains the main body of the customer review.
    * `Score`: Numerical rating from 1 to 5.
    * `Summary`: A short summary of the review (excluded from sentiment analysis to avoid providing excessive information to models).
    * `Product ID`: Identifier for the reviewed product.
    * `ProfileName`: User's name or nickname.
    * `Time`: Review timestamp.
* **Initial Observations**:
    * The `Score` field averages 4.183 with a standard deviation of 1.31, indicating a broad spectrum of customer satisfaction, typical for Amazon products.
    * A significant bias towards positive ratings was observed, with 63.9% (363,122 reviews) having a 5-star rating.
    * Average review length is approximately 436 characters or 80 words.
    * Review volume showed an exponential increase from 2007 onwards, attributed to increased internet adoption and Amazon's growing brand recognition.

---

## Project Phases and Methodologies

### 1. Data Understanding & Exploration

This phase involved a detailed examination of the dataset's characteristics and patterns.

* **Statistical Summary**: Analyzed ratings distribution, review lengths, and temporal trends.
* **Sentiment Trends Over Time**: Observed an exponential growth in positive reviews significantly outpacing negative reviews from 2007, indicating increased customer satisfaction and Amazon's popularity.
* **Demographic Insights**: Explored review patterns based on gender and day of the week.

### 2. Data Preparation and Cleaning

This crucial phase transformed raw text into a machine-ready format.

* **Sentiment Classification**: Reviews with ratings 4 or 5 were classified as positive, and ratings 1 or 2 as negative. Neutral reviews (rating 3) were excluded.
* **Class Imbalance Handling**: The initial dataset was heavily imbalanced (443,777 positive vs. 82,037 negative reviews). To ensure robust model training, the dataset was balanced by downsampling the majority class, resulting in 82,037 positive and 82,037 negative reviews.
* **Text Preprocessing**: A series of advanced techniques were applied:
    * **Tokenization & Lemmatization**: Performed using spaCy, reducing words to their base forms.
    * **Contraction Expansion**: Utilized the `contractions` library to expand contracted forms (e.g., "don't" to "do not").
    * **Noise Removal**: Eliminated numbers, punctuation, useless spaces, web/email addresses, and HTML/XML tags using `RE` (Regular Expressions) and `BeautifulSoup`.
    * **Stopword Removal**: Filtered out common, uninformative words using NLTK's comprehensive English stopwords list.
    * **Negation Handling**: Custom functions were developed to accurately manage negations (e.g., "not good" becomes "not_good") to preserve contextual sentiment.
    * **Collocation Detection (Bigrams)**: Gensim was used to identify frequently paired words, enhancing semantic understanding.
    * **Emoji Preservation**: Emojis were intentionally kept as they are valuable sentiment indicators.
* **Wordcloud Visualization**: Generated word clouds for both raw and cleaned text (for positive and negative sentiments) to visually demonstrate the effectiveness of the cleaning process.

### 3. Machine Learning Models for Sentiment Analysis

The project explored and evaluated multiple machine learning and deep learning models for sentiment classification.

#### 3.1. Naïve Bayes: MultinomialNB

* **Pipeline**: Utilized a pipeline incorporating TF-IDF for text vectorization, `SelectKBest` with chi-squared for feature extraction (selecting top 20,000 features), and `Multinomial Naïve Bayes` for classification.
* **Hyperparameter Tuning**: Employed 5-fold cross-validation with grid search for optimal parameters (e.g., `max_df=0.5`, `min_df=10` for TF-IDF; `alpha=0.5` for model smoothing).
* **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC Curve with AUC, and Precision-Recall Curve were used for comprehensive model assessment.

#### 3.2. Support Vector Machine: LinearSVC

* **Strengths**: Known for handling high-dimensional data, margin maximization, scalability, and flexibility. Generally outperforms Naïve Bayes due to better handling of feature interactions.
* **Weaknesses**: Sensitivity to noisy data and parameter tuning.
* **Hyperparameter Tuning**: 5-fold cross-validation with grid search. Key parameters included TF-IDF settings (similar to Naïve Bayes) and LinearSVC specific parameters (`C=1`, `loss='squared_hinge'`, `max_iter=1000`, `penalty='l2'`).
* **Evaluation Metrics**: Same comprehensive set of metrics as Naïve Bayes.

#### 3.3. DistilBERT Pre-trained Model

* **Architecture**: A streamlined version of BERT, retaining 97% of BERT's language understanding while being smaller and faster (6 Transformer layers, ~66 million parameters). Uses knowledge distillation to emulate BERT's behavior.
* **Initialization**: `DistilBertForSequenceClassification` initialized from `distilbert-base-uncased` with two labels for binary classification.
* **Optimization**: AdamW optimizer with a learning rate of $5 \times 10^{-5}$ and an epsilon of $10^{-8}$. A linear scheduler adjusts the learning rate.
* **Training**: Conducted on GPU using Apple's Metal Performance S (MPS), with gradient clipping to prevent exploding gradients.
* **Key Achievement**: The custom deep learning model achieved performance metrics comparable to DistilBERT, despite having significantly fewer parameters (2.5 million vs. 66 million), highlighting efficiency.

#### 3.4. Custom Deep Learning Model

* **Architecture**: Designed for sentiment analysis of textual data, incorporating various layers to capture sequential and contextual information.
    * **Custom Embedding Layer**: Converts word indices into dense, updatable vectors.
    * **1D Convolutional Layer**: Extracts local features with ReLU activation, followed by batch normalization.
    * **MaxPooling1D Layer**: Reduces dimensionality for computational efficiency.
    * **Bidirectional LSTM Layers (Two)**: Crucial for capturing long-term dependencies in both directions of the sequence, enhancing pattern learning.
    * **Dropout Layers**: Strategically placed after LSTM and dense layers to combat overfitting.
    * **Dense Layer**: Preceding the output, with L2 regularization and Kaiming He initialization.
    * **Output Layer**: Sigmoid activation for binary classification.
* **Compilation**: Uses RMSprop optimizer (often preferred for text data due to stable learning rates in noisy/sparse gradient scenarios) and binary crossentropy loss function.
* **Overall Optimization**: Balances depth, complexity, computational efficiency, and robustness against overfitting, making it highly suitable for voluminous and nuanced Amazon review data.

---
