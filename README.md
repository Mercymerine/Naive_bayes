# Naive_bayes
# Fake News Detection Using Naive Bayes
This project applies the Naive Bayes algorithm to classify news articles as fake or real. The goal is to build an effective and efficient model for detecting fake news, leveraging probabilistic machine learning techniques.

### Table of Contents
Overview

Dataset

Data Preprocessing

Exploratory Data Analysis (EDA)

Modeling

Evaluation

Technologies Used


Key Steps in the Workflow

### Overview
Naive Bayes is a simple yet powerful probabilistic machine learning model used for classification problems. It is particularly useful for large datasets and works by calculating the probability of an item belonging to a certain class based on its features. In this project, we use Naive Bayes to classify news articles as fake or real, which is a crucial task in today's digital world.

### Key Goals:
Build a robust and scalable pipeline for data preprocessing, training, and evaluation.

Leverage cross-validation to enhance model reliability.

Visualize data relationships and identify correlations between features.

### Dataset
The dataset used in this project was obtained from Kaggle:

Fake News Detection Dataset

Features:
Text: The content of the news article.

Label: The target variable, indicating whether the news is fake (0) or real (1).

### Data Preprocessing
Loading and Cleaning Data:
Download the dataset from Kaggle.

Extract the dataset and load it into a pandas DataFrame.

Inspect the structure and check for missing values.

#### Feature Selection:
Convert labels to binary values (Fake to 0, Real to 1).

Use CountVectorizer to convert text data into numeric features suitable for machine learning models.

#### Train-Test Split:
Data split into 80% training and 20% testing subsets.

### Exploratory Data Analysis (EDA)
Visualize the distribution of fake and real news articles.

Explore the text data to identify common words and phrases in fake and real news.

### Modeling
Model Implemented:
Naive Bayes:
Applied Multinomial Naive Bayes to the text classification problem.

Converted text data into numeric features using CountVectorizer.

Trained the model on the training set and evaluated it on the testing set.

### Evaluation
Cross-validation was used to ensure robust model performance across multiple folds. The Naive Bayes model demonstrated high accuracy in classifying news articles as fake or real.

### Model Evaluation Metrics:
Accuracy: Percentage of correctly classified instances.

Confusion Matrix: Summary of prediction results.

Classification Report: Precision, recall, and F1-score for each class.

Technologies Used
Programming Language: Python

Libraries:

pandas, numpy for data manipulation

matplotlib, seaborn for visualization

scikit-learn for machine learning

### Conclusion
The dataset preprocessing was thorough, and the Naive Bayes model was effective in achieving high accuracy for classifying news articles.

