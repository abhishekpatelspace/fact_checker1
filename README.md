# 🕵️‍♂️ Fact-Check ML Pipeline

An end-to-end Machine Learning pipeline and interactive Streamlit web application designed to automatically classify and fact-check political statements and general claims.

## 🌟 Project Overview

This project provides a comprehensive framework for detecting misinformation. It automates the collection of ground-truth data, performs advanced linguistic feature engineering, trains multiple classification models (both traditional ML and Deep Learning), and evaluates them against real-world data using the Google Fact Check Tools API.

The entire pipeline is wrapped in a premium, glassmorphism-themed Streamlit dashboard, allowing users to interactively scrape data, train models, and visualize performance metrics in real-time.

## 🚀 Key Features

*   **Automated Web Scraping:** Extracts fact-checked political statements, sources, dates, and truth labels directly from PolitiFact within user-defined date ranges.
*   **Advanced Feature Engineering:** Analyzes text across five linguistic dimensions:
    *   **Lexical & Morphological:** Lemma extraction and N-gram vectorization using SpaCy.
    *   **Syntactic:** Part-of-Speech (POS) tagging combined with TF-IDF.
    *   **Semantic:** Sentiment polarity and subjectivity scoring via TextBlob.
    *   **Discourse:** Sentence structure and transition analysis.
    *   **Pragmatic:** Detection of modal verbs and specific punctuation patterns.
*   **Diverse Model Architecture:** Implements and compares 6 distinct algorithms:
    *   Naive Bayes (MultinomialNB)
    *   Decision Tree
    *   Logistic Regression
    *   Support Vector Machine (SVM)
    *   Artificial Neural Network (ANN - Feedforward)
    *   Recurrent Neural Network (RNN - Bidirectional LSTM)
*   **Robust Training Pipeline:** 
    *   Maps granular truth ratings (e.g., "Pants on Fire", "Half True") into binary classifications (Fake/Real).
    *   Utilizes **Stratified 5-Fold Cross-Validation** to ensure reliable evaluation.
    *   Applies **SMOTE (Synthetic Minority Over-sampling Technique)** to handle severe class imbalances inherent in fact-checking datasets.
*   **Real-World API Benchmarking:** Connects to the **Google Fact Check Tools API** to dynamically fetch diverse, real-world claims and benchmark the trained models against external ground-truth data, measuring Accuracy, F1-Score, Precision, Recall, and Inference Latency.

## 📊 Key Findings & Insights

Through the development and testing of this pipeline, several key observations were made regarding automated fact-checking:

1.  **Feature Importance:** Syntactic features (TF-IDF on POS tags) and Lexical N-grams proved to be the most robust standalone predictors. Semantic and Pragmatic features are valuable but function best as auxiliary inputs rather than primary classifiers.
2.  **Class Imbalance is Critical:** Political datasets heavily skew toward specific ratings. Implementing SMOTE inside the cross-validation pipeline significantly improved the **Recall** of minority classes, preventing the models from simply guessing the majority class.
3.  **Deep Learning vs. Traditional ML:** 
    *   The **Bidirectional LSTM (RNN)** excels at capturing the sequential context and nuances of longer, complex claims, though it requires more training data and has higher inference latency.
    *   **Logistic Regression** and **SVM** offer excellent baseline accuracy with remarkably low latency, making them highly efficient for real-time deployment.
4.  **Real-World Generalization:** Models trained strictly on political data (PolitiFact) often experience an accuracy drop when benchmarked against broad, diverse topics (health, science, general news) via the Google Fact Check API. This highlights the necessity of domain-specific training or massive, multi-domain datasets for universal fact-checking.

## 🛠️ Technology Stack

*   **Core:** Python 3.12
*   **UI/Frontend:** Streamlit, Vanilla CSS (Glassmorphism UI)
*   **Data Processing:** Pandas, NumPy, BeautifulSoup4, Requests
*   **NLP:** SpaCy (`en_core_web_sm`), TextBlob
*   **Machine Learning:** Scikit-Learn, Imbalanced-Learn (SMOTE)
*   **Deep Learning:** TensorFlow / Keras
*   **Visualization:** Plotly, Matplotlib

## ⚙️ Installation & Deployment

This application is designed to be easily deployed on Streamlit Community Cloud.

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Add your Google API key:
   * **Locally:** Create `.streamlit/secrets.toml` and add `GOOGLE_API_KEY = "your_key_here"`
   * **Cloud:** Add the exact same string to the Streamlit Cloud Dashboard's Secret management.
4. Run locally: `streamlit run app.py`

