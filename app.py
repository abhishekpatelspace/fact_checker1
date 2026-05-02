import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import csv
from urllib.parse import urljoin
import time
import random
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# --- NEW DEP: Imbalanced-learn ---
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline 

# --- NLP & ML Imports ---
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from textblob import TextBlob
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from scipy import sparse
import io
import os

# --- Deep Learning Imports ---
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer as KerasTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Configuration ---
SCRAPED_DATA_PATH = 'politifact_data.csv'
N_SPLITS = 5

# Google Fact Check API rating mappings (for binary classification)
GOOGLE_TRUE_RATINGS = ["True", "Mostly True", "Accurate", "Correct"]
GOOGLE_FALSE_RATINGS = ["False", "Mostly False", "Pants on Fire", "Pants on Fire!", "Fake", "Incorrect", "Baseless", "Misleading"] 

# --- SpaCy Loading Function (Robust for Streamlit Cloud) ---
@st.cache_resource
def load_spacy_model():
    """Attempts to load SpaCy model, relying on the model being in requirements.txt."""
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError as e:
        st.error(f"SpaCy model 'en_core_web_sm' not found. Please ensure the direct GitHub link for the model is correctly listed in your 'requirements.txt' file.")
        st.code("""
        # Example of the line needed in requirements.txt:
        https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
        imbalanced-learn # Required for SMOTE
        """, language='text')
        # Try to download the model if not available
        try:
            import subprocess
            import sys
            st.info("Attempting to download spaCy model...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            nlp = spacy.load("en_core_web_sm")
            return nlp
        except:
            st.error("Failed to download spaCy model automatically. Please check your requirements.txt")
            raise e

# Load resources outside main app flow
try:
    NLP_MODEL = load_spacy_model()
except Exception:
    st.stop() 

stop_words = STOP_WORDS
pragmatic_words = ["must", "should", "might", "could", "will", "?", "!"]

# ============================
# DEMO DATA FUNCTION
# ============================

def get_demo_google_claims():
    """Provides demo fact-check data for testing without API key"""
    demo_claims = [
        {
            'claim_text': 'The earth is flat and NASA is hiding the truth from us.',
            'rating': 'False'
        },
        {
            'claim_text': 'Vaccines are completely safe and effective for 95% of the population.',
            'rating': 'Mostly True'
        },
        {
            'claim_text': 'The moon landing was filmed in a Hollywood studio in 1969.',
            'rating': 'False'
        },
        {
            'claim_text': 'Climate change is primarily caused by human activities and carbon emissions.',
            'rating': 'True'
        },
        {
            'claim_text': 'You can cure COVID-19 by drinking bleach and taking horse medication.',
            'rating': 'False'
        },
        {
            'claim_text': 'Regular exercise and balanced diet improve overall health and longevity.',
            'rating': 'True'
        },
        {
            'claim_text': '5G towers spread coronavirus and should be taken down immediately.',
            'rating': 'False'
        },
        {
            'claim_text': 'The Great Wall of China is visible from space with the naked eye.',
            'rating': 'Mostly False'
        },
        {
            'claim_text': 'Solar energy has become more affordable and efficient in the last decade.',
            'rating': 'True'
        },
        {
            'claim_text': 'Bill Gates is using vaccines to implant microchips in people.',
            'rating': 'Pants on Fire'
        },
        {
            'claim_text': 'Drinking 8 glasses of water daily is essential for human health.',
            'rating': 'Mostly True'
        },
        {
            'claim_text': 'Sharks don\'t get cancer and their cartilage can cure it in humans.',
            'rating': 'False'
        },
        {
            'claim_text': 'Electric vehicles produce zero emissions and are completely eco-friendly.',
            'rating': 'Mostly True'
        },
        {
            'claim_text': 'Humans only use 10% of their brain capacity.',
            'rating': 'False'
        },
        {
            'claim_text': 'Antibiotics are effective against viral infections like flu and colds.',
            'rating': 'False'
        }
    ]
    return demo_claims

# ============================
# GOOGLE FACT CHECK API INTEGRATION
# ============================

def fetch_google_claims(api_key, num_claims=100):
    """
    Fetches claims from Google Fact Check API with pagination handling.
    Uses multiple broad query terms to fetch diverse fact-checked claims.
    """
    base_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    collected_claims = []
    placeholder = st.empty()
    
    # The API requires a 'query' parameter; use broad topics to get diverse claims
    search_queries = [
        "politics", "health", "economy", "climate", "election",
        "vaccine", "immigration", "education", "crime", "tax",
        "government", "president", "congress", "covid", "energy"
    ]

    try:
        for query_term in search_queries:
            if len(collected_claims) >= num_claims:
                break
            
            page_token = None
            
            while len(collected_claims) < num_claims:
                # Build request parameters
                params = {
                    'key': api_key,
                    'query': query_term,
                    'languageCode': 'en',
                    'pageSize': min(100, num_claims - len(collected_claims))
                }

                if page_token:
                    params['pageToken'] = page_token

            # Update progress
                # Update progress
                placeholder.text(f"Fetching Google claims... {len(collected_claims)} collected so far (query: '{query_term}')")

                # Make API request
                response = requests.get(base_url, params=params, timeout=15)

                # Check for HTTP errors
                if response.status_code == 401:
                    st.error("Invalid API key. Please check your GOOGLE_API_KEY in .streamlit/secrets.toml")
                    return []
                elif response.status_code == 403:
                    st.error("API access forbidden. Ensure 'Fact Check Tools API' is enabled in Google Cloud Console.")
                    return []
                elif response.status_code == 429:
                    st.error("API rate limit exceeded. Please try again later with fewer claims.")
                    return []

                response.raise_for_status()
                data = response.json()

                # Check if response has claims
                if 'claims' not in data or not data['claims']:
                    break  # No more results for this query, try next one

                # Process each claim
                for claim_obj in data['claims']:
                    if len(collected_claims) >= num_claims:
                        break

                    # Extract claim text
                    claim_text = claim_obj.get('text', '')

                    # Extract rating from first claimReview
                    claim_reviews = claim_obj.get('claimReview', [])
                    if not claim_reviews or len(claim_reviews) == 0:
                        continue  # Skip claims without reviews

                    textual_rating = claim_reviews[0].get('textualRating', '')

                    # Skip if missing required fields
                    if not claim_text or not textual_rating:
                        continue

                    collected_claims.append({
                        'claim_text': claim_text,
                        'rating': textual_rating
                    })

                # Check for next page
                page_token = data.get('nextPageToken')
                if not page_token:
                    break  # No more pages for this query, try next one

        placeholder.success(f"Successfully fetched {len(collected_claims)} claims from Google Fact Check API")
        return collected_claims

    except requests.exceptions.RequestException as e:
        placeholder.error(f"Network error while fetching Google claims: {e}")
        return collected_claims if collected_claims else []
    except Exception as e:
        placeholder.error(f"Error processing Google API response: {e}")
        return collected_claims if collected_claims else []


def process_and_map_google_claims(api_results):
    """
    Converts Google's granular ratings into binary format (1=True, 0=False) and creates DataFrame.
    Discards ambiguous ratings like 'Half True', 'Mixed', etc.
    """
    if not api_results:
        return pd.DataFrame(columns=['claim_text', 'ground_truth'])

    processed_claims = []
    true_count = 0
    false_count = 0
    discarded_count = 0

    for claim_data in api_results:
        claim_text = claim_data.get('claim_text', '').strip()
        rating = claim_data.get('rating', '').strip()

        # Data quality checks
        if not claim_text or len(claim_text) < 10:
            discarded_count += 1
            continue

        if not rating:
            discarded_count += 1
            continue

        # Normalize rating for comparison (remove punctuation, lowercase)
        rating_normalized = rating.lower().strip().rstrip('!').rstrip('?')

        # Map to binary
        is_true = any(rating_normalized == r.lower() for r in GOOGLE_TRUE_RATINGS)
        is_false = any(rating_normalized == r.lower() for r in GOOGLE_FALSE_RATINGS)

        if is_true:
            processed_claims.append({
                'claim_text': claim_text,
                'ground_truth': 1
            })
            true_count += 1
        elif is_false:
            processed_claims.append({
                'claim_text': claim_text,
                'ground_truth': 0
            })
            false_count += 1
        else:
            # Ambiguous rating - discard
            discarded_count += 1

    # Create DataFrame
    google_df = pd.DataFrame(processed_claims)

    if not google_df.empty:
        # Remove duplicates (keep first occurrence)
        google_df = google_df.drop_duplicates(subset=['claim_text'], keep='first')

    # Display statistics
    total_processed = len(api_results)
    st.info(f"Processed {total_processed} claims: {true_count} True, {false_count} False, {discarded_count} ambiguous (discarded)")

    # Warn if only one class
    if not google_df.empty and len(google_df['ground_truth'].unique()) < 2:
        st.warning("Only one class found in processed claims. Results may not be meaningful.")

    return google_df


def run_google_benchmark(google_df, trained_models, vectorizer, selected_phase, rnn_tokenizer=None):
    """
    Tests trained models on Google claims and calculates performance metrics.
    """
    if google_df.empty:
        st.error("No Google claims available for benchmarking.")
        return pd.DataFrame()

    # Extract claim texts and ground truth labels
    X_raw = google_df['claim_text']
    y_true = google_df['ground_truth'].values

    # Apply same feature extraction as training
    try:
        if selected_phase == "Lexical & Morphological":
            X_processed = X_raw.apply(lexical_features)
            if vectorizer is None:
                st.error("Vectorizer not found for Lexical phase. Please retrain models.")
                return pd.DataFrame()
            X_features = vectorizer.transform(X_processed)

        elif selected_phase == "Syntactic":
            X_processed = X_raw.apply(syntactic_features)
            if vectorizer is None:
                st.error("Vectorizer not found for Syntactic phase. Please retrain models.")
                return pd.DataFrame()
            X_features = vectorizer.transform(X_processed)

        elif selected_phase == "Discourse":
            X_processed = X_raw.apply(discourse_features)
            if vectorizer is None:
                st.error("Vectorizer not found for Discourse phase. Please retrain models.")
                return pd.DataFrame()
            X_features = vectorizer.transform(X_processed)

        elif selected_phase == "Semantic":
            # Dense features - no vectorizer needed
            X_features = pd.DataFrame(X_raw.apply(semantic_features).tolist(), columns=["polarity", "subjectivity"]).values

        elif selected_phase == "Pragmatic":
            # Dense features - no vectorizer needed
            X_features = pd.DataFrame(X_raw.apply(pragmatic_features).tolist(), columns=pragmatic_words).values

        else:
            st.error(f"Unknown feature phase: {selected_phase}")
            return pd.DataFrame()

    except Exception as e:
        st.error(f"Feature extraction failed for Google claims: {e}")
        return pd.DataFrame()

    # Test each trained model
    results_list = []

    for model_name, model in trained_models.items():
        if model is None:
            results_list.append({
                'Model': model_name, 'Accuracy': 0, 'F1-Score': 0,
                'Precision': 0, 'Recall': 0, 'Inference Latency (ms)': 9999
            })
            continue

        try:
            # --- ANN prediction ---
            if model_name == "ANN":
                X_ann_bench = X_features
                if sparse.issparse(X_ann_bench):
                    X_ann_bench = X_ann_bench.toarray()
                elif isinstance(X_ann_bench, pd.DataFrame):
                    X_ann_bench = X_ann_bench.values
                
                start_inference = time.time()
                y_pred_prob = model.predict(X_ann_bench, verbose=0).flatten()
                y_pred = (y_pred_prob >= 0.5).astype(int)
                inference_time = (time.time() - start_inference) * 1000

            # --- RNN prediction ---
            elif model_name == "RNN (LSTM)":
                if rnn_tokenizer is None:
                    st.warning("RNN tokenizer not available for benchmark.")
                    results_list.append({
                        'Model': model_name, 'Accuracy': 0, 'F1-Score': 0,
                        'Precision': 0, 'Recall': 0, 'Inference Latency (ms)': 9999
                    })
                    continue
                X_rnn_bench, _ = prepare_rnn_data(X_raw.tolist(), tokenizer=rnn_tokenizer, fit=False)
                
                start_inference = time.time()
                y_pred_prob = model.predict(X_rnn_bench, verbose=0).flatten()
                y_pred = (y_pred_prob >= 0.5).astype(int)
                inference_time = (time.time() - start_inference) * 1000

            # --- Naive Bayes ---
            elif model_name == "Naive Bayes":
                X_features_model = np.abs(X_features).astype(float)
                start_inference = time.time()
                y_pred = model.predict(X_features_model)
                inference_time = (time.time() - start_inference) * 1000

            # --- Other ML models ---
            else:
                start_inference = time.time()
                y_pred = model.predict(X_features)
                inference_time = (time.time() - start_inference) * 1000

            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred) * 100
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)

            results_list.append({
                'Model': model_name,
                'Accuracy': accuracy,
                'F1-Score': f1,
                'Precision': precision,
                'Recall': recall,
                'Inference Latency (ms)': round(inference_time, 2)
            })

        except Exception as e:
            st.error(f"Prediction failed for {model_name}: {e}")
            results_list.append({
                'Model': model_name,
                'Accuracy': 0,
                'F1-Score': 0,
                'Precision': 0,
                'Recall': 0,
                'Inference Latency (ms)': 9999
            })

    return pd.DataFrame(results_list)

# ============================
# 1. WEB SCRAPING FUNCTION
# ============================

def scrape_data_by_date_range(start_date: pd.Timestamp, end_date: pd.Timestamp):
    base_url = "https://www.politifact.com/factchecks/list/"
    current_url = base_url
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["author", "statement", "source", "date", "label"])
    scraped_rows_count = 0
    page_count = 0
    st.caption(f"Starting scrape from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    placeholder = st.empty()

    while current_url and page_count < 100: 
        page_count += 1
        placeholder.text(f"Fetching page {page_count}... Scraped {scraped_rows_count} claims so far.")

        try:
            # Retry up to 3 times with increasing timeout
            response = None
            for attempt in range(3):
                try:
                    placeholder.text(f"Fetching page {page_count}... (attempt {attempt+1}/3) Scraped {scraped_rows_count} claims so far.")
                    response = requests.get(current_url, timeout=60, headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                    })
                    response.raise_for_status()
                    break
                except requests.exceptions.RequestException as retry_e:
                    if attempt == 2:
                        raise retry_e
                    time.sleep(2 * (attempt + 1))
            soup = BeautifulSoup(response.text, "html.parser")
        except requests.exceptions.RequestException as e:
            placeholder.error(f"Network Error after 3 attempts: {e}. Stopping scrape.")
            break

        rows_to_add = []

        for card in soup.find_all("li", class_="o-listicle__item"):
            date_div = card.find("div", class_="m-statement__desc")
            date_text = date_div.get_text(strip=True) if date_div else None
            claim_date = None
            
            if date_text:
                match = re.search(r"stated on ([A-Za-z]+\s+\d{1,2},\s+\d{4})", date_text)
                if match:
                    try:
                        claim_date = pd.to_datetime(match.group(1), format='%B %d, %Y')
                    except ValueError:
                        continue
            
            if claim_date:
                if start_date <= claim_date <= end_date:
                    statement_block = card.find("div", class_="m-statement__quote")
                    statement = statement_block.find("a", href=True).get_text(strip=True) if statement_block and statement_block.find("a", href=True) else None
                    source_a = card.find("a", class_="m-statement__name")
                    source = source_a.get_text(strip=True) if source_a else None
                    footer = card.find("footer", class_="m-statement__footer")
                    author = None
                    if footer:
                        author_match = re.search(r"By\s+([^•]+)", footer.get_text(strip=True))
                        if author_match:
                            author = author_match.group(1).strip()
                            
                    label_img = card.find("img", alt=True)
                    label = label_img['alt'].replace('-', ' ').title() if label_img and 'alt' in label_img.attrs else None

                    rows_to_add.append([author, statement, source, claim_date.strftime('%Y-%m-%d'), label])

                elif claim_date < start_date:
                    placeholder.warning(f"Encountered claim older than start date ({start_date.strftime('%Y-%m-%d')}). Stopping scrape.")
                    current_url = None
                    break 

        if current_url is None:
            break

        writer.writerows(rows_to_add)
        scraped_rows_count += len(rows_to_add)

        next_link = soup.find("a", class_="c-button c-button--hollow", string=re.compile(r"Next", re.I))
        if next_link and 'href' in next_link.attrs:
            next_href = next_link['href'].rstrip('&').rstrip('?')
            current_url = urljoin(base_url, next_href)
        else:
            placeholder.success("No more pages found or last page reached.")
            current_url = None

    placeholder.success(f"Scraping finished! Total claims processed: {scraped_rows_count}")
    
    output.seek(0)
    df = pd.read_csv(output, header=0, keep_default_na=False)
    df = df.dropna(subset=['statement', 'label'])
    
    df.to_csv(SCRAPED_DATA_PATH, index=False)
    return df

# ============================
# 2. FEATURE EXTRACTION (SPA/TEXTBLOB)
# ============================

def lexical_features(text):
    doc = NLP_MODEL(text.lower())
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and token.is_alpha]
    return " ".join(tokens)

def syntactic_features(text):
    doc = NLP_MODEL(text)
    pos_tags = " ".join([token.pos_ for token in doc])
    return pos_tags

def semantic_features(text):
    blob = TextBlob(text)
    return [blob.sentiment.polarity, blob.sentiment.subjectivity]

def discourse_features(text):
    doc = NLP_MODEL(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return f"{len(sentences)} {' '.join([s.split()[0].lower() for s in sentences if len(s.split()) > 0])}"

def pragmatic_features(text):
    text = text.lower()
    return [text.count(w) for w in pragmatic_words]

# ============================
# 3. MODEL TRAINING AND EVALUATION (K-FOLD & SMOTE)
# ============================

def get_classifier(name):
    """Initializes a classifier instance with hyperparameter tuning for imbalance."""
    if name == "Naive Bayes":
        return MultinomialNB()
    elif name == "Decision Tree":
        return DecisionTreeClassifier(random_state=42, class_weight='balanced') 
    elif name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, solver='liblinear', random_state=42, class_weight='balanced')
    elif name == "SVM":
        return SVC(kernel='linear', C=0.5, random_state=42, class_weight='balanced')
    return None

def apply_feature_extraction(X, phase, vectorizer=None):
    """Applies the chosen feature extraction technique and optimization (e.g., N-Grams)."""
    if phase == "Lexical & Morphological":
        X_processed = X.apply(lexical_features)
        vectorizer = vectorizer if vectorizer else CountVectorizer(binary=True, ngram_range=(1,2))
        X_features = vectorizer.fit_transform(X_processed)
        return X_features, vectorizer
    
    elif phase == "Syntactic":
        X_processed = X.apply(syntactic_features)
        vectorizer = vectorizer if vectorizer else TfidfVectorizer(max_features=5000)
        X_features = vectorizer.fit_transform(X_processed)
        return X_features, vectorizer

    elif phase == "Semantic":
        X_features = pd.DataFrame(X.apply(semantic_features).tolist(), columns=["polarity", "subjectivity"])
        return X_features, None

    elif phase == "Discourse":
        X_processed = X.apply(discourse_features)
        vectorizer = vectorizer if vectorizer else CountVectorizer(ngram_range=(1,2), max_features=5000)
        X_features = vectorizer.fit_transform(X_processed)
        return X_features, vectorizer

    elif phase == "Pragmatic":
        X_features = pd.DataFrame(X.apply(pragmatic_features).tolist(), columns=pragmatic_words)
        return X_features, None
    
    return None, None

# ============================
# 3b. DEEP LEARNING MODEL BUILDERS
# ============================

# RNN config constants
RNN_MAX_WORDS = 10000
RNN_MAX_LEN = 100

def build_ann_model(input_dim):
    """Builds a feedforward Artificial Neural Network for binary classification."""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_rnn_model(vocab_size=RNN_MAX_WORDS, max_len=RNN_MAX_LEN):
    """Builds a Bidirectional LSTM network for binary text classification."""
    model = Sequential([
        Embedding(vocab_size, 64, input_length=max_len),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def prepare_rnn_data(X_texts, tokenizer=None, fit=True):
    """Tokenizes and pads text data for RNN input."""
    if tokenizer is None:
        tokenizer = KerasTokenizer(num_words=RNN_MAX_WORDS, oov_token='<OOV>')
    if fit:
        tokenizer.fit_on_texts(X_texts)
    sequences = tokenizer.texts_to_sequences(X_texts)
    padded = pad_sequences(sequences, maxlen=RNN_MAX_LEN, padding='post', truncating='post')
    return padded, tokenizer


def evaluate_models(df: pd.DataFrame, selected_phase: str):
    """Trains and evaluates models using Stratified K-Fold Cross-Validation and SMOTE."""
    
    # 1. FEATURE ENGINEERING: BINARY TARGET MAPPING
    
    # Define mapping groups
    REAL_LABELS = ["True", "No Flip", "Mostly True", "Half Flip", "Half True"]
    FAKE_LABELS = ["False", "Barely True", "Pants On Fire", "Full Flop"]
    
    # Create the new binary target column
    def create_binary_target(label):
        if label in REAL_LABELS:
            return 1 # Real/True
        elif label in FAKE_LABELS:
            return 0 # Fake/False
        else:
            return np.nan # Mark unmappable/error labels

    df['target_label'] = df['label'].apply(create_binary_target)
    
    # 2. DATA CLEANING AND FILTERING
    
    # Drop rows where mapping failed
    df = df.dropna(subset=['target_label'])
    
    # Remove rows with short statements (noise/lack of context)
    df = df[df['statement'].astype(str).str.len() > 10]
    
    X_raw = df['statement'].astype(str)
    y_raw = df['target_label'].astype(int) # Target is now explicitly 0 or 1
    
    if len(np.unique(y_raw)) < 2:
        st.error("After binary mapping, only one class remains (all Real or all Fake). Cannot train classifier.")
        return pd.DataFrame() 
    
    # 3. Feature Extraction (Apply to all data once per phase)
    X_features_full, vectorizer = apply_feature_extraction(X_raw, selected_phase)
    
    if X_features_full is None:
        st.error("Feature extraction failed.")
        return pd.DataFrame()
        
    # Prepare data for K-Fold
    if isinstance(X_features_full, pd.DataFrame):
        X_features_full = X_features_full.values
    
    y = y_raw.values
    
    # 4. K-Fold Setup
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    models_to_run = {
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear', random_state=42, class_weight='balanced'),
        "SVM": SVC(kernel='linear', C=0.5, random_state=42, class_weight='balanced')
    }

    model_metrics = {name: [] for name in models_to_run.keys()}
    X_raw_list = X_raw.tolist()

    for name, model in models_to_run.items():
        st.caption(f"Training {name} with {N_SPLITS}-Fold CV & SMOTE...")
        
        fold_metrics = {
            'accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'train_time': [], 'inference_time': []
        }
        
        for fold, (train_index, test_index) in enumerate(skf.split(X_features_full, y)):
            
            # 4a. Get data indices for this fold
            X_train_raw = pd.Series([X_raw_list[i] for i in train_index])
            X_test_raw = pd.Series([X_raw_list[i] for i in test_index])
            y_train = y[train_index]
            y_test = y[test_index]
            
            # 4b. Transform the features using the fitted vectorizer (if applicable)
            if vectorizer is not None:
                # Need to run the phase's preprocessing (lexical_features or syntactic_features) on the raw text first
                X_train = vectorizer.transform(X_train_raw.apply(lexical_features if 'Lexical' in selected_phase else syntactic_features))
                X_test = vectorizer.transform(X_test_raw.apply(lexical_features if 'Lexical' in selected_phase else syntactic_features))
            else:
                # Dense feature sets (Semantic/Pragmatic)
                X_train, _ = apply_feature_extraction(X_train_raw, selected_phase)
                X_test, _ = apply_feature_extraction(X_test_raw, selected_phase)
            
            
            start_time = time.time()
            try:
                # --- SMOTE PIPELINE & Naive Bayes Fix ---
                if name == "Naive Bayes":
                    # FIX: Use np.abs on sparse matrix to get positive counts, then convert to int/float as needed.
                    X_train_final = np.abs(X_train).astype(float) 
                    clf = model
                    model.fit(X_train_final, y_train)
                else:
                    # Apply SMOTE to training data for other models
                    smote_pipeline = ImbPipeline([
                        ('sampler', SMOTE(random_state=42, k_neighbors=3)),
                        ('classifier', model)
                    ])
                    smote_pipeline.fit(X_train, y_train)
                    clf = smote_pipeline
                
                train_time = time.time() - start_time
                
                start_inference = time.time()
                y_pred = clf.predict(X_test)
                inference_time = (time.time() - start_inference) * 1000 
                
                # Metrics
                fold_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
                fold_metrics['f1'].append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
                fold_metrics['precision'].append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
                fold_metrics['recall'].append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
                fold_metrics['train_time'].append(train_time)
                fold_metrics['inference_time'].append(inference_time)

            except Exception as e:
                st.warning(f"Fold {fold+1} failed for {name}: {e}")
                for key in fold_metrics: fold_metrics[key].append(0)
                continue

        # Calculate means across all folds
        if fold_metrics['accuracy']:
            model_metrics[name] = {
                "Model": name,
                "Accuracy": np.mean(fold_metrics['accuracy']) * 100,
                "F1-Score": np.mean(fold_metrics['f1']),
                "Precision": np.mean(fold_metrics['precision']),
                "Recall": np.mean(fold_metrics['recall']),
                "Training Time (s)": round(np.mean(fold_metrics['train_time']), 2),
                "Inference Latency (ms)": round(np.mean(fold_metrics['inference_time']), 2),
            }
        else:
             st.error(f"{name} failed across all folds.")
             model_metrics[name] = {
                "Model": name, "Accuracy": 0, "F1-Score": 0, "Precision": 0, "Recall": 0,
                "Training Time (s)": 0, "Inference Latency (ms)": 9999,
            }

    # 5. DEEP LEARNING MODELS — ANN & RNN with K-Fold CV
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=0)
    
    # --- ANN K-Fold ---
    st.caption(f"Training ANN with {N_SPLITS}-Fold CV...")
    ann_fold_metrics = {
        'accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'train_time': [], 'inference_time': []
    }
    
    for fold, (train_index, test_index) in enumerate(skf.split(X_features_full, y)):
        try:
            X_train_ann = X_features_full[train_index]
            X_test_ann = X_features_full[test_index]
            y_train_ann = y[train_index]
            y_test_ann = y[test_index]
            
            # Convert sparse to dense if needed
            if sparse.issparse(X_train_ann):
                X_train_ann = X_train_ann.toarray()
                X_test_ann = X_test_ann.toarray()
            elif isinstance(X_train_ann, pd.DataFrame):
                X_train_ann = X_train_ann.values if hasattr(X_train_ann, 'values') else np.array(X_train_ann)
                X_test_ann = X_test_ann.values if hasattr(X_test_ann, 'values') else np.array(X_test_ann)
            
            # Apply SMOTE to training data
            try:
                smote = SMOTE(random_state=42, k_neighbors=min(3, min(np.bincount(y_train_ann)) - 1))
                X_train_ann_sm, y_train_ann_sm = smote.fit_resample(X_train_ann, y_train_ann)
            except:
                X_train_ann_sm, y_train_ann_sm = X_train_ann, y_train_ann
            
            # Compute class weights
            class_counts = np.bincount(y_train_ann_sm.astype(int))
            total = len(y_train_ann_sm)
            cw = {0: total / (2.0 * class_counts[0]) if class_counts[0] > 0 else 1.0,
                  1: total / (2.0 * class_counts[1]) if len(class_counts) > 1 and class_counts[1] > 0 else 1.0}
            
            ann_model = build_ann_model(X_train_ann_sm.shape[1])
            
            start_time = time.time()
            ann_model.fit(X_train_ann_sm, y_train_ann_sm, epochs=20, batch_size=32,
                          validation_split=0.15, callbacks=[early_stop],
                          class_weight=cw, verbose=0)
            train_time = time.time() - start_time
            
            start_inference = time.time()
            y_pred_prob = ann_model.predict(X_test_ann, verbose=0).flatten()
            y_pred_ann = (y_pred_prob >= 0.5).astype(int)
            inference_time = (time.time() - start_inference) * 1000
            
            ann_fold_metrics['accuracy'].append(accuracy_score(y_test_ann, y_pred_ann))
            ann_fold_metrics['f1'].append(f1_score(y_test_ann, y_pred_ann, average='weighted', zero_division=0))
            ann_fold_metrics['precision'].append(precision_score(y_test_ann, y_pred_ann, average='weighted', zero_division=0))
            ann_fold_metrics['recall'].append(recall_score(y_test_ann, y_pred_ann, average='weighted', zero_division=0))
            ann_fold_metrics['train_time'].append(train_time)
            ann_fold_metrics['inference_time'].append(inference_time)
        except Exception as e:
            st.warning(f"ANN Fold {fold+1} failed: {e}")
            for key in ann_fold_metrics: ann_fold_metrics[key].append(0)
    
    if ann_fold_metrics['accuracy']:
        model_metrics["ANN"] = {
            "Model": "ANN",
            "Accuracy": np.mean(ann_fold_metrics['accuracy']) * 100,
            "F1-Score": np.mean(ann_fold_metrics['f1']),
            "Precision": np.mean(ann_fold_metrics['precision']),
            "Recall": np.mean(ann_fold_metrics['recall']),
            "Training Time (s)": round(np.mean(ann_fold_metrics['train_time']), 2),
            "Inference Latency (ms)": round(np.mean(ann_fold_metrics['inference_time']), 2),
        }
    
    # --- RNN (LSTM) K-Fold ---
    st.caption(f"Training RNN (LSTM) with {N_SPLITS}-Fold CV...")
    rnn_fold_metrics = {
        'accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'train_time': [], 'inference_time': []
    }
    rnn_tokenizer = None
    
    for fold, (train_index, test_index) in enumerate(skf.split(X_features_full, y)):
        try:
            X_train_text_rnn = [X_raw_list[i] for i in train_index]
            X_test_text_rnn = [X_raw_list[i] for i in test_index]
            y_train_rnn = y[train_index]
            y_test_rnn = y[test_index]
            
            # Tokenize and pad
            X_train_pad, rnn_tokenizer = prepare_rnn_data(X_train_text_rnn, tokenizer=None, fit=True)
            X_test_pad, _ = prepare_rnn_data(X_test_text_rnn, tokenizer=rnn_tokenizer, fit=False)
            
            # Compute class weights for RNN (no SMOTE for sequential data)
            class_counts_rnn = np.bincount(y_train_rnn.astype(int))
            total_rnn = len(y_train_rnn)
            cw_rnn = {0: total_rnn / (2.0 * class_counts_rnn[0]) if class_counts_rnn[0] > 0 else 1.0,
                      1: total_rnn / (2.0 * class_counts_rnn[1]) if len(class_counts_rnn) > 1 and class_counts_rnn[1] > 0 else 1.0}
            
            rnn_model = build_rnn_model()
            
            start_time = time.time()
            rnn_model.fit(X_train_pad, y_train_rnn, epochs=15, batch_size=32,
                          validation_split=0.15, callbacks=[early_stop],
                          class_weight=cw_rnn, verbose=0)
            train_time = time.time() - start_time
            
            start_inference = time.time()
            y_pred_prob_rnn = rnn_model.predict(X_test_pad, verbose=0).flatten()
            y_pred_rnn = (y_pred_prob_rnn >= 0.5).astype(int)
            inference_time = (time.time() - start_inference) * 1000
            
            rnn_fold_metrics['accuracy'].append(accuracy_score(y_test_rnn, y_pred_rnn))
            rnn_fold_metrics['f1'].append(f1_score(y_test_rnn, y_pred_rnn, average='weighted', zero_division=0))
            rnn_fold_metrics['precision'].append(precision_score(y_test_rnn, y_pred_rnn, average='weighted', zero_division=0))
            rnn_fold_metrics['recall'].append(recall_score(y_test_rnn, y_pred_rnn, average='weighted', zero_division=0))
            rnn_fold_metrics['train_time'].append(train_time)
            rnn_fold_metrics['inference_time'].append(inference_time)
        except Exception as e:
            st.warning(f"RNN Fold {fold+1} failed: {e}")
            for key in rnn_fold_metrics: rnn_fold_metrics[key].append(0)
    
    if rnn_fold_metrics['accuracy']:
        model_metrics["RNN (LSTM)"] = {
            "Model": "RNN (LSTM)",
            "Accuracy": np.mean(rnn_fold_metrics['accuracy']) * 100,
            "F1-Score": np.mean(rnn_fold_metrics['f1']),
            "Precision": np.mean(rnn_fold_metrics['precision']),
            "Recall": np.mean(rnn_fold_metrics['recall']),
            "Training Time (s)": round(np.mean(rnn_fold_metrics['train_time']), 2),
            "Inference Latency (ms)": round(np.mean(rnn_fold_metrics['inference_time']), 2),
        }

    # 6. TRAIN FINAL MODELS ON FULL DATASET (for Google benchmark)
    st.caption("Training final models on complete dataset for benchmarking...")
    trained_models_final = {}

    # --- Final ML Models ---
    for name in models_to_run.keys():
        try:
            # Get fresh model instance
            final_model = get_classifier(name)

            # Prepare features for final training
            if vectorizer is not None:
                # Transform using the fitted vectorizer
                if 'Lexical' in selected_phase:
                    X_final_processed = X_raw.apply(lexical_features)
                elif 'Syntactic' in selected_phase:
                    X_final_processed = X_raw.apply(syntactic_features)
                elif 'Discourse' in selected_phase:
                    X_final_processed = X_raw.apply(discourse_features)
                else:
                    X_final_processed = X_raw
                X_final = vectorizer.transform(X_final_processed)
            else:
                # Dense features (Semantic/Pragmatic)
                X_final = X_features_full

            # Apply SMOTE and train (same pattern as K-Fold)
            if name == "Naive Bayes":
                X_final_train = np.abs(X_final).astype(float)
                final_model.fit(X_final_train, y)
                trained_models_final[name] = final_model
            else:
                # Apply SMOTE to full dataset for other models
                smote_pipeline_final = ImbPipeline([
                    ('sampler', SMOTE(random_state=42, k_neighbors=3)),
                    ('classifier', final_model)
                ])
                smote_pipeline_final.fit(X_final, y)
                trained_models_final[name] = smote_pipeline_final

        except Exception as e:
            st.warning(f"Failed to train final {name} model: {e}")
            trained_models_final[name] = None

    # --- Final ANN Model ---
    try:
        st.caption("Training final ANN model...")
        X_ann_full = X_features_full
        if sparse.issparse(X_ann_full):
            X_ann_full = X_ann_full.toarray()
        elif isinstance(X_ann_full, pd.DataFrame):
            X_ann_full = X_ann_full.values
        
        try:
            smote_ann = SMOTE(random_state=42, k_neighbors=min(3, min(np.bincount(y)) - 1))
            X_ann_sm, y_ann_sm = smote_ann.fit_resample(X_ann_full, y)
        except:
            X_ann_sm, y_ann_sm = X_ann_full, y
        
        final_ann = build_ann_model(X_ann_sm.shape[1])
        final_ann.fit(X_ann_sm, y_ann_sm, epochs=20, batch_size=32,
                      validation_split=0.1, callbacks=[early_stop], verbose=0)
        trained_models_final["ANN"] = final_ann
    except Exception as e:
        st.warning(f"Failed to train final ANN model: {e}")
        trained_models_final["ANN"] = None

    # --- Final RNN Model ---
    try:
        st.caption("Training final RNN (LSTM) model...")
        X_rnn_full, rnn_tokenizer_final = prepare_rnn_data(X_raw.tolist(), tokenizer=None, fit=True)
        
        class_counts_final = np.bincount(y.astype(int))
        total_final = len(y)
        cw_final = {0: total_final / (2.0 * class_counts_final[0]) if class_counts_final[0] > 0 else 1.0,
                    1: total_final / (2.0 * class_counts_final[1]) if len(class_counts_final) > 1 and class_counts_final[1] > 0 else 1.0}
        
        final_rnn = build_rnn_model()
        final_rnn.fit(X_rnn_full, y, epochs=15, batch_size=32,
                      validation_split=0.1, callbacks=[early_stop],
                      class_weight=cw_final, verbose=0)
        trained_models_final["RNN (LSTM)"] = final_rnn
    except Exception as e:
        st.warning(f"Failed to train final RNN model: {e}")
        trained_models_final["RNN (LSTM)"] = None

    results_list = list(model_metrics.values())
    return pd.DataFrame(results_list), trained_models_final, vectorizer, rnn_tokenizer_final if 'rnn_tokenizer_final' in dir() else rnn_tokenizer

# ============================
# 4. HUMOR & CRITIQUE FUNCTIONS
# ============================

def get_phase_critique(best_phase: str) -> str:
    critiques = {
        "Lexical & Morphological": ["Ah, the Lexical phase. Proving that sometimes, all you need is raw vocabulary and minimal effort. It's the high-school dropout that won the Nobel Prize.", "Just words, nothing fancy. This phase decided to ditch the deep thought and focus on counting. Turns out, quantity has a quality all its own.", "The Lexical approach: when in doubt, just scream the words louder. It lacks elegance but gets the job done."],
        "Syntactic": ["Syntactic features won? So grammar actually matters! We must immediately inform Congress. This phase is the meticulous editor who corrects everyone's texts.", "The grammar police have prevailed. This model focused purely on structure, proving that sentence construction is more important than meaning... wait, is that how politics works?", "It passed the grammar check! This phase is the sensible adult in the room, refusing to process any nonsense until the parts of speech align."],
        "Semantic": ["The Semantic phase won by feeling its feelings. It's highly emotional, heavily relying on vibes and tone. Surprisingly effective, just like a good political ad.", "It turns out sentiment polarity is the secret sauce! This model just needed to know if the statement felt 'good' or 'bad.' Zero complex reasoning required.", "Semantic victory! The model simply asked, 'Are they being optimistic or negative?' and apparently that was enough to crush the competition."],
        "Discourse": ["Discourse features won! This phase is the over-analyzer, counting sentences and focusing on the rhythm of the argument. It knows the debate structure better than the content.", "The long-winded champion! This model cared about how the argument was *structured*—the thesis, the body, the conclusion. It's basically the high school debate team captain.", "Discourse is the winner! It successfully mapped the argument's flow, proving that presentation beats facts."],
        "Pragmatic": ["The Pragmatic phase won by focusing on keywords like 'must' and '?'. It just needed to know the speaker's intent. It's the Sherlock Holmes of NLP.", "It's all about intent! This model ignored the noise and hunted for specific linguistic tells. It's concise, ruthless, and apparently correct.", "Pragmatic features for the win! The model knows that if someone uses three exclamation marks, they're either lying or selling crypto. Either way, it's a clue."],
    }
    return random.choice(critiques.get(best_phase, ["The results are in, and the system is speechless. It seems we need to hire a better comedian."]))

def get_model_critique(best_model: str) -> str:
    critiques = {
        "Naive Bayes": ["Naive Bayes: It's fast, it's simple, and it assumes every feature is independent. The model is either brilliant or blissfully unaware, but hey, it works!", "The Simpleton Savant has won! Naive Bayes brings zero drama and just counts things. It's the least complicated tool in the box, which is often the best.", "NB pulled off a victory. It's the 'less-is-more' philosopher who manages to outperform all the complex math majors."],
        "Decision Tree": ["The Decision Tree won by asking a series of simple yes/no questions until it got tired. It's transparent, slightly judgmental, and surprisingly effective.", "The Hierarchical Champion! It built a beautiful, intricate set of if/then statements. It's the most organized person in the office, and the accuracy shows it.", "Decision Tree victory! It achieved success by splitting the data until it couldn't be split anymore. A classic strategy in science and divorce."],
        "Logistic Regression": ["Logistic Regression: The veteran politician of ML. It draws a clean, straight line to victory. Boring, reliable, and hard to beat.", "The Straight-Line Stunner. It uses simple math to predict complex reality. It's predictable, efficient, and definitely got tenure.", "LogReg prevails! The model's philosophy is: 'Probability is all you need.' It's the safest bet, and the accuracy score agrees."],
        "SVM": ["SVM: It found the biggest, widest gap between the truth and the lies, and parked its hyperplane right there. Aggressive but effective boundary enforcement.", "The Maximizing Margin Master! SVM doesn't just separate classes; it builds a fortress between them. It's the most dramatic and highly paid algorithm here.", "SVM crushed it! It's the model that believes in extreme boundaries. No fuzzy logic, just a hard, clean, dividing line."],
        "ANN": ["The Neural Network has spoken! It fired up 128 neurons, asked them to argue about politics, and somehow reached a consensus. Democracy works—even in hidden layers.", "ANN for the win! This feedforward overachiever has more layers than a political scandal. It learned patterns that even the data didn't know existed.", "The Artificial Neural Network proved that stacking layers of math on top of each other can solve anything. It's basically the corporate hierarchy of algorithms."],
        "RNN (LSTM)": ["The LSTM remembers EVERYTHING. Every word, every sequence, every lie. It's the elephant of deep learning, and it just trampled the competition.", "RNN victory! This model read each statement word by word, like a suspicious detective reading a ransom note. Its memory is long and its judgment is harsh.", "The LSTM won by actually reading the sentences in order—what a revolutionary concept! While other models just counted words, this one understood the story."],
    }
    return random.choice(critiques.get(best_model, ["This model broke the simulation, so we have nothing funny to say."]))


def generate_humorous_critique(df_results: pd.DataFrame, selected_phase: str) -> str:
    if df_results.empty:
        return "The system failed to train anything. We apologize; our ML models are currently on strike demanding better data and less existential dread."

    df_results['F1-Score'] = pd.to_numeric(df_results['F1-Score'], errors='coerce').fillna(0)
    best_model_row = df_results.loc[df_results['F1-Score'].idxmax()]
    best_model = best_model_row['Model']
    max_f1 = best_model_row['F1-Score']
    max_acc = best_model_row['Accuracy']
    
    phase_critique = get_phase_critique(selected_phase)
    model_critique = get_model_critique(best_model)
    
    headline = f"The Golden Snitch Award goes to the {best_model}!"
    
    summary = (
        f"**Accuracy Report Card:** {headline}\n\n"
        f"This absolute unit achieved a **{max_acc:.2f}% Accuracy** (and {max_f1:.2f} F1-Score) on the `{selected_phase}` feature set. "
        f"It beat its rivals, proving that when faced with political statements, the winning strategy was to rely on: **{selected_phase} features!**\n\n"
    )
    
    roast = (
        f"### The AI Roast (Certified by a Data Scientist):\n"
        f"**Phase Performance:** {phase_critique}\n\n"
        f"**Model Personality:** {model_critique}\n\n"
        f"*(Disclaimer: All models were equally confused by the 'Mostly True' label, which they collectively deemed an existential threat.)*"
    )
    
    return summary + roast

# ============================
# 5. STREAMLIT APP — PREMIUM UI
# ============================

def app():
    st.set_page_config(
        page_title='FactChecker — AI Fact-Checking Platform',
        page_icon='🔍',
        layout='wide',
        initial_sidebar_state='expanded'
    )

    # ─────────────────────────────────────────────
    # PREMIUM CSS THEME — Glassmorphism Dark Mode
    # ─────────────────────────────────────────────
    st.markdown("""
    <style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── CSS Variables ── */
    :root {
        --bg-deep:        #06080f;
        --bg-primary:     #0a0e1a;
        --bg-card:        rgba(255,255,255,0.04);
        --bg-card-hover:  rgba(255,255,255,0.07);
        --glass-border:   rgba(255,255,255,0.08);
        --glass-border-h: rgba(255,255,255,0.15);
        --accent-indigo:  #6366f1;
        --accent-violet:  #8b5cf6;
        --accent-cyan:    #22d3ee;
        --accent-emerald: #10b981;
        --accent-amber:   #f59e0b;
        --accent-rose:    #f43f5e;
        --text-primary:   #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted:     #475569;
        --shadow-lg:      0 20px 40px rgba(0,0,0,0.4);
        --shadow-glow:    0 0 30px rgba(99,102,241,0.15);
    }

    /* ── Global Reset ── */
    *:not([class*="material"]):not([data-testid="stIconMaterial"]):not(.material-icons):not(.material-symbols-rounded),
    *:not([class*="material"]):not([data-testid="stIconMaterial"]):not(.material-icons):not(.material-symbols-rounded)::before,
    *:not([class*="material"]):not([data-testid="stIconMaterial"]):not(.material-icons):not(.material-symbols-rounded)::after {
        font-family: 'Inter', sans-serif !important;
    }

    /* Restore Material Icons font */
    .material-symbols-rounded,
    .material-icons,
    [data-testid="stIconMaterial"],
    [class*="material-symbols"],
    span[class*="material"] {
        font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1021 0%, #0a0e1a 100%) !important;
        border-right: 1px solid var(--glass-border) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text-secondary) !important; }
    [data-testid="stSidebar"] .stRadio label { font-weight: 500 !important; }
    [data-testid="stSidebar"] .stRadio label:hover { color: var(--text-primary) !important; }
    [data-testid="stSidebar"] hr { border-color: var(--glass-border) !important; }

    /* ── Headers ── */
    h1 { color: var(--text-primary) !important; font-weight: 800 !important; letter-spacing: -0.03em; }
    h2 { color: var(--text-primary) !important; font-weight: 700 !important; letter-spacing: -0.02em; }
    h3 { color: var(--text-primary) !important; font-weight: 600 !important; }
    h4, h5, h6 { color: var(--text-secondary) !important; }

    /* ── Body text ── */
    p, li, span, div { color: var(--text-secondary) !important; }

    /* ── Hero Banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 30%, #4338ca 60%, #6366f1 100%);
        padding: 2.5rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow-glow), var(--shadow-lg);
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -30%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(139,92,246,0.2) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-banner::after {
        content: '';
        position: absolute;
        bottom: -40%;
        left: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(34,211,238,0.1) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-banner h1 {
        color: #ffffff !important;
        font-size: 2.4rem !important;
        margin-bottom: 0.3rem;
        position: relative;
        z-index: 1;
    }
    .hero-banner p {
        color: rgba(255,255,255,0.8) !important;
        font-size: 1.05rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
        margin: 0;
    }

    /* ── Glass Card ── */
    .glass-card {
        background: var(--bg-card);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--glass-border);
        border-radius: 14px;
        padding: 1.6rem;
        margin-bottom: 1rem;
        transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
    }
    .glass-card:hover {
        background: var(--bg-card-hover);
        border-color: var(--glass-border-h);
        transform: translateY(-2px);
        box-shadow: var(--shadow-glow);
    }
    .glass-card h3 { margin-top: 0; font-size: 1.15rem; }
    .glass-card p  { font-size: 0.92rem; line-height: 1.6; }

    /* ── KPI Metric Card ── */
    .kpi-card {
        background: var(--bg-card);
        backdrop-filter: blur(12px);
        border: 1px solid var(--glass-border);
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        text-align: center;
        transition: all 0.35s cubic-bezier(0.4,0,0.2,1);
        position: relative;
        overflow: hidden;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        border-radius: 14px 14px 0 0;
    }
    .kpi-card.indigo::before { background: linear-gradient(90deg, var(--accent-indigo), var(--accent-violet)); }
    .kpi-card.emerald::before { background: linear-gradient(90deg, #059669, var(--accent-emerald)); }
    .kpi-card.cyan::before { background: linear-gradient(90deg, #0891b2, var(--accent-cyan)); }
    .kpi-card.amber::before { background: linear-gradient(90deg, #d97706, var(--accent-amber)); }
    .kpi-card.rose::before { background: linear-gradient(90deg, #e11d48, var(--accent-rose)); }
    .kpi-card.violet::before { background: linear-gradient(90deg, #7c3aed, var(--accent-violet)); }

    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 28px rgba(0,0,0,0.35);
        border-color: var(--glass-border-h);
    }
    .kpi-card .kpi-icon { font-size: 1.6rem; margin-bottom: 0.3rem; }
    .kpi-card .kpi-value {
        font-size: 1.9rem;
        font-weight: 800;
        color: var(--text-primary) !important;
        letter-spacing: -0.03em;
        line-height: 1.2;
    }
    .kpi-card .kpi-label {
        font-size: 0.78rem;
        color: var(--text-muted) !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
        margin-top: 0.3rem;
    }

    /* ── Pipeline Stepper ── */
    .pipeline-stepper {
        display: flex;
        gap: 0;
        margin: 1.5rem 0;
    }
    .step-item {
        flex: 1;
        text-align: center;
        padding: 1rem 0.5rem;
        position: relative;
    }
    .step-item::after {
        content: '';
        position: absolute;
        top: 28px;
        right: -50%;
        width: 100%;
        height: 2px;
        background: var(--glass-border);
        z-index: 0;
    }
    .step-item:last-child::after { display: none; }
    .step-dot {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.95rem;
        font-weight: 700;
        position: relative;
        z-index: 1;
        margin-bottom: 0.5rem;
        transition: all 0.3s ease;
    }
    .step-dot.done {
        background: linear-gradient(135deg, var(--accent-emerald), #34d399);
        color: #fff !important;
        box-shadow: 0 0 12px rgba(16,185,129,0.4);
    }
    .step-dot.active {
        background: linear-gradient(135deg, var(--accent-indigo), var(--accent-violet));
        color: #fff !important;
        box-shadow: 0 0 14px rgba(99,102,241,0.5);
        animation: pulse-ring 2s ease-in-out infinite;
    }
    .step-dot.pending {
        background: rgba(255,255,255,0.06);
        border: 2px solid var(--glass-border);
        color: var(--text-muted) !important;
    }
    .step-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-muted) !important;
    }
    .step-item.completed .step-label { color: var(--accent-emerald) !important; }
    .step-item.is-active .step-label { color: var(--accent-indigo) !important; }

    /* ── Podium Cards ── */
    .podium-card {
        background: var(--bg-card);
        backdrop-filter: blur(12px);
        border: 1px solid var(--glass-border);
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .podium-card:hover { transform: translateY(-3px); box-shadow: var(--shadow-glow); }
    .podium-card.gold   { border-color: #fbbf24; box-shadow: 0 0 20px rgba(251,191,36,0.15); }
    .podium-card.silver { border-color: #94a3b8; }
    .podium-card.bronze { border-color: #d97706; }
    .podium-medal { font-size: 2.2rem; margin-bottom: 0.3rem; }
    .podium-name { font-size: 1rem; font-weight: 700; color: var(--text-primary) !important; }
    .podium-score { font-size: 1.6rem; font-weight: 800; color: var(--accent-indigo) !important; margin: 0.3rem 0; }
    .podium-sub { font-size: 0.78rem; color: var(--text-muted) !important; }

    /* ── Phase Card ── */
    .phase-card {
        background: var(--bg-card);
        backdrop-filter: blur(12px);
        border: 1px solid var(--glass-border);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
        transition: all 0.25s ease;
        cursor: default;
    }
    .phase-card:hover {
        background: var(--bg-card-hover);
        border-color: var(--accent-indigo);
    }
    .phase-icon { font-size: 1.5rem; }
    .phase-name { font-weight: 600; font-size: 0.9rem; color: var(--text-primary) !important; }
    .phase-desc { font-size: 0.78rem; color: var(--text-muted) !important; }

    /* ── Status Badge ── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 0.25rem 0.7rem;
        border-radius: 20px;
        letter-spacing: 0.03em;
    }
    .status-badge.ready { background: rgba(16,185,129,0.15); color: #34d399 !important; }
    .status-badge.pending { background: rgba(245,158,11,0.15); color: #fbbf24 !important; }
    .status-badge.error { background: rgba(244,63,94,0.12); color: #fb7185 !important; }

    /* ── Buttons ── */
    .stButton>button {
        background: linear-gradient(135deg, var(--accent-indigo) 0%, var(--accent-violet) 100%) !important;
        color: #ffffff !important;
        border: none !important;
        padding: 0.65rem 1.5rem !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.01em;
        transition: all 0.3s cubic-bezier(0.4,0,0.2,1) !important;
        box-shadow: 0 4px 14px rgba(99,102,241,0.3) !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 22px rgba(99,102,241,0.4) !important;
    }
    .stButton>button:active { transform: translateY(0) !important; }

    /* ── Selectbox / Inputs ── */
    .stSelectbox > div > div,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stDateInput > div > div > input {
        background-color: rgba(255,255,255,0.04) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
    }
    .stSelectbox label, .stSlider label, .stDateInput label, .stRadio label, .stCheckbox label {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
    }

    /* ── Dataframe ── */
    .stDataFrame { border-radius: 12px; overflow: hidden; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid var(--glass-border); }
    .stTabs [data-baseweb="tab"] {
        color: var(--text-muted) !important;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        border-radius: 8px 8px 0 0;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent-indigo) !important;
        border-bottom: 2px solid var(--accent-indigo);
        background: rgba(99,102,241,0.06);
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 10px !important;
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
    }

    /* ── Metrics ── */
    [data-testid="metric-container"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 12px !important;
        padding: 0.8rem 1rem !important;
    }
    [data-testid="metric-container"] label { color: var(--text-secondary) !important; font-weight: 500 !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: var(--text-primary) !important; font-weight: 700 !important; }

    /* ── Divider ── */
    hr { border-color: var(--glass-border) !important; margin: 1.5rem 0 !important; }

    /* ── Critique / Quote Card ── */
    .critique-card {
        background: linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(139,92,246,0.05) 100%);
        border: 1px solid rgba(99,102,241,0.2);
        border-left: 4px solid var(--accent-indigo);
        border-radius: 0 14px 14px 0;
        padding: 1.5rem 1.8rem;
        margin: 1rem 0;
    }
    .critique-card p, .critique-card li, .critique-card h3 { color: var(--text-secondary) !important; }
    .critique-card strong { color: var(--text-primary) !important; }

    /* ── Animations ── */
    @keyframes pulse-ring {
        0%   { box-shadow: 0 0 0 0 rgba(99,102,241,0.4); }
        70%  { box-shadow: 0 0 0 8px rgba(99,102,241,0); }
        100% { box-shadow: 0 0 0 0 rgba(99,102,241,0); }
    }
    @keyframes fade-in-up {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .animate-in {
        animation: fade-in-up 0.5s cubic-bezier(0.4,0,0.2,1) both;
    }
    .animate-in-d1 { animation-delay: 0.1s; }
    .animate-in-d2 { animation-delay: 0.2s; }
    .animate-in-d3 { animation-delay: 0.3s; }
    .animate-in-d4 { animation-delay: 0.4s; }

    /* ── Plotly Chart container ── */
    .js-plotly-plot .plotly .main-svg { border-radius: 12px; }

    </style>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # SESSION STATE
    # ─────────────────────────────────────────────
    defaults = {
        'scraped_df': pd.DataFrame(),
        'df_results': pd.DataFrame(),
        'trained_models': {},
        'trained_vectorizer': None,
        'rnn_tokenizer': None,
        'google_benchmark_results': pd.DataFrame(),
        'google_df': pd.DataFrame(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ─────────────────────────────────────────────
    # SIDEBAR
    # ─────────────────────────────────────────────
    with st.sidebar:
        # Brand
        st.markdown("""
        <div style='text-align:center; padding:1.2rem 0 1rem;'>
            <div style='font-size:2rem; margin-bottom:0.2rem;'>🔍</div>
            <div style='font-size:1.3rem; font-weight:800; letter-spacing:-0.02em;
                        background:linear-gradient(135deg,#6366f1,#a78bfa);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>FactChecker</div>
            <div style='font-size:0.72rem; color:#64748b !important; text-transform:uppercase;
                        letter-spacing:0.1em; font-weight:600; margin-top:0.2rem;'>AI Fact-Checking Platform</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Navigation
        page = st.radio(
            "NAVIGATE",
            ["🏠  Dashboard", "📥  Data Collection", "🧠  Model Training",
             "🎯  Benchmark Testing", "📊  Results & Analysis"],
            key='navigation',
            label_visibility='collapsed'
        )

        st.markdown("---")

        # Pipeline progress
        has_data = not st.session_state['scraped_df'].empty
        has_models = bool(st.session_state['trained_models'])
        has_bench = not st.session_state['google_benchmark_results'].empty
        has_results = not st.session_state['df_results'].empty

        st.markdown("##### ⚙️ Pipeline Status")

        def _badge(label, ok):
            cls = "ready" if ok else "pending"
            dot = "●" if ok else "○"
            return f'<span class="status-badge {cls}">{dot} {label}</span>'

        st.markdown(
            f"""<div style='display:flex; flex-direction:column; gap:0.45rem; margin:0.5rem 0 0.8rem;'>
            {_badge("Data Collected", has_data)}
            {_badge("Models Trained", has_models)}
            {_badge("Benchmark Run", has_bench)}
            {_badge("Results Ready", has_results)}
            </div>""",
            unsafe_allow_html=True
        )

        st.markdown("---")

        # Quick actions
        if st.button("🗑️  Clear All Data", key="sidebar_clear", use_container_width=True):
            st.session_state.clear()
            st.rerun()

        # Feature reference
        with st.expander("📚 Feature Reference"):
            st.markdown("""
            **Lexical & Morphological**
            Lemmatization · Stopwords · N-grams

            **Syntactic**
            POS tags · Grammar patterns

            **Semantic**
            Sentiment polarity · Subjectivity

            **Discourse**
            Sentence count · Discourse markers

            **Pragmatic**
            Modal verbs · Intent signals
            """)

    # ─────────────────────────────────────────────
    # PAGE ROUTING
    # ─────────────────────────────────────────────

    # ╔══════════════════════════════════════════╗
    # ║           DASHBOARD  PAGE                ║
    # ╚══════════════════════════════════════════╝
    if "Dashboard" in page:
        # Hero
        st.markdown("""
        <div class="hero-banner animate-in">
            <h1>🔍 FactChecker Dashboard</h1>
            <p>AI-powered misinformation detection with NLP feature engineering & multi-model evaluation</p>
        </div>
        """, unsafe_allow_html=True)

        # KPI Cards
        total_claims = len(st.session_state['scraped_df']) if has_data else 0
        models_count = len(st.session_state['trained_models']) if has_models else 0
        best_acc = f"{st.session_state['df_results']['Accuracy'].max():.1f}%" if has_results and not st.session_state['df_results'].empty else "—"
        bench_count = len(st.session_state['google_df']) if has_bench else 0

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="kpi-card indigo animate-in animate-in-d1">
                <div class="kpi-icon">📄</div>
                <div class="kpi-value">{total_claims}</div>
                <div class="kpi-label">Claims Collected</div>
            </div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="kpi-card emerald animate-in animate-in-d2">
                <div class="kpi-icon">🤖</div>
                <div class="kpi-value">{models_count}</div>
                <div class="kpi-label">Models Trained</div>
            </div>""", unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="kpi-card cyan animate-in animate-in-d3">
                <div class="kpi-icon">🎯</div>
                <div class="kpi-value">{best_acc}</div>
                <div class="kpi-label">Best Accuracy</div>
            </div>""", unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="kpi-card amber animate-in animate-in-d4">
                <div class="kpi-icon">🧪</div>
                <div class="kpi-value">{bench_count}</div>
                <div class="kpi-label">Benchmark Claims</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Visual Pipeline Stepper
        steps = [
            ("1", "Collect Data", has_data),
            ("2", "Train Models", has_models),
            ("3", "Benchmark", has_bench),
            ("4", "Analyze", has_results),
        ]
        step_html = '<div class="pipeline-stepper animate-in">'
        for num, label, done in steps:
            cls = "completed" if done else ""
            dot_cls = "done" if done else "pending"
            # Mark the first incomplete step as active
            if not done and all(s[2] for s in steps[:steps.index((num, label, done))]):
                dot_cls = "active"
                cls = "is-active"
            step_html += f"""
            <div class="step-item {cls}">
                <div class="step-dot {dot_cls}">{'✓' if done else num}</div>
                <div class="step-label">{label}</div>
            </div>"""
        step_html += '</div>'
        st.markdown(step_html, unsafe_allow_html=True)

        st.markdown("---")

        # Getting started + Status
        g1, g2 = st.columns([3, 2])
        with g1:
            st.markdown("""
            <div class="glass-card">
                <h3>🚀 Getting Started</h3>
                <p style='margin-bottom:0.8rem;'>Follow these steps to run the complete fact-checking pipeline:</p>
                <ol style='padding-left:1.2rem; line-height:2;'>
                    <li><strong>Data Collection</strong> — Scrape PolitiFact claims by date range</li>
                    <li><strong>Model Training</strong> — Choose an NLP feature phase and train 6 classifiers (4 ML + ANN + RNN)</li>
                    <li><strong>Benchmark Testing</strong> — Validate models against Google Fact Check data</li>
                    <li><strong>Results & Analysis</strong> — Compare metrics, visualize performance, read AI critique</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)

        with g2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 📋 Current Status")
            if has_data:
                st.success(f"✅ Data: {total_claims} claims loaded")
            else:
                st.warning("⏳ Data: No data collected yet")
            if has_models:
                st.success(f"✅ Models: {models_count} models trained")
            else:
                st.info("⏳ Models: Awaiting training")
            if has_bench:
                st.success(f"✅ Benchmark: {bench_count} claims tested")
            else:
                st.info("⏳ Benchmark: Ready when models are trained")
            st.markdown('</div>', unsafe_allow_html=True)

    # ╔══════════════════════════════════════════╗
    # ║         DATA COLLECTION PAGE             ║
    # ╚══════════════════════════════════════════╝
    elif "Data Collection" in page:
        st.markdown("""
        <div class="hero-banner animate-in">
            <h1>📥 Data Collection</h1>
            <p>Scrape verified political claims from PolitiFact archives</p>
        </div>
        """, unsafe_allow_html=True)

        col_form, col_stats = st.columns([3, 2])

        with col_form:
            st.markdown('<div class="glass-card animate-in animate-in-d1">', unsafe_allow_html=True)
            st.markdown("#### 🌐 PolitiFact Scraper")

            min_date = pd.to_datetime('2007-01-01')
            max_date = pd.to_datetime('today').normalize()

            d1, d2 = st.columns(2)
            with d1:
                start_date = st.date_input("Start Date", min_value=min_date, max_value=max_date,
                                           value=pd.to_datetime('2023-01-01'))
            with d2:
                end_date = st.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("🔄 Start Scraping", key="scrape_btn", use_container_width=True):
                if start_date > end_date:
                    st.error("Start date must be before end date.")
                else:
                    with st.spinner("Scraping political claims…"):
                        scraped_df = scrape_data_by_date_range(pd.to_datetime(start_date), pd.to_datetime(end_date))
                    if not scraped_df.empty:
                        st.session_state['scraped_df'] = scraped_df
                        st.success(f"✅ Successfully scraped **{len(scraped_df)}** claims!")
                    else:
                        st.warning("No data found. Try adjusting the date range.")
            st.markdown('</div>', unsafe_allow_html=True)

            # Data preview
            if has_data:
                st.markdown('<div class="glass-card animate-in animate-in-d2">', unsafe_allow_html=True)
                st.markdown("#### 📋 Data Preview")
                st.dataframe(st.session_state['scraped_df'].head(12), use_container_width=True, height=440)
                st.markdown('</div>', unsafe_allow_html=True)

        with col_stats:
            st.markdown('<div class="glass-card animate-in animate-in-d2">', unsafe_allow_html=True)
            st.markdown("#### 📊 Data Statistics")

            if has_data:
                df = st.session_state['scraped_df']
                st.metric("Total Claims", len(df))
                st.metric("Unique Labels", df['label'].nunique())
                st.metric("Date Range", f"{df['date'].min()} → {df['date'].max()}" if 'date' in df.columns else "N/A")

                st.markdown("---")
                st.markdown("##### Label Distribution")

                label_counts = df['label'].value_counts().reset_index()
                label_counts.columns = ['Label', 'Count']

                colors = ['#6366f1', '#8b5cf6', '#a78bfa', '#22d3ee', '#10b981', '#f59e0b', '#f43f5e', '#ec4899']
                fig = px.bar(label_counts, x='Count', y='Label', orientation='h',
                             color='Label', color_discrete_sequence=colors)
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#94a3b8', family='Inter'),
                    showlegend=False, height=300,
                    margin=dict(l=0, r=0, t=10, b=10),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown("""
                <div style='text-align:center; padding:3rem 1rem; color:#475569 !important;'>
                    <div style='font-size:3rem; margin-bottom:0.5rem;'>📭</div>
                    <div style='font-weight:600; font-size:1rem;'>No Data Yet</div>
                    <div style='font-size:0.85rem; margin-top:0.3rem;'>Configure the scraper and click Start Scraping</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ╔══════════════════════════════════════════╗
    # ║         MODEL TRAINING PAGE              ║
    # ╚══════════════════════════════════════════╝
    elif "Model Training" in page:
        st.markdown("""
        <div class="hero-banner animate-in">
            <h1>🧠 Model Training</h1>
            <p>NLP feature extraction · 6 classifiers (4 ML + 2 Deep Learning) · Stratified K-Fold CV · SMOTE balancing</p>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state['scraped_df'].empty:
            st.warning("⚠️ Please collect data first from the **Data Collection** page!")
        else:
            col_config, col_info = st.columns([3, 2])

            with col_config:
                st.markdown('<div class="glass-card animate-in animate-in-d1">', unsafe_allow_html=True)
                st.markdown("#### ⚙️ Training Configuration")

                phases = ["Lexical & Morphological", "Syntactic", "Semantic", "Discourse", "Pragmatic"]
                phase_meta = {
                    "Lexical & Morphological": ("📝", "Word-level analysis with lemmatization, stopword removal & n-grams"),
                    "Syntactic":               ("🔤", "Grammar structure via part-of-speech tags & sentence patterns"),
                    "Semantic":                ("💭", "Sentiment analysis — polarity & subjectivity scoring"),
                    "Discourse":               ("📐", "Text structure — sentence count & discourse markers"),
                    "Pragmatic":               ("🎯", "Intent analysis — modal verbs & emphasis markers"),
                }

                selected_phase = st.selectbox("Feature Extraction Method", phases, key='selected_phase')

                icon, desc = phase_meta[selected_phase]
                st.markdown(f"""
                <div class="phase-card">
                    <div class="phase-icon">{icon}</div>
                    <div>
                        <div class="phase-name">{selected_phase}</div>
                        <div class="phase-desc">{desc}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                if st.button("🚀 Run Model Analysis", key="analyze_btn", use_container_width=True):
                    with st.spinner(f"Training 6 models (4 ML + 2 Deep Learning) with {N_SPLITS}-Fold Cross Validation…"):
                        result = evaluate_models(st.session_state['scraped_df'], selected_phase)
                        if isinstance(result, tuple) and len(result) == 4:
                            df_results, trained_models, trained_vectorizer, rnn_tok = result
                        elif isinstance(result, tuple) and len(result) == 3:
                            df_results, trained_models, trained_vectorizer = result
                            rnn_tok = None
                        else:
                            df_results = result
                            trained_models = {}
                            trained_vectorizer = None
                            rnn_tok = None
                        st.session_state['df_results'] = df_results
                        st.session_state['trained_models'] = trained_models
                        st.session_state['trained_vectorizer'] = trained_vectorizer
                        st.session_state['rnn_tokenizer'] = rnn_tok
                        st.session_state['selected_phase_run'] = selected_phase
                        st.success("✅ Analysis complete! View results in **Results & Analysis** page.")
                st.markdown('</div>', unsafe_allow_html=True)

                # Quick results preview
                if has_results:
                    st.markdown('<div class="glass-card animate-in animate-in-d2">', unsafe_allow_html=True)
                    st.markdown("#### ⚡ Quick Results Preview")
                    df_r = st.session_state['df_results']
                    cols = st.columns(len(df_r))
                    accents = ['indigo', 'emerald', 'cyan', 'amber', 'rose', 'violet']
                    for i, (_, row) in enumerate(df_r.iterrows()):
                        with cols[i]:
                            acc_cls = accents[i % len(accents)]
                            st.markdown(f"""
                            <div class="kpi-card {acc_cls}">
                                <div class="kpi-value">{row['Accuracy']:.1f}%</div>
                                <div class="kpi-label">{row['Model']}</div>
                            </div>""", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

            with col_info:
                st.markdown('<div class="glass-card animate-in animate-in-d2">', unsafe_allow_html=True)
                st.markdown("#### 🤖 Model Arsenal")
                models_info = [
                    ("🟣", "Naive Bayes", "Fast probabilistic classifier"),
                    ("🟢", "Decision Tree", "Interpretable rule-based splits"),
                    ("🔵", "Logistic Regression", "Linear boundary with regularization"),
                    ("🟠", "SVM", "Maximum margin hyperplane classifier"),
                    ("🔴", "ANN", "Feedforward neural network (Dense layers)"),
                    ("🟡", "RNN (LSTM)", "Bidirectional LSTM for sequence learning"),
                ]
                for dot, name, desc_ in models_info:
                    st.markdown(f"""
                    <div class="phase-card">
                        <div class="phase-icon">{dot}</div>
                        <div>
                            <div class="phase-name">{name}</div>
                            <div class="phase-desc">{desc_}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="glass-card animate-in animate-in-d3">', unsafe_allow_html=True)
                st.markdown("#### 📋 Training Pipeline")
                st.markdown("""
                - 🔀 **Stratified K-Fold** — 5 balanced folds
                - ⚖️ **SMOTE** — Synthetic oversampling for class balance
                - 🧠 **Deep Learning** — ANN + Bidirectional LSTM
                - 📈 **Metrics** — Accuracy, F1, Precision, Recall
                - ⏱️ **Timing** — Training + inference latency
                """)
                if has_models:
                    st.success(f"✅ {len(st.session_state['trained_models'])} models trained")
                    st.info(f"Phase: **{st.session_state.get('selected_phase_run', 'N/A')}**")
                st.markdown('</div>', unsafe_allow_html=True)

    # ╔══════════════════════════════════════════╗
    # ║        BENCHMARK TESTING PAGE            ║
    # ╚══════════════════════════════════════════╝
    elif "Benchmark" in page:
        st.markdown("""
        <div class="hero-banner animate-in">
            <h1>🎯 Benchmark Testing</h1>
            <p>Validate trained models against real-world fact-check data</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="glass-card animate-in animate-in-d1">', unsafe_allow_html=True)
        st.markdown("#### 🧪 Fact Check Benchmark Configuration")

        m1, m2, m3 = st.columns([2, 2, 1])
        with m1:
            use_demo = st.checkbox("Use Demo Data (no API key needed)", value=True,
                                   help="Test with 15 built-in sample fact-check claims")
        with m2:
            if not use_demo:
                if 'GOOGLE_API_KEY' not in st.secrets:
                    st.error("API Key not found in **.streamlit/secrets.toml**")
                else:
                    st.success("✅ API Key found!")
        with m3:
            num_claims = st.number_input("Claims to fetch", min_value=5, max_value=500, value=50, step=10, key='num_claims',
                                          help="Number of claims to fetch from Google Fact Check API (no upper limit enforced by API)")

        st.markdown("<br>", unsafe_allow_html=True)

        b1, b2 = st.columns([3, 1])
        with b1:
            if st.button("🚀 Run Benchmark Test", key="benchmark_btn", use_container_width=True):
                if not st.session_state.get('trained_models'):
                    st.error("Please train models first in the **Model Training** page!")
                else:
                    with st.spinner('Loading and testing fact-check data…'):
                        if use_demo:
                            api_results = get_demo_google_claims()
                            st.success("✅ Demo data loaded!")
                        else:
                            api_key = st.secrets["GOOGLE_API_KEY"]
                            api_results = fetch_google_claims(api_key, num_claims)
                            if api_results:
                                st.success(f"✅ Fetched {len(api_results)} claims!")

                        google_df = process_and_map_google_claims(api_results)

                        if not google_df.empty:
                            benchmark_df = run_google_benchmark(
                                google_df,
                                st.session_state['trained_models'],
                                st.session_state['trained_vectorizer'],
                                st.session_state['selected_phase_run'],
                                st.session_state.get('rnn_tokenizer', None)
                            )
                            st.session_state['google_benchmark_results'] = benchmark_df
                            st.session_state['google_df'] = google_df
                            st.success(f"✅ Benchmark complete — tested on **{len(google_df)}** claims!")
                        else:
                            st.warning("No claims processed. Try adjusting parameters.")
        with b2:
            st.caption("Tests trained models against independent fact-check data")

        st.markdown('</div>', unsafe_allow_html=True)

        # Benchmark results
        if has_bench:
            st.markdown('<div class="glass-card animate-in animate-in-d2">', unsafe_allow_html=True)
            st.markdown("#### 📊 Benchmark Results")
            bench_df = st.session_state['google_benchmark_results']

            # Metric cards
            cols = st.columns(len(bench_df))
            accents = ['indigo', 'emerald', 'cyan', 'amber', 'rose', 'violet']
            for i, (_, row) in enumerate(bench_df.iterrows()):
                with cols[i]:
                    st.markdown(f"""
                    <div class="kpi-card {accents[i % len(accents)]}">
                        <div class="kpi-value">{row['Accuracy']:.1f}%</div>
                        <div class="kpi-label">{row['Model']}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(bench_df, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ╔══════════════════════════════════════════╗
    # ║       RESULTS & ANALYSIS PAGE            ║
    # ╚══════════════════════════════════════════╝
    elif "Results" in page:
        st.markdown("""
        <div class="hero-banner animate-in">
            <h1>📊 Results & Analysis</h1>
            <p>Comprehensive performance metrics, visualizations & AI critique</p>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state['df_results'].empty:
            st.warning("⚠️ No results available. Please train models first in the **Model Training** page!")
        else:
            df_results = st.session_state['df_results']

            # ── Winner Podium ──
            sorted_df = df_results.sort_values('F1-Score', ascending=False).reset_index(drop=True)
            medals = ["🥇", "🥈", "🥉"]
            card_cls = ["gold", "silver", "bronze"]
            st.markdown("### 🏆 Winner's Podium")

            podium_cols = st.columns(min(len(sorted_df), 3))
            for i in range(min(len(sorted_df), 3)):
                row = sorted_df.iloc[i]
                with podium_cols[i]:
                    st.markdown(f"""
                    <div class="podium-card {card_cls[i]} animate-in animate-in-d{i+1}">
                        <div class="podium-medal">{medals[i]}</div>
                        <div class="podium-name">{row['Model']}</div>
                        <div class="podium-score">{row['Accuracy']:.1f}%</div>
                        <div class="podium-sub">F1: {row['F1-Score']:.3f} · {row['Inference Latency (ms)']}ms</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")

            # ── Charts Tabs ──
            tab1, tab2, tab3 = st.tabs(["📊 Metrics Comparison", "🕸️ Radar Chart", "⚡ Speed vs Accuracy"])

            with tab1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                metric_choice = st.selectbox(
                    "Select metric:",
                    ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'Training Time (s)', 'Inference Latency (ms)'],
                    key='chart_metric'
                )
                fig = px.bar(
                    df_results, x='Model', y=metric_choice,
                    color='Model',
                    color_discrete_sequence=['#6366f1', '#10b981', '#22d3ee', '#f59e0b', '#f43f5e', '#a78bfa'],
                    text=df_results[metric_choice].apply(lambda v: f"{v:.2f}" if isinstance(v, float) else str(v))
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#94a3b8', family='Inter'),
                    showlegend=False, height=420,
                    margin=dict(l=40, r=20, t=30, b=60),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', title=''),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title=metric_choice),
                )
                fig.update_traces(textposition='outside', textfont=dict(color='#e2e8f0', size=12, family='Inter'))
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with tab2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                categories = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
                fig_radar = go.Figure()
                radar_colors = ['#6366f1', '#10b981', '#22d3ee', '#f59e0b', '#f43f5e', '#a78bfa']
                for i, (_, row) in enumerate(df_results.iterrows()):
                    values = [row[c] if c != 'Accuracy' else row[c] / 100 for c in categories]
                    values.append(values[0])
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories + [categories[0]],
                        fill='toself',
                        name=row['Model'],
                        line=dict(color=radar_colors[i % len(radar_colors)]),
                        fillcolor=radar_colors[i % len(radar_colors)].replace(')', ',0.1)').replace('rgb', 'rgba') if 'rgb' in radar_colors[i % len(radar_colors)] else None,
                        opacity=0.8,
                    ))
                fig_radar.update_layout(
                    polar=dict(
                        bgcolor='rgba(0,0,0,0)',
                        radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(255,255,255,0.08)',
                                        tickfont=dict(color='#64748b', size=10)),
                        angularaxis=dict(gridcolor='rgba(255,255,255,0.08)',
                                         tickfont=dict(color='#94a3b8', size=11, family='Inter')),
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#94a3b8', family='Inter'),
                    legend=dict(font=dict(color='#94a3b8')),
                    height=450, margin=dict(l=60, r=60, t=40, b=40),
                )
                st.plotly_chart(fig_radar, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with tab3:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                fig_scatter = px.scatter(
                    df_results, x='Inference Latency (ms)', y='Accuracy',
                    size='F1-Score', color='Model',
                    color_discrete_sequence=['#6366f1', '#10b981', '#22d3ee', '#f59e0b', '#f43f5e', '#a78bfa'],
                    text='Model', size_max=40,
                )
                fig_scatter.update_traces(textposition='top center',
                                          textfont=dict(color='#e2e8f0', size=11, family='Inter'))
                fig_scatter.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#94a3b8', family='Inter'),
                    height=420, margin=dict(l=40, r=20, t=30, b=60),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='Inference Latency (ms)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='Accuracy (%)'),
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # ── Detailed Metrics Table ──
            st.markdown("---")
            st.markdown("### 📋 Detailed Metrics")
            st.dataframe(df_results, use_container_width=True, hide_index=True)

            # ── Google Benchmark Comparison ──
            if has_bench:
                st.markdown("---")
                st.markdown("### 🔄 PolitiFact vs Google Benchmark Comparison")

                google_results = st.session_state['google_benchmark_results']
                comp_cols = st.columns(len(google_results))

                for idx, (_, row) in enumerate(google_results.iterrows()):
                    model_name = row['Model']
                    google_acc = row['Accuracy']
                    pf_row = df_results[df_results['Model'] == model_name]
                    if not pf_row.empty:
                        pf_acc = pf_row['Accuracy'].values[0]
                        delta = google_acc - pf_acc
                        delta_color = "normal" if delta >= 0 else "inverse"
                    else:
                        delta = None
                        delta_color = "off"

                    with comp_cols[idx]:
                        if delta is not None:
                            st.metric(label=model_name, value=f"{google_acc:.1f}%",
                                      delta=f"{delta:+.1f}%", delta_color=delta_color)
                        else:
                            st.metric(label=model_name, value=f"{google_acc:.1f}%")

            # ── AI Humorous Critique ──
            st.markdown("---")
            st.markdown("### 🎭 AI Performance Review")

            cr1, cr2 = st.columns([3, 2])
            with cr1:
                critique_text = generate_humorous_critique(
                    st.session_state['df_results'],
                    st.session_state.get('selected_phase_run', 'Unknown')
                )
                st.markdown(f'<div class="critique-card">{critique_text}</div>', unsafe_allow_html=True)

            with cr2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("#### 🏅 Champion Model")
                if not df_results.empty:
                    best = df_results.loc[df_results['F1-Score'].idxmax()]
                    st.markdown(f"""
                    <div style='text-align:center; padding:0.5rem 0;'>
                        <div style='font-size:2.5rem;'>🏆</div>
                        <div style='font-size:1.2rem; font-weight:800; color:#f1f5f9 !important; margin:0.3rem 0;'>{best['Model']}</div>
                        <div style='font-size:2rem; font-weight:800; color:#6366f1 !important;'>{best['Accuracy']:.1f}%</div>
                        <div style='font-size:0.82rem; color:#64748b !important; margin-top:0.3rem;'>
                            F1: {best['F1-Score']:.3f} · Precision: {best['Precision']:.3f} · Recall: {best['Recall']:.3f}
                        </div>
                        <div style='margin-top:0.8rem;'>
                            <span class="status-badge ready">{st.session_state.get('selected_phase_run', 'N/A')}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)


# ── Run App ──
if __name__ == '__main__':
    app()

