# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
from features import extract_features_from_text
from tqdm import tqdm
import pickle
import os
import multiprocessing as mp
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === PATHS ===
CHECKPOINT_FILE = 'models/XGB/features_checkpoint.pkl'
MODEL_PATH = 'models/XGB/phishing_model_xgboost.pkl'
VECTORIZER_PATH = 'models/XGB/tfidf_vectorizer.pkl'
IMPUTER_PATH = 'models/XGB/imputer.pkl'
SCALER_PATH = 'models/XGB/scaler.pkl'

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'rb') as f:
            data = pickle.load(f)
        logging.info(f"Resuming from index {data['last_index']}")
        return data['features'], data['last_index']
    return [], -1

def save_checkpoint(features, idx):
    os.makedirs('models', exist_ok=True)
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump({'features': features, 'last_index': idx}, f)

def process_row(text):
    feats, _, _, _ = extract_features_from_text(text)
    return [
        feats['keyword_count'],
        feats['sentiment_neg'],
        feats['length'],
        feats['url_count'],
        feats['suspicious_urls'],
        feats['domain_age_days'] if pd.notna(feats['domain_age_days']) else -1,
        feats['google_safe']
    ]

# === LOAD DATA ===
logging.info("Loading Phishing_Email.csv...")
if not os.path.exists('Phishing_Email.csv'):
    logging.error("Phishing_Email.csv not found!")
    exit(1)

df = pd.read_csv('Phishing_Email.csv')
df = df.dropna(subset=['Email Text', 'Email Type']).reset_index(drop=True)

# Map labels
label_map = {'Phishing Email': 1, 'Safe Email': 0}
df['label'] = df['Email Type'].map(label_map)
if df['label'].isnull().any():
    logging.error("Unknown label in Email Type. Use only 'Phishing Email' or 'Safe Email'.")
    exit(1)

logging.info(f"Loaded {len(df)} emails | Phishing: {df['label'].sum()}")

# === CLASS BALANCE ===
scale_pos_weight = (df['label'] == 0).sum() / (df['label'] == 1).sum()
logging.info(f"scale_pos_weight = {scale_pos_weight:.2f}")

# === FEATURES ===
features_list, last_idx = load_checkpoint()
start = last_idx + 1 if last_idx >= 0 else 0

if start < len(df):
    for i in range(start, len(df), 500):
        batch_texts = df['Email Text'].iloc[i:i+500].tolist()
        logging.info(f"Extracting features: {i}–{i+len(batch_texts)-1}")
        with mp.Pool(mp.cpu_count()) as pool:
            batch_feats = list(tqdm(pool.imap(process_row, batch_texts), total=len(batch_texts)))
        features_list.extend(batch_feats)
        save_checkpoint(features_list, i + len(batch_texts) - 1)
else:
    logging.info("Using cached features")

cols = ['keyword_count', 'sentiment_neg', 'length', 'url_count', 'suspicious_urls', 'domain_age_days', 'google_safe']
X_struct = pd.DataFrame(features_list, columns=cols)

# === TF-IDF ===
logging.info("Computing TF-IDF...")
vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1,2), stop_words='english', min_df=2)
X_tfidf = vectorizer.fit_transform(df['Email Text']).toarray()
X_tfidf_df = pd.DataFrame(X_tfidf, columns=[f'tfidf_{i}' for i in range(X_tfidf.shape[1])])

# === IMPUTE & SCALE ===
imputer = SimpleImputer(strategy='mean')
X_struct_imp = pd.DataFrame(imputer.fit_transform(X_struct), columns=cols)
scaler = StandardScaler()
X_struct_scaled = pd.DataFrame(scaler.fit_transform(X_struct_imp), columns=cols)

# === COMBINE ===
X = pd.concat([X_struct_scaled.reset_index(drop=True), X_tfidf_df.reset_index(drop=True)], axis=1)
y = df['label']

# === TRAIN ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    n_estimators=500,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    n_jobs=-1,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# === RESULTS ===
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nXGBoost Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred, target_names=['Safe', 'Phishing']))

# === SAVE ===
os.makedirs('models', exist_ok=True)
joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)
joblib.dump(imputer, IMPUTER_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"Model saved: {MODEL_PATH}")