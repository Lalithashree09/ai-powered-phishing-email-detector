# features.py
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
from urllib.parse import urlparse
import whois
import hashlib
import pickle
import os
import pandas as pd

nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

PHISHING_KEYWORDS = [
    'urgent', 'verify', 'account', 'suspend', 'click', 'login', 'password',
    'bank', 'paypal', 'security', 'update', 'immediate', 'action required'
]

WHOIS_CACHE = 'models/whois_cache.pkl'
whois_cache = {}

if os.path.exists(WHOIS_CACHE):
    with open(WHOIS_CACHE, 'rb') as f:
        whois_cache = pickle.load(f)

def save_cache():
    os.makedirs('models', exist_ok=True)
    with open(WHOIS_CACHE, 'wb') as f:
        pickle.dump(whois_cache, f)

def extract_urls(text):
    return re.findall(r'http[s]?://[^\s<>"]+|www\.[^\s<>"]+', str(text))

def domain_entropy(domain):
    freq = {c: domain.count(c) for c in set(domain)}
    entropy = 0
    for count in freq.values():
        p = count / len(domain)
        entropy -= p * np.log2(p)
    return entropy if entropy > 0 else 0

def get_domain_age(domain):
    key = hashlib.md5(domain.encode()).hexdigest()
    if key in whois_cache:
        return whois_cache[key]
    try:
        w = whois.whois(domain)
        date = w.creation_date
        if isinstance(date, list):
            date = date[0]
        if date:
            age = (pd.Timestamp.now() - pd.to_datetime(date)).days
            whois_cache[key] = age
            save_cache()
            return age
    except:
        pass
    return np.nan

def extract_features_from_text(email_text):
    if not isinstance(email_text, str):
        email_text = str(email_text)
    
    full_text = email_text.lower()
    urls = extract_urls(full_text)
    
    # Text features
    keyword_count = sum(kw in full_text for kw in PHISHING_KEYWORDS)
    sentiment = sia.polarity_scores(full_text)
    length = len(full_text)
    url_count = len(urls)
    
    # URL features
    suspicious_urls = 0
    domain_age_days = np.nan
    
    for url in urls:
        domain = urlparse(url).netloc.lower().replace('www.', '')
        if not domain:
            continue
        if domain_entropy(domain) > 3.5:
            suspicious_urls += 1
        age = get_domain_age(domain)
        if pd.notna(age):
            if np.isnan(domain_age_days) or age < domain_age_days:
                domain_age_days = age
            if age < 180:
                suspicious_urls += 1

    return {
        'keyword_count': keyword_count,
        'sentiment_neg': sentiment['neg'],
        'length': length,
        'url_count': url_count,
        'suspicious_urls': suspicious_urls,
        'domain_age_days': domain_age_days,
        'google_safe': 0  # API not used
    }, full_text, urls, email_text