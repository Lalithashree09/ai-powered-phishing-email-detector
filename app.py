# app.py
import streamlit as st
import pandas as pd
import joblib
from features import extract_features_from_text

st.set_page_config(page_title="Phishing Detector", page_icon="shield")

@st.cache_resource
def load_model():
    model = joblib.load('models/phishing_model_xgboost.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    imputer = joblib.load('models/imputer.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, vectorizer, imputer, scaler

model, vectorizer, imputer, scaler = load_model()

st.title("AI Phishing Detector")
st.write("**XGBoost + Kaggle Dataset** | **97%+ Accuracy**")

with st.sidebar:
    st.header("Safety Tips")
    st.write("- Never click unverified links")
    st.write("- Check sender email")
    st.write("- Report to security")

st.header("Scan Email")
email = st.text_area("Paste full email:", height=200)

if st.button("Scan Now", type="primary"):
    if email.strip():
        with st.spinner("Analyzing..."):
            feats, _, urls, _ = extract_features_from_text(email)
            
            struct = pd.DataFrame([[
                feats['keyword_count'], feats['sentiment_neg'], feats['length'],
                feats['url_count'], feats['suspicious_urls'],
                feats['domain_age_days'] if pd.notna(feats['domain_age_days']) else -1,
                feats['google_safe']
            ]], columns=[
                'keyword_count', 'sentiment_neg', 'length', 'url_count',
                'suspicious_urls', 'domain_age_days', 'google_safe'
            ])
            struct_imp = pd.DataFrame(imputer.transform(struct), columns=struct.columns)
            struct_scaled = pd.DataFrame(scaler.transform(struct_imp), columns=struct.columns)
            
            tfidf = vectorizer.transform([email]).toarray()
            tfidf_df = pd.DataFrame(tfidf, columns=[f'tfidf_{i}' for i in range(tfidf.shape[1])])
            
            final = pd.concat([struct_scaled, tfidf_df], axis=1)
            prob = model.predict_proba(final)[0][1]
            pred = int(prob > 0.5)

        col1, col2 = st.columns([1, 2])
        with col1:
            if pred == 1:
                st.error("PHISHING")
            else:
                st.success("SAFE")
        with col2:
            st.metric("Threat Level", f"{prob:.1%}")
        st.progress(prob)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Keywords", feats['keyword_count'])
            st.metric("Tone", f"{feats['sentiment_neg']:.2f}")
        with c2:
            st.metric("Length", feats['length'])
            st.metric("URLs", len(urls))
        with c3:
            st.metric("Suspicious", feats['suspicious_urls'])
            age = feats['domain_age_days']
            st.metric("Domain Age", f"{int(age)} days" if pd.notna(age) else "N/A")

        if urls:
            st.write("**Links Found:**")
            for u in urls[:5]:
                st.code(u)
    else:
        st.warning("Enter email text.")