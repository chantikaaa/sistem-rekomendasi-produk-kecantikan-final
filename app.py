import streamlit as st
import pandas as pd
import re
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# 1. SETUP HALAMAN
st.set_page_config(
    page_title="Skincare Recommender",
    page_icon="💄",
    layout="wide"
)

st.title("Sistem Rekomendasi Produk Kecantikan Lokal 💄")
st.markdown("Berbasis Content-Based Filtering menggunakan Model Natural Language Processing (NLP)")
st.write("---")


# 2. LOAD DATA & MODEL
@st.cache_resource
def load_model():
    return SentenceTransformer('indobenchmark/indobert-base-p1')

@st.cache_data
def load_data():
    df_model = pd.read_csv('data_final.csv')
    df_ui = pd.read_csv('data-ui.csv', sep=';')
    embeddings = np.load('embeddings.npy')
    tfidf_matrix = np.load('tfidf_matrix.npy')
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    return df_model, df_ui, embeddings, tfidf_matrix, tfidf

model = load_model()
df_model, df_ui, embeddings, tfidf_matrix, tfidf = load_data()


# 3. PREPROCESSING & FUNGSI REKOMENDASI
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'expired date.*?(\n|$)', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'peringatan.*?(\n|$)', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def simplify_query(text):
    text = clean_text(text)
    words = text.split()
    keywords = [w for w in words if len(w) > 2]
    return ' '.join(keywords[:20])

def preprocess_query_tfidf(text):
    text = clean_text(text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = stemmer.stem(text)
    text = stopword_remover.remove(text)
    words = text.split()
    keywords = [w for w in words if len(w) > 2]
    return ' '.join(keywords)

def get_embedding(text):
    return model.encode(text, normalize_embeddings=True)

def choose_diverse_top_indices(similarities, top_k=15, result_n=5, max_pairwise=0.88):
    top_indices = similarities.argsort()[::-1][:top_k]
    chosen = []
    for idx in top_indices:
        idx = int(idx)
        if not chosen:
            chosen.append(idx)
            continue
        pair_sims = cosine_similarity([embeddings[idx]], embeddings[chosen])[0]
        if pair_sims.max() < max_pairwise:
            chosen.append(idx)
        if len(chosen) >= result_n:
            break
    if len(chosen) < result_n:
        for idx in top_indices:
            idx = int(idx)
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) >= result_n:
                break
    return chosen


# Modal untuk deskripsi produk
@st.dialog("Deskripsi Produk")
def show_description_modal(idx):
    st.write(f"**{df_ui.loc[idx, 'Product Name']}**")
    st.write(df_ui.loc[idx, 'short description'])
    if st.button("Tutup"):
        st.session_state.pop('selected_idx', None)
        st.rerun()


# 4. INPUT USER
user_input = st.text_area(
    "Masukkan produk acuan dan permasalahan kulit Anda:",
    placeholder="Contoh: 'Wardah Lightening Serum, kulit berjerawat..."
)


# 5. TOMBOL
if st.button("Rekomendasikan Produk"):
    st.session_state.pop('selected_idx', None)
    st.session_state.pop('top_indices', None)

    if not user_input.strip():
        st.warning("Harap isi input dahulu!")
    else:
        # IndoBERT + TF-IDF Weighted 50:50
        query_bert = simplify_query(user_input)
        query_tfidf_clean = preprocess_query_tfidf(user_input)

        query_embedding = get_embedding(query_bert)
        sim_bert = cosine_similarity([query_embedding], embeddings)[0]

        query_vec = tfidf.transform([query_tfidf_clean])
        sim_tfidf_query = cosine_similarity(query_vec, tfidf_matrix)[0]

        # Normalisasi min-max
        sim_bert_norm = (sim_bert - sim_bert.min()) / (sim_bert.max() - sim_bert.min() + 1e-9)
        sim_tfidf_norm = (sim_tfidf_query - sim_tfidf_query.min()) / (sim_tfidf_query.max() - sim_tfidf_query.min() + 1e-9)

        similarities = 0.5 * sim_bert_norm + 0.5 * sim_tfidf_norm
        top_indices = similarities.argsort()[::-1][:5].tolist()
        st.session_state['top_indices'] = top_indices


# 6. DISPLAY HASIL
if 'top_indices' in st.session_state:
    top_indices = st.session_state['top_indices']

    st.write("---")
    st.subheader("Produk Lokal Rekomendasi untuk kamu:")

    num_cols = min(len(top_indices), 5)
    cols = st.columns(num_cols)
    for i, idx in enumerate(top_indices):
        with cols[i]:
            st.image(df_ui.loc[idx, 'Product Image 1'], caption="Produk")
            st.write(f"**{df_ui.loc[idx, 'Product Name']}**")
            st.caption(df_ui.loc[idx, 'short description'][:80] + "...")

            if st.button("Lihat Deskripsi", key=f"desc_{idx}"):
                st.session_state['selected_idx'] = idx

    if 'selected_idx' in st.session_state:
        show_description_modal(st.session_state['selected_idx'])


# FOOTER
st.write("---")
st.caption("Prototype ini dibuat dengan Streamlit")