import streamlit as st
import pandas as pd
from tabs.limpeza import render_pre_processamento
from tabs.treino_teste import render_treino_teste
from tabs.random_forest import render_random_forest
from tabs.knn import render_knn
from tabs.eda import render_eda
from tabs.naive_bayes import render_naive_bayes
from tabs.docs import render_doc

st.set_page_config(page_title='Credit Card Default - UCI')
st.title('Análise de Inadimplência - UCI Credit Card')

# Carregar dados
@st.cache_data
def load_data():
    return pd.read_csv('src/data/UCI_Credit_Card.csv')

df_raw = load_data()

TABS = [
    'Documentação',
    'Análise Exploratória',
    'Pré-processamento',
    'Treino/Teste',
    'Modelos',
]

tabs = st.tabs(TABS)

# --- Aba 1: Documentação ---
with tabs[0]:
    render_doc()

# --- Aba 2: Análise Exploratória ---
with tabs[1]:
    render_eda(df_raw)

# --- Aba 3: Pré-processamento ---
with tabs[2]:
    render_pre_processamento(df_raw)

# --- Aba 4: Treino/Teste ---
with tabs[3]:
    train_df, test_df = render_treino_teste(df_raw)

# --- Aba 5: Modelos ---
with tabs[4]:
    st.header('Modelos de Classificação')
    model_tabs = st.tabs(['Random Forest', 'KNN', 'Naive Bayes'])
    with model_tabs[0]:
        render_random_forest(train_df, test_df)
    with model_tabs[1]:
        render_knn(train_df, test_df)
    with model_tabs[2]:
        render_naive_bayes(train_df, test_df)
