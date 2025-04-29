import streamlit as st

def render_dados_originais(df):
    st.header('Visualização dos Dados Originais')
    st.write('Primeiras linhas do dataset original:')
    st.dataframe(df.head())
    st.subheader('Estatísticas Descritivas (Original)')
    st.write(df.describe())
