import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def render_eda(df):
    st.header('Análise Exploratória (EDA)')
    st.markdown('''
Esta aba permite explorar o dataset de forma interativa antes de qualquer modelagem. Aqui você pode:

- Visualizar estatísticas descritivas das variáveis numéricas, como média, mediana, desvio padrão, valores mínimos e máximos, facilitando a compreensão geral dos dados.
- Gerar gráficos de distribuição para qualquer variável numérica do dataset, escolhendo entre histograma (para analisar a frequência dos valores) ou boxplot (para identificar a dispersão e possíveis outliers).
- Visualizar a matriz de correlação entre as variáveis numéricas por meio de um heatmap, identificando relações lineares e possíveis colinearidades que podem impactar a modelagem.

As opções de seleção de variáveis e tipo de gráfico permitem analisar diferentes aspectos dos dados de forma visual e intuitiva, ajudando a identificar padrões, tendências e possíveis problemas de qualidade nos dados.
''')
    st.markdown('''
◦ Estatísticas descritivas
◦ Gráficos de distribuição (histogramas, boxplots)
◦ Análise de correlação
''')

    st.subheader('Estatísticas descritivas')
    st.write(df.describe())

    st.subheader('Gráficos de distribuição')
    col = st.selectbox('Selecione uma variável numérica:', df.select_dtypes(include='number').columns, key='eda_col')
    tipo_grafico = st.radio('Tipo de gráfico:', ['Histograma', 'Boxplot'], key='eda_grafico')
    fig, ax = plt.subplots()
    if tipo_grafico == 'Histograma':
        sns.histplot(df[col], kde=True, ax=ax)
    else:
        sns.boxplot(x=df[col], ax=ax)
    st.pyplot(fig)

    st.subheader('Análise de correlação')
    corr = df.corr()
    fig_corr, ax_corr = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax_corr)
    st.pyplot(fig_corr)
    st.write('Matriz de correlação:', corr)
