import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from tabs.dados_originais import render_dados_originais

def render_eda(df):
    st.header('Análise Exploratória (EDA)')

    st.markdown('''
Esta aba permite explorar o dataset de forma interativa antes de qualquer modelagem. Aqui você pode:

- Visualizar estatísticas descritivas das variáveis numéricas, como média, mediana, desvio padrão, valores mínimos e máximos, facilitando a compreensão geral dos dados.
- Gerar gráficos de distribuição para qualquer variável numérica do dataset, escolhendo entre histograma (para analisar a frequência dos valores) ou boxplot (para identificar a dispersão e possíveis outliers).
- Visualizar a matriz de correlação entre as variáveis numéricas por meio de um heatmap, identificando relações lineares e possíveis colinearidades que podem impactar a modelagem.

As opções de seleção de variáveis e tipo de gráfico permitem analisar diferentes aspectos dos dados de forma visual e intuitiva, ajudando a identificar padrões, tendências e possíveis problemas de qualidade nos dados.
''')

    render_dados_originais(df)
    st.markdown('---')

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
    st.markdown('''
**O que é Análise de Correlação?**

A análise de correlação avalia o grau de relação linear entre duas variáveis numéricas. O coeficiente de correlação varia de -1 a 1:
- Valores próximos de 1 indicam forte correlação positiva (as variáveis crescem juntas).
- Valores próximos de -1 indicam forte correlação negativa (uma cresce enquanto a outra diminui).
- Valores próximos de 0 indicam pouca ou nenhuma relação linear.

**O que é a Matriz de Correlação?**

A matriz de correlação é uma tabela que mostra o coeficiente de correlação entre todas as variáveis numéricas do dataset. Cada célula da matriz indica o grau de relação linear entre um par de variáveis. Valores próximos de 1 ou -1 indicam forte relação, enquanto valores próximos de 0 indicam pouca ou nenhuma relação linear. O heatmap abaixo facilita a visualização dessas relações, destacando padrões e possíveis colinearidades.
''')
    corr = df.corr()
    fig_corr, ax_corr = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax_corr)
    st.pyplot(fig_corr)
    st.write('Matriz de correlação:', corr)
