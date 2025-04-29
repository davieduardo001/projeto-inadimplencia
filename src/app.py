import streamlit as st
import pandas as pd
from tabs.limpeza import render_limpeza
from tabs.treino_teste import render_treino_teste
from tabs.random_forest import render_random_forest
from tabs.knn import render_knn
from tabs.eda import render_eda
from tabs.dados_originais import render_dados_originais
from tabs.naive_bayes import render_naive_bayes

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
    'Limpeza',
    'Treino/Teste',
    'Modelos',
]

tabs = st.tabs(TABS)

# --- Aba 1: Documentação ---
with tabs[0]:
    st.header('Documentação do Aplicativo')
    st.caption('''
**Aplicativo interativo para previsão de inadimplência em cartões de crédito.**

Este dashboard foi desenvolvido para demonstrar, de forma didática e visual, todas as etapas de um projeto real de Ciência de Dados aplicado ao problema de inadimplência, utilizando dados públicos da UCI.

- Pipeline completo: visualização, limpeza, divisão treino/teste, classificação (com escolha de algoritmo e métrica), visualização gráfica e documentação.
- Interface organizada em abas, facilitando a navegação e evitando o scroll infinito.
- Todo o código está comentado e explicado em português.
- Pode ser adaptado para outros datasets tabulares e outros problemas de classificação.
''')

    st.markdown('---')
    st.header('Explicando as Métricas de Avaliação')

    st.subheader('Matriz de Confusão')
    st.markdown('''
A matriz de confusão é uma tabela que mostra como o modelo acerta e erra as previsões, comparando os valores reais com os previstos.

|                 | Predito: NÃO | Predito: SIM |
|-----------------|:------------:|:------------:|
| **Real: NÃO**   |     VN       |     FP       |
| **Real: SIM**   |     FN       |     VP       |

- **VN (Verdadeiro Negativo):** O modelo previu NÃO e era realmente NÃO.
- **VP (Verdadeiro Positivo):** O modelo previu SIM e era realmente SIM.
- **FP (Falso Positivo):** O modelo previu SIM, mas era NÃO (falso alarme).
- **FN (Falso Negativo):** O modelo previu NÃO, mas era SIM (deixou passar um caso real).

> No contexto de crédito:
> - **SIM** = Inadimplente (deu calote)
> - **NÃO** = Não inadimplente

A diagonal principal (VN e VP) mostra os acertos, enquanto os outros valores mostram erros do modelo.
''')

    st.subheader('Precision e Recall')
    st.markdown('''
- **Precisão (Precision):** Entre todas as previsões positivas, quantas realmente eram positivas.
    - Fórmula: VP / (VP + FP)
- **Recall (Sensibilidade):** Entre todas as amostras realmente positivas, quantas o modelo identificou corretamente.
    - Fórmula: VP / (VP + FN)

Essas métricas são importantes para entender não só quantos acertos o modelo teve, mas também a qualidade desses acertos.
''')

    st.subheader('Curva ROC e AUC')
    st.markdown('''
A curva ROC (Receiver Operating Characteristic) mostra a relação entre taxa de verdadeiros positivos (sensibilidade) e taxa de falsos positivos para diferentes limiares de decisão. O AUC (Área sob a curva) mede a capacidade do modelo em distinguir entre as classes:

- **AUC = 1:** Modelo perfeito
- **AUC = 0.5:** Modelo aleatório
- **Quanto mais próximo de 1, melhor**
''')

# --- Aba 2: Análise Exploratória ---
with tabs[1]:
    st.subheader('Dados Originais')
    render_dados_originais(df_raw)
    st.markdown('---')
    render_eda(df_raw)

# --- Aba 3: Limpeza ---
with tabs[2]:
    render_limpeza(df_raw)

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
