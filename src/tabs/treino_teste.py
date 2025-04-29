import streamlit as st
from sklearn.model_selection import train_test_split

def render_treino_teste(df):
    st.header('Divisão dos Dados em Treino e Teste')
    st.markdown('''
A divisão dos dados em conjuntos de treino e teste é uma etapa fundamental em projetos de Machine Learning. 

**Por quê dividir os dados?**
- O objetivo é avaliar o desempenho do modelo em dados que ele nunca viu antes, simulando como ele se comportaria no mundo real.
- Treinar e testar no mesmo conjunto pode gerar resultados enganosos (overfitting), pois o modelo "decoraria" os dados.

**Por que 80% para treino e 20% para teste?**
- Essa proporção é uma prática comum e recomendada, pois garante que o modelo tenha dados suficientes para aprender (80%) e também uma quantidade relevante para ser avaliado (20%).
- Proporções muito pequenas para teste podem não representar bem a realidade, enquanto proporções muito grandes para teste deixam poucos dados para o modelo aprender.
- Dependendo do tamanho do dataset, essa proporção pode ser ajustada, mas 80/20 é um ponto de partida seguro e amplamente utilizado.

> **Resumo:** Separar os dados dessa forma é considerado uma boa prática em projetos de ciência de dados e aprendizado de máquina, ajudando a garantir resultados mais confiáveis e generalizáveis.
''')
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    col1, col2 = st.columns(2)
    with col1:
        st.metric('Linhas de treino', train_df.shape[0])
    with col2:
        st.metric('Linhas de teste', test_df.shape[0])
    return train_df, test_df
