import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay, precision_score, recall_score
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

st.title('Análise de Dados - UCI Credit Card')

# Carregar os dados
data_path = 'src/data/UCI_Credit_Card.csv'
@st.cache_data
def load_data():
    return pd.read_csv(data_path)

df_raw = load_data()

# --------------------
# DASHBOARD COM ABAS
# --------------------
tabs = st.tabs([
    ' Dados Originais',
    ' Limpeza',
    ' Treino/Teste',
    ' Classificação Supervisionada',
    ' Visualização Gráfica',
    ' Documentação'
])

# --- Aba 1: Dados Originais ---
with tabs[0]:
    st.header('Visualização dos Dados Originais')
    st.write('Primeiras linhas do dataset original:')
    st.dataframe(df_raw.head())
    st.subheader('Estatísticas Descritivas (Original)')
    st.write(df_raw.describe())

# --- Aba 2: Limpeza ---
with tabs[1]:
    st.header('Limpeza de Dados')
    st.markdown('''
**Processo de limpeza realizado:**
- Remoção de linhas com valores ausentes (missing values).
- Ajuste de valores inconsistentes nas variáveis categóricas:
    - `EDUCATION`: valores diferentes de 1, 2, 3 ou 4 foram substituídos por 4 ("outros/desconhecido").
    - `MARRIAGE`: valores diferentes de 1, 2 ou 3 foram substituídos por 3 ("outros").
''')

    def clean_data(df):
        df = df.copy()
        # Remover linhas com valores ausentes
        df = df.dropna()
        # Corrigir EDUCATION
        df['EDUCATION'] = df['EDUCATION'].apply(lambda x: x if x in [1,2,3,4] else 4)
        # Corrigir MARRIAGE
        df['MARRIAGE'] = df['MARRIAGE'].apply(lambda x: x if x in [1,2,3] else 3)
        return df

    df_clean = clean_data(df_raw)

    st.write('Após limpeza:')
    st.dataframe(df_clean.head())
    st.subheader('Estatísticas Descritivas (Após Limpeza)')
    st.write(df_clean.describe())
    st.info(f'Foram removidas {len(df_raw) - len(df_clean)} linhas com valores ausentes e corrigidos valores inconsistentes nas colunas EDUCATION e MARRIAGE.')

# --- Aba 3: Treino/Teste ---
with tabs[2]:
    st.header('Divisão dos Dados em Treino e Teste')
    st.markdown('''
Os dados foram divididos em 80% para treino e 20% para teste, garantindo avaliação em dados nunca vistos durante o treinamento.
''')
    train_df, test_df = train_test_split(df_clean, test_size=0.2, random_state=42)
    col1, col2 = st.columns(2)
    with col1:
        st.metric('Linhas de treino', train_df.shape[0])
    with col2:
        st.metric('Linhas de teste', test_df.shape[0])

# --- Aba 4: Modelagem ---
with tabs[3]:
    st.header('Classificação Supervisionada')
    st.markdown('''
Escolha o algoritmo de classificação e o tipo de avaliação para prever a coluna de inadimplência (`default.payment.next.month`).
''')
    algoritmo = st.selectbox('Escolha o algoritmo:', ['Random Forest', 'KNN', 'Regressão Logística'], key='alg')
    tipo_avaliacao = st.radio('Tipo de avaliação:', ['Acurácia', 'Precision e Recall', 'AUC-ROC'], key='tipo_avaliacao')
    # Definir X e y
    target = 'default.payment.next.month'
    features = [col for col in train_df.columns if col not in ['ID', target]]
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]
    if algoritmo == 'Random Forest':
        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    elif algoritmo == 'KNN':
        n_vizinhos = st.slider('Número de vizinhos (K):', min_value=1, max_value=20, value=5, key='knn_k')
        modelo = KNeighborsClassifier(n_neighbors=n_vizinhos)
    elif algoritmo == 'Regressão Logística':
        modelo = LogisticRegression(max_iter=500, random_state=42)
    # Avaliação
    def avaliar_com_acuracia(modelo, X_train, y_train, X_test, y_test, nome_modelo):
        """
        Treina e avalia o modelo exibindo apenas a acurácia e a matriz de confusão.
        """
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.subheader(f'Acurácia no conjunto de teste ({nome_modelo}): **{acc:.2%}**')
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        ConfusionMatrixDisplay(cm, display_labels=['Não inadimplente', 'Inadimplente']).plot(ax=ax_cm, cmap='Blues')
        ax_cm.set_xlabel('Classe prevista (Predição)')
        ax_cm.set_ylabel('Classe real (Verdadeiro)')
        ax_cm.set_xticklabels(['Previsto: Não inadimplente', 'Previsto: Inadimplente'])
        ax_cm.set_yticklabels(['Verdadeiro: Não inadimplente', 'Verdadeiro: Inadimplente'])
        st.pyplot(fig_cm)
        st.caption(f'Matriz de confusão para o modelo {nome_modelo}.')

    def avaliar_com_precision_recall(modelo, X_train, y_train, X_test, y_test, nome_modelo):
        """
        Treina e avalia o modelo exibindo apenas precisão, recall e a matriz de confusão.
        """
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        st.write(f'Precisão (Precision): **{precision:.2%}**')
        st.write(f'Recall (Sensibilidade): **{recall:.2%}**')
        st.caption('''
**Precisão (Precision):** entre todas as previsões positivas do modelo, quantas realmente eram positivas.
**Recall (Sensibilidade):** entre todas as amostras realmente positivas, quantas o modelo identificou corretamente.
''')
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        ConfusionMatrixDisplay(cm, display_labels=['Não inadimplente', 'Inadimplente']).plot(ax=ax_cm, cmap='Blues')
        ax_cm.set_xlabel('Classe prevista (Predição)')
        ax_cm.set_ylabel('Classe real (Verdadeiro)')
        ax_cm.set_xticklabels(['Previsto: Não inadimplente', 'Previsto: Inadimplente'])
        ax_cm.set_yticklabels(['Verdadeiro: Não inadimplente', 'Verdadeiro: Inadimplente'])
        st.pyplot(fig_cm)
        st.caption(f'Matriz de confusão para o modelo {nome_modelo}.')

    def avaliar_com_auc_roc(modelo, X_train, y_train, X_test, y_test, nome_modelo):
        """
        Treina e avalia o modelo exibindo apenas o AUC-ROC, curva ROC e matriz de confusão.
        """
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        # Matriz de confusão
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        ConfusionMatrixDisplay(cm, display_labels=['Não inadimplente', 'Inadimplente']).plot(ax=ax_cm, cmap='Blues')
        ax_cm.set_xlabel('Classe prevista (Predição)')
        ax_cm.set_ylabel('Classe real (Verdadeiro)')
        ax_cm.set_xticklabels(['Previsto: Não inadimplente', 'Previsto: Inadimplente'])
        ax_cm.set_yticklabels(['Verdadeiro: Não inadimplente', 'Verdadeiro: Inadimplente'])
        st.pyplot(fig_cm)
        st.caption(f'Matriz de confusão para o modelo {nome_modelo}.')
        # Cálculo e gráfico da curva ROC/AUC
        st.markdown('**Curva ROC e AUC:**')
        if hasattr(modelo, "predict_proba"):
            y_score = modelo.predict_proba(X_test)[:, 1]
        elif hasattr(modelo, "decision_function"):
            y_score = modelo.decision_function(X_test)
        else:
            y_score = y_pred  # fallback
        try:
            auc = roc_auc_score(y_test, y_score)
            st.write(f'AUC-ROC: **{auc:.3f}**')
            fig_roc, ax_roc = plt.subplots()
            RocCurveDisplay.from_predictions(y_test, y_score, ax=ax_roc, name=nome_modelo)
            st.pyplot(fig_roc)
            st.caption('''
**AUC-ROC:** Mede a capacidade do modelo em distinguir entre as classes. Quanto mais próximo de 1, melhor o desempenho.
''')
        except Exception as e:
            st.warning(f'Não foi possível calcular a curva ROC/AUC: {e}')

    if tipo_avaliacao == 'Acurácia':
        avaliar_com_acuracia(modelo, X_train, y_train, X_test, y_test, algoritmo)
    elif tipo_avaliacao == 'Precision e Recall':
        avaliar_com_precision_recall(modelo, X_train, y_train, X_test, y_test, algoritmo)
    elif tipo_avaliacao == 'AUC-ROC':
        avaliar_com_auc_roc(modelo, X_train, y_train, X_test, y_test, algoritmo)
    st.markdown('''\
**Explicação dos algoritmos:**
- **Random Forest:** Conjunto de várias árvores de decisão, robusto a overfitting e bom para dados tabulares.
- **KNN (K-Nearest Neighbors):** Classifica uma amostra com base nos vizinhos mais próximos no espaço de atributos.
- **Regressão Logística:** Modelo linear para classificação binária, útil para interpretar probabilidades.
''')

# --- Aba 5: Visualização Gráfica ---
with tabs[4]:
    st.header('Visualização Gráfica')
    st.markdown('''
Selecione uma coluna numérica e o tipo de gráfico para explorar a distribuição dos dados.
''')
    col = st.selectbox('Selecione a coluna para visualizar:', df_clean.select_dtypes(include='number').columns, key='col_viz')
    grafico = st.radio('Tipo de gráfico:', ['Histograma', 'Boxplot'], key='grafico_viz')
    fig, ax = plt.subplots()
    if grafico == 'Histograma':
        sns.histplot(df_clean[col], kde=True, ax=ax)
    else:
        sns.boxplot(x=df_clean[col], ax=ax)
    st.pyplot(fig)

# --- Aba 6: Documentação ---
with tabs[5]:
    st.header('Documentação do Aplicativo')
    st.caption('''\
**Aplicativo interativo para previsão de inadimplência em cartões de crédito.**

Este dashboard foi desenvolvido para demonstrar, de forma didática e visual, todas as etapas de um projeto real de Ciência de Dados aplicado ao problema de inadimplência, utilizando dados públicos da UCI.

- Pipeline completo: visualização, limpeza, divisão treino/teste, classificação (com escolha de algoritmo e métrica), visualização gráfica e documentação.
- Interface organizada em abas, facilitando a navegação e evitando o scroll infinito.
- Todo o código está comentado e explicado em português.
- Pode ser adaptado para outros datasets tabulares e outros problemas de classificação.
''')