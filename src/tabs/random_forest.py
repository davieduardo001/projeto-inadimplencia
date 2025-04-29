import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


def render_random_forest(train_df, test_df):
    st.header('Classificação: Random Forest')
    st.markdown('''
**O que é Random Forest?**

Random Forest ("Floresta Aleatória") é um algoritmo de aprendizado supervisionado muito utilizado para tarefas de classificação e regressão. Ele funciona criando várias árvores de decisão independentes e combinando seus resultados para obter uma predição mais robusta e precisa.

- Cada árvore aprende de forma diferente, usando subconjuntos aleatórios dos dados e das variáveis.
- A predição final é feita por votação (no caso de classificação) ou média (no caso de regressão).
- Vantagens: geralmente apresenta alta acurácia, é robusto a overfitting e lida bem com dados complexos e não-lineares.

> Random Forest é considerado um dos algoritmos mais poderosos e versáteis em Machine Learning!
''')

    st.subheader('Configuração do Modelo')
    n_estimators = st.slider('Número de árvores (n_estimators):', min_value=10, max_value=300, value=100, step=10)
    target = 'default.payment.next.month'
    features = [col for col in train_df.columns if col not in ['ID', target]]
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    if st.button('Treinar Modelo Random Forest'):
        modelo = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.success(f'Acurácia no conjunto de teste: {acc:.2%}')

        # 1. Matriz de Confusão
        st.subheader('Matriz de Confusão')
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        ConfusionMatrixDisplay(cm, display_labels=['Não inadimplente', 'Inadimplente']).plot(ax=ax_cm, cmap='Blues')
        st.pyplot(fig_cm)
        st.caption('A matriz de confusão mostra como o modelo está errando e acertando as classes.')

        # 2. Precision, Recall, F1-score
        st.subheader('Precision, Recall e F1-score')
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        st.write(f'**Precisão (Precision):** {precision:.2%}')
        st.write(f'**Recall (Sensibilidade):** {recall:.2%}')
        st.write(f'**F1-score:** {f1:.2%}')
        st.caption('Precisão: entre todas as previsões positivas, quantas realmente eram positivas.\nRecall: entre todas as amostras realmente positivas, quantas o modelo identificou corretamente.\nF1-score: média harmônica entre precisão e recall, útil para datasets desbalanceados.')

        # 3. Curva ROC e AUC
        st.subheader('Curva ROC e AUC')
        if hasattr(modelo, "predict_proba"):
            y_score = modelo.predict_proba(X_test)[:, 1]
        else:
            y_score = y_pred
        auc = roc_auc_score(y_test, y_score)
        st.write(f'**AUC-ROC:** {auc:.3f}')
        fig_roc, ax_roc = plt.subplots()
        RocCurveDisplay.from_predictions(y_test, y_score, ax=ax_roc, name='Random Forest')
        st.pyplot(fig_roc)
        st.caption('AUC-ROC mede a capacidade do modelo em distinguir entre as classes. Quanto mais próximo de 1, melhor.')
