import streamlit as st
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


def render_xgboost(train_df, test_df):
    st.header('Classificação: XGBoost')
    st.markdown('''
**O que é XGBoost?**

XGBoost (Extreme Gradient Boosting) é um dos algoritmos de aprendizado de máquina mais poderosos e populares para tarefas de classificação e regressão. Ele utiliza o método de boosting, combinando várias árvores de decisão fracas para formar um modelo forte, otimizando o desempenho e evitando overfitting.

- Muito eficiente e rápido, mesmo para grandes volumes de dados.
- Oferece regularização para evitar overfitting.
- Venceu várias competições de ciência de dados.
''')

    st.subheader('Configuração do Modelo')
    n_estimators = st.slider('Número de árvores (n_estimators):', min_value=10, max_value=300, value=100, step=10, key='xgb_n_estimators')
    learning_rate = st.slider('Taxa de aprendizado (learning_rate):', min_value=0.01, max_value=0.5, value=0.1, step=0.01, key='xgb_learning_rate')
    target = 'default.payment.next.month'
    features = [col for col in train_df.columns if col not in ['ID', target]]
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    if st.button('Treinar Modelo XGBoost', key='xgb_train_button'):
        modelo = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, use_label_encoder=False, eval_metric='logloss', random_state=42)
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
        RocCurveDisplay.from_predictions(y_test, y_score, ax=ax_roc, name='XGBoost')
        st.pyplot(fig_roc)
        st.caption('AUC-ROC mede a capacidade do modelo em distinguir entre as classes. Quanto mais próximo de 1, melhor.')
