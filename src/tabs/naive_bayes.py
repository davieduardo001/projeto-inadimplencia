import streamlit as st
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, RocCurveDisplay, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix_explained(cm, labels):
    fig, ax = plt.subplots(figsize=(4,4))
    # Mostrar matriz com cmap
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # Títulos
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predito', fontsize=12)
    ax.set_ylabel('Real', fontsize=12)
    ax.set_title('Matriz de Confusão', fontsize=14, pad=15)
    # Escrever valores absolutos e percentuais
    total = cm.sum()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            percent = cm[i, j] / total * 100 if total > 0 else 0
            ax.text(j, i, f'{cm[i, j]}\n({percent:.1f}%)',
                    ha='center', va='center', color='black', fontsize=12, fontweight='bold')
    # Explicação dos quadrantes
    ax.text(-0.6, -0.6, 'Verdadeiro Negativo', color='gray', fontsize=10)
    ax.text(1.6, -0.6, 'Falso Positivo', color='gray', fontsize=10)
    ax.text(-0.6, 1.6, 'Falso Negativo', color='gray', fontsize=10)
    ax.text(1.6, 1.6, 'Verdadeiro Positivo', color='gray', fontsize=10)
    fig.tight_layout()
    return fig

def render_naive_bayes(train_df, test_df):
    st.header('Classificação: Naive Bayes')
    st.markdown('''
**O que é Naive Bayes?**

Naive Bayes é uma família de algoritmos probabilísticos baseada no Teorema de Bayes, assumindo independência entre as variáveis preditoras. É simples, rápido e funciona bem em muitos cenários de classificação, especialmente quando as features são aproximadamente independentes.

- Baseia-se na probabilidade condicional de cada classe dado os atributos.
- Muito utilizado em problemas de texto e classificação binária.
- Vantagens: rápido, robusto a dados ruidosos, fácil de interpretar.

> Embora simples, pode servir como um ótimo baseline para comparação!
''')

    target = 'default.payment.next.month'
    features = [col for col in train_df.columns if col not in ['ID', target]]
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    if st.button('Treinar Modelo Naive Bayes'):
        modelo = GaussianNB()
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.success(f'Acurácia no conjunto de teste: {acc:.2%}')

        # 1. Matriz de Confusão Explicativa
        st.subheader('Matriz de Confusão')
        cm = confusion_matrix(y_test, y_pred)
        labels = ['Não inadimplente', 'Inadimplente']
        fig_cm = plot_confusion_matrix_explained(cm, labels)
        st.pyplot(fig_cm)
        st.caption('Cada célula mostra o número absoluto e percentual das previsões. Quadrantes: VN (superior-esquerda), FP (superior-direita), FN (inferior-esquerda), VP (inferior-direita).')

        # 2. Precision e Recall
        st.subheader('Precision e Recall')
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        st.write(f'**Precisão (Precision):** {precision:.2%}')
        st.write(f'**Recall (Sensibilidade):** {recall:.2%}')
        st.caption('Precisão: entre todas as previsões positivas, quantas realmente eram positivas.\nRecall: entre todas as amostras realmente positivas, quantas o modelo identificou corretamente.')

        # 3. Curva ROC e AUC
        st.subheader('Curva ROC e AUC')
        if hasattr(modelo, "predict_proba"):
            y_score = modelo.predict_proba(X_test)[:, 1]
        else:
            y_score = y_pred
        auc = roc_auc_score(y_test, y_score)
        st.write(f'**AUC-ROC:** {auc:.3f}')
        fig_roc, ax_roc = plt.subplots()
        RocCurveDisplay.from_predictions(y_test, y_score, ax=ax_roc, name='Naive Bayes')
        st.pyplot(fig_roc)
        st.caption('AUC-ROC mede a capacidade do modelo em distinguir entre as classes. Quanto mais próximo de 1, melhor.')
