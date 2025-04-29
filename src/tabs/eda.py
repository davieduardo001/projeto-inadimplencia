import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def render_eda(df):
    st.header('Análise Exploratória de Dados (EDA)')
    st.markdown('''
A **Análise Exploratória de Dados (EDA)** é o processo de investigar, visualizar e resumir os dados antes de aplicar modelos de machine learning. Aqui você pode explorar distribuições, relações e padrões nos dados de forma interativa.

**Descrição das principais variáveis:**
- **SEX:** Gênero do cliente (1 = masculino, 2 = feminino)
- **EDUCATION:** Escolaridade (1 = pós-graduação, 2 = universitário, 3 = ensino médio, 4 = outros, 5-6 = desconhecido)
- **MARRIAGE:** Estado civil (1 = casado, 2 = solteiro, 3 = outros)
- **AGE:** Idade em anos
- **PAY_0 a PAY_6:** Status de pagamento dos últimos meses (-1 = pago em dia, 1 = atraso 1 mês, ..., 9 = atraso ≥9 meses)
- **BILL_AMT1 a BILL_AMT6:** Valor da fatura dos últimos 6 meses
- **PAY_AMT1 a PAY_AMT6:** Valor pago nos últimos 6 meses
- **default.payment.next.month:** Inadimplência no próximo mês (0 = não, 1 = sim)
''')

    st.subheader('Estatísticas Descritivas')
    st.write(df.describe())

    st.subheader('Distribuição de Variáveis Categóricas')
    # SEX
    st.markdown('**Distribuição por Gênero (SEX):** 1 = masculino, 2 = feminino')
    fig_sex, ax_sex = plt.subplots()
    sns.countplot(x='SEX', data=df, ax=ax_sex)
    ax_sex.set_xticklabels(['Masculino', 'Feminino'])
    st.pyplot(fig_sex)
    
    # EDUCATION
    st.markdown('**Distribuição por Escolaridade (EDUCATION):**')
    fig_edu, ax_edu = plt.subplots()
    order_edu = [1,2,3,4,5,6]
    labels_edu = ['Pós-graduação', 'Universitário', 'Ensino médio', 'Outros', 'Desconhecido(5)', 'Desconhecido(6)']
    sns.countplot(x='EDUCATION', data=df, order=order_edu, ax=ax_edu)
    ax_edu.set_xticklabels(labels_edu, rotation=15)
    st.pyplot(fig_edu)
    
    # MARRIAGE
    st.markdown('**Distribuição por Estado Civil (MARRIAGE):** 1 = casado, 2 = solteiro, 3 = outros')
    fig_mar, ax_mar = plt.subplots()
    order_mar = [1,2,3]
    labels_mar = ['Casado', 'Solteiro', 'Outros']
    sns.countplot(x='MARRIAGE', data=df, order=order_mar, ax=ax_mar)
    ax_mar.set_xticklabels(labels_mar)
    st.pyplot(fig_mar)

    st.subheader('Distribuição de Variáveis Numéricas')
    col = st.selectbox('Selecione uma variável numérica:', df.select_dtypes(include='number').columns, key='eda_col')
    tipo_grafico = st.radio('Tipo de gráfico:', ['Histograma', 'Boxplot'], key='eda_grafico')
    fig, ax = plt.subplots()
    if tipo_grafico == 'Histograma':
        sns.histplot(df[col], kde=True, ax=ax)
    else:
        sns.boxplot(x=df[col], ax=ax)
    st.pyplot(fig)

    st.subheader('Correlação entre Variáveis')
    if st.checkbox('Mostrar matriz de correlação'):
        fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
        corr = df.corr()
        sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax_corr)
        st.pyplot(fig_corr)
        st.caption('A matriz de correlação mostra o grau de relação linear entre as variáveis numéricas.')

    st.subheader('Relação com Inadimplência')
    target = 'default.payment.next.month'
    if target in df.columns:
        col_cat = st.selectbox('Selecione uma variável categórica para analisar relação com inadimplência:',
                              [c for c in df.columns if c not in ['ID', target] and str(df[c].dtype) in ['int64', 'object']],
                              key='eda_cat')
        if col_cat:
            st.write(f'Distribuição de {col_cat} por inadimplência:')
            fig_cat, ax_cat = plt.subplots()
            # Mapeamento para o eixo y (inadimplência)
            labels_default = ['Não inadimplente', 'Inadimplente']
            sns.countplot(data=df, x=col_cat, hue=target, ax=ax_cat)
            # Substitui as legendas numéricas por texto na legenda do gráfico
            handles, labels = ax_cat.get_legend_handles_labels()
            ax_cat.legend(handles, labels_default, title='Inadimplência')
            st.pyplot(fig_cat)
            st.caption('Veja como a variável selecionada se relaciona com a inadimplência. (0 = Não inadimplente, 1 = Inadimplente)')
