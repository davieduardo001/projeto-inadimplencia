import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Renomeado para pré-processamento e expandido conforme README

def render_pre_processamento(df):
    st.header('Pré-processamento')
    st.markdown('''
O pré-processamento é uma etapa fundamental para preparar os dados antes da modelagem. Inclui tratamento de outliers, codificação de variáveis categóricas e escalonamento dos dados.

**Tópicos abordados:**
- ◦ Tratamento de outliers (se necessário)
- ◦ Codificação de variáveis categóricas
- ◦ Escalonamento (StandardScaler ou MinMaxScaler)
''')

    st.subheader('1. Tratamento de outliers (se necessário)')
    st.info('Esta etapa pode incluir a remoção ou ajuste de valores extremos. (Exemplo ilustrativo, ajuste conforme necessário para seu dataset)')
    # Exemplo: Remover outliers em LIMIT_BAL (3 desvios padrão)
    df_proc = df.copy()
    lim = 3 * df_proc['LIMIT_BAL'].std()
    media = df_proc['LIMIT_BAL'].mean()
    df_proc = df_proc[(df_proc['LIMIT_BAL'] >= media - lim) & (df_proc['LIMIT_BAL'] <= media + lim)]
    st.write(f'Após remoção de outliers em LIMIT_BAL: {df_proc.shape[0]} linhas restantes.')

    st.subheader('2. Codificação de variáveis categóricas')
    st.info('Transforma variáveis categóricas em formato numérico para uso em modelos.')
    df_proc['SEX'] = df_proc['SEX'].replace({1: 0, 2: 1})  # Exemplo: 0=masculino, 1=feminino
    df_proc = pd.get_dummies(df_proc, columns=['EDUCATION', 'MARRIAGE'], drop_first=True)
    st.write('Exemplo de dados após codificação:')
    st.dataframe(df_proc.head())

    st.subheader('3. Escalonamento (StandardScaler ou MinMaxScaler)')
    st.info('Padroniza ou normaliza variáveis numéricas para melhorar o desempenho dos modelos.')
    scaler = st.radio('Escolha o tipo de escalonamento:', ['StandardScaler', 'MinMaxScaler'])
    cols_num = ['LIMIT_BAL', 'AGE']
    if scaler == 'StandardScaler':
        scaled = StandardScaler().fit_transform(df_proc[cols_num])
    else:
        scaled = MinMaxScaler().fit_transform(df_proc[cols_num])
    df_proc[cols_num] = scaled
    st.write('Dados após escalonamento:')
    st.dataframe(df_proc[cols_num].head())

    st.success('Pré-processamento concluído!')
