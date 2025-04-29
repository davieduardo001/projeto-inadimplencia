import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def render_pre_processamento(df):
    st.header('Pré-processamento')
    st.markdown('''
O pré-processamento é uma etapa fundamental para garantir a qualidade dos dados antes de aplicar modelos de machine learning. Aqui, realizamos três operações principais:

**1. Tratamento de outliers (valores extremos):**
Outliers são valores muito distantes do padrão dos dados, que podem distorcer estatísticas e prejudicar o desempenho dos modelos. Por exemplo, um cliente com limite de crédito absurdamente alto em relação aos demais. 

No exemplo abaixo, mostramos como remover outliers considerando apenas a variável `LIMIT_BAL` (limite de crédito), mas o mesmo raciocínio pode ser aplicado a outras variáveis numéricas do seu dataset. O critério utilizado é excluir registros cujo valor está além de 3 desvios padrão da média.

**2. Codificação de variáveis categóricas:**
Algumas variáveis possuem valores em formato de texto ou categorias (como SEXO, ESCOLARIDADE, ESTADO CIVIL). Para que os modelos matemáticos possam utilizá-las, transformamos essas categorias em números usando técnicas como one-hot encoding.

**3. Escalonamento (StandardScaler ou MinMaxScaler):**
Modelos de machine learning geralmente se beneficiam quando os dados numéricos estão na mesma escala. Por isso, aplicamos o escalonamento para padronizar (média 0, desvio padrão 1) ou normalizar (valores entre 0 e 1) os dados, conforme a escolha do usuário.

---
**O que foi removido ou alterado?**
- Registros com valores extremos em `LIMIT_BAL` (outliers) foram excluídos (exemplo).
- Variáveis categóricas foram convertidas para formato numérico.
- Colunas numéricas foram escalonadas para facilitar o aprendizado dos modelos.

Essas etapas aumentam a robustez e a performance dos modelos, além de reduzir o risco de interpretações erradas causadas por dados inconsistentes ou escalas diferentes.
''')

    st.subheader('1. Tratamento de outliers (se necessário)')
    st.info('Exemplo: Remoção de outliers apenas na variável LIMIT_BAL. Para um tratamento mais completo, avalie outras variáveis numéricas do seu dataset.')
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
    st.markdown('''
**Entendendo os dados pós-escalonamento:**

Após o escalonamento, os valores das variáveis numéricas mudam de escala:
- Com o StandardScaler, os dados ficam com média 0 e desvio padrão 1 (valores podem ser negativos e positivos, centrados em torno de zero).
- Com o MinMaxScaler, os dados ficam entre 0 e 1 (ou muito próximos disso).

Isso não altera a informação do dado, apenas coloca todas as variáveis na mesma base numérica, facilitando o aprendizado dos modelos. Não se preocupe se os números parecerem diferentes do original — é esperado!
''')
    st.write('Dados após escalonamento:')
    st.dataframe(df_proc[cols_num].head())

    st.success('Pré-processamento concluído!')
