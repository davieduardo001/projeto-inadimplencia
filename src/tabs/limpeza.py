import streamlit as st
import pandas as pd

def render_limpeza(df):
    st.header('Limpeza de Dados e Feature Engineering')
    st.markdown('''
**Processo de limpeza realizado:**
- Remoção de linhas com valores ausentes (missing values).
- Ajuste de valores inconsistentes nas variáveis categóricas:
    - `EDUCATION`: valores diferentes de 1, 2, 3 ou 4 foram substituídos por 4 ("outros/desconhecido").
    - `MARRIAGE`: valores diferentes de 1, 2 ou 3 foram substituídos por 3 ("outros").

**Feature Engineering (Engenharia de Atributos) aplicada:**
- Criação da variável `FAIXA_IDADE` que categoriza os clientes em grupos de idade: 'Jovem' (até 29 anos), 'Adulto' (30-59), 'Idoso' (60+).
- Criação da variável `PAGOU_TUDO` que indica se o cliente pagou o valor total da fatura nos últimos 6 meses.

Essas transformações ajudam o modelo a identificar padrões mais facilmente e podem melhorar o desempenho da classificação.
''')

    def clean_data(df):
        df = df.copy()
        # Remover linhas com valores ausentes
        df = df.dropna()
        # Corrigir EDUCATION
        df['EDUCATION'] = df['EDUCATION'].apply(lambda x: x if x in [1,2,3,4] else 4)
        # Corrigir MARRIAGE
        df['MARRIAGE'] = df['MARRIAGE'].apply(lambda x: x if x in [1,2,3] else 3)
        # Feature Engineering: criar faixa de idade
        df['FAIXA_IDADE'] = pd.cut(df['AGE'], bins=[0,29,59,150], labels=['Jovem','Adulto','Idoso'])
        # Feature Engineering: cliente pagou tudo nos últimos 6 meses?
        bill_cols = [f'BILL_AMT{i}' for i in range(1,7)]
        pay_cols = [f'PAY_AMT{i}' for i in range(1,7)]
        df['PAGOU_TUDO'] = (df[pay_cols].sum(axis=1) >= df[bill_cols].sum(axis=1)).astype(int)
        return df

    df_clean = clean_data(df)

    st.write('Após limpeza e feature engineering:')
    st.dataframe(df_clean.head())
    st.subheader('Estatísticas Descritivas (Após Limpeza)')
    st.write(df_clean.describe())
    st.info(f'Foram removidas {len(df) - len(df_clean)} linhas com valores ausentes e corrigidos valores inconsistentes nas colunas EDUCATION e MARRIAGE.')
    st.markdown('''
**Explicação das novas variáveis:**
- `FAIXA_IDADE`: Agrupa clientes por faixa etária, facilitando a análise de comportamento por idade.
- `PAGOU_TUDO`: Indica (1 = sim, 0 = não) se o cliente pagou o valor total das faturas recentes, o que pode ser útil para prever inadimplência.
''')
