# Projeto de Previsão de Inadimplência em Cartões de Crédito

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-orange)](https://streamlit.io)

## Objetivo
Prever a probabilidade de um cliente inadimplir (não pagar) no próximo mês, com base em dados históricos de pagamentos.

## Dataset
**Fonte**: [Default of Credit Card Clients](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset) no Kaggle

## Variáveis do Dataset

| Variável | Descrição | Valores/Formato |
|----------|-----------|-----------------|
| ID | Identificação do cliente | Número único |
| LIMIT_BAL | Limite de crédito | NT$ (inclui crédito individual e familiar) |
| SEX | Gênero | 1=masculino, 2=feminino |
| EDUCATION | Escolaridade | 1=pós-graduação, 2=universitário, 3=ensino médio, 4=outros, 5-6=desconhecido |
| MARRIAGE | Estado civil | 1=casado, 2=solteiro, 3=outros |
| AGE | Idade | Anos completos |
| PAY_0 a PAY_6 | Status de pagamento (-1 a 9) | -1=pago em dia, 1=atraso 1 mês, ..., 9=atraso ≥9 meses |
| BILL_AMT1 a BILL_AMT6 | Valor da fatura | NT$ (últimos 6 meses) |
| PAY_AMT1 a PAY_AMT6 | Valor pago | NT$ (últimos 6 meses) |
| default.payment.next.month | Inadimplência | 0=não, 1=sim |

*Observação: PAY_0 refere-se a setembro/2005, PAY_1 a agosto/2005, ..., PAY_6 a abril/2005*

## Métodos
- **Pré-processamento**:
  - Limpeza de dados
  - Análise exploratória (EDA)
  - Feature engineering

- **Modelos**:
  - Aprendizado supervisionado (classificação)
  - Comparação de algoritmos

## Métricas
- Matriz de confusão
- AUC-ROC
- Precision/Recall

## Como Usar
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Saídas
- Notebook Jupyter com análise completa
- Dashboard interativo no Streamlit

![Exemplo de Visualização](image.png)

- normalizacao eh feita automatica