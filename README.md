# Projeto Acadêmico: Previsão de Inadimplência em Cartões de Crédito

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)  
[![Streamlit](https://img.shields.io/badge/Streamlit-App-orange)](https://streamlit.io)

---

## 1. Introdução

Este projeto visa o desenvolvimento de uma aplicação interativa em Python, utilizando a biblioteca Streamlit, para análise e comparação de modelos de aprendizado de máquina aplicados à previsão de inadimplência em cartões de crédito. A aplicação tem como objetivo didático de finalizar a atividade de aprendizado de máquina, para que os estudantes explorarem conceitos fundamentais de análise exploratória de dados (EDA), pré-processamento, modelagem preditiva e avaliação de desempenho.

---

## 2. Objetivos

- Explorar estatísticas e padrões do conjunto de dados de inadimplência de clientes de cartão de crédito.
- Implementar diferentes algoritmos de classificação.
- Comparar os desempenhos dos modelos com base em métricas específicas.
- Disponibilizar uma aplicação interativa para fins educacionais.

---

## 3. Dataset

### Origem

A **Análise de Inadimplência - UCI Credit Card** refere-se a um estudo baseado no conjunto de dados **[Default of Credit Card Clients](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)**, disponibilizado pelo **UCI Machine Learning Repository**.

### Descrição

O dataset contém informações financeiras e socioeconômicas de **30 mil clientes** de cartões de crédito, incluindo:

- Limite de crédito
- Idade
- Histórico de pagamentos (últimos 6 meses)
- Valores faturados mensalmente
- Valores pagos mensalmente
- Dados demográficos (sexo, escolaridade, estado civil)

A variável-alvo (`default.payment.next.month`) indica se o cliente entrou ou não em inadimplência no mês seguinte.

### Estrutura das Variáveis

| Variável | Descrição |
|----------|-----------|
| `LIMIT_BAL` | Limite de crédito concedido (NT$) |
| `SEX`, `EDUCATION`, `MARRIAGE` | Variáveis demográficas codificadas |
| `AGE` | Idade do cliente |
| `PAY_0` a `PAY_6` | Histórico de pagamento (últimos 6 meses) |
| `BILL_AMT1` a `BILL_AMT6` | Valor das faturas mensais |
| `PAY_AMT1` a `PAY_AMT6` | Valor pago nos meses correspondentes |
| `default.payment.next.month` | Variável-alvo (0 = adimplente, 1 = inadimplente) |

---

## 4. Metodologia

### 4.1 Análise Exploratória de Dados (EDA)

#### Principais Insights do EDA

1. **Limite de Crédito vs Inadimplência**  
Correlação negativa com a inadimplência: limites mais altos tendem a reduzir o risco.

2. **Idade e Pontualidade**  
Clientes mais velhos costumam ser mais pontuais nos pagamentos.

3. **Consistência nos Gastos**  
Alta correlação entre os valores de fatura dos meses, indicando padrão estável de consumo.

4. **Educação como Indicador Secundário**  
Leve correlação com inadimplência, mas pouco significativa isoladamente.

5. **Histórico de Pagamento**  
Atrasos são fortemente preditivos de futuros atrasos.

6. **Pagamentos Realizados**  
Fraca correlação com a inadimplência, sugerindo menor valor preditivo.

### 4.2 Pré-processamento

- Divisão dos dados em treino e teste (80/20)
- Seleção de features (excluindo ID e target)

### 4.3 Modelagem Preditiva

Modelos aplicados:

| Modelo | Descrição | Vantagens |
|--------|-----------|-----------|
| **Regressão Logística** | Modelo estatístico linear | Simples, interpretável |
| **Random Forest** | Conjunto de árvores de decisão | Robusto, bom para não-linearidades |
| **KNN** | Classificação baseada em vizinhos | Fácil de implementar |
| **XGBoost** | Técnica de boosting com regularização | Alta performance e escalabilidade |

---

## 5. Avaliação dos Modelos

### 5.1 Métricas de Desempenho

1. **Acurácia**
   - Proporção de previsões corretas em relação ao total
   - Não é suficiente para datasets desbalanceados
   - Calculada como: (TP + TN) / (TP + TN + FP + FN)

2. **Precisão**
   - Proporção de previsões positivas corretas
   - Importante para minimizar falsos positivos
   - Calculada como: TP / (TP + FP)

3. **Recall (Sensibilidade)**
   - Proporção de positivos reais identificados
   - Importante para identificar todos os casos positivos
   - Calculada como: TP / (TP + FN)

4. **F1-Score**
   - Média harmônica entre Precisão e Recall
   - Balanceia os dois aspectos
   - Calculada como: 2 * (Precision * Recall) / (Precision + Recall)

### 5.2 Visualizações de Desempenho

1. **Matriz de Confusão**
   - Mostra os acertos e erros do modelo
   - Indica:
     - Verdadeiros Positivos (TP)
     - Verdadeiros Negativos (TN)
     - Falsos Positivos (FP)
     - Falsos Negativos (FN)

2. **Curva ROC e AUC**
   - Avalia a capacidade do modelo em distinguir classes
   - ROC: Gráfico de True Positive Rate vs False Positive Rate
   - AUC: Área sob a curva ROC
   - Valores próximos de 1 indicam melhor desempenho

### 5.3 Importância das Métricas no Contexto

- **Precisão**: Importante para evitar aprovação de clientes inadimplentes
- **Recall**: Crítico para identificar o máximo possível de inadimplentes
- **F1-Score**: Balanceia precisão e recall, útil para datasets desbalanceados
- **AUC-ROC**: Indica capacidade geral do modelo em distinguir classes

---

## 6. Conclusões

Claro! Aqui está uma versão revisada e mais acadêmica das **conclusões** do seu projeto:

---

## 6. Conclusões

A análise realizada evidencia padrões relevantes de comportamento de crédito, com implicações diretas para estratégias de mitigação de risco. Os principais achados são:

1. **Histórico de pagamento como fator determinante**  
Variáveis relacionadas a atrasos anteriores (PAY_0 a PAY_6) mostraram-se os principais preditores de inadimplência futura, indicando que padrões de comportamento são fortemente recorrentes.

2. **Perfil do cliente impacta o risco**  
Características demográficas como idade e limite de crédito apresentaram correlação inversa com inadimplência. Clientes mais velhos e com limites mais elevados tendem a apresentar menor propensão ao atraso.

3. **Educação e padrão de gastos contribuem de forma secundária**  
Embora menos preditivas isoladamente, variáveis como escolaridade e consistência no valor das faturas ajudam a compor um perfil mais completo do comportamento do cliente.

4. **Modelos de boosting demonstram maior desempenho preditivo**  
O modelo XGBoost superou os demais em métricas como F1-score e AUC, evidenciando a eficácia de técnicas baseadas em ensemble e regularização para problemas com dados desbalanceados.

---

## 7. Aplicação Interativa

A aplicação permite:

- Explorar visualmente os dados e os principais insights
- Treinar e comparar modelos
- Avaliar o desempenho de forma interativa

---

## 8. Como Executar

### 1. Via Docker (Recomendado)

#### 1.1 Usando Docker Compose (Recomendado)
```bash
docker compose up --build
```

#### 1.2 Construção e Execução Manual (Alternativa)
```bash
# Construir a imagem
docker build -t inadimplencia-app .

# Executar o container
docker run -d -p 8501:8501 --name inadimplencia-app inadimplencia-app
```

> **Nota:** Acesse a aplicação em http://localhost:8501 após a execução.

### 2. Manualmente com Python

#### 2.1 Usando Ambiente Virtual (Recomendado)

##### Para Linux/MacOS
```bash
# Criar ambiente virtual
python3 -m venv .venv

# Ativar ambiente virtual
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt

# Executar a aplicação
streamlit run src/app.py
```

##### Para Windows
```cmd
# Criar ambiente virtual
python -m venv .venv

# Ativar ambiente virtual
.venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt

# Executar a aplicação
streamlit run src/app.py
```

> **Nota:** No Windows, use `python` em vez de `python3` e o comando de ativação é diferente.

#### 2.2 Instalação Direta (Não Recomendado)
```bash
# Instalar dependências globalmente
pip install -r requirements.txt

# Executar a aplicação
streamlit run src/app.py
```

> **Nota:** A utilização de ambiente virtual é fortemente recomendada para evitar conflitos com outras instalações Python no sistema.

---

## 9. Estrutura do Projeto

```txt
inadimplencia-streamlit/
├── src/
│   ├── app.py
│   ├── tabs/
│   └── data/
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```