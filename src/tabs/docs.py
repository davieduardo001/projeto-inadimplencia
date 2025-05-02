import streamlit as st

def render_doc(): 
    st.header('Documentação do Aplicativo')
    st.caption('''
**Aplicativo interativo para previsão de inadimplência em cartões de crédito.**

Este dashboard foi desenvolvido para demonstrar, de forma didática e visual, todas as etapas de um projeto real de Ciência de Dados aplicado ao problema de inadimplência, utilizando dados públicos da UCI.

- Pipeline completo: visualização, limpeza, divisão treino/teste, classificação (com escolha de algoritmo e métrica), visualização gráfica e documentação.
- Interface organizada em abas, facilitando a navegação e evitando o scroll infinito.
- Todo o código está comentado e explicado em português.
- Pode ser adaptado para outros datasets tabulares e outros problemas de classificação.
''')

    st.markdown('---')
    st.header('Explicando as Métricas de Avaliação')

    st.subheader('Matriz de Confusão')
    st.markdown('''
A matriz de confusão é uma tabela que mostra como o modelo acerta e erra as previsões, comparando os valores reais com os previstos.

|                 | Predito: NÃO | Predito: SIM |
|-----------------|:------------:|:------------:|
| **Real: NÃO**   |     VN       |     FP       |
| **Real: SIM**   |     FN       |     VP       |

- **VN (Verdadeiro Negativo):** O modelo previu NÃO e era realmente NÃO.
- **VP (Verdadeiro Positivo):** O modelo previu SIM e era realmente SIM.
- **FP (Falso Positivo):** O modelo previu SIM, mas era NÃO (falso alarme).
- **FN (Falso Negativo):** O modelo previu NÃO, mas era SIM (deixou passar um caso real).

> No contexto de crédito:
> - **SIM** = Inadimplente (deu calote)
> - **NÃO** = Não inadimplente

A diagonal principal (VN e VP) mostra os acertos, enquanto os outros valores mostram erros do modelo.
''')

    st.subheader('Precision e Recall')
    st.markdown('''
- **Precisão (Precision):** Entre todas as previsões positivas, quantas realmente eram positivas.
    - Fórmula: VP / (VP + FP)
- **Recall (Sensibilidade):** Entre todas as amostras realmente positivas, quantas o modelo identificou corretamente.
    - Fórmula: VP / (VP + FN)

Essas métricas são importantes para entender não só quantos acertos o modelo teve, mas também a qualidade desses acertos.
''')

    st.subheader('Curva ROC e AUC')
    st.markdown('''
A curva ROC (Receiver Operating Characteristic) mostra a relação entre taxa de verdadeiros positivos (sensibilidade) e taxa de falsos positivos para diferentes limiares de decisão. O AUC (Área sob a curva) mede a capacidade do modelo em distinguir entre as classes:

- **AUC = 1:** Modelo perfeito
- **AUC = 0.5:** Modelo aleatório
- **Quanto mais próximo de 1, melhor**

> Made with 🐳 && 🐍
''')