# Dockerfile para o Streamlit App de Previsão de Inadimplência
FROM python:3.10-slim

# Evita prompts de interação
ENV DEBIAN_FRONTEND=noninteractive

# Cria diretório de trabalho
WORKDIR /app

# Copia requirements e instala dependências
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código do projeto
COPY . .

# Expõe a porta padrão do Streamlit
EXPOSE 8501

# Comando para rodar o app
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
