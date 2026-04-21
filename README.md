# Descrição

Projeto de analise preditiva de credito com treinamento de modelo, API de previsão e dashboard.

Este repositorio contem um pipeline para treinar modelos de score de credito usando MLflow, alem de uma API para fazer as predisões e um painel streamlit para interagir com o dados.

# Estrutura do Projeto

- `main.py` - Treina os dados e usa o optuna para otimizar os paramentros do modelo para que tenhamos os melhores resultados e registro o melhor modelo no MLflow.

- `api.py` - API utilizando o FastAPI e expõe o rota `/predict`.

- `home.py` - Cria as paginas no streamlit.

- `pages/painel.py` - Utiliza metricas e cria dashboard

- `pages/app.py` - Utiliza os dados para fazer a predição do modelo

- `pyproject.toml` - Utiliza as varias dependencias para o desenvolvimento do projeto.

# Ferramentas
- Python (pandas, sklearn, plotly)
- Streamlit
- Vscode
- FastAPI
- MLflow
- Optuna

# Desenvolvimento do Projeto

## 1ª etapa
-Carrega os dados da pasta data;
- Dividi os dados em treino e validação;
- Cria o pipeline para o preprocessamento dos dados
- Utiliza o optuna para encontrar os melhores parametros para a escolha do melhor modelo.
- Faz o registro do melhor modelo no framework mlflow

## 2ª Etapa
- Implantação da API (Interface de Programação de Aplicativos) para diponibilizar a predição do melhor modelo.

- Cria a rota /predict para mostrar os parametros da predição do modelo

## 3ª Etapa
Criação dos paineis no pagina do streamlit;

- 1ª painel: Informações de metricas do score de crédito e predição do modelo
- 2ª painel: informações de metricas do dashboard de score de crédito, grafico de barra dos clientes inadimplentes, grafico de barra por faixa etaria, entre outras informações analisadas.

# Resultados obtidos
 - O Dashboard visualizado no streamlit;
 - API.