"""
Este código cria um Dashboard interativo de risco de crédito utilizando Streamlit, com foco na análise exploratória dos dados de clientes.

A aplicação carrega um conjunto de dados históricos de crédito a partir de um arquivo CSV e realiza um processo inicial de limpeza, tratando valores ausentes e removendo registros fora de faixas consideradas válidas para variáveis financeiras.

O dashboard permite que o usuário aplique filtros dinâmicos de renda mensal e idade, ajustando a população analisada em tempo real. Com base nesses filtros, são calculadas métricas principais como o número total de clientes, a quantidade de inadimplentes e a taxa de inadimplência.

Além disso, a interface apresenta visualizações interativas geradas com Plotly, incluindo:

- Distribuição de inadimplentes em formato de gráfico de pizza;

- Análise de inadimplência por faixa etária;

- Distribuição de inadimplência por faixas de renda mensal;

- Relação entre taxa de endividamento e inadimplência.

Esses gráficos permitem identificar padrões e comportamentos de risco, facilitando a compreensão dos principais fatores associados à inadimplência. O objetivo do dashboard é apoiar a análise de crédito de forma visual, intuitiva e orientada a dados.
"""

import os

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Dashboard de Crédito")
col1, col2 = st.columns(2, border=True)
col1.page_link("pages/app.py", label="Score de Crédito (1º Painel)", icon="💳")
col2.page_link(
    "pages/painel.py", label="Dashboard de Risco (2º Painel)", icon="📊"
)


@st.cache_data
def load_data():
    home_path = os.getcwd()

    data_path = os.path.join(home_path, "data", "train.csv")

    data = pd.read_csv(data_path)
    return data


data = load_data()
data = data.fillna({"RendaMensal": 0, "NumeroDeDependentes": 0})
data = data[data["TaxaDeUtilizacaoDeLinhasNaoGarantidas"] <= 1]
data = data[data["NumeroDeVezes30-59DiasAtrasoNaoPior"] <= 20]
data = data[data["TaxaDeEndividamento"] <= 12]
data = data[data["NumeroDeVezes90DiasAtraso"] <= 20]
data = data[data["NumeroDeVezes60-89DiasAtrasoNaoPior"] <= 20]

renda_min = st.sidebar.number_input(
    label="Renda minima",
    value=1_000,
)
renda_max = st.sidebar.number_input(
    label="Renda maxima",
    value=int(data["RendaMensal"].max()),
)

idade_mim, idade_max = st.sidebar.slider(
    label="Idade",
    min_value=int(data["Idade"].min()),
    max_value=int(data["Idade"].max()),
    value=(18, data["Idade"].max()),
)

data = data[
    (data["RendaMensal"] >= renda_min) & (data["RendaMensal"] <= renda_max)
]
data = data[(data["Idade"] >= idade_mim) & (data["Idade"] <= idade_max)]

total = len(data)
inadimplentes = data["target"].sum()
taxa = inadimplentes / total

st.title("Dashboard de Risco de Crédito")


col1, col2, col3 = st.columns(3)


col1.metric("👥 Total de Clientes", f"{total:,.0f}".replace(",", "."))
col2.metric("⚠️ Inadimpletes", f"{inadimplentes:,.0f}".replace(",", "."))
col3.metric("📉 Taxa de Inadimpletes", f"{taxa:.2%}")

aux = data.copy()
aux["target_label"] = aux["target"].map(
    {0: "Não inadimpletes", 1: "Inadimpletes"}
)
fig = px.pie(
    data_frame=aux,
    names="target_label",
    title="Distribuição do Inadimpletes",
)
fig.update_layout(title_x=0.2)
st.plotly_chart(fig)

temp = data.copy()
bins = [0, 30, 40, 50, 60, 70, 80, 120]
labels = ["<=30", "31-40", "41-50", "51-60", "61-70", "71-80", "81+"]
age_group = pd.cut(
    temp["Idade"],
    bins=bins,
    labels=labels,
    include_lowest=True,
)
temp["age_group"] = age_group

group = (
    temp.groupby("age_group")
    .agg(
        count=("target", "count"),
    )
    .reset_index()
)

fig = px.bar(
    group,
    x="age_group",
    y="count",
    title=" Quantidades de Inadimples por Faixa Etária",
    labels={"age_group": "Faixas", "count": "Quantidades"},
    text_auto=True,
)
fig.update_layout(title_x=0.2)
st.plotly_chart(fig)


temp = data.copy()
bins = [0, 1_000, 2_000, 4_000, 7_000, 10_000, 50_000, 5_000_000]
labels = [
    "<=1_000",
    "1_001-2_000",
    "2_001-4_000",
    "4_001-7_000",
    "7_001-10_000",
    "10_001-50_000",
    "50_00+",
]

renda_group = pd.cut(
    temp["RendaMensal"],
    bins=bins,
    labels=labels,
    include_lowest=True,
)
temp["renda_group"] = renda_group

group = (
    temp.groupby("renda_group")
    .agg(
        count=("target", "count"),
    )
    .reset_index()
)

fig = px.bar(
    group,
    x="renda_group",
    y="count",
    title=" Quantidades de Inadimples por Renda Mensal",
    labels={"renda_group": "Faixas de renda", "count": "Quantidades"},
    text_auto=True,
)
fig.update_layout(title_x=0.2)
st.plotly_chart(fig)


temp = data.copy()
bins = [0.0, 0.15, 0.3, 0.5, 1.0, 12]
labels = ["0.0-0.15", "0.16-0.3", "0.31-0.5", "0.51-1.0", "1.0+"]

taxa_group = pd.cut(
    temp["TaxaDeEndividamento"],
    bins=bins,
    labels=labels,
    include_lowest=True,
)
temp["taxa_group"] = taxa_group

group = (
    temp.groupby("taxa_group")
    .agg(
        count=("target", "count"),
    )
    .reset_index()
)

fig = px.bar(
    group,
    x="taxa_group",
    y="count",
    title="Taxa de endividamento por Inadimpletes",
    labels={"taxa_group": "Taxas", "count": "Quantidades"},
    text_auto=True,
)
fig.update_layout(title_x=0.2)
st.plotly_chart(fig)
