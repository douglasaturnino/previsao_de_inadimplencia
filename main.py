# %%

import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# %%
path_home = os.getcwd()
trein_path = os.path.join(path_home, "data", "train.csv")
df = pd.read_csv(trein_path)

# %%

columns = [
    "target",
    "TaxaDeUtilizacaoDeLinhasNaoGarantidas",
    "Idade",
    "NumeroDeVezes30-59DiasAtrasoNaoPior",
    "TaxaDeEndividamento",
    "RendaMensal",
    "NumeroDeLinhasDeCreditoEEmprestimosAbertos",
    "NumeroDeVezes90DiasAtraso",
    "NumeroDeEmprestimosOuLinhasImobiliarias",
    "NumeroDeVezes60-89DiasAtrasoNaoPior",
    "NumeroDeDependentes",
]

df = df[columns]
# %%
X = df.drop("target", axis=1)
y = df["target"]

# %%
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, stratify=y, test_size=0.3
)

# %%
column_median = ["RendaMensal", "NumeroDeDependentes"]
remover = ["target", "RendaMensal", "NumeroDeDependentes"]

column_scaler = [col for col in columns if col not in remover]

# %%

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

scaler = Pipeline(steps=[("scaler", StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            column_median,
        ),
        (
            "scaler",
            scaler,
            column_scaler,
        ),
    ],
    remainder="passthrough",
)
# %%
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=6, class_weight="balanced"
    ),
    "GradientBoostingClassifier": GradientBoostingClassifier(
        n_estimators=200, max_depth=3
    ),
    "LogisticRegression": LogisticRegression(
        max_iter=1000, class_weight="balanced"
    ),
}

# %%
result = {}
for name, model in models.items():
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict_proba(X_valid)[:, 1]
    metric = roc_auc_score(y_valid, pred)

    result[name] = metric
    print(f"{name} ROC-AUC: {metric:.4f}")
