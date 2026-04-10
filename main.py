# %%

import os

import mlflow
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# %%
path_home = os.getcwd()
trein_path = os.path.join(path_home, "data", "train.csv")
df = pd.read_csv(trein_path)


# %%
mlflow.set_experiment("credit_scoring_optuna")

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
def objective(trial: optuna.trial.Trial):
    with mlflow.start_run(nested=True):
        model_name = trial.suggest_categorical(
            "model",
            [
                "RandomForest",
                "GradientBoostingClassifier",
                "LogisticRegression",
            ],
        )
        if model_name == "RandomForest":
            model = RandomForestClassifier(
                n_estimators=trial.suggest_int("rf_n_estimators", 50, 500),
                max_depth=trial.suggest_int("rf_max_depth", 3, 12),
                min_samples_split=trial.suggest_int(
                    "rf_min_samples_split", 2, 10
                ),
                class_weight="balanced",
            )
        elif model_name == "GradientBoostingClassifier":
            model = GradientBoostingClassifier(
                n_estimators=trial.suggest_int("gb_n_estimators", 50, 500),
                max_depth=trial.suggest_int("gb_max_depth", 3, 12),
                learning_rate=trial.suggest_float(
                    "gb_learning_rate", 0.01, 0.3
                ),
            )
        else:
            model = LogisticRegression(
                C=trial.suggest_float("lr_c", 0.01, 10),
                max_iter=trial.suggest_int("lr_max_iter", 100, 2000),
                class_weight="balanced",
            )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        result = cross_val_score(
            pipeline, X_train, y_train, cv=3, scoring="roc_auc"
        ).mean()

        mlflow.log_params(trial.params)
        mlflow.log_metric("roc_auc", result)

        model_info = mlflow.sklearn.log_model(pipeline, name=model_name)

        trial.set_user_attr("model_uri", model_info.model_uri)

        return result


# %%
with mlflow.start_run(run_name="optuna_optimization"):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=3)

print(f"Melhor ROC-AUC: {study.best_value}")
print(f"Melhes parâmetros: {study.best_params}")

best_trial = study.best_trial

model_uri = best_trial.user_attrs["model_uri"]

mlflow.register_model(
    model_uri=model_uri,
    name="credit_scoring_model",
)
