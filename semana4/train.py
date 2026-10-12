import matplotlib

matplotlib.use("Agg")  # backend sin ventana: permite guardar figuras al ejecutar desde terminal

import json
import os
import random

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import argparse



# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
TARGET = "Churn"
ID_COLUMN = "customerID"
NUMERICAS = ["tenure", "MonthlyCharges", "TotalCharges"]

# Valores por defecto
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_ITER = 1000
C = 1.0
SOLVER = "lbfgs"
CLASS_WEIGHT = "none"
SOLVERS = ["lbfgs", "liblinear", "newton-cg", "saga"]

# MLflow
TRACKING_URI = "sqlite:///mlflow_telco.db"
EXPERIMENT_NAME = "telco_churn_logreg_hiperparametros"

# Artefactos
RUTA_MATRIZ_CONFUSION = "matriz_confusion.png"
RUTA_METRICAS = "metricas.json"

print("Configuración inicial terminada")

# ---------------------------------------------------------------------------
# 0. Argumentos de línea de comandos
# ---------------------------------------------------------------------------
def parsear_argumentos(argv=None) -> argparse.Namespace:
    """Define los hiperparámetros y la semilla configurables desde la terminal."""
    parser = argparse.ArgumentParser(
        description="Entrena una regresión logística para predecir churn (Telco).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # muestra los defaults en --help
    )
    # Hiperparámetros del modelo
    parser.add_argument("--C", type=float, default=C,
                        help="Inverso de la fuerza de regularización (menor = más regularización).")
    parser.add_argument("--solver", choices=SOLVERS, default=SOLVER,
                        help="Algoritmo de optimización de la regresión logística.")
    parser.add_argument("--class-weight", choices=["none", "balanced"], default=CLASS_WEIGHT,
                        help="'balanced' da más peso a la clase minoritaria (churn = Yes).")
    parser.add_argument("--max-iter", type=int, default=MAX_ITER,
                        help="Número máximo de iteraciones del optimizador.")
    # Datos y reproducibilidad
    parser.add_argument("--test-size", type=float, default=TEST_SIZE,
                        help="Proporción de datos para prueba.")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE,
                        help="Semilla aleatoria (partición de datos y modelo).")
    parser.add_argument("--data-path", default=DATA_PATH, help="Ruta al CSV del dataset.")
    # MLflow
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME,
                        help="Nombre del experimento en MLflow.")
    parser.add_argument("--tracking-uri", default=TRACKING_URI,
                        help="Dónde guarda MLflow las corridas.")
    parser.add_argument("--run-name", default=None,
                        help="Nombre de la corrida en MLflow (opcional).")

    args = parser.parse_args(argv)
    if args.C <= 0:
        parser.error("--C debe ser mayor que 0")
    if not 0 < args.test_size < 1:
        parser.error("--test-size debe estar entre 0 y 1")
    return args

# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------
def fijar_semilla(seed: int) -> None:
    """
    Fija la semilla en todas las fuentes de aleatoriedad para que dos ejecuciones
    con los mismos argumentos den exactamente el mismo resultado.

    - random / numpy: generadores globales de Python y NumPy.
    - PYTHONHASHSEED: orden de hashing (afecta a procesos hijos que se lancen).
    - Además, la misma semilla se pasa explícitamente como random_state a
      train_test_split (partición) y a LogisticRegression (solvers con
      componente aleatoria, como liblinear y saga).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# 1. Carga de datos
# ---------------------------------------------------------------------------
def cargar_datos(ruta: str) -> pd.DataFrame:
    """Lee el CSV del dataset y lo devuelve como DataFrame."""
    return pd.read_csv(ruta)

print("Carga de datos terminada")

# ---------------------------------------------------------------------------
# 2. Limpieza
# ---------------------------------------------------------------------------
def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza de datos (de conformidad con el notebook de la semana anterior):
    - TotalCharges viene como texto por 11 valores en blanco (' '); se convierte a numérico.
    - Esos 11 registros son clientes con tenure = 0 (sin cobros todavía), así que se rellenan con 0.
    - No hay duplicados reales ni outliers (regla IQR), por lo que no se eliminan filas.
    - customerID es un identificador único que no aporta información al modelo; se elimina.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df = df.drop(columns=ID_COLUMN)
    return df

print("Limpieza de datos terminada")

# ---------------------------------------------------------------------------
# 3. Preparación de variables y partición
# ---------------------------------------------------------------------------
def separar_variables(df: pd.DataFrame):
    """Separa variables predictoras (X) y variable objetivo (y), codificando Churn como 1/0."""
    X = df.drop(columns=TARGET)
    y = (df[TARGET] == "Yes").astype(int)
    return X, y

def dividir_datos(X, y, test_size: float, random_state: int):
    """Divide en entrenamiento y prueba de forma estratificada para conservar la proporción de churn."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

print("Preparación de variables y partición terminada")




# ---------------------------------------------------------------------------
# 4. Preprocesamiento y modelo
# ---------------------------------------------------------------------------
def construir_preprocesamiento(numericas: list, categoricas: list) -> ColumnTransformer:
    """Escala las variables numéricas y codifica con one-hot las categóricas."""
    return ColumnTransformer([
        ("num", StandardScaler(), numericas),
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), categoricas),
    ])


def construir_pipeline(numericas: list, categoricas: list, C: float, solver: str,
                       class_weight: str, max_iter: int, random_state: int) -> Pipeline:
    """Une preprocesamiento y regresión logística (con los hiperparámetros recibidos) en un Pipeline."""
    modelo = LogisticRegression(
        C=C,
        solver=solver,
        class_weight=None if class_weight == "none" else class_weight,
        max_iter=max_iter,
        random_state=random_state,
    )
    return Pipeline([
        ("preprocesamiento", construir_preprocesamiento(numericas, categoricas)),
        ("modelo", modelo),
    ])


def entrenar_modelo(pipeline: Pipeline, X_train, y_train) -> Pipeline:
    """Ajusta el pipeline con los datos de entrenamiento."""
    pipeline.fit(X_train, y_train)
    return pipeline

print("Preprocesamiento y modelo terminados")



# ---------------------------------------------------------------------------
# 5. Evaluación
# ---------------------------------------------------------------------------
def calcular_metricas(y_true, y_pred, y_proba) -> dict:
    """Calcula las métricas de clasificación usadas en el notebook."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def evaluar_modelo(modelo: Pipeline, X_test, y_test):
    """Predice sobre el conjunto de prueba y devuelve métricas y predicciones."""
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]
    return calcular_metricas(y_test, y_pred, y_proba), y_pred


def evaluar_baseline_dummy(X_train, y_train, X_test, y_test) -> dict:
    """Referencia mínima: clasificador que siempre predice la clase mayoritaria."""
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    y_pred = dummy.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred, zero_division=0),
    }


def guardar_matriz_confusion(y_true, y_pred, ruta: str) -> str:
    """Genera la matriz de confusión y la guarda como imagen."""
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["No", "Yes"], cmap="Blues", ax=ax
    )
    ax.set_title("Matriz de confusión - Regresión logística")
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    return ruta

def guardar_metricas_json(metricas: dict, ruta: str) -> str:
    """Guarda las métricas en un archivo JSON (se registra como artefacto)."""
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump({k: round(v, 4) for k, v in metricas.items()}, f, indent=2)
    return ruta


# ---------------------------------------------------------------------------
# 6. Registro en MLflow
# ---------------------------------------------------------------------------
def configurar_mlflow(tracking_uri: str, experiment_name: str) -> None:
    """Indica dónde guardar las corridas y en qué experimento agruparlas."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def registrar_parametros(parametros: dict) -> None:
    """Registra cada hiperparámetro de la corrida con log_param."""
    for nombre, valor in parametros.items():
        mlflow.log_param(nombre, valor)


def registrar_metricas(metricas: dict, prefijo: str = "") -> None:
    """Registra cada métrica con log_metric (el prefijo distingue prueba/entrenamiento)."""
    for nombre, valor in metricas.items():
        mlflow.log_metric(f"{prefijo}{nombre}", valor)


def registrar_modelo(modelo: Pipeline, X_ejemplo) -> None:
    """Registra el pipeline completo (preprocesamiento + modelo) con su firma de entrada/salida."""
    firma = infer_signature(X_ejemplo, modelo.predict(X_ejemplo))
    mlflow.sklearn.log_model(
        sk_model=modelo,
        name="modelo",
        signature=firma,
        input_example=X_ejemplo.head(5),
    )

print("Evaluación terminada")

# ---------------------------------------------------------------------------
# Flujo principal
# ---------------------------------------------------------------------------
def main():
    args = parsear_argumentos()
    fijar_semilla(args.seed)
    print("Configuración:", vars(args))

    # Carga y limpieza
    df = limpiar_datos(cargar_datos(args.data_path))
    X, y = separar_variables(df)
    categoricas = [col for col in X.columns if col not in NUMERICAS]
    print(f"Datos: {X.shape[0]} registros, {X.shape[1]} variables "
          f"({len(NUMERICAS)} numéricas, {len(categoricas)} categóricas)")

    # Partición
    X_train, X_test, y_train, y_test = dividir_datos(X, y, args.test_size, args.seed)
    print(f"Entrenamiento: {X_train.shape[0]} | Prueba: {X_test.shape[0]} | "
          f"Proporción de churn en prueba: {y_test.mean():.3f}")

    configurar_mlflow(args.tracking_uri, args.experiment_name)
    with mlflow.start_run(run_name=args.run_name) as run:
        # Hiperparámetros
        registrar_parametros({
            "C": args.C,
            "solver": args.solver,
            "class_weight": args.class_weight,
            "max_iter": args.max_iter,
            "test_size": args.test_size,
            "seed": args.seed,
        })

        # Entrenamiento
        modelo = construir_pipeline(NUMERICAS, categoricas, C=args.C, solver=args.solver,
                                    class_weight=args.class_weight, max_iter=args.max_iter,
                                    random_state=args.seed)
        modelo = entrenar_modelo(modelo, X_train, y_train)

        # Evaluación en prueba
        metricas, y_pred = evaluar_modelo(modelo, X_test, y_test)
        print("\nMétricas en prueba:")
        for nombre, valor in metricas.items():
            print(f"  {nombre}: {valor:.3f}")
        print("\n" + classification_report(y_test, y_pred, target_names=["No", "Yes"], digits=3))

        # Comparación contra el clasificador trivial
        dummy = evaluar_baseline_dummy(X_train, y_train, X_test, y_test)
        print(f"Dummy (clase mayoritaria) - accuracy: {dummy['accuracy']:.3f} | "
              f"recall: {dummy['recall']:.3f}")

        # Revisión de sobreajuste
        metricas_train, _ = evaluar_modelo(modelo, X_train, y_train)
        print(f"Entrenamiento - accuracy: {metricas_train['accuracy']:.3f} | f1: {metricas_train['f1']:.3f}")
        print(f"Prueba        - accuracy: {metricas['accuracy']:.3f} | f1: {metricas['f1']:.3f}")

        # Métricas
        registrar_metricas(metricas, prefijo="test_")
        registrar_metricas({"accuracy": metricas_train["accuracy"],
                            "f1": metricas_train["f1"]}, prefijo="train_")

        # Artefactos
        mlflow.log_artifact(guardar_matriz_confusion(y_test, y_pred, RUTA_MATRIZ_CONFUSION))
        mlflow.log_artifact(guardar_metricas_json(metricas, RUTA_METRICAS))

        # Modelo
        registrar_modelo(modelo, X_train)

        print(f"\nCorrida registrada en MLflow - experimento: '{args.experiment_name}' | "
              f"run_id: {run.info.run_id}")



print("Flujo principal terminado")


if __name__ == "__main__":
    main()

print("Ejecución completa")