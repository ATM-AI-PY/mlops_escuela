# Telco Customer Churn — experimentación reproducible con MLflow

Tarea de la semana 4 (TC5061). Entrena una regresión logística para predecir el abandono de clientes (*churn*) del dataset **Telco Customer Churn** y registra cada corrida en **MLflow**: parámetros, métricas, artefactos y modelo.

## Contenido

| Archivo | Descripción |
|---|---|
| `train.py` | Script de entrenamiento: carga, limpieza, preprocesamiento, entrenamiento, evaluación y registro en MLflow |
| `requirements.txt` | Dependencias con versiones exactas |
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Dataset ([Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)) |
| `*.ipynb` | Notebook con el análisis exploratorio (EDA) |

## Instalación

Probado con Python 3.10.

```bash
pip install -r requirements.txt
```

## Ejecución

Desde la carpeta del repositorio, con el CSV en la misma carpeta:

```bash
# Valores por defecto (reproduce el modelo base)
python train.py

# Mejor configuración encontrada
python train.py --run-name 04_C0.01_balanced --C 0.01 --class-weight balanced

# Lista de argumentos
python train.py --help
```

### Argumentos

| Argumento | Default | Descripción |
|---|---|---|
| `--C` | `1.0` | Inverso de la fuerza de regularización |
| `--class-weight` | `none` | `balanced` da más peso a la clase minoritaria |
| `--solver` | `lbfgs` | `lbfgs`, `liblinear`, `newton-cg` o `saga` |
| `--max-iter` | `1000` | Iteraciones máximas del optimizador |
| `--seed` | `42` | Semilla aleatoria |
| `--test-size` | `0.2` | Proporción de datos para prueba |
| `--run-name` | — | Nombre de la corrida en MLflow |
| `--experiment-name` | `telco_churn_logreg_hiperparametros` | Experimento de MLflow |

## Ver resultados en MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow_telco.db --port 5003
```

Abrir http://localhost:5003. Cada corrida registra:

- **Parámetros:** C, class_weight, solver, max_iter, seed, test_size.
- **Métricas:** accuracy, precision, recall, F1 y ROC-AUC en prueba; accuracy y F1 en entrenamiento.
- **Artefactos:** `matriz_confusion.png` y `metricas.json`.
- **Modelo:** el pipeline completo (preprocesamiento + regresión logística), con firma.

## Reproducibilidad

La semilla (`--seed`, 42 por defecto) se fija en `random`, NumPy y `PYTHONHASHSEED`. Además, se pasa como `random_state` a `train_test_split` y a `LogisticRegression`. Dos ejecuciones con los mismos argumentos producen exactamente las mismas métricas.

## Corridas realizadas

```bash
python train.py --run-name 01_base
python train.py --run-name 02_balanced --class-weight balanced
python train.py --run-name 03_C0.01 --C 0.01
python train.py --run-name 04_C0.01_balanced --C 0.01 --class-weight balanced
python train.py --run-name 05_C10_balanced --C 10 --class-weight balanced
python train.py --run-name 06_liblinear_balanced --solver liblinear --class-weight balanced
python train.py --run-name 07_maxiter10_balanced --max-iter 10 --class-weight balanced
```
