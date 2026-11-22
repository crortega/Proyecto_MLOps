# ==============================================================================
# ARCHIVO: train.py
# QUE HACE: Entrena varios modelos, los trackea en MLflow y registra el mejor.
# ==============================================================================

import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score




def train_models():
    print("Arrancando el entrenamiento con MLflow...")

    # Configuramos MLflow para que guarde todo en una carpeta local llamada 'mlruns'
    # Asi no necesitamos levantar un servidor externo complejo
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Diabetes_Progression_Experiment")




    # Cargamos los datos limpios
    data_path = os.path.join("data", "processed", "diabetes_clean.csv")
    df = pd.read_csv(data_path)

    # Separamos caracteristicas (X) de lo que queremos predecir (y)[cite: 11]
    X = df.drop(columns=["target"])
    y = df["target"]

    # Dividimos en 80% para entrenar y 20% para examinar al modelo[cite: 11]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




    # Definimos los 3 modelos que probamos en el notebook[cite: 11]
    models = {
        "LinearRegression": {
            "model": LinearRegression(),
            "params": {"fit_intercept": True}
        },
        "RandomForest": {
            "model": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
            "params": {"n_estimators": 100, "max_depth": 6, "random_state": 42}
        },
        "GradientBoosting": {
            "model": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
            "params": {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42}
        }
    }




    # Variables para ir guardando cual es el mejor de esta tanda
    best_rmse = float("inf")
    best_run_id = None
    best_model_name = ""




    # Bucle para entrenar cada modelo y apuntar sus notas en MLflow
    for name, config in models.items():
        # start_run abre una "sesion" de grabacion en MLflow para este modelo
        with mlflow.start_run(run_name=name) as run:
            print(f"Entrenando {name}...")
            model = config["model"]
            model.fit(X_train, y_train)

            # Hacemos predicciones sobre el test
            preds = model.predict(X_test)

            # Calculamos las metricas de error
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            mae = float(mean_absolute_error(y_test, preds))
            r2 = float(r2_score(y_test, preds))

            print(f" -> Resultados {name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.2f}")




            # Guardamos parametros y metricas en MLflow
            mlflow.log_params(config["params"])
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            # Guardamos el modelo fisico dentro de MLflow (como artefacto)
            mlflow.sklearn.log_model(sk_model=model, artifact_path="model")




            # Comprobamos si este modelo mejora el record actual (menor RMSE es mejor)
            if rmse < best_rmse:
                best_rmse = rmse
                best_run_id = run.info.run_id
                best_model_name = name




    # Una vez probados todos, registramos el ganador en el Model Registry de MLflow[cite: 1, 4]
    print(f"\nEl ganador del entrenamiento es {best_model_name} con RMSE: {best_rmse:.2f}")
    model_uri = f"runs:/{best_run_id}/model"
    registered_model_name = "DiabetesRegressionModel"

    # Esto crea una nueva version del modelo en el registro central
    result = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
    print(f"Modelo registrado como '{registered_model_name}', version: {result.version}")




if __name__ == "__main__":
    train_models()