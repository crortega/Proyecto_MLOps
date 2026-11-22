# ==============================================================================
# ARCHIVO: evaluate_champion.py
# QUE HACE: Revisa la ultima version del modelo. Si su RMSE es menor al umbral,
#           le pone el alias "champion" para que pase a produccion.
# ==============================================================================

import sys
import mlflow
from mlflow.tracking import MlflowClient




def evaluate_and_promote():
    print("Evaluando candidato a Modelo Champion...")

    mlflow.set_tracking_uri("file:./mlruns")
    client = MlflowClient()

    model_name = "DiabetesRegressionModel"
    # Nuestro umbral de calidad (Quality Gate)[cite: 1, 4]
    # En el notebook vimos que los modelos sacaban unos 52-56 de RMSE[cite: 11].
    # Ponemos 60.0 como limite maximo de error permitido. Si falla mas que eso, no vale.
    RMSE_THRESHOLD = 60.0




    # Buscamos todas las versiones registradas de nuestro modelo y cogemos la ultima
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        print("ERROR: No hay ningun modelo registrado todavia!")
        sys.exit(1)

    # Ordenamos por numero de version de mayor a menor y pillamos la primera (la mas nueva)
    latest_version = sorted(versions, key=lambda x: int(x.version), reverse=True)[0]
    version_num = latest_version.version
    run_id = latest_version.run_id




    # Vamos a mirar las metricas que saco ese entrenamiento en concreto
    run_data = client.get_run(run_id).data
    rmse = run_data.metrics.get("rmse")

    print(f"Version evaluada: v{version_num} | RMSE obtenido: {rmse:.2f} | Umbral maximo: {RMSE_THRESHOLD}")




    # QUALITY GATE: Comprobamos si pasa el examen[cite: 1, 4]
    if rmse is not None and rmse < RMSE_THRESHOLD:
        print("PRUEBA SUPERADA! El modelo es lo bastante bueno.")
        
        # Le asignamos el alias 'champion'. Si ya lo tenia otra version vieja, MLflow se lo quita y se lo da a esta[cite: 1, 4]
        client.set_registered_model_alias(
            name=model_name,
            alias="champion",
            version=version_num
        )
        print(f"Modelo '{model_name}' v{version_num} ascendido a @champion!")
    else:
        print(f"CALIDAD INSUFICIENTE: El RMSE ({rmse:.2f}) supera el limite de {RMSE_THRESHOLD}.")
        print("No se promociona a champion y paramos el pipeline.")
        sys.exit(1) # Devolvemos error para frenar el despliegue en GitHub Actions




if __name__ == "__main__":
    evaluate_and_promote()