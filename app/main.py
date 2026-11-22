# ==============================================================================
# ARCHIVO: main.py
# QUE HACE: Levanta la API REST con FastAPI, carga el modelo champion y sirve
#           los endpoints /health, /model-info y /predict.
# ==============================================================================

import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException

from app.schemas import PatientData, PredictionResponse




# Inicializamos la aplicacion FastAPI
app = FastAPI(
    title="API Prediccion Progresion Diabetes",
    description="Servicio MLOps que expone el modelo champion registrado en MLflow",
    version="1.0.0"
)




# Variables globales para guardar el modelo cargado en memoria y sus datos
MODEL_NAME = "DiabetesRegressionModel"
MODEL_ALIAS = "champion"
loaded_model = None
model_metadata = {}




# Este evento se ejecuta automaticamente UNA VEZ justo cuando arranca la API
@app.on_event("startup")
def load_champion_model():
    global loaded_model, model_metadata
    print("Arrancando API... buscando modelo champion en MLflow...")

    mlflow.set_tracking_uri("file:./mlruns")
    client = MlflowClient()

    try:
        # Buscamos que version tiene puesta la etiqueta 'champion'
        version_info = client.get_model_version_by_alias(name=MODEL_NAME, alias=MODEL_ALIAS)
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

        # Cargamos el modelo en memoria para que las predicciones sean rapidisimas
        loaded_model = mlflow.sklearn.load_model(model_uri)

        # Guardamos info util para enseñarla luego en /model-info[cite: 5]
        run_data = client.get_run(version_info.run_id).data
        model_metadata = {
            "model_name": MODEL_NAME,
            "model_version": version_info.version,
            "alias": MODEL_ALIAS,
            "rmse": run_data.metrics.get("rmse"),
            "r2": run_data.metrics.get("r2")
        }
        print(f"Modelo {MODEL_NAME} v{version_info.version} (@{MODEL_ALIAS}) cargado con exito!")

    except Exception as e:
        # Si no hay modelo champion o falla algo, no tiramos el servidor abajo,
        # pero dejamos loaded_model como None para devolver error 503 si alguien pide prediccion
        print(f"AVISO: No se pudo cargar el modelo champion: {e}")
        loaded_model = None




# ENDPOINT 1: Comprobar que el servidor esta vivo[cite: 5]
@app.get("/health")
def health_check():
    return {"status": "ok"}




# ENDPOINT 2: Ver que modelo estamos usando ahora mismo[cite: 5]
@app.get("/model-info")
def get_model_info():
    if loaded_model is None:
        # Si el modelo no cargo al arrancar, devolvemos error 503 (Servicio no disponible)
        raise HTTPException(status_code=503, detail="El modelo champion no esta disponible en este momento.")
    return model_metadata




# ENDPOINT 3: Hacer predicciones con datos de un paciente nuevo[cite: 5]
@app.post("/predict", response_model=PredictionResponse)
def predict_progression(patient: PatientData):
    # Primero miramos si tenemos modelo cargado
    if loaded_model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado. No se pueden hacer predicciones.")

    try:
        # Convertimos los datos que manda el usuario (Pydantic) a un diccionario y luego a DataFrame
        input_dict = patient.model_dump()
        df = pd.DataFrame([input_dict])

        # OJO MUY IMPORTANTE: Como entrenamos el modelo con las 3 columnas nuevas de ingenieria
        # de caracteristicas, tenemos que calcularlas aqui tambien antes de predecir!
        df['bmi_age_interaction'] = df['bmi'] * df['age']
        df['bp_bmi_interaction'] = df['bp'] * df['bmi']
        df['s1_s2_ratio'] = df['s1'] / (df['s2'] + 1e-9)

        # Hacemos la prediccion
        pred = loaded_model.predict(df)[0]

        return PredictionResponse(
            prediction=round(float(pred), 2),
            model_name=model_metadata["model_name"],
            model_version=str(model_metadata["model_version"])
        )

    except Exception as e:
        # Si ocurre algun fallo inesperado calculando, devolvemos error 500
        raise HTTPException(status_code=500, detail=f"Error interno al predecir: {str(e)}")