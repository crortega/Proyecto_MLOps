# Sistema MLOps End-to-End: Predicción de Progresión de Diabetes

## 1. Objetivo del Proyecto
Este proyecto implementa un flujo completo de **MLOps** para resolver un problema de regresión
 cuantitativa: predecir la progresión de la diabetes en pacientes un año después de su evaluación 
 inicial. Partiendo de un notebook exploratorio, el código se ha refactorizado en módulos
 independientes para automatizar la extracción de datos, validación, limpieza, entrenamiento
 con **MLflow**, selección de modelo campeón, despliegue con **FastAPI** e integración 
 continua mediante **GitHub Actions**.

## 2. Dataset Utilizado
Se utiliza el dataset clásico **Diabetes** incluido en `scikit-learn`. Contiene 442 registros
 de pacientes con 10 variables fisiológicas y sanguíneas normalizadas 
 (`age`, `sex`, `bmi`, `bp`, `s1` a `s6`) y una variable objetivo (`target`) 
 que mide la progresión de la enfermedad.

## 3. Estructura del Repositorio
```text
├── .github/workflows/ci.yml   # Pipeline automatizado de CI/CD con GitHub Actions
├── data/                      # Carpeta donde se guardan los CSV raw y procesados
├── notebooks/                 # Notebook original con el analisis exploratorio (EDA)
├── src/                       # Scripts modulares del pipeline de datos y ML
│   ├── data_extraction.py     # Descarga y guarda los datos en crudo
│   ├── data_validation.py     # Comprueba esquema, tipos y nulos
│   ├── data_cleaning.py       # Limpia datos y crea variables de interaccion
│   ├── train.py               # Entrena modelos, registra metricas en MLflow y guarda el mejor
│   └── evaluate_champion.py   # Evalua el Quality Gate y asigna el alias 'champion'
├── app/                       # Codigo de la API REST
│   ├── main.py                # Endpoints de FastAPI y manejo de errores
│   └── schemas.py             # Validacion de entrada/salida con Pydantic
├── requirements.txt           # Dependencias del proyecto
└── README.md                  # Documentacion del proyecto
```

## 4. Instalación de Dependencias
Clona el repositorio, crea un entorno virtual e instala las librerías necesarias:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 5. Ejecución del Pipeline (Datos, Entrenamiento y Champion)
Ejecuta los scripts en orden secuencial desde la raíz del proyecto:
```bash
# 1. Pipeline de datos
python src/data_extraction.py
python src/data_validation.py
python src/data_cleaning.py

# 2. Entrenamiento y seleccion de modelo
python src/train.py
python src/evaluate_champion.py
```

### Selección del Modelo Champion (Quality Gate)
En `train.py` se entrenan tres algoritmos (*Linear Regression*, *Random Forest* y *Gradient Boosting*)
 registrando sus métricas (`RMSE`, `MAE`, `R2`) en MLflow. El modelo con menor `RMSE` se sube al
 **MLflow Model Registry**.
Posteriormente, `evaluate_champion.py` aplica un **Quality Gate**:
* **Umbral definido:** `RMSE < 60.0`.
* **Justificación:** En la fase exploratoria los modelos base rondaban un `RMSE` de ~52-55. Fijar 
un límite de 60.0 garantiza que ningún modelo degradado pase a producción si cambian los datos, 
permitiendo margen frente a variaciones normales. Si supera el corte, recibe el alias `@champion`.

*(Opcional) Para ver la interfaz visual de MLflow en local:* `mlflow ui --backend-store-uri file:./mlruns`

## 6. Ejecución y Pruebas de la API
Para levantar el servidor de FastAPI en local:
```bash
uvicorn app.main:app --reload
```
La documentación interactiva estará disponible en `http://127.0.0.1:8000/docs`.

### Ejemplos de prueba con `curl`:
1. **Health Check (`GET /health`):**
```bash
curl [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
```
2. **Información del Modelo (`GET /model-info`):**
```bash
curl [http://127.0.0.1:8000/model-info](http://127.0.0.1:8000/model-info)
```
3. **Predicción (`POST /predict`):**
```bash
curl -X POST "[http://127.0.0.1:800](http://127.0.0.1:800)
