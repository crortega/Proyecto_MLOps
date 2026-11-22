# ==============================================================================
# ARCHIVO: schemas.py
# QUE HACE: Define la estructura de entrada y salida de la API usando Pydantic.
#           Si alguien manda datos mal formados, esto devuelve error 422 solo.
# ==============================================================================

from pydantic import BaseModel, Field




# Molde de los datos que nos tiene que mandar el usuario para pedir una prediccion[cite: 5, 6]
class PatientData(BaseModel):
    # Field(...) obliga a que el campo venga si o si, y ademas podemos poner ejemplos
    age: float = Field(..., description="Edad normalizada del paciente", example=0.038)
    sex: float = Field(..., description="Sexo normalizado", example=0.050)
    bmi: float = Field(..., description="Indice de masa corporal", example=0.061)
    bp: float = Field(..., description="Presion arterial media", example=0.021)
    s1: float = Field(..., description="Colesterol total en suero", example=-0.044)
    s2: float = Field(..., description="Lipoproteinas de baja densidad (LDL)", example=-0.034)
    s3: float = Field(..., description="Lipoproteinas de alta densidad (HDL)", example=-0.043)
    s4: float = Field(..., description="Colesterol total / HDL", example=-0.002)
    s5: float = Field(..., description="Logaritmo de nivel de trigliceridos", example=0.019)
    s6: float = Field(..., description="Nivel de azucar en sangre", example=-0.017)




# Molde de lo que nuestra API va a responder cuando todo vaya bien[cite: 5]
class PredictionResponse(BaseModel):
    prediction: float
    model_name: str
    model_version: str