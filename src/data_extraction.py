# ==============================================================================
# ARCHIVO: data_extraction.py
# QUE HACE: Descarga los datos de diabetes de sklearn y los guarda en crudo.
# ==============================================================================

import os
import pandas as pd
from sklearn.datasets import load_diabetes




def extract_data():
    print("Empezando la extraccion de datos...")

    # Definimos donde vamos a guardar el archivo crudo (raw)
    # Usamos os.makedirs por si la carpeta no existe todavia, asi no peta
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    output_path = os.path.join(raw_dir, "diabetes_raw.csv")




    # Cargamos el dataset que ya viene integrado en scikit-learn
    # as_frame=True hace que nos lo devuelva directamente como DataFrame de pandas, mucho mas comodo
    diabetes = load_diabetes(as_frame=True)
    df = diabetes.frame




    # Guardamos el dataframe en un CSV sin el indice (index=False) para que no meta una columna basura
    df.to_csv(output_path, index=False)
    print(f"De lujo! Datos en crudo guardados en: {output_path}")
    print(f"Total de filas y columnas: {df.shape}")




# Este bloque hace que el codigo solo se ejecute si llamamos al archivo directamente
# Si lo importamos desde otro sitio, no se ejecutara solo
if __name__ == "__main__":
    extract_data()