# ==============================================================================
# ARCHIVO: data_validation.py
# QUE HACE: Revisa que el CSV descargado no venga corrupto, vacio o con columnas raras.
# ==============================================================================

import os
import sys
import pandas as pd
import numpy as np




def validate_data():
    print("Revisando que los datos esten bien (Validacion)...")

    raw_path = os.path.join("data", "raw", "diabetes_raw.csv")

    # Primero miramos si el archivo existe de verdad
    if not os.path.exists(raw_path):
        print(f"ERROR: No encuentro el archivo en {raw_path}. Ejecuta primero data_extraction.py!")
        sys.exit(1) # Salimos con codigo 1 para que GitHub Actions sepa que esto ha fallado




    # Leemos los datos
    df = pd.read_csv(raw_path)




    # CHECK 1: Que el dataset no venga vacio[cite: 11]
    if len(df) == 0:
        print("ERROR: El dataset esta vacio, no hay filas!")
        sys.exit(1)




    # CHECK 2: Que esten todas las columnas que esperamos (las 10 variables + el target)[cite: 11]
    expected_columns = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6', 'target']
    for col in expected_columns:
        if col not in df.columns:
            print(f"ERROR: Falta la columna '{col}' en los datos!")
            sys.exit(1)




    # CHECK 3: Que todos los datos sean numericos (nada de textos colados)[cite: 11]
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) != len(df.columns):
        print("ERROR: Hay columnas que no son numericas y deberian serlo!")
        sys.exit(1)




    # CHECK 4: Que no haya valores nulos (NaN) criticos[cite: 11]
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        print(f"ERROR: Se han encontrado {null_count} valores nulos en el dataset!")
        sys.exit(1)




    print("Todo correcto! Los datos han pasado la validacion sin problemas.")




if __name__ == "__main__":
    validate_data()