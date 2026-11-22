# ==============================================================================
# ARCHIVO: data_cleaning.py
# QUE HACE: Limpia duplicados/nulos, crea variables nuevas y guarda el dataset listo.
# ==============================================================================

import os
import pandas as pd




def clean_and_engineer_data():
    print("Empezando la limpieza y creacion de variables...")

    raw_path = os.path.join("data", "raw", "diabetes_raw.csv")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    output_path = os.path.join(processed_dir, "diabetes_clean.csv")




    # Cargamos los datos ya validados
    df = pd.read_csv(raw_path)
    filas_antes = len(df)




    # 1. LIMPIEZA BASICA (por si acaso entran datos sucios en el futuro)[cite: 4, 11]
    # Borramos filas con nulos si hubiera
    df = df.dropna()
    # Borramos filas exactamente iguales
    df = df.drop_duplicates()

    # Nos aseguramos de que todo sea tipo float/numerico[cite: 4, 11]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')




    # 2. INGENIERIA DE CARACTERISTICAS (Feature Engineering)
    # En el notebook vimos que cruzar variables ayudaba a la regresion lineal[cite: 11]
    # Multiplicamos IMC por edad
    df['bmi_age_interaction'] = df['bmi'] * df['age']
    # Multiplicamos presion arterial por IMC
    df['bp_bmi_interaction'] = df['bp'] * df['bmi']
    # Ratio entre s1 y s2 (sumamos 1e-9 para evitar dividir entre cero y que explote)
    df['s1_s2_ratio'] = df['s1'] / (df['s2'] + 1e-9)




    # Guardamos el archivo limpito y listo para entrenar
    df.to_csv(output_path, index=False)
    print(f"Limpieza terminada. Filas antes: {filas_antes} | Filas despues: {len(df)}")
    print(f"Archivo listo guardado en: {output_path}")




if __name__ == "__main__":
    clean_and_engineer_data()