import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib 

script_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.abspath(os.path.join(script_dir, os.pardir))

DATA_DIR = os.path.join(project_root, 'data')

MODELS_DIR = os.path.join(project_root, 'trained_models')

# Rutas a los modelos y el dataset
RUTA_DATASET = os.path.join(DATA_DIR, 'winequality-red.csv')
RUTA_GUARDADO_MODELO = os.path.join(MODELS_DIR, 'wine_quality_model.joblib')
RUTA_VALORES_IMPUTACION = os.path.join(MODELS_DIR, 'imputation_values.joblib')

# características esperadas por el modelo
CARACTERISTICAS = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]
VARIABLE_OBJETIVO = 'quality' # La variable que queremos predecir

# Carga y preparación de datos 
def cargar_y_preparar_datos(ruta_dataset):
    """
    Carga el dataset, separa características y objetivo, y divide en conjuntos de entrenamiento y prueba.
    """
    try:
        df = pd.read_csv(ruta_dataset, sep=';') 
    except FileNotFoundError:
        print(f"Error: El archivo del dataset no fue encontrado en {ruta_dataset}")
        return None, None, None, None, None

    print("Dataset cargado exitosamente.")
    print("Vista previa del Dataset:")
    print(df.head())
    print("\nInformación del Dataset:")
    df.info()

    X = df[CARACTERISTICAS]
    y = df[VARIABLE_OBJETIVO]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    valores_imputacion = X_train.mean()
    joblib.dump(valores_imputacion, RUTA_VALORES_IMPUTACION)
    print(f"\nValores de imputación (medias de los datos de entrenamiento) guardados en {RUTA_VALORES_IMPUTACION}")
    print(valores_imputacion)

    return X_train, X_test, y_train, y_test, valores_imputacion

# Entrenar y Evaluar Modelo 
def entrenar_evaluar_modelo(X_train, X_test, y_train, y_test):
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    print("\nEntrenando el modelo...")
    modelo.fit(X_train, y_train)
    print("Entrenamiento del modelo completo.")

    # Evaluo el modelo
    predicciones = modelo.predict(X_test)
    mse = mean_squared_error(y_test, predicciones) 
    print(f"Evaluación del modelo (Error Cuadrático Medio en el conjunto de prueba): {mse:.4f}")

    # Guardo el modelo entrenado
    joblib.dump(modelo, RUTA_GUARDADO_MODELO)
    print(f"Modelo entrenado guardado en {RUTA_GUARDADO_MODELO}")
    return modelo

# Se define la lógica de predicción
def predecir_calidad_vino(datos_entrada_dict, modelo, valores_imputacion):

    df_entrada = pd.DataFrame(columns=CARACTERISTICAS)
    fila_datos = {}

    print("\nEntrada recibida para predicción:")
    print(datos_entrada_dict)

    for caracteristica in CARACTERISTICAS:
        valor = datos_entrada_dict.get(caracteristica)
        if valor is None or str(valor).strip() == '': 
            fila_datos[caracteristica] = valores_imputacion[caracteristica] 
            print(f"Valor ausente para '{caracteristica}', imputado con {fila_datos[caracteristica]:.4f}")
        else:
            try:
                fila_datos[caracteristica] = float(valor)
            except ValueError: 
                print(f"Error: Entrada no numérica inválida para {caracteristica}: '{valor}'. Usando imputación.")
                fila_datos[caracteristica] = valores_imputacion[caracteristica] 

    df_entrada = pd.DataFrame([fila_datos], columns=CARACTERISTICAS)
    print("\nDatos preparados para el modelo (después de la imputación):")
    print(df_entrada)

    # Se realiza predicción
    calidad_numerica = modelo.predict(df_entrada)[0]
    # Se asegura que la predicción esté dentro del rango 0-10, ya que el modelo podría predecir ligeramente fuera
    calidad_numerica = np.clip(calidad_numerica, 0, 10)


    # Se categoriza la calidad
    if calidad_numerica < 5:
        categoria_cualitativa = "Malo"
    elif calidad_numerica < 7:
        categoria_cualitativa = "Regular"
    else:
        categoria_cualitativa = "Bueno"

    return round(calidad_numerica, 2), categoria_cualitativa # [cite: 13]


if __name__ == "__main__":
   
    print("--- Fase de Entrenamiento del Modelo ---")
    X_train, X_test, y_train, y_test, valores_imputacion_cargados = cargar_y_preparar_datos(RUTA_DATASET)

    modelo_entrenado = None
    if X_train is not None: 
        
        modelo_entrenado = entrenar_evaluar_modelo(X_train, X_test, y_train, y_test)


    if modelo_entrenado and valores_imputacion_cargados is not None:
       
        print("\n\n--- Ejemplo de Fase de Predicción ---")

       
        entrada_muestra_1 = {
            'fixed acidity': 7.4, 'volatile acidity': 0.70, 'citric acid': 0.00,
            'residual sugar': 1.9, 'chlorides': 0.076, 'free sulfur dioxide': 11.0,
            'total sulfur dioxide': 34.0, 'density': 0.9978, 'pH': 3.51,
            'sulphates': 0.56, 'alcohol': 9.4
        }
        puntaje1, categoria1 = predecir_calidad_vino(entrada_muestra_1, modelo_entrenado, valores_imputacion_cargados)
        print(f"Puntaje de Calidad Predicho: {puntaje1}, Categoría: {categoria1}") 

       
        entrada_muestra_2 = {
            'fixed acidity': 8.1, 'volatile acidity': None, 'citric acid': 0.22, 
            'residual sugar': 2.2, 'chlorides': 0.082, 'free sulfur dioxide': None, 
            'total sulfur dioxide': 50.0, 'density': 0.9960, 'pH': 3.27,
            'sulphates': 0.66, 'alcohol': 10.8
        }
        puntaje2, categoria2 = predecir_calidad_vino(entrada_muestra_2, modelo_entrenado, valores_imputacion_cargados)
        print(f"Puntaje de Calidad Predicho: {puntaje2}, Categoría: {categoria2}")

        entrada_muestra_3 = {
            'fixed acidity': None, 'volatile acidity': 0.5, 'citric acid': '', 
            'residual sugar': 2.0, 'chlorides': 0.07, 'free sulfur dioxide': 15.0,
            'total sulfur dioxide': 40.0, 'density': None, 'pH': 3.3,          
            'sulphates': 0.6, 'alcohol': None                                   
        }
        puntaje3, categoria3 = predecir_calidad_vino(entrada_muestra_3, modelo_entrenado, valores_imputacion_cargados)
        print(f"Puntaje de Calidad Predicho: {puntaje3}, Categoría: {categoria3}")

        
        entrada_muestra_4 = {
            'fixed acidity': 7.0, 'volatile acidity': 'alto', 'citric acid': 0.1, 
            'residual sugar': 1.8, 'chlorides': 0.06, 'free sulfur dioxide': 10.0,
            'total sulfur dioxide': 30.0, 'density': 0.995, 'pH': 3.4,
            'sulphates': 0.50, 'alcohol': 10.0
        }
        puntaje4, categoria4 = predecir_calidad_vino(entrada_muestra_4, modelo_entrenado, valores_imputacion_cargados)
        print(f"Puntaje de Calidad Predicho: {puntaje4}, Categoría: {categoria4}")
    else:
        print("No se pudo continuar con las predicciones debido a problemas en el entrenamiento del modelo o la carga de datos.")