import streamlit as st
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from predictor_vino import predecir_calidad_vino, CARACTERISTICAS, RUTA_GUARDADO_MODELO, RUTA_VALORES_IMPUTACION

st.set_page_config(page_title="Evaluador de Calidad de Vino", layout="wide")


@st.cache_resource 
def cargar_recursos():
    try:
        modelo = joblib.load(RUTA_GUARDADO_MODELO)
        valores_imputacion = joblib.load(RUTA_VALORES_IMPUTACION)
        return modelo, valores_imputacion
    except FileNotFoundError:
        st.error(f"Error: No se encontró el modelo ('{RUTA_GUARDADO_MODELO}') o los valores de imputación ('{RUTA_VALORES_IMPUTACION}'). "
                 "Asegúrate de que el script del backend ('predictor_vino_nucleo.py') se haya ejecutado al menos una vez "
                 "para generar estos archivos y que estén en la ruta correcta.")
        return None, None

modelo_cargado, valores_imputacion_cargados = cargar_recursos()

# Interfaz gráfica


st.title("🍷 Sistema de Evaluación de Calidad de Vino con IA")

st.markdown("""
Esta aplicación utiliza un modelo de Machine Learning para predecir la calidad de un vino
basándose en sus características fisicoquímicas.
**Instrucciones:**
1.  Ingrese los valores conocidos del vino en los campos de abajo.
2.  Si alguna medición no está disponible, puede dejar el campo vacío. El sistema lo manejará.
3.  Presione el botón "Evaluar Calidad del Vino" para obtener la predicción.
""")

# Formulario de Entrada de Datos
if modelo_cargado is not None and valores_imputacion_cargados is not None:
    with st.form("formulario_vino"):
        st.subheader("Ingrese los Parámetros Fisicoquímicos del Vino")

        col1, col2 = st.columns(2)
        
        # Diccionario para guardar las entradas del usuario
        entradas_usuario = {}

        campos_por_columna = (len(CARACTERISTICAS) + 1) // 2

        for i, caracteristica in enumerate(CARACTERISTICAS):
            
            nombre_legible = caracteristica.replace('_', ' ').title()
            placeholder_text = f"Ej: {valores_imputacion_cargados[caracteristica]:.2f}" # Se sugiere un valor típico

            if i < campos_por_columna:
                with col1:
                    entradas_usuario[caracteristica] = st.text_input(
                        label=nombre_legible,
                        key=caracteristica,
                        placeholder=placeholder_text
                    )
            else:
                with col2:
                    entradas_usuario[caracteristica] = st.text_input(
                        label=nombre_legible,
                        key=caracteristica,
                        placeholder=placeholder_text
                    )
        
        # Botón para enviar el formulario
        boton_evaluar = st.form_submit_button("Evaluar Calidad del Vino")

    # Lógica de Predicción y Visualización de Resultados
    if boton_evaluar:
        # Se convierte las entradas vacías a None para la función de predicción
        datos_para_predecir = {
            car: valor if valor.strip() != "" else None
            for car, valor in entradas_usuario.items()
        }

        st.write("---") 
        st.subheader("Resultado de la Evaluación")

        puntaje_numerico, categoria_cualitativa = predecir_calidad_vino(
            datos_para_predecir,
            modelo_cargado,
            valores_imputacion_cargados
        )

        # Mostramos los resultados
        st.metric(label="Puntaje Numérico Estimado", value=f"{puntaje_numerico:.2f} / 10") 
        
        if categoria_cualitativa == "Bueno":
            st.success(f"Categoría de Calidad: **{categoria_cualitativa}** 👍") 
        elif categoria_cualitativa == "Regular":
            st.warning(f"Categoría de Calidad: **{categoria_cualitativa}** 😐")
        else: 
            st.error(f"Categoría de Calidad: **{categoria_cualitativa}** 👎")

        # valores que fueron imputados
        valores_imputados_info = []
        for car, valor_original in datos_para_predecir.items():
            if valor_original is None:
                valor_usado = valores_imputacion_cargados[car]
                valores_imputados_info.append(f"- *{car.replace('_', ' ').title()}*: Usado valor por defecto ({valor_usado:.3f})")
        
        if valores_imputados_info:
            st.info("Información sobre Datos Ausentes:\n" + "\n".join(valores_imputados_info))

else:
    st.warning("La aplicación no puede iniciar porque faltan los archivos del modelo o de imputación. "
               "Por favor, ejecuta primero el script del backend para generarlos.")

st.markdown("---")
st.caption("Taller de Evaluación de Calidad de Vino")