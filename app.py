import lzma
from pathlib import Path

import joblib
import streamlit as st

st.set_page_config(
    page_title="Clasificador de Solicitudes Bancarias",
    page_icon="🏦",
    layout="centered",
)

@st.cache_resource
def cargar_modelo():
    ruta = Path(__file__).parent / "modelo_banking77_final.joblib.xz"
    with lzma.open(ruta, "rb") as archivo:
        return joblib.load(archivo)

datos = cargar_modelo()
modelo = datos["modelo"]
intenciones = datos["intenciones"]

st.title("🏦 Clasificador Inteligente de Solicitudes Bancarias")
st.write(
    "Sistema basado en Procesamiento de Lenguaje Natural para "
    "identificar automáticamente la intención de una solicitud bancaria."
)
st.info("El modelo fue entrenado con BANKING77. Escribe consultas en inglés.")

consulta = st.text_area(
    "Escribe la solicitud bancaria:",
    placeholder="I lost my bank card and need a replacement",
    height=120,
)

if st.button("Clasificar solicitud", type="primary"):
    if not consulta.strip():
        st.warning("Debes escribir una consulta.")
    else:
        prediccion = int(modelo.predict([consulta])[0])
        categoria = intenciones[prediccion]
        st.success("Solicitud procesada.")
        st.subheader("Intención identificada")
        st.write(categoria.replace("_", " ").title())
        st.caption("Categoría original: " + categoria)
        st.divider()
        st.write(
            "El resultado es una clasificación automática de apoyo. "
            "No sustituye la revisión humana."
        )

st.divider()
st.subheader("Información del sistema")
st.write("Modelo seleccionado: Linear SVC")
st.write("Representación textual: TF-IDF")
st.write("Categorías disponibles: 77")
st.write("F1 Macro en prueba: 89,03 %")
st.caption(
    "Proyecto académico — Procesamiento de Lenguaje Natural "
    "| Universidad de Cundinamarca"
)
st.warning(
    "No introduzcas contraseñas, números de tarjeta "
    "ni información bancaria personal."
)
