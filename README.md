# Clasificación de solicitudes bancarias con BANKING77

Proyecto académico de Procesamiento de Lenguaje Natural de la Universidad de Cundinamarca.

## Objetivo
Identificar automáticamente la intención de una consulta bancaria escrita en inglés entre las 77 categorías de BANKING77.

## Modelo y evaluación
- Representación: TF-IDF con unigramas y bigramas.
- Clasificador seleccionado: Linear SVC, C=1, min_df=2.
- Comparación en validación: Naive Bayes, Logistic Regression, Linear SVC y SGD Classifier.
- Evaluación final: 3.080 consultas del conjunto de prueba oficial; Accuracy y F1 Macro de 0,8903.
- El modelo final se entrenó con 9.992 consultas de desarrollo depuradas.

## Aplicación
La interfaz está en `app.py`. El archivo `modelo_banking77_final.joblib.xz` contiene exactamente el mismo modelo serializado de Google Colab, comprimido con XZ para facilitar su almacenamiento en GitHub. La aplicación lo descomprime en memoria al iniciarse.

Ejecutar localmente:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Para Streamlit Community Cloud, desplegar desde la rama `main` con archivo principal `app.py`. Elegir Python 3.13, compatible con las dependencias fijadas.

## Alcance y límites
El prototipo clasifica intenciones; no ejecuta operaciones bancarias ni está validado con clientes reales de una entidad financiera colombiana. Utilizar consultas en inglés y no incluir datos bancarios personales. Linear SVC no genera probabilidades calibradas de forma nativa.

Datos: https://huggingface.co/datasets/PolyAI/banking77

Notebook: https://colab.research.google.com/drive/1vM5nerYagnb7pXHB2q477riqmbTcYjTv
