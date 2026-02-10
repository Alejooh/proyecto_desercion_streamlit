# Predicción de deserción estudiantil

Proyecto de minería de datos orientado a la predicción de deserción estudiantil utilizando información académica histórica.

## Objetivo
Desarrollar un modelo predictivo que permita identificar estudiantes con riesgo de deserción en el siguiente período académico, aplicando la metodología CRISP-DM.

## Estructura del proyecto
- `notebooks/01_eda.ipynb`: análisis exploratorio, preparación de datos y modelado.
- `app/app.py`: aplicación web desarrollada en Streamlit para la visualización y predicción.
- `data/processed/data_ml.csv`: dataset procesado utilizado para el entrenamiento y evaluación.
- `models/modelo_desercion_logit.pkl`: modelo entrenado.
- `reports/Informe_Tecnico_Desercion_Estudiantil.pdf`: informe técnico del proyecto.
- `requirements.txt`: dependencias del proyecto.

## Tecnologías utilizadas
- Python 3.12
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib

## Ejecución de la aplicación
```bash
pip install -r requirements.txt
streamlit run app/app.py