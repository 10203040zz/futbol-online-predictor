# Fútbol Online Predictor

Modelo de predicción de fútbol con **autoaprendizaje** (River) + inyección de contexto (clima, lesiones, motivación...).

## Archivos

- `app_futbol_online.py` → App Streamlit (móvil-friendly)
- `football_model_v29_online.py` → Modelo online
- `requirements.txt`

## Desplegar en Streamlit Cloud

1. Este repositorio ya está listo.
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta este repo.
4. Archivo principal: `app_futbol_online.py`
5. Despliega.

## Uso local

```bash
pip install -r requirements.txt
python -m streamlit run app_futbol_online.py
```

El modelo aprende de cada resultado que introduzcas después del partido.
