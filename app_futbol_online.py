"""
App Streamlit - Modelo de Fútbol Online (versión móvil-friendly)
================================================================
Funciona en navegador de Android / iOS / PC.

Ejecutar localmente:
    pip install -r requirements.txt
    python -m streamlit run app_futbol_online.py

Para usarla en Android:
1. Súbela a Streamlit Community Cloud (gratis) o Hugging Face Spaces.
2. Abre el enlace en el navegador del móvil.
"""

import streamlit as st
import pandas as pd
import os
from football_model_v29_online import FootballOnlineModel

st.set_page_config(
    page_title="Fútbol Online",
    page_icon="⚽",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("⚽ Fútbol Online")
st.caption("Autoaprendizaje + contexto")

MODEL_FILE = "modelo_online.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_FILE):
        return FootballOnlineModel.load(MODEL_FILE)
    return FootballOnlineModel()

model = load_model()

with st.sidebar:
    st.header("Modelo")
    st.metric("Partidos", model.n_learned)
    st.metric("Accuracy", f"{model.metric_1x2.get():.1%}")
    csv_file = st.file_uploader("CSV histórico", type=["csv"])
    if csv_file and st.button("Entrenar"):
        df = pd.read_csv(csv_file)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        model.learn_from_dataframe(df)
        model.save(MODEL_FILE)
        st.success("Entrenado")
        st.rerun()

st.subheader("Partido")
home = st.text_input("Local", "Arsenal")
away = st.text_input("Visitante", "Chelsea")

with st.expander("Contexto (clima, lesiones...)", expanded=False):
    weather_rain = st.checkbox("Lluvia")
    weather_temp = st.number_input("Temp °C", value=15.0)
    home_inj = st.number_input("Bajas Local", 0, 10, 0)
    away_inj = st.number_input("Bajas Visitante", 0, 10, 0)
    home_star = st.checkbox("Estrella ausente Local")
    away_star = st.checkbox("Estrella ausente Visitante")
    rest_home = st.number_input("Descanso Local", 1, 14, 7)
    rest_away = st.number_input("Descanso Visitante", 1, 14, 7)
    motiv_home = st.slider("Motivación Local", 0.7, 1.3, 1.0)
    motiv_away = st.slider("Motivación Visitante", 0.7, 1.3, 1.0)

extra = {
    "weather_rain": int(weather_rain),
    "weather_temp": weather_temp,
    "home_key_injuries": home_inj,
    "away_key_injuries": away_inj,
    "home_missing_star": int(home_star),
    "away_missing_star": int(away_star),
    "rest_days_home": rest_home,
    "rest_days_away": rest_away,
    "motivation_home": motiv_home,
    "motivation_away": motiv_away,
}

if st.button("🔮 Predecir", type="primary", use_container_width=True) and home and away:
    result = model.predict(home, away, extra_features=extra)
    st.session_state.last = (home, away, extra)

    c1, c2, c3 = st.columns(3)
    p = result["1X2"]
    c1.metric("1", f"{p.get('1', 0)}%")
    c2.metric("X", f"{p.get('X', 0)}%")
    c3.metric("2", f"{p.get('2', 0)}%")
    st.metric("Over 2.5", f"{result['over_2.5']}%")
    st.metric("BTTS", f"{result['btts']}%")

if "last" in st.session_state:
    st.divider()
    st.subheader("Resultado real")
    h, a, _ = st.session_state.last
    hg = st.number_input("Goles Local", 0, 10, 1)
    ag = st.number_input("Goles Visitante", 0, 10, 1)
    if st.button("Actualizar modelo", use_container_width=True):
        model.update_with_result(h, a, int(hg), int(ag), extra_features=extra)
        model.save(MODEL_FILE)
        st.success("Modelo actualizado")
        st.rerun()
