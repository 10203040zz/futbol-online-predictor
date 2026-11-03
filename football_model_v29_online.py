"""
Modelo de Predicción de Fútbol v2.9 ONLINE + Inyección de contexto
===================================================================
Versión con AUTOAPRENDIZAJE y soporte para datos externos inyectados
(clima, lesiones, alineaciones, motivación, etc.)

Uso:
    model = FootballOnlineModel()
    model.learn_from_csv("EPL_2020_2026.csv")

    # Predicción con datos inyectados por Grok / bot
    pred = model.predict(
        "Arsenal", "Chelsea",
        extra_features={
            "weather_rain": 1,
            "home_key_injuries": 1,
            "away_key_injuries": 2,
            "home_missing_star": 1,
            "away_missing_star": 0,
            "rest_days_home": 4,
            "rest_days_away": 3,
        }
    )

    # Aprendizaje después del partido (también puede recibir contexto)
    model.update_with_result("Arsenal", "Chelsea", 2, 1, extra_features={...})
"""

import pandas as pd
import numpy as np
from river import compose, linear_model, preprocessing, metrics, optim
from collections import defaultdict
import pickle
import os

# =============================================================================
# MODELO ONLINE CON INYECCIÓN DE CONTEXTO
# =============================================================================
class FootballOnlineModel:
    def __init__(self, elo_k=32, elo_home=60, initial_elo=1500):
        self.elo_k = elo_k
        self.elo_home = elo_home
        self.initial_elo = initial_elo
        
        self.elo = defaultdict(lambda: initial_elo)
        
        self.team_stats = defaultdict(lambda: {
            "matches": 0, "points": 0, "xg_for": 0.0, "xg_against": 0.0,
            "goals_for": 0, "goals_against": 0, "recent": []
        })
        
        # Features por defecto para contexto externo (se pueden ampliar)
        self.default_extra = {
            "weather_rain": 0,
            "weather_temp": 15.0,
            "weather_wind": 0,
            "home_key_injuries": 0,
            "away_key_injuries": 0,
            "home_missing_star": 0,
            "away_missing_star": 0,
            "rest_days_home": 7,
            "rest_days_away": 7,
            "motivation_home": 1.0,
            "motivation_away": 1.0,
        }
        
        self.model_1x2 = compose.Pipeline(
            preprocessing.StandardScaler(),
            linear_model.SoftmaxRegression(
                optimizer=optim.Adam(lr=0.01),
                loss=optim.losses.CrossEntropy()
            )
        )
        
        self.model_over25 = compose.Pipeline(
            preprocessing.StandardScaler(),
            linear_model.LogisticRegression(optimizer=optim.Adam(lr=0.01))
        )
        self.model_btts = compose.Pipeline(
            preprocessing.StandardScaler(),
            linear_model.LogisticRegression(optimizer=optim.Adam(lr=0.01))
        )
        
        self.metric_1x2 = metrics.Accuracy()
        self.n_learned = 0

    def _expected_score(self, elo_a, elo_b):
        return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

    def _update_elo(self, home, away, result):
        eh = self.elo[home] + self.elo_home
        ea = self.elo[away]
        exp_h = self._expected_score(eh, ea)
        exp_a = 1 - exp_h
        self.elo[home] += self.elo_k * (result - exp_h)
        self.elo[away] += self.elo_k * ((1 - result) - exp_a)

    def _get_features(self, home, away, extra_features=None):
        h = self.team_stats[home]
        a = self.team_stats[away]
        
        def avg(stat, key):
            return stat[key] / max(stat["matches"], 1)
        
        feats = {
            "elo_diff": (self.elo[home] + self.elo_home) - self.elo[away],
            "home_elo": self.elo[home],
            "away_elo": self.elo[away],
            "home_form": avg(h, "points"),
            "away_form": avg(a, "points"),
            "home_xg_avg": avg(h, "xg_for"),
            "away_xg_avg": avg(a, "xg_for"),
            "home_xga_avg": avg(h, "xg_against"),
            "away_xga_avg": avg(a, "xg_against"),
            "home_n": h["matches"],
            "away_n": a["matches"],
        }
        
        # Añadir features de contexto (con valores por defecto)
        context = self.default_extra.copy()
        if extra_features:
            context.update(extra_features)
        feats.update(context)
        
        return feats

    def predict(self, home, away, extra_features=None):
        """Predicción con posibilidad de inyectar contexto externo"""
        feats = self._get_features(home, away, extra_features)
        
        p1x2 = self.model_1x2.predict_proba_one(feats)
        if not p1x2:
            p1x2 = {"1": 0.45, "X": 0.25, "2": 0.30}
        
        p_over = self.model_over25.predict_proba_one(feats).get(True, 0.5)
        p_btts = self.model_btts.predict_proba_one(feats).get(True, 0.5)
        
        return {
            "1X2": {k: round(v * 100, 1) for k, v in p1x2.items()},
            "over_2.5": round(p_over * 100, 1),
            "btts": round(p_btts * 100, 1),
            "elo": {"home": round(self.elo[home], 1), "away": round(self.elo[away], 1)},
            "context_used": extra_features or {},
            "features": feats
        }

    def update_with_result(self, home, away, home_goals, away_goals,
                           home_xg=None, away_xg=None, extra_features=None):
        """Autoaprendizaje. Puede recibir el mismo contexto que se usó en la predicción."""
        if home_xg is None:
            home_xg = float(home_goals)
        if away_xg is None:
            away_xg = float(away_goals)
            
        if home_goals > away_goals:
            result_1x2 = "1"
            result_elo = 1.0
        elif home_goals < away_goals:
            result_1x2 = "2"
            result_elo = 0.0
        else:
            result_1x2 = "X"
            result_elo = 0.5
            
        over25 = 1 if (home_goals + away_goals) > 2.5 else 0
        btts = 1 if home_goals > 0 and away_goals > 0 else 0
        
        feats = self._get_features(home, away, extra_features)
        
        pred = self.model_1x2.predict_one(feats) or "1"
        self.metric_1x2.update(result_1x2, pred)
        
        self.model_1x2.learn_one(feats, result_1x2)
        self.model_over25.learn_one(feats, bool(over25))
        self.model_btts.learn_one(feats, bool(btts))
        
        self._update_elo(home, away, result_elo)
        
        for team, gf, ga, xgf, xga, pts in [
            (home, home_goals, away_goals, home_xg, away_xg,
             3 if result_elo == 1 else 1 if result_elo == 0.5 else 0),
            (away, away_goals, home_goals, away_xg, home_xg,
             3 if result_elo == 0 else 1 if result_elo == 0.5 else 0),
        ]:
            s = self.team_stats[team]
            s["matches"] += 1
            s["points"] += pts
            s["xg_for"] += xgf
            s["xg_against"] += xga
            s["goals_for"] += gf
            s["goals_against"] += ga
            s["recent"].append(pts)
            if len(s["recent"]) > 10:
                s["recent"].pop(0)
        
        self.n_learned += 1

    def learn_from_dataframe(self, df):
        df = df.sort_values("date").dropna(subset=["home_goals", "away_goals"])
        for _, row in df.iterrows():
            self.update_with_result(
                row["home_team"], row["away_team"],
                int(row["home_goals"]), int(row["away_goals"]),
                row.get("home_xg"), row.get("away_xg")
            )
        print(f"Aprendidos {len(df)} partidos. Accuracy 1X2: {self.metric_1x2.get():.1%}")

    def learn_from_csv(self, path):
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        self.learn_from_dataframe(df)

    def save(self, path="modelo_online.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Modelo guardado en {path}")

    @staticmethod
    def load(path="modelo_online.pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    print("=== Modelo v2.9 ONLINE con inyección de contexto ===\n")
    model = FootballOnlineModel()
    
    csv_path = "EPL_2020_2026.csv"
    if os.path.exists(csv_path):
        model.learn_from_csv(csv_path)
    
    # Ejemplo de predicción con datos inyectados
    print("\n--- Predicción con contexto ---")
    pred = model.predict(
        "Arsenal", "Chelsea",
        extra_features={
            "weather_rain": 1,
            "home_key_injuries": 1,
            "away_key_injuries": 2,
            "home_missing_star": 1,
        }
    )
    print(pred)
    
    model.save()
