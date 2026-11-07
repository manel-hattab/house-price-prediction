# backend/app.py
import os
import joblib
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS
from preprocessing import preprocess_input, preprocess_for_ann

app = Flask(__name__)
CORS(app)

# ── INPUT_DIM : nombre exact de colonnes après ton encoding
#    (261 d'après les outputs de ton notebook)
INPUT_DIM = 260

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(x + self.block(x))

class ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(INPUT_DIM, 256), nn.BatchNorm1d(256), nn.ReLU()
        )
        self.res_blocks = nn.Sequential(
            ResBlock(256), ResBlock(256), ResBlock(256)
        )
        self.head = nn.Linear(256, 1)
    def forward(self, x):
        x = self.input_proj(x)
        x = self.res_blocks(x)
        return self.head(x)

# ════════════════════════════════════════
# Chargement des modèles au démarrage
# ════════════════════════════════════════
BASE = os.path.dirname(__file__)

lgbm_model = joblib.load(os.path.join(BASE, "models/lgbm_model.pkl"))
scaler     = joblib.load(os.path.join(BASE, "models/scaler.pkl"))

ann_model = ResidualMLP()
ann_model.load_state_dict(
    torch.load(
        os.path.join(BASE, "models/best_ann_ANN-4_Residual_MLP.pt"),
        map_location="cpu"
    )
)
ann_model.eval()

print("✅ LightGBM chargé")
print("✅ ANN-4 Residual MLP chargé")
print("✅ StandardScaler chargé")

# ════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models": ["LightGBM", "ANN-4 Residual MLP"],
        "input_dim": INPUT_DIM
    })


@app.route("/predict/lgbm", methods=["POST"])
def predict_lgbm():
    try:
        data = request.get_json()
        X = preprocess_input(data)
        log_pred  = lgbm_model.predict(X)[0]        # prédiction en log scale
        price_usd = float(np.expm1(log_pred))        # np.expm1 = inverse de np.log1p
        return jsonify({
            "model":     "LightGBM",
            "price_usd": round(price_usd, 2),
            "log_pred":  round(float(log_pred), 4),
            "r2_test":   0.9092,
            "rmse_test": 0.1139
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict/ann", methods=["POST"])
def predict_ann():
    """
    Prédit avec ANN-4 Residual MLP.
    preprocess_for_ann → StandardScaler → tensor → ann_model → np.expm1
    """
    try:
        data = request.get_json()
        X_sc = preprocess_for_ann(data, scaler)   # même que cell 87
        tensor = torch.tensor(X_sc, dtype=torch.float32)
        with torch.no_grad():
            log_pred = ann_model(tensor).item()
        price_usd = float(np.expm1(log_pred))
        return jsonify({
            "model":     "ANN-4 Residual MLP",
            "price_usd": round(price_usd, 2),
            "log_pred":  round(log_pred, 4),
            "r2_test":   0.7531,
            "rmse_test": 0.1877
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict/both", methods=["POST"])
def predict_both():
    """Appel les 2 modèles et retourne la comparaison."""
    try:
        data = request.get_json()

        # LightGBM
        X_lgbm   = preprocess_input(data)
        log_lgbm  = lgbm_model.predict(X_lgbm)[0]
        price_lgbm = float(np.expm1(log_lgbm))

        # ANN
        X_sc     = preprocess_for_ann(data, scaler)
        tensor   = torch.tensor(X_sc, dtype=torch.float32)
        with torch.no_grad():
            log_ann = ann_model(tensor).item()
        price_ann = float(np.expm1(log_ann))

        return jsonify({
            "LightGBM": {
                "price_usd": round(price_lgbm, 2),
                "r2_test":   0.9092
            },
            "ANN-4_Residual_MLP": {
                "price_usd": round(price_ann, 2),
                "r2_test":   0.7531
            },
            "recommended": "LightGBM"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)