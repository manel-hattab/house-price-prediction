# test_api.py  (à la racine de house_price_app/)
import requests, json

BASE = "http://127.0.0.1:5000"

# ── Exemple de maison basé sur les vraies colonnes de ton train.csv
#    (valeurs brutes — le preprocessing s'occupe du reste)
sample = {
    "MSSubClass":   60,
    "MSZoning":     "RL",
    "LotFrontage":  65.0,
    "LotArea":      8450,
    "Street":       "Pave",
    "LotShape":     "Reg",
    "LandContour":  "Lvl",
    "Utilities":    "AllPub",
    "LotConfig":    "Inside",
    "LandSlope":    "Gtl",
    "Neighborhood": "CollgCr",
    "Condition1":   "Norm",
    "Condition2":   "Norm",
    "BldgType":     "1Fam",
    "HouseStyle":   "2Story",
    "OverallQual":  7,
    "OverallCond":  5,
    "YearBuilt":    2003,
    "YearRemodAdd": 2003,
    "RoofStyle":    "Gable",
    "RoofMatl":     "CompShg",
    "Exterior1st":  "VinylSd",
    "Exterior2nd":  "VinylSd",
    "MasVnrType":   "BrkFace",
    "MasVnrArea":   196.0,
    "ExterQual":    "Gd",
    "ExterCond":    "TA",
    "Foundation":   "PConc",
    "BsmtQual":     "Gd",
    "BsmtCond":     "TA",
    "BsmtExposure": "No",
    "BsmtFinType1": "GLQ",
    "BsmtFinSF1":   706,
    "BsmtFinType2": "Unf",
    "BsmtFinSF2":   0,
    "BsmtUnfSF":    150,
    "TotalBsmtSF":  856,
    "Heating":      "GasA",
    "HeatingQC":    "Ex",
    "CentralAir":   "Y",
    "Electrical":   "SBrkr",
    "1stFlrSF":     856,
    "2ndFlrSF":     854,
    "LowQualFinSF": 0,
    "GrLivArea":    1710,
    "BsmtFullBath": 1,
    "BsmtHalfBath": 0,
    "FullBath":     2,
    "HalfBath":     1,
    "Bedroom":      3,
    "Kitchen":      1,
    "KitchenQual":  "Gd",
    "TotRmsAbvGrd": 8,
    "Functional":   "Typ",
    "Fireplaces":   0,
    "GarageType":   "Attchd",
    "GarageYrBlt":  2003.0,
    "GarageFinish": "RFn",
    "GarageCars":   2,
    "GarageArea":   548,
    "GarageQual":   "TA",
    "GarageCond":   "TA",
    "PavedDrive":   "Y",
    "WoodDeckSF":   0,
    "OpenPorchSF":  61,
    "EnclosedPorch":0,
    "3SsnPorch":    0,
    "ScreenPorch":  0,
    "PoolArea":     0,
    "MiscVal":      0,
    "MoSold":       2,
    "YrSold":       2008,
    "SaleType":     "WD",
    "SaleCondition":"Normal"
}

print("=" * 50)

# ── Test 1 : Health
r = requests.get(f"{BASE}/health")
print("1. Health check :", r.json())

# ── Test 2 : LightGBM seul
r = requests.post(f"{BASE}/predict/lgbm", json=sample)
d = r.json()
print(f"\n2. LightGBM prediction :")
print(f"   Prix prédit  : ${d['price_usd']:,.2f}")
print(f"   Log pred     : {d['log_pred']}")
print(f"   R² test      : {d['r2_test']}")

# ── Test 3 : ANN seul
r = requests.post(f"{BASE}/predict/ann", json=sample)
d = r.json()
print(f"\n3. ANN-4 Residual MLP prediction :")
print(f"   Prix prédit  : ${d['price_usd']:,.2f}")
print(f"   Log pred     : {d['log_pred']}")
print(f"   R² test      : {d['r2_test']}")

# ── Test 4 : Les deux
r = requests.post(f"{BASE}/predict/both", json=sample)
d = r.json()
print(f"\n4. Comparaison des deux modèles :")
print(f"   LightGBM          : ${d['LightGBM']['price_usd']:,.2f}  (R²={d['LightGBM']['r2_test']})")
print(f"   ANN-4 Residual MLP: ${d['ANN-4_Residual_MLP']['price_usd']:,.2f}  (R²={d['ANN-4_Residual_MLP']['r2_test']})")
print(f"   Recommandé      : {d['recommended']}")