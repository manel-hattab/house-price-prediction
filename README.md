# Prediction des prix immobiliers

Projet de machine learning visant à estimer le prix de vente de maisons à partir de leurs caractéristiques. Le travail comprend l'analyse des données dans un notebook, la comparaison de modèles de régression et une petite application web de prédiction.

## Contenu du projet

- `projet_immobiliers.ipynb` : exploration des données, préparation, entraînement et évaluation des modèles.
- `rapport_immobiliers.pdf` : rapport du projet.
- `house_price_app/backend/` : API Flask et artefacts nécessaires aux prédictions.
- `house_price_app/frontend/` : formulaire web pour comparer les estimations.

Le projet compare notamment :

| Modèle | RMSE (test) | R² (test) |
|---|---:|---:|
| LightGBM | 0.1139 | 0.9092 |
| ANN-4 Residual MLP | 0.1877 | 0.7531 |

Ces métriques sont celles obtenues dans le notebook. LightGBM est le modèle recommandé. La cible est transformée en logarithme pendant l'entraînement; l'API reconvertit les prédictions en prix en dollars.

## Prérequis

- Windows
- Python 3.11 recommandé
- Git

Les dépendances Python de l'API sont listées dans `house_price_app/backend/requirements.txt`.

## Lancer l'application en local

Ouvrir un terminal PowerShell à la racine du dépôt, puis créer l'environnement et installer les dépendances :

```powershell
cd .\house_price_app\backend
py -3.11 -m venv ..\venv
..\venv\Scripts\python.exe -m pip install --upgrade pip
..\venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

### Démarrer l'API

Depuis `house_price_app\backend` :

```powershell
..\venv\Scripts\python.exe .\app.py
```

L'API démarre sur `http://localhost:5000`. Pour vérifier qu'elle fonctionne, ouvrir `http://localhost:5000/health`.



### Ouvrir l'interface web

Dans un deuxième terminal PowerShell, depuis la racine du dépôt :

```powershell
cd .\house_price_app\frontend
..\venv\Scripts\python.exe -m http.server 8000
```

Ouvrir ensuite `http://localhost:8000` dans le navigateur. L'API doit rester lancée dans le premier terminal. Cette configuration sert à une démonstration locale, pas à un déploiement en production.

