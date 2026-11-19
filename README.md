# Classification de troubles mentaux

Projet d'apprentissage automatique realise dans un notebook Python. Il explore un jeu de donnees de reponses comportementales, prepare les variables, puis compare plusieurs modeles pour predire la colonne `Expert Diagnose`.

> **Important :** ce projet est a but educatif. Il ne constitue pas un outil de diagnostic medical et ne doit pas servir a prendre des decisions de sante.

## Contenu

- `projet_bipolarité.ipynb` : exploration, visualisations, pretraitement et comparaison de modeles.
- `Dataset-Mental-Disorders.csv` : donnees utilisees par le notebook (120 lignes et 19 colonnes).
- `article_bipolarité.pdf` : document de reference, a publier uniquement si tu as l'autorisation de le partager.

Le notebook compare notamment la regression logistique, la foret aleatoire, SVM, KNN, l'arbre de decision et des reseaux de neurones.

## Installation et execution

Python 3.11 est recommande. Depuis le dossier du projet, installe les dependances :

```bash
python -m pip install -r requirements.txt
```

Lance ensuite Jupyter :

```bash
jupyter notebook
```

Ouvre `projet_bipolarité.ipynb` et execute les cellules dans l'ordre. Le fichier CSV doit rester dans le meme dossier que le notebook.


