# 🧠 Clustering des unités de conditionnement – Projet Data Science

## 🌟 Objectif

L'objectif de ce projet est d'analyser les produits achetés par des restaurateurs afin d'identifier les opportunités de réduction des coûts. Une dimension clé de l'analyse repose sur le **conditionnement des produits** (ex : 1kg, 500g, bouteille, unité…).

Nous avons cherché à regrouper automatiquement les produits selon leurs types de conditionnement à l’aide d’un modèle de **Machine Learning non supervisé (KMeans)**.

---

## 🛠️ Méthodologie

### 1. ✨ Analyse exploratoire (EDA)
- Suppression des valeurs aberrantes : prix unitaires et quantités négatifs ou extrêmement élevés
- Calcul de la distribution des prix et détection des outliers via histogrammes et percentiles
- Analyse du pourcentage de valeurs manquantes (ex : `conditioning_unit`, `ia_product`, `item_code`)
- Imputation des `conditioning_unit` manquants à partir de la valeur dominante pour un même `ia_product`
- Extraction standardisée des poids depuis les unités (ex: 500g → 0.5kg)
- Comparaison des prix par kg pour repérer les variations liées à la taille du conditionnement

### 2. **Nettoyage et normalisation des unités**
- Regroupement manuel des variantes : `KG`, `kg`, `kilo` → `kg`, `BOT`, `BTL`, `bt` → `bottle`, etc.
- Conservation des 20 unités les plus fréquentes, les autres regroupées sous `other`.

### 3. **Encodage des données**
- Utilisation de **OneHotEncoder** sur les unités nettoyées pour transformer la variable catégorielle en vecteurs numériques exploitables par le modèle.

### 4. **Modèle de clustering KMeans**
- KMeans avec `n_clusters=5` par défaut (configurable), choisi comme point de départ raisonnable.
- Possibilité de déterminer dynamiquement le nombre optimal de clusters via la **méthode du coude**.

### 5. **Réduction de dimension (PCA)**
- Utilisation de PCA pour projeter les données encodées dans un espace 2D pour la visualisation.
- Affichage des clusters colorés selon leurs regroupements dans un graphique clair et interprétable.

---

## 🚀 Utilisation du script

Le script `cluster_conditioning.py` encapsule toute la logique du projet et permet un usage simple en ligne de commande.

### 📦 Commande
```bash
python cluster_conditioning.py --input fichier.csv --output resultat.csv --n_clusters 5
```

### 📌 Paramètres
- `--input` : chemin vers le fichier CSV contenant une colonne `conditioning_unit`
- `--output` : chemin du fichier CSV de sortie avec les clusters ajoutés
- `--n_clusters` : nombre de groupes à former (optionnel, défaut = 5)

---

## 📔 Notebook Google Colab

Une version interactive de ce projet est disponible sous forme de notebook Jupyter, incluant :
- Le nettoyage des données,
- L’analyse exploratoire (EDA),
- Le clustering KMeans avec visualisation PCA,
- L'interprétation des clusters.

🔗 Tu peux l’ouvrir et l’exécuter directement dans Google Colab :

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bastGit387/clustering-conditioning-ml/blob/main/clustering_conditioning.ipynb)

---

## 📊 Résultats et interprétation

Grâce au clustering, les produits ont été automatiquement répartis dans des groupes cohérents :
- Groupes dominés par des unités `kg`, `bottle`, `unit`, `package`, etc.
- Identification facilitée des familles de produits pour une analyse de prix intra-cluster
- Ce regroupement peut être réutilisé pour standardiser des bases produits ou détecter des incohérences tarifaires

---

## ✅ Conclusion

Ce projet démontre comment une approche de Machine Learning simple (KMeans) peut être appliquée avec efficacité à un problème métier concret. L’utilisation d’un script autonome permet une réutilisabilité immédiate sur d'autres jeux de données similaires. Le modèle est rapide, interprétable et facile à ajuster.

🔀 Ce pipeline peut être enrichi dans le futur avec l'extraction de **quantités normalisées** (ex: prix par kilo) pour une analyse plus fine des variations de prix.

---

📂 Auteur : Bastien Di Caro 
🗓️ Date : Mai 2025

