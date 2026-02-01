
# Optimisation de Portefeuille Multi-Objectifs (NSGA-II)

Ce projet implémente une approche progressive pour l'optimisation de portefeuille d'actifs financiers, allant de la théorie moderne classique (Markowitz) à des algorithmes génétiques avancés (NSGA-II) prenant en compte des contraintes réalistes.

Le projet inclut une **application interactive Streamlit** permettant d'explorer le Front de Pareto en 3D et de construire des portefeuilles optimisés.

## Structure du Projet

Le projet est organisé en trois niveaux de complexité croissante :

### Niveau 1 : L'Approche Classique
*   **Notebook :** `notebooks/niveau_1_markowitz.ipynb`
*   **Objectif :** Optimisation moyenne-variance classique (Harry Markowitz).
*   **Résultats :** Calcul de la Frontière Efficiente théorique et du portefeuille Tangent (Max Sharpe).
*   **Limitations :** Ne prend pas en compte les coûts de transaction ni les contraintes de cardinalité (nombre d'actifs).

### Niveau 2 : L'Algorithme Génétique (NSGA-II)
*   **Notebook :** `notebooks/niveau_2_couts_card.ipynb`
*   **Objectif :** Optimisation Multi-Objectifs réaliste.
    1.  Maximiser le Rendement.
    2.  Minimiser le Risque.
    3.  **Minimiser les Coûts de Transaction** (vs portefeuille précédent).
*   **Contrainte :** **Cardinalité** (ex: ne sélectionner que 10 actifs parmi 190).
*   **Technologie :** Utilisation de la librairie `pymoo` pour l'algorithme génétique NSGA-II.
*   **Comparaison :** Une section compare NSGA-II à une recherche aléatoire (Monte Carlo) pour démontrer la supériorité de l'IA.

### Niveau 3 : Application Interactive (Dashboard)
*   **Fichier :** `streamlit_app.py`
*   **Fonctionnalités :**
    *   Visualisation interactive du Front de Pareto en 3D (Risque, Rendement, Coûts).
    *   Sélection dynamique des contraintes (Cardinalité, budget de coûts).
    *   Analyse détaillée des portefeuilles sélectionnés (Composition, Métriques).

---

## Installation et Démarrage

### Pré-requis
Une des deux options suivantes :
*   **Option 1 (Docker)** : [Docker Desktop](https://www.docker.com/products/docker-desktop) installé.
*   **Option 2 (Local)** : Python 3.12+ installé.

### Installation

#### Via Docker (Recommandé)
Pour lancer le projet de manière robuste (gestion automatique des ports) :

```bash
# Sur Windows (PowerShell)
.\run.ps1

# Sur Mac ou Linux
chmod +x run.sh && ./run.sh
```

#### Via Python (Local)
Si vous préférez l'installation manuelle :

```bash
pip install -r requirements.txt
python src/prepare_data.py
python -m streamlit run streamlit_app.py
```
---

### Explorer les Notebooks
Pour comprendre la logique et voir les analyses détaillées :
1.  Exécutez `notebooks/niveau_0_data.ipynb` pour la préparation des données.
2.  Exécutez `notebooks/niveau_1_markowitz.ipynb` pour la théorie de base.
3.  Exécutez `notebooks/niveau_2_couts_card.ipynb` pour l'optimisation avancée (IA).

---

## Fonctionnalités Clés

*   **Optimisation Réaliste :** Prise en compte des frais de courtage qui impactent la performance réelle.
*   **Diversification Intelligente :** L'algorithme sélectionne les meilleurs actifs (Cardinalité $K$) pour réduire le risque sans diluer la performance.
*   **Visualisation 3D :** Comprendre les compromis (trade-offs) entre Risque, Rendement et Coûts grâce à `plotly`.