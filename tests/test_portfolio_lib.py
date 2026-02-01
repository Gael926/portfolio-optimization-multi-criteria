"""
Tests unitaires pour la bibliothèque d'optimisation de portefeuille.
Vérifie que l'algorithme NSGA-II produit des portefeuilles valides.
"""

import numpy as np
import pytest
from src.portfolio_lib import optimize_moo, PortfolioProblem

# Fixtures (données de test réutilisables)

@pytest.fixture
def mock_data():
    """Crée des données synthétiques pour les tests (5 actifs)."""
    np.random.seed(42)
    n_assets = 5
    
    # Rendements annuels moyens (entre 5% et 25%)
    mu = np.random.uniform(0.05, 0.25, n_assets)
    
    # Matrice de covariance (symétrique, définie positive)
    A = np.random.rand(n_assets, n_assets)
    sigma = np.dot(A, A.T) * 0.01  # Petite variance
    
    return mu, sigma


# Tests

def test_optimize_moo_returns_valid_weights(mock_data):
    """Vérifie que les poids retournés sont valides (somme = 1)."""
    mu, sigma = mock_data
    k_card = 3
    trans_cost = 0.01
    
    # Exécution (petite population pour vitesse)
    res, weights = optimize_moo(mu, sigma, k_card, trans_cost, pop_size=20, n_gen=10)
    
    # Vérifications
    assert weights.shape[1] == len(mu), "Nombre d'actifs incorrect"
    
    for w in weights:
        # La somme des poids doit être ~1
        assert np.isclose(w.sum(), 1.0, atol=1e-6), f"Somme des poids = {w.sum()}"
        
        # Les poids doivent être >= 0
        assert np.all(w >= 0), "Poids négatifs détectés"


def test_cardinality_constraint(mock_data):
    """Vérifie que la contrainte de cardinalité (K actifs max) est respectée."""
    mu, sigma = mock_data
    k_card = 3
    
    res, weights = optimize_moo(mu, sigma, k_card, trans_cost=0.01, pop_size=20, n_gen=10)
    
    for w in weights:
        # Nombre d'actifs avec poids > 0
        n_active = np.sum(w > 1e-6)
        assert n_active <= k_card, f"Cardinalité violée: {n_active} > {k_card}"


def test_pareto_front_has_multiple_solutions(mock_data):
    """Vérifie que le front de Pareto contient plusieurs solutions."""
    mu, sigma = mock_data
    
    res, weights = optimize_moo(mu, sigma, k_card=3, trans_cost=0.01, pop_size=30, n_gen=20)
    
    # On attend plusieurs solutions non-dominées
    assert len(weights) > 1, "Le front de Pareto devrait contenir plusieurs solutions"


def test_different_k_values_produce_different_results(mock_data):
    """Vérifie que changer K produit des résultats différents."""
    mu, sigma = mock_data
    
    _, weights_k2 = optimize_moo(mu, sigma, k_card=2, trans_cost=0.01, pop_size=20, n_gen=10)
    _, weights_k5 = optimize_moo(mu, sigma, k_card=5, trans_cost=0.01, pop_size=20, n_gen=10)
    
    # Les portefeuilles devraient être différents
    assert not np.allclose(weights_k2[0], weights_k5[0]), "K différents mais mêmes poids"
