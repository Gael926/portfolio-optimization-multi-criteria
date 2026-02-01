import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.repair import Repair
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

class DiverseSampling(Sampling):
    """
    Génère des solutions initiales avec des poids diversifiés (Dirichlet).
    Évite le problème des poids uniformes qu'on avait initialementen créant des distributions variées.
    """
    def __init__(self, k_card):
        super().__init__()
        self.k_card = k_card
    
    def _do(self, problem, n_samples, **kwargs):
        n_var = problem.n_var
        X = np.zeros((n_samples, n_var))
        
        for i in range(n_samples):
            # Sélection aléatoire de K actifs
            selected_indices = np.random.choice(n_var, size=self.k_card, replace=False)
            
            # Distribution Dirichlet pour des poids très variés (alpha < 1 = concentré)
            # Alpha petit = quelques gros poids, beaucoup de petits
            alpha = np.random.uniform(0.3, 0.8)  # Varie entre concentré et diversifié
            weights = np.random.dirichlet(np.ones(self.k_card) * alpha)
            
            X[i, selected_indices] = weights
        
        return X


class CardinalityRepair(Repair):
    """
    Répare les solutions pour respecter la contrainte de cardinalité (K actifs max)
    tout en préservant la diversité des poids.
    """
    def __init__(self, k_card):
        super().__init__()
        self.k_card = k_card
    
    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            x = X[i]
            
            # Identifier les K plus grands poids
            idx_sorted = np.argsort(x)[::-1]  # Décroissant
            idx_top_k = idx_sorted[:self.k_card]
            
            # Créer un nouveau vecteur avec seulement les top K
            x_new = np.zeros_like(x)
            x_new[idx_top_k] = x[idx_top_k]
            
            # Normaliser pour que la somme = 1 (budget)
            total = np.sum(x_new)
            if total > 1e-9:
                x_new = x_new / total
            else:
                # Fallback: distribution uniforme sur les K sélectionnés
                x_new[idx_top_k] = 1.0 / self.k_card
            
            X[i] = x_new
        
        return X


# Portfolio Problem Definition

class PortfolioProblem(ElementwiseProblem):
    def __init__(self, mu, sigma, k_card=10, trans_cost=0.005, w_prev=None):
        """
        Optimisation Multi-Objectifs avec contraintes de Cardinalité et Coûts.
        Objectifs:
        1. Maximiser Rendement (-f1, car pymoo minimise)
        2. Minimiser Risque (f2 = variance)
        3. Minimiser Coûts de Transaction (f3)
        """
        self.mu = mu.values if hasattr(mu, 'values') else mu
        self.sigma = sigma.values if hasattr(sigma, 'values') else sigma
        self.k_card = k_card
        self.trans_cost = trans_cost
        self.w_prev = w_prev if w_prev is not None else np.zeros(len(self.mu))
        
        super().__init__(
            n_var=len(self.mu),
            n_obj=3,
            n_ieq_constr=0,
            xl=0.0,
            xu=1.0
        )

    def _evaluate(self, x, out, *args, **kwargs):
        # x est déjà réparé par CardinalityRepair (somme = 1, K actifs)
        w = x
        
        # f1: Rendement (Minimiser -R pour maximiser R)
        f1 = -np.dot(w, self.mu)
        
        # f2: Risque (Variance du portefeuille)
        f2 = np.dot(w, np.dot(self.sigma, w))
        
        # f3: Coûts de Transaction (Turnover × taux)
        turnover = np.sum(np.abs(w - self.w_prev))
        f3 = turnover * self.trans_cost
        
        out["F"] = [f1, f2, f3]


# Main Optimization Function

def optimize_moo(mu, sigma, k_card, trans_cost, w_prev=None, pop_size=100, n_gen=100):
    """
    Lance l'optimisation NSGA-II avec opérateurs personnalisés.
    
    Returns:
        res: Résultat pymoo (Front de Pareto)
        final_weights: Matrice numpy des poids finaux pour chaque solution
    """
    problem = PortfolioProblem(
        mu, sigma, 
        k_card=k_card, 
        trans_cost=trans_cost, 
        w_prev=w_prev
    )
    
    # Opérateurs personnalisés pour éviter le bug des poids uniformes
    sampling = DiverseSampling(k_card)
    repair = CardinalityRepair(k_card)
    
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=sampling,
        repair=repair,
        eliminate_duplicates=True
    )
    
    termination = get_termination("n_gen", n_gen)
    
    res = minimize(
        problem, 
        algorithm, 
        termination, 
        seed=42,  # Reproductibilité
        verbose=False
    )
    
    # Les poids sont déjà normalisés grâce au Repair mais on s'assure de la cohérence finale
    final_weights = []
    for x in res.X:
        # Double vérification de la normalisation
        w = x.copy()
        w[w < 1e-6] = 0  # Nettoyer les très petits poids
        total = np.sum(w)
        if total > 0:
            w = w / total
        final_weights.append(w)
    
    return res, np.array(final_weights)

