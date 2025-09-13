"""Weight optimization for ensemble blending."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Callable, Any
import warnings
from itertools import product
import random


class WeightOptimizer:
    """Weight optimizer for ensemble blending."""
    
    def __init__(self, scorer: Callable, random_state: Optional[int] = None):
        """Initialize weight optimizer.
        
        Args:
            scorer: Scorer function that takes (y_true, y_pred) and returns score
            random_state: Random state for reproducibility
        """
        self.scorer = scorer
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)
    
    def objective(self, weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Objective function for optimization.
        
        Args:
            weights: Weight vector (n_models,)
            X: Prediction matrix (n_samples, n_models)
            y: True labels (n_samples,)
            
        Returns:
            Negative score (for maximization)
        """
        # Ensure weights sum to 1
        weights = weights / np.sum(weights)
        
        # Compute weighted prediction
        y_pred = np.dot(X, weights)
        
        # Compute score
        try:
            score = self.scorer.score(y, y_pred) if hasattr(self.scorer, 'score') else self.scorer(y, y_pred)
            return -score  # Negative for minimization
        except Exception as e:
            warnings.warn(f"Error in objective function: {e}")
            return 1e6  # Large penalty for invalid predictions
    
    def optimize_single_restart(self, X: np.ndarray, y: np.ndarray, 
                              n_models: int) -> Tuple[np.ndarray, float]:
        """Single optimization restart.
        
        Args:
            X: Prediction matrix (n_samples, n_models)
            y: True labels (n_samples,)
            n_models: Number of models
            
        Returns:
            Tuple of (best_weights, best_score)
        """
        # Random initialization
        weights_init = np.random.dirichlet(np.ones(n_models))
        
        # Constraints: sum(weights) = 1, weights >= 0
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_models)]
        
        # Optimize
        try:
            result = minimize(
                self.objective,
                weights_init,
                args=(X, y),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if result.success:
                weights = result.x / np.sum(result.x)  # Normalize
                score = -result.fun
                return weights, score
            else:
                # Fallback to uniform weights
                uniform_weights = np.ones(n_models) / n_models
                uniform_score = -self.objective(uniform_weights, X, y)
                return uniform_weights, uniform_score
                
        except Exception as e:
            warnings.warn(f"Optimization failed: {e}")
            # Fallback to uniform weights
            uniform_weights = np.ones(n_models) / n_models
            uniform_score = -self.objective(uniform_weights, X, y)
            return uniform_weights, uniform_score
    
    def optimize_parallel(self, X: np.ndarray, y: np.ndarray,
                         n_restarts: int = 16, max_workers: int = 4) -> Tuple[np.ndarray, float, Dict]:
        """Parallel optimization with multiple restarts.
        
        Args:
            X: Prediction matrix (n_samples, n_models)
            y: True labels (n_samples,)
            n_restarts: Number of random restarts
            max_workers: Maximum number of workers
            
        Returns:
            Tuple of (best_weights, best_score, optimization_info)
        """
        n_models = X.shape[1]
        
        print(f"🔍 Optimizing weights with {n_restarts} restarts...")
        
        # Parallel optimization
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all restarts
            futures = []
            for _ in range(n_restarts):
                future = executor.submit(self.optimize_single_restart, X, y, n_models)
                futures.append(future)
            
            # Collect results
            for i, future in enumerate(as_completed(futures)):
                try:
                    weights, score = future.result()
                    results.append((weights, score))
                    print(f"  Restart {i+1}/{n_restarts}: score = {score:.6f}")
                except Exception as e:
                    warnings.warn(f"Restart {i+1} failed: {e}")
        
        if not results:
            raise RuntimeError("All optimization restarts failed")
        
        # Find best result
        best_idx = np.argmax([score for _, score in results])
        best_weights, best_score = results[best_idx]
        
        # Prepare optimization info
        all_scores = [score for _, score in results]
        optimization_info = {
            'n_restarts': n_restarts,
            'best_score': best_score,
            'mean_score': np.mean(all_scores),
            'std_score': np.std(all_scores),
            'min_score': np.min(all_scores),
            'max_score': np.max(all_scores),
            'convergence_rate': len([s for s in all_scores if s > best_score * 0.99]) / len(all_scores)
        }
        
        print(f"✅ Best score: {best_score:.6f} (mean: {np.mean(all_scores):.6f})")
        
        return best_weights, best_score, optimization_info


def prepare_weight_data(oof_data: Dict[str, pd.DataFrame],
                       sub_data: Dict[str, pd.DataFrame],
                       target_col: str = "target") -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Prepare data for weight optimization.
    
    Args:
        oof_data: Dictionary of OOF DataFrames
        sub_data: Dictionary of submission DataFrames
        target_col: Name of target column in OOF data
        
    Returns:
        Tuple of (X_oof, y_oof, X_sub, model_names)
    """
    # Get common models
    oof_models = set(oof_data.keys())
    sub_models = set(sub_data.keys())
    common_models = list(oof_models.intersection(sub_models))
    
    if len(common_models) < 2:
        raise ValueError(f"Need at least 2 common models for weight optimization, got {len(common_models)}")
    
    # Sort for consistency
    common_models.sort()
    
    # Prepare OOF data
    oof_predictions = []
    y_oof = None
    
    for model in common_models:
        df = oof_data[model]
        if target_col in df.columns:
            oof_predictions.append(df['pred'].values)
            if y_oof is None:
                y_oof = df[target_col].values
        else:
            warnings.warn(f"No target column '{target_col}' in {model}, skipping weight optimization")
            continue
    
    if len(oof_predictions) < 2:
        raise ValueError("Not enough valid OOF data for weight optimization")
    
    X_oof = np.column_stack(oof_predictions)
    
    # Prepare submission data
    sub_predictions = []
    for model in common_models:
        if model in sub_data:
            sub_predictions.append(sub_data[model]['pred'].values)
    
    X_sub = np.column_stack(sub_predictions)
    
    return X_oof, y_oof, X_sub, common_models


def optimize_weights(oof_data: Dict[str, pd.DataFrame],
                    sub_data: Dict[str, pd.DataFrame],
                    scorer: Callable,
                    target_col: str = "target",
                    n_restarts: int = 16,
                    max_workers: int = 4,
                    random_state: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Optimize ensemble weights.
    
    Args:
        oof_data: Dictionary of OOF DataFrames
        sub_data: Dictionary of submission DataFrames
        scorer: Scorer function
        target_col: Name of target column
        n_restarts: Number of random restarts
        max_workers: Maximum number of workers
        random_state: Random state
        
    Returns:
        Tuple of (weighted_predictions, weight_info)
    """
    print(f"⚖️  Optimizing ensemble weights...")
    
    # Prepare data
    X_oof, y_oof, X_sub, model_names = prepare_weight_data(oof_data, sub_data, target_col)
    
    # Initialize optimizer
    optimizer = WeightOptimizer(scorer, random_state=random_state)
    
    # Optimize weights
    best_weights, best_score, opt_info = optimizer.optimize_parallel(
        X_oof, y_oof, n_restarts=n_restarts, max_workers=max_workers
    )
    
    # Apply weights to submission data
    weighted_preds = np.dot(X_sub, best_weights)
    
    # Create result DataFrame
    result_df = pd.DataFrame({
        'id': sub_data[model_names[0]]['id'].values,
        'pred': weighted_preds
    })
    
    # Prepare weight info
    weight_info = {
        'model_names': model_names,
        'weights': {model_names[i]: float(best_weights[i]) for i in range(len(model_names))},
        'best_score': best_score,
        'optimization_info': opt_info,
        'n_models': len(model_names)
    }
    
    print(f"📊 Optimized weights: {weight_info['weights']}")
    
    return result_df, weight_info


def validate_weights(weights: Dict[str, float], 
                    X_oof: np.ndarray, y_oof: np.ndarray,
                    scorer: Callable) -> float:
    """Validate weights on OOF data.
    
    Args:
        weights: Dictionary of model weights
        X_oof: OOF prediction matrix
        y_oof: True labels
        scorer: Scorer function
        
    Returns:
        Validation score
    """
    # Convert weights to array
    weight_array = np.array(list(weights.values()))
    weight_array = weight_array / np.sum(weight_array)  # Normalize
    
    # Compute weighted prediction
    y_pred = np.dot(X_oof, weight_array)
    
    # Compute score
    return scorer(y_oof, y_pred)
