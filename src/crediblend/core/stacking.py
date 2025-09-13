"""Stacking methods for ensemble learning."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple, Any
import warnings


class StackingBlender:
    """Stacking blender using meta-learner."""
    
    def __init__(self, meta_learner: str = "lr", random_state: Optional[int] = None):
        """Initialize stacking blender.
        
        Args:
            meta_learner: Meta-learner type ('lr' for LogisticRegression, 'ridge' for Ridge)
            random_state: Random state for reproducibility
        """
        self.meta_learner_type = meta_learner
        self.random_state = random_state
        self.meta_learner = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Initialize meta-learner
        if meta_learner == "lr":
            self.meta_learner = LogisticRegression(
                random_state=random_state,
                max_iter=1000,
                solver='liblinear'
            )
        elif meta_learner == "ridge":
            self.meta_learner = Ridge(
                random_state=random_state,
                alpha=1.0
            )
        else:
            raise ValueError(f"Unsupported meta-learner: {meta_learner}")
    
    def fit(self, X_oof: np.ndarray, y_oof: np.ndarray) -> 'StackingBlender':
        """Fit the stacking blender on OOF data.
        
        Args:
            X_oof: OOF predictions matrix (n_samples, n_models)
            y_oof: True labels
            
        Returns:
            Self
        """
        if X_oof.shape[1] < 2:
            raise ValueError("Need at least 2 models for stacking")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_oof)
        
        # Fit meta-learner
        self.meta_learner.fit(X_scaled, y_oof)
        self.is_fitted = True
        
        return self
    
    def predict(self, X_sub: np.ndarray) -> np.ndarray:
        """Predict on submission data.
        
        Args:
            X_sub: Submission predictions matrix (n_samples, n_models)
            
        Returns:
            Blended predictions
        """
        if not self.is_fitted:
            raise ValueError("Stacking blender not fitted yet")
        
        # Scale features
        X_scaled = self.scaler.transform(X_sub)
        
        # Predict
        if self.meta_learner_type == "lr":
            # LogisticRegression returns probabilities
            predictions = self.meta_learner.predict_proba(X_scaled)[:, 1]
        else:
            # Ridge returns continuous values
            predictions = self.meta_learner.predict(X_scaled)
        
        return predictions
    
    def get_coefficients(self) -> Dict[str, float]:
        """Get stacking coefficients.
        
        Returns:
            Dictionary mapping model index to coefficient
        """
        if not self.is_fitted:
            return {}
        
        if hasattr(self.meta_learner, 'coef_'):
            coefs = self.meta_learner.coef_.flatten()
            return {f"model_{i}": float(coefs[i]) for i in range(len(coefs))}
        else:
            return {}


def prepare_stacking_data(oof_data: Dict[str, pd.DataFrame],
                         sub_data: Dict[str, pd.DataFrame],
                         target_col: str = "target") -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Prepare data for stacking.
    
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
        raise ValueError(f"Need at least 2 common models for stacking, got {len(common_models)}")
    
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
            warnings.warn(f"No target column '{target_col}' in {model}, skipping stacking")
            continue
    
    if len(oof_predictions) < 2:
        raise ValueError("Not enough valid OOF data for stacking")
    
    X_oof = np.column_stack(oof_predictions)
    
    # Prepare submission data
    sub_predictions = []
    for model in common_models:
        if model in sub_data:
            sub_predictions.append(sub_data[model]['pred'].values)
    
    X_sub = np.column_stack(sub_predictions)
    
    return X_oof, y_oof, X_sub, common_models


def stacking_blend(oof_data: Dict[str, pd.DataFrame],
                  sub_data: Dict[str, pd.DataFrame],
                  meta_learner: str = "lr",
                  target_col: str = "target",
                  random_state: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Perform stacking blend.
    
    Args:
        oof_data: Dictionary of OOF DataFrames
        sub_data: Dictionary of submission DataFrames
        meta_learner: Meta-learner type ('lr' or 'ridge')
        target_col: Name of target column
        random_state: Random state
        
    Returns:
        Tuple of (blended_predictions, stacking_info)
    """
    print(f"🔄 Performing stacking blend with {meta_learner}...")
    
    # Prepare data
    X_oof, y_oof, X_sub, model_names = prepare_stacking_data(oof_data, sub_data, target_col)
    
    # Initialize and fit blender
    blender = StackingBlender(meta_learner=meta_learner, random_state=random_state)
    blender.fit(X_oof, y_oof)
    
    # Predict
    blended_preds = blender.predict(X_sub)
    
    # Create result DataFrame
    result_df = pd.DataFrame({
        'id': sub_data[model_names[0]]['id'].values,
        'pred': blended_preds
    })
    
    # Prepare stacking info
    stacking_info = {
        'meta_learner': meta_learner,
        'model_names': model_names,
        'coefficients': blender.get_coefficients(),
        'n_models': len(model_names),
        'random_state': random_state
    }
    
    print(f"✅ Stacking completed with {len(model_names)} models")
    print(f"📊 Coefficients: {blender.get_coefficients()}")
    
    return result_df, stacking_info


def cross_validate_stacking(X_oof: np.ndarray, y_oof: np.ndarray,
                          meta_learner: str = "lr",
                          cv: int = 5,
                          random_state: Optional[int] = None) -> Tuple[float, np.ndarray]:
    """Cross-validate stacking performance.
    
    Args:
        X_oof: OOF predictions matrix
        y_oof: True labels
        meta_learner: Meta-learner type
        cv: Number of CV folds
        random_state: Random state
        
    Returns:
        Tuple of (mean_score, cv_scores)
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import roc_auc_score
    
    # Initialize blender
    blender = StackingBlender(meta_learner=meta_learner, random_state=random_state)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_oof)
    
    # Cross-validate
    if meta_learner == "lr":
        # For logistic regression, use predict_proba
        def scorer(estimator, X, y):
            pred_proba = estimator.predict_proba(X)[:, 1]
            return roc_auc_score(y, pred_proba)
    else:
        # For ridge, use predict
        def scorer(estimator, X, y):
            pred = estimator.predict(X)
            return roc_auc_score(y, pred)
    
    cv_scores = cross_val_score(blender.meta_learner, X_scaled, y_oof, 
                               cv=cv, scoring=scorer)
    
    return cv_scores.mean(), cv_scores
