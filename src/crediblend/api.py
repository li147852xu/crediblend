"""Stable Python API for CrediBlend."""

from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, field_validator
from pathlib import Path
import warnings

from .core.metrics import Scorer
from .core.blend import mean_blend, rank_mean_blend, logit_mean_blend, get_best_blend
from .core.weights import optimize_weights
from .core.stacking import stacking_blend
from .core.decorrelate import filter_redundant_models


class BlendConfig(BaseModel):
    """Configuration for blending operations."""
    
    method: str = Field(default="mean", description="Blending method")
    metric: str = Field(default="auc", description="Evaluation metric")
    target_col: str = Field(default="target", description="Target column name")
    random_state: Optional[int] = Field(default=None, description="Random seed")
    
    # Advanced options
    decorrelate: bool = Field(default=False, description="Enable decorrelation")
    correlation_threshold: float = Field(default=0.8, description="Correlation threshold")
    stacking: Optional[str] = Field(default=None, description="Stacking method (lr/ridge)")
    weight_search: bool = Field(default=False, description="Enable weight optimization")
    search_params: Dict[str, int] = Field(default_factory=lambda: {"iters": 200, "restarts": 16})
    
    @field_validator('method')
    @classmethod
    def validate_method(cls, v):
        valid_methods = ['mean', 'rank_mean', 'logit_mean', 'best_single', 'weighted', 'stacking']
        if v not in valid_methods:
            raise ValueError(f"Invalid method: {v}. Must be one of {valid_methods}")
        return v
    
    @field_validator('metric')
    @classmethod
    def validate_metric(cls, v):
        valid_metrics = ['auc', 'mse', 'mae']
        if v not in valid_metrics:
            raise ValueError(f"Invalid metric: {v}. Must be one of {valid_metrics}")
        return v


class BlendModel(BaseModel):
    """Trained blending model."""
    
    method: str
    config: BlendConfig
    weights: Optional[Dict[str, float]] = None
    stacking_info: Optional[Dict[str, Any]] = None
    decorrelation_info: Optional[Dict[str, Any]] = None
    oof_metrics: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    best_single_model: Optional[str] = None
    improvement_over_best_single: Optional[float] = None
    
    model_config = {"arbitrary_types_allowed": True}


class BlendResult(BaseModel):
    """Result of blending operation."""
    
    predictions: pd.DataFrame
    model: BlendModel
    improvement_over_best_single: Optional[float] = None
    warnings: List[str] = Field(default_factory=list)
    
    model_config = {"arbitrary_types_allowed": True}


def fit_blend(oof_frames: List[pd.DataFrame], 
              method: str = "mean",
              config: Optional[BlendConfig] = None,
              **kwargs) -> BlendModel:
    """Fit a blending model on OOF data.
    
    Args:
        oof_frames: List of OOF DataFrames with columns [id, pred, target, fold?]
        method: Blending method ('mean', 'rank_mean', 'logit_mean', 'best_single', 'weighted', 'stacking')
        config: Optional configuration object
        **kwargs: Additional configuration parameters
        
    Returns:
        Trained BlendModel
        
    Raises:
        ValueError: If input data is invalid or method is not supported
    """
    if not oof_frames:
        raise ValueError("At least one OOF frame is required")
    
    # Merge configuration
    if config is None:
        config = BlendConfig(method=method, **kwargs)
    else:
        # Update config with kwargs, but don't override method if config already has one
        config_dict = config.model_dump()
        config_dict.update(kwargs)
        if method != 'mean':  # Only override if method is explicitly provided
            config_dict['method'] = method
        config = BlendConfig(**config_dict)
    
    # Set random seed if provided
    if config.random_state is not None:
        np.random.seed(config.random_state)
        import random
        random.seed(config.random_state)
    
    # Validate OOF frames
    for i, df in enumerate(oof_frames):
        if 'id' not in df.columns or 'pred' not in df.columns:
            raise ValueError(f"OOF frame {i} missing required columns 'id' or 'pred'")
        if config.target_col not in df.columns:
            raise ValueError(f"OOF frame {i} missing target column '{config.target_col}'")
    
    # Create model names
    model_names = [f"model_{i}" for i in range(len(oof_frames))]
    oof_data = {name: df for name, df in zip(model_names, oof_frames)}
    
    # Initialize scorer
    scorer = Scorer(metric=config.metric)
    
    # Compute OOF metrics
    oof_metrics = {}
    for name, df in oof_data.items():
        if 'fold' in df.columns:
            # Per-fold metrics
            fold_scores = []
            for fold in sorted(df['fold'].unique()):
                fold_data = df[df['fold'] == fold]
                if len(fold_data) > 0:
                    score = scorer.score(fold_data[config.target_col].values, 
                                       fold_data['pred'].values)
                    fold_scores.append(score)
            
            if fold_scores:
                oof_metrics[name] = {
                    'overall_oof': np.mean(fold_scores),
                    'mean_fold': np.mean(fold_scores),
                    'std_fold': np.std(fold_scores)
                }
        else:
            # Overall metrics
            score = scorer.score(df[config.target_col].values, df['pred'].values)
            oof_metrics[name] = {'overall_oof': score, 'mean_fold': score}
    
    # Find best single model
    best_single_model = None
    best_single_score = -np.inf
    for name, metrics in oof_metrics.items():
        score = metrics.get('overall_oof', -np.inf)
        if score > best_single_score:
            best_single_score = score
            best_single_model = name
    
    # Apply decorrelation if enabled
    decorrelation_info = None
    if config.decorrelate and len(oof_data) > 1:
        try:
            filtered_oof_data, filtered_metrics, decorrelation_info = filter_redundant_models(
                oof_data, oof_metrics, config.target_col, config.correlation_threshold
            )
            oof_data = filtered_oof_data
            oof_metrics = filtered_metrics
        except Exception as e:
            warnings.warn(f"Decorrelation failed: {e}")
    
    # Prepare model
    model = BlendModel(
        method=config.method,
        config=config,
        oof_metrics=oof_metrics,
        best_single_model=best_single_model,
        decorrelation_info=decorrelation_info
    )
    
    # Fit specific method
    if config.method == 'weighted' and len(oof_data) > 1:
        try:
            # Create dummy submission data for weight optimization
            sub_data = {name: df[['id', 'pred']].copy() for name, df in oof_data.items()}
            scorer = Scorer(metric=config.metric)
            predictions, weight_info = optimize_weights(
                oof_data, sub_data, scorer, config.target_col,
                config.search_params.get('restarts', 16),
                4, config.random_state
            )
            weights = weight_info['weights']
            model.weights = weights
        except Exception as e:
            warnings.warn(f"Weight optimization failed: {e}")
            # Fall back to mean blending
            model.method = 'mean'
    
    elif config.method == 'stacking' and config.stacking and len(oof_data) > 1:
        try:
            # Create dummy submission data for stacking
            sub_data = {name: df[['id', 'pred']].copy() for name, df in oof_data.items()}
            _, stacking_info = stacking_blend(
                oof_data, sub_data, config.stacking, config.random_state
            )
            model.stacking_info = stacking_info
        except Exception as e:
            warnings.warn(f"Stacking failed: {e}")
            # Fall back to mean blending
            model.method = 'mean'
    
    return model


def predict_blend(model: BlendModel, sub_frames: List[pd.DataFrame]) -> BlendResult:
    """Generate predictions using a trained blending model.
    
    Args:
        model: Trained BlendModel
        sub_frames: List of submission DataFrames with columns [id, pred]
        
    Returns:
        BlendResult with predictions and metadata
        
    Raises:
        ValueError: If input data is invalid
    """
    if not sub_frames:
        raise ValueError("At least one submission frame is required")
    
    # Validate submission frames
    for i, df in enumerate(sub_frames):
        if 'id' not in df.columns or 'pred' not in df.columns:
            raise ValueError(f"Submission frame {i} missing required columns 'id' or 'pred'")
    
    # Create model names (should match training)
    model_names = [f"model_{i}" for i in range(len(sub_frames))]
    sub_data = {name: df for name, df in zip(model_names, sub_frames)}
    
    # Generate predictions based on method
    warnings_list = []
    
    if model.method == 'mean':
        predictions = mean_blend(sub_data)
    elif model.method == 'rank_mean':
        predictions = rank_mean_blend(sub_data)
    elif model.method == 'logit_mean':
        predictions = logit_mean_blend(sub_data)
    elif model.method == 'best_single':
        if model.best_single_model and model.best_single_model in sub_data:
            predictions = sub_data[model.best_single_model][['id', 'pred']].copy()
            predictions.columns = ['id', 'pred']
        else:
            # Fall back to first model
            predictions = sub_data[model_names[0]][['id', 'pred']].copy()
            predictions.columns = ['id', 'pred']
            warnings_list.append("Best single model not found, using first model")
    elif model.method == 'weighted' and model.weights:
        # Weighted blending
        pred_arrays = []
        for name in model_names:
            if name in sub_data and name in model.weights:
                pred_arrays.append(sub_data[name]['pred'].values)
        
        if pred_arrays:
            pred_matrix = np.column_stack(pred_arrays)
            weights = np.array([model.weights.get(name, 0) for name in model_names if name in model.weights])
            weights = weights / weights.sum()  # Normalize
            weighted_pred = np.dot(pred_matrix, weights)
            predictions = pd.DataFrame({
                'id': sub_data[model_names[0]]['id'].values,
                'pred': weighted_pred
            })
        else:
            # Fall back to mean
            predictions = mean_blend(sub_data)
            warnings_list.append("Weighted blending failed, using mean blending")
    elif model.method == 'stacking' and model.stacking_info:
        # Stacking prediction
        try:
            predictions, _ = stacking_blend(
                {name: sub_data[name] for name in model_names if name in sub_data},
                sub_data, model.stacking_info.get('meta_learner', 'lr'), 
                model.config.random_state
            )
        except Exception as e:
            predictions = mean_blend(sub_data)
            warnings_list.append(f"Stacking prediction failed: {e}, using mean blending")
    else:
        # Fall back to mean blending
        predictions = mean_blend(sub_data)
        warnings_list.append(f"Method {model.method} not supported, using mean blending")
    
    # Calculate improvement over best single model
    improvement = None
    if model.best_single_model and model.best_single_model in model.oof_metrics:
        best_single_score = model.oof_metrics[model.best_single_model].get('overall_oof', 0)
        # This would require re-evaluating the blended predictions
        # For now, we'll set it to None and let the caller calculate if needed
        improvement = None
    
    return BlendResult(
        predictions=predictions,
        model=model,
        improvement_over_best_single=improvement,
        warnings=warnings_list
    )


def search_weights(oof_frames: List[pd.DataFrame],
                   target_col: str = "target",
                   metric: str = "auc",
                   iters: int = 200,
                   restarts: int = 16,
                   random_state: Optional[int] = None) -> Tuple[Dict[str, float], float, Dict[str, Any]]:
    """Search for optimal ensemble weights.
    
    Args:
        oof_frames: List of OOF DataFrames
        target_col: Target column name
        metric: Evaluation metric
        iters: Number of optimization iterations
        restarts: Number of random restarts
        random_state: Random seed
        
    Returns:
        Tuple of (weights, best_score, optimization_info)
    """
    if len(oof_frames) < 2:
        raise ValueError("At least 2 OOF frames required for weight optimization")
    
    # Create model names
    model_names = [f"model_{i}" for i in range(len(oof_frames))]
    oof_data = {name: df for name, df in zip(model_names, oof_frames)}
    
    # Create dummy submission data
    sub_data = {name: df[['id', 'pred']].copy() for name, df in oof_data.items()}
    
    # Initialize scorer
    scorer = Scorer(metric=metric)
    
    # Optimize weights
    predictions, info = optimize_weights(
        oof_data, sub_data, scorer, target_col, restarts, 4, random_state
    )
    
    # Extract weights and score from info
    weights = info['weights']
    best_score = info['best_score']
    
    return weights, best_score, info


# Convenience functions for common use cases
def quick_blend(oof_frames: List[pd.DataFrame], 
                sub_frames: List[pd.DataFrame],
                method: str = "mean",
                **kwargs) -> BlendResult:
    """Quick blending with minimal configuration.
    
    Args:
        oof_frames: List of OOF DataFrames
        sub_frames: List of submission DataFrames
        method: Blending method
        **kwargs: Additional configuration
        
    Returns:
        BlendResult with predictions
    """
    model = fit_blend(oof_frames, method=method, **kwargs)
    return predict_blend(model, sub_frames)


def load_model(file_path: Union[str, Path]) -> BlendModel:
    """Load a saved model from file.
    
    Args:
        file_path: Path to saved model file
        
    Returns:
        Loaded BlendModel
    """
    import json
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Convert back to proper types
    if 'config' in data and isinstance(data['config'], dict):
        data['config'] = BlendConfig(**data['config'])
    
    return BlendModel(**data)


def save_model(model: BlendModel, file_path: Union[str, Path]) -> None:
    """Save a model to file.
    
    Args:
        model: BlendModel to save
        file_path: Path to save model
    """
    import json
    
    # Convert to dict for JSON serialization
    model_dict = model.model_dump()
    
    with open(file_path, 'w') as f:
        json.dump(model_dict, f, indent=2, default=str)
