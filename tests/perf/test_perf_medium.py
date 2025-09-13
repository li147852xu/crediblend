"""Performance tests for medium-scale data."""

import pytest
import pandas as pd
import numpy as np
import time
from pathlib import Path
import tempfile
import os

from crediblend.api import fit_blend, predict_blend, quick_blend
from crediblend.core.performance import (get_memory_usage, optimize_dtypes, 
                                       memory_efficient_blend, estimate_memory_usage)


@pytest.mark.slow
class TestPerformanceMedium:
    """Performance tests for medium-scale data (200k rows x 8 models)."""
    
    @pytest.fixture
    def medium_oof_data(self):
        """Create medium-scale OOF data (200k rows x 8 models)."""
        np.random.seed(42)
        n_samples = 200000
        n_models = 8
        
        oof_data = []
        for i in range(n_models):
            # Create realistic predictions with some correlation
            base_pred = np.random.beta(2, 5, n_samples)  # Skewed towards lower values
            noise = np.random.normal(0, 0.1, n_samples)
            pred = np.clip(base_pred + noise, 0, 1)
            
            # Create target with some signal
            target = (base_pred + np.random.normal(0, 0.2, n_samples) > 0.3).astype(int)
            
            # Create folds
            fold = np.random.randint(0, 5, n_samples)
            
            df = pd.DataFrame({
                'id': range(n_samples),
                'pred': pred,
                'target': target,
                'fold': fold
            })
            
            oof_data.append(df)
        
        return oof_data
    
    @pytest.fixture
    def medium_sub_data(self):
        """Create medium-scale submission data."""
        np.random.seed(42)
        n_samples = 200000
        n_models = 8
        
        sub_data = []
        for i in range(n_models):
            pred = np.random.beta(2, 5, n_samples)
            pred = np.clip(pred, 0, 1)
            
            df = pd.DataFrame({
                'id': range(n_samples),
                'pred': pred
            })
            
            sub_data.append(df)
        
        return sub_data
    
    def test_memory_usage_tracking(self, medium_oof_data):
        """Test memory usage tracking."""
        initial_memory = get_memory_usage()
        print(f"Initial memory: {initial_memory:.1f}MB")
        
        # Process data
        optimized_data = [optimize_dtypes(df) for df in medium_oof_data]
        
        final_memory = get_memory_usage()
        print(f"Final memory: {final_memory:.1f}MB")
        print(f"Memory increase: {final_memory - initial_memory:.1f}MB")
        
        # Should not use more than 1GB
        assert final_memory - initial_memory < 1000
    
    def test_dtype_optimization(self, medium_oof_data):
        """Test dtype optimization reduces memory usage."""
        original_memory = sum(df.memory_usage(deep=True).sum() for df in medium_oof_data) / 1024 / 1024
        print(f"Original memory: {original_memory:.1f}MB")
        
        optimized_data = [optimize_dtypes(df) for df in medium_oof_data]
        optimized_memory = sum(df.memory_usage(deep=True).sum() for df in optimized_data) / 1024 / 1024
        print(f"Optimized memory: {optimized_memory:.1f}MB")
        
        # Should reduce memory usage by at least 20%
        reduction = (original_memory - optimized_memory) / original_memory
        assert reduction > 0.2, f"Memory reduction {reduction:.1%} is too small"
    
    def test_memory_efficient_blend(self, medium_oof_data, medium_sub_data):
        """Test memory-efficient blending."""
        start_time = time.time()
        start_memory = get_memory_usage()
        
        # Convert lists to dictionaries
        oof_dict = {f'model_{i}': df for i, df in enumerate(medium_oof_data)}
        sub_dict = {f'model_{i}': df for i, df in enumerate(medium_sub_data)}
        
        result = memory_efficient_blend(oof_dict, sub_dict, 
                                      method='mean', memory_cap_mb=2048)
        
        end_time = time.time()
        end_memory = get_memory_usage()
        
        duration = end_time - start_time
        memory_used = end_memory - start_memory
        
        print(f"Blending duration: {duration:.2f}s")
        print(f"Memory used: {memory_used:.1f}MB")
        
        # Should complete within reasonable time and memory
        assert duration < 30, f"Blending took too long: {duration:.2f}s"
        assert memory_used < 1000, f"Used too much memory: {memory_used:.1f}MB"
        assert len(result) == len(medium_sub_data[0]), "Result length mismatch"
    
    def test_quick_blend_performance(self, medium_oof_data, medium_sub_data):
        """Test quick_blend performance on medium data."""
        start_time = time.time()
        
        result = quick_blend(medium_oof_data, medium_sub_data, method='mean')
        
        duration = time.time() - start_time
        
        print(f"Quick blend duration: {duration:.2f}s")
        
        # Should complete within reasonable time
        assert duration < 60, f"Quick blend took too long: {duration:.2f}s"
        assert len(result.predictions) == len(medium_sub_data[0])
    
    def test_parallel_weight_optimization(self, medium_oof_data, medium_sub_data):
        """Test parallel weight optimization performance."""
        from crediblend.core.performance import parallel_weight_optimization
        from crediblend.core.metrics import Scorer
        
        scorer = Scorer(metric='auc')
        
        start_time = time.time()
        
        result = parallel_weight_optimization(
            {f'model_{i}': df for i, df in enumerate(medium_oof_data)},
            {f'model_{i}': df for i, df in enumerate(medium_sub_data)},
            scorer, 'target', n_restarts=8, n_jobs=2
        )
        
        # Handle different return formats
        if len(result) == 2:
            predictions, info = result
            weights = info.get('weights', {})
            score = info.get('best_score', 0)
        else:
            weights, score, info = result
        
        duration = time.time() - start_time
        
        print(f"Parallel weight optimization duration: {duration:.2f}s")
        print(f"Best score: {score:.4f}")
        print(f"Weights: {weights}")
        
        # Should complete within reasonable time
        assert duration < 120, f"Weight optimization took too long: {duration:.2f}s"
        assert isinstance(weights, dict)
        assert len(weights) == len(medium_oof_data)
    
    def test_estimate_memory_usage(self, medium_oof_data):
        """Test memory usage estimation."""
        data_dict = {f'model_{i}': df for i, df in enumerate(medium_oof_data)}
        
        estimated_memory = estimate_memory_usage(data_dict)
        
        # Should be reasonable estimate
        assert estimated_memory > 0
        assert estimated_memory < 2000  # Should not exceed 2GB
    
    def test_auto_strategy_selection(self, medium_oof_data, medium_sub_data):
        """Test auto strategy selection."""
        from crediblend.core.performance import auto_strategy_selection
        
        oof_dict = {f'model_{i}': df for i, df in enumerate(medium_oof_data)}
        sub_dict = {f'model_{i}': df for i, df in enumerate(medium_sub_data)}
        
        strategy = auto_strategy_selection(oof_dict, sub_dict, memory_cap_mb=2048, n_jobs=2)
        
        print(f"Selected strategy: {strategy}")
        
        # Should select a valid strategy
        assert strategy in ['mean', 'weighted', 'decorrelate_weighted']
    
    def test_performance_guardrails(self, medium_oof_data):
        """Test performance guardrails."""
        from crediblend.core.performance import performance_guardrails
        
        data_dict = {f'model_{i}': df for i, df in enumerate(medium_oof_data)}
        
        # Test with high memory cap
        result_high = performance_guardrails(data_dict, memory_cap_mb=4096, max_models=20)
        assert len(result_high) == len(data_dict)
        
        # Test with low memory cap
        result_low = performance_guardrails(data_dict, memory_cap_mb=100, max_models=5)
        assert len(result_low) <= 5
    
    def test_end_to_end_performance(self, medium_oof_data, medium_sub_data):
        """Test end-to-end performance on medium data."""
        start_time = time.time()
        start_memory = get_memory_usage()
        
        # Fit model
        model = fit_blend(medium_oof_data, method='mean', random_state=42)
        
        # Predict
        result = predict_blend(model, medium_sub_data)
        
        end_time = time.time()
        end_memory = get_memory_usage()
        
        duration = end_time - start_time
        memory_used = end_memory - start_memory
        
        print(f"End-to-end duration: {duration:.2f}s")
        print(f"Memory used: {memory_used:.1f}MB")
        
        # Should complete within reasonable time and memory
        assert duration < 120, f"End-to-end took too long: {duration:.2f}s"
        assert memory_used < 1500, f"Used too much memory: {memory_used:.1f}MB"
        assert len(result.predictions) == len(medium_sub_data[0])


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
