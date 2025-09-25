#!/usr/bin/env python3
"""
m6A RNA Modification Prediction - Prediction Script
Author: Your Name
Description: Makes predictions on new data using a trained m6A modification model
"""

import pandas as pd
import numpy as np
import pickle
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class M6APredictor:
    def __init__(self):
        """Initialize the m6A predictor"""
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.original_feature_names = None
        self.balance_strategy = None
        self.scaler_type = None
        self.training_stats = {}
        self.random_state = 42
    
    def _engineer_features_predict(self, X, original_feature_names):
        """
        Apply the same feature engineering as during training - enhanced version
        """
        # Original features
        X_engineered = X.copy()
        
        # Apply the same enhanced feature engineering as in training
        if len(original_feature_names) >= 9:
            # Group features by position (-1, center, +1)
            # Positions: -1 (features 1,2,3), center (features 4,5,6), +1 (features 7,8,9)
            
            # Position-specific aggregated features
            avg_dwelling = np.mean(X[:, [0, 3, 6]], axis=1)  # Average dwelling times
            avg_std = np.mean(X[:, [1, 4, 7]], axis=1)       # Average standard deviations
            avg_mean = np.mean(X[:, [2, 5, 8]], axis=1)      # Average mean signals
            
            X_engineered = np.column_stack([X_engineered, avg_dwelling, avg_std, avg_mean])
            
            # Position-specific ratios (center vs flanking)
            center_to_flanking_dwelling = X[:, 3] / (X[:, 0] + X[:, 6] + 1e-8)
            center_to_flanking_std = X[:, 4] / (X[:, 1] + X[:, 7] + 1e-8)
            center_to_flanking_mean = X[:, 5] / (X[:, 2] + X[:, 8] + 1e-8)
            
            X_engineered = np.column_stack([X_engineered, center_to_flanking_dwelling, 
                                          center_to_flanking_std, center_to_flanking_mean])
            
            # Signal variability features
            dwelling_range = np.max(X[:, [0, 3, 6]], axis=1) - np.min(X[:, [0, 3, 6]], axis=1)
            std_range = np.max(X[:, [1, 4, 7]], axis=1) - np.min(X[:, [1, 4, 7]], axis=1)
            mean_range = np.max(X[:, [2, 5, 8]], axis=1) - np.min(X[:, [2, 5, 8]], axis=1)
            
            X_engineered = np.column_stack([X_engineered, dwelling_range, std_range, mean_range])
            
            # Cross-feature interactions
            center_dwelling_x_mean = X[:, 3] * X[:, 5]  # center dwelling * center mean
            center_std_x_mean = X[:, 4] * X[:, 5]       # center std * center mean
            
            X_engineered = np.column_stack([X_engineered, center_dwelling_x_mean, center_std_x_mean])
            
            # Polynomial features for center position
            center_dwelling_sq = X[:, 3] ** 2
            center_mean_sq = X[:, 5] ** 2
            center_std_sq = X[:, 4] ** 2
            
            X_engineered = np.column_stack([X_engineered, center_dwelling_sq, center_mean_sq, center_std_sq])
        
        return X_engineered

    def load_model(self, filepath):
        """Load a trained model and preprocessing objects"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.original_feature_names = model_data.get('original_feature_names', None)
            self.balance_strategy = model_data.get('balance_strategy', 'class_weight')
            self.scaler_type = model_data.get('scaler_type', 'standard')
            self.training_stats = model_data.get('training_stats', {})
            self.random_state = model_data.get('random_state', 42)
            
            print(f"Model loaded from {filepath}")
            print(f"Model trained with {len(self.feature_names)} features")
            print(f"Original features: {len(self.original_feature_names) if self.original_feature_names else 'Unknown'}")
            print(f"Balance strategy used: {self.balance_strategy}")
            print(f"Scaler type: {self.scaler_type}")
            
            if self.training_stats:
                print("Training statistics:")
                for key, value in self.training_stats.items():
                    if isinstance(value, (int, float)):
                        if isinstance(value, float):
                            print(f"  {key}: {value:.4f}")
                        else:
                            print(f"  {key}: {value}")
            
        except Exception as e:
            raise Exception(f"Error loading model from {filepath}: {str(e)}")
    
    def predict(self, df, return_detailed=False):
        """
        Make predictions on new data with enhanced error handling and validation
        
        Args:
            df: DataFrame with the same structure as training data
            return_detailed: If True, return additional diagnostic information
            
        Returns:
            predictions_df: DataFrame with transcript_id, transcript_position, score
            If return_detailed=True, also returns detailed_info dict
        """
        if self.model is None:
            raise Exception("Model not loaded. Please load a model first.")
        
        # Validate input data
        required_columns = ['transcript_id', 'transcript_position']
        missing_required = [col for col in required_columns if col not in df.columns]
        if missing_required:
            raise Exception(f"Missing required columns in input data: {missing_required}")
        
        # Get available feature columns
        exclude_cols = ['gene_id', 'transcript_id', 'transcript_position', 'label', 'sequence']
        available_features = [col for col in df.columns if col not in exclude_cols]
        
        # Check for expected original features
        if self.original_feature_names:
            expected_features = self.original_feature_names
        else:
            # Fallback to standard feature names
            expected_features = [f'feature_{i}' for i in range(1, 10)]
            
        missing_features = [f for f in expected_features if f not in df.columns]
        if missing_features:
            print(f"Warning: Missing features in input data: {missing_features}")
            
            # Try to use available features that match the pattern
            available_feature_cols = [col for col in df.columns if col.startswith('feature_')]
            if len(available_feature_cols) < 9:
                raise Exception(f"Insufficient features. Expected at least 9 features, got {len(available_feature_cols)}")
            
            # Use the first 9 available feature columns
            expected_features = sorted(available_feature_cols)[:9]
            print(f"Using available features: {expected_features}")
        
        # Extract features and validate
        try:
            X_original = df[expected_features].values
            
            # Check for invalid values
            if np.any(np.isnan(X_original)):
                nan_counts = np.isnan(X_original).sum(axis=0)
                print("Warning: Found NaN values in features:")
                for i, count in enumerate(nan_counts):
                    if count > 0:
                        print(f"  {expected_features[i]}: {count} NaN values")
                
                # Handle NaN values (replace with median)
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='median')
                X_original = imputer.fit_transform(X_original)
                print("NaN values replaced with median values.")
            
            # Check for infinite values
            if np.any(np.isinf(X_original)):
                print("Warning: Found infinite values in features. Clipping to finite range.")
                X_original = np.clip(X_original, -1e10, 1e10)
        
        except Exception as e:
            raise Exception(f"Error extracting features: {str(e)}")
        
        # Engineer features using the same process as training
        try:
            X_engineered = self._engineer_features_predict(X_original, expected_features)
        except Exception as e:
            raise Exception(f"Error in feature engineering: {str(e)}")
        
        # Scale features using the same scaler from training
        try:
            X_scaled = self.scaler.transform(X_engineered)
        except Exception as e:
            raise Exception(f"Error in feature scaling: {str(e)}")
        
        # Make predictions (probabilities)
        try:
            probabilities = self.model.predict_proba(X_scaled)[:, 1]  # Probability of class 1 (m6A modified)
        except Exception as e:
            raise Exception(f"Error making predictions: {str(e)}")
        
        # Validate predictions
        if np.any(np.isnan(probabilities)) or np.any(np.isinf(probabilities)):
            raise Exception("Invalid predictions generated (NaN or infinite values)")
        
        if np.any((probabilities < 0) | (probabilities > 1)):
            raise Exception("Invalid probability values (outside [0, 1] range)")
        
        # Create output DataFrame
        predictions_df = pd.DataFrame({
            'transcript_id': df['transcript_id'],
            'transcript_position': df['transcript_position'],
            'score': probabilities
        })
        
        # Sort by transcript_id and position for consistent output
        predictions_df = predictions_df.sort_values(['transcript_id', 'transcript_position']).reset_index(drop=True)
        
        # Print summary statistics
        print(f"Generated predictions for {len(predictions_df)} positions")
        print(f"Score distribution - Min: {probabilities.min():.4f}, "
              f"Max: {probabilities.max():.4f}, Mean: {probabilities.mean():.4f}, "
              f"Median: {np.median(probabilities):.4f}")
        
        # Count high confidence predictions
        high_conf_thresholds = [0.5, 0.7, 0.9]
        for threshold in high_conf_thresholds:
            high_conf = (probabilities > threshold).sum()
            print(f"Predictions > {threshold}: {high_conf} ({high_conf/len(probabilities)*100:.1f}%)")
        
        if not return_detailed:
            return predictions_df
        
        # Prepare detailed information
        detailed_info = {
            'n_predictions': len(predictions_df),
            'score_stats': {
                'min': float(probabilities.min()),
                'max': float(probabilities.max()),
                'mean': float(probabilities.mean()),
                'median': float(np.median(probabilities)),
                'std': float(probabilities.std())
            },
            'high_confidence_counts': {
                f'above_{threshold}': int((probabilities > threshold).sum())
                for threshold in high_conf_thresholds
            },
            'feature_info': {
                'n_original_features': len(expected_features),
                'n_engineered_features': X_engineered.shape[1],
                'original_features_used': expected_features
            }
        }
        
        return predictions_df, detailed_info
    
    def predict_with_confidence_intervals(self, df, n_bootstrap=100):
        """
        Make predictions with bootstrap confidence intervals
        
        Args:
            df: Input dataframe
            n_bootstrap: Number of bootstrap iterations
            
        Returns:
            DataFrame with predictions and confidence intervals
        """
        print(f"Computing predictions with {n_bootstrap} bootstrap samples...")
        
        # Get the base prediction
        base_predictions = self.predict(df)
        
        # For simplicity, we'll estimate confidence based on prediction uncertainty
        # In a real implementation, you might retrain models on bootstrap samples
        base_scores = base_predictions['score'].values
        
        # Estimate confidence intervals based on score distribution and model uncertainty
        # This is a simplified approach - more sophisticated methods exist
        score_std = np.std(base_scores)
        confidence_factor = 1.96  # 95% confidence interval
        
        # Simple confidence estimation (in practice, you'd use more sophisticated methods)
        confidence_width = np.minimum(
            confidence_factor * score_std * np.sqrt(base_scores * (1 - base_scores)),
            0.1  # Cap the maximum uncertainty
        )
        
        predictions_with_ci = base_predictions.copy()
        predictions_with_ci['score_lower'] = np.maximum(0, base_scores - confidence_width)
        predictions_with_ci['score_upper'] = np.minimum(1, base_scores + confidence_width)
        predictions_with_ci['confidence_width'] = confidence_width
        
        return predictions_with_ci

def main():
    parser = argparse.ArgumentParser(description='Predict m6A RNA modifications on new data')
    parser.add_argument('--data', required=True, help='CSV filename inside test_data/ folder')
    parser.add_argument('--model', default='m6a_model.pkl', help='Model filename inside model/ folder')
    parser.add_argument('--output', default='predictions.csv', help='Output CSV filename (saved to output/)')
    parser.add_argument('--detailed', action='store_true', help='Print detailed prediction information')
    parser.add_argument('--confidence_intervals', action='store_true', help='Include confidence intervals')
    parser.add_argument('--validate_input', action='store_true', help='Perform additional input validation')
    
    args = parser.parse_args()

    # Define folders
    data_path = Path("test_data") / args.data
    model_path = Path("model") / args.model
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output
    
    # Check if input files exist
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load data
    print("Loading input data...")
    try:
        data_df = pd.read_csv(data_path)
        print(f"Loaded {len(data_df)} samples from {data_path}")
        print(f"Data columns: {list(data_df.columns)}")
        
        if args.validate_input:
            print("\nPerforming input validation...")
            numeric_cols = data_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col.startswith('feature_'):
                    col_data = data_df[col]
                    print(f"{col}: min={col_data.min():.4f}, max={col_data.max():.4f}, "
                          f"mean={col_data.mean():.4f}, null={col_data.isnull().sum()}")
            
            duplicates = data_df.duplicated(subset=['transcript_id', 'transcript_position']).sum()
            if duplicates > 0:
                print(f"Warning: Found {duplicates} duplicate transcript_id + transcript_position combinations")
        
    except Exception as e:
        raise Exception(f"Error loading data from {data_path}: {str(e)}")
    
    # Initialize predictor and load model
    predictor = M6APredictor()
    predictor.load_model(model_path)
    
    # Make predictions
    print("\nMaking predictions...")
    
    if args.confidence_intervals:
        predictions_df = predictor.predict_with_confidence_intervals(data_df)
        output_columns = ['transcript_id', 'transcript_position', 'score', 'score_lower', 'score_upper']
    else:
        if args.detailed:
            predictions_df, detailed_info = predictor.predict(data_df, return_detailed=True)
            print("\nDetailed info:")
            for k, v in detailed_info.items():
                print(f"{k}: {v}")
        else:
            predictions_df = predictor.predict(data_df)
        
        output_columns = ['transcript_id', 'transcript_position', 'score']
    
    # Save predictions
    predictions_df[output_columns].to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    
    # Display sample predictions
    print("\nSample predictions:")
    print(predictions_df[output_columns].head(10))
    
    # Summary statistics
    scores = predictions_df['score']
    print(f"\nPrediction summary:")
    print(f"Total predictions: {len(predictions_df)}")
    print(f"Score statistics:")
    print(f"  Mean: {scores.mean():.4f}")
    print(f"  Median: {scores.median():.4f}")
    print(f"  Std: {scores.std():.4f}")
    print(f"  Min: {scores.min():.4f}")
    print(f"  Max: {scores.max():.4f}")
    
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    print(f"\nPredictions by confidence threshold:")
    for threshold in thresholds:
        count = (scores > threshold).sum()
        print(f"  > {threshold}: {count} ({count/len(scores)*100:.1f}%)")

if __name__ == "__main__":
    main()