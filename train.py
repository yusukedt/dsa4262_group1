import pandas as pd
import numpy as np
import pickle
import argparse
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
import warnings
warnings.filterwarnings('ignore')

class M6APredictor:
    def __init__(self, balance_strategy='class_weight', scaler_type='standard', random_state=42):
        """
        Initialize the m6A predictor
        
        Args:
            balance_strategy: Strategy to handle imbalanced data
                - 'class_weight': Use class weights in logistic regression
                - 'smote': Use SMOTE oversampling
                - 'undersample': Use random undersampling
                - 'enn': Use Edited Nearest Neighbours undersampling
            scaler_type: Type of feature scaling ('standard' or 'robust')
            random_state: Random seed for reproducibility
        """
        self.balance_strategy = balance_strategy
        self.scaler_type = scaler_type
        self.random_state = random_state
        
        # Choose scaler
        if scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
            
        self.model = None
        self.feature_names = None
        self.original_feature_names = None
        self.training_stats = {}

    def _get_feature_columns(self, df):
        """
        Returns a list of feature columns from the dataframe.
        This includes all columns starting with 'feature_' and 'read_count' if present.
        """
        feature_cols = [c for c in df.columns if c.startswith('feature_')]
        if 'read_count' in df.columns:
            feature_cols.append('read_count')
        return feature_cols

    def _engineer_features(self, X, feature_names, fit=True):
        """
        Engineer additional features to improve separability with better genomics-aware features
        """
        print("Engineering additional features...")
        
        # Original features
        X_engineered = X.copy()
        engineered_names = feature_names.copy()
        
        # Enhanced feature engineering based on RNA modification patterns
        if len(feature_names) >= 9:
            # Group features by position (-1, center, +1)
            # Positions: -1 (features 1,2,3), center (features 4,5,6), +1 (features 7,8,9)
            pos_minus1 = [0, 1, 2]  # feature_1, feature_2, feature_3
            pos_center = [3, 4, 5]  # feature_4, feature_5, feature_6
            pos_plus1 = [6, 7, 8]   # feature_7, feature_8, feature_9
            
            # Position-specific aggregated features
            # Average signals across positions
            avg_dwelling = np.mean(X[:, [0, 3, 6]], axis=1)  # Average dwelling times
            avg_std = np.mean(X[:, [1, 4, 7]], axis=1)       # Average standard deviations
            avg_mean = np.mean(X[:, [2, 5, 8]], axis=1)      # Average mean signals
            
            X_engineered = np.column_stack([X_engineered, avg_dwelling, avg_std, avg_mean])
            engineered_names.extend(['avg_dwelling_time', 'avg_std_signal', 'avg_mean_signal'])
            
            # Position-specific ratios (center vs flanking)
            # Center to flanking ratios
            center_to_flanking_dwelling = X[:, 3] / (X[:, 0] + X[:, 6] + 1e-8)
            center_to_flanking_std = X[:, 4] / (X[:, 1] + X[:, 7] + 1e-8)
            center_to_flanking_mean = X[:, 5] / (X[:, 2] + X[:, 8] + 1e-8)
            
            X_engineered = np.column_stack([X_engineered, center_to_flanking_dwelling, 
                                          center_to_flanking_std, center_to_flanking_mean])
            engineered_names.extend(['center_flanking_dwelling_ratio', 'center_flanking_std_ratio', 'center_flanking_mean_ratio'])
            
            # Signal variability features
            dwelling_range = np.max(X[:, [0, 3, 6]], axis=1) - np.min(X[:, [0, 3, 6]], axis=1)
            std_range = np.max(X[:, [1, 4, 7]], axis=1) - np.min(X[:, [1, 4, 7]], axis=1)
            mean_range = np.max(X[:, [2, 5, 8]], axis=1) - np.min(X[:, [2, 5, 8]], axis=1)
            
            X_engineered = np.column_stack([X_engineered, dwelling_range, std_range, mean_range])
            engineered_names.extend(['dwelling_range', 'std_range', 'mean_range'])
            
            # Cross-feature interactions for most informative features
            # Based on typical m6A signal patterns, dwelling time and signal mean are often most informative
            center_dwelling_x_mean = X[:, 3] * X[:, 5]  # center dwelling * center mean
            center_std_x_mean = X[:, 4] * X[:, 5]       # center std * center mean
            
            X_engineered = np.column_stack([X_engineered, center_dwelling_x_mean, center_std_x_mean])
            engineered_names.extend(['center_dwelling_x_mean', 'center_std_x_mean'])
            
            # Polynomial features for center position (most important for m6A)
            center_dwelling_sq = X[:, 3] ** 2
            center_mean_sq = X[:, 5] ** 2
            center_std_sq = X[:, 4] ** 2
            
            X_engineered = np.column_stack([X_engineered, center_dwelling_sq, center_mean_sq, center_std_sq])
            engineered_names.extend(['center_dwelling_squared', 'center_mean_squared', 'center_std_squared'])
        
        print(f"Original features: {len(feature_names)}")
        print(f"Engineered features: {len(engineered_names)}")
        
        return X_engineered, engineered_names
    
    def _split_by_gene(self, df, test_size=0.2, val_size=0.1):
        """
        Split data by gene_id to avoid data leakage with improved stratification
        
        Returns:
            train_df, val_df, test_df
        """
        # Get unique genes
        unique_genes = df['gene_id'].unique()
        print(f"Total unique genes: {len(unique_genes)}")
        
        # Calculate class distribution per gene to ensure stratification
        # More sophisticated stratification: consider both presence and abundance
        gene_stats = df.groupby('gene_id').agg({
            'label': ['sum', 'count', 'mean']
        }).reset_index()
        gene_stats.columns = ['gene_id', 'positive_count', 'total_count', 'positive_rate']
        
        # Create stratification categories based on positive rate
        def categorize_gene(rate):
            if rate == 0:
                return 0  # No positives
            elif rate <= 0.02:
                return 1  # Very low rate
            elif rate <= 0.1:
                return 2  # Low rate
            else:
                return 3  # High rate
        
        gene_stats['strat_category'] = gene_stats['positive_rate'].apply(categorize_gene)
        
        print("Gene stratification distribution:")
        print(gene_stats['strat_category'].value_counts().sort_index())
        
        # Split genes into train+val and test
        genes_train_val, genes_test = train_test_split(
            gene_stats['gene_id'].values, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=gene_stats['strat_category'].values
        )
        
        # Split train+val into train and validation
        gene_stats_train_val = gene_stats[gene_stats['gene_id'].isin(genes_train_val)]
        genes_train, genes_val = train_test_split(
            gene_stats_train_val['gene_id'].values,
            test_size=val_size/(1-test_size),
            random_state=self.random_state,
            stratify=gene_stats_train_val['strat_category'].values
        )
        
        # Create dataframes
        train_df = df[df['gene_id'].isin(genes_train)].copy()
        val_df = df[df['gene_id'].isin(genes_val)].copy()
        test_df = df[df['gene_id'].isin(genes_test)].copy()
        
        print(f"Train genes: {len(genes_train)}, samples: {len(train_df)}, m6A ratio: {train_df['label'].mean():.4f}")
        print(f"Val genes: {len(genes_val)}, samples: {len(val_df)}, m6A ratio: {val_df['label'].mean():.4f}")
        print(f"Test genes: {len(genes_test)}, samples: {len(test_df)}, m6A ratio: {test_df['label'].mean():.4f}")
        
        return train_df, val_df, test_df
    
    def _handle_imbalanced_data(self, X, y):
        """
        Apply selected strategy to handle imbalanced data with improved options
        """
        print(f"Original class distribution: {np.bincount(y)}")
        print(f"Original imbalance ratio: {np.bincount(y)[0] / np.bincount(y)[1]:.1f}:1")
        
        if self.balance_strategy == 'class_weight':
            return X, y
        
        # Determine appropriate k_neighbors based on minority class size
        minority_count = np.bincount(y)[1]
        k_neighbors = min(5, minority_count - 1) if minority_count > 1 else 1
        
        if self.balance_strategy == 'smote':
            sampler = SMOTE(random_state=self.random_state, k_neighbors=k_neighbors)
        
        elif self.balance_strategy == 'borderline_smote':
            sampler = BorderlineSMOTE(random_state=self.random_state, k_neighbors=k_neighbors)
        
        elif self.balance_strategy == 'adasyn':
            sampler = ADASYN(random_state=self.random_state, n_neighbors=k_neighbors)
        
        elif self.balance_strategy == 'smote_tomek':
            sampler = SMOTETomek(random_state=self.random_state)
        
        elif self.balance_strategy == 'smote_enn':
            sampler = SMOTEENN(random_state=self.random_state)
        
        elif self.balance_strategy == 'undersample':
            sampler = RandomUnderSampler(random_state=self.random_state)
            
        elif self.balance_strategy == 'enn':
            sampler = EditedNearestNeighbours()
        
        else:
            raise ValueError(f"Unknown balance strategy: {self.balance_strategy}")
        
        try:
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            print(f"Resampled class distribution: {np.bincount(y_resampled)}")
            print(f"New imbalance ratio: {np.bincount(y_resampled)[0] / np.bincount(y_resampled)[1]:.1f}:1")
            return X_resampled, y_resampled
        except Exception as e:
            print(f"Warning: Resampling failed ({str(e)}), using original data with class weights")
            return X, y
    
    def _evaluate_model(self, X, y, data_name=""):
        """
        Comprehensive model evaluation with multiple metrics
        """
        probabilities = self.model.predict_proba(X)[:, 1]
        
        # Calculate metrics
        roc_auc = roc_auc_score(y, probabilities)
        precision, recall, _ = precision_recall_curve(y, probabilities)
        pr_auc = auc(recall, precision)
        avg_precision = average_precision_score(y, probabilities)
        
        print(f"{data_name} Metrics:")
        print(f"  AUC-ROC: {roc_auc:.4f}")
        print(f"  AUC-PR: {pr_auc:.4f}")
        print(f"  Average Precision: {avg_precision:.4f}")
        
        return {
            f'{data_name.lower()}_auc_roc': roc_auc,
            f'{data_name.lower()}_auc_pr': pr_auc,
            f'{data_name.lower()}_avg_precision': avg_precision
        }
    
    def train(self, df, validation_df=None, use_cv=True, cv_folds=5):
        """
        Train the m6A prediction model with enhanced evaluation
        
        Args:
            df: Training dataframe
            validation_df: Optional validation dataframe for evaluation
            use_cv: Whether to perform cross-validation
            cv_folds: Number of CV folds
        """
        # Get features and labels
        self.original_feature_names = self._get_feature_columns(df)
        X_original = df[self.original_feature_names].values
        y = df['label'].values
        
        print(f"Training with {len(self.original_feature_names)} original features and {len(X_original)} samples")
        print(f"Original m6A modification rate: {y.mean():.4f}")
        
        # Engineer additional features
        X_engineered, self.feature_names = self._engineer_features(X_original, self.original_feature_names)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_engineered)
        
        # Handle imbalanced data
        X_balanced, y_balanced = self._handle_imbalanced_data(X_scaled, y)
        
        # Set up model with optimized parameters for imbalanced data
        if self.balance_strategy == 'class_weight':
            class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
            class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
            print(f"Using class weights: {class_weight_dict}")
            
            self.model = LogisticRegression(
                random_state=self.random_state,
                max_iter=2000,
                class_weight=class_weight_dict,
                penalty='l2',
                C=1.0,
                solver='liblinear'
            )
        else:
            self.model = LogisticRegression(
                random_state=self.random_state,
                max_iter=2000,
                penalty='l2',
                C=1.0,
                solver='liblinear'
            )
        
        # Perform cross-validation
        if use_cv:
            print(f"\nPerforming {cv_folds}-fold cross-validation...")
            cv_scores_roc = cross_val_score(self.model, X_balanced, y_balanced, 
                                          cv=cv_folds, scoring='roc_auc')
            cv_scores_pr = cross_val_score(self.model, X_balanced, y_balanced, 
                                         cv=cv_folds, scoring='average_precision')
            
            print(f"CV AUC-ROC: {cv_scores_roc.mean():.4f} ± {cv_scores_roc.std():.4f}")
            print(f"CV AUC-PR: {cv_scores_pr.mean():.4f} ± {cv_scores_pr.std():.4f}")
            
            self.training_stats.update({
                'cv_auc_roc_mean': cv_scores_roc.mean(),
                'cv_auc_roc_std': cv_scores_roc.std(),
                'cv_auc_pr_mean': cv_scores_pr.mean(),
                'cv_auc_pr_std': cv_scores_pr.std()
            })
        
        # Train final model
        print("\nTraining final model...")
        self.model.fit(X_balanced, y_balanced)
        
        # Evaluate on training data
        train_metrics = self._evaluate_model(X_scaled, y, "Training")
        self.training_stats.update(train_metrics)
        
        # Evaluate on validation data
        if validation_df is not None:
            X_val_original = validation_df[self.original_feature_names].values
            X_val_engineered, _ = self._engineer_features(X_val_original, self.original_feature_names)
            X_val_scaled = self.scaler.transform(X_val_engineered)
            y_val = validation_df['label'].values
            
            val_metrics = self._evaluate_model(X_val_scaled, y_val, "Validation")
            self.training_stats.update(val_metrics)
        
        # Store additional training information
        self.training_stats.update({
            'balance_strategy': self.balance_strategy,
            'scaler_type': self.scaler_type,
            'n_features_original': len(self.original_feature_names),
            'n_features_engineered': len(self.feature_names),
            'n_samples_original': len(y),
            'n_samples_balanced': len(y_balanced),
            'm6a_rate_original': y.mean(),
            'm6a_rate_balanced': y_balanced.mean()
        })
    
    def save_model(self, filepath):
        """Save the trained model and preprocessing objects"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'original_feature_names': self.original_feature_names,
            'balance_strategy': self.balance_strategy,
            'scaler_type': self.scaler_type,
            'training_stats': self.training_stats,
            'random_state': self.random_state
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")

def main():
    parser = argparse.ArgumentParser(description='Train m6A RNA modification prediction model')
    parser.add_argument('--data', default='50_genes.csv', help='Path to CSV file containing both features and labels')
    parser.add_argument('--output', default='m6a_model.pkl', help='Output model file path')
    parser.add_argument('--balance_strategy', default='class_weight', 
                       choices=['class_weight', 'smote', 'borderline_smote', 'adasyn', 
                               'smote_tomek', 'smote_enn', 'undersample', 'enn'],
                       help='Strategy to handle imbalanced data')
    parser.add_argument('--scaler_type', default='standard', choices=['standard', 'robust'],
                       help='Type of feature scaling')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size (fraction)')
    parser.add_argument('--val_size', type=float, default=0.1, help='Validation set size (fraction)')
    parser.add_argument('--cv_folds', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--no_cv', action='store_true', help='Skip cross-validation')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Ensure model folder exists
    model_dir = Path("model")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Build full output paths under "model/"
    output_path = model_dir / args.output
    stats_path = output_path.with_suffix('').as_posix() + "_stats.json"

    # Load data
    print("Loading data...")
    merged_df = pd.read_csv(args.data)
    
    # Check if required columns exist
    required_columns = ['gene_id', 'transcript_id', 'transcript_position', 'label']
    missing_columns = [col for col in required_columns if col not in merged_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    print(f"Dataset shape: {merged_df.shape}")
    print(f"Columns: {list(merged_df.columns)}")
    print(f"m6A modification rate: {merged_df['label'].mean():.4f}")
    print(f"Number of unique genes: {merged_df['gene_id'].nunique()}")
    print(f"Number of unique transcripts: {merged_df['transcript_id'].nunique()}")
    
    # Initialise predictor
    predictor = M6APredictor(
        balance_strategy=args.balance_strategy,
        scaler_type=args.scaler_type,
        random_state=args.random_state
    )
    
    # Split data by gene
    train_df, val_df, test_df = predictor._split_by_gene(
        merged_df, 
        test_size=args.test_size, 
        val_size=args.val_size
    )
    
    # Train model
    predictor.train(train_df, val_df, use_cv=not args.no_cv, cv_folds=args.cv_folds)
    
    # Evaluate on test set
    if len(test_df) > 0:
        X_test_original = test_df[predictor.original_feature_names].values
        X_test_engineered, _ = predictor._engineer_features(X_test_original, predictor.original_feature_names)
        X_test_scaled = predictor.scaler.transform(X_test_engineered)
        y_test = test_df['label'].values
        
        test_metrics = predictor._evaluate_model(X_test_scaled, y_test, "Test")
        predictor.training_stats.update(test_metrics)
    
    # Save model
    predictor.save_model(output_path)
    
    # Save training statistics
    with open(stats_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        stats_to_save = {}
        for k, v in predictor.training_stats.items():
            if isinstance(v, np.ndarray):
                stats_to_save[k] = v.tolist()
            elif isinstance(v, np.integer):
                stats_to_save[k] = int(v)
            elif isinstance(v, np.floating):
                stats_to_save[k] = float(v)
            else:
                stats_to_save[k] = v
        json.dump(stats_to_save, f, indent=2)
    
    print(f"\nTraining completed. Model saved to {output_path}")
    print(f"Training statistics saved to {stats_path}")
    
    # Print final summary
    print("\n=== TRAINING SUMMARY ===")
    for key, value in predictor.training_stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()