import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

def analyze_data_imbalance(csv_file):
    """Analyze the class imbalance in the data"""
    print("="*50)
    print("DATA IMBALANCE ANALYSIS")
    print("="*50)
    
    df = pd.read_csv(csv_file)
    total = len(df)
    positive = df['label'].sum()
    negative = (df['label'] == 0).sum()
    
    print(f"Total samples: {total:,}")
    print(f"Positive (m6A): {positive:,} ({positive/total*100:.2f}%)")
    print(f"Negative (no m6A): {negative:,} ({negative/total*100:.2f}%)")
    print(f"Imbalance ratio: {negative/positive:.1f}:1")
    
    # Calculate what a random classifier would achieve
    baseline_precision = positive / total
    print(f"Baseline precision (random): {baseline_precision:.4f}")
    print(f"Baseline PR-AUC (random): ~{baseline_precision:.4f}")
    
    return df, positive/total

def analyze_feature_quality(df):
    """Analyze feature distributions and separability"""
    print("\n" + "="*50)
    print("FEATURE QUALITY ANALYSIS")
    print("="*50)
    
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    
    print(f"Number of features: {len(feature_cols)}")
    
    # Check feature separability
    pos_data = df[df['label'] == 1][feature_cols]
    neg_data = df[df['label'] == 0][feature_cols]
    
    if len(pos_data) > 0 and len(neg_data) > 0:
        print("\nFeature separability (mean difference):")
        for col in feature_cols:
            pos_mean = pos_data[col].mean()
            neg_mean = neg_data[col].mean()
            diff = abs(pos_mean - neg_mean)
            pos_std = pos_data[col].std()
            neg_std = neg_data[col].std()
            pooled_std = np.sqrt((pos_std**2 + neg_std**2) / 2)
            effect_size = diff / pooled_std if pooled_std > 0 else 0
            
            print(f"  {col}: diff={diff:.4f}, effect_size={effect_size:.3f}")
    
    return feature_cols

def baseline_comparison(df, feature_cols):
    """Compare against baseline classifiers"""
    print("\n" + "="*50)
    print("BASELINE COMPARISON")
    print("="*50)
    
    X = df[feature_cols].values
    y = df['label'].values
    
    # Dummy classifiers
    classifiers = {
        'Most Frequent': DummyClassifier(strategy='most_frequent'),
        'Stratified': DummyClassifier(strategy='stratified'),
        'Uniform': DummyClassifier(strategy='uniform')
    }
    
    for name, clf in classifiers.items():
        clf.fit(X, y)
        y_pred = clf.predict_proba(X)[:, 1]
        
        if len(np.unique(y)) == 2:
            roc_auc = roc_auc_score(y, y_pred)
            pr_auc = average_precision_score(y, y_pred)
            print(f"{name:12} - ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")

def suggest_improvements(imbalance_ratio, n_features, pr_auc):
    """Suggest specific improvements"""
    print("\n" + "="*50)
    print("IMPROVEMENT SUGGESTIONS")
    print("="*50)
    
    if imbalance_ratio > 0.95:  # Very imbalanced
        print("🔴 SEVERE IMBALANCE DETECTED")
        
    if n_features < 15:
        print("🟡 LIMITED FEATURES")
    
    if pr_auc < 0.1:
        print("🔴 VERY LOW PR-AUC")
        
    print("\nREALISTIC EXPECTATIONS:")
    print(f"With {imbalance_ratio*100:.1f}% imbalance:")
    print(f"- Random PR-AUC: ~{imbalance_ratio:.4f}")
    print(f"- Good PR-AUC: >{imbalance_ratio*2:.4f}")
    print(f"- Excellent PR-AUC: >{imbalance_ratio*5:.4f}")

def main():
    try:
        # Analyze your real data
        df, pos_rate = analyze_data_imbalance('50_genes.csv')
        feature_cols = analyze_feature_quality(df)
        baseline_comparison(df, feature_cols)
        suggest_improvements(1 - pos_rate, len(feature_cols), 0.007)
        
    except FileNotFoundError:
        print("50_genes.csv not found. Using synthetic data analysis...")
        df, pos_rate = analyze_data_imbalance('test_50_genes.csv')
        feature_cols = analyze_feature_quality(df)
        baseline_comparison(df, feature_cols)
        suggest_improvements(1 - pos_rate, len(feature_cols), 0.007)

if __name__ == "__main__":
    main()