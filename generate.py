import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def generate_realistic_features(n_samples, m6a_labels, sequence_contexts, random_state=42):
    """
    Generate more realistic features based on known m6A signal characteristics
    
    Args:
        n_samples: Number of samples
        m6a_labels: Binary labels for m6A modification
        sequence_contexts: Sequence contexts for each sample
        random_state: Random seed
    """
    np.random.seed(random_state)
    
    features = np.zeros((n_samples, 9))
    
    for i in range(n_samples):
        is_m6a = m6a_labels[i]
        sequence = sequence_contexts[i]
        
        # Base feature values - different distributions for each position and feature type
        for pos in range(3):  # 3 positions: -1, center, +1
            base_idx = pos * 3
            
            # Dwelling time features (indices 0, 3, 6)
            dwelling_idx = base_idx + 0
            if is_m6a and pos == 1:  # Center position shows most change for m6A
                # m6A sites tend to have longer dwelling times at center
                features[i, dwelling_idx] = np.random.lognormal(mean=np.log(0.008), sigma=0.6)
            else:
                features[i, dwelling_idx] = np.random.lognormal(mean=np.log(0.005), sigma=0.5)
            
            # Standard deviation features (indices 1, 4, 7)
            std_idx = base_idx + 1
            if is_m6a and pos == 1:
                # m6A sites tend to have higher signal variability
                features[i, std_idx] = np.random.gamma(shape=3, scale=3)
            else:
                features[i, std_idx] = np.random.gamma(shape=2, scale=2)
            
            # Mean signal features (indices 2, 5, 8)
            mean_idx = base_idx + 2
            base_mean = 100 if pos == 1 else 95  # Center position typically higher
            
            if is_m6a and pos == 1:
                # m6A modifications affect the center position signal
                features[i, mean_idx] = np.random.normal(loc=base_mean + 15, scale=20)
            else:
                features[i, mean_idx] = np.random.normal(loc=base_mean, scale=15)
        
        # Add sequence-specific effects
        if 'AAACC' in sequence or 'AGACC' in sequence:
            # Strong DRACH motifs show different signal patterns
            features[i, 5] += np.random.normal(0, 5)  # Center mean signal variation
        
        # Add some realistic correlations between features
        if is_m6a:
            # m6A sites show correlated changes across positions
            center_effect = np.random.normal(0, 2)
            features[i, [3, 4, 5]] += center_effect * np.array([0.5, 0.3, 0.8])
    
    return features

def generate_test_data(n_samples=1000, n_genes=20, m6a_rate=0.05, 
                      realistic_features=True, random_state=42):
    """
    Generate synthetic test data that mimics the structure of processed m6Anet data
    
    Args:
        n_samples: Number of samples to generate
        n_genes: Number of unique genes
        m6a_rate: Fraction of positions with m6A modifications
        realistic_features: Whether to use realistic feature generation
        random_state: Random seed for reproducibility
    """
    np.random.seed(random_state)
    
    # Generate gene and transcript IDs with more realistic patterns
    gene_ids = [f"ENSG{i:011d}" for i in range(100000 + random_state, 100000 + random_state + n_genes)]
    transcript_ids = [f"ENST{i:011d}" for i in range(200000 + random_state, 200000 + random_state + n_genes)]
    
    # Create base data
    data = []
    used_positions = set()  # Track used transcript-position combinations
    
    for i in range(n_samples):
        # Select gene and transcript
        gene_idx = np.random.randint(0, n_genes)
        gene_id = gene_ids[gene_idx]
        transcript_id = transcript_ids[gene_idx]
        
        # Generate unique position for this transcript
        max_attempts = 100
        for attempt in range(max_attempts):
            position = np.random.randint(100, 5000)  # Realistic transcript positions
            pos_key = (transcript_id, position)
            if pos_key not in used_positions:
                used_positions.add(pos_key)
                break
        else:
            # If we can't find a unique position, just use a random one
            position = np.random.randint(100, 10000)
        
        # Generate 7-mer sequence with realistic DRACH motifs
        drach_motifs = [
            'AAACC', 'AGACC', 'GAACC', 'TAACC',  # High-confidence DRACH
            'AGACT', 'GGACT', 'AAACT', 'GAACT',  # Medium-confidence DRACH
            'AATCC', 'AGTCC', 'ATTCC'            # Lower-confidence motifs
        ]
        
        # Weight motifs by their likelihood to have m6A
        motif_weights = [3, 3, 2, 2, 2, 2, 1.5, 1.5, 1, 1, 1]
        motif_weights = np.array(motif_weights) / sum(motif_weights)
        
        motif = np.random.choice(drach_motifs, p=motif_weights)
        
        # Create full 7-mer sequence
        flanking_bases = ['A', 'T', 'G', 'C']
        sequence = (np.random.choice(flanking_bases) + 
                   motif + 
                   np.random.choice(flanking_bases))
        
        # Create row
        row = {
            'gene_id': gene_id,
            'transcript_id': transcript_id,
            'transcript_position': position,
            'sequence': sequence,
        }
        data.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Remove any remaining duplicates
    df = df.drop_duplicates(subset=['transcript_id', 'transcript_position']).reset_index(drop=True)
    n_samples = len(df)
    
    # Generate labels with realistic patterns
    labels = []
    for _, row in df.iterrows():
        # Base probability
        prob = m6a_rate
        
        # Sequence context effects
        motif_in_sequence = None
        for motif in ['AAACC', 'AGACC', 'GAACC', 'TAACC']:
            if motif in row['sequence']:
                motif_in_sequence = motif
                break
        
        if motif_in_sequence:
            if motif_in_sequence in ['AAACC', 'AGACC']:
                prob *= 4  # Strong DRACH motifs
            else:
                prob *= 2  # Moderate DRACH motifs
        
        # Gene-specific effects
        gene_hash = hash(row['gene_id']) % 100
        if gene_hash < 20:  # 20% of genes are "high m6A" genes
            prob *= 2
        elif gene_hash > 80:  # 20% of genes are "low m6A" genes
            prob *= 0.5
        
        # Position effects
        if 500 <= row['transcript_position'] <= 2000:
            prob *= 1.5
        
        # Ensure probability does not exceed 1
        prob = min(prob, 0.8)  # Cap at 80% to maintain realism
        
        # Generate label
        label = 1 if np.random.random() < prob else 0
        labels.append(label)
    
    df['label'] = labels
    
    # Generate features
    if realistic_features:
        feature_matrix = generate_realistic_features(n_samples, labels, df['sequence'].values, random_state)
    else:
        # Simple feature generation
        feature_matrix = np.random.randn(n_samples, 9)
    
    # Add features to dataframe
    for i in range(9):
        df[f'feature_{i+1}'] = feature_matrix[:, i]
    
    # Ensure all features are positive where appropriate
    dwelling_features = ['feature_1', 'feature_4', 'feature_7']
    std_features = ['feature_2', 'feature_5', 'feature_8']
    
    for feat in dwelling_features + std_features:
        df[feat] = np.abs(df[feat])  # Ensure positive values
    
    return df

def create_multiple_datasets(base_name="test_genes", n_datasets=3, random_state=42):
    """
    Create multiple test datasets with different characteristics
    """
    datasets = {}
    
    # Dataset 1: Small, balanced
    print("Generating balanced dataset...")
    balanced_df = generate_test_data(
        n_samples=200, n_genes=5, m6a_rate=0.2, 
        realistic_features=True, random_state=random_state
    )
    datasets[f"{base_name}_balanced.csv"] = balanced_df
    
    # Dataset 2: Larger, realistic imbalance
    print("Generating realistic dataset...")
    realistic_df = generate_test_data(
        n_samples=1000, n_genes=20, m6a_rate=0.05,
        realistic_features=True, random_state=random_state + 1
    )
    datasets[f"{base_name}_realistic.csv"] = realistic_df
    
    # Dataset 3: Very imbalanced, challenging
    print("Generating challenging dataset...")
    challenging_df = generate_test_data(
        n_samples=2000, n_genes=50, m6a_rate=0.02,
        realistic_features=True, random_state=random_state + 2
    )
    datasets[f"{base_name}_challenging.csv"] = challenging_df
    
    return datasets

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic test data for m6A prediction')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--n_genes', type=int, default=20, help='Number of unique genes')
    parser.add_argument('--m6a_rate', type=float, default=0.05, help='Base m6A modification rate')
    parser.add_argument('--output_prefix', default='test_genes', help='Output file prefix')
    parser.add_argument('--realistic_features', action='store_true', default=True, 
                       help='Use realistic feature generation')
    parser.add_argument('--multiple_datasets', action='store_true', 
                       help='Generate multiple datasets with different characteristics')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Ensure test_data folder exists
    test_data_dir = Path("test_data")
    test_data_dir.mkdir(parents=True, exist_ok=True)

    if args.multiple_datasets:
        # Generate multiple datasets
        datasets = create_multiple_datasets(
            base_name=args.output_prefix,
            n_datasets=3,
            random_state=args.random_state
        )
        
        for filename, dataset in datasets.items():
            out_file = test_data_dir / filename
            dataset.to_csv(out_file, index=False)
            print(f"\nDataset saved: {out_file}")
            print(f"  Samples: {len(dataset)}")
            print(f"  Genes: {dataset['gene_id'].nunique()}")
            print(f"  m6A rate: {dataset['label'].mean():.4f}")
            print(f"  Features: {[col for col in dataset.columns if col.startswith('feature_')]}")
            
        # Create prediction-only datasets (without labels)
        for filename, dataset in datasets.items():
            pred_filename = filename.replace('.csv', '_predict.csv')
            pred_dataset = dataset.drop('label', axis=1)
            pred_dataset.to_csv(pred_filename, index=False)
            print(f"Prediction dataset (no labels): {pred_filename}")
            
    else:
        # Generate single dataset
        print("Generating test data...")
        test_df = generate_test_data(
            n_samples=args.n_samples,
            n_genes=args.n_genes,
            m6a_rate=args.m6a_rate,
            realistic_features=args.realistic_features,
            random_state=args.random_state
        )
        
        # Create prediction-only dataset (without labels)
        pred_file = test_data_dir / f"{args.output_prefix}_predict.csv"
        pred_data = test_df.drop('label', axis=1)
        pred_data.to_csv(pred_file, index=False)
        print(f"Prediction data (without labels) saved to {pred_file}")
        
        # Display summary
        print(f"\nDataset Summary:")
        print(f"Total samples: {len(test_df)}")
        print(f"Unique genes: {test_df['gene_id'].nunique()}")
        print(f"Unique transcripts: {test_df['transcript_id'].nunique()}")
        print(f"m6A modification rate: {test_df['label'].mean():.4f}")
        print(f"m6A positive samples: {test_df['label'].sum()}")
        
        # Feature statistics
        feature_cols = [col for col in test_df.columns if col.startswith('feature_')]
        print(f"\nFeature Statistics:")
        for col in feature_cols:
            values = test_df[col]
            print(f"{col}: mean={values.mean():.4f}, std={values.std():.4f}, "
                  f"min={values.min():.4f}, max={values.max():.4f}")
        
        # Sequence analysis
        print(f"\nSequence Analysis:")
        motif_counts = {}
        for seq in test_df['sequence']:
            for motif in ['AAACC', 'AGACC', 'GAACC', 'TAACC', 'AGACT', 'GGACT']:
                if motif in seq:
                    motif_counts[motif] = motif_counts.get(motif, 0) + 1
        
        for motif, count in sorted(motif_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {motif}: {count} sequences ({count/len(test_df)*100:.1f}%)")
        
        # Display sample data
        print("\nSample data (first 5 rows):")
        print(test_df.head())
    
    print(f"\nData generation completed successfully!")

if __name__ == "__main__":
    main()