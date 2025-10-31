# m6A Site Prediction using Logistic Regression

This project develops a machine learning pipeline to **predict m6A RNA modification sites** from Nanopore direct RNA-Seq data.  
It uses engineered features from per-position signal data, applies data balancing, and evaluates model performance using classification and probabilistic metrics.

---

## Overview

**Goal:**  
Identify m6A modification sites from RNA sequencing features derived from Nanopore direct RNA-Seq data.

**Approach:**  
1. Load and aggregate per-read signal features into per-site summaries.  
2. Engineer biologically relevant and interaction-based features.  
3. Split data by gene to avoid data leakage.  
4. Handle class imbalance using **SMOTEENN** (combines oversampling + noise reduction).  
5. Train a **Logistic Regression** model with feature scaling and class balancing.  
6. Evaluate using ROC-AUC, PR-AUC, F1-based thresholding, and confusion matrices.  
7. Output per-position m6A prediction scores.

---

## Requirements

The script automatically installs dependencies if missing, including:

- `pandas`
- `numpy`
- `scikit-learn`
- `imbalanced-learn`
- `matplotlib`

No manual installation needed, but you can install them directly via:

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib
```

---

## Input Data

**Expected input file:**  
`Full_genes.csv`

This CSV should include:

| Column | Description |
|---------|--------------|
| `transcript_id` | Transcript identifier |
| `transcript_position` | Position within transcript |
| `gene_id` | Gene identifier |
| `label` | Binary label (1 = m6A modified, 0 = unmodified) |
| `feature_*` | Numeric feature columns (e.g. signal mean, std, dwell time) |

The script groups per-read data into per-position averages:

```python
df = df.groupby(['transcript_id', 'transcript_position'], as_index=False).agg({
    **{col: 'mean' for col in feature_cols}, 
    'label': 'mean',
    'gene_id': 'first'
})
```

---

## Feature Engineering

The `engineer_features()` function expands raw signal features by adding:

| Category | Example Features |
|-----------|------------------|
| **Averages** | Mean of dwell, std, and signal mean across flanking positions |
| **Ratios** | Center-to-flanking signal ratios |
| **Ranges** | Max–min signal differences |
| **Interactions** | Products of key center features |
| **Polynomials** | Squared center signal terms |

This enhances signal interpretability and model performance.

---

## Model Training

**Steps:**
1. Split data using **GroupShuffleSplit** to ensure genes do not overlap between training/testing.
2. Apply **SMOTEENN** to balance modified/unmodified site ratio.
3. Scale features with **RobustScaler** (less sensitive to outliers).
4. Train a **Logistic Regression** classifier with:
   - L2 regularization
   - `class_weight='balanced'`
   - `solver='liblinear'`

---

## Evaluation

Metrics reported:
- **Accuracy**
- **Precision, Recall, F1-score**
- **ROC-AUC**
- **PR-AUC**
- **Confusion Matrix**

Additionally, the script plots **Precision–Recall vs Threshold** to identify an optimal decision threshold by F1 score.

```python
best_threshold = thresholds[np.argmax(2 * (precisions * recalls) / (precisions + recalls + 1e-8))]
```

---

## Output

### 1. **Predicted m6A Sites**
The notebook saves a CSV:

```bash
predicted_m6a_sites.csv
```

```
Best threshold (by F1): 0.83
Accuracy: 0.93
ROC-AUC: 0.81
PR-AUC: 0.23
```

Columns:

| Column | Description |
|---------|-------------|
| `transcript_id` | Transcript identifier |
| `transcript_position` | Position in transcript |
| `score` | Predicted probability (0–1) of m6A modification |

### 2. **Post-analysis**
The script demonstrates how to:
- Average scores by transcript position
- Identify **high-confidence sites (score > 0.6)**
- Visualise them with a scatter plot

---