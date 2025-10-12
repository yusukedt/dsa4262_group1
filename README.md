# dsa4262_group1
DSA4262 Project Repository

## Task 1 Overview
This task focuses on the identification of RNA m6A modifications from direct RNA-Seq data.  
We implement different machine learning classifiers to predict modification sites based on features extracted from nanopore signals.

## My Contribution
- Implemented **XGBoost model** (`train_xgb.py`) as an alternative to logistic regression, SVM, and RF.  
- Used `5_genes.csv` dataset with gene-based split for train/val/test.  
- Handled class imbalance using `scale_pos_weight`.  

## train_xgb.ipynb

The `train_xgb.ipynb` notebook is used to train an **XGBoost classifier** for m6A site prediction.  
This serves as an alternative to logistic regression, SVM, and random forest models.

- **Dataset loading and summary**
  
  Input: `5_genes.csv`  
  Gene-based split into train/validation/test sets.  
  m6A modification rate: ~2.9% (severe class imbalance).

- **Class imbalance handling**
  
  Used `scale_pos_weight = (negative / positive)` in XGBoost to account for rare positives.

- **Model training**
  
  XGBoost parameters:  
  - `n_estimators=300`  
  - `max_depth=5`  
  - `learning_rate=0.05`  
  - `subsample=0.8`  
  - `colsample_bytree=0.8`  
  - Early stopping on validation set.

- **Evaluation metrics**
  
  - Test ROC AUC: **0.795**  
  - Test PR AUC: **0.138**  
  These results show the model can rank positives well (ROC AUC close to 0.8),  
  but PR AUC is lower due to class imbalance.

- **Outputs**
  
  Predictions saved to `xgb_predictions.csv` in required format


## Fine-Tuned XGBoost Model

The fine-tuning stage builds upon the baseline model (`train_xgb.ipynb`) by improving class balance handling and optimising hyperparameters through grid search.  
This work is implemented in the notebook **`finetuning (1).ipynb`**, and the trained outputs are saved as:

- `xgb_best_model (1).pkl` – fine-tuned XGBoost model  
- `xgb_best_params.json` – optimal hyperparameters  
- `xgb_eval_results.json` – evaluation metrics and confusion matrix  

---

### **Dataset loading and summary**

**Dataset:** `50_genes.csv`  
**Shape:** (128,461, 14)  
**Columns:** `['gene_id', 'transcript_id', 'transcript_position', 'sequence', 'label', 'feature_1', ..., 'feature_9']`  
**m6A rate:** 0.0291 (~3%)  
**Unique genes:** 50  
**Unique transcripts:** 62  

The dataset contains 128,461 transcript positions with numerical features and binary labels.  
It is **highly imbalanced**, with approximately **96 % negatives** and **4 % positives**.

---

### **Data splitting**

| Split | Samples | m6A Ratio |
|:--|:--:|:--:|
| Train | 76,984 | 0.037 |
| Validation | 25,692 | 0.030 |
| Test | 25,693 | 0.031 |

Splitting was performed using stratified sampling (60 / 20 / 20) to maintain class proportions.

---

### **Class balancing**

- Original distribution: `label = 0` ≈ 74 k | `label = 1` ≈ 3 k  
- Applied **random undersampling** of majority class (no SMOTE).  
- Final training ratio ≈ **3 : 1** (negative : positive).  
- Validation and test sets remain naturally imbalanced to reflect real data.

---

### **Model fine-tuning**

Fine-tuning was conducted with **GridSearchCV** on the following parameters:

| Parameter | Search Range |
|:--|:--|
| `n_estimators` | [100, 200, 300] |
| `max_depth` | [3, 5, 7] |
| `learning_rate` | [0.01, 0.1, 0.3] |
| `subsample` | [0.7, 1.0] |
| `colsample_bytree` | [0.7, 1.0] |
| `scale_pos_weight` | [3] |

**Best parameters:**
```json
{
  "colsample_bytree": 0.7,
  "learning_rate": 0.1,
  "max_depth": 7,
  "n_estimators": 100,
  "scale_pos_weight": 3,
  "subsample": 1.0
}
```

### **Evaluation results**

**Validation set**

- Accuracy = 0.556  
- ROC-AUC = 0.751  
- PR-AUC = 0.110  

**Test set**

| Metric | Value |
|:--|:--:|
| Accuracy | 0.744 |
| **ROC-AUC** | **0.832** |
| PR-AUC | 0.231 |
| **Weighted F1** | **0.823** |
| Precision (1) | 0.105 |
| Recall (1) | 0.752 |
| F1 (1) | 0.185 |

**Confusion Matrix**


---

### **Interpretation**

- **ROC-AUC = 0.832 → good separability** between m6A and non-m6A sites.  
- **High recall (0.75)** highlights the model successfully detects most true m6A sites.  
- **Low precision (0.11)** is expected but still low because positives form only ~4 % of the dataset.  
- **Weighted F1 = 0.823** shows strong overall balance between precision and recall.  
- Stable performance across validation and test confirms good generalisation.

---

### **Outputs**

| File | Description |
|:--|:--|
| `xgb_best_model (1).pkl` | Fine-tuned XGBoost model |
| `xgb_best_params.json` | Optimal hyperparameters |
| `xgb_eval_results.json` | Evaluation metrics |

All outputs allow direct reuse without retraining.

---



