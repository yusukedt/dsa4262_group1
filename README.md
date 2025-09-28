# dsa4262_group1
DSA4262 Project Repository

## Task 1 Overview
This task focuses on the identification of RNA m6A modifications from direct RNA-Seq data.  
We implement different machine learning classifiers to predict modification sites based on features extracted from nanopore signals.

## My Contribution
- Implemented **XGBoost model** (`train_xgb.py`) as an alternative to logistic regression, SVM, and RF.  
- Used `5_genes.csv` dataset with gene-based split for train/val/test.  
- Handled class imbalance using `scale_pos_weight`.  

## train_xgb.py

The `train_xgb.py` script is used to train an **XGBoost classifier** for m6A site prediction.  
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
  
  Predictions saved to `xgb_predictions.csv` in required format:  
