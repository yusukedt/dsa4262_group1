# dsa4262_group1
DSA4262 Project Repository

## Task 1 Overview
This task focuses on the identification of RNA m6A modifications from direct RNA-Seq data.  
We implement different machine learning classifiers to predict modification sites based on features extracted from nanopore signals.

## My Contribution (Yunxiu)
- Implemented **XGBoost model** (`train_xgb.py`) as an alternative to logistic regression, SVM, and RF.  
- Used `5_genes.csv` dataset with gene-based split for train/val/test.  
- Handled class imbalance using `scale_pos_weight`.  
- Results:
  - Test ROC AUC ≈ **0.795**  
  - Test PR AUC ≈ **0.138**  
- Generated predictions in required format:
transcript_id,transcript_position,score
ENST00000601697,1098,0.334869
ENST00000549920,403,0.483256
