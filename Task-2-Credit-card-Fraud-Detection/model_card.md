# Model Card - Credit Card Fraud Detection (Random Forest)

**Model Details**
- Type: RandomForestClassifier (200 trees, max_depth=12, balanced_subsample)
- Trained on: synthetic data mimicking Kaggle ULB creditcard (or real if provided)
- Features: V1-V28 + log_amount, hour_of_day, amount_bin
- Threshold: tuned to ~0.3-0.4 depending on run (cost-optimized)

**Performance (on holdout, typical run)**
- AUC-ROC: ~0.98+
- AUC-PR: ~0.75+ (key metric for imbalance)
- Recall on fraud: high after threshold tune
- Precision: trade-off via cost model

**Limitations**
- Synthetic data for demo, real data has more noise
- No real-time serving here, but predict.py shows scoring
- Cost model is simplified ($120 FN, $2 FP)

**Intended Use**
- Payments risk scoring / transaction monitoring
- Offline batch scoring or integrate into API

**Monitoring Checklist**
- Track AUC-PR and cost on live traffic weekly
- Monitor feature drift (esp. amount and time patterns)
- Retrain if fraud rate or patterns shift (concept drift common in fraud)
- Log false positive rate to avoid customer friction
- Have human review queue for flagged txns above certain score
