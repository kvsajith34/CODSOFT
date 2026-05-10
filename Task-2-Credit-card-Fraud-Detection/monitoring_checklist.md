# Fraud Detection Monitoring Checklist

## Daily
- [ ] Check fraud_score distribution on yesterday's txns (should match training)
- [ ] Count flagged txns vs expected (false positive rate < 1% ideally)
- [ ] Spot check 10-20 flagged cases for obvious errors

## Weekly
- [ ] Recompute AUC-PR on recent labeled data (if available) or use proxy metrics
- [ ] Compare to baseline from model_card.md
- [ ] Check for data drift: mean(log_amount), hour_of_day distribution
- [ ] Review top false positives - any pattern? (e.g. certain merchants)

## Monthly
- [ ] Retrain on latest 3 months data if performance dropped >5% on AUC-PR
- [ ] Update cost model if chargeback fees changed
- [ ] A/B test new threshold or new features (e.g. more bins)
- [ ] Document changes in model_card.md

## Alerts
- Fraud rate jumps >2x normal -> investigate (possible new attack)
- Model file missing or load error -> fallback to rule-based
- High latency in scoring -> optimize or cache

## Real-world tips
- Use predict.py logic in production API (add input validation!)
- Log every score + decision for audit
- Have "human in loop" for high-value txns
- Test adversarial: what if someone crafts txns to look legit?

This is basic but covers the main things for a payments risk system.
