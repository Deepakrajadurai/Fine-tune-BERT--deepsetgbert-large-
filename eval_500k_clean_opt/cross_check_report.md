# 5g. Test vs. Final Holdout Cross-Check

- test.csv macro_f1: 1.0000
- final_holdout.csv macro_f1: 0.3332
- difference (test - holdout): +0.6667

**WARNING**: difference exceeds 0.03 -- possible subtle overfitting to decisions made while iterating against test/val, even with group-disjoint splits. Report both numbers in the thesis, not just the better one.