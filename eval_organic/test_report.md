# Evaluation Report: test

## 5a. Aggregate metrics

```
              precision    recall  f1-score   support

       human       1.00      1.00      1.00      1674
          ai       1.00      1.00      1.00      1674

    accuracy                           1.00      3348
   macro avg       1.00      1.00      1.00      3348
weighted avg       1.00      1.00      1.00      3348

```

Confusion matrix (rows=true, cols=pred, order=[human, ai]):
```
[[1666    8]
 [   5 1669]]
```

## 5b. Per-domain Macro F1

- **Casual**: macro_f1=0.9955 (n=569)
- **News**: macro_f1=0.9960 (n=2779)

## 5c. Per-generator recall (AI class only, via `meta` column)

- **unknown**: recall=0.9970 (n=1674)

Spread across generators: 0.0000 (narrow -- check for uniform bimodal pattern (5d))

## 5d. Confidence distribution (P(AI))

```
  [0.00-0.05): ################################################# (1665)
  [0.05-0.10):  (1)
  [0.10-0.15):  (1)
  [0.15-0.20):  (2)
  [0.20-0.25):  (0)
  [0.25-0.30):  (0)
  [0.30-0.35):  (0)
  [0.35-0.40):  (1)
  [0.40-0.45):  (0)
  [0.45-0.50):  (1)
  [0.50-0.55):  (0)
  [0.55-0.60):  (0)
  [0.60-0.65):  (1)
  [0.65-0.70):  (0)
  [0.70-0.75):  (1)
  [0.75-0.80):  (0)
  [0.80-0.85):  (1)
  [0.85-0.90):  (2)
  [0.90-0.95):  (1)
  [0.95-1.00): ################################################## (1671)
```

- Fraction with extreme confidence (<0.05 or >0.95): 99.64%
- Fraction confidently WRONG (extreme confidence but incorrect): 0.24%

## 5e. Accuracy by length bucket and label

- 0-500, label=0 (human): acc=0.9948 (n=1532)
- 0-500, label=1 (ai): acc=0.9969 (n=1596)
- 500-650, label=0 (human): acc=1.0000 (n=116)
- 500-650, label=1 (ai): acc=1.0000 (n=78)
- 650-800, label=0 (human): acc=1.0000 (n=22)
- 800-950, label=0 (human): acc=1.0000 (n=3)
- 950-1100, label=0 (human): acc=1.0000 (n=1)

## 5f. Per human-source recall (human class only, via `meta` column)

- **unknown**: recall=0.9952 (n=1674)