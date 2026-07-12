# Evaluation Report: test

## 5a. Aggregate metrics

```
              precision    recall  f1-score   support

       human       1.00      1.00      1.00     41491
          ai       1.00      1.00      1.00     41491

    accuracy                           1.00     82982
   macro avg       1.00      1.00      1.00     82982
weighted avg       1.00      1.00      1.00     82982

```

Confusion matrix (rows=true, cols=pred, order=[human, ai]):
```
[[41489     2]
 [    1 41490]]
```

## 5b. Per-domain Macro F1


## 5c. Per-generator recall (AI class only, via `meta` column)

- **unknown**: recall=1.0000 (n=41491)

Spread across generators: 0.0000 (narrow -- check for uniform bimodal pattern (5d))

## 5d. Confidence distribution (P(AI))

```
  [0.00-0.05): ################################################# (41490)
  [0.05-0.10):  (0)
  [0.10-0.15):  (0)
  [0.15-0.20):  (0)
  [0.20-0.25):  (0)
  [0.25-0.30):  (0)
  [0.30-0.35):  (0)
  [0.35-0.40):  (0)
  [0.40-0.45):  (0)
  [0.45-0.50):  (0)
  [0.50-0.55):  (0)
  [0.55-0.60):  (0)
  [0.60-0.65):  (0)
  [0.65-0.70):  (0)
  [0.70-0.75):  (0)
  [0.75-0.80):  (1)
  [0.80-0.85):  (0)
  [0.85-0.90):  (0)
  [0.90-0.95):  (0)
  [0.95-1.00): ################################################## (41491)
```

- Fraction with extreme confidence (<0.05 or >0.95): 100.00%
- Fraction confidently WRONG (extreme confidence but incorrect): 0.00%

## 5e. Accuracy by length bucket and label

- 0-500, label=0 (human): acc=1.0000 (n=41324)
- 0-500, label=1 (ai): acc=1.0000 (n=41491)
- 500-650, label=0 (human): acc=1.0000 (n=105)
- 650-800, label=0 (human): acc=1.0000 (n=25)
- 800-950, label=0 (human): acc=1.0000 (n=17)
- 950-1100, label=0 (human): acc=1.0000 (n=8)
- 1100-1300, label=0 (human): acc=1.0000 (n=3)
- 1300-1500, label=0 (human): acc=1.0000 (n=4)
- 1500-1750, label=0 (human): acc=1.0000 (n=1)
- 1750-10000000, label=0 (human): acc=1.0000 (n=4)

## 5f. Per human-source recall (human class only, via `meta` column)

- **unknown**: recall=1.0000 (n=41491)