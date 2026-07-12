# Evaluation Report: test

## 5a. Aggregate metrics

```
              precision    recall  f1-score   support

       human       0.50      0.98      0.66      6798
          ai       0.52      0.02      0.05      6798

    accuracy                           0.50     13596
   macro avg       0.51      0.50      0.35     13596
weighted avg       0.51      0.50      0.35     13596

```

Confusion matrix (rows=true, cols=pred, order=[human, ai]):
```
[[6642  156]
 [6631  167]]
```

## 5b. Per-domain Macro F1

- **europarl_de**: macro_f1=0.3544 (n=13596)

## 5c. Per-generator recall (AI class only, via `meta` column)

- **unknown**: recall=0.0246 (n=6798)

Spread across generators: 0.0000 (narrow -- check for uniform bimodal pattern (5d))

## 5d. Confidence distribution (P(AI))

```
  [0.00-0.05): ################################################## (13107)
  [0.05-0.10):  (55)
  [0.10-0.15):  (33)
  [0.15-0.20):  (16)
  [0.20-0.25):  (16)
  [0.25-0.30):  (14)
  [0.30-0.35):  (9)
  [0.35-0.40):  (9)
  [0.40-0.45):  (9)
  [0.45-0.50):  (5)
  [0.50-0.55):  (6)
  [0.55-0.60):  (13)
  [0.60-0.65):  (7)
  [0.65-0.70):  (9)
  [0.70-0.75):  (9)
  [0.75-0.80):  (8)
  [0.80-0.85):  (9)
  [0.85-0.90):  (14)
  [0.90-0.95):  (25)
  [0.95-1.00):  (223)
```

- Fraction with extreme confidence (<0.05 or >0.95): 98.04%
- Fraction confidently WRONG (extreme confidence but incorrect): 48.85%

## 5e. Accuracy by length bucket and label

- 0-500, label=0 (human): acc=0.9770 (n=6776)
- 0-500, label=1 (ai): acc=0.0246 (n=6791)
- 500-650, label=0 (human): acc=1.0000 (n=15)
- 500-650, label=1 (ai): acc=0.0000 (n=3)
- 800-950, label=0 (human): acc=1.0000 (n=2)
- 800-950, label=1 (ai): acc=0.0000 (n=2)
- 950-1100, label=0 (human): acc=1.0000 (n=1)
- 1750-10000000, label=0 (human): acc=1.0000 (n=4)
- 1750-10000000, label=1 (ai): acc=0.0000 (n=2)

## 5f. Per human-source recall (human class only, via `meta` column)

- **unknown**: recall=0.9771 (n=6798)