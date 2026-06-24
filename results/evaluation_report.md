# Model Validation & Generalization Evaluation Report

This report summarizes the rigorous validation results of the GBERT-large German AI vs. Human text detector.

## Overall Calibration & Accuracy Diagnostics

| Model Setup | Accuracy | Macro F1 | ROC-AUC | ECE (10 bins) | Brier Score |
|---|---|---|---|---|---|
| Original | 58.70% | 57.38% | 0.6718 | 0.408622 | 0.408989 |
| Rotation 1 | 83.67% | 60.86% | 0.8692 | 0.160050 | 0.157452 |
| Rotation 2 | 94.67% | 90.69% | 0.9932 | 0.039817 | 0.045541 |

## Cross-Domain Transfer Matrix (Accuracy by Source)

| Source Domain | Class | Original Model | Rotation 1 (Gemini Held-Out) | Rotation 2 (Qwen Held-Out) |
|---|---|---|---|---|
| AI ChatGPT | AI | 38.33% | N/A | N/A |
| AI Claude | AI | 30.00% | N/A | N/A |
| AI Gemini | AI | 0.00% | 18.33% | N/A |
| AI Qwen | AI | 100.00% | N/A | 73.33% |
| Human Casual | Human | 100.00% | 100.00% | 100.00% |
| Human Essay | Human | 50.00% | 100.00% | 100.00% |
| Human News | Human | 100.00% | 100.00% | 100.00% |
| Human Wiki | Human | 93.33% | 100.00% | 100.00% |
| Humanized AI | AI | 16.67% | N/A | N/A |

## Performance by Text Length Bins

| Length Bin (Words) | Original (Acc / Count) | Rotation 1 (Acc / Count) | Rotation 2 (Acc / Count) |
|---|---|---|---|
| 20-40 | 51.97% (229) | 71.94% (139) | 97.96% (98) |
| 40-80 | 62.65% (257) | 91.96% (112) | 92.00% (150) |
| 80-120 | 80.00% (35) | 96.67% (30) | 93.94% (33) |
| 120-200 | 47.37% (19) | 100.00% (19) | 100.00% (19) |
| Other | 0.00% (0) | 0.00% (0) | 0.00% (0) |

