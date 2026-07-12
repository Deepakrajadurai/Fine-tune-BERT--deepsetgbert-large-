# Whitespace Artifact Fix Report

## Data/Final/training_pair_v5.csv

- Total rows: 435178

### Before normalization

  - label=0 (human), n=217589: newline=0 (0.00%), tab=0 (0.00%), double_space=0 (0.00%)

  - label=1 (AI), n=217589: newline=143723 (66.05%), tab=0 (0.00%), double_space=0 (0.00%)

### After normalization (should all be 0)

  - label=0: newline=0, tab=0, double_space=0

  - label=1: newline=0, tab=0, double_space=0
