# Dataset Directory

This directory should contain the raw training and test datasets.

## Required Files

- `train.csv` - Training dataset
- `test.csv` - Test dataset

## How to Get the Dataset

### Option 1: Amazon Review Polarity Dataset (Recommended for Demo)

1. Download from Kaggle: https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews
2. Extract the files
3. Run the data preparation script:
   ```bash
   python prepare_data.py
   ```

### Option 2: Use Your Own Dataset

Create CSV files with the following format:

**Format:**
```csv
text,label
"Review text here",0
"Another review",1
```

**Labels:**
- `0` = Genuine review
- `1` = Fake review

**Requirements:**
- Must have a `text` column with review content
- Must have a `label` column with 0 (genuine) or 1 (fake)
- Recommended: 10,000+ reviews for good model performance

## Note

The raw data files are **not included** in this repository due to size constraints.
Users must download their own datasets and place them here before training.
