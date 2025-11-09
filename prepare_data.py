"""
Data preparation script for Amazon Review Polarity Dataset.
This script loads, samples, and prepares the data for training.

Usage:
    python prepare_data.py [dataset_path]
    
    If dataset_path is not provided, uses default: ./amazon_review_polarity_csv
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

print("=" * 70)
print("Amazon Review Dataset Preparation")
print("=" * 70)

# Paths - Allow user to specify dataset location
if len(sys.argv) > 1:
    DATASET_PATH = sys.argv[1]
else:
    # Default to current directory or common locations
    DATASET_PATH = os.environ.get('DATASET_PATH', './amazon_review_polarity_csv')

print(f"\nDataset location: {DATASET_PATH}")

TRAIN_FILE = os.path.join(DATASET_PATH, "train.csv")
TEST_FILE = os.path.join(DATASET_PATH, "test.csv")

# Check if files exist
if not os.path.exists(TRAIN_FILE):
    print(f"\n✗ Error: Training file not found: {TRAIN_FILE}")
    print("\nPlease:")
    print("  1. Download the Amazon Review Polarity dataset from Kaggle")
    print("  2. Extract it to a folder")
    print("  3. Run: python prepare_data.py /path/to/dataset")
    print("  OR set DATASET_PATH environment variable")
    sys.exit(1)

if not os.path.exists(TEST_FILE):
    print(f"\n✗ Error: Test file not found: {TEST_FILE}")
    sys.exit(1)

OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration
TRAIN_SAMPLES = 50000  # Sample 50k from 1.8M for training
TEST_SAMPLES = 10000   # Sample 10k from 200k for testing

print(f"\nConfiguration:")
print(f"  Training samples: {TRAIN_SAMPLES:,}")
print(f"  Test samples: {TEST_SAMPLES:,}")
print(f"  Output directory: {OUTPUT_DIR}")

# Column names from readme
# Class index (1=negative, 2=positive), Title, Review Text
COLUMN_NAMES = ['label', 'title', 'text']

print("\n" + "-" * 70)
print("Step 1: Loading Training Data (this may take a moment...)")
print("-" * 70)

# Load training data
try:
    # Read with chunksize to handle large file
    chunks = []
    chunksize = 100000
    
    print(f"Reading {TRAIN_FILE} in chunks...")
    for i, chunk in enumerate(pd.read_csv(TRAIN_FILE, 
                                          names=COLUMN_NAMES,
                                          header=None,
                                          chunksize=chunksize)):
        if len(chunks) * chunksize >= TRAIN_SAMPLES:
            break
        chunks.append(chunk)
        print(f"  Loaded chunk {i+1} ({len(chunk):,} rows)")
    
    train_df = pd.concat(chunks, ignore_index=True)
    
    # Sample if we have more than needed
    if len(train_df) > TRAIN_SAMPLES:
        train_df = train_df.sample(n=TRAIN_SAMPLES, random_state=42)
    
    print(f"✓ Loaded {len(train_df):,} training samples")
    
except Exception as e:
    print(f"✗ Error loading training data: {e}")
    exit(1)

print("\n" + "-" * 70)
print("Step 2: Loading Test Data")
print("-" * 70)

try:
    test_df = pd.read_csv(TEST_FILE, 
                         names=COLUMN_NAMES,
                         header=None,
                         nrows=TEST_SAMPLES)
    
    print(f"✓ Loaded {len(test_df):,} test samples")
    
except Exception as e:
    print(f"✗ Error loading test data: {e}")
    exit(1)

print("\n" + "-" * 70)
print("Step 3: Data Preprocessing")
print("-" * 70)

def prepare_dataframe(df, name):
    """Prepare dataframe for our system."""
    print(f"\nProcessing {name} data...")
    
    # Combine title and text for richer content
    df['text'] = df['title'] + ". " + df['text']
    
    # Convert labels: 1 (negative) -> 1 (fake), 2 (positive) -> 0 (genuine)
    # Note: This is a simplification. Negative reviews aren't necessarily fake.
    # But for demonstration, we'll treat negative (class 1) as potentially suspicious
    df['label'] = df['label'].map({1: 1, 2: 0})
    
    # Keep only necessary columns
    df = df[['text', 'label']]
    
    # Remove any missing values
    df = df.dropna()
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['text'])
    
    # Remove very short reviews (less than 10 characters)
    df = df[df['text'].str.len() >= 10]
    
    print(f"  Samples after cleaning: {len(df):,}")
    print(f"  Label distribution:")
    print(f"    Genuine (0): {(df['label'] == 0).sum():,} ({(df['label'] == 0).sum() / len(df) * 100:.1f}%)")
    print(f"    Fake (1): {(df['label'] == 1).sum():,} ({(df['label'] == 1).sum() / len(df) * 100:.1f}%)")
    
    return df

# Process both datasets
train_df = prepare_dataframe(train_df, "training")
test_df = prepare_dataframe(test_df, "test")

print("\n" + "-" * 70)
print("Step 4: Saving Processed Data")
print("-" * 70)

try:
    train_output = os.path.join(OUTPUT_DIR, "train.csv")
    test_output = os.path.join(OUTPUT_DIR, "test.csv")
    
    train_df.to_csv(train_output, index=False)
    print(f"✓ Training data saved: {train_output}")
    print(f"  Size: {os.path.getsize(train_output) / (1024*1024):.2f} MB")
    
    test_df.to_csv(test_output, index=False)
    print(f"✓ Test data saved: {test_output}")
    print(f"  Size: {os.path.getsize(test_output) / (1024*1024):.2f} MB")
    
except Exception as e:
    print(f"✗ Error saving data: {e}")
    exit(1)

print("\n" + "-" * 70)
print("Step 5: Data Summary")
print("-" * 70)

print(f"\nTraining Data Sample:")
print(train_df.head(3).to_string(max_colwidth=60))

print(f"\n\nTest Data Sample:")
print(test_df.head(3).to_string(max_colwidth=60))

print("\n" + "=" * 70)
print("✓ Data Preparation Complete!")
print("=" * 70)
print("\nNext Steps:")
print("  1. Review the prepared data in data/raw/")
print("  2. Run: python train_model.py")
print("  3. Start backend: cd backend && python app.py")
print("  4. Start frontend: cd frontend && npm start")
print("=" * 70)
