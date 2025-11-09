# Dataset Information & Sources

This document provides information about datasets suitable for training the fake review detection model.

## Recommended Datasets

### 1. Amazon Product Reviews
**Source:** [Kaggle - Amazon Reviews](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews)

**Description:**
- Large collection of Amazon product reviews
- Contains both genuine and fake reviews
- Pre-labeled for supervised learning

**Format:**
- CSV file with columns: `text`, `label`
- Label: 0 = Genuine, 1 = Fake

**Size:** ~50,000+ reviews

**Download Instructions:**
```bash
# 1. Go to Kaggle link above
# 2. Click "Download"
# 3. Extract and place in data/raw/reviews.csv
```

---

### 2. Yelp Fake Review Dataset
**Source:** [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences)

**Description:**
- Restaurant and business reviews from Yelp
- Labeled by experts for authenticity
- Good for sentiment + fake detection

**Format:**
- Text files with tab-separated values
- Needs preprocessing to convert to CSV

---

### 3. OpSpam Dataset
**Source:** [Available on request from authors]

**Description:**
- Hotel reviews from TripAdvisor
- Gold standard dataset for fake review research
- Used in academic papers

**Size:** ~1,600 reviews

---

### 4. Deceptive Opinion Spam Corpus
**Source:** [Cornell CS Department](https://www.cs.uic.edu/~liub/FBS/fake-reviews.html)

**Description:**
- Manually created fake and genuine hotel reviews
- High-quality labeled data
- Small but reliable

---

## Dataset Format Requirements

Your CSV file should have the following structure:

```csv
text,label
"Great product! Really loved it.",0
"Amazing! Best purchase ever! WOW!!!",1
"Good quality. Works as expected.",0
```

### Column Specifications:

1. **text** (string):
   - Review text content
   - Can include punctuation, emojis
   - No length limit (will be truncated if > 10,000 chars)

2. **label** (integer):
   - `0` = Genuine/Real review
   - `1` = Fake/Fraudulent review

---

## Creating Your Own Dataset

If you want to create a custom dataset:

### 1. Data Collection
- Collect reviews from e-commerce platforms (ensure compliance with ToS)
- Mix of verified genuine reviews and identified fake reviews
- Aim for balanced dataset (50-50 split)

### 2. Labeling
- Manual labeling by multiple annotators
- Use inter-annotator agreement metrics
- Consider using crowd-sourcing platforms

### 3. Data Quality
- Remove duplicates
- Handle missing values
- Ensure text encoding is UTF-8
- Remove personally identifiable information (PII)

---

## Data Preprocessing

The system automatically handles:
- ✅ Text cleaning (URLs, special characters)
- ✅ Tokenization and lemmatization
- ✅ Stopword removal
- ✅ Feature extraction

You just need to provide raw text!

---

## Sample Dataset

A small sample dataset is included for testing:

```python
import pandas as pd

sample_data = {
    'text': [
        "This product exceeded my expectations. Good quality.",
        "AMAZING! BEST EVER! BUY NOW!!!",
        "Decent product, arrived on time.",
        "I received this for free. It's great! Five stars!",
        "Works fine. Nothing special but does the job."
    ],
    'label': [0, 1, 0, 1, 0]  # 0=genuine, 1=fake
}

df = pd.DataFrame(sample_data)
df.to_csv('data/raw/sample_reviews.csv', index=False)
```

---

## Dataset Statistics

For optimal model performance:

| Metric | Recommended | Minimum |
|--------|-------------|---------|
| Total samples | 10,000+ | 1,000 |
| Fake reviews | 40-60% | 30% |
| Genuine reviews | 40-60% | 40% |
| Avg review length | 50-200 words | 10 words |

---

## Legal & Ethical Considerations

⚠️ **Important:**
- Ensure you have rights to use the dataset
- Comply with data protection regulations (GDPR, CCPA)
- Remove personal information from reviews
- Respect platform Terms of Service
- Cite dataset sources in publications

---

## Citation

If using this system in research, please cite:

```bibtex
@misc{fake-review-detection-2025,
  author = {Your Name},
  title = {Fake Review Detection System Using Ensemble Machine Learning},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/fake-review-detection}
}
```

---

## Additional Resources

- **Papers on Fake Review Detection:**
  - Ott et al. (2011) - "Finding Deceptive Opinion Spam"
  - Mukherjee et al. (2013) - "Spotting Fake Reviewer Groups"
  - Rayana & Akoglu (2015) - "Collective Opinion Spam Detection"

- **Tools:**
  - [Label Studio](https://labelstud.io/) - Data labeling tool
  - [Amazon Mechanical Turk](https://www.mturk.com/) - Crowdsourcing labels

---

**Need Help?**
Open an issue on GitHub: https://github.com/aravindss2004/fake-review-detection/issues
