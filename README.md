# Daily News Category Classifier

A Bidirectional LSTM model that classifies news articles into 13 semantic categories — built from scratch with GloVe embeddings and real-world messy data.

**71.2% test accuracy** across 13 categories on the HuffPost News dataset (~200k articles)

---

## The Problem

News datasets are messy. The original HuffPost dataset had 40+ overlapping, redundant categories — `"Arts & Culture"` vs `"Culture & Arts"`, `"Worldpost"` vs `"World News"` vs `"The WorldPost"`. Before any model could learn anything meaningful, the data needed to be understood and reorganized.

The first real work in this project wasn't training a model. It was making sense of the data.

---

## What I Built

A custom grouping function consolidated 40+ labels into **13 high-level semantic categories**:

| Category | Examples of merged labels |
|---|---|
| Lifestyle & Wellness | WELLNESS, HEALTHY LIVING, STYLE & BEAUTY |
| Parenting & Education | PARENTING, PARENTS, EDUCATION, COLLEGE |
| Sports & Entertainment | SPORTS, ENTERTAINMENT, COMEDY, ARTS |
| Travel, Tourism & Culture | TRAVEL, ARTS & CULTURE, FOOD & DRINK |
| Empowered Voices | WOMEN, QUEER VOICES, BLACK VOICES, LATINO VOICES |
| Science & Tech | TECH, SCIENCE |
| World News | THE WORLDPOST, WORLDPOST, WORLD NEWS |
| Business & Money | BUSINESS, MONEY |
| Environment | ENVIRONMENT, GREEN |
| ... | ... |

---

## Model Architecture

```
Input (tokenized, padded to 100 tokens)
  → Embedding (GloVe 100d, fine-tuned)
  → Bidirectional LSTM (128 units, return_sequences=True)
  → Dropout (0.5) + BatchNormalization
  → Bidirectional LSTM (64 units)
  → Dropout (0.3)
  → Dense (32, ReLU)
  → Dropout (0.3)
  → Dense (13, Softmax)
```

**Training setup:**
- Optimizer: Adam (lr=0.001)
- Loss: Sparse Categorical Crossentropy
- Callbacks: EarlyStopping + ReduceLROnPlateau
- Batch size: 64 | Max epochs: 10

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **71.2%** |
| Test Loss | 0.9776 |
| Best category | Lifestyle & Wellness |
| Hardest category | U.S. News (class imbalance) |

The model handles well-defined, data-rich categories effectively. Categories with sparse samples or heavy contextual overlap with others (U.S. News) remain challenging — a known limitation of LSTM-based approaches without attention mechanisms.

---

## Stack

- Python 3.x
- TensorFlow / Keras
- GloVe 6B 100d embeddings
- scikit-learn
- NumPy, pandas, matplotlib

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/komorabi/news-category-classifier
cd news-category-classifier
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the data**
- Dataset: [HuffPost News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset) → save as `data/news.json`
- GloVe embeddings: [glove.6B.zip](https://nlp.stanford.edu/projects/glove/) → extract `glove.6B.100d.txt` to `data/glove/`

**4. Run the notebook**
```bash
jupyter notebook project.ipynb
```

---

## What I'd Do Differently

71% is a solid baseline for an LSTM on noisy multi-class text data. But the ceiling for this problem is higher. Next steps:

- **Fine-tune BERT or DistilBERT** — attention mechanisms handle overlapping categories much better than recurrence
- **SMOTE or class weights** — address the imbalance in underrepresented categories
- **Data augmentation** — back-translation or synonym replacement for sparse labels

---

## Author

Bahadir · [GitHub](https://github.com/komorabi)

