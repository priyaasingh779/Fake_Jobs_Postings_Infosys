import re
import string
import random
import joblib
import pandas as pd
import numpy as np

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('punkt_tab')   # 👈 Add this line


from bs4 import BeautifulSoup        # pip install beautifulsoup4
import nltk                          # pip install nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


# Ensure NLTK resources

nltk_packages = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
for pkg in nltk_packages:
    try:
        nltk.data.find(f'tokenizers/{pkg}')
    except LookupError:
        nltk.download(pkg)

STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# Utility: text cleaning

def clean_text(text):
    """Return cleaned, tokenized, lemmatized text string."""
    if not isinstance(text, str):
        return ""
    # 1. Lowercase
    text = text.lower()
    # 2. Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    # 3. Remove URLs
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    # 4. Remove punctuation and digits
    text = re.sub(r'[\d]', ' ', text)
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    # 5. Tokenize and remove short tokens / stopwords, lemmatize
    tokens = nltk.word_tokenize(text)
    cleaned_tokens = []
    for tok in tokens:
        tok = tok.strip()
        if not tok or tok in STOPWORDS or len(tok) <= 2:
            continue
        tok = lemmatizer.lemmatize(tok)
        cleaned_tokens.append(tok)
    return " ".join(cleaned_tokens)


# Part 1 — Data Understanding

print("=== Part 1: Data Understanding ===")
df = pd.read_csv('fake_job_postings.csv')

# Basic info
print("\nDataset shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())

# Missing values per column
print("\nMissing values per column:")
print(df.isna().sum())

# Distribution of fraudulent labels
if 'fraudulent' in df.columns:
    print("\nDistribution of 'fraudulent':")
    print(df['fraudulent'].value_counts(normalize=False).to_frame(name='count'))
    print("\nDistribution (proportion):")
    print(df['fraudulent'].value_counts(normalize=True).to_frame(name='proportion'))
else:
    raise KeyError("Column 'fraudulent' not found in dataset.")

# Three short insights (generic — adapt if needed)
print("\n3 Quick Insights (adapt if numbers differ on your data):")
insights = [
    "1) Class imbalance: real jobs typically outnumber fake jobs (check proportions above).",
    "2) Several columns contain missing values (e.g., company profile, salary, or employment_type) — fake posts often have more missing or placeholder info.",
    "3) Some text fields contain HTML/URLs/irrelevant tokens that must be cleaned before modeling."
]
for i in insights:
    print(i)


# Part 2 — Text Cleaning & Preprocessing

print("\n=== Part 2: Text Cleaning & Preprocessing ===")
# Use 'description' column (common in Kaggle dataset)
if 'description' not in df.columns:
    raise KeyError("Column 'description' not found in dataset. Make sure you have the correct CSV.")

# Example raw description for later comparison
example_raw = df['description'].fillna("").iloc[0]

# Create clean_description
print("\nCleaning text (this may take a moment)...")
df['description'] = df['description'].fillna("")
df['clean_description'] = df['description'].apply(clean_text)

# Compare average word count before and after cleaning
def avg_word_count(series):
    return series.apply(lambda t: len(str(t).split())).mean()

avg_before = avg_word_count(df['description'])
avg_after = avg_word_count(df['clean_description'])

print(f"\nAverage word count BEFORE cleaning: {avg_before:.2f}")
print(f"Average word count AFTER cleaning: {avg_after:.2f}")

# Show one raw vs cleaned example
print("\nExample - Raw description (truncated 300 chars):\n", example_raw[:300])
print("\nExample - Cleaned description:\n", df['clean_description'].iloc[0])


# Part 3 — Feature Extraction (TF-IDF)

print("\n=== Part 3: Feature Extraction (TF-IDF) ===")
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = tfidf.fit_transform(df['clean_description'])
print("\nTF-IDF matrix shape:", X_tfidf.shape)

# 10 sample feature names
feature_names = tfidf.get_feature_names_out()
print("\n10 sample feature names:", feature_names[:10].tolist())

# Top 15 words by global TF-IDF (sum across docs)
tfidf_sums = np.asarray(X_tfidf.sum(axis=0)).ravel()
top15_idx = np.argsort(tfidf_sums)[-15:][::-1]
top15 = [(feature_names[i], float(tfidf_sums[i])) for i in top15_idx]
print("\nTop 15 words by global TF-IDF score (word, summed_score):")
for w, s in top15:
    print(w, f"{s:.4f}")


# Part 4 — Model Building

print("\n=== Part 4: Model Building ===")
y = df['fraudulent'].astype(int)    # ensure int

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000, class_weight='balanced')  # balanced to help class imbalance
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\nEvaluation on test set:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nShort interpretation (2-3 sentences):")
print("The model achieves the metrics shown above. If precision is much higher than recall, the model is conservative (fewer false positives). If recall is higher, it catches more fakes but may include more false positives. Examine confusion matrix to judge trade-offs and consider more advanced models or feature engineering if needed.")


# Part 5 — Model Analysis & Save

print("\n=== Part 5: Model Analysis & Saving ===")
# 5 random job descriptions from whole dataset (not only test); show predicted fake probability via pipeline
random_idx = random.sample(range(len(df)), 5)
sample_df = df.iloc[random_idx].copy()
X_sample_tfidf = tfidf.transform(sample_df['clean_description'])
sample_proba = model.predict_proba(X_sample_tfidf)[:, 1]
sample_pred_label = (sample_proba >= 0.5).astype(int)

sample_df = sample_df[['title', 'description', 'clean_description']].reset_index(drop=True)
sample_df['predicted_proba'] = sample_proba
sample_df['predicted_label'] = sample_pred_label

print("\n5 random job predictions (probability of being fake):\n")
print(sample_df[['title', 'predicted_proba', 'predicted_label']])

# Manually inspect one predicted fake and one predicted real (choose from sample_df)
fake_examples = sample_df[sample_df['predicted_label'] == 1]
real_examples = sample_df[sample_df['predicted_label'] == 0]

if not fake_examples.empty:
    print("\nExample predicted FAKE (inspect description):\n")
    print("Title:", fake_examples.iloc[0]['title'])
    print("Raw description (truncated):", fake_examples.iloc[0]['description'][:400])
    print("Cleaned:", fake_examples.iloc[0]['clean_description'][:400])
else:
    print("\nNo predicted-fake examples found in the random sample. You can rerun to sample again.")

if not real_examples.empty:
    print("\nExample predicted REAL (inspect description):\n")
    print("Title:", real_examples.iloc[0]['title'])
    print("Raw description (truncated):", real_examples.iloc[0]['description'][:400])
    print("Cleaned:", real_examples.iloc[0]['clean_description'][:400])
else:
    print("\nNo predicted-real examples found in the random sample. You can rerun to sample again.")

# Save trained model and vectorizer
joblib.dump(model, 'fake_job_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
print("\nSaved model to 'fake_job_model.pkl' and vectorizer to 'tfidf_vectorizer.pkl'")

print("\n=== Pipeline complete ===")
