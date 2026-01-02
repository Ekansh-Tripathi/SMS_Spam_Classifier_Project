import os, re, pickle, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- 1) Load dataset (adjust path if needed) ---
# Try common files: SMSSpamCollection or spam.csv inside archive/
candidates = [
    "archive/SMSSpamCollection",
    "archive/spam.csv",
    "SMSSpamCollection",
    "spam.csv"
]
path = None
for c in candidates:
    if os.path.exists(c):
        path = c; break
if path is None:
    raise FileNotFoundError("Dataset not found. Keep it inside 'archive/' as SMSSpamCollection or spam.csv")

def read_data(p):
    if os.path.basename(p).lower()=="smsspamcollection":
        df = pd.read_csv(p, sep="\t", names=["label","text"])
    else:
        df = pd.read_csv(p, encoding_errors="ignore")
        # try to map common column names
        possible = [("v1","v2"), ("label","text"), ("Category","Message")]
        for a,b in possible:
            if a in df.columns and b in df.columns:
                df = df[[a,b]].rename(columns={a:"label", b:"text"})
                break
    return df

df = read_data(path).dropna()
df["label"] = df["label"].str.lower().map({"ham":0,"spam":1})
df = df[df["label"].isin([0,1])].reset_index(drop=True)

# --- 2) Clean + light preprocess (fast & simple) ---
stop = set(stopwords.words("english"))
lemm = WordNetLemmatizer()
url_pat = re.compile(r"http\S+|www\.\S+")
num_pat = re.compile(r"\d+")
nonalnum = re.compile(r"[^a-z0-9\s]")

def clean(s):
    s = s.lower()
    s = url_pat.sub(" url ", s)
    s = num_pat.sub(" num ", s)
    s = nonalnum.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def preprocess(s):
    s = clean(s)
    toks = nltk.word_tokenize(s)
    toks = [lemm.lemmatize(t) for t in toks if t not in stop]
    return " ".join(toks)

df["proc"] = df["text"].astype(str).apply(preprocess)

# --- 3) Split ---
X_train, X_test, y_train, y_test = train_test_split(
    df["proc"], df["label"].values, test_size=0.2, random_state=42, stratify=df["label"]
)

# --- 4) TF-IDF + Naive Bayes ---
vectorizer = TfidfVectorizer(max_features=5000)
Xtr = vectorizer.fit_transform(X_train)
Xte = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(Xtr, y_train)
pred = model.predict(Xte)

acc = accuracy_score(y_test, pred)
p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", pos_label=1)

print(f"Accuracy: {acc:.4f}")
print(f"Spam  -> Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
print("\nClassification report:\n", classification_report(y_test, pred, digits=4))

# --- 5) Save artifacts ---
os.makedirs("models", exist_ok=True)
pickle.dump(vectorizer, open("models/tfidf.pkl","wb"))
pickle.dump(model, open("models/nb_model.pkl","wb"))

print("\nSaved: models/tfidf.pkl & models/nb_model.pkl")
