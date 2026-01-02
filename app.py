import pickle, re, nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- setup ---
st.set_page_config(page_title="SMS Spam Classifier", page_icon="📱")
st.title("📱 SMS Spam Classifier")

# --- load model/vectorizer ---
vec = pickle.load(open("models/tfidf.pkl","rb"))
model = pickle.load(open("models/nb_model.pkl","rb"))

# --- preprocessing ---
stop = set(stopwords.words("english"))
lemm = WordNetLemmatizer()
url_pat = re.compile(r"http\S+|www\.\S+")
num_pat = re.compile(r"\d+")
nonalnum = re.compile(r"[^a-z0-9\s]")

def preprocess(s):
    s = s.lower()
    s = url_pat.sub(" url ", s)
    s = num_pat.sub(" num ", s)
    s = nonalnum.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    toks = nltk.word_tokenize(s)
    toks = [lemm.lemmatize(t) for t in toks if t not in stop]
    return " ".join(toks)

# --- UI ---
sms = st.text_area("Enter your SMS text:")
if st.button("Predict"):
    if sms.strip() == "":
        st.warning("Please enter some text first!")
    else:
        x = vec.transform([preprocess(sms)])
        pred = model.predict(x)[0]
        label = "🚫 **SPAM**" if pred == 1 else "✅ **HAM (Not Spam)**"
        st.markdown(label)
