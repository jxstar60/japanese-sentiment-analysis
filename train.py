import pandas as pd
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv("data/data.csv")

tokenizer = Tokenizer()
def tokenize(text):
    return " ".join([token.surface for token in tokenizer.tokenize(text)])

df["text"] = df["text"].apply(tokenize)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

model = LogisticRegression(max_iter=200)
model.fit(X, y)

# 保存模型
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("模型保存完成")