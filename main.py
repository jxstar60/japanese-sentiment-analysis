import pandas as pd
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# 读取数据
df = pd.read_csv("data/data.csv", encoding="utf-8")

# 分词
tokenizer = Tokenizer()
def tokenize(text):
    return " ".join([token.surface for token in tokenizer.tokenize(text)])

df["text"] = df["text"].apply(tokenize)

# 特征化
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# 切分训练/测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 模型1：朴素贝叶斯
model1 = MultinomialNB()
model1.fit(X_train, y_train)
pred1 = model1.predict(X_test)

# 模型2：逻辑回归
model2 = LogisticRegression(max_iter=200)
model2.fit(X_train, y_train)
pred2 = model2.predict(X_test)

# 输出结果
print("=== Naive Bayes ===")
print("Accuracy:", accuracy_score(y_test, pred1))

print("\n=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, pred2))

print("\n=== Classification Report ===")
print(classification_report(y_test, pred2, zero_division=0))
