import pandas as pd
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 简单数据（先用小数据，后面再换）
data = {
    "text": [
        "この商品はとても良い",
        "最悪です",
        "まあまあです",
        "素晴らしい体験でした",
        "もう買いません"
    ],
    "label": [1, 0, 1, 1, 0]  # 1=正面 0=负面
}

df = pd.DataFrame(data)

# 分词函数
tokenizer = Tokenizer()
def tokenize(text):
    return " ".join([token.surface for token in tokenizer.tokenize(text)])

df["text"] = df["text"].apply(tokenize)

# 向量化
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])

# 模型训练
model = MultinomialNB()
model.fit(X, df["label"])

# 测试
test_text = "この商品は悪い"
test_text = tokenize(test_text)
test_vec = vectorizer.transform([test_text])

prediction = model.predict(test_vec)
print("预测结果:", "正面" if prediction[0] == 1 else "负面")