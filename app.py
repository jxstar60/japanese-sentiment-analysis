import streamlit as st
import joblib
from janome.tokenizer import Tokenizer

st.title("日文情感分析系统")

user_input = st.text_input("请输入一句日语：")

# 加载模型
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

tokenizer = Tokenizer()
def tokenize(text):
    return " ".join([token.surface for token in tokenizer.tokenize(text)])

if user_input:
    text = tokenize(user_input)
    vec = vectorizer.transform([text])
    pred = model.predict(vec)

    if pred[0] == 1:
        st.success("正面 😊")
    else:
        st.error("负面 😡")