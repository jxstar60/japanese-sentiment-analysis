import streamlit as st
import joblib
from janome.tokenizer import Tokenizer

# ===== 页面设置 =====
st.set_page_config(
    page_title="日文情感分析系统",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 日本語感情分析システム")
st.markdown("テキストを入力すると、AIが感情（ポジティブ・ネガティブ）を分析します。")

# ===== 初始化历史记录 =====
if "history" not in st.session_state:
    st.session_state.history = []

# ===== 加载模型 =====
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ===== 日语分词 =====
tokenizer = Tokenizer()
def tokenize(text):
    return " ".join([token.surface for token in tokenizer.tokenize(text)])

# ===== 输入区域 =====
st.subheader("✏️ テキスト入力")
user_input = st.text_area("ここに日本語の文章を入力してください", height=100)

# ===== 示例按钮 =====
if st.button("📌 サンプルを試す"):
    user_input = "この商品はとても良い"

# ===== 分析按钮 =====
if st.button("🔍 分析する"):
    if user_input.strip() == "":
        st.warning("⚠️ テキストを入力してください")
    else:
        text = tokenize(user_input)
        vec = vectorizer.transform([text])

        # ===== 预测 =====
        prob = model.predict_proba(vec)[0]
        pred = model.predict(vec)[0]
        confidence = max(prob)

        # ===== 结果展示 =====
        st.subheader("📊 分析結果")

        if pred == 1:
            st.success(f"ポジティブ 😊（信頼度：{confidence:.2f}）")
        else:
            st.error(f"ネガティブ 😡（信頼度：{confidence:.2f}）")

        # ===== 置信度进度条 =====
        st.progress(int(confidence * 100))

        # ===== 保存历史 =====
        st.session_state.history.append((user_input, pred, confidence))

# ===== 历史记录 =====
st.subheader("🕘 履歴（直近5件）")

if len(st.session_state.history) == 0:
    st.write("まだ履歴がありません")
else:
    for text, p, c in reversed(st.session_state.history[-5:]):
        label = "ポジティブ 😊" if p == 1 else "ネガティブ 😡"
        st.write(f"「{text}」 → {label}（{c:.2f}）")

# ===== 页脚 =====
st.markdown("---")
st.caption("© 2026 Japanese Sentiment Analysis Demo")