import streamlit as st
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

# --- Загружаем обученную модель ---
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model("catboost_model.cbm")
    return model

model = load_model()

# --- Заголовок ---
st.title("🚀 Детектор мошеннических транзакций")
st.write("Введите параметры транзакции и нажмите **Предсказать**, чтобы проверить, является ли она мошеннической.")

# --- UI: Ползунки для признаков ---
v10 = st.slider("V10", min_value=-20.0, max_value=20.0, value=0.0)
v12 = st.slider("V12", min_value=-20.0, max_value=20.0, value=0.0)
v14 = st.slider("V14", min_value=-20.0, max_value=20.0, value=0.0)
v17 = st.slider("V17", min_value=-20.0, max_value=20.0, value=0.0)
amount = st.number_input("Сумма транзакции (Amount)", min_value=0.0, max_value=10000.0, value=100.0)

# --- Формируем входные данные ---
features = np.array([[v10, v12, v14, v17, amount]])


# --- Кнопка предсказания ---
if st.button("🔍 Предсказать"):
    prediction = model.predict(features)
    proba = model.predict_proba(features)[0][1]  # Вероятность мошенничества

    if prediction == 1:
        st.error(f"⚠️ Предупреждение! Вероятность мошенничества: {proba:.2%}")
    else:
        st.success(f"✅ Транзакция безопасна. Вероятность мошенничества: {proba:.2%}")



