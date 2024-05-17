import streamlit as st
import re
import joblib
import numpy as np
import pandas as pd

# Определение регулярных выражений для извлечения компонентов VIN
CHARS = [chr(x) for x in range(ord("A"), ord("Z") + 1)
         if chr(x) not in ("I", "O", "Q")]
NUMS = [str(x) for x in range(1, 10)] + ["0"]
ALLOWED = "".join(CHARS + NUMS)
WMI_RE = f"(?P<wmi>[{ALLOWED}]{{3}})"
RESTRAINT_RE = f"(?P<restraint>[{ALLOWED}])"
MODEL_RE = f"(?P<model>[{ALLOWED}]{{3}})"
ENGINE_RE = f"(?P<engine>[{ALLOWED}])"
CHECK_RE = f"([{ALLOWED}])"
YEAR_RE = f'(?P<year>[{ALLOWED.replace("U", "").replace("Z", "")}])'
PLANT_RE = f"(?P<plant>[{ALLOWED}])"
VIS_RE = f"([{ALLOWED}]{{3}}\\d{{3}})"
VIN_RE = (
    f"{WMI_RE}{RESTRAINT_RE}{MODEL_RE}{ENGINE_RE}{CHECK_RE}{YEAR_RE}{PLANT_RE}{VIS_RE}"
)

# Функция для проверки правильности VIN
def is_valid_vin(vin):
    vin_pattern = re.compile(r"^(?!\b[{ALLOWED}]{14}\d{3}\b)")
    return bool(vin_pattern.match(vin))

# Функция для получения VIN от пользователя
def get_vin():
    vin = st.text_input("Введите VIN автомобиля: ")
    if not vin:
        st.error("Пожалуйста, введите VIN.")
        return None
    if not is_valid_vin(vin):
        st.error("Неверный формат VIN. Пожалуйста, введите корректный VIN.")
        return None
    return vin

# Функция для извлечения компонентов VIN
def extract_vin_components(vin):
    matches = re.finditer(VIN_RE, vin)
    data = pd.DataFrame([match.groupdict() for match in matches])
    return data

# Загрузка OneHotEncoder и модели
loaded_ohe_encoder = joblib.load("ohe_encoder.joblib")
loaded_model = joblib.load("model.lgbm")

# Основной алгоритм
def predict_vehicle_price(vin):
    data = extract_vin_components(vin)
    cat_features = data.select_dtypes(include="object").columns.tolist()
    
    # Проверка на наличие ошибок перед преобразованием данных
    if data.empty:
        st.error("Неверный формат VIN. Пожалуйста, введите корректный VIN.")
        return
    
    # Исправленная проверка на наличие неизвестных категорий перед применением transform
    unknown_categories = set()
    for i, col in enumerate(cat_features):
        unique_values = set(data[col].unique())
        if unique_values - set(loaded_ohe_encoder.categories_[i]):
            unknown_categories.update(unique_values - set(loaded_ohe_encoder.categories_[i]))
    
    if unknown_categories:
        st.error("Неверный формат VIN. Пожалуйста, введите корректный VIN.")
        return
    
    try:
        data_temp = loaded_ohe_encoder.transform(data[cat_features])
    except ValueError as e:
        if "The number of features in X is different to the number of features of the fitted data" in str(e):
            st.error("Неверный формат VIN. Пожалуйста, введите корректный VIN.")
            return
        else:
            raise e  # Выводим другую ошибку, если это не ошибка, связанная с форматом VIN
    
    feature_names = loaded_ohe_encoder.get_feature_names(input_features=cat_features)
    data[feature_names] = data_temp
    data.drop(cat_features, axis=1, inplace=True)
    predictions = loaded_model.predict(data)
    rounded_prediction = np.round(predictions[0])
    st.write(f"Приблизительная стоимость автомобиля: {rounded_prediction}")


# Функция для запуска приложения
def main():
    st.title("Приложение для прогнозирования стоимости автомобилей Ford (произведенных в Северной Америке) по VIN")
    vin = get_vin()
    if vin is None:
        return
    if st.button("Предсказать стоимость"):
        predict_vehicle_price(vin)

if __name__ == "__main__":
    main()