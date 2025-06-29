import streamlit as st
import requests
import joblib
import google.generativeai as genai

# =============================
# Gemini API Setup (secure)
# =============================
genai.configure(api_key=st.secrets["gemini"]["api_key"])
model = genai.GenerativeModel("models/chat-bison-001")

# =============================
# Load Disease Prediction Model
# =============================
model_path = "disease_predictor_model.pkl"
try:
    predictor = joblib.load(model_path)
except FileNotFoundError:
    st.error("❌ Model file not found! Upload 'disease_model.pkl' to your repo.")
    st.stop()

# =============================
# OpenWeather API Key
# =============================
OPENWEATHER_API_KEY = st.secrets["openweather"]["api_key"]

# =============================
# UI Design
# =============================
st.set_page_config(page_title="🌡️ ClimaHealth AI", layout="centered")
st.title("🌍 ClimaHealth – Climate-Based Disease Predictor")
st.markdown("Enter a city and get health insights based on weather conditions.")

# =============================
# Get Weather Data
# =============================
def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        temperature = round(data["main"]["temp"] - 273.15, 2)  # Kelvin to Celsius
        humidity = data["main"]["humidity"]
        rainfall = data.get("rain", {}).get("1h", 0.0)
        return temperature, humidity, rainfall
    else:
        return None

# =============================
# Predict Disease
# =============================
def predict_disease(temp, humidity, rain):
    features = [[temp, humidity, rain]]
    return predictor.predict(features)

# =============================
# Gemini Explanation
# =============================
def explain_disease(disease):
    if disease == "none":
        return "✅ No climate-sensitive disease risk detected based on current weather."
    
    prompt = (
        f"Explain the disease '{disease}' caused by climate conditions. "
        "Include:\n- How weather affects it\n- Common symptoms\n"
        "- Prevention tips\n- Remedies\nExplain simply for the general public."
    )
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Gemini error: {e}"

# =============================
# Form & Output
# =============================
with st.form("weather_form"):
    city = st.text_input("🌆 Enter City Name", value="Chennai")
    submit = st.form_submit_button("Predict")

if submit:
    st.markdown("⏳ Fetching weather data...")
    weather = get_weather(city)

    if weather:
        temp, humidity, rain = weather
        st.success(f"📍 Weather in {city}: {temp}°C | 💧 Humidity: {humidity}% | ☔ Rainfall: {rain} mm")
        
        disease = predict_disease(temp, humidity, rain)[0]
        st.subheader(f"🦠 Predicted Disease: `{disease}`")
        
        with st.spinner("🤖 Asking Gemini for details..."):
            explanation = explain_disease(disease)
            st.markdown(f"### 🧠 AI Health Insight:\n{explanation}")
    else:
        st.error("⚠️ Could not fetch weather. Check city name or API key.")
