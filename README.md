# 🌾 Crop Recommendation System

An intelligent Machine Learning-based system designed to assist farmers and agricultural enthusiasts in choosing the most suitable crop for their land. By analyzing soil composition and environmental conditions, the system provides high-accuracy recommendations to maximize yield and sustainability.

---

## 🚀 Features

- **Precision Recommendations**: Predicts the best crop based on 7 critical environmental and soil parameters.
- **Top-3 Suggestions**: Provides the primary recommendation along with the next two most likely alternatives.
- **Confidence Scoring**: Displays a probability percentage for each recommendation.
- **Interactive UI**: A clean, modern, and responsive web interface built with Streamlit.
- **Instant Results**: Real-time processing of user inputs.

---

## 🧠 How It Works

The system utilizes a **Random Forest Classifier** trained on historical agricultural data. It considers the following inputs:

| Parameter          | Description                         |
| :----------------- | :---------------------------------- |
| **Nitrogen (N)**   | Ratio of Nitrogen content in soil   |
| **Phosphorus (P)** | Ratio of Phosphorus content in soil |
| **Potassium (K)**  | Ratio of Potassium content in soil  |
| **Temperature**    | Temperature in degrees Celsius (°C) |
| **Humidity**       | Relative humidity in %              |
| **pH Value**       | PH value of the soil (3.5 - 9.0)    |
| **Rainfall**       | Rainfall in mm                      |

---

## 🛠️ Tech Stack

- **Language**: Python
- **Framework**: Streamlit (Web App UI)
- **ML Library**: Scikit-Learn (Model training & processing)
- **Data Handling**: Pandas & NumPy
- **Serialization**: Joblib (Model & Scaler loading)

---

## 📁 Project Structure

```text
├── assets/                     # UI screenshots 
├── app.py                      # Main Streamlit application script
├── Crop_Recommendation.ipynb    # Research and Model training notebook
├── Crop_recommendation.csv      # Raw dataset for crop prediction
├── crop_recommendation_model.pkl # Saved Random Forest Classifier model
├── label_encoder.pkl            # Serialized label encoder for classes
├── scaler.pkl                  # Serialized feature scaler
├── requirements.txt            # Required Python libraries
└── README.md                   # Project documentation
```

---

## 📊 Model Performance

- **Algorithm**: Random Forest Classifier
- **Preprocessing**: StandardScaler for feature scaling.
- **Accuracy**: Optimized for high precision in diverse environmental conditions.

---

## 📸 Application Preview

### User Interface & Prediction

![Prediction Output](assets/ss1.png)

### Probability Analysis

![Prediction Output](assets/ss2.png)

---

## 🔮 Future Enhancements

- 📡 **IoT Integration**: Real-time soil monitoring using sensor networks.
- ☁️ **Weather API**: Automated climate data fetching based on GPS location.
- 🌐 **Localization**: Multi-language support to empower regional farmers.
- 🧪 **Fertilizer Guidance**: Smart nutrient recommendations based on crop choice.
