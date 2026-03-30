# Skin Cancer Detection App

A Streamlit web application that uses a trained CNN model to classify skin lesions and provide medical recommendations.

## Features

- **Image Upload**: Upload skin lesion images for analysis
- **AI Classification**: Classifies into 7 different skin condition categories
- **Severity Assessment**: Provides severity levels (Low, Medium, High, Critical)
- **Medical Recommendations**: Offers specific recommendations based on the predicted condition
- **Confidence Scoring**: Shows prediction confidence levels

## Skin Conditions Detected

1. **Actinic keratoses and intraepithelial carcinomae (akiec)** - High severity
2. **Basal cell carcinoma (bcc)** - High severity  
3. **Benign keratosis-like lesions (bkl)** - Low severity
4. **Dermatofibroma (df)** - Low severity
5. **Melanocytic nevi (nv)** - Low severity
6. **Pyogenic granulomas and hemorrhage (vasc)** - Medium severity
7. **Melanoma (mel)** - Critical severity

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Open the app in your browser (usually http://localhost:8501)
2. Upload an image of a skin lesion using the file uploader
3. View the AI analysis results including:
   - Predicted condition
   - Confidence level
   - Severity assessment
   - Medical recommendations

## Model Information

- **Input**: 28x28 RGB images
- **Architecture**: CNN with multiple convolutional and dense layers
- **Output**: 7-class classification with softmax activation
- **Training Data**: HAM10000 dataset

## Important Disclaimers

⚠️ **This application is for educational and screening purposes only**

- Not a substitute for professional medical diagnosis
- Always consult qualified healthcare professionals
- Do not delay seeking medical care based on these results
- High-risk conditions (melanoma, basal cell carcinoma) require immediate medical attention

## File Structure

```
├── app.py              # Main Streamlit application
├── models/
│   ├── Skin.h5         # Trained Keras model
│   └── Skin.tflite     # TensorFlow Lite model
├── requirements.txt    # Python dependencies
└── README.md          # This file
```