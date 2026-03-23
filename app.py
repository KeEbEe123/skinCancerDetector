import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🔬",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    # Try different model files in order of preference
    model_paths = [
        ('models/Skin_retrained.h5', 'Retrained Keras model'),
        ('models/Skin.h5', 'Original Keras model'),
        ('models/Skin_retrained.tflite', 'Retrained TFLite model'),
        ('models/Skin.tflite', 'Original TFLite model')
    ]
    
    for model_path, description in model_paths:
        try:
            logger.info(f"Attempting to load {description} from {model_path}")
            
            if model_path.endswith('.h5'):
                # Keras model
                model = tf.keras.models.load_model(model_path)
                logger.info(f"✓ {description} loaded. Input shape: {model.input_shape}")
                
                # Test model with dummy data to check if it's working
                test_inputs = [
                    np.zeros((1, 28, 28, 3)),
                    np.ones((1, 28, 28, 3)),
                    np.random.random((1, 28, 28, 3))
                ]
                
                predictions = []
                for test_input in test_inputs:
                    pred = model.predict(test_input, verbose=0)
                    predictions.append(np.argmax(pred[0]))
                
                if len(set(predictions)) == 1:
                    logger.warning(f"⚠️ {description} always predicts the same class - trying next model")
                    continue
                
                logger.info(f"✓ {description} appears to be working correctly")
                st.success(f"Using {description}")
                return model
                
            else:
                # TFLite model
                interpreter = tf.lite.Interpreter(model_path=model_path)
                interpreter.allocate_tensors()
                logger.info(f"✓ {description} loaded successfully")
                st.success(f"Using {description}")
                return interpreter
                
        except Exception as e:
            logger.warning(f"Could not load {description}: {e}")
            continue
    
    # If no model works
    st.error("❌ Could not load any working model. Please retrain the model.")
    st.info("Run `python retrain_model.py` to create a new working model.")
    return None

# Define classes and their information
classes = {
    0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),
    1: ('bcc', 'Basal cell carcinoma'),
    2: ('bkl', 'Benign keratosis-like lesions'),
    3: ('df', 'Dermatofibroma'),
    4: ('nv', 'Melanocytic nevi'),
    5: ('vasc', 'Pyogenic granulomas and hemorrhage'),
    6: ('mel', 'Melanoma')
}

# Severity and recommendations
severity_info = {
    'akiec': {
        'severity': 'High',
        'color': 'red',
        'description': 'Pre-cancerous lesions that can develop into squamous cell carcinoma',
        'recommendations': [
            'Seek immediate dermatological consultation',
            'Consider biopsy for definitive diagnosis',
            'Regular monitoring and follow-up',
            'Avoid sun exposure and use broad-spectrum sunscreen',
            'Consider treatment options like cryotherapy or topical medications'
        ]
    },
    'bcc': {
        'severity': 'High',
        'color': 'red',
        'description': 'Most common form of skin cancer, rarely metastasizes but can be locally destructive',
        'recommendations': [
            'Schedule urgent dermatological appointment',
            'Surgical removal is typically required',
            'Regular skin examinations',
            'Sun protection measures',
            'Monitor for new or changing lesions'
        ]
    },
    'mel': {
        'severity': 'Critical',
        'color': 'darkred',
        'description': 'Most dangerous form of skin cancer with high metastatic potential',
        'recommendations': [
            'URGENT: See dermatologist immediately',
            'Biopsy and staging required',
            'May require surgical excision with wide margins',
            'Possible sentinel lymph node biopsy',
            'Regular full-body skin examinations',
            'Strict sun protection'
        ]
    },
    'bkl': {
        'severity': 'Low',
        'color': 'green',
        'description': 'Benign skin lesions that are not cancerous',
        'recommendations': [
            'Routine dermatological check-up recommended',
            'Monitor for any changes in size, color, or texture',
            'General sun protection',
            'No immediate treatment required unless cosmetically bothersome'
        ]
    },
    'df': {
        'severity': 'Low',
        'color': 'green',
        'description': 'Benign fibrous skin tumor, harmless but may be cosmetically concerning',
        'recommendations': [
            'Routine monitoring sufficient',
            'Surgical removal only if cosmetically desired',
            'Regular self-examination',
            'No urgent medical intervention needed'
        ]
    },
    'nv': {
        'severity': 'Low',
        'color': 'green',
        'description': 'Common benign moles, usually harmless',
        'recommendations': [
            'Regular self-monitoring using ABCDE rule',
            'Annual dermatological check-up',
            'Sun protection to prevent changes',
            'Watch for asymmetry, border irregularity, color changes, diameter >6mm, evolution'
        ]
    },
    'vasc': {
        'severity': 'Medium',
        'color': 'orange',
        'description': 'Vascular lesions that may bleed but are typically benign',
        'recommendations': [
            'Dermatological evaluation recommended',
            'May require treatment if bleeding or growing',
            'Monitor for changes',
            'Consider removal if problematic'
        ]
    }
}

def preprocess_image(image):
    """Preprocess image for model prediction"""
    logger.info(f"Original image mode: {image.mode}, size: {image.size}")
    
    # Convert PIL image to numpy array
    img_array = np.array(image)
    logger.info(f"Image array shape after conversion: {img_array.shape}")
    logger.info(f"Image array dtype: {img_array.dtype}")
    logger.info(f"Image array min/max values: {img_array.min()}/{img_array.max()}")
    
    # Resize to 28x28
    img_resized = cv2.resize(img_array, (28, 28))
    logger.info(f"Image shape after resize: {img_resized.shape}")
    
    # Ensure 3 channels (RGB)
    if len(img_resized.shape) == 2:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        logger.info("Converted grayscale to RGB")
    elif img_resized.shape[2] == 4:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGBA2RGB)
        logger.info("Converted RGBA to RGB")
    
    logger.info(f"Final image shape before normalization: {img_resized.shape}")
    logger.info(f"Pixel value range before normalization: {img_resized.min()}-{img_resized.max()}")
    
    # Normalize pixel values to [0, 1] - SAME AS TRAINING
    img_normalized = img_resized.astype(np.float32) / 255.0
    logger.info(f"Pixel value range after normalization: {img_normalized.min()}-{img_normalized.max()}")
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    logger.info(f"Final batch shape: {img_batch.shape}")
    
    # Log some sample pixel values
    logger.info(f"Sample pixels from normalized image: {img_normalized[0, 0, :]} (top-left pixel)")
    logger.info(f"Sample pixels from normalized image: {img_normalized[14, 14, :]} (center pixel)")
    
    # Check if image is all the same value (which might cause issues)
    unique_values = np.unique(img_normalized)
    logger.info(f"Number of unique pixel values: {len(unique_values)}")
    if len(unique_values) < 10:
        logger.warning(f"Image has very few unique values: {unique_values}")
    
    return img_batch

def predict_skin_condition(model, image):
    """Make prediction on preprocessed image"""
    logger.info("Starting prediction process")
    
    processed_image = preprocess_image(image)
    logger.info(f"Processed image shape for prediction: {processed_image.shape}")
    
    # Check if model is TFLite interpreter or Keras model
    if hasattr(model, 'predict'):
        # Keras model
        logger.info("Using Keras model for prediction")
        predictions = model.predict(processed_image, verbose=0)
        predictions = predictions[0]
    else:
        # TFLite interpreter
        logger.info("Using TFLite model for prediction")
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        
        model.set_tensor(input_details[0]['index'], processed_image)
        model.invoke()
        predictions = model.get_tensor(output_details[0]['index'])[0]
    
    logger.info(f"Raw predictions: {predictions}")
    logger.info(f"Predictions sum: {np.sum(predictions)}")
    
    # Get predicted class and confidence
    predicted_class = np.argmax(predictions)
    confidence = float(predictions[predicted_class])
    
    logger.info(f"Predicted class index: {predicted_class}")
    logger.info(f"Confidence: {confidence}")
    
    # Log all class probabilities
    for i, prob in enumerate(predictions):
        class_code, class_name = classes[i]
        logger.info(f"Class {i} ({class_code}): {prob:.6f}")
    
    return predicted_class, confidence, predictions

def main():
    st.title("🔬 Skin Cancer Detection System")
    st.markdown("Upload an image of a skin lesion for AI-powered analysis")
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("This AI system analyzes skin lesions and provides:")
        st.write("• Classification of skin condition")
        st.write("• Severity assessment")
        st.write("• Medical recommendations")
        
        st.warning("⚠️ **Disclaimer**: This tool is for educational purposes only and should not replace professional medical diagnosis.")
        
        st.markdown("---")
        st.header("🔧 Troubleshooting")
        st.write("If the model always predicts the same condition:")
        st.code("python retrain_model.py", language="bash")
        st.write("This will create a new working model using your data.")
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Failed to load the model. Please check if the model file exists.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of the skin lesion"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Uploaded Image")
            st.image(image, caption="Uploaded skin lesion", use_column_width=True)
        
        with col2:
            st.subheader("🔍 Analysis Results")
            
            # Add debug toggle
            show_debug = st.checkbox("Show Debug Information", value=False)
            
            with st.spinner("Analyzing image..."):
                try:
                    predicted_class, confidence, all_predictions = predict_skin_condition(model, image)
                    
                    # Debug information
                    if show_debug:
                        st.subheader("🐛 Debug Information")
                        st.write(f"**Predicted class index:** {predicted_class}")
                        st.write(f"**Raw predictions array:** {all_predictions}")
                        st.write(f"**Predictions sum:** {np.sum(all_predictions)}")
                        st.write(f"**All predictions:**")
                        for i, prob in enumerate(all_predictions):
                            class_code, class_name = classes[i]
                            st.write(f"  - Class {i} ({class_code}): {prob:.6f}")
                    
                    # Get class information
                    class_code, class_name = classes[predicted_class]
                    severity_data = severity_info[class_code]
                    
                    # Display prediction
                    st.markdown(f"### Predicted Condition: **{class_name.title()}**")
                    st.markdown(f"**Confidence:** {confidence:.2%}")
                    
                    # Severity indicator
                    severity_color = severity_data['color']
                    st.markdown(f"**Severity:** <span style='color: {severity_color}; font-weight: bold;'>{severity_data['severity']}</span>", unsafe_allow_html=True)
                    
                    # Progress bar for confidence
                    st.progress(confidence)
                    
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    logger.error(f"Prediction error: {e}", exc_info=True)
                    return
        
        # Detailed information
        st.markdown("---")
        
        col3, col4 = st.columns([1, 1])
        
        with col3:
            st.subheader("📋 Condition Details")
            st.write(f"**Full Name:** {class_name}")
            st.write(f"**Description:** {severity_data['description']}")
            
            # All predictions
            st.subheader("📊 All Predictions")
            for i, (code, name) in classes.items():
                prob = all_predictions[i]
                st.write(f"**{name}:** {prob:.2%}")
        
        with col4:
            st.subheader("🏥 Medical Recommendations")
            
            # Color-coded severity box
            severity_color = severity_data['color']
            st.markdown(f"""
            <div style='padding: 10px; border-left: 5px solid {severity_color}; background-color: rgba(255,255,255,0.1); margin: 10px 0;'>
                <strong>Severity Level: {severity_data['severity']}</strong>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("**Recommended Actions:**")
            for i, recommendation in enumerate(severity_data['recommendations'], 1):
                st.write(f"{i}. {recommendation}")
        
        # Additional warnings for high-risk conditions
        if class_code in ['mel', 'bcc', 'akiec']:
            st.error("⚠️ **IMPORTANT**: This condition requires immediate medical attention. Please consult a dermatologist as soon as possible.")
        
        # General disclaimer
        st.markdown("---")
        st.info("**Medical Disclaimer**: This AI tool is designed to assist in preliminary screening only. Always consult with qualified healthcare professionals for proper diagnosis and treatment. Do not delay seeking medical care based on these results.")

if __name__ == "__main__":
    main()