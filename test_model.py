import tensorflow as tf
import numpy as np

def test_model():
    print("Testing model loading and prediction...")
    
    try:
        # Load model
        print("Loading model...")
        model = tf.keras.models.load_model('models/Skin.h5')
        print(f"Model loaded successfully!")
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
        
        # Print model summary
        print("\nModel Summary:")
        model.summary()
        
        # Test with random data
        print("\nTesting with random data...")
        for i in range(5):
            random_input = np.random.random((1, 28, 28, 3))
            prediction = model.predict(random_input, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            
            print(f"\nTest {i+1}:")
            print(f"  Predicted class: {predicted_class}")
            print(f"  Confidence: {confidence:.6f}")
            print(f"  All predictions: {prediction[0]}")
            print(f"  Predictions sum: {np.sum(prediction[0])}")
        
        # Test with different input patterns
        print("\nTesting with specific patterns...")
        
        # All zeros
        zeros_input = np.zeros((1, 28, 28, 3))
        prediction = model.predict(zeros_input, verbose=0)
        print(f"All zeros - Predicted class: {np.argmax(prediction[0])}, Confidence: {prediction[0][np.argmax(prediction[0])]:.6f}")
        
        # All ones
        ones_input = np.ones((1, 28, 28, 3))
        prediction = model.predict(ones_input, verbose=0)
        print(f"All ones - Predicted class: {np.argmax(prediction[0])}, Confidence: {prediction[0][np.argmax(prediction[0])]:.6f}")
        
        # Half values
        half_input = np.full((1, 28, 28, 3), 0.5)
        prediction = model.predict(half_input, verbose=0)
        print(f"All 0.5s - Predicted class: {np.argmax(prediction[0])}, Confidence: {prediction[0][np.argmax(prediction[0])]:.6f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model()