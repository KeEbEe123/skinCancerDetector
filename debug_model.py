import tensorflow as tf
import numpy as np

def debug_model():
    print("=== MODEL DEBUGGING ===")
    
    try:
        # Load model
        model = tf.keras.models.load_model('models/Skin.h5')
        print("✓ Model loaded successfully")
        
        # Check model architecture
        print(f"\nModel input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        print(f"Number of layers: {len(model.layers)}")
        
        # Check if model has been trained (look at weights)
        print("\n=== WEIGHT ANALYSIS ===")
        for i, layer in enumerate(model.layers[:5]):  # Check first 5 layers
            if hasattr(layer, 'get_weights') and layer.get_weights():
                weights = layer.get_weights()[0]  # Get first weight matrix
                print(f"Layer {i} ({layer.name}): weight std = {np.std(weights):.6f}, mean = {np.mean(weights):.6f}")
                
                # Check if weights are all similar (indicating untrained model)
                if np.std(weights) < 0.001:
                    print(f"  ⚠️  WARNING: Layer {i} has very low weight variance - might be untrained!")
        
        # Test with diverse inputs
        print("\n=== PREDICTION TESTS ===")
        
        test_cases = [
            ("Random noise", np.random.random((1, 28, 28, 3))),
            ("All zeros", np.zeros((1, 28, 28, 3))),
            ("All ones", np.ones((1, 28, 28, 3))),
            ("All 0.5", np.full((1, 28, 28, 3), 0.5)),
            ("Gradient pattern", np.tile(np.linspace(0, 1, 28).reshape(1, 28, 1), (1, 1, 28, 3))),
            ("Checkerboard", np.tile(np.array([[[0, 0, 0], [1, 1, 1]] * 14] * 14), (1, 1, 1, 1))),
        ]
        
        predictions_summary = []
        
        for name, test_input in test_cases:
            pred = model.predict(test_input, verbose=0)[0]
            predicted_class = np.argmax(pred)
            confidence = pred[predicted_class]
            
            print(f"\n{name}:")
            print(f"  Predicted class: {predicted_class}")
            print(f"  Confidence: {confidence:.6f}")
            print(f"  All predictions: {[f'{p:.6f}' for p in pred]}")
            
            predictions_summary.append((name, predicted_class, confidence))
        
        # Check if all predictions are the same
        print("\n=== SUMMARY ===")
        unique_predictions = set([p[1] for p in predictions_summary])
        if len(unique_predictions) == 1:
            print("🚨 PROBLEM: All inputs predict the same class!")
            print("This indicates the model is broken, corrupted, or not properly trained.")
        else:
            print(f"✓ Model produces {len(unique_predictions)} different predictions")
        
        # Check prediction confidence
        confidences = [p[2] for p in predictions_summary]
        avg_confidence = np.mean(confidences)
        if avg_confidence > 0.99:
            print(f"🚨 PROBLEM: Average confidence is {avg_confidence:.4f} - too high!")
            print("This suggests the model is overconfident or broken.")
        
        # Try to load the original training script's model save
        print("\n=== CHECKING ALTERNATIVE MODEL PATHS ===")
        alternative_paths = [
            'Skin Cancer.h5',
            'Skin.h5', 
            './Skin Cancer.h5',
            './Skin.h5'
        ]
        
        for path in alternative_paths:
            try:
                alt_model = tf.keras.models.load_model(path)
                print(f"✓ Found model at: {path}")
                
                # Quick test
                test_pred = alt_model.predict(np.random.random((1, 28, 28, 3)), verbose=0)[0]
                print(f"  Sample prediction: class {np.argmax(test_pred)}, confidence {test_pred[np.argmax(test_pred)]:.6f}")
                
            except Exception as e:
                print(f"✗ No model at: {path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model()