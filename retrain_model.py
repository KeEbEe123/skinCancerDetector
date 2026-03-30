import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from keras.utils import to_categorical
import warnings
warnings.filterwarnings("ignore")

def retrain_simple_model():
    print("Loading data...")
    
    # Load data
    data = pd.read_csv('hmnist_28_28_RGB.csv')
    print(f"Data shape: {data.shape}")
    
    # Separate features and labels
    Label = data["label"].values
    Data = data.drop(columns=["label"]).values
    
    # Reshape data to image format
    Data = Data.reshape(-1, 28, 28, 3)
    print(f"Reshaped data: {Data.shape}")
    print(f"Labels shape: {Label.shape}")
    print(f"Unique labels: {np.unique(Label)}")
    print(f"Label distribution: {np.bincount(Label)}")
    
    # Normalize data
    Data = Data.astype(np.float32) / 255.0
    
    # Convert labels to categorical
    Label_categorical = to_categorical(Label, num_classes=7)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        Data, Label_categorical, test_size=0.2, random_state=42, stratify=Label
    )
    
    print(f"Training set: {X_train.shape}, {y_train.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")
    
    # Create a simpler model
    print("Creating model...")
    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28, 3)),
        
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.BatchNormalization(),
        
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.BatchNormalization(),
        
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.BatchNormalization(),
        
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(7, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model summary:")
    model.summary()
    
    # Calculate class weights for imbalanced data
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(Label),
        y=Label
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class weights: {class_weight_dict}")
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=10,  # Reduced for quick training
        validation_data=(X_test, y_test),
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Evaluate model
    print("Evaluating model...")
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Test with different inputs to ensure model is working
    print("\nTesting model with different inputs...")
    test_inputs = [
        ("Random", np.random.random((1, 28, 28, 3))),
        ("Zeros", np.zeros((1, 28, 28, 3))),
        ("Ones", np.ones((1, 28, 28, 3))),
    ]
    
    predictions = []
    for name, test_input in test_inputs:
        pred = model.predict(test_input, verbose=0)[0]
        predicted_class = np.argmax(pred)
        confidence = pred[predicted_class]
        predictions.append(predicted_class)
        print(f"{name}: Class {predicted_class}, Confidence {confidence:.4f}")
    
    # Check if model produces different predictions
    if len(set(predictions)) > 1:
        print("✓ Model produces different predictions for different inputs")
        
        # Save the working model
        model.save('models/Skin_retrained.h5')
        print("✓ Model saved as models/Skin_retrained.h5")
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        with open('models/Skin_retrained.tflite', 'wb') as f:
            f.write(tflite_model)
        print("✓ TFLite model saved as models/Skin_retrained.tflite")
        
    else:
        print("⚠️ Model still has issues - all predictions are the same")
    
    return model

if __name__ == "__main__":
    retrain_simple_model()