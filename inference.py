import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

def load_model_and_preprocessor():
    """Load the trained model and preprocessor."""
    try:
        # Load the model
        model = tf.keras.models.load_model('patient_survival_model.h5')
        print("Model loaded successfully!")
        
        # Load the preprocessor
        preprocessor = joblib.load('preprocessor.pkl')
        print("Preprocessor loaded successfully!")
        
        return model, preprocessor
    except Exception as e:
        print(f"Error loading model or preprocessor: {e}")
        return None, None

def preprocess_test_data(test_df, preprocessor):
    """Preprocess test data using the fitted preprocessor."""
    # Keep record_id for later use
    record_ids = test_df['record_id'].copy()
    
    # Remove columns that were not used in training
    # Assuming the same columns were dropped during training
    columns_to_drop = ['record_id', 'first_name', 'last_name', 'survival_status']
    X_test = test_df.drop(columns_to_drop, axis=1, errors='ignore')
    
    # Transform the test data using the fitted preprocessor
    X_test_processed = preprocessor.transform(X_test)
    
    return X_test_processed, record_ids

def make_predictions(model, X_test_processed):
    """Make predictions using the trained model."""
    # Get prediction probabilities
    y_pred_proba = model.predict(X_test_processed)
    
    # Convert probabilities to binary predictions (threshold = 0.5)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    return y_pred, y_pred_proba.flatten()

def save_predictions(record_ids, predictions, probabilities, output_file='test_predictions.csv'):
    """Save predictions to CSV file."""
    results_df = pd.DataFrame({
        'record_id': record_ids,
        'prediction': predictions,
        'probability': probabilities
    })
    
    results_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    return results_df

def main():
    """Main inference function."""
    print("Starting inference process...")
    
    # Check if required files exist
    if not os.path.exists('test.csv'):
        print("Error: test.csv file not found!")
        return
    
    if not os.path.exists('patient_survival_model.h5'):
        print("Error: patient_survival_model.h5 not found!")
        return
    
    if not os.path.exists('preprocessor.pkl'):
        print("Error: preprocessor.pkl not found!")
        return
    
    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv('test.csv')
    print(f"Test data shape: {test_df.shape}")
    
    # Load model and preprocessor
    model, preprocessor = load_model_and_preprocessor()
    if model is None or preprocessor is None:
        return
    
    # Preprocess test data
    print("Preprocessing test data...")
    X_test_processed, record_ids = preprocess_test_data(test_df, preprocessor)
    print(f"Processed test data shape: {X_test_processed.shape}")
    
    # Make predictions
    print("Making predictions...")
    predictions, probabilities = make_predictions(model, X_test_processed)
    
    # Save results
    results_df = save_predictions(record_ids, predictions, probabilities)
    
    # Display summary
    print(f"\nPrediction Summary:")
    print(f"Total predictions: {len(predictions)}")
    print(f"Predicted positive cases: {np.sum(predictions)}")
    print(f"Predicted negative cases: {len(predictions) - np.sum(predictions)}")
    print(f"Average prediction probability: {np.mean(probabilities):.3f}")
    
    print("\nFirst 10 predictions:")
    print(results_df.head(10))

if __name__ == "__main__":
    main()
