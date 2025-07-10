import os
import platform
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

# Configure TensorFlow for optimal performance
def configure_hardware():
    # Disable oneDNN custom operations for better compatibility
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # Check for Apple Silicon (MPS)
    if platform.system() == 'Darwin' and platform.processor() == 'arm':
        print("Apple Silicon (M1/M2) detected. Configuring for MPS...")
        # Enable memory growth for MPS if needed
        try:
            physical_devices = tf.config.list_physical_devices('MPS')
            if physical_devices:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
                print(f"Using Apple Silicon GPU (MPS): {physical_devices[0]}")
                return '/device:MPS:0'
        except (ImportError, RuntimeError) as e:
            print(f"Could not configure MPS: {e}")
    
    # Check for CUDA (NVIDIA GPU)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to prevent TensorFlow from allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
            print(f"Using GPU: {gpus[0]}")
            return '/device:GPU:0'
        except RuntimeError as e:
            print(f"Could not configure GPU: {e}")
    
    # Fall back to CPU
    print("No GPU/TPU found. Using CPU.")
    return '/device:CPU:0'

# Set device strategy
strategy = tf.distribute.get_strategy()
print(f"Using device: {strategy}")

# Set random seeds for reproducibility
def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

set_seeds(42)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load the dataset
train_df = pd.read_csv('Idealize 2025 Datathon Competition/train.csv')

# Display basic information about the dataset
print("Dataset shape:", train_df.shape)
print("\nFirst few rows:")
print(train_df.head())
print("\nDataset info:")
print(train_df.info())
print("\nMissing values per column:")
print(train_df.isnull().sum())

# Define features and target
# Assuming 'survival_status' is the target variable (you'll need to adjust this based on your actual target column)
# If the target column has a different name, please update the following line
X = train_df.drop(['record_id', 'first_name', 'last_name', 'survival_status'], axis=1, errors='ignore')
y = train_df['survival_status']  # Update this to your actual target column name

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("\nCategorical features:", categorical_features)
print("Numerical features:", numerical_features)

# Create preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocess the data
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)

# Get the number of features after one-hot encoding
num_features = X_train_processed.shape[1]
print(f"\nNumber of features after preprocessing: {num_features}")

# Build the model
def create_model(input_dim):
    with strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
    return model

# Create and train the model
print("\nCreating model...")
model = create_model(num_features)
model.summary()

# Define callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

# Learning rate reduction on plateau
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# Model checkpoint
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Train the model
print("\nTraining the model...")
history = model.fit(
    X_train_processed, y_train,
    validation_data=(X_val_processed, y_val),
    epochs=200,  # Increased max epochs since we have early stopping
    batch_size=256,  # Increased batch size for better GPU utilization
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1,
    workers=4,
    use_multiprocessing=True
)

# Evaluate the model
y_pred_proba = model.predict(X_val_processed)
y_pred = (y_pred_proba > 0.5).astype(int)

print("\nValidation Results:")
print("Accuracy:", accuracy_score(y_val, y_pred))
print("ROC AUC:", roc_auc_score(y_val, y_pred_proba))
print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
print("\nTraining history plot saved as 'training_history.png'")

# Save the model and preprocessor
model.save('patient_survival_model.h5')
import joblib
joblib.dump(preprocessor, 'preprocessor.pkl')
print("\nModel and preprocessor saved successfully!")
