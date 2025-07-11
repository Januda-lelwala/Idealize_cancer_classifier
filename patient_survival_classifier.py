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
train_df = pd.read_csv('train.csv')

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

# Print class distribution before oversampling
print("Class distribution before oversampling:", np.bincount(y_train))

# Improved oversampling of the minority class by duplication
def oversample_minority_class(X_processed, y, oversample_ratio=1.0):
    """
    Oversample minority class by duplication
    
    Args:
        X_processed: Preprocessed features
        y: Target labels
        oversample_ratio: Ratio to oversample (1.0 = balance classes, 0.5 = half balance, etc.)
    
    Returns:
        X_oversampled, y_oversampled: Oversampled data
    """
    unique, counts = np.unique(y, return_counts=True)
    
    if len(unique) != 2:
        raise ValueError(f"Expected binary classification, got {len(unique)} classes")
    
    majority_class = unique[np.argmax(counts)]
    minority_class = unique[np.argmin(counts)]
    majority_count = counts.max()
    minority_count = counts.min()
    
    print(f"Majority class {majority_class}: {majority_count} samples")
    print(f"Minority class {minority_class}: {minority_count} samples")
    
    # Calculate number of samples to add
    target_minority_count = int(minority_count + (majority_count - minority_count) * oversample_ratio)
    num_to_add = target_minority_count - minority_count
    
    if num_to_add <= 0:
        print("No oversampling needed or oversample_ratio too low")
        return X_processed, y
    
    print(f"Adding {num_to_add} samples to minority class")
    
    # Get indices of minority class samples
    minority_indices = np.where(y == minority_class)[0]
    
    # Randomly duplicate minority class samples
    np.random.seed(42)  # For reproducibility
    duplicated_indices = np.random.choice(minority_indices, size=num_to_add, replace=True)
    
    # Add duplicated samples
    X_duplicated = X_processed[duplicated_indices]
    y_duplicated = y[duplicated_indices]
    
    X_oversampled = np.vstack([X_processed, X_duplicated])
    y_oversampled = np.concatenate([y, y_duplicated])
    
    return X_oversampled, y_oversampled

# Apply oversampling
X_train_processed, y_train = oversample_minority_class(X_train_processed, y_train, oversample_ratio=1.0)

# Print class distribution after oversampling
print("Class distribution after oversampling:", np.bincount(y_train))

# Verify the oversampling worked correctly
unique_after, counts_after = np.unique(y_train, return_counts=True)
imbalance_ratio = counts_after.min() / counts_after.max()
print(f"Class imbalance ratio after oversampling: {imbalance_ratio:.3f} (1.0 = perfectly balanced)")

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
    epochs=100,
    batch_size=256,  # Increased batch size for better GPU utilization
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1
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
