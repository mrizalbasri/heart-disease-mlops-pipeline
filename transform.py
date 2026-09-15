import tensorflow as tf
import tensorflow_transform as tft

NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
LABEL_KEY = "target"

def transformed_name(key: str) -> str:
    """Generate transformed feature name."""
    return f"{key}_xf"

def preprocessing_fn(inputs):
    """Preprocess input features into transformed features."""
    outputs = {}
    
    # Scale numerical features using z-score normalization
    for feature in NUMERICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.scale_to_z_score(inputs[feature])
        
    # Cast categorical features to int64
    for feature in CATEGORICAL_FEATURES:
        outputs[transformed_name(feature)] = tf.cast(inputs[feature], tf.int64)
        
    # Target label
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)
    
    return outputs
