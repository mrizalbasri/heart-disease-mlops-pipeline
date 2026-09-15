import os
import tensorflow as tf
import tensorflow_transform as tft
import keras_tuner as kt
from tfx.components.trainer.fn_args_utils import FnArgs

NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
LABEL_KEY = "target"

def transformed_name(key: str) -> str:
    """Generate transformed feature name."""
    return f"{key}_xf"

def gzip_reader_fn(filenames):
    """Small utility for reading gzip-compressed TFRecord files."""
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")

def input_fn(file_pattern, tf_transform_output, batch_size=32):
    """Generate batched TFRecord dataset for training and evaluation."""
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=None,
        label_key=transformed_name(LABEL_KEY)
    )
    return dataset

def get_model(hp=None):
    """Build Keras Deep Neural Network model, optionally with tuned hyperparameters."""
    input_features = []
    float_inputs = []
    
    for feature in NUMERICAL_FEATURES:
        inp = tf.keras.Input(shape=(1,), name=transformed_name(feature), dtype=tf.float32)
        input_features.append(inp)
        float_inputs.append(inp)
        
    for feature in CATEGORICAL_FEATURES:
        inp = tf.keras.Input(shape=(1,), name=transformed_name(feature), dtype=tf.int64)
        input_features.append(inp)
        float_inputs.append(tf.cast(inp, tf.float32))
        
    concat = tf.keras.layers.concatenate(float_inputs)
    
    if hp is not None and hasattr(hp, "get"):
        units_1 = hp.get("units_1") if "units_1" in hp.values else 64
        units_2 = hp.get("units_2") if "units_2" in hp.values else 32
        dropout = hp.get("dropout") if "dropout" in hp.values else 0.2
        lr = hp.get("learning_rate") if "learning_rate" in hp.values else 0.001
    else:
        units_1 = 64
        units_2 = 32
        dropout = 0.2
        lr = 0.001
        
    x = tf.keras.layers.Dense(units_1, activation="relu")(concat)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(units_2, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    
    model = tf.keras.Model(inputs=input_features, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model

def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses raw serialized tf.Example for serving."""
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY, None)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)

    return serve_tf_examples_fn

def run_fn(fn_args: FnArgs):
    """Train the model based on given args."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    
    train_dataset = input_fn(fn_args.train_files, tf_transform_output, batch_size=32)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, batch_size=32)
    
    hp = None
    if fn_args.hyperparameters:
        hp = kt.HyperParameters.from_config(fn_args.hyperparameters)
        
    model = get_model(hp)
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir, update_freq="batch"
    )
    
    train_steps = fn_args.train_steps or 20
    eval_steps = fn_args.eval_steps or 5
    
    model.fit(
        train_dataset,
        steps_per_epoch=train_steps,
        validation_data=eval_dataset,
        validation_steps=eval_steps,
        callbacks=[tensorboard_callback],
        epochs=10
    )
    
    signatures = {
        "serving_default": _get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        )
    }
    
    model.save(fn_args.serving_model_dir, save_format="tf", signatures=signatures)
