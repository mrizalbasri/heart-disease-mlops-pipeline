from typing import NamedTuple, Dict, Text, Any
import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft
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

def model_builder(hp):
    """Build Keras model with tunable hyperparameters."""
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
    
    hp_units_1 = hp.Int("units_1", min_value=32, max_value=128, step=32, default=64)
    hp_units_2 = hp.Int("units_2", min_value=16, max_value=64, step=16, default=32)
    hp_dropout = hp.Float("dropout", min_value=0.1, max_value=0.3, step=0.1, default=0.2)
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4], default=1e-3)
    
    x = tf.keras.layers.Dense(hp_units_1, activation="relu")(concat)
    x = tf.keras.layers.Dropout(hp_dropout)(x)
    x = tf.keras.layers.Dense(hp_units_2, activation="relu")(x)
    x = tf.keras.layers.Dropout(hp_dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    
    model = tf.keras.Model(inputs=input_features, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model

TunerFnResult = NamedTuple("TunerFnResult", [
    ("tuner", kt.engine.base_tuner.BaseTuner),
    ("fit_kwargs", Dict[Text, Any])
])

def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """Build the tuner using KerasTuner RandomSearch."""
    transform_graph_path = getattr(fn_args, "transform_graph_path", None) or getattr(fn_args, "transform_output", None)
    tf_transform_output = tft.TFTransformOutput(transform_graph_path)
    
    train_dataset = input_fn(fn_args.train_files, tf_transform_output, batch_size=32)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, batch_size=32)
    
    tuner = kt.RandomSearch(
        hypermodel=model_builder,
        objective=kt.Objective("val_auc", direction="max"),
        max_trials=3,
        directory=fn_args.working_dir,
        project_name="heart_disease_tuning"
    )
    
    train_steps = fn_args.train_steps or 20
    eval_steps = fn_args.eval_steps or 5
    
    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "callbacks": [tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=3)],
            "x": train_dataset,
            "validation_data": eval_dataset,
            "steps_per_epoch": train_steps,
            "validation_steps": eval_steps,
            "epochs": 5
        }
    )
