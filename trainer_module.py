import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs
import keras_tuner as kt

NUMERIC_FEATURES = [
    'age', 'fnlwgt', 'education-num',
    'capital-gain', 'capital-loss', 'hours-per-week',
]

CATEGORICAL_FEATURES = {
    'workclass': 9,
    'education': 16,
    'marital-status': 7,
    'occupation': 15,
    'relationship': 6,
    'race': 5,
    'sex': 2,
    'native-country': 42,
}

LABEL_KEY = 'income'
EMBEDDING_DIM = 8


def transformed_name(key):
    return key + '_xf'


def _gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _get_serve_tf_examples_fn(model, tf_transform_output):
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec
        )
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)

    return serve_tf_examples_fn


def _input_fn(file_pattern, tf_transform_output, num_epochs=None, batch_size=64):
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY),
    )
    return dataset


def _build_keras_model(tf_transform_output, hp=None):
    units_1 = hp.Int('units_1', min_value=64, max_value=256, step=64) if hp else 128
    units_2 = hp.Int('units_2', min_value=32, max_value=128, step=32) if hp else 64
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1) if hp else 0.3
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]) if hp else 1e-3

    inputs = {}
    embedding_outputs = []

    for feature in NUMERIC_FEATURES:
        inputs[transformed_name(feature)] = tf.keras.Input(
            shape=(1,), name=transformed_name(feature)
        )

    for feature, vocab_size in CATEGORICAL_FEATURES.items():
        cat_input = tf.keras.Input(
            shape=(1,), name=transformed_name(feature), dtype=tf.int64,
            sparse=False,
        )
        inputs[transformed_name(feature)] = cat_input
        embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size + 1,
            output_dim=EMBEDDING_DIM,
            name=f'embedding_{feature}'
        )(cat_input)
        embedding_flat = tf.keras.layers.Flatten()(embedding)
        embedding_outputs.append(embedding_flat)

    numeric_inputs = [inputs[transformed_name(f)] for f in NUMERIC_FEATURES]
    numeric_concat = tf.keras.layers.concatenate(numeric_inputs)
    all_features = tf.keras.layers.concatenate([numeric_concat] + embedding_outputs)

    x = tf.keras.layers.Dense(units_1, activation='relu')(all_features)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(units_2, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
            tf.keras.metrics.AUC(name='auc'),
        ]
    )
    model.summary()
    return model


def run_fn(fn_args: FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = _input_fn(
        fn_args.train_files, tf_transform_output, num_epochs=10, batch_size=64
    )
    eval_dataset = _input_fn(
        fn_args.eval_files, tf_transform_output, num_epochs=1, batch_size=64
    )

    if fn_args.hyperparameters:
        hp = kt.HyperParameters.from_config(fn_args.hyperparameters)
        model = _build_keras_model(tf_transform_output, hp)
    else:
        model = _build_keras_model(tf_transform_output)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_binary_accuracy',
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=fn_args.model_run_dir,
            update_freq='batch',
        ),
    ]

    model.fit(
        train_dataset,
        epochs=10,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=callbacks,
    )

    signatures = {
        'serving_default': _get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
        )
    }

    model.save(
        fn_args.serving_model_dir,
        save_format='tf',
        signatures=signatures,
    )
