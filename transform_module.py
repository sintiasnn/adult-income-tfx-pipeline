import tensorflow as tf

NUMERIC_FEATURES = [
    'age', 'fnlwgt', 'education-num',
    'capital-gain', 'capital-loss', 'hours-per-week',
]

CATEGORICAL_FEATURES = [
    'workclass', 'education', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'native-country',
]

LABEL_KEY = 'income'


def transformed_name(key):
    return key + '_xf'


def preprocessing_fn(inputs):
    outputs = {}
    for feature in NUMERIC_FEATURES:
        outputs[transformed_name(feature)] = tf.cast(inputs[feature], tf.float32)
    for feature in CATEGORICAL_FEATURES:
        outputs[transformed_name(feature)] = tf.cast(inputs[feature], tf.int64)
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)
    return outputs
