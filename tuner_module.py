import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components.tuner.component import TunerFnResult
import keras_tuner as kt

from trainer_module import _build_keras_model, _input_fn


def tuner_fn(fn_args: FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = _input_fn(
        fn_args.train_files, tf_transform_output, num_epochs=1, batch_size=64
    )
    eval_dataset = _input_fn(
        fn_args.eval_files, tf_transform_output, num_epochs=1, batch_size=64
    )

    tuner = kt.RandomSearch(
        hypermodel=lambda hp: _build_keras_model(tf_transform_output, hp),
        objective=kt.Objective('val_binary_accuracy', direction='max'),
        max_trials=5,
        executions_per_trial=1,
        directory=fn_args.working_dir,
        project_name='adult_income_tuning',
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'x': train_dataset,
            'validation_data': eval_dataset,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps,
        }
    )
