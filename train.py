import os
import time
import itertools
import numpy as np
from argparse import ArgumentParser

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers.experimental.preprocessing import (
    StringLookup,
    TextVectorization,
)

AUTOTUNE = tf.data.experimental.AUTOTUNE
MAXLEN = 7
VOCAB = list("0123456789+")
embedding_size = 8


def gen():
    for _ in itertools.count(1):
        n1 = np.random.randint(0, 1000)
        n2 = np.random.randint(0, 1000)
        tot = n1 + n2
        yield str(n1) + "+" + str(n2), str(tot)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    args = parser.parse_args()

    vectorize_layer = TextVectorization(
        standardize=None,
        # Split on chars
        split=lambda x: tf.strings.unicode_split(x, "UTF-8"),
        output_mode="int",
    )
    vectorize_layer.set_vocabulary(VOCAB)
    vocab_size = len(vectorize_layer.get_vocabulary())
    i_layer = StringLookup(
        vocabulary=vectorize_layer.get_vocabulary(), invert=True
    )

    def vectorize_target(x, y):
        return x, vectorize_layer(y)

    dataset = tf.data.Dataset.from_generator(
        gen, (tf.string, tf.string), (tf.TensorShape([]), tf.TensorShape([]))
    )
    dataset = dataset.batch(args.batch_size)
    dataset = dataset.map(vectorize_target)
    dataset = dataset.prefetch(2)

    inputs, targets = next(iter(dataset))

    model = tf.keras.Sequential(
        [
            vectorize_layer,
            tf.keras.layers.Embedding(vocab_size, embedding_size),
            tf.keras.layers.LSTM(128),
            # As the decoder RNN's input, repeatedly provide with the last output of
            # RNN for each time step. Repeat 'DIGITS + 1' times as that's the maximum
            # length of output, e.g., when DIGITS=3, max output is 999+999=1998.
            tf.keras.layers.RepeatVector(4),
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.Dense(vocab_size, activation="softmax"),
        ]
    )

    optimizer = tf.keras.optimizers.Adam(args.lr)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    class TranslateMonitor(tf.keras.callbacks.Callback):
        def __init__(self, log_dir, inputs, targets):
            super().__init__()
            self.file_writer = tf.summary.create_file_writer(log_dir)
            self.inputs, self.targets = inputs, targets

        def on_epoch_end(self, epoch, logs=None):
            header_row = "Input | Target | Prediction | Correct"
            preds = model(inputs, training=False)
            preds_ints = tf.argmax(preds, axis=-1)

            preds_outs = i_layer(preds_ints)
            preds_strings = tf.strings.reduce_join(preds_outs, axis=-1)

            targets_outs = i_layer(targets)
            targets_strings = tf.strings.reduce_join(targets_outs, axis=-1)

            correct = tf.reduce_all(preds_ints == targets, axis=1)
            correct_str = tf.where(correct, "✅", "❌")

            table_rows = tf.strings.join(
                [
                    inputs,
                    " | ",
                    targets_strings,
                    " | ",
                    preds_strings,
                    " | ",
                    correct_str,
                ]
            )
            table_body = tf.strings.reduce_join(
                inputs=table_rows, separator="\n"
            )
            table = tf.strings.join(
                [header_row, "---|---|---|---", table_body], separator="\n"
            )

            with self.file_writer.as_default():
                tf.summary.text("Samples", table, step=epoch)

    model.fit(
        dataset,
        # epochs=args.epochs,
        epochs=1,
        steps_per_epoch=2000,
        callbacks=[
            TensorBoard(log_dir=args.log_dir, profile_batch=0),
            TranslateMonitor(
                log_dir=args.log_dir, inputs=inputs, targets=targets
            ),
        ],
    )

    inputs = tf.keras.Input(shape=(1,), dtype="string")
    outputs_preds = model(inputs)
    outputs_int = tf.argmax(outputs_preds, axis=-1)
    targets_outs = i_layer(outputs_int)
    outputs = tf.strings.reduce_join(targets_outs, axis=-1)

    end_to_end_model = tf.keras.Model(inputs, outputs)
    end_to_end_model.save(os.path.join(args.log_dir, "model"))

    # model.save(os.path.join(args.log_dir, "model"))
