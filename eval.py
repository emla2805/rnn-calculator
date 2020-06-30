from argparse import ArgumentParser
import tensorflow as tf


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-dir", default="logs/model")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model_dir)

    def eval(expr):
        return model(tf.constant([expr])).numpy()[0].decode()
