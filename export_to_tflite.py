import tensorflow as tf
from model import MicroISP


def convert_model(IMAGE_HEIGHT, IMAGE_WIDTH):

    g = tf.Graph()
    with g.as_default(), tf.compat.v1.Session() as sess:

        input_image = tf.compat.v1.placeholder(tf.float32, [1, IMAGE_HEIGHT, IMAGE_WIDTH, 4], name="input")
        output = MicroISP(input_image)

        sess.run(tf.compat.v1.global_variables_initializer())

        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, "pretrained_weights/MicroISP.ckpt")

        converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, [input_image], [output])

        converter.experimental_new_converter = True
        tflite_model = converter.convert()

        open("tflite_model/MicroISP.tflite", "wb").write(tflite_model)


if __name__ == "__main__":

    INPUT_IMAGE_HEIGHT = 1500
    INPUT_IMAGE_WIDTH = 2000

    convert_model(INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)
