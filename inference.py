import tensorflow as tf
import numpy as np
import imageio
import rawpy
import os

from model import MicroISP

IMAGE_HEIGHT = 1500
IMAGE_WIDTH = 2000


def extract_bayer_channels(raw):

    ch_B  = raw[1::2, 1::2]
    ch_Gb = raw[0::2, 1::2]
    ch_R  = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]

    RAW_combined = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))

    RAW_norm = RAW_combined.astype(np.float32) / (4 * 255)
    return RAW_norm


def process_photos():

    g = tf.Graph()
    with g.as_default(), tf.compat.v1.Session() as sess:

        input_image = tf.compat.v1.placeholder(tf.float32, [1, IMAGE_HEIGHT, IMAGE_WIDTH, 4], name="input")
        output_image = MicroISP(input_image)

        sess.run(tf.compat.v1.global_variables_initializer())

        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, "pretrained_weights/MicroISP.ckpt")

        files_list = os.listdir("sample_RAW_photos/")
        for file_name in files_list:

            raw = rawpy.imread("sample_RAW_photos/" + file_name)
            input_data = np.expand_dims(extract_bayer_channels(raw.raw_image), axis=0)

            print("Processing image " + str(file_name) + "...")

            output = sess.run(output_image, feed_dict={input_image: input_data})
            output_resized = np.squeeze(output * 255, axis=0)

            output_resized[output_resized < 0] = 0
            output_resized[output_resized > 255] = 255

            print("Saving image " + str(file_name))

            imageio.imsave("sample_visual_results/" + file_name.split(".")[0] + ".png", output_resized.astype(np.uint8), compress_level=3)


if __name__ == "__main__":

    process_photos()
