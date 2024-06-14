import tensorflow as tf


def attention(x, num_features, hidden_units, attention_dims=4):

    w = tf.Variable(tf.compat.v1.random_normal([1, 1, num_features, num_features], stddev=1e-3))
    b = tf.Variable(tf.zeros([num_features]))

    x = tf.nn.conv2d(x, w, strides=[1, 3, 3, 1], padding='VALID') + b
    x = tf.keras.layers.PReLU(shared_axes=[1,2])(x)

    w = tf.Variable(tf.compat.v1.random_normal([3, 3, num_features, num_features], stddev=1e-3))
    b = tf.Variable(tf.zeros([num_features]))

    x = tf.nn.conv2d(x, w, strides=[1, 3, 3, 1], padding='VALID') + b
    x = tf.keras.layers.PReLU(shared_axes=[1,2])(x)

    w = tf.Variable(tf.compat.v1.random_normal([3, 3, num_features, num_features], stddev=1e-3))
    b = tf.Variable(tf.zeros([num_features]))

    x = tf.nn.conv2d(x, w, strides=[1, 3, 3, 1], padding='VALID') + b
    x = tf.keras.layers.PReLU(shared_axes=[1,2])(x)

    w = tf.Variable(tf.compat.v1.random_normal([3, 3, num_features, num_features], stddev=1e-3))
    b = tf.Variable(tf.zeros([num_features]))

    x = tf.nn.conv2d(x, w, strides=[1, 3, 3, 1], padding='VALID') + b
    x = tf.keras.layers.PReLU(shared_axes=[1,2])(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.reshape(x, [-1, 1, 1, num_features])

    w = tf.Variable(tf.compat.v1.random_normal([1, 1, num_features, hidden_units], stddev=1e-3))
    b = tf.Variable(tf.zeros([hidden_units]))

    x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b
    x = tf.keras.layers.PReLU(shared_axes=[1,2])(x)

    w = tf.Variable(tf.compat.v1.random_normal([1, 1, hidden_units, attention_dims], stddev=1e-3))
    b = tf.Variable(tf.zeros([attention_dims]))

    x = tf.nn.sigmoid(tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b)

    return x


def MicroISP(image,
                    num_features=4,
                    input_channels=4,
                    output_channels=3,
                    num_blocks=7,
                    res_skip=3,
                    scale=2):

    with tf.compat.v1.variable_scope("generator-1"):

        weights = {}
        biases = {}

        for i in range(output_channels):

            for j in range(num_blocks):

                w_id = str(i) + '_w1_' + str(j)
                b_id = str(i) + '_b1_' + str(j)

                if j == 0:
                    weights[w_id] = tf.Variable(tf.compat.v1.random_normal([3, 3, input_channels, num_features], stddev=1e-3), name=w_id)
                else:
                    weights[w_id] = tf.Variable(tf.compat.v1.random_normal([3, 3, num_features, num_features], stddev=1e-3), name=w_id)

                biases[b_id] = tf.Variable(tf.zeros([num_features]), name=b_id)

            w_out = str(i) + '_w1_out'
            b_out = str(i) + '_b1_out'

            weights[w_out] = tf.Variable(tf.compat.v1.random_normal([3, 3, num_features, scale*scale], stddev=1e-3), name=w_out)
            biases[b_out] = tf.Variable(tf.zeros([scale*scale]), name=b_out)

        outputs_1 = []
        outputs_raw = []

        for i in range(output_channels):

            convs = []

            conv = None
            conv_skip = None

            for j in range(num_blocks):

                w_id = str(i) + '_w1_' + str(j)
                b_id = str(i) + '_b1_' + str(j)

                conv = tf.keras.layers.PReLU(shared_axes=[1,2])(tf.nn.conv2d(image if j == 0 else conv, weights[w_id], strides=[1, 1, 1, 1], padding='SAME') + biases[b_id])

                if j % res_skip == 0:

                    conv_skip = conv

                elif j % res_skip == res_skip - 1:

                    c = attention(conv, num_features, hidden_units=num_features)
                    conv = conv * c
                    conv = conv + conv_skip

                    convs.append(conv)

            w_out = str(i) + '_w1_out'
            b_out = str(i) + '_b1_out'

            conv_out = tf.nn.conv2d(conv, weights[w_out], strides=[1, 1, 1, 1], padding='SAME') + biases[b_out]

            conv_out = tf.nn.tanh(conv_out)
            outputs_raw.append(conv_out)

            conv_out = conv_out * 0.58 + 0.5
            outputs_1.append(tf.nn.depth_to_space(conv_out, scale))

    with tf.compat.v1.variable_scope("generator-2"):

        weights = {}
        biases = {}

        for i in range(output_channels):

            for j in range(num_blocks):

                w_id = str(i) + '_w2_' + str(j)
                b_id = str(i) + '_b2_' + str(j)

                if j == 0:
                    weights[w_id] = tf.Variable(tf.compat.v1.random_normal([3, 3, input_channels, num_features], stddev=1e-3), name=w_id)
                else:
                    weights[w_id] = tf.Variable(tf.compat.v1.random_normal([3, 3, num_features, num_features], stddev=1e-3), name=w_id)

                biases[b_id] = tf.Variable(tf.zeros([num_features]), name=b_id)

            w_out = str(i) + '_w2_out'
            b_out = str(i) + '_b2_out'

            weights[w_out] = tf.Variable(tf.compat.v1.random_normal([3, 3, num_features, scale*scale], stddev=1e-3), name=w_out)
            biases[b_out] = tf.Variable(tf.zeros([scale*scale]), name=b_out)

        outputs_2 = []

        for i in range(output_channels):

            convs = []

            conv = None
            conv_skip = None

            for j in range(num_blocks):

                w_id = str(i) + '_w2_' + str(j)
                b_id = str(i) + '_b2_' + str(j)

                conv = tf.keras.layers.PReLU(shared_axes=[1,2])(tf.nn.conv2d(outputs_raw[i] if j == 0 else conv, weights[w_id], strides=[1, 1, 1, 1], padding='SAME') + biases[b_id])

                if j % res_skip == 0:

                    conv_skip = conv

                elif j % res_skip == res_skip - 1:

                    c = attention(conv, num_features, hidden_units=num_features, attention_dims=4)
                    conv = conv * c
                    conv = conv + conv_skip

                    convs.append(conv)

            w_out = str(i) + '_w2_out'
            b_out = str(i) + '_b2_out'

            conv_out = tf.nn.conv2d(conv, weights[w_out], strides=[1, 1, 1, 1], padding='SAME') + biases[b_out]
            conv_out = tf.nn.tanh(conv_out) * 0.58 + 0.5

            outputs_2.append(tf.nn.depth_to_space(conv_out, scale))

        result = tf.concat(outputs_2, axis=-1)

    return result

