import tensorflow as tf
import tensorflow.keras.layers as layers

def norm_act(x, activation=tf.nn.relu):
    x = layers.BatchNormalization(axis=1)(x)

    if activation is not None:
        if activation == 'leaky_relu':
            activation = tf.nn.leaky_relu
        x = layers.Activation(activation)(x)
    return x


def conv_norm(x, filters, kernel_size=3, strides=1, activation=tf.nn.relu, do_norm_act=True):

    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same',
                      use_bias=not do_norm_act, data_format="channels_first")(x)
    if do_norm_act:
        x = norm_act(x, activation=activation)
    return x


def BasicBlock(inp, filters, strides=1, activation=tf.nn.relu, dp_rate=0,
               suffix=1, *args, **kwargs):

    in_filters = inp.shape[-1]

    x = norm_act(inp, activation=activation)

    if in_filters != filters:   # use conv_shortcut to increase the filters of identity.
        identity = conv_norm(x, filters, kernel_size=1, strides=1, activation=activation, do_norm_act=False)
    elif strides > 1:               # else just downsample or conv1x1 with strides can be tried.
        identity = layers.MaxPool2D(data_format="channels_first")(inp)
    else:                           # or keep the same.
        identity = inp

    x = conv_norm(x, filters, kernel_size=3, activation=activation, strides=strides)
    x = conv_norm(x, filters, kernel_size=3, activation=activation, do_norm_act=False)

    if dp_rate:
        x = layers.Dropout(dp_rate)(x)

    x = layers.Add()([identity, x])

    return x


def Bottleneck(inp, filters, strides=1, activation=tf.nn.relu, expansion=4,
               dp_rate=0, suffix=1, *args, **kwargs):

    in_filters = inp.shape[1]
    out_filters = filters*expansion

    x = norm_act(inp, activation=activation)

    identity = inp
    conv_shortcut = False
    if in_filters != out_filters:   # use conv_shortcut to increase the filters of identity.
        identity = conv_norm(x, out_filters, kernel_size=1, strides=1, activation=activation, do_norm_act=False)
        conv_shortcut = True
    if strides > 1:
        mip = identity if conv_shortcut else inp
        identity = layers.MaxPool2D(data_format="channels_first")(mip)

    x = conv_norm(x, filters, kernel_size=1, activation=activation)      # contract
    x = conv_norm(x, filters, kernel_size=3, activation=activation, strides=strides)
    x = conv_norm(x, out_filters, kernel_size=1, activation=activation, do_norm_act=False) # expand

    if dp_rate:
        x = layers.Dropout(dp_rate)(x)

    x = layers.Add()([identity, x])

    return x