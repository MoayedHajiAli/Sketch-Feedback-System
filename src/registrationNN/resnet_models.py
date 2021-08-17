# -*- coding: utf-8 -*-

"""
keras_resnet.models._1d
~~~~~~~~~~~~~~~~~~~~~~~
This module implements popular one-dimensional residual models.
"""

import keras.backend
import keras.layers
import keras.models
import keras.regularizers



class BatchNormalization(keras.layers.BatchNormalization):
    """
    Identical to keras.layers.BatchNormalization, but adds the option to freeze parameters.
    """
    def __init__(self, freeze, *args, **kwargs):
        self.freeze = freeze
        super(BatchNormalization, self).__init__(*args, **kwargs)

        # set to non-trainable if freeze is true
        self.trainable = not self.freeze

    def call(self, *args, **kwargs):
        # Force test mode if frozen, otherwise use default behaviour (i.e., training=None).
        if self.freeze:
            kwargs['training'] = False
        return super(BatchNormalization, self).call(*args, **kwargs)

    def get_config(self):
        config = super(BatchNormalization, self).get_config()
        config.update({'freeze': self.freeze})
        return config


class ResNet():
    """
    Constructs a `keras.models.Model` object using the given block count.
    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)
    :param blocks: the network’s residual architecture
    :param block: a residual block (e.g. an instance of `keras_resnet.blocks.basic_1d`)
    :param include_top: if true, includes classification layers
    :param classes: number of classes to classify (include_top must be true)
    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)
    :param numerical_names: list of bool, same size as blocks, used to indicate whether names of layers should include numbers or letters
    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)
    Usage:
        >>> import keras_resnet.blocks
        >>> import keras_resnet.models
        >>> shape, classes = (224, 224, 3), 1000
        >>> x = keras.layers.Input(shape)
        >>> blocks = [2, 2, 2, 2]
        >>> block = keras_resnet.blocks.basic_1d
        >>> model = keras_resnet.models.ResNet(x, classes, blocks, block, classes=classes)
        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    
    def bottleneck_1d(
        self,
        filters,
        stage=0,
        block=0,
        kernel_size=3,
        numerical_name=False,
        stride=None,
        freeze_bn=False
    ):

        """
        A one-dimensional bottleneck block.
        :param filters: the output’s feature space
        :param stage: int representing the stage of this block (starting from 0)
        :param block: int representing this block (starting from 0)
        :param kernel_size: size of the kernel
        :param numerical_name: if true, uses numbers to represent blocks instead of chars (ResNet{101, 152, 200})
        :param stride: int representing the stride used in the shortcut and the first conv layer, default derives stride from block id
        :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)
        Usage:
            >>> import keras_resnet.blocks
            >>> keras_resnet.blocks.bottleneck_1d(64)
        """

        parameters = {"kernel_initializer": "he_normal"}

        if stride is None:
            stride = 1 if block != 0 or stage == 0 else 2

        if keras.backend.image_data_format() == "channels_last":
            axis = -1
        else:
            axis = 1

        if block > 0 and numerical_name:
            block_char = "b{}".format(block)
        else:
            block_char = chr(ord('a') + block)

        stage_char = str(stage + 2)

        def f(x):
            y = keras.layers.Conv1D(
                filters,
                1,
                strides=stride,
                use_bias=False,
                name="res{}{}_branch2a".format(stage_char, block_char),
                **parameters
            )(x)

            y = BatchNormalization(
                axis=axis,
                epsilon=1e-5,
                freeze=freeze_bn,
                name="bn{}{}_branch2a".format(stage_char, block_char)
            )(y)

            y = keras.layers.Activation(
                "relu",
                name="res{}{}_branch2a_relu".format(stage_char, block_char)
            )(y)

            y = keras.layers.ZeroPadding1D(
                padding=1,
                name="padding{}{}_branch2b".format(stage_char, block_char)
            )(y)

            y = keras.layers.Conv1D(
                filters,
                kernel_size,
                use_bias=False,
                name="res{}{}_branch2b".format(stage_char, block_char),
                **parameters
            )(y)

            y = BatchNormalization(
                axis=axis,
                epsilon=1e-5,
                freeze=freeze_bn,
                name="bn{}{}_branch2b".format(stage_char, block_char)
            )(y)

            y = keras.layers.Activation(
                "relu",
                name="res{}{}_branch2b_relu".format(stage_char, block_char)
            )(y)

            y = keras.layers.Conv1D(
                filters * 4,
                1,
                use_bias=False,
                name="res{}{}_branch2c".format(stage_char, block_char),
                **parameters
            )(y)

            y = BatchNormalization(
                axis=axis,
                epsilon=1e-5,
                freeze=freeze_bn,
                name="bn{}{}_branch2c".format(stage_char, block_char)
            )(y)

            if block == 0:
                shortcut = keras.layers.Conv1D(
                    filters * 4,
                    1,
                    strides=stride,
                    use_bias=False,
                    name="res{}{}_branch1".format(stage_char, block_char),
                    **parameters
                )(x)

                shortcut = BatchNormalization(
                    axis=axis,
                    epsilon=1e-5,
                    freeze=freeze_bn,
                    name="bn{}{}_branch1".format(stage_char, block_char)
                )(shortcut)
            else:
                shortcut = x

            y = keras.layers.Add(
                name="res{}{}".format(stage_char, block_char)
            )([y, shortcut])

            y = keras.layers.Activation(
                "relu",
                name="res{}{}_relu".format(stage_char, block_char)
            )(y)

            return y

        return f


    def basic_1d(self, filters,
        stage=0,
        block=0,
        kernel_size=3,
        numerical_name=False,
        stride=None,
        freeze_bn=False):


        parameters = {"kernel_initializer": "he_normal"}

        if stride is None:
            if block != 0 or stage == 0:
                stride = 1
            else:
                stride = 2

        if keras.backend.image_data_format() == "channels_last":
            axis = -1
        else:
            axis = 1

        if block > 0 and numerical_name:
            block_char = "b{}".format(block)
        else:
            block_char = chr(ord('a') + block)

        stage_char = str(stage + 2)

        def f(x):
            y = keras.layers.ZeroPadding1D(
                padding=1, 
                name="padding{}{}_branch2a".format(stage_char, block_char)
            )(x)
            
            y = keras.layers.Conv1D(
                filters,
                kernel_size,
                strides=stride,
                use_bias=False,
                name="res{}{}_branch2a".format(stage_char, block_char),
                **parameters
            )(y)
            
            y = BatchNormalization(
                axis=axis,
                epsilon=1e-5,
                freeze=freeze_bn,
                name="bn{}{}_branch2a".format(stage_char, block_char)
            )(y)
            
            y = keras.layers.Activation(
                "relu",
                name="res{}{}_branch2a_relu".format(stage_char, block_char)
            )(y)

            y = keras.layers.ZeroPadding1D(
                padding=1,
                name="padding{}{}_branch2b".format(stage_char, block_char)
            )(y)
            
            y = keras.layers.Conv1D(
                filters,
                kernel_size,
                use_bias=False,
                name="res{}{}_branch2b".format(stage_char, block_char),
                **parameters
            )(y)
            
            y = BatchNormalization(
                axis=axis,
                epsilon=1e-5,
                freeze=freeze_bn,
                name="bn{}{}_branch2b".format(stage_char, block_char)
            )(y)

            if block == 0:
                shortcut = keras.layers.Conv1D(
                    filters,
                    1,
                    strides=stride,
                    use_bias=False,
                    name="res{}{}_branch1".format(stage_char, block_char),
                    **parameters
                )(x)

                shortcut = BatchNormalization(
                    axis=axis,
                    epsilon=1e-5,
                    freeze=freeze_bn,
                    name="bn{}{}_branch1".format(stage_char, block_char)
                )(shortcut)
            else:
                shortcut = x

            y = keras.layers.Add(
                name="res{}{}".format(stage_char, block_char)
            )([y, shortcut])
            
            y = keras.layers.Activation(
                "relu",
                name="res{}{}_relu".format(stage_char, block_char)
            )(y)

            return y

        return f


    def ResNet1D(
        self,
        inputs,
        blocks = [3, 8, 36, 3],
        block = basic_1d,
        include_top=False,
        final_layer=6,
        freeze_bn=False,
        numerical_names=None,
        id_shift = 0,
        *args,
        **kwargs
    ):
        if keras.backend.image_data_format() == "channels_last":
            axis = 3
        else:
            axis = 1

        if numerical_names is None:
            numerical_names = [True] * len(blocks)

        # x = keras.layers.ZeroPadding1D(padding=3, name="padding_conv1")(inputs)
        # x = keras.layers.Conv1D(64, 7, strides=2, use_bias=False, name="conv1")(x)
        # x = BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn_conv1")(x)
        # x = keras.layers.Activation("relu", name="conv1_relu")(x)
        # x = keras.layers.MaxPooling1D(3, strides=2, padding="same", name="pool1")(x)
        x = inputs
        
        features = 64

        outputs = []

        for stage_id, iterations in enumerate(blocks):
            for block_id in range(iterations):
                x = block(
                    features,
                    stage_id + id_shift,
                    block_id,
                    numerical_name=(block_id > 0 and numerical_names[stage_id]),
                    freeze_bn=freeze_bn
                )(x)

            features *= 2

            outputs.append(x)

        
        if include_top:
            assert final_layer > 0

            x = keras.layers.GlobalAveragePooling1D(name=f"pool5_{id_shift}")(x)
            x = keras.layers.Dense(final_layer, activation="softmax", name=f"fc1000_{id_shift}")(x)

        return x


# class ResNet1D18(ResNet1D):
#     """
#     Constructs a `keras.models.Model` according to the ResNet18 specifications.
#     :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)
#     :param blocks: the network’s residual architecture
#     :param include_top: if true, includes classification layers
#     :param classes: number of classes to classify (include_top must be true)
#     :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)
#     :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)
#     Usage:
#         >>> import keras_resnet.models
#         >>> shape, classes = (224, 224, 3), 1000
#         >>> x = keras.layers.Input(shape)
#         >>> model = keras_resnet.models.ResNet18(x, classes=classes)
#         >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
#     """
#     def __init__(self, inputs, blocks=None, include_top=True, classes=1000, freeze_bn=False, *args, **kwargs):
#         if blocks is None:
#             blocks = [2, 2, 2, 2]

#         super(ResNet1D18, self).__init__(
#             inputs,
#             blocks,
#             block=keras_resnet.blocks.basic_1d,
#             include_top=include_top,
#             classes=classes,
#             freeze_bn=freeze_bn,
#             *args,
#             **kwargs
#         )


# class ResNet1D34(ResNet1D):
#     """
#     Constructs a `keras.models.Model` according to the ResNet34 specifications.
#     :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)
#     :param blocks: the network’s residual architecture
#     :param include_top: if true, includes classification layers
#     :param classes: number of classes to classify (include_top must be true)
#     :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)
#     :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)
#     Usage:
#         >>> import keras_resnet.models
#         >>> shape, classes = (224, 224, 3), 1000
#         >>> x = keras.layers.Input(shape)
#         >>> model = keras_resnet.models.ResNet34(x, classes=classes)
#         >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
#     """
#     def __init__(self, inputs, blocks=None, include_top=True, classes=1000, freeze_bn=False, *args, **kwargs):
#         if blocks is None:
#             blocks = [3, 4, 6, 3]

#         super(ResNet1D34, self).__init__(
#             inputs,
#             blocks,
#             block=keras_resnet.blocks.basic_1d,
#             include_top=include_top,
#             classes=classes,
#             freeze_bn=freeze_bn,
#             *args,
#             **kwargs
#         )


# class ResNet1D50(ResNet1D):
#     """
#     Constructs a `keras.models.Model` according to the ResNet50 specifications.
#     :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)
#     :param blocks: the network’s residual architecture
#     :param include_top: if true, includes classification layers
#     :param classes: number of classes to classify (include_top must be true)
#     :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)
#     :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)
#     Usage:
#         >>> import keras_resnet.models
#         >>> shape, classes = (224, 224, 3), 1000
#         >>> x = keras.layers.Input(shape)
#         >>> model = keras_resnet.models.ResNet50(x)
#         >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
#     """
#     def __init__(self, inputs, blocks=None, include_top=True, classes=1000, freeze_bn=False, *args, **kwargs):
#         if blocks is None:
#             blocks = [3, 4, 6, 3]

#         numerical_names = [False, False, False, False]

#         super(ResNet1D50, self).__init__(
#             inputs,
#             blocks,
#             numerical_names=numerical_names,
#             block=keras_resnet.blocks.bottleneck_1d,
#             include_top=include_top,
#             classes=classes,
#             freeze_bn=freeze_bn,
#             *args,
#             **kwargs
#         )


# class ResNet1D101(ResNet1D):
#     """
#     Constructs a `keras.models.Model` according to the ResNet101 specifications.
#     :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)
#     :param blocks: the network’s residual architecture
#     :param include_top: if true, includes classification layers
#     :param classes: number of classes to classify (include_top must be true)
#     :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)
#     :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)
#     Usage:
#         >>> import keras_resnet.models
#         >>> shape, classes = (224, 224, 3), 1000
#         >>> x = keras.layers.Input(shape)
#         >>> model = keras_resnet.models.ResNet101(x, classes=classes)
#         >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
#     """
#     def __init__(self, inputs, blocks=None, include_top=True, classes=1000, freeze_bn=False, *args, **kwargs):
#         if blocks is None:
#             blocks = [3, 4, 23, 3]

#         numerical_names = [False, True, True, False]

#         super(ResNet1D101, self).__init__(
#             inputs,
#             blocks,
#             numerical_names=numerical_names,
#             block=keras_resnet.blocks.bottleneck_1d,
#             include_top=include_top,
#             classes=classes,
#             freeze_bn=freeze_bn,
#             *args,
#             **kwargs
#         )


# class ResNet1D152(ResNet1D):
#     """
#     Constructs a `keras.models.Model` according to the ResNet152 specifications.
#     :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)
#     :param blocks: the network’s residual architecture
#     :param include_top: if true, includes classification layers
#     :param classes: number of classes to classify (include_top must be true)
#     :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)
#     :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)
#     Usage:
#         >>> import keras_resnet.models
#         >>> shape, classes = (224, 224, 3), 1000
#         >>> x = keras.layers.Input(shape)
#         >>> model = keras_resnet.models.ResNet152(x, classes=classes)
#         >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
#     """
#     def __init__(self, inputs, blocks=None, include_top=True, classes=1000, freeze_bn=False, *args, **kwargs):
#         if blocks is None:
#             blocks = [3, 8, 36, 3]

#         numerical_names = [False, True, True, False]

#         super(ResNet1D152, self).__init__(
#             inputs,
#             blocks,
#             numerical_names=numerical_names,
#             block=keras_resnet.blocks.bottleneck_1d,
#             include_top=include_top,
#             classes=classes,
#             freeze_bn=freeze_bn,
#             *args,
#             **kwargs
#         )


# class ResNet1D200(ResNet1D):
#     """
#     Constructs a `keras.models.Model` according to the ResNet200 specifications.
#     :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)
#     :param blocks: the network’s residual architecture
#     :param include_top: if true, includes classification layers
#     :param classes: number of classes to classify (include_top must be true)
#     :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)
#     :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)
#     Usage:
#         >>> import keras_resnet.models
#         >>> shape, classes = (224, 224, 3), 1000
#         >>> x = keras.layers.Input(shape)
#         >>> model = keras_resnet.models.ResNet200(x, classes=classes)
#         >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
#     """
#     def __init__(self, inputs, blocks=None, include_top=True, classes=1000, freeze_bn=False, *args, **kwargs):
#         if blocks is None:
#             blocks = [3, 24, 36, 3]

#         numerical_names = [False, True, True, False]

#         super(ResNet1D200, self).__init__(
#             inputs,
#             blocks,
#             numerical_names=numerical_names,
#             block=keras_resnet.blocks.bottleneck_1d,
#             include_top=include_top,
#             classes=classes,
#             freeze_bn=freeze_bn,
#             *args,
#             **kwargs
#         )