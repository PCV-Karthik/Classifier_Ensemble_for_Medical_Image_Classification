from tensorflow.keras import Sequential
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, concatenate, Dropout, \
    BatchNormalization, Input, Add, Activation, Flatten, AveragePooling2D, ZeroPadding2D, MaxPool2D


# Alex Net
def alexnet(input_shape, num_classes):
    model = Sequential([
        Input(input_shape),
        Conv2D(96, kernel_size=11, strides=4, activation='relu'),
        MaxPooling2D(pool_size=3, strides=2),
        Conv2D(256, kernel_size=5, padding='same', activation='relu'),
        MaxPooling2D(pool_size=3, strides=2),
        Conv2D(384, kernel_size=3, padding='same', activation='relu'),
        Conv2D(384, kernel_size=3, padding='same', activation='relu'),
        Conv2D(256, kernel_size=3, padding='same', activation='relu'),
        MaxPooling2D(pool_size=3, strides=2),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    return model


# Google Net
def inception_module(x, filters):
    branch1x1 = Conv2D(filters=filters[0], kernel_size=(1, 1), padding='same', activation='relu')(x)

    branch3x3 = Conv2D(filters=filters[1], kernel_size=(1, 1), padding='same', activation='relu')(x)
    branch3x3 = Conv2D(filters=filters[2], kernel_size=(3, 3), padding='same', activation='relu')(branch3x3)

    branch5x5 = Conv2D(filters=filters[3], kernel_size=(1, 1), padding='same', activation='relu')(x)
    branch5x5 = Conv2D(filters=filters[4], kernel_size=(5, 5), padding='same', activation='relu')(branch5x5)

    branch_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = Conv2D(filters=filters[5], kernel_size=(1, 1), padding='same', activation='relu')(branch_pool)

    output = concatenate([branch1x1, branch3x3, branch5x5, branch_pool], axis=-1)
    return output


def googlenet(input_shape, num_classes):
    input_layer = Input(input_shape)

    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(filters=192, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = inception_module(x, filters=[64, 96, 128, 16, 32, 32])
    x = inception_module(x, filters=[128, 128, 192, 32, 96, 64])
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = inception_module(x, filters=[192, 96, 208, 16, 48, 64])

    x = Flatten()(x)
    x = Dense(units=1024, activation='relu')(x)
    output_layer = Dense(units=num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return model


# Resnet
def convolutional_block(x, filter_size):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = Conv2D(filter_size, (3, 3), padding='same', strides=(2, 2))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    # Layer 2
    x = Conv2D(filter_size, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    # Processing Residue with conv(1,1)
    x_skip = Conv2D(filter_size, (1, 1), strides=(2, 2))(x_skip)
    # Add Residue
    x = Add()([x, x_skip])
    x = Activation('relu')(x)
    return x


def identity_block(x, filter_size):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = Conv2D(filter_size, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    # Layer 2
    x = Conv2D(filter_size, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    # Add Residue
    x = Add()([x, x_skip])
    x = Activation('relu')(x)
    return x


def resnet(shape, classes):
    # Step 1 (Setup Input Layer)
    x_input = Input(shape)
    x = ZeroPadding2D((3, 3))(x_input)
    # Step 2 (Initial Conv layer along with maxPool)
    x = Conv2D(64, kernel_size=7, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    # Define size of sub-blocks and initial filter size
    block_layers = [3, 4, 6, 3]
    filter_size = 64
    # Step 3 Add the Resnet Blocks
    for i in range(4):
        if i == 0:
            # For sub-block 1 Residual/Convolutional block not needed
            for j in range(block_layers[i]):
                x = identity_block(x, filter_size)
        else:
            # One Residual/Convolutional Block followed by Identity blocks
            # The filter size will go on increasing by a factor of 2
            filter_size = filter_size * 2
            x = convolutional_block(x, filter_size)
            for j in range(block_layers[i] - 1):
                x = identity_block(x, filter_size)
    # Step 4 End Dense Network
    x = AveragePooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(classes, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=x_input, outputs=x, name="ResNet34")
    return model
