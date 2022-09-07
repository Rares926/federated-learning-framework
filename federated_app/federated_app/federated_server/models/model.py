import tensorflow as tf

IMG_SIZE = 224
NUM_CLASSES = 11
NUM_FEATURES = 7 * 7 * 1280
BATCH_SIZE = 32

head_MLP = tf.keras.Sequential([tf.keras.layers.Dense(128, activation='relu', name='dense_1', input_shape=([NUM_FEATURES])),
                                tf.keras.layers.Dense(NUM_CLASSES, name='dense_2')])

head_CNN = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(6, 5, activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(16, 5, activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=120, activation="relu"),
        tf.keras.layers.Dense(units=84, activation="relu"),
        tf.keras.layers.Dense(units=NUM_CLASSES, activation="softmax"),
    ]
)
