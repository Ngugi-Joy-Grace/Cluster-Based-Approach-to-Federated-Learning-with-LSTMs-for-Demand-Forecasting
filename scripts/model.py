import tensorflow as tf


def build_lstm_model(input_shape):
    """
    Create and compile an LSTM model for demand forecasting.

    Args:
        input_shape (tuple): (timesteps, features)

    Returns:
        tf.keras.Model: Compiled LSTM model.
    """
    # Input layer
    inputs = tf.keras.Input(shape=input_shape)

    # First LSTM layer (return_sequences=True for stacked LSTM)
    x = tf.keras.layers.LSTM(64, activation='relu', return_sequences=True)(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)

    # Second LSTM layer
    x = tf.keras.layers.LSTM(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # Output layer: single value (Sales)
    outputs = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    return model
