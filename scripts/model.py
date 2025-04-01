import tensorflow as tf


def build_lstm_model(input_shape,
                     lstm_units_1=64,
                     lstm_units_2=32,
                     dropout_rate=0.2,
                     learning_rate=1e-3):
    """
    Create and compile an LSTM model for demand forecasting.

    Args:
        input_shape (tuple): (timesteps, features)
        lstm_units_1 (int): Number of units in the first LSTM layer
        lstm_units_2 (int): Number of units in the second LSTM layer
        dropout_rate (float): Dropout rate
        learning_rate (float): Learning rate for Adam

    Returns:
        tf.keras.Model: Compiled LSTM model.
    """
    # Input layer
    inputs = tf.keras.Input(shape=input_shape)

    # First LSTM layer
    x = tf.keras.layers.LSTM(lstm_units_1,
                             activation='relu',
                             return_sequences=True)(inputs)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    # Second LSTM layer
    x = tf.keras.layers.LSTM(lstm_units_2,
                             activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    # Optional Dense "head"
    x = tf.keras.layers.Dense(lstm_units_2, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    # Output layer
    outputs = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )

    return model
