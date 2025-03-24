import tensorflow as tf

def lstm_model_1(input_shape):
    """
    Create and compile an LSTM model for time-series forecasting.

    Args:
        input_shape (tuple): Shape of input data (timesteps, features).

    Returns:
        tf.keras.Model: Compiled LSTM model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, activation='relu', return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)  # Predicting single numeric value (Sales)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
