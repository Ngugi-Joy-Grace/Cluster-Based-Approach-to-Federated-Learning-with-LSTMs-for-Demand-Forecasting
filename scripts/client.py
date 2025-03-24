import flwr as fl
from model import lstm_model_1

class StoreClient(fl.client.NumPyClient):
    """
    Flower client implementation for federated learning.

    Attributes:
        model: Local LSTM model.
        x_train, y_train: Training dataset.
        x_val, y_val: Validation dataset.
    """

    def __init__(self, train_data, val_data):
        self.x_train, self.y_train = train_data
        self.x_val, self.y_val = val_data
        self.model = lstm_model_1((self.x_train.shape[1], self.x_train.shape[2]))

    def get_parameters(self, config):
        """Returns current model parameters to the server."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Trains local model with parameters from the server."""
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=15, batch_size=32, verbose=0)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        """Evaluates local model performance."""
        self.model.set_weights(parameters)
        loss, mae = self.model.evaluate(self.x_val, self.y_val, verbose=0)
        return loss, len(self.x_val), {"mae": mae}
