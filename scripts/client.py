import logging
import pickle
from pathlib import Path
import flwr as fl
import numpy as np
from model import build_lstm_model


class StoreClient(fl.client.NumPyClient):
    """
    Flower client implementation for federated learning on a store's data.

    Attributes:
        cluster_id (int): The cluster number.
        store_id (int): The store ID number.
        x_train (np.ndarray): Feature matrix for training set.
        y_train (np.ndarray): Labels for training set.
        x_val (np.ndarray): Feature matrix for validation set.
        y_val (np.ndarray): Labels for validation set.
        model (tf.keras.Model): Local LSTM model for training and evaluation.
    """

    def __init__(self, cluster_id: int, store_id: int, data_dir: Path):
        """
        Initialize the StoreClient by loading pickle data and building the model.

        Args:
            cluster_id (int): Cluster number.
            store_id (int): Store ID.
            data_dir (Path): Path to the federated data directory.
        """
        self.cluster_id = cluster_id
        self.store_id = store_id
        # Load local data from pickle
        self.x_train, self.y_train, self.x_val, self.y_val = self._load_data(data_dir)

        # Build the LSTM model
        # shape => (timesteps, num_features)
        timesteps = self.x_train.shape[1]
        num_features = self.x_train.shape[2]
        self.model = build_lstm_model((timesteps, num_features))

    def _load_data(self, data_dir: Path):
        """
        Loads training and validation data from the pickled file
        for this store and cluster.

        Args:
            data_dir (Path): Base path to ../data/federated_data

        Returns:
            x_train, y_train, x_val, y_val (np.ndarray)
        """
        cluster_folder = data_dir / f"cluster_{self.cluster_id}"
        data_file = cluster_folder / f"store_{self.store_id}.pkl"
        logging.info(f"Loading pickle data for store {self.store_id}, cluster {self.cluster_id}...")

        with open(data_file, "rb") as f:
            data = pickle.load(f)

        x_train, y_train = data["train"]
        x_val, y_val = data["val"]

        # Verify the shapes are correct
        expected_timesteps = 7  # 7 days of data
        if len(x_train.shape) != 3 or x_train.shape[1] != expected_timesteps:
            error_msg = f"Invalid shape for x_train: expected (*, {expected_timesteps}, features), got {x_train.shape}"
            logging.error(error_msg)
            raise ValueError(error_msg)

        logging.info(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
        logging.info(f"x_val shape: {x_val.shape}, y_val shape: {y_val.shape}")

        return x_train, y_train, x_val, y_val

    def get_parameters(self, config):
        """
        Return local model parameters to the server.
        """
        logging.info(f"[Store {self.store_id}] Sending model parameters to server.")
        return self.model.get_weights()

    def fit(self, parameters, config):
        """
        Train the local model for a few epochs with the given parameters.
        """
        self.model.set_weights(parameters)  # Set global model parameters

        logging.info(f"[Store {self.store_id}] Starting local training.")
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=10,
            batch_size=32,
            verbose=0
        )
        logging.info(f"[Store {self.store_id}] Local training complete.")

        # Return updated weights and the number of training samples
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        """
        Evaluate the local model on the validation set.
        """
        self.model.set_weights(parameters)
        loss, mae = self.model.evaluate(self.x_val, self.y_val, verbose=0)
        logging.info(f"[Store {self.store_id}] Evaluation - Loss: {loss:.4f}, MAE: {mae:.4f}")

        # Return (loss, number_of_validation_samples, metrics_dict)
        return loss, len(self.x_val), {"mae": float(mae)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    client = StoreClient(cluster_id=0, store_id=1, data_dir=Path("../data/federated_data"))
    logging.info("Testing single-client local training...")
    client.model.fit(client.x_train, client.y_train, epochs=1, batch_size=32)
    loss, mae = client.model.evaluate(client.x_val, client.y_val)
    logging.info(f"Single-client test complete - Loss: {loss:.4f}, MAE: {mae:.4f}")
