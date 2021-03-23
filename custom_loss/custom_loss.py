import tensorflow as tf
from tensorflow.keras import metrics

from vae.vae_architectures import VAEArchitecture

class CustomLoss:
    def __init__(self, losses=None):
        self.loss_calculators = {}
        self.loss_trackers = {"total_loss": metrics.Mean(name="total_loss")}

        if losses is not None:
            for loss_name, loss_calculator in losses.items():
                self.add_loss(loss_name, loss_calculator)

    def add_loss(self, loss_name, loss_calculator):
        if loss_name not in self.loss_calculators.keys():
            self.loss_calculators[loss_name] = loss_calculator
            self.loss_trackers[loss_name] = metrics.Mean(name=loss_name)

    def calculate_loss(self, original, architecture: VAEArchitecture):

        z_mean, z_log_var, z = (architecture.get_encoder()(original))
        reconstructed = (architecture.get_decoder())(z)

        total = tf.constant(0.0)

        for loss_name, loss_calculator in self.loss_calculators.items():
            loss = loss_calculator.calculate_loss(original=original,
                    reconstruction=reconstructed, z_mean=z_mean, z_log_var=z_log_var
            )

            self.loss_trackers[loss_name].update_state(loss)
            total = tf.add(total, loss)

        self.loss_trackers["total_loss"].update_state(total)

        return total

    def get_metrics(self):

        # Sorted ensures order is the same
        metric_names = sorted(list(self.loss_trackers.keys()))

        return [self.loss_trackers[tracker_name] for tracker_name in metric_names]

    def get_loss_trackers(self):
        results = {}

        for loss_name, loss_tracker in self.loss_trackers.items():
            results[loss_name] = loss_tracker.result()

        return results




