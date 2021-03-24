import tensorflow as tf
from tensorflow.keras import metrics

from vae.vae_architectures import VAEArchitecture


class CustomLoss:
    def __init__(self, losses=None):
        self.loss_calculators = {}
        self.loss_trackers = {"total_loss": metrics.Mean(name="total_loss")}

        self.cached_metrics = None

        if losses is not None:
            for loss_name, loss_calculator in losses.items():
                self.add_loss(loss_name, loss_calculator)

    def add_loss(self, loss_name, loss_calculator):
        if loss_name not in self.loss_calculators.keys():
            self.loss_calculators[loss_name] = loss_calculator
            self.loss_trackers[loss_name] = metrics.Mean(name=loss_name)
            self.cached_metrics = None

    def calculate_loss(self, original, architecture: VAEArchitecture):

        enc = architecture.get_encoder()
        dec = architecture.get_decoder()

        z_mean, z_log_var, z = enc(original)
        reconstructed = dec(z)

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

        if self.cached_metrics is None:
            metric_names = sorted(list(self.loss_trackers.keys()))
            self.cached_metrics = [self.loss_trackers[tracker_name] for tracker_name in metric_names]

        return self.cached_metrics

    def get_loss_trackers(self):
        results = {}

        for loss_name, loss_tracker in self.loss_trackers.items():
            results[loss_name] = loss_tracker.result()

        return results




