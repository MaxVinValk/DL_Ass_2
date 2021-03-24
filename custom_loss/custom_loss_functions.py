from custom_loss.custom_loss import CustomLoss
from custom_loss.vgg_relu_layers import VGG_ReLu_Layer
from custom_loss.custom_loss_calculators import KLDCalculator, FPLCalculator, ReconLossCalculator

'''
Here we can define which total loss functions to use. Each loss function is a combination of one or more
loss calculator. All loss calculators must be added in init using the add_loss function.
'''


'''
A generalized form of the loss for 123 from the paper, with custom alpha & betas:
'''


class PaperLoss(CustomLoss):
    def __init__(self, input_shape, batch_size, loss_layers, alpha, beta):
        super().__init__()

        self.loss_layers = loss_layers
        self.alpha = alpha
        self.beta = beta

        self.add_loss("kl_loss", KLDCalculator(alpha=alpha))
        self.add_loss("fpl_loss", FPLCalculator(input_shape, batch_size, loss_layers, beta))

    def __str__(self):
        return "FPL + KLD:\n" + \
                f"Layers:\t{self.loss_layers}\n" + \
                f"Beta:\t{self.beta}\n" + \
                f"Alpha:\t{self.alpha}\n"


'''
The loss function as used in the paper
'''


class PaperLoss123(PaperLoss):
    def __init__(self, input_shape, batch_size):
        loss_layers = [VGG_ReLu_Layer.ONE, VGG_ReLu_Layer.TWO, VGG_ReLu_Layer.THREE]
        alpha = 1.0
        beta = [.5, .5, .5]
        super().__init__(input_shape, batch_size, loss_layers, alpha, beta)


'''
The standard VAE loss
'''


class ReconLoss(CustomLoss):
    def __init__(self):
        super().__init__()
        self.add_loss("kl_loss", KLDCalculator())
        self.add_loss("recon_loss", ReconLossCalculator())

    def __str__(self):
        return "Recon + KLD"
