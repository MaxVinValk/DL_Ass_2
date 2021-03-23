from custom_loss.custom_loss import CustomLoss
from custom_loss.vgg_relu_layers import VGG_ReLu_Layer
from custom_loss.custom_loss_calculators import KLDCalculator, FPLCalculator, ReconLossCalculator

'''
Here we can define which total loss functions to use. Each loss function is a combination of one or more
loss calculator. All loss calculators must be added in init using the add_loss function.
'''


'''
The loss function as used in the paper
'''


class PaperLoss123(CustomLoss):
    def __init__(self, input_shape, batch_size):
        super().__init__()

        loss_layers = [VGG_ReLu_Layer.ONE, VGG_ReLu_Layer.TWO, VGG_ReLu_Layer.THREE]
        beta = [.5, .5, .5]

        self.add_loss("kl_loss", KLDCalculator(alpha=1.0))
        self.add_loss("fpl_loss", FPLCalculator(input_shape, batch_size, loss_layers, beta))


'''
A generalized form of the loss for 123 from the paper, with custom alpha & betas:
'''


class Loss123(CustomLoss):
    def __init__(self, input_shape, batch_size, alpha, beta):
        super().__init__()

        loss_layers = [VGG_ReLu_Layer.ONE, VGG_ReLu_Layer.TWO, VGG_ReLu_Layer.THREE]

        self.add_loss("kl_loss", KLDCalculator(alpha=alpha))
        self.add_loss("fpl_loss", FPLCalculator(input_shape, batch_size, loss_layers, beta))


'''
The standard VAE loss
'''


class ReconLoss(CustomLoss):
    def __init__(self):
        super().__init__()
        self.add_loss("kl_loss", KLDCalculator())
        self.add_loss("recon_loss", ReconLossCalculator())
