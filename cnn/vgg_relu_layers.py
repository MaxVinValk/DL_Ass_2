from enum import Enum


class VGG_ReLu_Layer(Enum):
    """Layers with the first ReLU for feature perceptual loss

    Args:
        Enum (int): Number, indicateing the block
    """
    ONE = 1
    TWO = 4
    THREE = 7
    FOUR = 12
    FIVE = 17
