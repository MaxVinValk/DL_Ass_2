from enum import IntEnum


class VGG_ReLu_Layer(IntEnum):
    """Layers with the first ReLU for feature perceptual loss

    Args:
        Enum (int): Number, indicating the block
    """
    ONE = 1
    TWO = 4
    THREE = 7
    FOUR = 12
    FIVE = 17
