from torchvision import transforms

class ContrastiveTransform:
    """Returns two randomly transformed versions of an image."""

    def __init__(self, transform: transforms):
        self.base_transform = transform

    def __call__(self, image):
        query_image = self.base_transform(image)
        key_image = self.base_transform(image)
        return [query_image, key_image]