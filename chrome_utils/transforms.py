from torchvision import transforms

class ContrastiveTransform:
    """Returns two randomly transformed versions of an image."""

    def __init__(self, transform: transforms):
        self.base_transform = transform

    def __call__(self, image):
        query_image = self.base_transform(image)
        key_image = self.base_transform(image)
        return [query_image, key_image]
    
evalTransform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])