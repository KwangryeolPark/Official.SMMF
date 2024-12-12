from torchvision.models import resnet50

def get_model(args=None):
    return resnet50(num_classes=100)
