# EECS 553 Winter 2025
import torch
import torchvision.models as models
from dataset import DogDataset
from train import train


def load_pretrained(num_classes=5):
    """
    Load a ResNet-18 model from `torchvision.models` with pre-trained weights. Freeze all the parameters besides the
    final layer by setting the flag `requires_grad` for each parameter to False. Replace the final fully connected layer
    with another fully connected layer with `num_classes` many output units.
    Inputs:
        - num_classes: int
    Returns:
        - model: PyTorch model
    """
    # TODO (part f): load a pre-trained ResNet-18 model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)


    for param in model.parameters():
        param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    return model


if __name__ == '__main__':
    config = {
        'dataset_path': 'data/images/dogs',
        'batch_size': 4,
        'if_resize': False,             
        'ckpt_path': 'checkpoints/transfer',
        'plot_name': 'Transfer',
        'num_epoch': 5,
        'learning_rate': 1e-3,
        'momentum': 0.9,
    }
    dataset = DogDataset(config['batch_size'], config['dataset_path'],config['if_resize'])
    model = load_pretrained()
    train(config, dataset, model)