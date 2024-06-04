
import torch
import torch.nn as nn
import torchvision.models as tvmodels
from torchvision.models import AlexNet_Weights
import timm

__all__ = ["mobilenet_v3_small", "vgg16", "alexnet", "vit_base_patch16_224"]

class TorchVisionModel(nn.Module):
    def __init__(self, name, num_classes, loss, weights=None, pretrained=False, **kwargs):
        super().__init__()
        self.loss = loss
        self.name = name
        kwargs.pop('use_gpu', None)

        if 'vit' in name:
            # Initialize ViT with specified parameters
            self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=num_classes, **kwargs)
            # Use num_features to get the number of features before the classifier layer
            self.feature_dim = self.backbone.num_features
        else:
            # Initialize traditional models
            if name == 'mobilenet_v3_small':
                self.backbone = tvmodels.mobilenet_v3_small(weights=weights, pretrained=pretrained, **kwargs)
            elif name == 'vgg16':
                self.backbone = tvmodels.vgg16(weights=weights, pretrained=pretrained, **kwargs)
            elif name == 'alexnet':
                self.backbone = tvmodels.alexnet(weights=weights, pretrained=pretrained, **kwargs)
            else:
                raise ValueError(f"Unsupported model: {name}")

            if name == 'alexnet':
                self.feature_dim = 256 * 6 * 6  # AlexNet specific feature dimensions
            else:
                # Get the number of output features from the final FC layer (if applicable)
                self.feature_dim = getattr(self.backbone, 'classifier')[-1].out_features

            # Replace the classifier for traditional models
            self.backbone.classifier[-1] = nn.Linear(self.feature_dim, num_classes)

        # The classifier is a single linear layer for both ViTs and traditional models
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        assert x.dim() == 4 and x.size(1) == 3, "Input tensor must be [batch, 3, height, width]"

        if 'vit' in self.name:
            # For Vision Transformers, use the forward pass provided by timm which includes feature extraction
            features = self.backbone.forward_features(x)
            # Use the class token (position 0) for classification
            features = features[:, 0]
        else:
            # For other models, use the features method and flatten the output
            features = self.backbone.features(x)
            features = torch.flatten(features, 1)

        # Classifier is applied in both cases
        y = self.classifier(features)
        return self.handle_loss(y, features)

    def handle_loss(self, y, x):
        if not self.training:
            return y

        if self.loss == {"xent"}:
            return y
        elif self.loss == {"xent", "htri"}:
            return y, x
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

# Define any models supported by torchvision below
# https://pytorch.org/vision/0.11/models.html

def mobilenet_v3_small(pretrained=False, **kwargs):
    weights = tvmodels.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    return TorchVisionModel('mobilenet_v3_small', weights=weights, pretrained=pretrained, **kwargs)

def vgg16(pretrained=False, **kwargs):
    weights = tvmodels.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    return TorchVisionModel('vgg16', weights=weights, pretrained=pretrained, **kwargs)

def alexnet(pretrained=False, **kwargs):

    weights = AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
    return TorchVisionModel('alexnet', weights=weights, pretrained=pretrained, **kwargs)

def vit_base_patch16_224(pretrained=False, **kwargs):
    return TorchVisionModel('vit_base_patch16_224', pretrained=pretrained, **kwargs)