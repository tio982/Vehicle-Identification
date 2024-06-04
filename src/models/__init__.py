
from .resnet import (
    resnet18,
    resnet18_fc512,
    resnet34,
    resnet34_fc512,
    resnet50,
    resnet50_fc512,
)
from .tvmodels import mobilenet_v3_small, vgg16, alexnet, vit_base_patch16_224

# Here we add alexnet to the model factory
__model_factory = {
    # image classification models
    "resnet18": resnet18,
    "resnet18_fc512": resnet18_fc512,
    "resnet34": resnet34,
    "resnet34_fc512": resnet34_fc512,
    "resnet50": resnet50,
    "resnet50_fc512": resnet50_fc512,
    "mobilenet_v3_small": mobilenet_v3_small,
    "vgg16": vgg16,
    "alexnet": alexnet,  # Ensure alexnet is updated properly
    "vit_base_patch16_224": vit_base_patch16_224, 
}

def get_names():
    return list(__model_factory.keys())

def init_model(name, *args, **kwargs):
    if name not in get_names():
        raise KeyError(f"Unknown model: {name}")
    return __model_factory[name](*args, **kwargs)


