import torch
import torchvision.transforms as transforms
from PIL import ImageEnhance

from .dataset_spec import Split

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

jitter_param = dict(Brightness=0.4, Contrast=0.4, Color=0.4)

TRAIN_TRANSFORMS = ['random_resized_crop', 'jitter', 'random_flip', 'to_tensor', 'normalize']
TEST_TRANSFORMS = ['resize', 'center_crop', 'to_tensor', 'normalize']

class ImageJitter(object):
    def __init__(self, transformdict):
        transformtypedict = dict(Brightness=ImageEnhance.Brightness,
                                 Contrast=ImageEnhance.Contrast,
                                 Sharpness=ImageEnhance.Sharpness,
                                 Color=ImageEnhance.Color)
        self.params = [(transformtypedict[k], transformdict[k]) for k in transformdict]

    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.params))

        for i, (transformer, alpha) in enumerate(self.params):
            r = alpha * (randtensor[i] * 2.0 - 1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')

        return out


def get_transforms(image_size, split, **kwargs):
    if split == Split["TRAIN"]:
        return train_transform(image_size, **kwargs)
    else:
        return test_transform(image_size, **kwargs)


def test_transform(image_size, transform_list=TEST_TRANSFORMS):
    #resize_size = int(image_size * 256 / 224)
    #assert resize_size == image_size * 256 // 224
    resize_size = image_size

    transf_dict = {'resize': transforms.Resize(resize_size),
                   'center_crop': transforms.CenterCrop(image_size),
                   'to_tensor': transforms.ToTensor(),
                   'normalize': normalize}
    augmentations = transform_list

    return transforms.Compose([transf_dict[key] for key in augmentations])


def train_transform(image_size, transform_list=TRAIN_TRANSFORMS):
    transf_dict = {'resize': transforms.Resize(image_size),
                   'center_crop': transforms.CenterCrop(image_size),
                   'random_resized_crop': transforms.RandomResizedCrop(image_size),
                   'jitter': ImageJitter(jitter_param),
                   'random_flip': transforms.RandomHorizontalFlip(),
                   'to_tensor': transforms.ToTensor(),
                   'normalize': normalize}
    augmentations = transform_list

    return transforms.Compose([transf_dict[key] for key in augmentations])
