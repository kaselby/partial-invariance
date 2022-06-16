import torchvision
import torch
from torch.utils.data import IterableDataset, DataLoader, Dataset, Subset, ConcatDataset
from torchvision.datasets import Omniglot
from PIL import Image

import os

#from torchvision
def list_dir(root: str, prefix: bool = False):
    """List all directories at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = [p for p in os.listdir(root) if os.path.isdir(os.path.join(root, p))]
    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]
    return directories

def list_files(root: str, suffix: str, prefix: bool = False):
    """List all files ending with a suffix at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = [p for p in os.listdir(root) if os.path.isfile(os.path.join(root, p)) and p.endswith(suffix)]
    if prefix is True:
        files = [os.path.join(root, d) for d in files]
    return files

class ModifiedOmniglotDataset(Dataset):
    folder="omniglot-py"

    @classmethod
    def make_dataset(cls, root_dir, n_alphabets=-1, img_dir="images_background", transform=None):
        target_folder = os.path.join(root_dir, cls.folder, img_dir)
        all_alphabets = list_dir(target_folder)
        n_alphabets = n_alphabets if n_alphabets > 0 else len(all_alphabets)
        perm = torch.randperm(len(all_alphabets))
        alphabets = [all_alphabets[i] for i in perm[:n_alphabets]]
        return cls(target_folder, alphabets, transform)

    @classmethod
    def splits(cls, root_dir, *n, img_dir="images_background", transform=None):
        def validate_n(n, n_tot):
            n_out = []
            s = 0
            for i in range(len(n)):
                if n[i] > 0:
                    n_out.append(n[i])
                    s += n[i]
                else:
                    n_out.append(n_tot - s)
                    s = n_tot
            assert s <= n_tot
            return n_out

        target_folder = os.path.join(root_dir, cls.folder, img_dir)
        all_alphabets = list_dir(target_folder)
        n = validate_n(n, len(all_alphabets))
        perm = torch.randperm(len(all_alphabets))
        alphabet_splits = []
        i=0
        for ni in n:
            alphabet_splits.append([all_alphabets[i] for i in perm[i:i+ni]])
            i += ni
        
        return [cls(target_folder, alphabet, transform) for alphabet in alphabet_splits]

    def __init__(self, target_folder, alphabets, transform):
        super().__init__()
        self.target_folder = target_folder
        self.transform = transform
        self._alphabets = alphabets
        self._characters: List[str] = sum([[os.path.join(a, c) for c in list_dir(os.path.join(self.target_folder, a))]
                                           for a in self._alphabets], [])
        self._character_images = [[(image, idx) for image in list_files(os.path.join(self.target_folder, character), '.png')]
                                  for idx, character in enumerate(self._characters)]
        self._flat_character_images: List[Tuple[str, int]] = sum(self._character_images, [])

    def __len__(self) -> int:
        return len(self._flat_character_images)

    def _make_output(self, image_name, character_class):
        image_path = os.path.join(self.target_folder, self._characters[character_class], image_name)
        image = Image.open(image_path, mode='r').convert('L')

        if self.transform:
            image = self.transform(image)

        return image, character_class

    def get_character_image(self, character_index, img_index):
        image_name, character_class = self._character_images[character_index][img_index]
        return self._make_output(image_name, character_class)

    def get_image(self, index):
        image_name, character_class = self._flat_character_images[index]
        return self._make_output(image_name, character_class)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        return self.get_image(index)


class DatasetByClass():
    @staticmethod
    def _subsets_by_class(dataset, n_classes):
        indices = {i:[] for i in range(n_classes)}
        for i, (_, label) in enumerate(dataset):
            indices[label].append(i)
        return {k: Subset(dataset, v) for k,v in indices.items()}

    @classmethod
    def splits(cls, dataset, class_splits):
        n_classes = sum(class_splits)
        subsets = cls._subsets_by_class(dataset, n_classes)
        if len(class_splits) == 1:
            return cls(dataset, subsets)
        else:
            perm = torch.randperm(n_classes)
            j=0
            datasets=[]
            for split in class_splits:
                datasets.append(cls(dataset, {i:subsets[i] for i in perm[j:j+split]}))
                j += split
            return datasets
            
    def __init__(self, dataset, subsets_by_class):
        self.n_classes = len(subsets_by_class.keys())
        self.dataset = dataset
        self.subsets_by_class = subsets_by_class

    def get_item_by_class(self, cls_label, i):
        return self.subsets_by_class[cls_label][i]

    def get_subset_by_class(self, cls_labels):
        return ConcatDataset([self.subsets_by_class[i] for i in cls_labels])

    def __getitem__(self, i):
        return self.dataset[i]


def load_omniglot(root_folder="./data"):
    transforms = torchvision.transforms.ToTensor()
    train_dataset, = ModifiedOmniglotDataset.splits(root_folder, -1, transform=transforms, img_dir="images_background")
    val_dataset, test_dataset = ModifiedOmniglotDataset.splits(root_folder, 5, -1, transform=transforms, img_dir="images_evaluation")
    
    return train_dataset, val_dataset, test_dataset


def load_mnist(root_folder="./data"):
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root=root_folder, download=True, transform=transform, train=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root=root_folder, download=True, transform=transform, train=False
    )

    return train_dataset, test_dataset

def load_cifar(root_folder="./data"):
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5071, 0.4866, 0.4409), (0.1642, 0.1496, 0.1728))
    ])

    train_dataset = torchvision.datasets.CIFAR100(
        root=root_folder, download=True, transform=transform, train=True
    )

    test_dataset = torchvision.datasets.CIFAR100(
        root=root_folder, download=True, transform=transform, train=False
    )

    return train_dataset, test_dataset

class ImageCooccurenceGenerator():
    def __init__(self, dataset):
        self.dataset = dataset
        self.image_size = dataset[0][0].size()[1:]

    def _sample_batch(self, batch_size, x_samples, y_samples):
        #indices = torch.randperm(len(self.dataset))
        for j in range(batch_size):
            #mindex = j * (x_samples + y_samples)
            indices = torch.randperm(len(self.dataset))
            X_j = [self.dataset[i] for i in indices[:x_samples]]
            Y_j = [self.dataset[i] for i in indices[x_samples: x_samples + y_samples]]
            yield X_j, Y_j

    def _generate(self, batch_size, set_size=(50,75), **kwargs):
        n_samples = torch.randint(*set_size, (2,))
        X, Y, targets = [], [], []
        for (X_j, Y_j) in self._sample_batch(batch_size, n_samples[0].item(), n_samples[1].item(), **kwargs):
            Xdata, Xlabels = zip(*X_j)
            Ydata, Ylabels = zip(*Y_j)
            target = len(set(Xlabels) & set(Ylabels))
            X.append(torch.stack(Xdata, 0))
            Y.append(torch.stack(Ydata, 0))
            targets.append(target)
        return (torch.stack(X, 0), torch.stack(Y, 0)), torch.tensor(targets, dtype=torch.float)

    def __call__(self, *args, **kwargs):
        return self._generate(*args, **kwargs)

class CIFARCooccurenceGenerator(ImageCooccurenceGenerator):
    def _sample_batch(self, batch_size, x_samples, y_samples):
        batch_n_classes = max(x_samples, y_samples)
        for j in range(batch_size):
            classes = torch.randperm(self.dataset.n_classes)[:batch_n_classes]
            subset = self.dataset.get_subset_by_class(classes.tolist())
            indices = torch.randperm(len(subset))
            X_j = [subset[i] for i in indices[:x_samples]]
            Y_j = [subset[i] for i in indices[x_samples: x_samples + y_samples]]
            yield X_j, Y_j


class OmniglotCooccurenceGenerator(ImageCooccurenceGenerator):
    def _make_output(self, image_name, character_class):
        image_path = os.path.join(self.dataset.target_folder, self.dataset._characters[character_class], image_name)
        image = Image.open(image_path, mode='r').convert('L')

        if self.dataset.transform:
            image = self.dataset.transform(image)
        
        return image, character_class

    def _sample_batch(self, batch_size, x_samples, y_samples, n_chars=-1):
        n_chars = max(x_samples, y_samples)
        for j in range(batch_size):
            character_indices = [i for i in torch.randperm(len(self.dataset._characters))[:n_chars]]
            flat_character_images= sum([self.dataset._character_images[i] for i in character_indices], [])
            indices = torch.randperm(len(flat_character_images))
            X_j = [self._make_output(*flat_character_images[i]) for i in indices[:x_samples]]
            Y_j = [self._make_output(*flat_character_images[i]) for i in indices[x_samples: x_samples + y_samples]]
            yield (X_j, Y_j)