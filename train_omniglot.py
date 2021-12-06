import torchvision
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader, Dataset
from torchvision.datasets import Omniglot

from PIL import Image
from typing import Any, Callable, List, Optional, Tuple

import os
import argparse
import math
import tqdm

from models2 import MultiSetTransformer, PINE, MultiSetModel
from generators import ImageCooccurenceGenerator

from train_omniglot import *
transform=torchvision.transforms.ToTensor()
train,val,test=ModifiedOmniglotDataset.splits("./data",15,5,5,transform=transform)


'''
class ImageCooccurenceDataset(IterableDataset):
    def __init__(self, dataset, set_size):
        self.dataset = dataset
        self.images_by_class = self._split_classes(dataset)
        self.set_size = set_size


    def _split_classes(self, dataset):
        images_by_class = {}
        for image, label in dataset:
            if label not in images_by_class:
                images_by_class[label] = []
            images_by_class[label].append(image)
        return images_by_class

    def __iter__(self):
        n_sets = len(self.dataset) / self.set_size / 2
        indices= torch.randperm(len(self.dataset))
        for j in range(n_sets):
            mindex = j * self.set_size * 2
            X = [self.dataset[i] for i in indices[mindex:mindex+self.set_size]]
            Y = [self.dataset[i] for i in indices[mindex+self.set_size+1:mindex+2*self.set_size]]
            Xdata, Xlabels = zip(*X)
            Ydata, Ylabels = zip(*Y)
            target = len(set(Xlabels) & set(Ylabels))

            yield (Xdata, Ydata), target
'''

#from torchvision
def list_dir(root: str, prefix: bool = False) -> List[str]:
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

def list_files(root: str, suffix: str, prefix: bool = False) -> List[str]:
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
    def make_dataset(cls, root_dir, n_alphabets, img_dir="images_background", transform=None):
        target_folder = os.path.join(root_dir, cls.folder, img_dir)
        all_alphabets = list_dir(target_folder)
        perm = torch.randperm(len(all_alphabets))
        alphabets = [all_alphabets[i] for i in perm[:n_alphabets]]
        return cls(target_folder, alphabets, transform)

    @classmethod
    def splits(cls, root_dir, *n, img_dir="images_background", transform=None):
        target_folder = os.path.join(root_dir, cls.folder, img_dir)
        all_alphabets = list_dir(target_folder)
        assert sum(n) <= len(all_alphabets)
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

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image_name, character_class = self._flat_character_images[index]
        image_path = os.path.join(self.target_folder, self._characters[character_class], image_name)
        image = Image.open(image_path, mode='r').convert('L')

        if self.transform:
            image = self.transform(image)

        return image, character_class


class ConvLayer(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.net = nn.Sequential(
            nn.Conv2d(in_filters, out_filters, kernel_size, stride=stride),
            nn.BatchNorm2d(out_filters),
            nn.ReLU()
        )

    def size_transform(self, input_size):
        return int(math.floor((input_size - kernel_size)/self.stride))
    
    def forward(self, inputs):
        return self.net(inputs)

class ConvBlock(nn.Module):
    def __init__(self, in_filters, out_filters, n_conv=2, pool='max'):
        super().__init__()
        self.n_conv = n_conv
        self.in_filters = in_filters
        self.out_filters = out_filters
        layers = [ConvLayer(in_filters, out_filters, 3)]
        for i in range(n_conv-1):
            layers.append(ConvLayer(out_filters, out_filters, 3))

        if pool == 'max':
            layers.append(nn.MaxPool2d(2,2))
        elif pool == 'avg':
            layers.append(nn.AvgPool2d(2,2))
        else:
            pool = 'none'
        self.pool = pool
        self.net = nn.Sequential(*layers)

    def size_transform(self, input_size):
        conv_size = input_size - 2 * self.n_conv
        pool_size = conv_size if self.pool == 'none' else int(math.floor(conv_size/2))
        return pool_size
    
    def forward(self, inputs):
        return self.net(inputs)

class ConvEncoder(nn.Module):
    @classmethod
    def make_omniglot_model(cls, output_size):
        layers = [
            ConvBlock(1, 16, pool='max'),
            ConvBlock(16, 32, pool='max'),
            ConvBlock(32, 64, pool='max'),
            ConvBlock(64, 128, n_conv=3, pool='none')
        ]
        return cls(layers, 105, output_size)
    
    @classmethod
    def make_mnist_model(cls, output_size):
        layers = [
            ConvBlock(1, 32, n_conv=1, pool='max'),
            ConvBlock(32, 64, n_conv=1, pool='max'),
        ]
        return cls(layers, 28, output_size)

    def __init__(self, layers, img_size, output_size, avg_pool=False):
        super().__init__()
        self.output_size = output_size
        conv_out_size = self._get_output_size(layers, img_size)
        if avg_pool:
            self.conv = nn.Sequential(*layers, nn.AvgPool2d(conv_out_size))
            self.fc = nn.Linear(layers[-1].out_filters, output_size)
        else:
            self.conv = nn.Sequential(*layers)
            self.fc = nn.Linear(layers[-1].out_filters * conv_out_size * conv_out_size, output_size)

    def _get_output_size(self, layers, input_size):
        x = input_size
        for layer in layers:
            x = layer.size_transform(x)
        return x

    def forward(self, inputs):
        conv_out = self.conv(inputs)
        fc_out = self.fc(conv_out.view(*inputs.size()[:-3], -1))
        return fc_out

class MultiSetImageModel(nn.Module):
    def __init__(self, encoder, set_model):
        super().__init__()
        self.set_model = set_model
        self.encoder = encoder
    
    def forward(self, X, Y, **kwargs):
        ZX = self.encoder(X.view(-1, *X.size()[-3:]))
        ZY = self.encoder(Y.view(-1, *Y.size()[-3:]))
        ZX = ZX.view(*X.size()[:-3], ZX.size(-1))
        ZY = ZY.view(*Y.size()[:-3], ZY.size(-1))
        return self.set_model(ZX, ZY, **kwargs)


def load_omniglot(root_folder="./data"):
    train_dataset = torchvision.datasets.Omniglot(
        root=root_folder, download=True, transform=torchvision.transforms.ToTensor(), background=True
    )

    test_dataset = torchvision.datasets.Omniglot(
        root=root_folder, download=True, transform=torchvision.transforms.ToTensor(), background=False
    )

    return train_dataset, test_dataset

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

def poisson_loss(outputs, targets):
    return -1 * (targets * outputs - torch.exp(outputs)).mean()

def train(model, optimizer, train_generator, val_generator, test_generator, steps, poisson=False, batch_size=64, eval_every=500, save_every=2000, eval_steps=100, checkpoint_dir=None, data_kwargs={}):
    losses = []
    eval_accs = []
    initial_step=1
    if checkpoint_dir is not None:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        else:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            if os.path.exists(checkpoint_path):
                load_dict = torch.load(checkpoint_path)
                model, optimizer, initial_step, losses, eval_accs = load_dict['model'], load_dict['optimizer'], load_dict['step'], load_dict['losses'], load_dict['accs']
    
    loss_fct = nn.MSELoss() if not poisson else poisson_loss
    avg_loss=0
    for i in tqdm.tqdm(range(steps)):
        optimizer.zero_grad()

        (X,Y), target = train_generator(batch_size, **data_kwargs)

        out = model(X,Y)
        loss = loss_fct(out.squeeze(-1), target)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        avg_loss += loss.item()

        if i % eval_every == 0 and i > 0:
            acc = evaluate(model, val_generator, eval_steps, poisson, batch_size, data_kwargs)
            eval_accs.append(acc)
            print("Step: %d\tAccuracy:%f\tTraining Loss: %f" % (i, acc, avg_loss/eval_every))
            avg_loss=0

        if checkpoint_dir is not None and i % save_every == 0 and i > 0:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            torch.save({'model':model,'optimizer':optimizer, 'step': i, 'losses':losses, 'accs': eval_accs}, checkpoint_path)
    
    test_acc = evaluate(model, test_generator, eval_steps, poisson, batch_size, data_kwargs)
    
    return model, (losses, eval_accs, test_acc)

def evaluate(model, eval_generator, steps, poisson=False, batch_size=64, data_kwargs={}):
    n_correct = 0
    with torch.no_grad():
        for i in range(steps):
            (X,Y), target = eval_generator(batch_size, **data_kwargs)
            out = torch.exp(model(X,Y).squeeze(-1))
            if poisson:
                n_correct += torch.logical_or(torch.eq(out.ceil(), target.int()), torch.eq(out.ceil()-1, target.int())).sum().item()
            else:
                n_correct += torch.eq(out.round(), target.int()).sum().item()
    return n_correct / (batch_size * steps)


def pretrain(encoder, n_classes, train_dataset, val_dataset, epochs, lr, batch_size, device, val_split=0.1):
    model = nn.Sequential(encoder, nn.Linear(encoder.output_size, n_classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for i in range(epochs):
        loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        avg_loss = 0
        for batch, targets in loader:
            optimizer.zero_grad()

            out = model(batch.to(device))
            loss = criterion(out, targets.to(device))
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
        avg_loss /= len(train_dataset)/batch_size
        eval_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size)
        acc = 0
        for batch, targets in eval_loader:
            with torch.no_grad():
                out = model(batch.cuda())
                acc += out.argmax(dim=-1).eq(targets.cuda()).sum().item()
        acc /= len(val_dataset)
        print("Epoch: %d\tTraining Loss: %f\t Eval Acc: %f" % (i, avg_loss, acc))
        
    return encoder



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str)
    parser.add_argument('--model', type=str, default='csab', choices=['csab', 'rn', 'pine'])
    parser.add_argument('--checkpoint_dir', type=str, default="/checkpoint/kaselby")
    parser.add_argument('--checkpoint_name', type=str, default=None)
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--latent_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--set_size', type=int, nargs=2, default=[6,10])
    parser.add_argument('--basedir', type=str, default="final-runs")
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'omniglot'], default='mnist')
    parser.add_argument('--pretrain_epochs', type=int, default=0)
    parser.add_argument('--poisson', action='store_true')
    parser.add_argument('--val_split', type=float, default=0.1)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    run_dir = os.path.join(args.basedir, args.dataset, args.run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    device = torch.device("cuda:0")

    if args.dataset == "mnist":
        trainval_dataset, test_dataset = load_mnist(args.data_dir)
        conv_encoder = ConvEncoder.make_mnist_model(args.latent_size)
        n_classes=10
    else:
        trainval_dataset, test_dataset = load_omniglot(args.data_dir)
        conv_encoder = ConvEncoder.make_omniglot_model(args.latent_size)
        n_classes=1623
    n_val = int(len(trainval_dataset) * args.val_split)
    train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [len(trainval_dataset)-n_val, n_val])

    train_generator = ImageCooccurenceGenerator(train_dataset, device)
    val_generator = ImageCooccurenceGenerator(val_dataset, device)
    test_generator = ImageCooccurenceGenerator(test_dataset, device)

    if args.pretrain_epochs > 0:
        pretrain_lr = 1e-3
        pretrain_bs = 64
        print("Beginning Pretraining...")
        conv_encoder = pretrain(conv_encoder, n_classes, train_dataset, val_dataset, args.pretrain_epochs, pretrain_lr, pretrain_bs, device)        

    if args.model == 'csab':
        model_kwargs={
            'ln':True,
            'remove_diag':False,
            'num_blocks':args.num_blocks,
            'num_heads':args.num_heads,
            'dropout':args.dropout,
            'equi':False
        }
        set_model = MultiSetTransformer(args.latent_size, args.latent_size, args.hidden_size, 1, **model_kwargs)
    elif args.model == 'pine':
        set_model = PINE(args.latent_size, args.latent_size/4, 16, 2, args.hidden_size, 1)
    else:
        raise NotImplementedError("Model type not recognized.")
    model = MultiSetImageModel(conv_encoder, set_model).to(device)

    batch_size = args.batch_size
    steps = args.steps
    if torch.cuda.device_count() > 1:
        n_gpus = torch.cuda.device_count()
        print("Let's use", n_gpus, "GPUs!")
        model = nn.DataParallel(model)
        batch_size *= n_gpus
        steps = int(steps/n_gpus)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.checkpoint_name) if args.checkpoint_name is not None else None
    data_kwargs = {'set_size':args.set_size}
    model, (losses, accs, test_acc) = train(model, optimizer, train_generator, val_generator, test_generator, steps, batch_size, checkpoint_dir=checkpoint_dir, data_kwargs=data_kwargs)

    print("Test Accuracy:", test_acc)

    model_out = model._modules['module'] if torch.cuda.device_count() > 1 else model
    torch.save(model_out, os.path.join(run_dir,"model.pt"))  
    torch.save({'losses':losses, 'eval_accs': accs, 'test_acc': test_acc, 'args':args}, os.path.join(run_dir,"logs.pt"))  

