import torchvision
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader, Dataset, Subset, ConcatDataset
from torchvision.datasets import Omniglot

from PIL import Image
from typing import Any, Callable, List, Optional, Tuple

import os
import argparse
import math
import tqdm

from models2 import MultiSetTransformer, PINE, MultiSetModel, NaiveMultiSetModel, CrossOnlyModel, MultiRNModel
from generators import ImageCooccurenceGenerator, OmniglotCooccurenceGenerator, CIFARCooccurenceGenerator


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

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
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
        return int(math.floor((input_size - self.kernel_size)/self.stride))
    
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
            ConvLayer(1, 32, kernel_size=7, stride=2),
            ConvBlock(32, 32, n_conv=2, pool='max'),
            ConvBlock(32, 64, n_conv=2,pool='max'),
            ConvBlock(64, 128, n_conv=2, pool='max')
        ]
        return cls(layers, 105, output_size)
    
    @classmethod
    def make_mnist_model(cls, output_size):
        layers = [
            ConvBlock(1, 32, n_conv=1, pool='max'),
            ConvBlock(32, 64, n_conv=1, pool='max'),
        ]
        return cls(layers, 28, output_size)
    
    @classmethod
    def make_cifar_model(cls, output_size):
        layers = [
            ConvBlock(3, 32, n_conv=2, pool='max'),
            ConvBlock(32, 64, n_conv=2, pool='max'),
        ]
        return cls(layers, 32, output_size)
    
    @classmethod
    def make_coco_model(cls, output_size):
        layers = [
            ConvLayer(3, 32, kernel_size=7, stride=2),
            ConvBlock(32, 32, n_conv=2, pool='max'),
            ConvBlock(32, 64, n_conv=2, pool='max'),
            ConvBlock(64, 128, n_conv=2, pool='max'),
            ConvBlock(128, 256, n_conv=2, pool='max'),
        ]
        return cls(layers, 224, output_size)


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


def make_resnet_model(latent_size):
    model = torchvision.models.resnet34(pretrained=False)
    model.fc = nn.Linear(512, latent_size)
    return model


def load_omniglot(root_folder="./data"):
    '''
    train_dataset = torchvision.datasets.Omniglot(
        root=root_folder, download=True, transform=torchvision.transforms.ToTensor(), background=True
    )

    test_dataset = torchvision.datasets.Omniglot(
        root=root_folder, download=True, transform=torchvision.transforms.ToTensor(), background=False
    )
    '''
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


def poisson_loss(outputs, targets):
    return -1 * (targets * outputs - torch.exp(outputs)).mean()

def train(model, optimizer, train_generator, val_generator, test_generator, steps, scheduler=None, poisson=False, batch_size=64, eval_every=500, save_every=2000, eval_steps=200, test_steps=500, checkpoint_dir=None, data_kwargs={}):
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
                model, optimizer, scheduler, initial_step, losses, eval_accs = load_dict['model'], load_dict['optimizer'], load_dict['scheduler'], load_dict['step'], load_dict['losses'], load_dict['accs']
    
    loss_fct = nn.MSELoss() if not poisson else poisson_loss
    avg_loss=0
    for i in tqdm.tqdm(range(initial_step, steps)):
        optimizer.zero_grad()

        (X,Y), target = train_generator(batch_size, **data_kwargs)

        out = model(X,Y)
        loss = loss_fct(out.squeeze(-1), target)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

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
            torch.save({'model':model,'optimizer':optimizer, 'scheduler':scheduler, 'step': i, 'losses':losses, 'accs': eval_accs}, checkpoint_path)
    
    test_acc = evaluate(model, test_generator, test_steps, poisson, batch_size, data_kwargs)
    
    return model, (losses, eval_accs, test_acc)

def evaluate(model, eval_generator, steps, poisson=False, batch_size=64, data_kwargs={}):
    n_correct = 0
    with torch.no_grad():
        for i in range(steps):
            (X,Y), target = eval_generator(batch_size, **data_kwargs)
            out = model(X,Y).squeeze(-1)
            if poisson:
                out = torch.exp(out)
            n_correct += torch.logical_or(torch.eq(out.ceil(), target.int()), torch.eq(out.ceil()-1, target.int())).sum().item()
    return n_correct / (batch_size * steps)


def pretrain(encoder, n_classes, train_dataset, val_dataset, steps, lr, batch_size, device, val_split=0.1, eval_every=300):
    model = nn.Sequential(encoder, nn.Linear(encoder.output_size, n_classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    step = 0
    avg_loss = 0
    while step < steps:
        loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        for batch, targets in loader:
            optimizer.zero_grad()

            out = model(batch.to(device))
            loss = criterion(out, targets.to(device))
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

            step += 1

            if step % eval_every == 0 and step > 0:
                avg_loss /= eval_every
                eval_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size)
                acc = 0
                for batch, targets in eval_loader:
                    with torch.no_grad():
                        out = model(batch.cuda())
                        acc += out.argmax(dim=-1).eq(targets.cuda()).sum().item()
                acc /= len(val_dataset)
                print("Step: %d\tTraining Loss: %f\t Eval Acc: %f" % (step, avg_loss, acc))
                avg_loss = 0

            if step >= steps:
                break
        
    return encoder



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str)
    parser.add_argument('--model', type=str, default='csab', choices=['csab', 'rn', 'pine', 'naive', 'cross-only'])
    parser.add_argument('--checkpoint_dir', type=str, default="/checkpoint/kaselby")
    parser.add_argument('--checkpoint_name', type=str, default=None)
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--steps', type=int, default=16000)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--latent_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--set_size', type=int, nargs=2, default=[6,10])
    parser.add_argument('--basedir', type=str, default="final-runs")
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar100', 'omniglot'], default='mnist')
    parser.add_argument('--pretrain_steps', type=int, default=0)
    parser.add_argument('--poisson', action='store_true')
    parser.add_argument('--weight_sharing', type=str, choices=['none', 'cross', 'sym'], default='none')
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--eval_every', type=int, default=500)
    parser.add_argument('--eval_steps', type=int, default=200)
    parser.add_argument('--merge_type', type=str, default='concat', choices=['concat', 'sum', 'lambda'])
    parser.add_argument('--warmup_steps', type=int, default=1000)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    run_dir = os.path.join(args.basedir, args.dataset, args.run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    device = torch.device("cuda:0")
    data_kwargs = {'set_size':args.set_size}

    if args.dataset == "mnist":
        trainval_dataset, test_dataset = load_mnist(args.data_dir)
        n_val = int(len(trainval_dataset) * args.val_split)
        train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [len(trainval_dataset)-n_val, n_val])
        conv_encoder = ConvEncoder.make_mnist_model(args.latent_size)
        n_classes=10
        generator_cls = ImageCooccurenceGenerator
        pretrain_val = val_dataset
    elif args.dataset == "cifar100":
        trainval_dataset, test_dataset = load_cifar(args.data_dir)
        n_val = int(len(trainval_dataset) * args.val_split)
        train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [len(trainval_dataset)-n_val, n_val])
        conv_encoder = make_resnet_model(args.latent_size)
        #conv_encoder = ConvEncoder.make_cifar_model(args.latent_size)
        n_classes=100
        generator_cls = CIFARCooccurenceGenerator
        pretrain_val = val_dataset
    else:
        train_dataset, val_dataset, test_dataset = load_omniglot(args.data_dir)
        conv_encoder = ConvEncoder.make_omniglot_model(args.latent_size)
        n_classes=len(train_dataset._characters)
        generator_cls = OmniglotCooccurenceGenerator
        pretrain_val = train_dataset
        data_kwargs['n_chars'] = 50

    if args.pretrain_steps > 0:
        pretrain_lr = 3e-4
        pretrain_bs = 64
        print("Beginning Pretraining...")
        conv_encoder = pretrain(conv_encoder, n_classes, train_dataset, pretrain_val, args.pretrain_steps, pretrain_lr, pretrain_bs, device)        

    if args.dataset == 'cifar100':
        train_dataset = DatasetByClass.splits(train_dataset, (100,))
        val_dataset = DatasetByClass.splits(val_dataset, (100,))
        test_dataset = DatasetByClass.splits(test_dataset, (100,))

    train_generator = generator_cls(train_dataset, device)
    val_generator = generator_cls(val_dataset, device)
    test_generator = generator_cls(test_dataset, device)

    if args.model == 'csab':
        model_kwargs={
            'ln':True,
            'remove_diag':False,
            'num_blocks':args.num_blocks,
            'num_heads':args.num_heads,
            'dropout':args.dropout,
            'equi':False,
            'weight_sharing': args.weight_sharing,
            'merge': args.merge_type
        }
        set_model = MultiSetTransformer(args.latent_size, args.latent_size, args.hidden_size, 1, **model_kwargs)
    elif args.model == 'cross-only':
        model_kwargs={
            'ln':True,
            'num_blocks':args.num_blocks,
            'num_heads':args.num_heads,
            'dropout':args.dropout,
            'equi':False,
            'weight_sharing': args.weight_sharing
        }
        set_model = CrossOnlyModel(args.latent_size, args.latent_size, args.hidden_size, 1, **model_kwargs)
    elif args.model == 'naive':
        model_kwargs={
            'ln':True,
            'remove_diag':False,
            'num_blocks':args.num_blocks,
            'num_heads':args.num_heads,
            'dropout':args.dropout,
            'equi':False,
            'weight_sharing': args.weight_sharing
        }
        set_model = NaiveMultiSetModel(args.latent_size, args.latent_size, args.hidden_size, 1, **model_kwargs)
    elif args.model == 'pine':
        set_model = PINE(args.latent_size, int(args.latent_size/4), 16, 2, args.hidden_size, 1)
    elif args.model == 'rn':
        model_kwargs={
            'ln':True,
            'remove_diag':False,
            'num_blocks':args.num_blocks,
            'dropout':args.dropout,
            'equi':False,
            'weight_sharing': args.weight_sharing,
            'pool1': 'max',
            'pool2': 'max'
        }
        set_model = MultiRNModel(args.latent_size, args.latent_size, args.hidden_size, 1, **model_kwargs)
    else:
        raise NotImplementedError("Model type not recognized.")
    model = MultiSetImageModel(conv_encoder, set_model).to(device)

    batch_size = args.batch_size
    steps = args.steps
    eval_every=args.eval_every
    eval_steps=args.eval_steps
    if torch.cuda.device_count() > 1:
        n_gpus = torch.cuda.device_count()
        print("Let's use", n_gpus, "GPUs!")
        model = nn.DataParallel(model)
        batch_size *= n_gpus
        steps = int(steps/n_gpus)
        eval_every = int(eval_every/n_gpus)
        eval_steps = int(eval_steps/n_gpus)

    print("Beginning Training...")

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-8, total_iters=args.warmup_steps) if args.warmup_steps > 0 else None
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.checkpoint_name) if args.checkpoint_name is not None else None
    model, (losses, accs, test_acc) = train(model, optimizer, train_generator, val_generator, test_generator, steps, 
        scheduler=scheduler, batch_size=batch_size, poisson=args.poisson, checkpoint_dir=checkpoint_dir, data_kwargs=data_kwargs, eval_every=eval_every, eval_steps=eval_steps)

    print("Test Accuracy:", test_acc)

    model_out = model._modules['module'] if torch.cuda.device_count() > 1 else model
    torch.save(model_out, os.path.join(run_dir,"model.pt"))  
    torch.save({'losses':losses, 'eval_accs': accs, 'test_acc': test_acc, 'args':args}, os.path.join(run_dir,"logs.pt"))  

