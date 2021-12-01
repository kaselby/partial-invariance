import torchvision
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset

import os
import argparse
import math
import tqdm

from models2 import MultiSetTransformer, PINE, MultiSetModel
from generators import ImageCooccurenceGenerator

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
    def __init__(self, img_size, output_size):
        super().__init__()
        layers = [
            ConvBlock(1, 16, pool='max'),
            ConvBlock(16, 32, pool='max'),
            ConvBlock(32, 64, pool='max'),
            ConvBlock(64, 128, n_conv=3, pool='none')
        ]
        out_size = self._get_output_size(layers, img_size)
        self.conv = nn.Sequential(*layers, nn.AvgPool2d(out_size))
        self.fc = nn.Linear(128, output_size)

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


def load_datasets(root_folder="./data", device=torch.device('cpu')):
    train_dataset = torchvision.datasets.Omniglot(
        root=root_folder, download=True, transform=torchvision.transforms.ToTensor(), background=True
    )

    test_dataset = torchvision.datasets.Omniglot(
        root=root_folder, download=True, transform=torchvision.transforms.ToTensor(), background=False
    )

    return ImageCooccurenceGenerator(train_dataset, device), ImageCooccurenceGenerator(test_dataset, device)

def train(model, optimizer, train_dataset, test_dataset, steps, batch_size=64, eval_every=500, save_every=2000, eval_steps=100, checkpoint_dir=None, data_kwargs={}):
    train_losses = []
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
    
    loss_fct = nn.MSELoss()
    for i in tqdm.tqdm(range(steps)):
        optimizer.zero_grad()

        (X,Y), target = train_dataset(batch_size, **data_kwargs)

        out = model(X,Y)
        loss = loss_fct(out.squeeze(-1), target)
        loss.backward()
        optimizer.step()

        if i % eval_every == 0:
            acc = evaluate(model, train_dataset, eval_steps, batch_size, data_kwargs)
            eval_accs.append(acc)
            print("Step: %d\tAccuracy:%f" % (i, acc))

        if i % save_every == 0 and checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            torch.save({'model':model,'optimizer':optimizer, 'step': i, 'losses':losses, 'accs': eval_accs}, checkpoint_path)
    
    test_acc = evaluate(model, test_dataset, eval_steps, batch_size, data_kwargs)
    
    return model, (losses, accs, test_acc)

def evaluate(model, eval_dataset, steps, batch_size=64, data_kwargs={}):
    n_correct = 0
    with torch.no_grad():
        for i in range(steps):
            (X,Y), target = eval_dataset(batch_size, data_kwargs)
            out = model(X,Y).squeeze(-1)
            n_correct += (out == target).sum().item()
    
    return n_correct / (batch_size * steps)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str)
    parser.add_argument('--model', type=str, default='csab', choices=['csab', 'rn', 'pine'])
    parser.add_argument('--checkpoint_dir', type=str, default="/checkpoint/kaselby")
    parser.add_argument('--checkpoint_name', type=str, default=None)
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--latent_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--set_size', type=int, nargs=2, default=[6,10])
    parser.add_argument('--basedir', type=str, default="final-runs")
    parser.add_argument('--data_dir', type=str, default='./data')
    return parser.parse_args()

IMG_SIZE=105

if __name__ == '__main__':
    args = parse_args()

    run_dir = os.path.join(args.basedir, "omniglot", args.run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    device = torch.device("cuda:0")

    train_dataset, test_dataset = load_datasets(args.data_dir, device)

    conv_encoder = ConvEncoder(IMG_SIZE, args.latent_size)
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
    model, (losses, accs, test_acc) = train(model, optimizer, train_dataset, test_dataset, steps, batch_size, checkpoint_dir=checkpoint_dir, data_kwargs=data_kwargs)

    print("Test Accuracy:", test_acc)

    model_out = model._modules['module'] if torch.cuda.device_count() > 1 else model
    torch.save(model_out, os.path.join(run_dir,"model.pt"))  
    torch.save({'losses':losses, 'eval_accs': accs, 'test_acc': test_acc, 'args':args}, os.path.join(run_dir,"logs.pt"))  

