import torch
import torchvision
import torch.nn as nn

from train_omniglot import ConvEncoder, ConvBlock, ConvLayer, MultiSetImageModel
from md_generator import MetaDatasetGenerator
from meta_dataset.dataset_spec import Split
from models2 import MultiSetTransformer, PINE, NaiveMultiSetModel

import argparse
import os
import tqdm

def train_adv(discriminator, generator, d_opt, g_opt, dataset, steps, device, set_size=(10,15),batch_size=64, save_every=2000, checkpoint_dir=None, data_kwargs={}):
    d_losses = []
    g_losses = []
    step=0
    if checkpoint_dir is not None:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        else:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            if os.path.exists(checkpoint_path):
                load_dict = torch.load(checkpoint_path)
                discriminator, generator, d_opt, g_opt, step, d_losses, g_losses = load_dict['discriminator'], load_dict['generator'], load_dict['d_opt'], load_dict['g_opt'], load_dict['step'], load_dict['d_losses'], load_dict['g_losses']
    
    criterion = nn.BCEWithLogitsLoss()
    ones = torch.ones(batch_size).to(device)
    zeros = torch.zeros(batch_size).to(device)  #can these just always be the same tensors? does that break anything?
    while step < steps:
        for batch in dataset:
            d_opt.zero_grad()
            outputs = discriminator(batch.to(device))
            d_loss1 = criterion(outputs, ones)
            d_loss1.backward()

            nsamples = torch.randint(*set_size, (1,))
            noise = torch.randn(batch_size * n_samples, 1, 1).to(device)
            fake_batch = generator(noise).view(batch_size, n_samples, -1)

            outputs2 = discriminator(fake_batch.detach())
            d_loss2 = criterion(outputs2, zeros)
            d_loss2.backward()
            d_opt.step()

            g_opt.zero_grad()
            outputs3 = discriminator(fake_batch)
            g_loss = criterion(outputs3, ones)
            g_loss.backward()
            g_opt.step()

            d_losses.append((d_loss1+d_loss2).item()/2)
            g_losses.append(g_loss.item())

            step += 1
            if step % save_every == 0 and checkpoint_dir is not None:
                checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                torch.save({
                    'discriminator':discriminator, 
                    'generator':generator,
                    'd_opt':d_opt, 
                    'g_opt':g_opt, 
                    'step': i, 
                    'd_loss': d_losses,
                    'g_loss': g_losses
                    }, checkpoint_path)

    return discriminator, generator, d_losses, g_losses


def train_disc(model, optimizer, train_dataset, val_dataset, test_dataset, steps, batch_size=64, eval_every=500, save_every=2000, eval_steps=100, checkpoint_dir=None, data_kwargs={}):
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
                model, optimizer, initial_step, train_losses, eval_accs = load_dict['model'], load_dict['optimizer'], load_dict['step'], load_dict['losses'], load_dict['accs']
    
    avg_loss = 0
    loss_fct = nn.BCEWithLogitsLoss()
    for i in tqdm.tqdm(range(steps)):
        optimizer.zero_grad()

        (X,Y), target = train_dataset(batch_size, **data_kwargs)

        out = model(X,Y)
        loss = loss_fct(out.squeeze(-1), target)
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()
        train_losses.append(loss.item())

        if i % eval_every == 0 and i > 0:
            acc = eval_disc(model, val_dataset, eval_steps, batch_size, data_kwargs)
            eval_accs.append(acc)
            avg_loss /= eval_every
            print("Step: %d\tLoss: %f\tAccuracy: %f" % (i, avg_loss, acc))
            avg_loss = 0

        if i % save_every == 0 and i > 0 and checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            torch.save({'model':model,'optimizer':optimizer, 'step': i, 'losses':train_losses, 'accs': eval_accs}, checkpoint_path)
    
    test_acc = eval_disc(model, test_dataset, eval_steps, batch_size, data_kwargs)
    
    return model, (train_losses, accs, test_acc)

def eval_disc(model, dataset, steps, batch_size, data_kwargs):
    with torch.no_grad():
        n_correct = 0
        for i in tqdm.tqdm(range(steps)):
            (X,Y), target = dataset(batch_size, **data_kwargs)
            out = model(X,Y).squeeze(-1)
            n_correct += torch.eq((out > 0), target).sum().item()
    return n_correct / (batch_size * steps)



def train_gen(generator, discriminator, optimizer, train_dataset, steps, batch_size=64, save_every=2000, print_every=250, checkpoint_dir=None, data_kwargs={}):
    train_losses = []
    initial_step=1
    if checkpoint_dir is not None:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        else:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            if os.path.exists(checkpoint_path):
                load_dict = torch.load(checkpoint_path)
                generator, discriminator, optimizer, initial_step, train_losses = load_dict['generator'], load_dict['discriminator'], load_dict['optimizer'], load_dict['step'], load_dict['losses']

    avg_loss = 0
    labels = torch.ones(batch_size)
    loss_fct = nn.BCEWithLogitsLoss()
    for i in tqdm.tqdm(range(steps)):
        optimizer.zero_grad()

        X = train_dataset(batch_size, **data_kwargs)
        noise = torch.randn(*X.size()).to(X.device)
        Y = generator(noise)

        out = discriminator(X,Y)
        loss = loss_fct(out.squeeze(-1), labels)
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()
        train_losses.append(loss.item())

        if i % print_every == 0:
            avg_loss /= print_every
            print("Step: %d\tLoss: %f" % (i, avg_loss))
            avg_loss = 0

        if i % save_every == 0 and checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            torch.save({'generator':generator, 'discriminator': discriminator, 'optimizer':optimizer, 'step': i, 'losses':train_losses}, checkpoint_path)
    
    return generator, train_losses   


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str)
    parser.add_argument('--model', type=str, default='csab', choices=['csab', 'rn', 'pine'])
    parser.add_argument('--checkpoint_dir', type=str, default="/checkpoint/kaselby")
    parser.add_argument('--checkpoint_name', type=str, default=None)
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--latent_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--set_size', type=int, nargs=2, default=[10,15])
    parser.add_argument('--basedir', type=str, default="final-runs")
    parser.add_argument('--eval_every', type=int, default=500)
    parser.add_argument('--eval_steps', type=int, default=200)
    parser.add_argument('--weight_sharing', type=str, choices=['none', 'cross', 'sym'], default='none')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    run_dir = os.path.join(args.basedir, "meta-dataset", "discriminator", args.run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    device = torch.device("cuda")

    layers = [
        ConvLayer(3, 32, kernel_size=7, stride=2),
        ConvBlock(32, 32, n_conv=2, pool='max'),
        ConvBlock(32, 64, n_conv=2,pool='max'),
        ConvBlock(64, 128, n_conv=2, pool='max')
    ]
    encoder = ConvEncoder(layers, 84, args.latent_size)
    if args.model == 'csab':
        model_kwargs={
            'ln':True,
            'remove_diag':False,
            'num_blocks':args.num_blocks,
            'num_heads':args.num_heads,
            'dropout':args.dropout,
            'equi':False,
            'weight_sharing': args.weight_sharing
        }
        set_model = MultiSetTransformer(args.latent_size, args.latent_size, args.hidden_size, 1, **model_kwargs)
    elif args.model == 'naive':
        model_kwargs={
            'ln':True,
            'remove_diag':False,
            'num_blocks':args.num_blocks,
            'num_heads':args.num_heads,
            #'dropout':args.dropout,
            'equi':False,
            'weight_sharing': args.weight_sharing
        }
        set_model = NaiveMultiSetModel(args.latent_size, args.latent_size, args.hidden_size, 1, **model_kwargs)
    elif args.model == 'pine':
        set_model = PINE(args.latent_size, int(args.latent_size/4), 16, 2, args.hidden_size, 1)
    else:
        raise NotImplementedError
    discriminator = MultiSetImageModel(encoder, set_model).to(device)

    train_generator = MetaDatasetGenerator(split=Split.TRAIN, device=device)
    val_generator = MetaDatasetGenerator(split=Split.VALID, device=device)
    test_generator = MetaDatasetGenerator(split=Split.TEST, device=device)
    
    batch_size = args.batch_size
    steps = args.steps
    eval_every=args.eval_every
    eval_steps=args.eval_steps
    if torch.cuda.device_count() > 1:
        n_gpus = torch.cuda.device_count()
        print("Let's use", n_gpus, "GPUs!")
        discriminator = nn.DataParallel(discriminator)
        batch_size *= n_gpus
        steps = int(steps/n_gpus)
        eval_every = int(eval_every/n_gpus)
        eval_steps = int(eval_steps/n_gpus)

    print("Beginning Training...")

    data_kwargs = {'set_size':args.set_size}
    optimizer = torch.optim.Adam(discriminator.parameters(), args.lr)
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.checkpoint_name) if args.checkpoint_name is not None else None
    model, (losses, accs, test_acc) = train_disc(discriminator, optimizer, train_generator, val_generator, test_generator, steps, 
        batch_size=batch_size, checkpoint_dir=checkpoint_dir, data_kwargs=data_kwargs, eval_every=eval_every, eval_steps=eval_steps)

    print("Test Accuracy:", test_acc)

    model_out = discriminator._modules['module'] if torch.cuda.device_count() > 1 else discriminator
    torch.save(model_out, os.path.join(run_dir, "model.pt"))  
    torch.save({'losses':losses, 'eval_accs': accs, 'test_acc': test_acc, 'args':args}, os.path.join(run_dir,"logs.pt"))  





    
