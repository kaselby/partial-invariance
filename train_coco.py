import torchvision
from torchvision.datasets import CocoCaptions
import torchvision.transforms as T
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, Dataset

from transformers import BertModel, BertTokenizer

import os
import argparse
import math
import tqdm
import json
import fasttext

from models2 import MultiSetTransformer, PINE, MultiSetModel, BertEncoderWrapper, ImageEncoderWrapper
from generators import CaptionGenerator, bert_tokenize_batch



#def fasttext_encoder_preproc():



def load_caption_data(imgdir, anndir):
    transforms = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    
    train_dataset = CocoCaptions(root=os.path.join(imgdir, "train2014"), annFile=os.path.join(anndir, "captions_train2014.json"), transform=transforms)
    val_dataset = CocoCaptions(root=os.path.join(imgdir, "val2014"), annFile=os.path.join(anndir, "captions_val2014.json"), transform=transforms)

    return train_dataset, val_dataset


def make_model(set_model, text_model='bert', embed_dim=300):
    if text_model == 'bert':
        model = BertModel.from_pretrained("bert-base-uncased")
        text_encoder = BertEncoderWrapper(model)
    else:
        text_encoder = EmbeddingEncoderWrapper(embed_dim)

    vgg = torchvision.models.vgg16(pretrained=True)
    vgg.classifier = nn.Sequential(*list(vgg.classifier.children())[:-3])
    img_encoder = ImageEncoderWrapper(vgg, 4096)

    for param in img_encoder.parameters():
        param.requires_grad = False
    for param in text_encoder.parameters():
        param.requires_grad = False
    
    #set_model = MultiSetTransformer(*args, **kwargs)
    return MultiSetModel(set_model, img_encoder, text_encoder)



def train(model, optimizer, train_dataset, test_dataset, steps, batch_size=64, eval_every=500, save_every=2000, eval_steps=100, checkpoint_dir=None, data_kwargs={}):
    train_losses = []
    eval_accs = []
    initial_step=0
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
    for i in tqdm.tqdm(range(initial_step, steps)):
        optimizer.zero_grad()

        (X,Y), target = train_dataset(batch_size, **data_kwargs)

        out = model(X,Y)
        loss = loss_fct(out.squeeze(-1), target)
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()
        train_losses.append(loss.item())

        if i % eval_every == 0 and i > 0:
            acc = evaluate(model, train_dataset, eval_steps, batch_size, data_kwargs)
            eval_accs.append(acc)
            avg_loss /= eval_every
            print("Step: %d\tLoss: %f\tAccuracy: %f" % (i, avg_loss, acc))
            avg_loss = 0

        if i % save_every == 0 and i > 0 and checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            torch.save({'model':model,'optimizer':optimizer, 'step': i, 'losses':train_losses, 'accs': eval_accs}, checkpoint_path)
    
    test_acc = evaluate(model, test_dataset, eval_steps, batch_size, data_kwargs)
    
    return model, (train_losses, eval_accs, test_acc)

def evaluate(model, eval_dataset, steps, batch_size=64, data_kwargs={}):
    n_correct = 0
    with torch.no_grad():
        for i in range(steps):
            (X,Y), target = eval_dataset(batch_size, **data_kwargs)
            out = model(X,Y).squeeze(-1)
            n_correct += torch.eq((out > 0), target).sum().item()
    
    return n_correct / (batch_size * steps)




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
    parser.add_argument('--data_dir', type=str, default='./coco')
    parser.add_argument('--eval_every', type=int, default=500)
    parser.add_argument('--eval_steps', type=int, default=200)
    parser.add_argument('--text_model', type=str, choices=['bert', 'ft'], default='bert')
    parser.add_argument('--embed_path', type=str, default="wiki-news-300d-1M.vec")
    parser.add_argument('--embed_dim', type=int, default=300)
    return parser.parse_args()

#IMG_SIZE=105

if __name__ == '__main__':
    args = parse_args()

    run_dir = os.path.join(args.basedir, "coco", args.run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    device = torch.device("cuda:0")

    if args.text_model == 'bert':
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        tokenize_fct = bert_tokenize_batch
        tokenize_args = (tokenizer,)
    elif args.text_model == 'ft':
        ft = fasttext.load_model(args.embed_path)
        tokenize_fct = fasttext_tokenize_batch
        tokenize_args = (ft,)

    train_dataset, test_dataset = load_caption_data(os.path.join(args.data_dir, "images"), os.path.join(args.data_dir, "annotations"))
    train_generator = CaptionGenerator(train_dataset, tokenize_fct, tokenize_args, device=device)
    test_generator = CaptionGenerator(test_dataset, tokenize_fct, tokenize_args, device=device)
    
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
    model = make_model(set_model, text_model=args.text_model, embed_dim=args.embed_dim).to(device)

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

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.checkpoint_name) if args.checkpoint_name is not None else None
    data_kwargs = {'set_size':args.set_size}
    print("Beginning training...")
    model, (losses, accs, test_acc) = train(model, optimizer, train_generator, train_generator, steps, batch_size, 
        checkpoint_dir=checkpoint_dir, data_kwargs=data_kwargs, eval_every=eval_every, eval_steps=eval_steps)

    print("Test Accuracy:", test_acc)

    model_out = model._modules['module'] if torch.cuda.device_count() > 1 else model
    torch.save(model_out, os.path.join(run_dir,"model.pt"))  
    torch.save({'losses':losses, 'eval_accs': accs, 'test_acc': test_acc, 'args':args}, os.path.join(run_dir,"logs.pt"))  

