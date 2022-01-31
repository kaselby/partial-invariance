import torchvision
from torchvision.datasets import CocoCaptions, Flickr30k
import torchvision.transforms as T
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, Dataset, Subset

from transformers import BertModel, BertTokenizer

import os
import argparse
import math
import tqdm
import json
import fasttext

from models2 import MultiSetTransformer, PINE, MultiSetModel, NaiveMultiSetModel, BertEncoderWrapper, ImageEncoderWrapper, EmbeddingEncoderWrapper, CrossOnlyModel, MultiRNModel
from generators import CaptionGenerator, bert_tokenize_batch, fasttext_tokenize_batch, EmbeddingAlignmentGenerator, load_pairs
from train_omniglot import ConvEncoder


#def fasttext_encoder_preproc():


SS_SCHEDULE=[{'set_size':(1,5), 'steps':2000}, {'set_size':(3,10), 'steps':4000}, {'set_size':(8,15), 'steps':10000}]


def load_coco_data(imgdir, anndir):
    transforms = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    
    train_dataset = CocoCaptions(root=os.path.join(imgdir, "train2014"), annFile=os.path.join(anndir, "captions_train2014.json"), transform=transforms)
    val_dataset = CocoCaptions(root=os.path.join(imgdir, "val2014"), annFile=os.path.join(anndir, "captions_val2014.json"), transform=transforms)

    return train_dataset, train_dataset, val_dataset

def load_flickr_data(imgdir, annfile, split_file):
    transforms = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    
    dataset = Flickr30k(root=os.path.join(imgdir), ann_file=annfile, transform=transforms)

    with open(split_file, 'r') as f:
        splits_dict=json.load(f)
    splits = {'train':[], 'val':[], 'test':[]}
    for i in range(len(splits_dict['images'])):
        img_dict = splits_dict['images'][i]
        splits[img_dict["split"]].append(img_dict["imgid"])

    train_dataset = Subset(dataset, splits["train"])
    val_dataset = Subset(dataset, splits["val"])
    test_dataset = Subset(dataset, splits["test"])

    return train_dataset, val_dataset, test_dataset




def make_model(set_model, text_model='bert', img_model='vgg', embed_dim=300):
    if text_model == 'bert':
        model = BertModel.from_pretrained("bert-base-uncased")
        text_encoder = BertEncoderWrapper(model)
        #for param in text_encoder.parameters():
         #   param.requires_grad = False
    else:
        text_encoder = EmbeddingEncoderWrapper(embed_dim)

    if img_model == 'resnet':
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.fc = nn.Identity()#nn.Sequential(*list(vgg.classifier.children())[:-3])
        img_encoder = ImageEncoderWrapper(resnet, 2048)
        #for param in img_encoder.parameters():
        #    param.requires_grad = False
    else:
        enc = ConvEncoder.make_coco_model(256)
        img_encoder = ImageEncoderWrapper(enc, 256)
    
    #set_model = MultiSetTransformer(*args, **kwargs)
    return MultiSetModel(set_model, img_encoder, text_encoder)



def train(model, optimizer, train_dataset, test_dataset, steps, batch_size=64, eval_every=500, save_every=2000, eval_steps=100, test_steps=500, checkpoint_dir=None, data_kwargs={}):
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

            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            torch.save({'model':model,'optimizer':optimizer, 'step': i, 'losses':train_losses, 'accs': eval_accs}, checkpoint_path)

    
    test_acc = evaluate(model, test_dataset, test_steps, batch_size, data_kwargs)
    
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
    parser.add_argument('--model', type=str, default='csab', choices=['csab', 'rn', 'pine', 'naive', 'cross-only'])
    parser.add_argument('--dataset', type=str, default='coco', choices=['coco', 'flickr30k', 'fasttext'])
    parser.add_argument('--checkpoint_dir', type=str, default="/checkpoint/kaselby")
    parser.add_argument('--checkpoint_name', type=str, default=None)
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--latent_size', type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--set_size', type=int, nargs=2, default=[6,10])
    parser.add_argument('--anneal_set_size', action='store_true')
    parser.add_argument('--basedir', type=str, default="final-runs2")
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--eval_every', type=int, default=500)
    parser.add_argument('--eval_steps', type=int, default=200)
    parser.add_argument('--test_steps', type=int, default=500)
    parser.add_argument('--text_model', type=str, choices=['bert', 'ft'], default='bert')
    parser.add_argument('--img_model', type=str, choices=['resnet', 'base'], default='resnet')
    parser.add_argument('--embed_path', type=str, default="cc.en.300.bin")
    parser.add_argument('--embed_dim', type=int, default=300)
    parser.add_argument('--merge_type', type=str, default='concat', choices=['concat', 'sum'])
    return parser.parse_args()

#IMG_SIZE=105

if __name__ == '__main__':
    args = parse_args()

    run_dir = os.path.join(args.basedir, args.dataset, args.run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    device = torch.device("cuda:0")
    captioning = args.dataset != "fasttext"

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    if captioning:
        if args.text_model == 'bert':
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            tokenize_fct = bert_tokenize_batch
            tokenize_args = (tokenizer,)
        elif args.text_model == 'ft':
            ft = fasttext.load_model(args.embed_path)
            tokenize_fct = fasttext_tokenize_batch
            tokenize_args = (ft,)

        if args.dataset == "coco":
            train_dataset, val_dataset, test_dataset = load_coco_data(os.path.join(dataset_dir, "images"), os.path.join(dataset_dir, "annotations"))
        else:
            train_dataset, val_dataset, test_dataset = load_flickr_data(os.path.join(dataset_dir, "images"), os.path.join(dataset_dir, "annotations.token"), os.path.join(dataset_dir, "splits.json"))
        train_generator = CaptionGenerator(train_dataset, tokenize_fct, tokenize_args, device=device)
        val_generator = CaptionGenerator(val_dataset, tokenize_fct, tokenize_args, device=device)
        test_generator = CaptionGenerator(test_dataset, tokenize_fct, tokenize_args, device=device)
    else:
        src_emb = fasttext.load_model(os.path.join(dataset_dir, "cc.en.300.bin"))
        tgt_emb = fasttext.load_model(os.path.join(dataset_dir, "cc.fr.300.bin"))
        pairs = load_pairs(os.path.join(dataset_dir, "valid_en-fr.txt"))
        train_pairs, test_pairs = split_pairs(pairs, 0.1)
        train_generator = EmbeddingAlignmentGenerator(src_emb, tgt_emb, train_pairs, device=device)
        test_generator = EmbeddingAlignmentGenerator(src_emb, tgt_emb, test_pairs, device=device)
    
    if args.model == 'csab':
        model_kwargs={
            'ln':True,
            'remove_diag':False,
            'num_blocks':args.num_blocks,
            'num_heads':args.num_heads,
            'dropout':args.dropout,
            'equi':False,
            'decoder_layers': 0,
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
            #'weight_sharing': args.weight_sharing
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
            #'weight_sharing': args.weight_sharing,
            'pool1': 'max',
            'pool2': 'max'
        }
        set_model = MultiRNModel(args.latent_size, args.latent_size, args.hidden_size, 1, **model_kwargs)
    else:
        raise NotImplementedError("Model type not recognized.")
    if captioning:
        model = make_model(set_model, text_model=args.text_model, img_model=args.img_model, embed_dim=args.embed_dim).to(device)
    else:
        model = set_model.to(device)

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
    model, (losses, accs, test_acc) = train(model, optimizer, train_generator, test_generator, steps, batch_size, 
        checkpoint_dir=checkpoint_dir, data_kwargs=data_kwargs, eval_every=eval_every, eval_steps=eval_steps, test_steps=args.test_steps)

    print("Test Accuracy:", test_acc)

    model_out = model._modules['module'] if torch.cuda.device_count() > 1 else model
    torch.save(model_out, os.path.join(run_dir,"model.pt"))  
    torch.save({'losses':losses, 'eval_accs': accs, 'test_acc': test_acc, 'args':args}, os.path.join(run_dir,"logs.pt"))  

