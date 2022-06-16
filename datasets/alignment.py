import torchvision
from torchvision.datasets import CocoCaptions, Flickr30k
import torchvision.transforms as T
import torch
from torch.utils.data import IterableDataset, Dataset, Subset

import os

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

class CaptionGenerator():
    def __init__(self, dataset, tokenize_fct, tokenize_args, p=0.5):
        self.N = len(dataset)
        self.dataset = dataset
        #self.img_encoder = img_encoder
        #self.text_encoder = text_encoder
        self.tokenize_fct = tokenize_fct
        self.tokenize_args = tokenize_args
        self.p = p
    
    def _split_dataset(self, dataset):
        imgs, text = [], []
        for img, captions in dataset:
            imgs.append(img)
            text.append(captions[0])
        return imgs, text
    
    def _build_text_batch(self, captions, use_first=True):
        return self.tokenize_fct(captions, *self.tokenize_args, use_first=use_first)

    def _build_img_batch(self, imgs):
        bs = len(imgs)
        ss = len(imgs[0])
        batch = torch.stack([torch.stack(batch_j, 0) for batch_j in imgs], 0)
        return batch
        

    def _generate(self, batch_size, set_size=(25,50)):
        aligned = (torch.rand(batch_size) < self.p)
        n_samples = torch.randint(*set_size, (1,)).item()

        indices = torch.randperm(self.N)
        X, Y = [], []
        for i in range(batch_size):
            mindex = n_samples * 2 * i

            imgs, captions = zip(*[self.dataset[i] for i in indices[mindex:mindex+n_samples]])
            X.append(imgs)
            if aligned[i].item():
                Y.append(captions)
            else:
                _, captions2 = zip(*[self.dataset[i] for i in indices[mindex+n_samples:mindex+n_samples*2]])
                Y.append(captions2)

        X = self._build_img_batch(X)
        Y = self._build_text_batch(Y)
        return (X, Y), aligned.float()
    
    def _generate_overlap(self, batch_size, set_size=(25,50), overlap_mult=3):
        aligned = (torch.rand(batch_size) < self.p)
        n_samples = torch.randint(*set_size, (1,)).item()

        indices = torch.randperm(self.N)
        X, Y = [], []
        for i in range(batch_size):
            mindex = n_samples * overlap_mult * i

            imgs, captions = zip(*[self.dataset[i] for i in indices[mindex:mindex+n_samples]])
            X.append(imgs)
            if aligned[i].item():
                Y.append(captions)
            else:
                unaligned_indices = torch.multinomial(torch.ones(n_samples * overlap_mult), n_samples)
                _, captions2 = zip(*[self.dataset[indices[mindex+i]] for i in unaligned_indices])
                Y.append(captions2)

        X = self._build_img_batch(X)
        Y = self._build_text_batch(Y)
        return (X, Y), aligned.float()
    
    def __call__(self, *args, overlap=False, **kwargs):
        if overlap:
            return self._generate_overlap(*args, **kwargs)
        else:
            return self._generate(*args, **kwargs)


def bert_tokenize_batch(captions, tokenizer, use_first=True):
    bs = len(captions)
    ss = len(captions[0])
    ns = 1 if use_first else len(captions[0][0]) 
    flattened_seqs = []
    for batch_element in captions:
        for set_element in batch_element:
            if use_first:
                flattened_seqs.append(set_element[0])
            else:
                flattened_seqs += set_element
    
    tokenized_seqs = tokenizer(flattened_seqs, padding=True, truncation=True, return_tensors='pt')
    return {'set_size':ss, 'n_seqs': ns, 'inputs': tokenized_seqs}

def fasttext_tokenize_batch(captions, ft, use_first=True):
    def preproc(s):
        s = s.translate(str.maketrans('', '', string.punctuation)).replace("\n", "")
        return s.lower().strip()
    batch = []
    for batch_element in captions:
        seqs = []
        for set_element in batch_element:
            try:
                if use_first:
                    seqs.append(torch.tensor(ft.get_sentence_vector(preproc(set_element[0]))))
                else:
                    seqs.append(torch.tensor([ft.get_sentence_vector(preproc(s)) for s in set_element]))
            except:
                import pdb;pdb.set_trace()
        batch.append(torch.stack(seqs, 0))
    batch = torch.stack(batch, 0)
    return batch


def load_pairs(pair_file):
    with open(pair_file, 'r') as infile:
        lines = infile.readlines()
    pairs = [line.strip().split(" ") for line in lines]
    return pairs

import random
def split_pairs(pairs, val_frac, test_frac):
    N = len(pairs)
    shuffled_pairs = random.sample(pairs, k=N)
    r1 = int(round(val_frac * N))
    r2 = int(round(test_frac * N))
    return shuffled_pairs[:N-r1-r2], shuffled_pairs[N-r1-r2:N-r2], shuffled_pairs[N-r2:]

import fasttext
class EmbeddingAlignmentGenerator():
    @classmethod
    def from_files(cls, src_file, tgt_file, dict_file, **kwargs):
        src_emb = fasttext.load_model(src_file)
        tgt_emb = fasttext.load_model(tgt_file)
        pairs=load_pairs(dict_file)
        return cls(src_emb, tgt_emb, pairs, **kwargs)

    def __init__(self, src_emb, tgt_emb, pairs, device=torch.device('cpu')):
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.pairs = pairs#[p for p in pairs if (p[0] in src_emb and p[1] in tgt_emb)] ASSUME THIS IS ALREADY DONE
        self.N = len(pairs)
        self.device = device

    def _generate_sets(self, indices):
        X,Y = [],[]
        for i in indices:
            word_x, word_y = self.pairs[i]
            X.append(self.src_emb[word_x])
            Y.append(self.tgt_emb[word_y])
        return torch.tensor(np.array(X)), torch.tensor(np.array(Y))

    def _generate(self, batch_size, p_aligned=0.5, set_size=(10,30), overlap_mult=3):
        aligned = (torch.rand(batch_size) < p_aligned).to(self.device)
        n_samples = torch.randint(*set_size, (1,)).item()
        indices = torch.randperm(self.N)
        X, Y_aligned = self._generate_sets(indices[:batch_size*n_samples])
        _, Y_unaligned = self._generate_sets(indices[batch_size*n_samples:batch_size*n_samples*2])
        X = X.view(batch_size, n_samples, -1).to(self.device)
        Y_aligned = Y_aligned.view(batch_size, n_samples, -1).to(self.device)
        Y_unaligned = Y_unaligned.view(batch_size, n_samples, -1).to(self.device)
        Y = torch.where(aligned.view(-1,1,1), Y_aligned, Y_unaligned)
        return (X, Y), aligned.float()

    def _generate_overlap(self, batch_size, p_aligned=0.5, set_size=(10,30), overlap_mult=3):
        aligned = (torch.rand(batch_size) < p_aligned).to(self.device)
        n_samples = torch.randint(*set_size, (1,)).item()

        indices = torch.randperm(self.N)
        X, Y = [], []
        for i in range(batch_size):
            mindex = n_samples * overlap_mult * i
            X_i, Y_aligned_i = self._generate_sets(indices[mindex:mindex+n_samples])
            X.append(X_i)
            if aligned[i].item():
                Y.append(Y_aligned_i)
            else:
                unaligned_indices = torch.multinomial(torch.ones(n_samples * overlap_mult), n_samples)
                _, Y_unaligned_i = self._generate_sets(indices[mindex+unaligned_indices])
                Y.append(Y_unaligned_i)
        X = torch.stack(X, dim=0).to(self.device)
        Y = torch.stack(Y, dim=0).to(self.device)
        return (X, Y), aligned.float()

    def __call__(self, *args, overlap=False, **kwargs):
        if overlap:
            return self._generate_overlap(*args, **kwargs)
        else:
            return self._generate(*args, **kwargs)
