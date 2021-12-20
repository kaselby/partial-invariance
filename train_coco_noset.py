import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, IterableDataset
from train_omniglot import ConvEncoder

import torchvision
from torchvision.datasets import CocoCaptions
import torchvision.transforms as T

import os
import fasttext
import tqdm

import re
from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()

def preprocess_text(document):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(document))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()

        # Lemmatization
        tokens = document.split()
        tokens = [stemmer.lemmatize(word) for word in tokens]
        #tokens = [word for word in tokens if word not in en_stop]
        #tokens = [word for word in tokens if len(word) > 3]

        preprocessed_text = ' '.join(tokens)

        return preprocessed_text

def load_caption_data(imgdir, anndir):
    transforms = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    
    train_dataset = CocoCaptions(root=os.path.join(imgdir, "train2014"), annFile=os.path.join(anndir, "captions_train2014.json"), transform=transforms)
    val_dataset = CocoCaptions(root=os.path.join(imgdir, "val2014"), annFile=os.path.join(anndir, "captions_val2014.json"), transform=transforms)

    return train_dataset, val_dataset


class CocoMatchingModel(nn.Module):
    def __init__(self, text_enc, img_enc, latent_size, hidden_size, output_size):
        super().__init__()
        self.text_encoder = text_enc
        self.img_encoder = img_enc
        self.decoder = nn.Sequential(
            nn.Linear(2*latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        #self.X_proj = nn.Linear(img_encoder.output_size, self.latent_size) if img_encoder.output_size != self.latent_size else None
        #self.Y_proj = nn.Linear(text_encoder.output_size, self.latent_size) if text_encoder.output_size != self.latent_size else None
    
    def forward(self, imgs, texts):
        #packed_texts = torch.nn.utils.rnn.pack_sequence([torch.tensor(seq) for seq in texts], enforce_sorted=False)
        encoded_texts = self.text_encoder(texts)
        ZY, _ = torch.nn.utils.rnn.pad_packed_sequence(encoded_texts, batch_first=True)[:,0]
        ZX = self.img_encoder(imgs)
        return self.decoder(torch.cat([ZX, ZY], dim=1), **kwargs)



def build_model(latent_size, hidden_size, embed_size=300):
    text_enc = nn.LSTM(embed_size, latent_size, batch_first=True, bidirectional=True)
    img_enc = ConvEncoder.make_coco_model(latent_size)

    model = CocoMatchingModel(text_enc, img_enc, latent_size, hidden_size, 1)
    return model

def process_captions(ft, captions, start_tok="cls"):
    processed_seqs = start_tok + " " + preprocess_text(captions[0]) 
    seq_tensors = torch.tensor([ft[x] for x in seq.split(" ") if x in ft])
    return seq_tensors
    #return torch.nn.utils.rnn.pad_sequence(seq_tensors, batch_first=True), [seq.size(1) for seq in seq_tensors]

def collate_with_padding(batch):
    inputs, labels = zip(*batch)
    imgs, texts = zip(*inputs)
    packed_text = torch.nn.utils.rnn.pack_sequence(texts, enforce_sorted=False)
    return (torch.stack(imgs, 0), packed_text), labels

class CaptionMatchingDataset(IterableDataset):
    def __init__(self, dataset, embeddings, device=torch.device('cpu')):
        self.dataset=dataset
        self.embeddings = embeddings
        self.device = device

    def __iter__(self, p=0.5):
        N = len(self.dataset)
        indices = torch.randperm(N)
        aligned = (torch.rand(N) > p)
        unaligned_indices = torch.nonzero(aligned.logical_not())
        unaligned_pairs = torch.cat([unaligned_indices, unaligned_indices.roll(-1, dims=0)], 1)
        unaligned_map = {unaligned_pairs[i,0].item():unaligned_pairs[i,1].item() for i in range(unaligned_pairs.size(0))}

        for i in range(N):
            j = indices[i]
            imgs, captions = self.dataset[j]
            if not aligned[j].item():
                captions = self.dataset[unaligned_map[j.item()]][1]
            yield (imgs, process_captions(self.embeddings, captions)), aligned[j]
    
    def __len__(self):
        return len(self.dataset)



def train(model, optimizer, train_dataset, val_dataset, epochs, batch_size):
    criterion = nn.BCEWithLogitsLoss()

    for i in range(epochs):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_with_padding)
        train_loss = 0
        for (imgs, captions), aligned in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            yhat = model(imgs.to(model.device()), captions.to(model.device()))
            loss = criterion(model.squeeze(-1), aligned)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        with torch.no_grad():
            val_loss = 0
            acc = 0
            val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_with_padding)
            for (imgs, captions), aligned in val_loader:
                yhat = model(imgs, captions)
                loss = criterion(model.squeeze(-1), aligned)
                val_loss += loss.item()
                acc += (yhat > 0).sum()
        print("Epoch: %d\tTraining Loss:%f\tValidation Loss:%f\tValidation Accuracy:%f" % (i, avg_loss/len(train_dataset), val_loss/len(val_dataset), acc/len(val_dataset)))


if __name__ == "__main__":
    lr = 1e-4
    bs = 64
    ls = 256
    hs = 512
    epochs = 10
    embed_dim=300
    embed_path="cc.en.300.bin"
    coco_path="./coco"

    ft = fasttext.load_model(embed_path)

    device = torch.device("cuda")

    base_train_dataset, base_val_dataset = load_caption_data(os.path.join(coco_path, "images"), os.path.join(coco_path, "annotations"))
    train_dataset = CaptionMatchingDataset(base_train_dataset, ft, device)
    val_dataset = CaptionMatchingDataset(base_val_dataset, ft, device)

    model = build_model(ls, hs, embed_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr)

    train(model, optimizer, train_dataset, val_dataset, epochs, bs)

    
        


            

