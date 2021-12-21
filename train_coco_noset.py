import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, IterableDataset
from train_omniglot import ConvEncoder

import torchvision
from torchvision.datasets import CocoCaptions
from torchvision.models import resnet101
import torchvision.transforms as T

from transformers import BertModel, BertTokenizer

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

class Processor():
    def collate(self, batch):
        inputs, labels = zip(*batch)
        imgs, texts = zip(*inputs)
        collated_text = self.collate_text(texts)
        return (torch.stack(imgs, 0), collated_text), torch.stack(labels, dim=0).float()

    def collate_text(self, text_batch):
        pass

class LSTMProcessor(Processor):
    def __init__(self, embeddings, start_tok="cls"):
        super().__init__()
        self.embeddings = embeddings
        self.start_tok = start_tok

    def process_example(self, example):
        processed_seq = self.start_tok + " " + preprocess_text(captions[0]) 
        seq_tensors = torch.tensor([self.embeddings[x] for x in processed_seq.split(" ") if x in self.embeddings])
        return seq_tensors

    def collate_text(self, text_batch):
        return torch.nn.utils.rnn.pack_sequence([self.process_example(x) for x in text_batch], enforce_sorted=False)

class BERTProcessor(Processor):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
    
    def collate_text(self, text_batch):
        return tokenizer([captions[0] for captions in text_batch], padding=True, truncation=True, return_tensors='pt')


def load_caption_data(imgdir, anndir):
    transforms = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    #transforms = T.ToTensor()
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
        ZX = self.img_encoder(imgs)
        ZY = self.text_encoder(texts)
        return self.decoder(torch.cat([ZX, ZY], dim=1))

class BERTEncoder(nn.Module):
    def __init__(self, bert, latent_size):
        super().__init__()
        self.bert=bert
        self.proj = nn.Linear(bert.config.hidden_size, latent_size)

    def forward(self, inputs):
        return self.proj(self.bert(**inputs).last_hidden_state[:,0])

class LSTMEncoder(nn.Module):
    def __init__(self, lstm):
        super().__init__()
        self.lstm=lstm

    def forward(self, inputs):
        packed_output, (h,c) = self.lstm(inputs)
        ZY = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)[0][:,0]
        ZY = ZY.view(bs, 2, -1).sum(dim=1)
        return ZY



def build_model(latent_size, hidden_size, embed_size=300):
    #text_enc = LSTMEncoder(nn.LSTM(embed_size, latent_size, batch_first=True, bidirectional=True))
    text_enc = BERTEncoder(BertModel.from_pretrained("bert-base-uncased"), latent_size)
    
    #img_enc = ConvEncoder.make_coco_model(latent_size)
    resnet = resnet101(pretrained=True)
    resnet.fc=nn.Linear(2048, latent_size)
    img_enc=resnet

    model = CocoMatchingModel(text_enc, img_enc, latent_size, hidden_size, 1)
    return model

'''
def process_captions(ft, captions, start_tok="cls"):
    processed_seq = start_tok + " " + preprocess_text(captions[0]) 
    seq_tensors = torch.tensor([ft[x] for x in processed_seq.split(" ") if x in ft])
    return seq_tensors
    #return torch.nn.utils.rnn.pad_sequence(seq_tensors, batch_first=True), [seq.size(1) for seq in seq_tensors]

def collate_with_padding(batch):
    inputs, labels = zip(*batch)
    imgs, texts = zip(*inputs)
    packed_text = torch.nn.utils.rnn.pack_sequence(texts, enforce_sorted=False)
    return (torch.stack(imgs, 0), packed_text), torch.stack(labels, dim=0).float()
'''


class CaptionMatchingDataset(IterableDataset):
    def __init__(self, dataset, N=-1):
        self.dataset=dataset
        self.N = len(self.dataset) if N <= 0 else N
        #self.embeddings = embeddings

    def __iter__(self, p=0.5):
        indices = torch.randperm(self.N)
        aligned = (torch.rand(self.N) > p)
        unaligned_indices = torch.nonzero(aligned.logical_not())
        unaligned_pairs = torch.cat([unaligned_indices, unaligned_indices.roll(-1, dims=0)], 1)
        unaligned_map = {unaligned_pairs[i,0].item():unaligned_pairs[i,1].item() for i in range(unaligned_pairs.size(0))}

        for i in range(self.N):
            j = indices[i].item()
            imgs, captions = self.dataset[j]
            if not aligned[j].item():
                captions = self.dataset[unaligned_map[j]][1]
            yield (imgs, captions), aligned[j].float()
    
    def __len__(self):
        return self.N



def train(model, optimizer, processor, train_dataset, val_dataset, epochs, batch_size, device=torch.device('cpu')):
    criterion = nn.BCEWithLogitsLoss()

    for i in range(epochs):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=processor.collate)
        train_loss = 0
        for (imgs, captions), aligned in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            yhat = model(imgs.to(device), captions.to(device))
            loss = criterion(yhat.squeeze(-1), aligned.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        with torch.no_grad():
            val_loss = 0
            acc = 0
            val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=processor.collate)
            for (imgs, captions), aligned in val_loader:
                yhat = model(imgs.to(device), captions.to(device))
                loss = criterion(yhat.squeeze(-1), aligned.to(device))
                val_loss += loss.item()
                acc += torch.eq((yhat.squeeze(-1) >= 0), aligned.to(device)).sum().item()
        print("Epoch: %d\tTraining Loss:%f\tValidation Loss:%f\tValidation Accuracy:%f" % (i, train_loss/len(train_loader), val_loss/len(val_loader), acc/len(val_dataset)))


if __name__ == "__main__":
    lr = 1e-5
    bs = 64
    ls = 1024
    hs = 4096
    epochs = 5
    embed_dim=300
    embed_path="cc.en.300.bin"
    coco_path="./coco"

    #ft = fasttext.load_model(embed_path)
    #processor = LSTMProcessor(ft)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    processor = BERTProcessor(tokenizer)

    device = torch.device("cuda")

    base_train_dataset, base_val_dataset = load_caption_data(os.path.join(coco_path, "images"), os.path.join(coco_path, "annotations"))
    train_dataset = CaptionMatchingDataset(base_train_dataset)
    val_dataset = CaptionMatchingDataset(base_val_dataset)

    model = build_model(ls, hs, embed_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr)

    train(model, optimizer, processor, train_dataset, val_dataset, epochs, bs, device)

    
        


            

