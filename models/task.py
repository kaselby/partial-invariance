
import torch
import torch.nn as nn
import torch.nn.init
import math

class ImageEncoderWrapper(nn.Module):
    def __init__(self, encoder, output_size):
        super().__init__()
        self.encoder = encoder
        self.output_size = output_size

    def forward(self, inputs):
        encoded_batch = self.encoder(inputs.view(-1, *inputs.size()[-3:]))
        return encoded_batch.view(*inputs.size()[:-3], encoded_batch.size(-1))

class BertEncoderWrapper(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.output_size = bert.config.hidden_size

    def forward(self, inputs):
        ss, n_seqs, bert_inputs = inputs['set_size'], inputs['n_seqs'], inputs['inputs']
        encoded_seqs = self.bert(**bert_inputs).last_hidden_state
        if n_seqs == 1:
            out = encoded_seqs[:,0].reshape(-1, ss, encoded_seqs.size(-1))
        else:
            out = encoded_seqs[:,0].reshape(-1, ss, n_seqs, encoded_seqs.size(-1)).mean(2)
        return out

class EmbeddingEncoderWrapper(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.output_size=embed_dim

    def forward(self, inputs):
        return inputs


class MultiSetModel(nn.Module):
    def __init__(self, set_model, X_encoder, Y_encoder):
        super().__init__()
        self.set_model = set_model
        self.X_encoder = X_encoder
        self.Y_encoder = Y_encoder
        self.latent_size = set_model.input_size
        self.X_proj = nn.Linear(X_encoder.output_size, self.latent_size) if X_encoder.output_size != self.latent_size else None
        self.Y_proj = nn.Linear(Y_encoder.output_size, self.latent_size) if Y_encoder.output_size != self.latent_size else None

    def forward(self, X, Y, **kwargs):
        ZX = self.X_encoder(X)
        ZY = self.Y_encoder(Y)

        if self.X_proj is not None:
            ZX = self.X_proj(ZX)
        if self.Y_proj is not None:
            ZY = self.Y_proj(ZY)
        
        return self.set_model(ZX, ZY, **kwargs)



class CocoTrivialModel(nn.Module):
    def __init__(self, text_enc, img_enc, latent_size, hidden_size, output_size):
        self.text_encoder = text_enc
        self.img_encoder = img_enc
        self.decoder = nn.Sequential(
            nn.Linear(2*latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.X_proj = nn.Linear(X_encoder.output_size, self.latent_size) if X_encoder.output_size != self.latent_size else None
        self.Y_proj = nn.Linear(Y_encoder.output_size, self.latent_size) if Y_encoder.output_size != self.latent_size else None
    
    def forward(self, imgs, texts):
        ZX = self.img_encoder(imgs)
        ZY = self.text_encoder(texts)

        if self.X_proj is not None:
            ZX = self.X_proj(ZX)
        if self.Y_proj is not None:
            ZY = self.Y_proj(ZY)
        
        return self.set_model(ZX, ZY, **kwargs)


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