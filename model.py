'''
Author: Hao Luo
Sep. 2021
'''

import torch
import torch.nn as nn
import random

class Encoder(nn.Module):
  def __init__(self, cb_size, emb_dim, hid_dim, n_layers, dropout):
    super().__init__()
    self.embedding = nn.Embedding(cb_size, emb_dim)
    self.hid_dim = hid_dim
    self.n_layers = n_layers
    self.rnn = nn.GRU( emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=True)
    self.dropout = nn.Dropout(dropout)

  def forward(self, input):
    # input[0] = [batch_size, sequence_len, 1]
    # input[1] = [batch_size, sequence_len, img_feat_dim]
    batch_size = list(input.size())[0]
    embedding = self.embedding(input)
    
    outputs, hidden = self.rnn(self.dropout(embedding))
    # outputs = [batch size, sequence len, hid dim * directions]
    # hidden =  [num_layers * directions, batch size  , hid dim]
        
    return outputs, hidden

class Decoder(nn.Module):
  def __init__(self, cb_size, emb_dim, hid_dim, n_layers, dropout):
    super().__init__()
    self.cb_size = cb_size
    self.hid_dim = hid_dim * 2
    self.n_layers = n_layers
    self.embedding = nn.Embedding(cb_size, emb_dim)
    self.input_dim = emb_dim 
    self.rnn = nn.GRU(self.input_dim, self.hid_dim, self.n_layers, dropout = dropout, batch_first=True)
    self.embedding2beam1 = nn.Linear(self.hid_dim, self.hid_dim * 2)
    self.embedding2beam2 = nn.Linear(self.hid_dim * 2, self.hid_dim * 4)
    self.embedding2beam3 = nn.Linear(self.hid_dim * 4, self.cb_size)
    self.dropout = nn.Dropout(dropout)

  def forward(self, input, hidden, encoder_outputs):
    # input = [batch size, 1]
    # hidden = [batch size, n layers * directions, hid dim]
    if input.dim() != 1:
      embedded = input
    else:
      input = input.unsqueeze(1)
      embedded = self.dropout(self.embedding(input))
    # embedded = [batch size, 1, emb dim]

    output, hidden = self.rnn(embedded, hidden)
    # output = [batch size, 1, hid dim]
    # hidden = [num_layers, batch size, hid dim]

    output = self.embedding2beam1(output.squeeze(1))
    output = self.embedding2beam2(output)
    prediction = self.embedding2beam3(output)
    # prediction = [batch size, codebook size]
    return prediction, hidden

    
class Seq2Seq(nn.Module):
  def __init__(self, config, encoder, decoder, device):
    super().__init__()
    self.config = config
    self.encoder = encoder
    self.decoder = decoder
    self.device = device
    assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
            
  def forward(self, input, target, teacher_forcing_ratio):
    # input  = [batch size, input len, 1]
    # target = [batch size, target len, 1]
    batch_size = target.shape[0]
    target_len = target.shape[1]
    cb_size = self.decoder.cb_size

    outputs = torch.zeros(batch_size, target_len, cb_size).to(self.device)
    encoder_outputs, hidden = self.encoder(input)
    # hidden =  [num_layers * directions, batch size  , hid dim]  --> [num_layers, directions, batch size  , hid dim]
    hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)
    hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)
    # start token
    input = torch.zeros(batch_size, 1, self.config.emb_dim).to(self.device)
    preds = []
    for t in range(target_len):
      output, hidden = self.decoder(input, hidden, encoder_outputs)
      outputs[:, t] = output
      teacher_force = random.random() <= teacher_forcing_ratio
      top1 = output.argmax(1)
      input = target[:, t] if teacher_force and t < target_len else top1
      preds.append(top1.unsqueeze(1))
    preds = torch.cat(preds, 1)
    return outputs, preds

  def inference(self, input, target):
    with torch.no_grad():
      # input  = [batch size, input len, vocab size]
      # target = [batch size, target len, vocab size]
      batch_size = input.shape[0]
      cb_size = self.decoder.cb_size

      outputs = torch.zeros(batch_size, self.config.output_len, cb_size).to(self.device)
      encoder_outputs, hidden = self.encoder(input)
      # hidden =  [num_layers * directions, batch size  , hid dim]  --> [num_layers, directions, batch size  , hid dim]
      hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)
      hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)
      # start token
      input = torch.zeros(batch_size, 1, self.config.emb_dim).to(self.device)
      preds = []
      for t in range(self.config.output_len):
        output, hidden = self.decoder(input, hidden, encoder_outputs)
        # output = [batch_size, 128]
        outputs[:, t] = output

        top1 = output.argmax(1)
        # top1 = [batch_size, 1]
        input = top1
        preds.append(top1.unsqueeze(1))
        
      preds = torch.cat(preds, 1)
      # preds = [batch_size, output_len, 1]
    return outputs, preds
