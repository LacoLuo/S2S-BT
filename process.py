'''
Author: Hao Luo
Sep. 2021
'''

import os 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.utils.data as data 
import torchvision.transforms as trf 
from torch.utils.tensorboard import SummaryWriter

from data_feed import DataFeed
from model import Seq2Seq, Encoder, Decoder
from utils import save_model, load_model, infinite_iter, schedule_sampling

def build_model(config, device):

  encoder = Encoder(config.cb_size, config.emb_dim, config.hid_dim, config.n_layers, config.dropout)
  decoder = Decoder(config.cb_size, config.emb_dim, config.hid_dim, config.n_layers, config.dropout)
  model = Seq2Seq(config, encoder, decoder, device)
  # Setup the optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
  if config.load_model_path:
    model = load_model(model, config.load_model_path)
  model = model.to(device)

  return model, optimizer

def train(config, model, optimizer, train_iter, loss_function, total_steps, summary_steps, device, writer):
  model.train()
  model.zero_grad()
  losses = []
  loss_sum = 0.0
  top_1_acc_n5, top_1_acc_n3, top_1_acc_n1 = 0.0, 0.0, 0.0
  exp_decay_score_n5, exp_decay_score_n3, exp_decay_score_n1 = 0.0, 0.0, 0.0
  for step in range(summary_steps):

    beams = next(train_iter)

    src_beams = beams[:, :config.input_len].type(torch.LongTensor).to(device)
    trg_beams = beams[:, config.input_len : config.input_len + config.output_len].type(torch.LongTensor).to(device)

    outputs, preds = model(src_beams, trg_beams, schedule_sampling(total_steps + step + 1))

    outputs = outputs.reshape(-1, outputs.size(2))
    targets = trg_beams.reshape(-1)
    loss = loss_function(outputs, targets)
    
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()

    loss_sum += loss.item()

    # Calculate the Top-1 Accuracy
    targets = targets.view(beams.size(0), -1)
    top_1_acc_n5 += torch.sum( torch.prod(preds == targets, dim=1, dtype=torch.float) ).item() / beams.size(0)
    top_1_acc_n3 += torch.sum( torch.prod(preds[:, :3] == targets[:, :3], dim=1, dtype=torch.float) ).item() / beams.size(0)
    top_1_acc_n1 += torch.sum( torch.prod(preds[:, :1] == targets[:, :1], dim=1, dtype=torch.float) ).item() / beams.size(0)


    # Calculate the Exponential-Decay Score
    exp_decay_score_n5 += torch.sum( torch.exp( - torch.norm(((preds - targets)), 1, -1, dtype=torch.float) / (5 * 0.5) ) ).item() / beams.size(0)
    exp_decay_score_n3 += torch.sum( torch.exp( - torch.norm(((preds[:, :3] - targets[:, :3])), 1, -1, dtype=torch.float) / (3 * 0.5) ) ).item() / beams.size(0)
    exp_decay_score_n1 += torch.sum( torch.exp( - torch.norm(((preds[:, :1] - targets[:, :1])), 1, -1, dtype=torch.float) / (1 * 0.5) ) ).item() / beams.size(0)

    if (step + 1) % (summary_steps / 5) == 0:
      loss_sum = loss_sum / (summary_steps / 5)
      top_1_acc_n1, top_1_acc_n3, top_1_acc_n5 = top_1_acc_n1 / (summary_steps / 5), top_1_acc_n3 / (summary_steps / 5), top_1_acc_n5 / (summary_steps / 5)
      exp_decay_score_n1, exp_decay_score_n3, exp_decay_score_n5 = exp_decay_score_n1 / (summary_steps / 5), exp_decay_score_n3 / (summary_steps / 5), exp_decay_score_n5 / (summary_steps / 5)

      # write training summmary
      writer.add_scalar('Loss/train', loss_sum, total_steps + step + 1)
      writer.add_scalars('Top-1_Accuracy/train', {'n=1': top_1_acc_n1, 'n=3': top_1_acc_n3, 'n=5': top_1_acc_n5}, total_steps + step + 1)
      writer.add_scalars('Exp-Decay_Score/train', {'n=1': exp_decay_score_n1, 'n=3': exp_decay_score_n3, 'n=5': exp_decay_score_n5}, total_steps + step + 1)

      print ("\r", "train [{}] loss: {:.3f}, Top-1 Acc (n=1): {:.3f}, Top-1 Acc(n=3): {:.3f}, Top-1 Acc (n=5): {:.3f}      ".format(total_steps + step + 1, loss_sum, top_1_acc_n1, top_1_acc_n3, top_1_acc_n5), end=" ")
      losses.append(loss_sum)
      loss_sum = 0.0
      top_1_acc_n5, top_1_acc_n3, top_1_acc_n1 = 0.0, 0.0, 0.0
      exp_decay_score_n5, exp_decay_score_n3, exp_decay_score_n1 = 0.0, 0.0, 0.0

  return model, optimizer, losses

def valid(config, model, val_loader, loss_function, device):
  model.eval()
  loss_sum = 0.0
  top_1_acc_n5, top_1_acc_n3, top_1_acc_n1 = 0.0, 0.0, 0.0
  exp_decay_score_n5, exp_decay_score_n3, exp_decay_score_n1 = 0.0, 0.0, 0.0
  n = 0
  result = []
  for beams in val_loader:
    src_beams = beams[:, :config.input_len].type(torch.LongTensor).to(device)
    trg_beams = beams[:, config.input_len : config.input_len + config.output_len].type(torch.LongTensor).to(device)

    batch_size = beams.size(0)
    with torch.no_grad():
      outputs, preds = model.inference(src_beams, trg_beams)

    outputs = outputs.reshape(-1, outputs.size(2))
    targets = trg_beams.reshape(-1)

    loss = loss_function(outputs, targets)
    loss_sum += loss.item()

    targets = targets.view(batch_size, -1)
    # Calculate Top-1 Accuracy
    top_1_acc_n5 += torch.sum( torch.prod(preds == targets, dim=1, dtype=torch.float) ).item()
    top_1_acc_n3 += torch.sum( torch.prod(preds[:, :3] == targets[:, :3], dim=1, dtype=torch.float) ).item()
    top_1_acc_n1 += torch.sum( torch.prod(preds[:, :1] == targets[:, :1], dim=1, dtype=torch.float) ).item()

    # Calculate Exponential-Decay Score
    exp_decay_score_n5 += torch.sum( torch.exp( - torch.norm(((preds - targets)), 1, -1, dtype=torch.float) / (5 * 0.5) ) ).item()
    exp_decay_score_n3 += torch.sum( torch.exp( - torch.norm(((preds[:, :3] - targets[:, :3])), 1, -1, dtype=torch.float) / (3 * 0.5) ) ).item()
    exp_decay_score_n1 += torch.sum( torch.exp( - torch.norm(((preds[:, :1] - targets[:, :1])), 1, -1, dtype=torch.float) / (1 * 0.5) ) ).item()

    n += batch_size

  return loss_sum / len(val_loader), (top_1_acc_n1 / n, top_1_acc_n3 / n, top_1_acc_n5 / n), (exp_decay_score_n1 / n, exp_decay_score_n3 / n, exp_decay_score_n5 / n), result

def test(config, model, dataloader, device):
  n = 0
  top_1_acc_n5, top_1_acc_n3, top_1_acc_n1 = 0.0, 0.0, 0.0
  exp_decay_score_n5, exp_decay_score_n3, exp_decay_score_n1 = 0.0, 0.0, 0.0
  for i, datas in enumerate(dataloader):
    beams = datas

    src_beams = beams[:, :config.input_len].type(torch.LongTensor).to(device)
    trg_beams = beams[:, config.input_len : config.input_len + config.output_len].type(torch.LongTensor).to(device)

    batch_size = beams.size(0)
    outputs, preds = model.inference(src_beams, None)

    targets = trg_beams.view(batch_size, -1)
    # Calculate Top-1 Accuracy
    top_1_acc_n5 += torch.sum( torch.prod(preds == targets, dim=1, dtype=torch.float) ).item()
    top_1_acc_n3 += torch.sum( torch.prod(preds[:, :3] == targets[:, :3], dim=1, dtype=torch.float) ).item()
    top_1_acc_n1 += torch.sum( torch.prod(preds[:, :1] == targets[:, :1], dim=1, dtype=torch.float) ).item()

    # Calculate Exponential-Decay Score
    exp_decay_score_n5 += torch.sum( torch.exp( - torch.norm(((preds - targets)), 1, -1, dtype=torch.float) / (5 * 0.5) ) ).item()
    exp_decay_score_n3 += torch.sum( torch.exp( - torch.norm(((preds[:, :3] - targets[:, :3])), 1, -1, dtype=torch.float) / (3 * 0.5) ) ).item()
    exp_decay_score_n1 += torch.sum( torch.exp( - torch.norm(((preds[:, :1] - targets[:, :1])), 1, -1, dtype=torch.float) / (1 * 0.5) ) ).item()

    n += batch_size
  return (top_1_acc_n1 / n, top_1_acc_n3 / n, top_1_acc_n5 / n), (exp_decay_score_n1 / n, exp_decay_score_n3 / n, exp_decay_score_n5 / n)

def train_process(config):
  device = torch.device(f'cuda:{config.gpu}')

  # Get output_directory ready
  if not os.path.isdir(config.store_model_path):
    os.makedirs(config.store_model_path)

  # create a summary writer with the specified folder name.
  writer = SummaryWriter(os.path.join(config.store_model_path, 'summary'))

  train_feed = DataFeed(
                      root_dir = config.trn_data_path,
                      n = config.input_len + config.output_len)
  train_loader = data.DataLoader(train_feed, batch_size=config.batch_size, num_workers=8, pin_memory = True)
  train_iter = infinite_iter(train_loader)

  # Prepare validation data
  val_feed = DataFeed(
                     root_dir = config.val_data_path,
                     n = config.input_len + config.output_len)
  val_loader = data.DataLoader(val_feed, batch_size=config.val_batch_size, num_workers=8, pin_memory = True)

  # Build model
  model, optimizer = build_model(config, device)
  print ("Finish building model")

  # Define loss function
  loss_function = nn.CrossEntropyLoss()

  train_losses, val_losses, top_1_accs = [], [], []
  total_steps = 0
  best_result = 0.0
  count_for_early_stop = 0
  print("---training start---")
  while (total_steps < config.num_steps):
    
    # Train
    model, optimizer, loss = train(config, model, optimizer, train_iter, loss_function, total_steps, config.summary_steps, device, writer)
    train_losses += loss

    # Validate
    val_loss, top_1_acc, exp_decay_score, result = valid(config, model, val_loader, loss_function, device)
    val_losses.append(val_loss)
    top_1_accs.append(top_1_acc)

    total_steps += config.summary_steps

    # Write validation summary
    writer.add_scalar('Loss/validation', val_loss, total_steps)
    writer.add_scalars('Top-1_Accuracy/validation', {'n=1': top_1_acc[0], 'n=3': top_1_acc[1], 'n=5': top_1_acc[2]}, total_steps)
    writer.add_scalars('Exp-Decay_Score/validation', {'n=1': exp_decay_score[0], 'n=3': exp_decay_score[1], 'n=5': exp_decay_score[2]}, total_steps)

    print ("\r", "val [{}] loss: {:.3f}, Top-1 Acc (n=1): {:.3f}, Top-1 Acc(n=3): {:.3f}, Top-1 Acc (n=5): {:.3f}, Exp-Decay Score (n=1): {:.3f}, Exp-Decay Score (n=3): {:.3f}, Exp-Decay Score (n=5): {:.3f}       "\
                    .format(total_steps, val_loss, top_1_acc[0], top_1_acc[1], top_1_acc[2], exp_decay_score[0], exp_decay_score[1], exp_decay_score[2]))

    result = (1*exp_decay_score[0]+3*exp_decay_score[1]+5*exp_decay_score[2])/9.0
    # Save checkpoint
    if total_steps % config.store_steps == 0.0:
        if result > best_result:
            best_result = result
            torch.save(model.state_dict(), f'{config.store_model_path}/best.ckpt')
            print(f'Record the best result with score: {best_result}')
        else:
            count_for_early_stop += 1

        if count_for_early_stop == 10:
            break
  
  writer.close()

  return train_losses, val_losses, top_1_accs
  
def test_process(config):
  device = torch.device(f'cuda:{config.gpu}')    


  test_feed = DataFeed(
                     root_dir = config.test_data_path,
                     n = config.input_len + config.output_len,
                     init_shuffle=False)
  test_loader = data.DataLoader(test_feed, batch_size=1024, num_workers=8, pin_memory = True)

  # Build model
  model, _ = build_model(config, device)
  #print ("Finish building model")

  model.eval()
  # Test
  top_1_acc, exp_decay_score = test(config, model, test_loader, device)
  
  print ("\r", "Top-1 Acc (n=1): {:.3f}, Top-1 Acc(n=3): {:.3f}, Top-1 Acc (n=5): {:.3f}, Exp-Decay Score (n=1): {:.3f}, Exp-Decay Score (n=3): {:.3f}, Exp-Decay Score (n=5): {:.3f}       "\
                    .format(top_1_acc[0], top_1_acc[1], top_1_acc[2], exp_decay_score[0], exp_decay_score[1], exp_decay_score[2]))

  return 
