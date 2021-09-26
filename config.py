'''
Author: Hao Luo
Sep. 2021 
'''

class configurations(object):
  def __init__(self):
    # Architecture
    self.emb_dim = 512
    self.hid_dim = 256
    self.img_feature_dim = 512
    self.n_layers = 3
    self.dropout = 0.5
    
    # Train param.
    self.batch_size = 64
    self.val_batch_size = 64
    self.test_batch_size = 64
    self.learning_rate = 0.00001
    self.cb_size = 128
    self.input_len = 8
    self.output_len = 5          
    self.num_steps = 200000                     
    self.store_steps = 500                     
    self.summary_steps = 100                
    self.gpu = 0
