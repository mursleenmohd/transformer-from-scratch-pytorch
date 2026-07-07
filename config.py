import torch

SRC_VOCAB_SIZE = 5000  
TGT_VOCAB_SIZE = 5000


D_MODEL = 128          
NUM_HEADS = 4         
NUM_LAYERS = 2        
D_FF = 512        
MAX_SEQ_LENGTH = 20   
DROPOUT = 0.1


BATCH_SIZE = 32
EPOCHS = 30
LR = 0.0005

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")