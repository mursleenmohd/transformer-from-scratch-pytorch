import torch
import torch.nn as nn
import torch.optim as optim
import config
from dataset import get_train_loader
from model.transformer import Transformer


train_loader, tokenizer = get_train_loader()

transformer = Transformer(
    config.SRC_VOCAB_SIZE, config.TGT_VOCAB_SIZE, config.D_MODEL, 
    config.NUM_HEADS, config.NUM_LAYERS, config.D_FF, config.MAX_SEQ_LENGTH, config.DROPOUT
).to(config.DEVICE)


criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=config.LR, betas=(0.9, 0.98), eps=1e-9)

transformer.train()
for epoch in range(config.EPOCHS):
    total_loss = 0
    for src_data, tgt_data in train_loader:

        src_data, tgt_data = src_data.to(config.DEVICE), tgt_data.to(config.DEVICE)
        
        optimizer.zero_grad()
        

        output = transformer(src_data, tgt_data[:, :-1])
        
        
        loss = criterion(output.contiguous().view(-1, config.TGT_VOCAB_SIZE), tgt_data[:, 1:].contiguous().view(-1))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    print(f"Epoch: {epoch+1}/{config.EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")


torch.save(transformer.state_dict(), "transformer.pth")
print("Real-text Model successfully saved to transformer.pth")