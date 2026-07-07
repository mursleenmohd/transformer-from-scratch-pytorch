import torch
import config
from model.transformer import Transformer
from dataset import get_train_loader  


def greedy_decode(model, src, max_len=10, start_symbol=1, end_symbol=2):
    model.eval()
    src = src.to(config.DEVICE)
    
    tgt_input = torch.tensor([[start_symbol]], device=config.DEVICE) 

    with torch.no_grad():
        for _ in range(max_len):
            output = model(src, tgt_input)
            next_token_logits = output[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
            
            tgt_input = torch.cat([tgt_input, next_token], dim=1)

            if next_token.item() == end_symbol:
                break
                
    return tgt_input

if __name__ == "__main__":
    _, tokenizer = get_train_loader()

    
    model = Transformer(
        config.SRC_VOCAB_SIZE, config.TGT_VOCAB_SIZE, config.D_MODEL, 
        config.NUM_HEADS, config.NUM_LAYERS, config.D_FF, config.MAX_SEQ_LENGTH, config.DROPOUT
    ).to(config.DEVICE)
    
    model.load_state_dict(torch.load("transformer.pth", map_location=config.DEVICE))
    print("Trained model weights loaded successfully!")

    test_english_sentence = "The color and opacity of the highlight border." 
    
    print(f"\n[English]: {test_english_sentence}")
    
    src_tensor = tokenizer.encode(test_english_sentence, config.MAX_SEQ_LENGTH).unsqueeze(0)
    
    print("Translating...")
    generated_ids = greedy_decode(model, src_tensor, max_len=15)
    
    predicted_hindi_sentence = tokenizer.decode(generated_ids[0])
    
    print(f"[Hindi]: {predicted_hindi_sentence}\n")