import torch
from utils import load_lora_model_from_run, get_device
from data import tokenize
import torch.nn.utils.rnn as rnn_utils

# 1) 指定 LoRA 运行目录（包含 rnn_lora_adapters_*.pt / vocab.pkl / label_encoder.pkl / finetune_lora_config.json）
lora_run_dir = "Checkpoints_lora/rnn_lora_20250819_003150"  # 按你的目录修改

device = get_device()
model, vocab, label_encoder, cfg = load_lora_model_from_run(lora_run_dir, map_location=str(device))
model.to(device).eval()

def encode(text, vocab, max_len=200):
    tokens = tokenize(text)
    unk = vocab.get('<unk>', 0)
    ids = [vocab.get(t, unk) for t in tokens][:max_len]
    return torch.tensor(ids, dtype=torch.long)

def predict(texts):
    pad_idx = vocab.get('<pad>', 1)
    max_len = int(cfg.get("max_len", 200))
    seqs = [encode(t, vocab, max_len=max_len) for t in texts]
    batch = rnn_utils.pad_sequence(seqs, batch_first=True, padding_value=pad_idx).to(device)
    with torch.no_grad():
        logits = model(batch)
        pred_ids = logits.argmax(dim=1).cpu().numpy()
    return label_encoder.inverse_transform(pred_ids)

print(predict(["This movie is great!", "Terrible plot."]))