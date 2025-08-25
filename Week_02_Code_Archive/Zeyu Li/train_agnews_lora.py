import torch, numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
import evaluate

BASE = "distilbert-base-uncased"   # 也可换 prajjwal1/bert-tiny 先跑通
NUM_LABELS = 4
device = "cuda" if torch.cuda.is_available() else "cpu"

ds = load_dataset("ag_news")
tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)

def tokenize_fn(examples):
    return tok(examples["text"], truncation=True, max_length=256)
tok_ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
collator = DataCollatorWithPadding(tokenizer=tok)

acc = evaluate.load("accuracy")
f1 = evaluate.load("f1")
def metrics(p):
    logits, labels = p
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
            "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"]}

model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=NUM_LABELS)

# DistilBERT 常用注意力权重名：q_lin / k_lin / v_lin / out_lin
lora_cfg = LoraConfig(
    task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.1, bias="none",
    target_modules=["q_lin","k_lin","v_lin","out_lin"]
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir="./agnews_lora_out",
    learning_rate=2e-4,
    per_device_train_batch_size=8,      # 显存不够可降到 4/2
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,      # 等效更大 batch
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tok_ds["train"],
    eval_dataset=tok_ds["test"],
    tokenizer=tok,
    data_collator=collator,
    compute_metrics=metrics
)

if __name__ == "__main__":
    model.to(device)
    trainer.train()
    print(trainer.evaluate())
    trainer.save_model("./agnews_lora_out")  # 保存 LoRA 适配器
