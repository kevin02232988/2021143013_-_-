import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup, logging
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

logging.set_verbosity_error()


df = pd.read_csv("cleaned_sampled_12_reviews_final.csv", encoding="utf-8")
df["rating_review"] = df["rating_review"].astype(float)
df = df[df['rating_review'] != 3]
df['Sentiment'] = df['rating_review'].apply(lambda x: 1 if x >= 4 else 0)


train_df = df.sample(n=2000, random_state=42).reset_index(drop=True)
train_texts = list(train_df['review_full'].values)
train_labels = train_df['Sentiment'].values


eval_texts = list(df['review_full'].values)
eval_labels = df['Sentiment'].values


tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased', do_lower_case=True)

train_encodings = tokenizer(train_texts, truncation=True, max_length=256, padding="max_length")
eval_encodings = tokenizer(eval_texts, truncation=True, max_length=256, padding="max_length")

input_ids = np.array(train_encodings['input_ids'])
attention_masks = np.array(train_encodings['attention_mask'])


train_inputs, val_inputs, train_masks, val_masks, train_y, val_y = train_test_split(
    input_ids, attention_masks, train_labels, test_size=0.2, random_state=42)


batch_size = 8

train_dataset = TensorDataset(torch.tensor(train_inputs),
                              torch.tensor(train_masks),
                              torch.tensor(train_y).long())

val_dataset = TensorDataset(torch.tensor(val_inputs),
                            torch.tensor(val_masks),
                            torch.tensor(val_y).long())

train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)


eval_dataset = TensorDataset(torch.tensor(eval_encodings['input_ids']),
                             torch.tensor(eval_encodings['attention_mask']),
                             torch.tensor(eval_labels).long())
eval_dataloader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset), batch_size=batch_size)


model = MobileBertForSequenceClassification.from_pretrained('google/mobilebert-uncased', num_labels=2)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 4
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=len(train_dataloader) * epochs)


for epoch in range(epochs):
    model.train()
    total_loss = 0
    loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
    for batch in loop:
        b_input_ids, b_mask, b_labels = [x.to(device) for x in batch]

        model.zero_grad()
        outputs = model(input_ids=b_input_ids,
                        attention_mask=b_mask,
                        labels=b_labels)

        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} - Avg Train Loss: {total_loss / len(train_dataloader):.4f}")


    model.eval()
    val_preds, val_true = [], []
    for batch in val_dataloader:
        b_input_ids, b_mask, b_labels = [x.to(device) for x in batch]
        with torch.no_grad():
            outputs = model(input_ids=b_input_ids, attention_mask=b_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        val_preds.extend(preds.cpu().numpy())
        val_true.extend(b_labels.cpu().numpy())

    val_acc = accuracy_score(val_true, val_preds)
    print(f"Epoch {epoch+1} - Validation Accuracy: {val_acc:.4f}")


model.eval()
all_preds = []
all_labels = []

for batch in tqdm(eval_dataloader, desc="Evaluating on full dataset"):
    b_input_ids, b_mask, b_labels = [x.to(device) for x in batch]
    with torch.no_grad():
        outputs = model(input_ids=b_input_ids, attention_mask=b_mask)
    logits = outputs.logits
    preds = torch.argmax(logits, dim=1)
    all_preds.extend(preds.cpu().numpy())
    all_labels.extend(b_labels.cpu().numpy())


acc = accuracy_score(all_labels, all_preds)
print(f"\n🔎 Accuracy on full data (25,000): {acc:.4f}\n")
print(classification_report(all_labels, all_preds, digits=4))

df['Predicted_Sentiment'] = all_preds
df.to_csv("mobilebert_predictions_with_validation.csv", index=False)
print("예측 결과 저장 완료: mobilebert_predictions_with_validation.csv")
