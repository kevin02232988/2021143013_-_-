import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup, logging
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import matplotlib.pyplot as plt


GPU = torch.cuda.is_available()
device = torch.device("cuda" if GPU else "cpu")
print("Using device:", device)

logging.set_verbosity_error()


df = pd.read_csv("cleaned_sampled_12_reviews_final.csv", encoding="utf-8")
df["rating_review"] = df["rating_review"].astype(float)


df = df[df['rating_review'] != 3]
df['Sentiment'] = df['rating_review'].apply(lambda x: 1 if x >= 4 else 0)


sample_df = df.sample(n=2000, random_state=42).reset_index(drop=True)
data_X = list(sample_df['review_full'].values)
labels = sample_df['Sentiment'].values


tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased', do_lower_case=True)
inputs = tokenizer(data_X, truncation=True, max_length=256, padding="max_length", add_special_tokens=True)

input_ids = np.array(inputs['input_ids'])
attention_mask = np.array(inputs['attention_mask'])


train, validation, train_y, validation_y = train_test_split(input_ids, labels, test_size=0.2, random_state=2025)
train_mask, validation_mask, _, _ = train_test_split(attention_mask, labels, test_size=0.2, random_state=2025)

# 4. TensorDataset 및 DataLoader 구성
batch_size = 8

train_inputs = torch.tensor(train)
train_labels = torch.tensor(train_y).to(torch.long)
train_masks = torch.tensor(train_mask)
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_inputs = torch.tensor(validation)
validation_labels = torch.tensor(validation_y).to(torch.long)
validation_masks = torch.tensor(validation_mask)
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)


model = MobileBertForSequenceClassification.from_pretrained('google/mobilebert-uncased', num_labels=2)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 4
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=len(train_dataloader) * epochs)


epoch_result = []

for e in range(epochs):
    model.train()
    total_train_loss = 0

    progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {e+1}")
    for batch in progress_bar:
        batch_ids, batch_mask, batch_labels = batch
        batch_ids = batch_ids.to(device)
        batch_mask = batch_mask.to(device)
        batch_labels = batch_labels.to(device)

        model.zero_grad()
        output = model(batch_ids, attention_mask=batch_mask, labels=batch_labels)
        loss = output.loss
        total_train_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        progress_bar.set_postfix({'loss': loss.item()})

    avg_train_loss = total_train_loss / len(train_dataloader)


    model.eval()
    train_pred, train_true = [], []
    for batch in train_dataloader:
        batch_ids, batch_mask, batch_labels = [x.to(device) for x in batch]
        with torch.no_grad():
            output = model(batch_ids, attention_mask=batch_mask)
        logits = output.logits
        pred = torch.argmax(logits, dim=1)
        train_pred.extend(pred.cpu().numpy())
        train_true.extend(batch_labels.cpu().numpy())

    train_accuracy = (np.array(train_pred) == np.array(train_true)).mean()


    val_pred, val_true = [], []
    for batch in validation_dataloader:
        batch_ids, batch_mask, batch_labels = [x.to(device) for x in batch]
        with torch.no_grad():
            output = model(batch_ids, attention_mask=batch_mask)
        logits = output.logits
        pred = torch.argmax(logits, dim=1)
        val_pred.extend(pred.cpu().numpy())
        val_true.extend(batch_labels.cpu().numpy())

    val_accuracy = (np.array(val_pred) == np.array(val_true)).mean()

    epoch_result.append((avg_train_loss, train_accuracy, val_accuracy))


results = []
for idx, (loss, train_acc, val_acc) in enumerate(epoch_result, start=1):
    print(f"Epoch {idx}: Train loss: {loss:.4f}, Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}")
    results.append({"Epoch": idx, "Train Loss": loss, "Train Accuracy": train_acc, "Validation Accuracy": val_acc})

log_df = pd.DataFrame(results)
log_df.to_csv("training_log.csv", index=False)
print("\n✅ 학습 로그 저장 완료: training_log.csv")


plt.figure(figsize=(8, 5))
plt.plot(log_df['Epoch'], log_df['Train Accuracy'], label='Train Accuracy', marker='o')
plt.plot(log_df['Epoch'], log_df['Validation Accuracy'], label='Validation Accuracy', marker='o')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_curve.png")
plt.show()


print("\n📦 모델 저장 중...")
save_path = "mobilebert_custom_model_review"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("✅ 모델 저장 완료:", save_path)
