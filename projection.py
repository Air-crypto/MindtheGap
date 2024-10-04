import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset
from huggingface_hub import login
import numpy as np
from types import SimpleNamespace
from pytorch_metric_learning.losses import NPairsLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CodeDataset(Dataset):
    def __init__(self, c_code_list, pseudocode_list, label_list):
        self.c_code_list = c_code_list
        self.pseudocode_list = pseudocode_list
        self.label_list = label_list

    def __len__(self):
        return len(self.c_code_list)

    def __getitem__(self, idx):
        c_code = self.c_code_list[idx]
        pseudocode = self.pseudocode_list[idx]
        label = self.label_list[idx]
        return c_code, pseudocode, label

class ProjectionModel(nn.Module):
    def __init__(self, config):
        super(ProjectionModel, self).__init__()
        self.config = config
        self.c_code_encoder = AutoModel.from_pretrained("microsoft/codebert-base")
        self.pseudocode_encoder = AutoModel.from_pretrained("microsoft/codebert-base")

        # Projection network with 1 hidden layer
        self.projection = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim)
        )

    def forward(self, c_code_inputs, pseudocode_inputs):
        c_code_embedding = self.c_code_encoder(**c_code_inputs).last_hidden_state.mean(dim=1)
        pseudocode_embedding = self.pseudocode_encoder(**pseudocode_inputs).last_hidden_state.mean(dim=1)
        projected_pseudocode_embedding = self.projection(pseudocode_embedding)
        return c_code_embedding, projected_pseudocode_embedding
        
class CustomNPairsLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(CustomNPairsLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor_embeddings, positive_embeddings, labels):
        loss = 0.0
        count = 0
        
        for i in range(anchor_embeddings.size(0)):
            if labels[i] == 1:
                positive_distance = F.pairwise_distance(anchor_embeddings[i].unsqueeze(0), positive_embeddings[i].unsqueeze(0))
                
                for j in range(anchor_embeddings.size(0)):
                    if i != j:
                        negative_distance = F.pairwise_distance(anchor_embeddings[i].unsqueeze(0), positive_embeddings[j].unsqueeze(0))
                        loss += F.relu(positive_distance - negative_distance + self.margin)
                        count += 1
        
        return loss / count if count > 0 else torch.tensor(0.0, requires_grad=True).to(anchor_embeddings.device)

# Load the dataset and prepare for training
write_key = 'HF_KEY'
login(write_key)

dataset = load_dataset("aircrypto/English-French-Translations-Train-Large", split='train')

c_code_list = dataset['english']
pseudocode_list = dataset['french']
label_list = dataset['label']

c_code_train, c_code_val, pseudocode_train, pseudocode_val, label_train, label_val = train_test_split(c_code_list, pseudocode_list, label_list, test_size=0.3, random_state=42)

train_dataset = CodeDataset(c_code_train, pseudocode_train, label_train)
val_dataset = CodeDataset(c_code_val, pseudocode_val, label_val)

# Initialize the model, loss function, and optimizer
config_dict = {
    "model_type": "ProjectionModel",
    "embedding_dim": 768,
    "hidden_dim": 2048,
}

config = SimpleNamespace(**config_dict)

model = ProjectionModel(config).to(device)
criterion = CustomNPairsLoss(margin=1.0)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

dataset_size = len(c_code_train)
batch_size = 32

# Training loop
for epoch in range(5):
    model.train()
    running_loss = 0.0

    indices = np.arange(dataset_size)
    np.random.shuffle(indices)

    for i in range(0, dataset_size, batch_size):
        batch_indices = indices[i:i+batch_size]

        c_code_batch = [c_code_train[j] for j in batch_indices]
        pseudocode_batch = [pseudocode_train[j] for j in batch_indices]
        labels = [label_train[j] for j in batch_indices]

        c_code_inputs = tokenizer(c_code_batch, return_tensors='pt', padding=True, truncation=True).to(device)
        positive_inputs = tokenizer(pseudocode_batch, return_tensors='pt', padding=True, truncation=True).to(device)
        labels = torch.tensor(labels).to(device)

        anchor_embeddings = model.c_code_encoder(**c_code_inputs).last_hidden_state.mean(dim=1)
        positive_embeddings = model.pseudocode_encoder(**positive_inputs).last_hidden_state.mean(dim=1)

        valid_indices = [idx for idx, label in enumerate(labels) if label == 1]

        if valid_indices:
            filtered_anchor_embeddings = anchor_embeddings[valid_indices]
            filtered_positive_embeddings = positive_embeddings[valid_indices]
            filtered_labels = torch.tensor([labels[idx] for idx in valid_indices]).to(filtered_anchor_embeddings.device)

            loss = criterion(filtered_anchor_embeddings, filtered_positive_embeddings, filtered_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/5], Loss: {running_loss / len(valid_indices)}')

    # Validation step
    val_indices = np.arange(len(c_code_val))
    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(c_code_val), batch_size):
            batch_indices = val_indices[i:i+batch_size]

            c_code_batch = [c_code_val[j] for j in batch_indices]
            pseudocode_batch = [pseudocode_val[j] for j in batch_indices]
            labels = [label_val[j] for j in batch_indices]

            c_code_inputs = tokenizer(c_code_batch, return_tensors='pt', padding=True, truncation=True).to(device)
            positive_inputs = tokenizer(pseudocode_batch, return_tensors='pt', padding=True, truncation=True).to(device)
            labels = torch.tensor(labels).to(device)

            anchor_embeddings = model.c_code_encoder(**c_code_inputs).last_hidden_state.mean(dim=1)
            positive_embeddings = model.pseudocode_encoder(**positive_inputs).last_hidden_state.mean(dim=1)

            valid_indices = [idx for idx, label in enumerate(labels) if label == 1]

            if valid_indices:
                filtered_anchor_embeddings = anchor_embeddings[valid_indices]
                filtered_positive_embeddings = positive_embeddings[valid_indices]
                filtered_labels = torch.tensor([labels[idx] for idx in valid_indices]).to(filtered_anchor_embeddings.device)

                loss = criterion(filtered_anchor_embeddings, filtered_positive_embeddings, filtered_labels)
                val_loss += loss.item()

    print(f'Validation Loss: {val_loss / (len(val_dataset) // batch_size)}')

print("Training complete!")

# Save and upload the model to Hugging Face Hub
import os
import json
from huggingface_hub import create_repo, upload_folder

model_save_path = "./trained_projection_model"

def save_model(model, tokenizer, save_path):
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), f"{save_path}/pytorch_model.bin")
    tokenizer.save_pretrained(save_path)
    config = {
        "model_type": "ProjectionModel",
        "embedding_dim": 768,
        "hidden_dim": 2048
    }
    with open(f"{save_path}/config.json", "w") as f:
        json.dump(config, f)

save_model(model, tokenizer, model_save_path)

hub_model_id = "aircrypto/code-llama-7b-projection-largev5.2"
create_repo(hub_model_id)
upload_folder(
    folder_path=model_save_path,
    repo_id=hub_model_id,
    repo_type="model"
)

print(f"Model successfully uploaded to {hub_model_id}")
