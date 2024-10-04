import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, AutoModel, DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ProjectionModel(nn.Module):
    def __init__(self, config):
        super(ProjectionModel, self).__init__()
        self.config = config
        self.c_code_encoder = AutoModel.from_pretrained("microsoft/codebert-base")
        self.pseudocode_encoder = AutoModel.from_pretrained("microsoft/codebert-base")

        self.projection = nn.Sequential(
            nn.Linear(config['embedding_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'], config['embedding_dim'])
        )

    def forward(self, c_code_inputs, pseudocode_inputs):
        c_code_embedding = self.c_code_encoder(**c_code_inputs).last_hidden_state.mean(dim=1)
        pseudocode_embedding = self.pseudocode_encoder(**pseudocode_inputs).last_hidden_state.mean(dim=1)
        projected_pseudocode_embedding = self.projection(pseudocode_embedding)
        return c_code_embedding, projected_pseudocode_embedding

def compute_cosine_similarity(a, b):
    a, b = a.to(device), b.to(device)
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    return F.cosine_similarity(a, b)

def calculate_metrics(predicted, true):
    accuracy = accuracy_score(true, predicted)
    precision = precision_score(true, predicted, average='weighted', zero_division=0)
    recall = recall_score(true, predicted, average='weighted', zero_division=0)
    f1 = f1_score(true, predicted, average='weighted', zero_division=0)
    return accuracy, precision, recall, f1

def test_with_projection(query_c_code, c_code_encoder, tokenizer, rag_pseudocode_snippets, rag_pseudocode_embeddings):
    query_c_code_inputs = tokenizer(query_c_code, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        query_c_code_embedding = c_code_encoder(**query_c_code_inputs).last_hidden_state.mean(dim=1)
        query_transformed_embedding = model.projection(query_c_code_embedding)

    similarity_scores = [compute_cosine_similarity(query_transformed_embedding, rag_embedding.to(device)).item() for rag_embedding in rag_pseudocode_embeddings]
    top_index = torch.argmax(torch.tensor(similarity_scores).to(device))
    return rag_pseudocode_snippets[top_index]

def test_without_projection(query_c_code, base_model, base_tokenizer, rag_pseudocode_snippets, rag_pseudocode_embeddings_base):
    c_code_inputs = base_tokenizer(query_c_code, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        c_code_embedding = base_model(**c_code_inputs).last_hidden_state.mean(dim=1)

    similarity_scores = [compute_cosine_similarity(c_code_embedding, rag_embedding.to(device)).item() for rag_embedding in rag_pseudocode_embeddings_base]
    top_index = torch.argmax(torch.tensor(similarity_scores).to(device))
    return rag_pseudocode_snippets[top_index]

def test_with_bm25(query_c_code, bm25, rag_pseudocode_snippets):
    query_tokens = query_c_code.split()
    bm25_scores = bm25.get_scores(query_tokens)
    top_index = torch.argmax(torch.tensor(bm25_scores))
    return rag_pseudocode_snippets[top_index]

def test_with_sentence_transformer(query_c_code, st_model, rag_pseudocode_snippets, rag_pseudocode_embeddings_st):
    query_embedding = st_model.encode(query_c_code, convert_to_tensor=True).to(device)
    similarity_scores = [compute_cosine_similarity(query_embedding, rag_embedding.to(device)).item() for rag_embedding in rag_pseudocode_embeddings_st]
    top_index = torch.argmax(torch.tensor(similarity_scores).to(device))
    return rag_pseudocode_snippets[top_index]

def test_with_dpr(query_c_code, dpr_question_encoder, dpr_question_tokenizer, rag_pseudocode_snippets, rag_pseudocode_embeddings_dpr):
    question_inputs = dpr_question_tokenizer(query_c_code, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        question_embedding = dpr_question_encoder(**question_inputs).pooler_output

    similarity_scores = [compute_cosine_similarity(question_embedding, rag_embedding.to(device)).item() for rag_embedding in rag_pseudocode_embeddings_dpr]
    top_index = torch.argmax(torch.tensor(similarity_scores).to(device))
    return rag_pseudocode_snippets[top_index]

# Load models
model_name = "aircrypto/code-llama-7b-projection-largev5.2"
config_file = hf_hub_download(repo_id=model_name, filename="config.json")
with open(config_file, 'r') as f:
    config_dict = json.load(f)

model = ProjectionModel(config_dict)
model_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
state_dict = torch.load(model_path, map_location="cpu")
model.load_state_dict(state_dict)
model = model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModel.from_pretrained("microsoft/codebert-base").to(device)
base_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
st_model = SentenceTransformer('all-mpnet-base-v2').to(device)

dpr_question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base').to(device)
dpr_question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
dpr_context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base').to(device)
dpr_context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

c_code_encoder = model.c_code_encoder.to(device)

# Load dataset and prepare embeddings
new_dataset = load_dataset("aircrypto/English-French-Translations-Test", split='train')
new_dataset = new_dataset.select(range(250))

rag_pseudocode_snippets = new_dataset['french']

tokenized_snippets = [snippet.split() for snippet in rag_pseudocode_snippets]
bm25 = BM25Okapi(tokenized_snippets)

rag_pseudocode_embeddings = []
rag_pseudocode_embeddings_base = []
rag_pseudocode_embeddings_st = []
rag_pseudocode_embeddings_dpr = []

for code in rag_pseudocode_snippets:
    code_inputs = tokenizer(code, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        code_embedding = c_code_encoder(**code_inputs).last_hidden_state.mean(dim=1)
    rag_pseudocode_embeddings.append(code_embedding)

    code_inputs_base = base_tokenizer(code, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        code_embedding_base = base_model(**code_inputs_base).last_hidden_state.mean(dim=1)
    rag_pseudocode_embeddings_base.append(code_embedding_base)

    code_embedding_st = st_model.encode(code, convert_to_tensor=True).to(device)
    rag_pseudocode_embeddings_st.append(code_embedding_st)

    ctx_inputs = dpr_context_tokenizer(code, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        ctx_embedding = dpr_context_encoder(**ctx_inputs).pooler_output
    rag_pseudocode_embeddings_dpr.append(ctx_embedding)

# Test and evaluate models
true_labels = []
predictions_with_projection = []
predictions_without_projection = []
predictions_with_bm25 = []
predictions_with_st = []
predictions_with_dpr = []

total_time_projection = 0.0
total_time_without_projection = 0.0
total_time_bm25 = 0.0
total_time_st = 0.0
total_time_dpr = 0.0

num_queries = 0

for item in new_dataset:
    if item['label'] == 0:
        continue

    query_c_code = item['english']
    true_pseudocode = item['french']
    num_queries += 1

    start_time = time.time()
    predicted_with_projection = test_with_projection(query_c_code, c_code_encoder, tokenizer, rag_pseudocode_snippets, rag_pseudocode_embeddings)
    total_time_projection += (time.time() - start_time)
    predictions_with_projection.append(predicted_with_projection)

    start_time = time.time()
    predicted_without_projection = test_without_projection(query_c_code, base_model, base_tokenizer, rag_pseudocode_snippets, rag_pseudocode_embeddings_base)
    total_time_without_projection += (time.time() - start_time)
    predictions_without_projection.append(predicted_without_projection)

    start_time = time.time()
    predicted_with_bm25 = test_with_bm25(query_c_code, bm25, rag_pseudocode_snippets)
    total_time_bm25 += (time.time() - start_time)
    predictions_with_bm25.append(predicted_with_bm25)

    start_time = time.time()
    predicted_with_st = test_with_sentence_transformer(query_c_code, st_model, rag_pseudocode_snippets, rag_pseudocode_embeddings_st)
    total_time_st += (time.time() - start_time)
    predictions_with_st.append(predicted_with_st)

    start_time = time.time()
    predicted_with_dpr = test_with_dpr(query_c_code, dpr_question_encoder, dpr_question_tokenizer, rag_pseudocode_snippets, rag_pseudocode_embeddings_dpr)
    total_time_dpr += (time.time() - start_time)
    predictions_with_dpr.append(predicted_with_dpr)

    true_labels.append(true_pseudocode)

# Calculate and print metrics
metrics_with_projection = calculate_metrics(predictions_with_projection, true_labels)
metrics_without_projection = calculate_metrics(predictions_without_projection, true_labels)
metrics_with_bm25 = calculate_metrics(predictions_with_bm25, true_labels)
metrics_with_st = calculate_metrics(predictions_with_st, true_labels)
metrics_with_dpr = calculate_metrics(predictions_with_dpr, true_labels)

avg_time_projection = total_time_projection / num_queries
avg_time_without_projection = total_time_without_projection / num_queries
avg_time_bm25 = total_time_bm25 / num_queries
avg_time_st = total_time_st / num_queries
avg_time_dpr = total_time_dpr / num_queries

def print_metrics(name, metrics, avg_time):
    accuracy, precision, recall, f1 = metrics
    print(f"\n{name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Average Time per Query: {avg_time:.4f} seconds")

print_metrics("Model with Projection", metrics_with_projection, avg_time_projection)
print_metrics("CodeBERT without Projection", metrics_without_projection, avg_time_without_projection)
print_metrics("BM25 Model", metrics_with_bm25, avg_time_bm25)
print_metrics("Sentence Transformer Model", metrics_with_st, avg_time_st)
print_metrics("DPR Model", metrics_with_dpr, avg_time_dpr)
