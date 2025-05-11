import os
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
import json
import gzip
import time
from datetime import datetime
from datetime import timedelta
import gc
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.profiler import profile, record_function, ProfilerActivity
import torch.distributed as dist
    
# Import wandb for experiment tracking
import wandb


# For reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# Define constants
MAX_SEQ_LEN = 50
MASK_TOKEN = 0  # ID for mask token
PAD_TOKEN = 1   # ID for padding token

def setup_ddp(args):
    if args.local_rank == -1:
        # fallback to env variable if not explicitly passed
        args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(minutes=20))

    device = torch.device("cuda", args.local_rank)
    return device

# Data preprocessing
def preprocess_amazon_data(file_path, min_uc=5, min_sc=5, cache_path="cache/amazon_data.pkl"):
    """
    Preprocess the Amazon Electronics dataset.
    
    Args:
        file_path: Path to the raw dataset file (supports .json and .json.gz)
        min_uc: Minimum user count threshold
        min_sc: Minimum item count threshold
    
    Returns:
        Preprocessed data with user-item interactions
    """
    # Ensure cache folder exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path} …")
        with open(cache_path, "rb") as f:
            user_sequences, num_items = pickle.load(f)
        return user_sequences, num_items
    
    print(f"Reading raw data from {file_path}...")
    start_time = time.time()
    
    # Load raw JSON lines
    if file_path.endswith(".gz"):
        data = []
        with gzip.open(file_path, "rt") as f:
            for line in f:
                data.append(json.loads(line))
        df = pd.DataFrame(data)
    else:
        df = pd.read_json(file_path, lines=True)

    print(f"Loaded raw data in {time.time() - start_time:.2f}s")
    
    # Support for both .json and .json.gz files
    # if file_path.endswith('.gz'):
    #     # Read gzipped JSON file
    #     data = []
    #     with gzip.open(file_path, 'rb') as f:
    #         for line in f:
    #             data.append(json.loads(line))
    #     df = pd.DataFrame(data)
    # else:
    #     # Regular JSON file
    #     df = pd.read_json(file_path, lines=True)
    
    # print(f"Data loading completed in {time.time() - start_time:.2f} seconds")
    
    # Select relevant columns
    df = df[['reviewerID', 'asin', 'unixReviewTime']]
    df.columns = ['user_id', 'item_id', 'timestamp']
    
    # Convert to numeric IDs
    user_counts = df['user_id'].value_counts()
    item_counts = df['item_id'].value_counts()
    
    # Filter out users and items with fewer interactions than thresholds
    user_mask = user_counts >= min_uc
    item_mask = item_counts >= min_sc
    
    valid_users = user_counts[user_mask].index
    valid_items = item_counts[item_mask].index
    
    df = df[df['user_id'].isin(valid_users) & df['item_id'].isin(valid_items)]
    
    # Create user and item mappings
    user_map = {u: i+2 for i, u in enumerate(valid_users)}  # +2 because 0,1 are special tokens
    item_map = {i: j+2 for j, i in enumerate(valid_items)}  # +2 because 0,1 are special tokens
    
    # Replace IDs with mapped IDs
    df['user_id'] = df['user_id'].map(user_map)
    df['item_id'] = df['item_id'].map(item_map)
    
    # Sort by user ID and timestamp
    df = df.sort_values(['user_id', 'timestamp'])
    
    # Get user sequences
    user_sequences = []
    
    for user_id, group in tqdm(df.groupby('user_id')):
        items = list(group['item_id'])
        if len(items) > 2:  # At least 3 interactions
            user_sequences.append(items)
    
    num_users = len(user_sequences)
    num_items = len(valid_items) + 2  # +2 for mask and pad tokens
    
    print(f"Number of users: {num_users}")
    print(f"Number of items: {num_items}")

    # Cache the result for next time
    with open(cache_path, "wb") as f:
        pickle.dump((user_sequences, num_items), f)
        print(f"Cached processed data to {cache_path}")
    
    return user_sequences, num_items


class BERTRec4Dataset(Dataset):
    def __init__(self, user_sequences, num_items, max_len=MAX_SEQ_LEN, mask_prob=0.15):
        self.user_sequences = user_sequences
        self.num_items = num_items
        self.max_len = max_len
        self.mask_prob = mask_prob
        
    def __len__(self):
        return len(self.user_sequences)
    
    def __getitem__(self, index):
        seq = self.user_sequences[index]

        # 1️⃣ Pad / truncate so every sample is exactly max_len long
        if len(seq) > self.max_len:
            seq = seq[-self.max_len:]                     # most‑recent items
        else:
            seq = [PAD_TOKEN] * (self.max_len - len(seq)) + seq

        # Labels are the original (un‑masked) sequence
        labels = seq.copy()
        masked_seq = seq.copy()

        # 2️⃣ Always mask the *last* non‑PAD token ------------------------------
        last_idx = max(i for i, v in enumerate(seq) if v != PAD_TOKEN)
        masked_seq[last_idx] = MASK_TOKEN          # force‑mask
        # NOTE: `labels[last_idx]` is already the ground‑truth item.

        # 3️⃣ Randomly mask additional tokens (but never PAD or the last idx) ---
        for i in range(self.max_len):
            if i == last_idx or seq[i] == PAD_TOKEN:
                continue
            if random.random() < self.mask_prob:   # e.g. 0.20
                masked_seq[i] = MASK_TOKEN

        return (
            torch.tensor(masked_seq, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


# 3. Improved PositionalEncoding with dropout
class ImprovedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50, dropout=0.1):
        super(ImprovedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# 2. Enhanced BERT4Rec model with modifications
class EnhancedBERT4RecModule(nn.Module):
    def __init__(self, num_items, hidden_size, num_heads, num_layers, dropout=0.2, max_len=50):
        super(EnhancedBERT4RecModule, self).__init__()
        self.num_items = num_items
        self.hidden_size = hidden_size
        
        # Item embedding layer with larger embedding size
        self.item_embeddings = nn.Embedding(num_items, hidden_size, padding_idx=PAD_TOKEN)
        
        # Position encoding with dropout
        self.position_encoding = ImprovedPositionalEncoding(hidden_size, max_len, dropout)
        
        # Layer normalization before transformer
        self.pre_encoder_norm = nn.LayerNorm(hidden_size)
        
        # Transformer encoder layers with enhanced architecture
        encoder_layers = []
        for _ in range(num_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size*4,  # Larger feedforward dimension
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            encoder_layers.append(layer)
        
        self.transformer_layers = nn.ModuleList(encoder_layers)
        
        # Layer normalization after transformer
        self.post_encoder_norm = nn.LayerNorm(hidden_size)
        
        # Output layers with gradual dimension reduction
        self.output_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_items)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def forward(self, input_ids):
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != PAD_TOKEN).float()
        
        # Get embeddings and add positional encoding
        embeddings = self.item_embeddings(input_ids)
        embeddings = self.position_encoding(embeddings)
        
        # Apply pre-encoder normalization
        embeddings = self.pre_encoder_norm(embeddings)
        
        # Create attention mask for transformer (False = don't mask, True = mask)
        transformer_mask = ~attention_mask.bool()
        
        
        # Apply transformer layers with residual connections
        x = embeddings
        for layer in self.transformer_layers:
            x = layer(x, src_key_padding_mask=transformer_mask)
        
        # Apply post-encoder normalization
        x = self.post_encoder_norm(x)
        
        # Get predictions through output network
        predictions = self.output_ffn(x)
        
        return predictions


# 4. Enhanced masking strategy for more robust training
class AdvancedBERT4RecDataset(Dataset):
    def __init__(self, user_sequences, num_items, max_len=50, mask_prob=0.20, mask_token=0, pad_token=1):
        self.user_sequences = user_sequences
        self.num_items = num_items
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.pad_token = pad_token
        
    def __len__(self):
        return len(self.user_sequences)
    
    def __getitem__(self, index):
        seq = self.user_sequences[index]

        # Truncate sequence if needed
        if len(seq) > self.max_len:
            # Take the most recent items
            seq = seq[-self.max_len:]
        else:
            # Pad with PAD tokens to reach max_len
            seq = [self.pad_token] * (self.max_len - len(seq)) + seq

        # Save original sequence for labels
        labels = seq.copy()
        masked_seq = seq.copy()

        # Get positions of real items (not padding)
        real_items = [i for i, item in enumerate(seq) if item != self.pad_token]
        
        if real_items:
            # Always mask the last item for next-item prediction
            last_idx = real_items[-1]
            masked_seq[last_idx] = self.mask_token
            
            # Progressive masking - higher probability for items closer to the target item
            for i in real_items[:-1]:
                # Calculate distance-based masking probability
                # Items closer to the target have higher masking probability
                distance_to_last = last_idx - i
                position_factor = 1.0 - (distance_to_last / len(real_items))
                adjusted_mask_prob = self.mask_prob * (0.8 + 0.4 * position_factor)
                
                if random.random() < adjusted_mask_prob:
                    masked_seq[i] = self.mask_token

        return (
            torch.tensor(masked_seq, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )



def get_model_size(model):
    """Calculate the number of parameters in the model"""
    return sum(p.numel() for p in model.parameters())

def get_model_size_in_millions(model):
    """Calculate the number of parameters in the model in millions"""
    return get_model_size(model) / 1e6


# 1. Enhanced learning rate schedule
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    """
    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine annealing with warm restarts
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


# 5. Focal Loss for handling imbalanced item popularity
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        

# 6. Improved loss calculation with focal loss
def improved_loss_calculation(outputs, labels, input_ids, pad_token=1, gamma=1.0):
    # Calculate loss on all non-padding tokens
    non_pad_mask = (input_ids != pad_token)
    
    # Reshape for cross entropy
    outputs = outputs.view(-1, outputs.size(-1))  # (batch_size * seq_len, num_items)
    labels = labels.view(-1)  # (batch_size * seq_len)
    non_pad_mask = non_pad_mask.view(-1)  # (batch_size * seq_len)
    
    # Filter out padding
    outputs = outputs[non_pad_mask]
    labels = labels[non_pad_mask]
    
    # Calculate focal loss
    ce_loss = F.cross_entropy(outputs, labels, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = ((1 - pt) ** gamma) * ce_loss
    
    return focal_loss.mean()


def calculate_loss(outputs, labels, input_ids, mask_token=MASK_TOKEN, pad_token=PAD_TOKEN):
    mask = (input_ids == mask_token) & (input_ids != pad_token)
    mask = mask.view(-1)
    
    outputs = outputs.view(-1, outputs.size(-1))
    labels = labels.view(-1)
    
    outputs = outputs[mask]
    labels = labels[mask]
    
    return F.cross_entropy(outputs, labels)


def train_one_epoch(model, dataloader, optimizer, scheduler, device, scaler=None, use_amp=False, log_interval=10, use_wandb=False, rank=0):
    model.train()
    total_loss = 0
    batch_times = []
    start_time = time.time()
    
    for batch_idx, (input_ids, labels) in enumerate(tqdm(dataloader) if rank == 0 else dataloader):
        batch_start = time.time()
        input_ids, labels = input_ids.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        if use_amp:
            # Mixed precision training with autocast
            with torch.amp.autocast(device_type="cuda"):
                # Forward pass
                outputs = model(input_ids)
                loss = improved_loss_calculation(outputs, labels, input_ids)
                # loss = calculate_loss(outputs, labels, input_ids)
            
            # Backward pass with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard precision training
            # Forward pass
            outputs = model(input_ids)
            
            # Calculate loss only on masked tokens
            # mask = (input_ids == MASK_TOKEN)
            
            # # Reshape for cross entropy
            # outputs = outputs.view(-1, outputs.size(-1))
            # labels = labels.view(-1)
            # mask = mask.view(-1)
            
            # # Filter predictions and labels based on mask
            # outputs = outputs[mask]
            # labels = labels[mask]
            
            # Calculate loss
            # loss = F.cross_entropy(outputs, labels)
            # loss = calculate_loss(outputs, labels, input_ids)
            loss = improved_loss_calculation(outputs, labels, input_ids)
            
            # Backward pass and optimize
            loss.backward()
            apply_gradient_clipping(model, max_norm=1.0)
            optimizer.step()
        
        # Step scheduler
        scheduler.step()
        
        # Calculate batch time
        batch_end = time.time()
        batch_time = batch_end - batch_start
        batch_times.append(batch_time)
        
        # Log metrics
        total_loss += loss.item()

        # Log to wandb periodically
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            avg_batch_time = sum(batch_times[-log_interval:]) / min(log_interval, len(batch_times))
            samples_per_second = dataloader.batch_size / avg_batch_time
            
            if args.use_wandb and (not args.distributed or args.local_rank == 0):
                wandb.log({
                    "train/batch_loss": loss.item(),
                    "train/running_loss": avg_loss,
                    "train/batch_time": avg_batch_time,
                    "train/samples_per_second": samples_per_second,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/batch": batch_idx,
                })
    
    epoch_time = time.time() - start_time
    avg_loss = total_loss / len(dataloader)
    
    # Log epoch metrics
    if args.use_wandb and (not args.distributed or args.local_rank == 0):
        wandb.log({
            "train/epoch_loss": avg_loss,
            "train/epoch_time": epoch_time,
            "train/avg_batch_time": sum(batch_times) / len(batch_times),
        })
    
    return avg_loss

def evaluate(model, dataloader, device, k=10, use_amp=False):
    model.eval()
    ndcg_scores = []
    hit_scores = []
    mrr_scores = []  # Add Mean Reciprocal Rank
    start_time = time.time()
    inference_times = []
    
    with torch.no_grad():
        for batch_idx, (input_ids, labels) in enumerate(tqdm(dataloader)):
            input_ids, labels = input_ids.to(device), labels.to(device)
            
            # Measure inference time
            inference_start = time.time()

            with torch.amp.autocast(device_type="cuda") if use_amp else torch.no_grad():
                # Forward pass
                outputs = model(input_ids)
            
            # Record inference time
            inference_time = time.time() - inference_start
            inference_times.append(inference_time)
            
            # Get mask positions
            mask_positions = (input_ids == MASK_TOKEN)
            
            # For each sequence in the batch
            for i in range(input_ids.size(0)):
                seq_mask_positions = mask_positions[i]
                
                if not seq_mask_positions.any():
                    continue
                
                # # Get the last masked position (prediction target)
                # last_mask_idx = seq_mask_positions.nonzero()[-1].item()
                
                # # Get the true label
                # true_label = labels[i, last_mask_idx].item()
                
                # # Get prediction scores
                # pred_scores = outputs[i, last_mask_idx, :]
                
                # # Get top-k predictions
                # top_k_items = torch.topk(pred_scores, k=k).indices.tolist()
                
                # # Calculate metrics
                # hit = 1 if true_label in top_k_items else 0
                # hit_scores.append(hit)
                
                # # Calculate NDCG
                # if hit == 1:
                #     ndcg = 1 / math.log2(top_k_items.index(true_label) + 2)
                # else:
                #     ndcg = 0
                # ndcg_scores.append(ndcg)

                # Get all masked indices
                mask_indices = seq_mask_positions.nonzero().squeeze()
                if mask_indices.numel() == 0:
                    continue
                if mask_indices.dim() == 0:  # Single mask case
                    mask_indices = mask_indices.unsqueeze(0)
                
                for mask_idx in mask_indices:
                    mask_idx = mask_idx.item()
                    true_label = labels[i, mask_idx].item()
                    
                    if true_label == PAD_TOKEN:
                        continue
                    
                    pred_scores = outputs[i, mask_idx, :]
                    topk_scores, topk_indices = torch.topk(pred_scores, k=k)
                    rank = (topk_indices == true_label).nonzero(as_tuple=True)[0]
                    
                    hit = 1 if rank.numel() > 0 else 0
                    hit_scores.append(hit)
                    
                    if hit:
                        ndcg = 1 / math.log2(rank.item() + 2)
                        mrr = 1 / (rank.item() + 1)
                    else:
                        ndcg = 0
                        mrr = 0
                        
                    ndcg_scores.append(ndcg)
                    mrr_scores.append(mrr)

    
    # Calculate metrics
    hr_at_k = sum(hit_scores) / len(hit_scores) if hit_scores else 0
    ndcg_at_k = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0
    mrr = sum(mrr_scores)/len(mrr_scores) if mrr_scores else 0
    
    # Calculate inference performance metrics
    total_time = time.time() - start_time
    avg_inference_time = sum(inference_times) / len(inference_times)
    total_batches = len(dataloader)
    total_items = len(dataloader.dataset)
    batches_per_second = total_batches / total_time
    items_per_second = total_items / total_time
    
    # Log metrics to wandb
    if args.use_wandb and (not args.distributed or args.local_rank == 0):
        wandb.log({
            f"eval/HR@{k}": hr_at_k,
            f"eval/NDCG@{k}": ndcg_at_k,
            f"eval/MRR": mrr,
            "eval/total_time": total_time,
            "eval/avg_inference_time": avg_inference_time,
            "eval/batches_per_second": batches_per_second,
            "eval/items_per_second": items_per_second,
        })
    
    metrics = {
        f'HR@{k}': hr_at_k,
        f'NDCG@{k}': ndcg_at_k,
        'MRR': mrr,
        'total_eval_time': total_time,
        'avg_inference_time': avg_inference_time,
        'batches_per_second': batches_per_second,
        'items_per_second': items_per_second
    }
    
    return metrics

def run_pytorch_profiler(model, dataloader, device, use_amp=False, scaler=None):
    """Run PyTorch Profiler on a single batch"""
    print("Running PyTorch Profiler...")
    model.train()
    
    # Get a batch from the dataloader
    input_ids, labels = next(iter(dataloader))
    input_ids, labels = input_ids.to(device), labels.to(device)
    
    # Define the profiler
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True
    ) as prof:
        with record_function("model_inference"):
            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(input_ids)
            else:
                outputs = model(input_ids)
            
            # Calculate loss
            mask = (input_ids == MASK_TOKEN)
            outputs = outputs.view(-1, outputs.size(-1))
            labels = labels.view(-1)
            mask = mask.view(-1)
            outputs = outputs[mask]
            labels = labels[mask]
            loss = F.cross_entropy(outputs, labels)
        
        with record_function("model_backward"):
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
    
    # Print profiler results
    print(prof.key_averages().table(sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total", row_limit=20))
    
    # Export profiler results
    prof.export_chrome_trace(f"profile_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    return prof


def benchmark_inference(model, dataloader, device, use_amp=False, num_batches=100):
    """Benchmark inference performance"""
    print("Benchmarking inference performance...")
    model.eval()
    batch_times = []
    
    with torch.no_grad():
        for i, (input_ids, _) in enumerate(dataloader):
            if i >= num_batches:
                break
                
            input_ids = input_ids.to(device)
            
            # Warmup to avoid initial cuda overhead
            if i == 0:
                if use_amp:
                    with torch.amp.autocast(device_type="cuda"):
                        _ = model(input_ids)
                else:
                    _ = model(input_ids)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                continue
            
            # Benchmark
            start_time = time.time()
            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    _ = model(input_ids)
            else:
                _ = model(input_ids)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            batch_times.append(end_time - start_time)
    
    # Calculate statistics
    avg_time = sum(batch_times) / len(batch_times)
    std_time = np.std(batch_times)
    median_time = np.median(batch_times)
    throughput = dataloader.batch_size / avg_time
    
    # Log to wandb
    if args.use_wandb and (not args.distributed or args.local_rank == 0):
        wandb.log({
            "benchmark/avg_inference_time": avg_time,
            "benchmark/std_inference_time": std_time,
            "benchmark/median_inference_time": median_time,
            "benchmark/throughput_samples_per_second": throughput
        })
    
    print(f"Inference benchmarking results:")
    print(f"  Average time per batch: {avg_time*1000:.2f} ms")
    print(f"  Standard deviation: {std_time*1000:.2f} ms")
    print(f"  Median time: {median_time*1000:.2f} ms")
    print(f"  Throughput: {throughput:.2f} samples/second")
    
    return {
        "avg_time": avg_time,
        "std_time": std_time,
        "median_time": median_time,
        "throughput": throughput
    }


# 6. Improved evaluation function
def improved_evaluate(model, dataloader, device, k=10, use_amp=False):
    model.eval()
    ndcg_scores = []
    hit_scores = []
    mrr_scores = []  # Add Mean Reciprocal Rank
    start_time = time.time()
    inference_times = []
    
    with torch.no_grad():
        for batch_idx, (input_ids, labels) in enumerate(tqdm(dataloader)):
            input_ids, labels = input_ids.to(device), labels.to(device)
            
            # Measure inference time
            inference_start = time.time()
            
            with torch.amp.autocast(device_type="cuda") if use_amp else torch.no_grad():
                outputs = model(input_ids)
            
            # Record inference time
            inference_time = time.time() - inference_start
            inference_times.append(inference_time)
            
            # Get mask positions
            mask_positions = (input_ids == MASK_TOKEN)
            
            # For each sequence in the batch
            for i in range(input_ids.size(0)):
                seq_mask_positions = mask_positions[i]
                
                if not seq_mask_positions.any():
                    continue
                
                # Get the last masked position (prediction target)
                last_mask_idx = seq_mask_positions.nonzero()[-1].item()
                
                # Get the true label
                true_label = labels[i, last_mask_idx].item()
                
                # Skip padding
                if true_label == PAD_TOKEN:
                    continue
                
                # Get prediction scores
                pred_scores = outputs[i, last_mask_idx, :]
                
                # Get top-k predictions
                topk_scores, topk_indices = torch.topk(pred_scores, k=k)
                topk_indices = topk_indices.tolist()
                
                # Calculate metrics
                rank = topk_indices.index(true_label) + 1 if true_label in topk_indices else 0
                
                # Hit Rate
                hit = 1 if rank > 0 else 0
                hit_scores.append(hit)
                
                # NDCG
                ndcg = 1.0 / math.log2(rank + 1) if rank > 0 else 0
                ndcg_scores.append(ndcg)
                
                # MRR (Mean Reciprocal Rank)
                mrr = 1.0 / rank if rank > 0 else 0
                mrr_scores.append(mrr)
    
    # Calculate metrics
    hr_at_k = sum(hit_scores) / len(hit_scores) if hit_scores else 0
    ndcg_at_k = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0
    mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0
    
    # Calculate inference performance metrics
    total_time = time.time() - start_time
    avg_inference_time = sum(inference_times) / len(inference_times)
    total_batches = len(dataloader)
    total_items = len(dataloader.dataset)
    batches_per_second = total_batches / total_time
    items_per_second = total_items / total_time
    
    # Log metrics to wandb
    if args.use_wandb and (not args.distributed or args.local_rank == 0):
        wandb.log({
            f"eval/HR@{k}": hr_at_k,
            f"eval/NDCG@{k}": ndcg_at_k,
            "eval/MRR": mrr,
            "eval/total_time": total_time,
            "eval/avg_inference_time": avg_inference_time,
            "eval/batches_per_second": batches_per_second,
            "eval/items_per_second": items_per_second,
        })
    

    metrics = {
        f'HR@{k}': hr_at_k,
        f'NDCG@{k}': ndcg_at_k,
        'MRR': mrr,
        'total_eval_time': total_time,
        'avg_inference_time': avg_inference_time,
        'batches_per_second': batches_per_second,
        'items_per_second': items_per_second
    }
    
    return metrics


# 8. Improved optimizer with better weight decay and parameter grouping
def create_improved_optimizer(model, lr=5e-4, weight_decay=0.05):
    """
    Create an improved optimizer with parameter-specific learning rates and weight decay
    """
    # Group parameters by type
    no_decay = ["bias", "LayerNorm.weight"]
    
    # Group parameters for differential learning rates
    # Embeddings often need lower learning rates
    embedding_params = [(n, p) for n, p in model.named_parameters() if "embedding" in n]
    other_params = [(n, p) for n, p in model.named_parameters() if "embedding" not in n]
    
    optimizer_grouped_parameters = [
        # Embeddings with weight decay
        {
            "params": [p for n, p in embedding_params if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
            "lr": lr * 0.5  # Lower learning rate for embeddings
        },
        # Embeddings without weight decay
        {
            "params": [p for n, p in embedding_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": lr * 0.5  # Lower learning rate for embeddings
        },
        # Other parameters with weight decay
        {
            "params": [p for n, p in other_params if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        # Other parameters without weight decay
        {
            "params": [p for n, p in other_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    return optimizer



# 9. Gradient clipping function
def apply_gradient_clipping(model, max_norm=1.0):
    """
    Apply gradient clipping to prevent exploding gradients
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


# 7. Function to create the optimized model
def create_optimized_model(num_items, hidden_size=384, num_heads=6, num_layers=4, dropout=0.25):
    """
    Create an optimized BERT4Rec model with better parameters
    """
    model = EnhancedBERT4RecModule(
        num_items=num_items,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout
    )
    return model


def main(args):

    # Setup DDP and device
    if args.distributed:
        device = setup_ddp(args)
        rank = args.local_rank
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0

    try:
        if args.use_wandb and (not args.distributed or args.local_rank == 0):
            # Initialize wandb
            wandb.init(
                project=args.wandb_project_name,
                name=args.wandb_run_name if args.wandb_run_name else f"BERT4Rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=vars(args)
            )
    except Exception as e:
        print(f"wandb init failed: {e}") 
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize AMP scaler if using mixed precision
    scaler = torch.amp.GradScaler(device='cuda') if args.use_amp else None

    
    # Process data
    print("Processing Amazon Electronics dataset...")
    user_sequences, num_items = preprocess_amazon_data(args.data_path, min_uc=args.min_uc, min_sc=args.min_sc)
    
    # Log dataset stats
    if args.use_wandb and (not args.distributed or args.local_rank == 0):
        wandb.log({
            "dataset/num_users": len(user_sequences),
            "dataset/num_items": num_items,
            "dataset/avg_sequence_length": np.mean([len(seq) for seq in user_sequences]),
            "dataset/max_sequence_length": max([len(seq) for seq in user_sequences]),
            "dataset/min_sequence_length": min([len(seq) for seq in user_sequences]),
        })
    
    # Split data
    random.shuffle(user_sequences)
    train_ratio = 0.8
    valid_ratio = 0.1
    test_ratio = 0.1
    
    train_size = int(len(user_sequences) * train_ratio)
    valid_size = int(len(user_sequences) * valid_ratio)
    
    train_sequences = user_sequences[:train_size]
    valid_sequences = user_sequences[train_size:train_size+valid_size]
    test_sequences = user_sequences[train_size+valid_size:]
    
    print(f"Train size: {len(train_sequences)}")
    print(f"Valid size: {len(valid_sequences)}")
    print(f"Test size: {len(test_sequences)}")
    
    # Create datasets
    train_dataset = AdvancedBERT4RecDataset(train_sequences, num_items, max_len=args.max_len, mask_prob=args.mask_prob)
    valid_dataset = AdvancedBERT4RecDataset(valid_sequences, num_items, max_len=args.max_len, mask_prob=args.mask_prob)
    test_dataset = AdvancedBERT4RecDataset(test_sequences, num_items, max_len=args.max_len, mask_prob=args.mask_prob)
    
    train_sampler = DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = DistributedSampler(valid_dataset) if args.distributed else None
    test_sampler = DistributedSampler(test_dataset) if args.distributed else None    

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, shuffle=(train_sampler is None), num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, sampler=valid_sampler, shuffle=False, num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, shuffle=False, num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    
    # Calculate model size parameters to reach ~200M parameters
    # Most parameters are in embeddings and linear layers
    # Parameters = num_items * hidden_size + hidden_size * num_items + transformer parameters
    # Transformer parameters depend on hidden_size, num_heads, num_layers
    
    # We'll use a large hidden size and moderate number of layers to reach ~200M parameters
    # hidden_size = 1024
    # hidden_size = 512
    # hidden_size = 256
    hidden_size = 384
    # num_heads = 16
    # num_heads = 8
    # num_heads = 8
    num_heads = 6
    # num_layers = 8
    # num_layers = 6
    # num_layers = 4
    num_layers = 4


    model = create_optimized_model(num_items=num_items)

    # Calculate and print model size
    model_size_m = get_model_size_in_millions(model)
    print(f"Model size: {model_size_m:.2f}M parameters")
    
    # Log model architecture to wandb
    if args.use_wandb and (not args.distributed or args.local_rank == 0):
        wandb.config.update({
            "model/num_parameters": get_model_size(model),
            "model/num_parameters_millions": model_size_m,
            "model/hidden_size": hidden_size,
            "model/num_heads": num_heads,
            "model/num_layers": num_layers,
            "model/num_items": num_items,
        })
        
        # Create model architecture table
        model_architecture = []
        for name, param in model.named_parameters():
            model_architecture.append([name, param.shape, param.numel()])
        
        if args.use_wandb and (not args.distributed or args.local_rank == 0):
            wandb.log({
                "model/architecture": wandb.Table(
                    data=model_architecture,
                    columns=["Layer Name", "Shape", "Parameters"]
                )
            })
    
    # Move model to device
    model = model.to(device)
    
    if args.distributed:
        model = DDP(
            model, 
            device_ids=[args.local_rank], 
            bucket_cap_mb=10  # FIX: split big All‑Reduce into ~40 MB chunks
        )

    # Create optimizer
    # optimizer = Adam(model.parameters(), lr=args.lr)
    # 1. Improved optimizer with weight decay
    def create_optimizer(model, lr=1e-3, weight_decay=0.01):
        # Use AdamW instead of Adam for better weight decay handling
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(optimizer_grouped_parameters, lr=lr)


    # Create scheduler with warmup
    num_training_steps = len(train_dataloader) * args.epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    
    optimizer = create_improved_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Create a timestamp for saving
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f"models/bert4rec_{timestamp}"
    os.makedirs(save_path, exist_ok=True)
    
    # Run profiler if requested
    if args.run_profiler:
        prof = run_pytorch_profiler(model, train_dataloader, device, args.use_amp, scaler)
        if args.use_wandb and (not args.distributed or args.local_rank == 0):
            wandb.save(f"profile_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    # Run inference benchmarking if requested
    if args.run_benchmark:
        benchmark_results = benchmark_inference(model, test_dataloader, device, args.use_amp, args.benchmark_batches)
    
    # Training loop
    best_valid_hr = 0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_one_epoch(model, train_dataloader, optimizer, scheduler, device, scaler, args.use_amp, args.log_interval, args.use_wandb, rank)
        current_lr = scheduler.get_last_lr()[0]  # Get the last learning rate

        if rank == 0:
            print(f"Train loss: {train_loss:.4f}, LR: {current_lr:.6f}")

        # Memory cleanup
        if args.gc_after_epoch:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Evaluate on validation set
        # valid_metrics = evaluate(model, valid_dataloader, device, k=10, use_amp=args.use_amp)
        # valid_metrics = improved_evaluate(model, valid_dataloader, device, k=10, use_amp=args.use_amp)
        valid_metrics = evaluate(model, valid_dataloader, device, k=10, use_amp=args.use_amp)

        if rank == 0:
            print(f"Valid HR@10: {valid_metrics['HR@10']:.4f}, NDCG@10: {valid_metrics['NDCG@10']:.4f}, MRR: {valid_metrics['MRR']:.4f}")

        # Log epoch number
        if args.use_wandb and (not args.distributed or args.local_rank == 0):
            wandb.log({"epoch": epoch + 1})
        
        # Early stopping
        # Save best model
        if rank == 0:
            if valid_metrics['HR@10'] > best_valid_hr:
                best_valid_hr = valid_metrics['HR@10']
                best_epoch = epoch
                patience_counter = 0
                
                # Save model
                torch.save(model.state_dict(), f"{save_path}/bert4rec_best.pt")
                print(f"Saved best model with HR@10: {best_valid_hr:.4f}")
                
                # Save to wandb
                if args.use_wandb and (not args.distributed or args.local_rank == 0):
                    wandb.log({"best_epoch": epoch + 1, "best_hr@10": best_valid_hr})
                    wandb.save(f"{save_path}/bert4rec_best.pt")
            
            else:
                patience_counter += 1
                if patience_counter >= args.early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break


    if rank == 0:
        print(f"\nTraining complete. Best epoch: {best_epoch+1} with HR@10: {best_valid_hr:.4f}")
    

    if(args.distributed):
        dist.destroy_process_group()


    # Load best model
    model.load_state_dict(torch.load(f"{save_path}/bert4rec_best.pt"))
    
    # Evaluate on test set
    # test_metrics = improved_evaluate(model, test_dataloader, device, k=10, use_amp=args.use_amp)
    test_metrics = evaluate(model, test_dataloader, device, k=10, use_amp=args.use_amp)
    print(f"Test HR@10: {test_metrics['HR@10']:.4f}, NDCG@10: {test_metrics['NDCG@10']:.4f}")
    
    # Log final test metrics
    if args.use_wandb and (not args.distributed or args.local_rank == 0):
        wandb.log({
            "test/HR@10": test_metrics['HR@10'],
            "test/NDCG@10": test_metrics['NDCG@10'],
            "test/final_epoch": args.epochs,
        })
    
    # Run final benchmark on best model
    if args.final_benchmark:
        print("Running final benchmark on best model...")
        final_benchmark = benchmark_inference(model, test_dataloader, device, args.use_amp, args.benchmark_batches)
        
        if args.use_wandb and (not args.distributed or args.local_rank == 0):
            wandb.log({
                "final_benchmark/avg_inference_time": final_benchmark["avg_time"],
                "final_benchmark/throughput": final_benchmark["throughput"],
            })
    
    # Save final model and configuration
    final_save_path = f"{save_path}/bert4rec_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': args,
        'num_items': num_items,
        'metrics': test_metrics
    }, final_save_path)
    
    print(f"Saved final model and configuration to {final_save_path}")
    
    if args.use_wandb and (not args.distributed or args.local_rank == 0):
        wandb.save(final_save_path)
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train BERT4Rec model')
    
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--local_rank', type=int, default=-1)

    # Data parameters
    parser.add_argument('--data_path', type=str, default='Electronics_5.json.gz', help='Path to the Amazon Electronics dataset')
    parser.add_argument('--min_uc', type=int, default=5, help='Min user count')
    parser.add_argument('--min_sc', type=int, default=5, help='Min item count')
    parser.add_argument('--max_len', type=int, default=50, help='Max sequence length')
    
    # Model parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--mask_prob', type=float, default=0.15, help='Masking probability')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--early_stopping_patience', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    # Logging and profiling
    parser.add_argument('--use_wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb_project_name', type=str, default='BERT4Rec', help='wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='wandb run name')

    # AMP & performance
    parser.add_argument('--use_amp', action='store_true', help='Use mixed precision training (AMP)')
    parser.add_argument('--run_profiler', action='store_true', help='Run PyTorch profiler')
    parser.add_argument('--run_benchmark', action='store_true', help='Run inference benchmark')
    parser.add_argument('--final_benchmark', action='store_true', help='Run final benchmark on best model')
    parser.add_argument('--benchmark_batches', type=int, default=100, help='Number of batches for benchmarking')
    parser.add_argument('--gc_after_epoch', action='store_true', help='Run garbage collection after each epoch')
    parser.add_argument('--log_interval', type=int, default=10, help='WandB log interval')

    args = parser.parse_args()
    
    main(args)
