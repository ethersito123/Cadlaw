import os
import warnings
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from model import build_transformer     # Tu MoE Cadlaw sale de aquí
from config import get_config, get_weights_file_path, latest_weights_file_path


# ----------------------------
# Dataset Causal Autoregresivo
# ----------------------------

class LLMPretrainDataset(Dataset):
    def __init__(self, text_list, tokenizer, seq_len):
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        self.data = []
        for text in text_list:
            ids = tokenizer.encode(text).ids
            if len(ids) > seq_len:
                # cortar en ventanas
                for i in range(0, len(ids) - seq_len, seq_len):
                    self.data.append(ids[i:i + seq_len])
            else:
                padded = ids + [tokenizer.token_to_id("[PAD]")] * (seq_len - len(ids))
                self.data.append(padded)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx][:-1], dtype=torch.long)
        y = torch.tensor(self.data[idx][1:], dtype=torch.long)
        return x, y


# ----------------------------
# Tokenizer loader/trainer
# ----------------------------

def get_or_build_tokenizer(config, dataset_text):
    tokenizer_path = Path(config["tokenizer_file"])

    if not tokenizer_path.exists():
        print("Tokenizer not found. Training new tokenizer...")

        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = WordLevelTrainer(
            special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
            min_frequency=2
        )

        tokenizer.train_from_iterator(dataset_text, trainer)
        tokenizer.save(str(tokenizer_path))

    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


# ----------------------------
# Build model
# ----------------------------

def get_model(config, vocab_size):
    model = build_transformer(
        vocab_size=vocab_size,
        max_seq_len=config["seq_len"],
        dim=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        moe_experts=config["n_routed_experts"],
        moe_inter_dim=config["moe_inter_dim"],
        expert_top_k=config["n_activated_experts"]
    )
    return model


# ----------------------------
# Main Train Loop
# ----------------------------

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Load dataset
    # ----------------------------
    raw = load_dataset(config["datasource"], split="train")  # Ej: "wikipedia"
    text_list = [x["text"] for x in raw]

    # ----------------------------
    # Tokenizer
    # ----------------------------
    tokenizer = get_or_build_tokenizer(config, text_list)
    vocab_size = tokenizer.get_vocab_size()

    # ----------------------------
    # Dataset / Dataloader
    # ----------------------------
    train_ds = LLMPretrainDataset(
        text_list, tokenizer, config["seq_len"]
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True
    )

    # ----------------------------
    # Build Model
    # ----------------------------
    model = get_model(config, vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"))

    writer = SummaryWriter(config["experiment_name"])

    # ----------------------------
    # Load checkpoint
    # ----------------------------
    global_step = 0
    initial_epoch = 0

    ckpt_path = latest_weights_file_path(config)
    if ckpt_path:
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        initial_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("Training from scratch.")

    # ----------------------------
    # Training loop
    # ----------------------------
    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch, (inp, tgt) in enumerate(progress):
            inp = inp.to(device)
            tgt = tgt.to(device)

            logits = model(inp)
            loss = loss_fn(logits.view(-1, vocab_size), tgt.view(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.flush()

            progress.set_postfix({"loss": f"{loss.item():.4f}"})
            global_step += 1

        # Save checkpoint
        ckpt_name = get_weights_file_path(config, epoch)
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, ckpt_name)

        print(f"Saved checkpoint {ckpt_name}")


# ----------------------------
# Run
# ----------------------------

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
