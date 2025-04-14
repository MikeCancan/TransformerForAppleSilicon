import torch
import torch.nn as nn
from model import build_transformer
from dataset import LanguageSet
from torch.utils.data import DataLoader
from tokenizers import Tokenizer,models,trainers,pre_tokenizers
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.tensorboard import SummaryWriter
from config import get_config,load_path
from pathlib import Path
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from datasets import load_dataset
from tqdm import tqdm
from torch.amp import autocast,GradScaler
def get_sentence(ds,language):
    for item in ds:
        yield item['translation'][language]

def init_tokenizer(config,ds,language):
    tokenizer_path = Path(config['tokenizer_file'].format(language))
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            min_frequency=2,
            special_tokens = ["[UNK]","[SOS]","[PAD]","[EOS]"],
            continuing_subword_prefix = "##"
        )
        tokenizer.train_from_iterator(get_sentence(ds,language),trainer)
        tokenizer.save(str(tokenizer_path))
        return tokenizer
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return tokenizer

def init_ds(config):
    # Get the dataset
    ds_raw = load_dataset("IWSLT/iwslt2017",name="iwslt2017-en-de",split="train")
    #Get the tokenizer
    src_tokenizer = init_tokenizer(config,ds_raw,config["src_lan"])
    tgt_tokenizer = init_tokenizer(config,ds_raw,config["tgt_lan"])

    train_size = int(0.9 * len(ds_raw))
    test_size = len(ds_raw) - train_size

    split_ds = ds_raw.train_test_split(test_size=test_size)
    train_ds = split_ds["train"]
    test_ds = split_ds["test"]

    train_ds = LanguageSet(train_ds,src_tokenizer,tgt_tokenizer,config["src_lan"],config["tgt_lan"],config["seq_len"])
    test_ds = LanguageSet(test_ds,src_tokenizer,tgt_tokenizer,config["src_lan"],config["tgt_lan"],config["seq_len"])

    train_loader = DataLoader(train_ds,config["batch_size"],shuffle=True)
    test_loader = DataLoader(test_ds,config["batch_size"],shuffle=True)

    return train_loader,test_loader,src_tokenizer,tgt_tokenizer

def init_model(config,src_vocab,tgt_vocab):
    model = build_transformer(src_vocab,tgt_vocab,config["seq_len"],config["seq_len"])
    return model

def validate_model(model, test_loader, criterion, writer, epoch, device):
    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            label = batch["label"].to(device)

            with autocast(device_type='mps'):
                encoder_output = model.encode(encoder_input, encoder_mask)
                decoder_output = model.decode(encoder_output, decoder_input, encoder_mask, decoder_mask)
                proj_output = model.project(decoder_output)

                loss = criterion(proj_output.view(-1, proj_output.size(-1)), label.view(-1))
                total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(test_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")

    # Log validation loss
    writer.add_scalar("Loss/val", avg_val_loss, epoch)


def train_model(config):
    device = torch.device('mps')
    train_loader,test_loader,src_tokenizer,tgt_tokenizer = init_ds(config)

    src_vocab = len(src_tokenizer.get_vocab())
    tgt_vocab = len(tgt_tokenizer.get_vocab())
    model = init_model(config,src_vocab,tgt_vocab).to(device)
    scaler = GradScaler()

    criterion = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id("[PAD]"))
    optimizer =  torch.optim.Adam(model.parameters(),lr=config["lr"])

    global_step = 0
    writer = SummaryWriter(log_dir="runs/transformer_experiment")

    for epoch in range(1):
        model.train()
        batch_iterator = tqdm(train_loader)
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)

            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)

            label = batch['label'].to(device)

            with autocast(device_type='mps'):
                encoder_output = model.encode(encoder_input,encoder_mask)
                decoder_output = model.decode(encoder_output,decoder_input,encoder_mask,decoder_mask)
                proj_output = model.project(decoder_output)

                loss = criterion(proj_output.view(-1,tgt_tokenizer.get_vocab_size()),label.view(-1))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            writer.add_scalar('train_loss',loss.item(),global_step=global_step)
            writer.flush()
            global_step+=1
        validate_model(model,test_loader,criterion,writer,epoch,device=device)

        model_filename = load_path(config,epoch)
        
        torch.save(
            {
                'epoch':epoch,
                "model_state_dict":model.state_dict(),
                "optimizer_state_dict":optimizer.state_dict(),
                'global_step':global_step
            }
        ),model_filename
    writer.close()
if __name__ == '__main__':
    config = get_config()
    train_model(config)