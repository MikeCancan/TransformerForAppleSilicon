from datasets import load_dataset
from torch.utils.data import Dataset
import torch
import torch.nn as nn
'''
The purpose of the file

1. feature extracting
2. PAD,EOS,SOS padding
3. tokenizer??
'''
class LanguageSet(Dataset):
    def __init__(self,ds,tokenizer_src,tokenizer_tgt,src_lan,tgt_lan,seq_len):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lan = src_lan
        self.tgt_lan = tgt_lan
        self.seq_len = seq_len

        # SOS for begining, EOS for ending, PAD for padding
        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')],dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')],dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')],dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self,idx):
        text_pair = self.ds[idx]
        src_text = text_pair['translation'][self.src_lan]
        tgt_text = text_pair['translation'][self.tgt_lan]

        #pass the ids as the token by using tokenizer machine and the encode function

        enc_input_token = self.tokenizer_src.encode(src_text).ids
        dec_input_token = self.tokenizer_tgt.encode(tgt_text).ids

        #Calculate the num of padding tokens
        enc_padding_num = max(0,self.seq_len - len(enc_input_token) - 2)
        dec_padding_num = max(0,self.seq_len - len(dec_input_token) - 1)

        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_token[:self.seq_len-2]),
            self.eos_token,
            self.pad_token.repeat(max(0,enc_padding_num))
        ])

        decoder_input = torch.cat(
            [
                torch.tensor(dec_input_token[:self.seq_len-1]),
                self.eos_token,
                self.pad_token.repeat(max(0,self.seq_len-len(dec_input_token)-1))
            ]
        )
        label = torch.cat(
            [
                torch.tensor(dec_input_token,dtype=torch.int64),
                self.eos_token,
                torch.tensor(self.pad_token.repeat(dec_padding_num),dtype=torch.int64)
            ]
        )
        future_mask = torch.triu(torch.ones(self.seq_len,self.seq_len),diagonal=1).int()
        return{
            "encoder_input":encoder_input,
            "decoder_input":decoder_input,
            "src_text":src_text,
            "tgt_text":tgt_text,
            "label":label,
            "encoder_mask":(encoder_input!=self.pad_token).unsqueeze(0).unsqueeze(0).int()&future_mask,
            "decoder_mask":(decoder_input!=self.pad_token).unsqueeze(0).unsqueeze(0).int()&future_mask
        }



