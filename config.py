from pathlib import Path

def get_config():
    return {
        'batch_size':8,
        'num_epochs':20,
        'lr':0.001,
        'seq_len':128,
        'd_model':512,
        'd_ff':2048,
        'src_lan':'en',
        'tgt_lan':'de',
        'model_folder':'weights',
        'model_name':'TFModel',
        'preload':None,
        'tokenizer_file':'tokenizer_{0}.json',
        'temp':'./test',
        'log_path':'./runs/experiment'
    }

def load_path(config,epoch):
    model_folder = config['model_folder']
    model_name = config['model_name']
    model_filename = f"{model_name}{epoch}.pt"
    return str(Path('.')/model_folder/model_filename)