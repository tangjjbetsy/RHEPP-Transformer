from model import *
from main import *
from config import *
from torch import optim
from transformers import BertConfig
from torch.utils.data import DataLoader, Dataset
from data_preprocessing import *
from pretty_midi import PrettyMIDI

import os
import sys
import copy
import numpy as np
import itertools
import pytorch_lightning as pl

def generate(model, input_file, performer, output_file, full=True):
    tokenizer = load_tokenizer(PATH_TO_TOKENIZER)
    tokens = tokenize_midi(input_file,
                    performer,
                    tokenizer)
    tokens = np.concatenate([tokens, np.zeros((len(tokens), 1))], axis=1)
    print(tokens.shape)
    for i in range(len(tokens)):
        if i < len(tokens) - 1:
            tokens[i][6] = tokens[i+1][4]*1536 + tokens[i+1][3] - (tokens[i][4]*1536 + tokens[i][3])
        tokens[i][1] = 0
    
    if (full == False) & (len(tokens) > 1000):
        tokens = tokens[0:1000]
    token_list = []
        
    if len(tokens) < 1000:
        seq_len = len(tokens)
        tokens = np.concatenate([tokens, np.ones((MAX_LEN-seq_len, 7)) * 0])[0]
        token_list.append((tokens, seq_len))
    else:
        n = len(tokens) // 1000
        i = 0
        while i < n:
            token_list.append((tokens[i*1000:(i+1)*1000], 1000))
            i += 1
        seq_len = len(tokens[n*1000:])
        final = np.concatenate([tokens[n*1000:], np.ones((MAX_LEN-seq_len, 7)) * 0])
        token_list.append((final, seq_len))
    
    performance = []
    
    model.eval()
    for k in range(len(token_list)):
        seq_len = token_list[k][1]
        tokens = token_list[k][0]
    
        with torch.no_grad():
            inputs = torch.LongTensor(tokens)[None, ..., [0,1,2,3,4,6]]
            mask = np.ones(MAX_LEN)
            mask[seq_len:] = 0
            mask = torch.LongTensor(mask)[None, ...]
            performer = torch.LongTensor([performer])
            
            print(inputs.shape)
            print(mask.shape)
            print(performer.shape)
            outputs = model(inputs, mask, performer)
            outputs = [torch.ceil(i.squeeze()).long() for i in outputs]
            
            if k == 0:
                diff = outputs[2][-1]
                outputs[2] = torch.cumsum(outputs[2], dim=-1) + inputs.squeeze()[:, 3][0] + (inputs.squeeze()[:, 4][0] - 1) * 1536
                outputs[2] = torch.cat([inputs.squeeze()[:, 3][None, 0] + inputs.squeeze()[:, 4][None, 0]*1536, outputs[2][0:-1]])
            else:
                outputs[2] = torch.cat([diff[None], outputs[2][0:-1]])
                diff = outputs[2][-1]
                outputs[2] = torch.cumsum(outputs[2], dim=-1) + np.asarray(performance)[:, 3][-1] + (np.asarray(performance)[:, 4][-1] - 1) * 1536
            
            outputs[1] = torch.clamp(outputs[1] + inputs.squeeze()[:, 2], 0, 4608)
            
            generation = torch.stack([inputs.squeeze()[:, 0], 
                                    outputs[0],
                                    outputs[1],
                                    outputs[2] % 1536,
                                    outputs[2] // 1536 + 1],
                                    dim=-1).tolist()[0:seq_len]
            
            performance += generation
            
    token2midi([performance], tokenizer, output_file)
    
def get_args():
    parser = argparse.ArgumentParser(description='')
    ### parameter setting ###
    parser.add_argument('--max_seq_len', type=int, default=MAX_LEN, help='all sequences are padded to `max_seq_len`')
    parser.add_argument('--ckpt_path', type=str, default=CKPT_PATH)
    parser.add_argument('--input_file', type=str, default=None)
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--performer', type=int, default=0)
    parser.add_argument('--hs', type=int, default=128)
    ### cuda ###
    parser.add_argument("--cuda_devices", nargs='+', default=["0","1"], help="CUDA device ids")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.cuda_devices)
    
    configuration = BertConfig(
                            max_position_embeddings=args.max_seq_len,
                            position_embedding_type='relative_key_query',
                            hidden_size=args.hs,
                            num_hidden_layers=4,
                            precision = 16,
                            num_attention_heads=4,
                            intermediate_size=128,
                            )
    
    bertmodel = MidiBert(configuration)    
    mymodel = MidiBertLM(bertmodel)
    model = MyLightningModule(mymodel)
    
    sys.path.append(args.ckpt_path)
    checkpoint = torch.load(args.ckpt_path)
    model.load_state_dict(checkpoint)
    
    generate(model, args.input_file, args.performer, args.output_file)