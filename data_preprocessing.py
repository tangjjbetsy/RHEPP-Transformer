import pprint, os, copy
import numpy as np
import pandas as pd

from config import *
from octuple_performer import OctuplePerformer
from miditoolkit import MidiFile
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

def init_tokenizer(is_quantize=None, res=384):
    pitch_range = range(21,109)
    beat_res = {(0, 4): res, (4, 12): res}
    nb_velocities = 64
    additional_tockens = {'Tempo': False, 
                        'TimeSignature': False,
                        'Chord': False,
                        'Rest': False,
                        'Program': False,
                        'Pedal': False,
                        'Composition': False,
                        'nb_tempos': 64,  # nb of tempo bins
                        'tempo_range': (40, 250)}  # (min, max)

    tokenizer = OctuplePerformer(pitch_range, beat_res, 
                                    nb_velocities, additional_tockens, 
                                    mask=False, sos_eos_tokens=False, 
                                    num_of_performer=NUM_OF_PERFORMERS, is_quantize=is_quantize)
    return tokenizer

def load_tokenizer(path_to_tokenizer):
    tokenizer = np.load(path_to_tokenizer, allow_pickle=True).item()
    return tokenizer

def tokenize_midi(midi_file, performer, tokenizer):
    midi_file = MidiFile(midi_file)
    tokens = tokenizer.midi_to_tokens(midi_file, performer)
    return tokens

def token2midi(tokens, tokenizer, path_midi_output):
    midi_rep = tokenizer.tokens_to_midi(tokens)
    midi_rep.dump(filename=path_midi_output)

def token2event(tokens, tokenizer):
    events = tokenizer.tokens_to_events(tokens)
    return np.asarray(events).squeeze()

def tokenize_one_piece(path, performer_id, res=384, tokenizer=None):
    if tokenizer != None:
        tokenizer = load_tokenizer(tokenizer)
    else:
        tokenizer = init_tokenizer(res=res)
        
    pprint.pprint(tokenizer.vocab)
    tokens = tokenize_midi(path, performer_id, tokenizer)
    return tokens
    
def align_p_and_s_tokens(ref_token, target_token):
    target_list = []
    extra = []
    for i in range(len(ref_token)):
        if i > len(target_token):
            extra.append(i)
            continue
        is_find = False
        
        if i <= 10:
            start = 0
        else:
            start = i - 10
            
        if i <= len(target_token) - 20:
            end = i + 20    
        else:
            end = len(target_token)
            
        for j in range(start, end):
            if (ref_token[i][0] == target_token[j][0]) and \
            (ref_token[i][1] == target_token[j][1]):
                if len(target_list) > 0:
                    if ref_token[i][4] - ref_token[i-1][4] <= 1:
                        if target_token[j][4] * 1537 + target_token[j][3]- \
                            target_list[-1][4]*1537 - target_list[-1][3] > -1536:
                                target_list.append(target_token[j])
                                is_find = True
                                break
                    else:
                        if (target_token[j][4] - target_list[-1][4] > 1) and \
                            (target_token[j][4] - target_list[-1][4] < 10):
                            target_list.append(target_token[j])
                            is_find = True
                            break
                else:
                    target_list.append(target_token[j])
                    is_find = True
                    break
            
        if is_find == False:
            extra.append(i)

    ref_list = [i for j, i in enumerate(ref_token) if j not in extra]
    
    if len(ref_list) == len(target_list):
        if len(extra) > 0:
            print(len(extra))
        return ref_list, target_list
    else:
        print("fail to align two sequences")
        raise ValueError 

def create_dataset_for_s2p_ioi(csv_file=CSV_FILE):
    midi_csv = pd.read_csv(csv_file, header=0)
    tokenizer = init_tokenizer()
    
    p_list = list()
    s_list = list()
    mask_list = list()
    performer_list = list()
    piece_list = list()
    
    for idx, row in tqdm(midi_csv.iterrows(), 
                         total= midi_csv.shape[0], 
                         bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:20}{r_bar}'):
        
        scale_ratios = np.linspace(-0.25, 0.25, 11) + 1
        midi_paths = [os.path.join(os.path.dirname(row['midi_path']), 
                                os.path.basename(row['midi_path']).replace(".mid", "_" + str(ratio) + ".mid")) for ratio in scale_ratios]
        
        for path  in midi_paths:
            p_file = PATH_TO_PERFORMANCE_FOLDER + path
            s_file = PATH_TO_SCORE_FOLDER + path
            
            p_tokens = np.asarray(tokenize_midi(p_file, 
                                            int(row['artist_id']), 
                                            tokenizer)).squeeze()
            try:
                s_tokens = np.asarray(tokenize_midi(s_file, 
                                                int(row['artist_id']), 
                                                tokenizer)).squeeze()
            except:
                print("break")
                continue
            
            # Align Score with Performance    
            s_tokens, p_tokens = align_p_and_s_tokens(s_tokens, p_tokens)
            p_tokens = np.concatenate([p_tokens, np.zeros((len(p_tokens), 1))], axis=1)
            s_tokens = np.concatenate([s_tokens, np.zeros((len(s_tokens), 1))], axis=1)
            
            seq_len = len(p_tokens)
            if seq_len <= MAX_LEN:
                p = copy.deepcopy(p_tokens)
                p = np.concatenate([p, np.ones((MAX_LEN-seq_len, VOCAB_CATEGORIES)) * pad_id])
                
                s = copy.deepcopy(s_tokens)
                s = np.concatenate([s, np.ones((MAX_LEN-seq_len, VOCAB_CATEGORIES)) * pad_id])
                attn_mask = np.concatenate([np.ones(seq_len), np.zeros(MAX_LEN-seq_len)])
                
                
                for i in range(len(p)):
                    # Calculate IOI
                    if i < seq_len - 1:
                        p[i][6] = p[i+1][4]*1536 + p[i+1][3] - (p[i][4]*1536 + p[i][3])
                        s[i][6] = s[i+1][4]*1536 + s[i+1][3] - (s[i][4]*1536 + s[i][3])
                    
                    # Calculate position difference of the note in performance and score (not used)
                    p[i][3] = (p[i][4]*1536 + p[i][3]) - (s[i][4]*1536 + s[i][3])
                    # Calculate duration deviation of the note
                    p[i][2] -= s[i][2]
                
                p_list.append(p)
                s_list.append(s)
                mask_list.append(attn_mask)
                performer_list.append(int(row['artist_id']))
                piece_list.append(row['midi_path'])
        
            else:
                start_index = 0
                p_seq = copy.deepcopy(p_tokens)
                s_seq = copy.deepcopy(s_tokens)
                n = 0
                
                while start_index + MAX_LEN <= seq_len - 1:
                    overlap_length = np.random.randint(50, 100)
                    p = copy.deepcopy(p_seq[start_index:start_index + MAX_LEN])
                    s = copy.deepcopy(s_seq[start_index:start_index + MAX_LEN])
                    start_index += MAX_LEN - overlap_length    
                    attn_mask = np.ones(MAX_LEN)
                    
                    for i in range(len(p)):
                        if i < len(p) - 1:
                            p[i][6] = p[i+1][4]*1536 + p[i+1][3] - (p[i][4]*1536 + p[i][3])
                            s[i][6] = s[i+1][4]*1536 + s[i+1][3] - (s[i][4]*1536 + s[i][3])
                                                    
                        p[i][3] = (p[i][4]*1536 + p[i][3]) - (s[i][4]*1536 + s[i][3])
                        p[i][2] -= s[i][2]
                    
                    p_list.append(p)
                    s_list.append(s)
                    mask_list.append(attn_mask)
                    performer_list.append(int(row['artist_id']))
                    piece_list.append(row['midi_path'])
                    n += 1
                    
                try:
                    p = copy.deepcopy(p_seq[start_index:])
                    s = copy.deepcopy(s_seq[start_index:])
                    
                    end_len = len(p)
                    
                    p = np.concatenate([p, np.ones((MAX_LEN-end_len, VOCAB_CATEGORIES)) * pad_id])
                    s = np.concatenate([s, np.ones((MAX_LEN-end_len, VOCAB_CATEGORIES)) * pad_id])
                    attn_mask = np.concatenate([np.ones(end_len), np.zeros(MAX_LEN-end_len)])
                except:
                    print(end_len)
                    print(start_index)
                    raise ValueError
                
                for i in range(len(p)):
                    if i < end_len - 1:
                        p[i][6] = p[i+1][4]*1536 + p[i+1][3] - (p[i][4]*1536 + p[i][3])
                        s[i][6] = s[i+1][4]*1536 + s[i+1][3] - (s[i][4]*1536 + s[i][3])
                                                
                    p[i][3] = (p[i][4]*1536 + p[i][3]) - (s[i][4]*1536 + s[i][3])
                    p[i][2] -= s[i][2]
                    
                p_list.append(p)
                s_list.append(s)
                mask_list.append(attn_mask)
                performer_list.append(int(row['artist_id']))
                piece_list.append(row['midi_path'])
                
            
    # add to list
    p_final = np.asarray(p_list)
    s_final = np.asarray(s_list)

    performer_list = np.asarray(performer_list)
    mask_final = np.asarray(mask_list)
    piece_list = np.asarray(piece_list)
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=2)
    for train_index, val_index in sss.split(p_final, performer_list):
        train_x, valid_x = s_final[train_index], s_final[val_index]
        train_mask, valid_mask = mask_final[train_index], mask_final[val_index]
        train_y, valid_y = p_final[train_index], p_final[val_index]   
        train_performer, valid_performer = performer_list[train_index], performer_list[val_index]
        train_piece, valid_piece = piece_list[train_index], piece_list[val_index]
    
    np.savez(
        "data/s2p_data.npz", 
        x_train=train_x,
        y_train=train_y,
        mask_train=train_mask,
        x_valid=valid_x,
        y_valid=valid_y,
        performer_train = train_performer,
        performer_valid = valid_performer,
        mask_valid=valid_mask,
        piece_train=train_piece,
        piece_valid=valid_piece
    )
    
    pprint.pprint(tokenizer.vocab)
    np.save("tokenizers/mytokenizer_s2p.npy", tokenizer)

if __name__ == "__main__":
    create_dataset_for_s2p_ioi(CSV_FILE)