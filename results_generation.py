import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import sys

from models.model_vqa import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils

from dataset import create_dataset, create_sampler, create_loader, vqa_collate_fn

from scheduler import create_scheduler
from optim import create_optimizer

from dataset.relation_dataset import Relation_val_dataset, pre_caption

import re
import glob

def main(args, config):
    device = torch.device(args.device)
    
    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    unus = ['[unused{}]'.format(x) for x in range(200,800)]
    pos_token = ['@@']
    pos_token.extend([f'[pos_{x}]' for x in range(args.position_res)])
    pos_token.append('##')
    postoken_dict = {}
    tokenizer = BertTokenizer.from_pretrained('configs/vocab.txt')
    for x,y in zip(unus, pos_token):
        un_index = tokenizer.vocab[x]
        tokenizer.vocab[y] = un_index
        postoken_dict[y] = un_index
        _ = tokenizer.vocab.pop(x)
        tokenizer.basic_tokenizer.never_split.add(y)
    postoken_dict.pop('@@')
    postoken_dict.pop('##')
    postoken_index = torch.randn(30522).bool()
    postoken_index[:] = False
    for x in postoken_dict.values():
        postoken_index[x]=True
    
    # data 
    
    dt1 = Relation_val_dataset(root = '', ann_file = config['test_file'], img_res=config['image_res'], replace = False)
    
    samplers = [None]
    col_fn = vqa_collate_fn
    
    batch_size = args.batch_size
    print('batch size ', batch_size)
    
    data_loader = create_loader([dt1],samplers,
                                  batch_size=[batch_size],
                                  num_workers=[4],is_trains=[False], 
                                  collate_fns=[col_fn])[0]

    # model
    
    model = ALBEF(config=config, text_encoder='bert-base-uncased', text_decoder='bert-base-uncased', \
                      tokenizer=tokenizer, postoken_dict = postoken_dict)
    print('Load checkpoint from ...', args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu') 
    state_dict = checkpoint['model']

    # reshape positional embedding to accomodate for image resolution change
    pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
    state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped   

    msg = model.load_state_dict(state_dict,strict=False)  
    print(msg) 

    postoken_dict_rev = {v:k for k,v in postoken_dict.items()}
    postoken_dict_rev[int(tokenizer('@@').input_ids[-1])] = '@@'
    postoken_dict_rev[int(tokenizer('##').input_ids[-1])] = '##'
    
    
    
    model = model.cuda()
    model.eval()
    
    num_beams = args.num_beams
    num_seq = args.num_seq

    with_gt = args.with_gt
        
    save_json = {}
    
    start_point = args.chunk_size*args.chunk
    end_point = args.chunk_size*(args.chunk+1)
    
    print('chunk is ', args.chunk)
    print('start from idx ', start_point)
    print('end with idx ', end_point)
    
    with torch.no_grad():  
        for i,(image, question, answer, weights_, n) in enumerate(data_loader):
            if i < start_point:
                continue
            if i >= end_point:
                break 
            if i % 50 == 0:
                print(i)
                json.dump(save_json, open('{}/{}_round{}_beam{}_numseq{}_{}_chunk{}.json'.format(args.result_dir, args.split, args.round, num_beams, num_seq, args.checkpoint[-6:-4], args.chunk),'w'))
            

            image = image.cuda()
            image_embeds = model.visual_encoder(image) 
            image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

            question_input = tokenizer(question, padding='longest', return_tensors="pt").to(image.device)
            question_output = model.text_encoder(question_input.input_ids, 
                                                attention_mask = question_input.attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,                                    
                                                return_dict = True) 


            question_states = question_output.last_hidden_state.repeat_interleave(num_beams,dim=0)
            question_atts = torch.ones(question_states.size()[:-1],dtype=torch.long).to(question_states.device)
            model_kwargs = {"encoder_hidden_states": question_states, "encoder_attention_mask":question_atts}

            bos_ids = torch.full((image.size(0),1),fill_value=101,device=image.device)

            eos = int(tokenizer('@@').input_ids[-1])

            outputs = model.text_decoder.generate(input_ids=bos_ids,
                                                 max_length=20,
                                                 min_length=1,
                                                 num_beams=num_beams,
                                                 num_return_sequences=num_seq,
                                                 eos_token_id=eos,
                                                 pad_token_id=model.tokenizer.pad_token_id, 
                                                 return_dict_in_generate=True, 
                                                 output_scores=True,
                                                 **model_kwargs)
            sequences1 = outputs['sequences'] # batch*num_seq, 20
            for ii in range(0,image.size(0)):
                save_json[batch_size*i + ii] = {}

                save_json[batch_size*i + ii]['round1_scores'] = outputs['sequences_scores'][ii*num_seq:(ii+1)*num_seq].tolist()
                save_json[batch_size*i + ii]['round1_sequences'] = outputs['sequences'][ii*num_seq:(ii+1)*num_seq].tolist()
            round2_scores = []
            round2_raw_seqs = []
            for seq_idx in range(0,len(sequences1)):
                seq = sequences1[seq_idx]
                image_embeds_seq = image_embeds[seq_idx//num_seq].unsqueeze(0)
                image_atts_seq = image_atts[seq_idx//num_seq].unsqueeze(0)

                seq1 = seq[seq.nonzero()].squeeze(1).unsqueeze(0).cuda()


                num_beams_r2 = 5
                question_states = question_output.last_hidden_state[seq_idx//num_seq].unsqueeze(0).repeat_interleave(num_beams_r2,dim=0)
                question_atts = torch.ones(question_states.size()[:-1],dtype=torch.long).to(question_states.device)
                model_kwargs = {"encoder_hidden_states": question_states, "encoder_attention_mask":question_atts}

                max_len = 30
                outputs = model.text_decoder.generate(input_ids=seq1,
                                                     max_length=max_len,
                                                     min_length=1,
                                                     num_beams=num_beams_r2,
                                                     num_return_sequences=1,
                                                     eos_token_id=model.tokenizer.sep_token_id,
                                                     pad_token_id=model.tokenizer.pad_token_id, 
                                                     return_dict_in_generate=True, 
                                                     output_scores=True,
                                                     **model_kwargs)
                sequences2 = outputs['sequences']


                round2_scores.append(outputs['sequences_scores'].tolist())
                round2_raw_seqs.append(outputs['sequences'].tolist())

            for ii in range(0,image.size(0)):

                save_json[batch_size*i + ii]['round2_scores'] = round2_scores[ii*num_seq:(ii+1)*num_seq]
                save_json[batch_size*i + ii]['round2_sequences'] = round2_raw_seqs[ii*num_seq:(ii+1)*num_seq]

    json.dump(save_json, open('{}/{}_round{}_beam{}_numseq{}_{}_chunk{}.json'.format(args.result_dir, args.split, args.round, num_beams, num_seq, args.checkpoint[-6:-4], args.chunk),'w'))
        
    return 
    
if __name__ == '__main__':
              
    parser = argparse.ArgumentParser()
              
    parser.add_argument('--config', default='configs/relation_grounding.yaml')
    parser.add_argument('--bert_config', default='configs/config_bert.json')
    parser.add_argument('--root', default='checkpoints_folder/')
    parser.add_argument('--output_dir', default='')   
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--split', default='test')
    parser.add_argument('--with_gt', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--position_res', default=512, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_seq', default=50, type=int)
    parser.add_argument('--num_beams', default=100, type=int)
    parser.add_argument('--round', default=1, type=int)
    parser.add_argument('--start', default=1, type=int)
    parser.add_argument('--end', default=1, type=int)
    parser.add_argument('--chunk', default=0, type=int) # 0,1,2,3
    parser.add_argument('--chunk_size', default=5, type=int) # in each chunk, how many samples are processed
    parser.add_argument('--batch_size', default=16, type=int)
   
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    config['position_res'] = args.position_res
    config['bert_config'] = args.bert_config
    assert config['position_res'] == args.position_res
    
    args.result_dir = os.path.join(args.root, args.output_dir, 'oidv6_results')
     
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    
    yaml.dump(config, open(os.path.join(args.result_dir, 'config.yaml'), 'w')) 
    
    all_checkpoints = glob.glob(os.path.join(args.root, args.output_dir, '*.pth'))
    all_checkpoints = all_checkpoints[args.start:args.end]
    print(len(all_checkpoints))
    for ckpt in all_checkpoints:
        args.checkpoint = ckpt
        main(args, config)
