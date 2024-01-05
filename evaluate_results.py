import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import sys
import pickle


import torch
from models.tokenization_bert import BertTokenizer

import eval_utils

from dataset.relation_dataset import Relation_val_dataset
import glob
import re
from PIL import Image

    
def evaluate(args, config):
    # set up position tokens
    
    unus = ['[unused{}]'.format(x) for x in range(200,800)]
    pos_token = ['@@']
    pos_token.extend([f'[pos_{x}]' for x in range(512)])
    pos_token.append('##')
    pos_token.extend([f'[num_{x}]' for x in range(0,20)])
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

    postoken_dict_rev = {v:k for k,v in postoken_dict.items()}
    postoken_dict_rev[int(tokenizer('@@').input_ids[-1])] = '@@'
    postoken_dict_rev[int(tokenizer('##').input_ids[-1])] = '##'

    # get ground truth text and boxes from test data
    test_data = Relation_val_dataset(root = '', ann_file = config['test_file'], img_res=config['image_res'])
    gt_relation_obj_text = []
    gt_relation_obj_box = []
    for i in range(0,len(test_data.ann)):
        rel = test_data.ann[i]['relation'].replace('_',' ').split()
        rel.append(test_data.ann[i]['obj'])

        gt_relation_obj_text.append(rel)
        gt_relation_obj_box.append(test_data.ann[i]['obj_box'])
    
    # decode generated objects and boxes
    results_files = glob.glob(args.results_folder+'*.json') 
    results_files.sort()
    predicted_results = {}
    
    for n in results_files:
        temp = json.load(open(n,'r'))
        predicted_results.update(temp)
    print('There are', len(predicted_results),'predictions')
    
    p = re.compile('pos_(\d+)')
    img_info = {}
    text_res = []
    for img_id in predicted_results:
        if img_id not in img_info:
            image = Image.open(test_data.ann[int(img_id)]['root']+test_data.ann[int(img_id)]['file_name']+'.jpg').convert('RGB')
            w, h = image.size
            img_info[img_id] = [w,h]
        else:
            w, h = img_info[img_id]
        all_sentence_candidates = []
        results = predicted_results[img_id]
        result_sequence = results['round2_sequences']
        # deal with multiple predictions for each input subject
        
        for output in result_sequence:

            if type(output) == list:
                output = output[0]
            ans = []
            for tok in output:
                if int(tok) in postoken_dict_rev:
                    dec_tok = postoken_dict_rev[int(tok)]
                else:
                    dec_tok = tokenizer.decode(tok)
                ans.append(''.join(dec_tok.split()))
            sentence = ' '.join(ans)
            if ' @@' in sentence:
                decode_sentence = sentence.split(' @@')[0].split('[CLS] ')[-1].replace(' ##','')
                decode_positions = sentence.split(' @@')[1]
                four_pos = p.findall(decode_positions)
                if len(four_pos) != 4:
                    four_pos = [0,0,0,0]
                else:
                    four_pos = [int(x) for x in four_pos]
                    four_pos[0] = four_pos[0]/512*w
                    four_pos[1] = four_pos[1]/512*h
                    four_pos[2] = four_pos[2]/512*w
                    four_pos[3] = four_pos[3]/512*h
            elif ' [pos_' in sentence:
                decode_sentence = sentence.split('[CLS] ')[-1].split(' [pos_')[0].replace(' ##','')
                four_pos = [0,0,0,0]

            else:
                decode_sentence = sentence.split('[CLS] ')[-1].replace(' ##','')
                four_pos = [0,0,0,0]
            
            decode_sentence = decode_sentence.replace(' _ ','')
            decode_sentence = decode_sentence
            all_sentence_candidates.append([decode_sentence, four_pos])
        text_res.append(all_sentence_candidates)
        
    print('Finished decoding',len(text_res),'predicted sentences.')
    
    # prepare synsets to evaluate relation+object text
    oid_freq_rel_obj_valtest = oid_freq_valtest_verbs = oid_freq_valtest_objs = {}
    for test_file in config['test_file']:
        oid_freq, oid_freq_verbs, oid_freq_objs = eval_utils.get_synsets(test_file)
        oid_freq_rel_obj_valtest.update(oid_freq)
        oid_freq_valtest_verbs.update(oid_freq_verbs)
        oid_freq_valtest_objs.update(oid_freq_objs)
  
    # calculate accuracy
    correct_text = 0.0
    correct_box = 0.0
    total_samples = 0.0
    relation_type_results = {}
    already_count_text = {}
    for i in range(0,len(gt_relation_obj_text)):
        gt = gt_relation_obj_text[i] # GT relation+object text
        gt_box = gt_relation_obj_box[i] # GT object box
        rel_token = ' '.join(gt).lower().replace('_',' ') # GT relation+object text
        if rel_token not in relation_type_results:
            relation_type_results[rel_token] = [0,0,0] # successfully retrieved text, successfully retrieved box, all cases
        relation_type_results[rel_token][2] += 1    
        
        for res in text_res[i][:args.topk]:
            res_text = res[0]
            res_box = res[1]
            if args.mode == 'syn':
                pass_ = False
                rel_word,obj_word = oid_freq_rel_obj_valtest[rel_token]
                all_syn_rel = oid_freq_valtest_verbs[rel_word]
                all_syn_obj = oid_freq_valtest_objs[obj_word]
                pass_rel = False
                pass_obj = False
                res_text = ' ' + res_text + ' '


                for syn_rel in all_syn_rel:
                    if syn_rel in res_text:
                        pass_rel = True
                        break
                for syn_obj in all_syn_obj:
                    if syn_obj in res_text:
                        pass_obj = True
                        break               
                if pass_rel and pass_obj:
                    pass_ = True
                    
            elif args.mode == 'exact':
                pass_ = False
                if res_text == (' '.join(gt)).lower().replace('_',' '):
                    pass_ = True
                
            if pass_:
                # if this sample's relation+obj text is never predicted correctly, we count its correctness 
                if i not in already_count_text:
                    correct_text += 1
                    already_count_text[i] = 1
                    relation_type_results[rel_token][0] += 1
                    
                    
                # if this sample's relation+obj and box are both predicted correctly, we count the correctness and keep looking at the next sample
                if eval_utils.compute_iou(res_box,gt_box) >= args.box_threshold:
                    correct_box += 1
                    relation_type_results[rel_token][1] += 1
                    break
        total_samples += 1

    print('top k is ', args.topk)
    print('box threshold is ',args.box_threshold)
    print('Text Accuracy:',correct_text/total_samples)
    print('Box Accuracy:',correct_box/total_samples) 
    
    if args.report_unseen:
        bins = pickle.load(open('data_preparation/processed_data/coco_cc_oid_bins_52_100_20000.p','rb'))

        text_cor = {'set0':0,'set1':0,'others':0,'all':0}
        box_cor = {'set0':0,'set1':0,'others':0,'all':0}
        all_ = {'set0':0,'set1':0,'others':0,'all':0}

        unseen_perf = {name:[] for name in bins[1]}
        for name in relation_type_results:

            if name in bins[0]:
                all_['set0']+= float(relation_type_results[name][2])
                text_cor['set0'] += relation_type_results[name][0]
                box_cor['set0'] += relation_type_results[name][1]
            if name in bins[1]:
                unseen_perf[name].append([relation_type_results[name][0], relation_type_results[name][1], relation_type_results[name][2]])

                all_['set1']+= float(relation_type_results[name][2])
                text_cor['set1'] += relation_type_results[name][0]
                box_cor['set1'] += relation_type_results[name][1]
            if name not in bins[0]+bins[1]:
                all_['others']+= float(relation_type_results[name][2])
                text_cor['others'] += relation_type_results[name][0]
                box_cor['others'] += relation_type_results[name][1]
            all_['all']+= float(relation_type_results[name][2])
            text_cor['all'] += relation_type_results[name][0]
            box_cor['all'] += relation_type_results[name][1]

        print('Underseen classes text and box accuracy: ', text_cor['set0']/all_['set0'], box_cor['set0']/all_['set0'])

        print('Unseen classes text and box accuracy: ', text_cor['set1']/all_['set1'], box_cor['set1']/all_['set1'])
        print('all classes text and box accuracy: ', text_cor['all']/all_['all'], box_cor['all']/all_['all'])

        print('underseen + unseen classes text and box accuracy: ', (text_cor['set1'] + text_cor['set0'])/(all_['set1']+all_['set0']), (box_cor['set1'] + box_cor['set0'])/(all_['set1'] + all_['set0']))

        print('others classes text and box accuracy: ', text_cor['others']/all_['others'], box_cor['others']/all_['others'])


    return 

if __name__ == '__main__':
              
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_folder', default='checkpoints_folder/oidv6_results/')              
    parser.add_argument('--config', default='configs/relation_grounding.yaml')
    parser.add_argument('--mode', default='syn', choices=['syn', 'exact'], help='predicted synonyms|exact text as correct')
    parser.add_argument('--topk', default=3, type=int) # consider top-k predictions
    parser.add_argument('--box_threshold', default=0.5) # if iou(gt_box, predicted_box) >= 0.5, the prediction is correct
    parser.add_argument('--report_unseen', default=False, type=bool)
    
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    evaluate(args, config)

