from functools import partial
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel, BertLMHeadModel

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np


class ALBEF(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 text_decoder = None,
                 tokenizer = None,
                 config = None,   
                 postoken_dict = None,
                 ):
        super().__init__()
        
        self.min_pos = tokenizer('@@').input_ids[-1]
        self.max_pos = tokenizer('##').input_ids[-1]
        
        self.tokenizer = tokenizer 
        self.distill = config['distill']

        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))    

        config_encoder = BertConfig.from_json_file(config['bert_config'])   
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=config_encoder, add_pooling_layer=False)  
            
        config_decoder = BertConfig.from_json_file(config['bert_config'])
        config_decoder.fusion_layer = 0
        config_decoder.num_hidden_layers = 6
        self.text_decoder = BertLMHeadModel.from_pretrained(text_decoder, config=config_decoder)    

        if self.distill:
            self.visual_encoder_m = VisionTransformer(
                img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
                mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))             
            self.text_encoder_m = BertModel.from_pretrained(text_encoder, config=config_encoder, add_pooling_layer=False)   
            self.text_decoder_m = BertLMHeadModel.from_pretrained(text_decoder, config=config_decoder)   
            self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                                [self.text_encoder,self.text_encoder_m],
                                [self.text_decoder,self.text_decoder_m],
                               ]
            self.copy_params() 
            self.momentum = 0.995
        

    def forward(self, image, quesiton, answer=None, alpha=0, k=None, weights=None, return_state_idx=None):
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        
        '''
        k: number of answers for each question
        weights: weight for each answer
        '''          
        answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)      

        question_output = self.text_encoder(quesiton.input_ids, 
                                            attention_mask = quesiton.attention_mask, 
                                            encoder_hidden_states = image_embeds,
                                            encoder_attention_mask = image_atts,                             
                                            return_dict = True)    

        question_states = []                
        question_atts = []  

        for b, n in enumerate(k):
            question_states += [question_output.last_hidden_state[b]]*n
            question_atts += [quesiton.attention_mask[b]]*n 
        question_states = torch.stack(question_states,0)    
        question_atts = torch.stack(question_atts,0)     

        if self.distill:                    
            with torch.no_grad():
                self._momentum_update()
                image_embeds_m = self.visual_encoder_m(image) 
                question_output_m = self.text_encoder_m(quesiton.input_ids, 
                                                        attention_mask = quesiton.attention_mask, 
                                                        encoder_hidden_states = image_embeds_m,
                                                        encoder_attention_mask = image_atts,                             
                                                        return_dict = True)    

                question_states_m = []                
                for b, n in enumerate(k):
                    question_states_m += [question_output_m.last_hidden_state[b]]*n
                question_states_m = torch.stack(question_states_m,0)    

                logits_m = self.text_decoder_m(answer.input_ids, 
                                               attention_mask = answer.attention_mask, 
                                               encoder_hidden_states = question_states_m,
                                               encoder_attention_mask = question_atts,                                  
                                               return_logits = True,
                                              )                       
            answer_output = self.text_decoder(answer.input_ids, 
                                          attention_mask = answer.attention_mask, 
                                          encoder_hidden_states = question_states,
                                          encoder_attention_mask = question_atts,                  
                                          labels = answer_targets,
                                          return_dict = True,   
                                          soft_labels = F.softmax(logits_m,dim=-1),
                                          reduction = 'none',
                                         )   

        else:
            answer_output = self.text_decoder(answer.input_ids, 
                                              attention_mask = answer.attention_mask, 
                                              encoder_hidden_states = question_states,
                                              encoder_attention_mask = question_atts,                  
                                              labels = answer_targets,
                                              return_dict = True,   
                                              reduction = 'none',
                                             )                      
        loss = weights * answer_output.loss         
        loss = loss.sum()/image.size(0)

        return loss

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
