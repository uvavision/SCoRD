import os
import re
import PIL
import json
import torch
import random
import numpy as np
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms as T
import torchvision.transforms.functional as F
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from dataset.randaugment import RandomAugment


class Relation_train_dataset_mixed(Dataset):
    def __init__(self, ann_file, max_words=200,  resize_ratio=0.25, img_res=256, position_res = 512):
        super().__init__()
        self.img_res = img_res
        self.ann = []
        print("Creating dataset")
        for f in ann_file:
            self.ann.extend(json.load(open(f)))
        print(len(self.ann)) # number of images
        self.max_words = max_words
        #image augmentation func
        self.aug_transform = Augfunc(True, resize_ratio)
        #the number of position tokens is 512
        self.position_res = position_res
        self.pos_dict = {x:f"[pos_{x}]" for x in range(self.position_res)}

        self.eos = '[SEP]'
        self.flip = 'train' in ' '.join(ann_file)
        
        
    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index].copy()
        ann = random.choice(ann) # choose a subject
        if '/cc3m/' in ann['root']+ann['file_name']:
            image = Image.open(ann['root']+ann['file_name']).convert('RGB')
        else:
            image = Image.open(ann['root']+ann['file_name']+'.jpg').convert('RGB')
        w, h = image.size

        no_sub_box = False
        no_obj_box = []
        
        if len(ann['sub_box']) == 0:
            no_sub_box = True
            ann['sub_bbox_list'] = [[-1, -1, -1, -1]]
        else:
            ann['sub_bbox_list'] = [[ann['sub_box'][0],ann['sub_box'][1],ann['sub_box'][2]-1,ann['sub_box'][3]-1]]
        bbox_list_sub = torch.as_tensor(ann['sub_bbox_list'], dtype=torch.float32).clamp(min=0)
        ann['obj_bbox_list']  = []
        for b in ann['obj_box']:
            if len(b) == 0:
                no_obj_box.append(True)
                ann['obj_bbox_list'].append([-1,-1,-1,-1])
            else:
                no_obj_box.append(False)
                ann['obj_bbox_list'].append([b[0],b[1],b[2]-1,b[3]-1])
                
        bbox_list_obj = torch.as_tensor(ann['obj_bbox_list'], dtype=torch.float32).clamp(min=0)

        
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes_sub = torch.min(bbox_list_sub.reshape(-1, 2, 2), max_size)
        cropped_boxes_obj = torch.min(bbox_list_obj.reshape(-1, 2, 2), max_size)
        ann['sub_bbox_list'] = cropped_boxes_sub.reshape(-1,4).numpy().tolist()
        ann['obj_bbox_list'] = cropped_boxes_obj.reshape(-1,4).numpy().tolist()

        image, ann, do_horizontal = self.aug_transform.random_aug(image, ann, self.flip, False, self.img_res)

        sub_seq = [ann['sub'], '@@']
        bbs = random.choice(ann['sub_bbox_list'])

        if not no_sub_box:
            sub_bbox_512 = [int(xy*self.position_res/self.img_res) if int(xy*self.position_res/self.img_res) <=self.position_res-1 else self.position_res-1  for xy in bbs]  
            sub_seq.extend([self.pos_dict[int(x)] for x in sub_bbox_512])
            
        else:
            sub_seq.extend(['[PAD]','[PAD]','[PAD]','[PAD]'])
        sub_seq.append('##')
        sub_caption = ' '.join(sub_seq)

        obj_captions = []
        for box_index in range(0,len(ann['obj_bbox_list'])):
            bbo = ann['obj_bbox_list'][box_index]
            obj_seq = [ann['relation'][box_index], ann['obj'][box_index], '@@']
            
            if not no_obj_box[box_index]: 
                obj_bbox_512 = [int(xy*self.position_res/self.img_res) if int(xy*self.position_res/self.img_res) <=self.position_res-1 else self.position_res-1  for xy in bbo]
                obj_seq.extend([self.pos_dict[int(x)] for x in obj_bbox_512])
                
            else:
                obj_seq.extend(['[PAD]','[PAD]','[PAD]','[PAD]'])
                
            obj_seq.append('##')
            obj_seq.append(self.eos)
            obj_seq = [x for x in obj_seq if x != '']
            obj_caption = ' '.join(obj_seq)
            obj_captions.append(obj_caption)

        if do_horizontal:
            sub_caption = sub_caption.replace("left", "[TMP]").replace("right", "left").replace("[TMP]", "right")
            for idx in range(0,len(obj_captions)):
                obj_captions[idx] = obj_captions[idx].replace("left", "[TMP]").replace("right", "left").replace("[TMP]", "right")
        sub_caption = pre_caption(sub_caption, self.max_words)
        sub_caption = sub_caption.replace('[pad]','[PAD]')
        
        
        for idx in range(0,len(obj_captions)):
            obj_captions[idx] = pre_caption(obj_captions[idx], self.max_words).replace('[pad]','[PAD]').replace('[sep]', '[SEP]')

        obj_captions = list(set(obj_captions))
        answer_weight = {}
        for answer in obj_captions:
            answer_weight[answer] = 0.5
        obj_captions = list(answer_weight.keys())
        weights = list(answer_weight.values())

        return image, sub_caption, obj_captions, weights

    
class Relation_val_dataset(Dataset):
    def __init__(self, root, ann_file, max_words=200,  resize_ratio=0.25, img_res=256, position_res = 512, replace = False):   
        super().__init__()
        self.img_res = img_res     
        self.ann_img = []
        print("Creating dataset")
        for f in ann_file:
            self.ann_img.extend(json.load(open(f)))
        self.ann = []
        for x in self.ann_img:
            if type(x) == dict:
                self.ann.append(x)
            else:
                self.ann.extend(x)        
        print(len(self.ann))
        self.max_words = max_words
        #image augmentation func
        self.aug_transform = Augfunc(True, resize_ratio)
        #the number of position tokens is 512
        self.position_res = position_res
        self.pos_dict = {x:f"[pos_{x}]" for x in range(self.position_res)}
        self.root = root
        self.replace = replace
        self.eos = '[SEP]'
        
    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):    
        ann = self.ann[index].copy()

        if self.root != '':
            image = Image.open(self.root+ann['file_name']+'.jpg').convert('RGB')
        else:
            image = Image.open(ann['root']+ann['file_name']+'.jpg').convert('RGB')
                       
        if type(ann['relation']) == list:
            obj_idx = random.choice(range(0,len(ann['relation'])))
            ann['obj'] = ann['obj'][obj_idx]
            ann['obj_box'] = ann['obj_box'][obj_idx]
            ann['relation'] = ann['relation'][obj_idx]
        else:
            ann['obj'] = ann['obj'] 
            ann['obj_box'] = ann['obj_box'] 
            ann['relation'] = ann['relation'] 
   
        ann['sub_bbox_list'] = [[ann['sub_box'][0],ann['sub_box'][1],ann['sub_box'][2]-1,ann['sub_box'][3]-1]]
        bbox_list_sub = torch.as_tensor(ann['sub_bbox_list'], dtype=torch.float32).clamp(min=0)
        ann['obj_bbox_list'] = [[ann['obj_box'][0],ann['obj_box'][1],ann['obj_box'][2]-1,ann['obj_box'][3]-1]]
        bbox_list_obj = torch.as_tensor(ann['obj_bbox_list'], dtype=torch.float32).clamp(min=0)
        
        w, h = image.size
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes_sub = torch.min(bbox_list_sub.reshape(-1, 2, 2), max_size)
        cropped_boxes_obj = torch.min(bbox_list_obj.reshape(-1, 2, 2), max_size)
        ann['sub_bbox_list'] = cropped_boxes_sub.reshape(-1,4).numpy().tolist()
        ann['obj_bbox_list'] = cropped_boxes_obj.reshape(-1,4).numpy().tolist()
        image, ann, do_horizontal = self.aug_transform.random_aug(image, ann, False, False, self.img_res)

        sub_seq = [ann['sub'], '@@'] 
        
        
        bbs = random.choice(ann['sub_bbox_list'])
        
        sub_bbox_512 = [int(xy*self.position_res/self.img_res) if int(xy*self.position_res/self.img_res) <=self.position_res-1 else self.position_res-1  for xy in bbs]
        sub_seq.extend([self.pos_dict[int(x)] for x in sub_bbox_512])
        sub_seq.append('##')
        sub_caption = ' '.join(sub_seq)

        bbo = random.choice(ann['obj_bbox_list'])
        obj_seq = [ann['relation'], ann['obj'], '@@'] 
        obj_bbox_512 = [int(xy*self.position_res/self.img_res) if int(xy*self.position_res/self.img_res) <=self.position_res-1 else self.position_res-1  for xy in bbo]
        obj_seq.extend([self.pos_dict[int(x)] for x in obj_bbox_512])
        obj_seq.append('##')
        obj_caption = ' '.join(obj_seq)

        if do_horizontal:
            sub_caption = sub_caption.replace("left", "[TMP]").replace("right", "left").replace("[TMP]", "right")
            obj_caption = obj_caption.replace("left", "[TMP]").replace("right", "left").replace("[TMP]", "right")
        sub_caption = pre_caption(sub_caption, self.max_words)
        obj_caption = pre_caption(obj_caption, self.max_words)

            
        obj_caption = [obj_caption+' '+self.eos]
        weights = [0.5]  
        return image, sub_caption, obj_caption, weights

def make_pseudo_pos_seq(name, bbox, img_h, img_w, position_res):
    #the number of position tokens is 512
    pos_dict = {x:f"[pos_{x}]" for x in range(position_res)}

    hh = position_res/int(img_h)
    ww = position_res/int(img_w)
    bbox_xyxy_resize = resize_bbox(bbox, hh, ww, position_res)
    if bbox_xyxy_resize == ' ':
        return name
    else:
        pos_seq = [name,' @@ ' ]
        pos_seq.extend([pos_dict[m] for m in bbox_xyxy_resize])
        pos_seq.append(' ## ')
        pseudo_seq = ' '.join(pos_seq)
        return pseudo_seq                   


def resize_bbox(bbox, h, w, position_res):
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3] 
        x1 = max(int(x_min * w,), 0)
        y1 = max(int(y_min * h,), 0)
        x2 = min(int(x_max * w,), position_res-1)
        y2 = min(int(y_max * h,), position_res-1)
        if position_res in [x1, y1, x2, y2]:
            return ' '
        else:
            return [x1, y1, x2, y2]


class Augfunc(object):
    def __init__(self, horizontal=True, resize_ratio=0.25):
        self.resize_ratio = resize_ratio
        max_size=1333
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.random_size_crop = Compose(
                                            [
                                                RandomResize([400, 500, 600]),
                                                RandomSizeCrop(384, max_size),
                                            ]
                                        )    
        self.horizontal = horizontal
        if self.horizontal:
            self.random_horizontal = RandomHorizontalFlip()
        self.final_transform = transforms.Compose([
            # RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',]),
            transforms.ToTensor(),
            normalize,
        ])
    def random_aug(self, image, ann, do_hori=True, do_aug=True, img_res=256):
        do_horizontal=False
        if random.random() < self.resize_ratio:
            image, ann = resize(image, ann, (img_res, img_res))
        else:
            if do_aug:
                image, ann = self.random_size_crop(image, ann)
            image, ann = resize(image, ann, (img_res, img_res))
        if do_hori:
            image, ann, do_horizontal = self.random_horizontal(image, ann)
            if 'caption' in ann:
                ann['caption'].replace('[TMP', 'right').replace('left_', 'right')
            elif 'normal_caption' in ann:
                ann['normal_caption'].replace('[TMP', 'right').replace('left_', 'right')
            elif 'obj_label_list_can' in ann:
                ann['obj_label_list_can'] = [x.replace('[TMP', 'right').replace('left_', 'right') for x in ann['obj_label_list_can']]
                
            else:
                if type(ann['relation']) == list:
                    ann['relation'] = [x.replace('[TMP', 'right').replace('left_', 'right') for x in ann['relation']]
                else:
                    ann['relation'].replace('[TMP', 'right').replace('left_', 'right')
        image = self.final_transform(image)
        return image, ann, do_horizontal


def pre_caption(caption,max_words):
    caption = re.sub(
        r"([,.'!?\"()*:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')
    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
    return caption



def hflip(image, target):
    flipped_image = F.hflip(image)
    w, h = image.size
    target = target.copy()
    if "sub_bbox_list" in target:
        boxes = torch.as_tensor(target["sub_bbox_list"], dtype=torch.float32)
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1], dtype=torch.float32) + torch.as_tensor([w, 0, w, 0], dtype=torch.float32)
        target["sub_bbox_list"] = boxes.numpy().tolist()
        
    for box_key in target:
        
        if "obj_bbox_list" in box_key:
            boxes = torch.as_tensor(target[box_key], dtype=torch.float32)
            boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1], dtype=torch.float32) + torch.as_tensor([w, 0, w, 0], dtype=torch.float32)
            target[box_key] = boxes.numpy().tolist()
        
    do_horizontal = True
    return flipped_image, target, do_horizontal


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size
    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img, target):
        do_horizontal = False
        if random.random() < self.p:
            return hflip(img, target)
        return img, target, do_horizontal
    

class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int, respect_boxes: bool = True):
        self.min_size = min_size
        self.max_size = max_size
        self.respect_boxes = respect_boxes  # if True we can't crop a box out
    def __call__(self, img: PIL.Image.Image, target: dict):
        init_boxes = len(target["not_crop_bbox_list"])
        max_patience = 100
        for i in range(max_patience):
            w = random.randint(self.min_size, min(img.width, self.max_size))
            h = random.randint(self.min_size, min(img.height, self.max_size))
            region = T.RandomCrop.get_params(img, [h, w])
            result_img, result_target = crop(img, target, region)
            if not self.respect_boxes or len(result_target["not_crop_bbox_list"]) == init_boxes or i < max_patience - 1:
                return result_img, result_target
            elif not self.respect_boxes or len(result_target["not_crop_bbox_list"]) == init_boxes or i == max_patience - 1:
                return img, target
        #return result_img, result_target


def crop(image, target, region):
    cropped_image = F.crop(image, *region)
    target = target.copy()
    i, j, h, w = region
    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w]).numpy().tolist()
    if "not_crop_bbox_list" in target:
        not_crop_bboxes = torch.as_tensor(target["not_crop_bbox_list"], dtype=torch.float32)
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = not_crop_bboxes - torch.as_tensor([j, i, j, i], dtype=torch.float32)
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["not_crop_bbox_list"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        positive_bboxes = torch.as_tensor(target["bbox_list"], dtype=torch.float32)
        positive_cropped_bboxes = positive_bboxes - torch.as_tensor([j, i, j, i], dtype=torch.float32)
        positive_cropped_bboxes = torch.min(positive_cropped_bboxes.reshape(-1, 2, 2), max_size)
        positive_cropped_bboxes = positive_cropped_bboxes.clamp(min=0)
        target["bbox_list"] = positive_cropped_bboxes.reshape(-1, 4).numpy().tolist()
    cropped_boxes = target["not_crop_bbox_list"].reshape(-1, 2, 2)
    keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
    crop_bbox = target["not_crop_bbox_list"][keep]
    target["not_crop_bbox_list"] = crop_bbox.reshape(-1, 4).numpy().tolist()
    return cropped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))
        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        return (oh, ow)
    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)
    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)
    if target is None:
        return rescaled_image, None
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios
    target = target.copy()
    if "sub_bbox_list" in target:
        boxes = target["sub_bbox_list"]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height], dtype=torch.float32)
        target["sub_bbox_list"] = scaled_boxes.numpy().tolist()
        
    for box_key in target:
        
        if "obj_bbox_list" in box_key:
            boxes = target[box_key]
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height], dtype=torch.float32)
            target[box_key] = scaled_boxes.numpy().tolist()
    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area
    h, w = size
    target["size"] = torch.tensor([h, w]).numpy().tolist()
    return rescaled_image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
