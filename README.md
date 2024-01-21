# [SCoRD: Subject-Conditional Relation Detection With Text-Augmented Data](https://openaccess.thecvf.com/content/WACV2024/html/Yang_SCoRD_Subject-Conditional_Relation_Detection_With_Text-Augmented_Data_WACV_2024_paper.html)
Ziyan Yang, Kushal Kafle, Zhe Lin, Scott Cohen, Zhihong Ding, Vicente Ordonez

## Abstract
We propose Subject-Conditional Relation Detection SCoRD, where conditioned on an input subject, the goal is to predict all its relations to other objects in a scene along with their locations. Based on the Open Images dataset, we propose a challenging OIv6-SCoRD benchmark such that the training and testing splits have a distribution shift in terms of the occurrence statistics of <subject, relation, object> triplets. To solve this problem, we propose an auto-regressive model that given a subject, it predicts its relations, objects, and object locations by casting this output as a sequence of tokens. First, we show that previous scene-graph prediction methods fail to produce as exhaustive an enumeration of relation-object pairs when conditioned on a subject on this benchmark. Particularly, we obtain a recall@3 of 83.8% for our relation-object predictions compared to the 49.75% obtained by a recent scene graph detector. Then, we show improved generalization on both relation-object and object-box predictions by leveraging during training relation-object pairs obtained automatically from textual captions and for which no object-box annotations are available. Particularly, for <subject, relation, object> triplets for which no object locations are available during training, we are able to obtain a recall@3 of 33.80% for relation-object pairs and 26.75% for their box locations.

## Install

Please follow [ALBEF](https://github.com/salesforce/ALBEF) to install the required packages. 

## Data
Download the training and testing splits [here](https://drive.google.com/drive/folders/19Bk1mhaXvW8bAeMgmHvg7X3ZTAmZOwd0?usp=sharing). 
To download images:
- Download [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html), [MS COCO](https://cocodataset.org/#download), [Flickr30k](https://bryanplummer.com/Flickr30kEntities/) and [OpenImageV6](https://storage.googleapis.com/openimages/web/download.html) images from the corresponding websites
- Download [CC3M](https://ai.google.com/research/ConceptualCaptions/) using this [codebase](https://github.com/igorbrigadir/DownloadConceptualCaptions)
- Download [CC12M](https://github.com/google-research-datasets/conceptual-12m) using this [codebase](https://github.com/rom1504/img2dataset?tab=readme-ov-file)

## Checkpoint
Download the checkpoint for the removing 50% experiment [here](https://drive.google.com/drive/folders/1vuH6NiGLO-MYlgEdYPEkYEsbTt0iw4xA?usp=sharing).

## Evaluation 
First, run this command to generate <relation, object, object location> triples: 
```bash
# start and end indices indicate the index of your target checkpoint in the checkpoint folder. If you only have one checkpoint in the folder, the start flag should be 0 and the end flag should be 1
# chunk size indicates how many batches of evaluation samples should be processed
CUDA_VISIBLE_DEVICES=0 python results_generation.py --root your_checkpoint_folder --start 0 --end 1 --chunk 0 --num_seq 3 --num_beams 5 --chunk_size 100 --round 2
```

Then, run this command to get evaluation results:
```bash
python evaluate_results.py --results_folder your_checkpoint_folder/oidv6_results/  --report_unseen True --topk 3
```
## Training
First, download the pre-trained checkpoint from [PEVL](https://thunlp.oss-cn-qingdao.aliyuncs.com/pevl_pretrain.pth):
Run:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=12888 --use_env run_relation_train.py --config configs/relation_grounding.yaml --output_dir your_checkpoint_folder --checkpoint pevl_pretrain.pth
```

## Acknowledgement

We would like to thank [ALBEF](https://github.com/salesforce/ALBEF) and [PEVL](https://github.com/thunlp/PEVL/tree/main). Their released codebases help a lot in this project.

## Citing

If you think this work is interesting, please consider to cite it:
```bash
@inproceedings{yang2024scord,
  title={SCoRD: Subject-Conditional Relation Detection with Text-Augmented Data},
  author={Yang, Ziyan and Kafle, Kushal and Lin, Zhe and Cohen, Scott and Ding, Zhihong and Ordonez, Vicente},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={5731--5741},
  year={2024}
}
```