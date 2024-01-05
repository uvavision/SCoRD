import torch
import numpy as np
import json

def get_synsets(data_file):
     
    oid_valtest = json.load(open(data_file,'r')) 
#     print(len(oid_valtest))

    oid_freq_valtest = {}
    oid_freq_rel_obj_valtest = {}
    for x in oid_valtest:
        if 'oidv' not in x[0]['root']:
            continue
        for y in x:
            y['relation'] = y['relation'].lower().replace('_',' ')
            y['obj'] = y['obj'].lower().replace('_',' ')
            rel_obj = y['relation'] + ' ' + y['obj']
            if rel_obj not in oid_freq_valtest:
                oid_freq_valtest[rel_obj] = 0
                oid_freq_rel_obj_valtest[rel_obj] = [y['relation'], y['obj']]


            oid_freq_valtest[rel_obj] += 1
#     print(len(oid_freq_valtest))
    
    oid_freq_valtest_verbs = {oid_freq_rel_obj_valtest[x][0]:{' ' + oid_freq_rel_obj_valtest[x][0]+' '} for x in oid_freq_rel_obj_valtest}
    oid_freq_valtest_objs = {oid_freq_rel_obj_valtest[x][1]:{' ' + oid_freq_rel_obj_valtest[x][1]+' '} for x in oid_freq_rel_obj_valtest}

    irregular_rel = {'throw':{' threw '}, 'holds':{' held '}, 'hug':{' hugging ', ' hugged '},'hits':{' hitting '},'cut':{' cutting '}, 'eat':{' eatting ',' eaten ',' ate '}, \
                    'drink':{' drunk '},'wears':{' wore ',' worn '},'ride':{' rode ',' ridden '},'hang':{' hung '},'catch':{' caught '}}
    irregular_obj = {'man':{' men '}, 'woman':{' women '}, 'goggles':{' goggle '}, 'scarf':{' scarves '}, 'bus':{'buses'}, 'wine glass':{' wine glasses '}}
    for x in oid_freq_valtest_verbs:
        if x in {'on', 'in', 'at', 'under', 'inside of'}:
            continue
        if (x.split()[0][-1] == 's' or x.split()[0][-1] == 'e') and x.split()[0][-2:] != 'ss':
            if len(x.split()) == 1:
                oid_freq_valtest_verbs[x].add(' ' + x.split()[0][:-1]+' ')
                oid_freq_valtest_verbs[x].add(' ' + x.split()[0][:-1]+'ing ')
                oid_freq_valtest_verbs[x].add(' ' + x.split()[0][:-1]+'ed ')
            elif len(x.split()) == 2:
                oid_freq_valtest_verbs[x].add(' ' + x.split()[0][:-1]+' '+ x.split()[1] + ' ')
                oid_freq_valtest_verbs[x].add(' ' + x.split()[0][:-1]+'ed '+ x.split()[1] + ' ')
                oid_freq_valtest_verbs[x].add(' ' + x.split()[0][:-1]+'ing '+ x.split()[1] + ' ')

            if x.split()[0][-1] == 'e':
                oid_freq_valtest_verbs[x].add(' ' + x+'s ')

        elif x.split()[0][-2:] == 'ss':
            oid_freq_valtest_verbs[x].add(' ' + x+'es ')
            oid_freq_valtest_verbs[x].add(' ' + x+'ed ')
            oid_freq_valtest_verbs[x].add(' ' + x+'ing ')
        else:   
            oid_freq_valtest_verbs[x].add(' ' + x+'s ')
            oid_freq_valtest_verbs[x].add(' ' + x+'ed ')
            oid_freq_valtest_verbs[x].add(' ' + x+'ing ')
    for word in irregular_rel:
        if word in oid_freq_valtest_verbs:
            oid_freq_valtest_verbs[word].union(irregular_rel[word])

    
    for x in oid_freq_valtest_objs:
        if x[-2:] == 'es':
            oid_freq_valtest_objs[x].add(' ' + x[:-2]+' ')
            oid_freq_valtest_objs[x].add(' ' + x[:-2]+'e ')
        elif x[-1:] == 's':
            oid_freq_valtest_objs[x].add(' ' + x[:-1]+' ')
        else:
            oid_freq_valtest_objs[x].add(' ' + x+'s ')
            oid_freq_valtest_objs[x].add(' ' + x+'es ')
        if x[-1:] == 'y':
            oid_freq_valtest_objs[x].add(' ' + x[:-1]+'ies ')

        if len(x.split()) > 1:
            oid_freq_valtest_objs[x].add(' ' + x.replace(' ','')+' ')
            oid_freq_valtest_objs[x].add(' ' + x.replace(' ','')+'es ')
            oid_freq_valtest_objs[x].add(' ' + x.replace(' ','')+'s ')
            
    for word in irregular_obj:
        if word in oid_freq_valtest_objs:
            oid_freq_valtest_objs[word].union(irregular_obj[word])
    

    return oid_freq_rel_obj_valtest, oid_freq_valtest_verbs, oid_freq_valtest_objs


def compute_area(bbox,invalid=None):
    x1,y1,x2,y2 = bbox

    if (x2 <= x1) or (y2 <= y1):
        area = invalid
    else:
        area = (x2 - x1 + 1) * (y2 - y1 + 1)

    return area
def compute_iou(bbox1,bbox2,verbose=False):
    x1,y1,x2,y2 = bbox1
    x1_,y1_,x2_,y2_ = bbox2
    
    x1_in = max(x1,x1_)
    y1_in = max(y1,y1_)
    x2_in = min(x2,x2_)
    y2_in = min(y2,y2_)

    intersection = compute_area(bbox=[x1_in,y1_in,x2_in,y2_in],invalid=0.0)
    area1 = compute_area(bbox1)
    area2 = compute_area(bbox2)
    if area1 is None or area2 is None:
        return 0
    union = area1 + area2 - intersection
    iou = intersection / (union + 1e-6)

    if verbose:
        return iou, intersection, union

    return iou 
