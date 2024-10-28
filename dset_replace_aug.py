from utils.basic_utils import save_jsonl, load_jsonl
from copy import deepcopy
import random
import numpy as np
from utils.length_aug import *
import sys

def nonGT_replace(data, moments, non_moments, ctx_l, clip_len, db_range, moment_db):

    ###############################################
    # non_moment_segments replacement
    ###############################################

    non_moment_segments = []

    for i, (s, e) in enumerate(non_moments):
        non_moment = dict()

        need_len = (e- s)

        find = False
        db_range_idx = -1
        for db_range_ in db_range:
            if need_len > db_range_:
                find = True
                break
            db_range_idx += 1

        if not find:
            print(need_len)
            return None
        
        another_moment = random.choice(moment_db[db_range_idx])
        non_moment['vid'] = another_moment[0]


        ass, aee = another_moment[1]
        if aee - ass < need_len:
            assert False
        else:
            aee = ass + need_len

        if ass == 0:
            rss = 0
        else:
            rss = int(ass // clip_len) if ass % clip_len == 0 else int(ass // clip_len) + 1

        ree = int(aee // clip_len)


        non_moment['clip_id'] = [rss, ree]
        non_moment['len'] = (ree - rss)

        non_moment_segments.append(non_moment)


    ###############################################
    # 새로운 data 만들기
    ###############################################

    new_data = dict()
    new_data['qid'] = data['qid']
    new_data['query'] = data['query']
    new_data['duration'] = data['duration']
    new_data['vid'] = data['vid']
    new_data['relevant_windows'] = data['relevant_windows']
    new_data['relevant_clip_ids'] = data['relevant_clip_ids']

    if 'saliency_scores' in data:
        new_data['saliency_scores'] = data['saliency_scores']


    new_data['org_clip_ids_order'] = []

    non_moments_idx = 0
    if non_moments[0][0] == 0:
        non_moment_segment = non_moment_segments[0]
        new_data['org_clip_ids_order'].append((non_moment_segment['vid'], non_moment_segment['clip_id']))
        non_moments_idx += 1

    for i in range(len(moments) - 1):

        # moment segment
        s, e = moments[i]
        rs = int(s // clip_len) if s != 0 else 0
        re = int(e // clip_len) if e % clip_len == 0 else int(e // clip_len) + 1
        new_data['org_clip_ids_order'].append((data['vid'], [rs, re]))

        # non-moment segment
        non_moment_segment = non_moment_segments[non_moments_idx]
        new_data['org_clip_ids_order'].append((non_moment_segment['vid'], non_moment_segment['clip_id']))
        non_moments_idx += 1

    # moment segment
    s, e = moments[-1]
    rs = int(s // clip_len) if s != 0 else 0
    re = int(e // clip_len) if e % clip_len == 0 else int(e // clip_len) + 1
    new_data['org_clip_ids_order'].append((data['vid'], [rs, re]))

    if non_moments_idx < len(non_moment_segments):
        non_moment_segment = non_moment_segments[-1]
        new_data['org_clip_ids_order'].append((non_moment_segment['vid'], non_moment_segment['clip_id']))

    assert len(new_data['org_clip_ids_order']) == len(moments) + len(non_moments)
    return new_data


dset_name = sys.argv[1]
seed = int(sys.argv[2])

print(f" ========== {dset_name} augmentation =============")
print(f" seed : {seed}")

savefilename = f"data/{dset_name}"
savefilename += f"_replace"
savefilename += f"_seed_{seed}.jsonl"

random.seed(seed)
np.random.seed(seed)



if dset_name == 'hl':
    datalist = load_jsonl('data/highlight_train_release.jsonl')
    clip_len = 2
    print(f" dset : QVHighlights")

elif dset_name == 'tacos':
    datalist = load_jsonl('data/tacos/train.jsonl')
    clip_len = 2
    print(f" dset : TACoS")

elif 'charades' in dset_name :
    datalist = load_jsonl('data/charades/charades_sta_train_tvr_format.jsonl')
    if 'vgg' in dset_name:
        clip_len = 0.166666
        print(f" dset : Charades - VGG")

    else:
        clip_len = 1
        print(f" dset : Charades")

else:
    assert False

print(f" ============================================")



# another video moment database
db_range = [150, 130, 110, 90, 70, 50, 30, 10, 0] # moment class borderline
reverse_db_range = ([150] + db_range)[:-1][::-1] # moment class borderline
moment_db = [[], # longer than 100
             [], # longer than 50
             [],
             [],
             [],
             [], # longer than 50
             [],
             [],
             [],] # longer than 10

for data in datalist:
    moments = data['relevant_windows']
    for start, end in moments:
        if (end-start) >= db_range[0]:
            moment_db[0].append((data['vid'], [start, end]))
        elif (end-start) >= db_range[1]:
            moment_db[1].append((data['vid'], [start, end]))
        elif (end-start) >= db_range[2]:
            moment_db[2].append((data['vid'], [start, end]))
        elif (end-start) >= db_range[3]:
            moment_db[3].append((data['vid'], [start, end]))
        elif (end-start) >= db_range[4]:
            moment_db[4].append((data['vid'], [start, end]))
        elif (end-start) >= db_range[5]:
            moment_db[5].append((data['vid'], [start, end]))
        elif (end-start) >= db_range[6]:
            moment_db[6].append((data['vid'], [start, end]))
        elif (end-start) >= db_range[7]:
            moment_db[7].append((data['vid'], [start, end]))
        elif (end-start) >= db_range[8]:
            moment_db[8].append((data['vid'], [start, end]))
    
print(f"Moment Database")
print(f"moment db (>= {db_range[0]}) : {len(moment_db[0])}")
print(f"moment db (>= {db_range[1]}) : {len(moment_db[1])}")
print(f"moment db (>= {db_range[2]}) : {len(moment_db[2])}")
print(f"moment db (>= {db_range[3]}) : {len(moment_db[3])}")
print(f"moment db (>= {db_range[4]}) : {len(moment_db[4])}")
print(f"moment db (>= {db_range[5]}) : {len(moment_db[5])}")
print(f"moment db (>= {db_range[6]}) : {len(moment_db[6])}")
print(f"moment db (>= {db_range[7]}) : {len(moment_db[7])}")
print(f"moment db (>= {db_range[8]}) : {len(moment_db[8])}")

new_datalist = []

for data in datalist:

    new_datalist.append(deepcopy(data))

    ctx_l = int(data['duration'] // clip_len) if data['duration'] % clip_len == 0 else int(data['duration'] // clip_len) + 1

    ###############################################
    # moment와 non-moment 구하기
    ###############################################

    if 'relevant_clip_ids' in data: # QVHighlights

        all_clips = np.zeros(ctx_l)
        all_clips[data['relevant_clip_ids']] = 1

        moments = find_ones_groups(all_clips, clip_len)
        assert moments == data['relevant_windows']

        non_moments = find_zeros_groups(all_clips, clip_len)

    else: # Charades, TACoS (single moment)
        moments = data['relevant_windows']
        non_moments = []
        if moments[0][0] != 0:
            non_moments.append([0, moments[0][0]])
        if moments[0][1] != data['duration']:
            non_moments.append([moments[0][1], data['duration']])    

    # 만약 non-moment가 없다면 이 data는 pass
    if not non_moments:
        continue 
    
    # crop augmentation
    new_data = nonGT_replace(data, moments=moments, non_moments=non_moments, ctx_l=ctx_l, clip_len=clip_len, db_range=db_range, moment_db=moment_db)
    if new_data:
        new_datalist.append(new_data)


print(f"Length Augmentation : {len(datalist)} -> {len(new_datalist)}")
print(f"Saved File : {savefilename}")

save_jsonl(new_datalist, savefilename)
