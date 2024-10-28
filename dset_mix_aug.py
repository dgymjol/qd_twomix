from utils.basic_utils import save_jsonl, load_jsonl
from copy import deepcopy
import random
import numpy as np
from utils.length_aug import *
import sys

def mix(data, moments, non_moments, ctx_l, clip_len):


    ###############################################
    # moment segment 만들기
    ###############################################

    moment_segments = []

    ss_idx = 0
    for i, (s, e) in enumerate(moments):
        moment = dict()

        rs = int(s // clip_len) if s != 0 else 0
        re = int(e // clip_len) if e % clip_len == 0 else int(e // clip_len) + 1
        moment['clip_id'] = [rs, re]
        moment['seg_sec'] = [s - rs * clip_len, re * clip_len - e]
        moment['len'] = (re - rs)

        if 'saliency_scores' in data:
            ss_nxt_idx = ss_idx + moment['len']
            moment['saliency_scores'] = data['saliency_scores'][ss_idx : ss_nxt_idx]
            ss_idx = ss_nxt_idx

        moment_segments.append(moment)



    ###############################################
    # non_moment_segments 만들기
    ###############################################

    non_moment_segments = []

    for i, (s, e) in enumerate(non_moments):
        non_moment = dict()

        if s == 0:
            rs = 0
        else:
            rs = int(s // clip_len) if s % clip_len == 0 else int(s // clip_len) + 1
        re = int(e // clip_len)

        non_moment['clip_id'] = [rs, re]
        non_moment['len'] = (re - rs)

        non_moment_segments.append(non_moment)


    ###############################################
    # moment와 non-moment 섞기
    ###############################################

    random.shuffle(non_moment_segments)
    random.shuffle(moment_segments)

    ###############################################
    # 새로운 data 만들기
    ###############################################

    new_data = dict()
    new_data['qid'] = data['qid']
    new_data['query'] = data['query']
    new_data['duration'] = data['duration']
    new_data['vid'] = data['vid']

    new_clips = np.zeros(ctx_l)

    # new_data['saliency_scores'] ok
    # new_data['org_clip_ids_order'] ok
    cur_clip_id = 0
    new_data['org_clip_ids_order'] = []
    if 'saliency_scores' in data:
        new_data['saliency_scores'] = []
    seg_secs = []


    moments_idx = 0
    if len(non_moment_segments) < len(moment_segments):
        moment_segment = moment_segments[0]
        nxt_clip_id = cur_clip_id + moment_segment['len']
        new_clips[cur_clip_id:nxt_clip_id] = 1
        cur_clip_id = nxt_clip_id
        new_data['org_clip_ids_order'].append((data['vid'], moment_segment['clip_id']))
        if 'saliency_scores' in data:
            new_data['saliency_scores'] += moment_segment['saliency_scores']
        seg_secs.append(moment_segment['seg_sec'])
        moments_idx += 1

    for i in range(len(non_moment_segments) - 1):

        # non-moment segment
        non_moment_segment = non_moment_segments[i]
        cur_clip_id += non_moment_segment['len']
        new_data['org_clip_ids_order'].append((data['vid'], non_moment_segment['clip_id']))

        # moment segment
        moment_segment = moment_segments[moments_idx]
        nxt_clip_id = cur_clip_id + moment_segment['len']
        new_clips[cur_clip_id:nxt_clip_id] = 1
        cur_clip_id = nxt_clip_id
        new_data['org_clip_ids_order'].append((data['vid'], moment_segment['clip_id']))
        if 'saliency_scores' in data:
            new_data['saliency_scores'] += moment_segment['saliency_scores']
        seg_secs.append(moment_segment['seg_sec'])
        moments_idx += 1

    non_moment_segment = non_moment_segments[-1]
    new_data['org_clip_ids_order'].append((data['vid'], non_moment_segment['clip_id']))

    if moments_idx < len(moment_segments):
        moment_segment = moment_segments[moments_idx]
        nxt_clip_id = cur_clip_id + moment_segment['len']
        new_clips[cur_clip_id:nxt_clip_id] = 1
        cur_clip_id = nxt_clip_id
        new_data['org_clip_ids_order'].append((data['vid'], moment_segment['clip_id']))
        if 'saliency_scores' in data:
            new_data['saliency_scores'] += moment_segment['saliency_scores']
        seg_secs.append(moment_segment['seg_sec'])
        moments_idx += 1
    

    if 'relevant_clip_ids' in data:
        new_data['relevant_clip_ids'] = np.where(new_clips == 1)[0].tolist()
        new_data['relevant_windows'] = find_ones_groups(new_clips, clip_len)
    else:
        new_data['relevant_windows'] = []
        for sup_m, sub_m in zip(find_ones_groups(new_clips, clip_len), seg_secs):
            sups, supe = sup_m
            subs, sube = sub_m
            new_data['relevant_windows'].append([sups + subs, supe - sube])


    ### Test ####
    if 'saliency_scores' in data:
        assert len(data['saliency_scores']) == len(new_data['saliency_scores'])
        assert len(new_data['saliency_scores']) == len(new_data['relevant_clip_ids'])

    return new_data


dset_name = sys.argv[1]
seed = int(sys.argv[2])


print(f" ========== {dset_name} augmentation =============")
print(f" seed : {seed}")

savefilename = f"data/{dset_name}"
savefilename += f"_mix"
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
    new_crop_data = mix(data, moments=moments, non_moments=non_moments, ctx_l=ctx_l, clip_len=clip_len)
    if new_crop_data:
        new_datalist.append(new_crop_data)


print(f"Length Augmentation : {len(datalist)} -> {len(new_datalist)}")
print(f"Saved File : {savefilename}")

save_jsonl(new_datalist, savefilename)
