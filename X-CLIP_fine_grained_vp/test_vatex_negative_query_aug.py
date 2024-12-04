import torch
import numpy as np
import json
from tqdm import tqdm
import pandas as pd
import warnings
import pickle
import argparse
warnings.filterwarnings("ignore")

import sys
sys.path.append("./XCLIP")
import os

from get_xclip import get_xclip


def get_args(description='Fine evaluation on X-CLIP'):
    parser = argparse.ArgumentParser(description=description)


    parser.add_argument("--save_dir", default='/var/scratch/achen/VisualSearch/vln_uvo/xclip_vlnuvo_bs128_margin0.2_w0.1_triplet', type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    
    parser.add_argument("--visual_path", default='/var/scratch/achen/VisualSearch/vln_uvo/xclip_feats/xclip_vlnuvo_bs128_margin0.2_w0.1_triplet_ep2_vis_feat.pkl', type=str, required=True,
                        help="The path of visual features.")
    
    parser.add_argument("--visual_mask_path", default='/var/scratch/achen/VisualSearch/vln_uvo/xclip_feats/xclip_vlnuvo_bs128_margin0.2_w0.1_triplet_ep2_mask.pkl', type=str, required=True,
                        help="The path of visual mask.")
    
    parser.add_argument("--test_model", default='/home/achen/github/X-CLIP_ag_triplet/ckpts_dsw/xclip_vlnuvo_bs128_margin0.2_w0.1_triplet/pytorch_model.bin.1', type=str, required=False, help="Initial model.")
    parser.add_argument("--test_id_path", default='/home/caz/VisualSearch/vatex_data/VATEX_test.csv', type=str, required=True, help="Initial model.")

    args = parser.parse_args()
    return args

args = get_args()

config = ['--do_eval', '--max_words', '32', '--max_frames', '12', '--freeze_layer_num', '0', '--slice_framepos', '2', \
          '--loose_type', '--linear_patch', '2d', '--sim_header', 'seqTransf', '--pretrained_clip_name', 'ViT-B/32', \
          '--output_dir', './', '--local_rank', '0', \
          '--init_model', args.test_model]

model = get_xclip(config)


from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer

def txt_tokenizer(sentence,max_words=32):
    k = 1
    pairs_text = np.zeros((k,max_words), dtype=np.int64)
    pairs_mask = np.zeros((k, max_words), dtype=np.int64)
    pairs_segment = np.zeros((k, max_words), dtype=np.int64)
        
    SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
    tokenizer = ClipTokenizer()
    words = tokenizer.tokenize(sentence)
    words = [SPECIAL_TOKEN["CLS_TOKEN"]] + words
    total_length_with_CLS = max_words - 1
    if len(words) > total_length_with_CLS:
        words = words[:total_length_with_CLS]
    words = words + [SPECIAL_TOKEN["SEP_TOKEN"]]

    input_ids = tokenizer.convert_tokens_to_ids(words)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)
    while len(input_ids) < max_words:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    assert len(input_ids) == max_words
    assert len(input_mask) == max_words
    assert len(segment_ids) == max_words

    pairs_text = np.array(input_ids)
    pairs_mask = np.array(input_mask)
    pairs_segment = np.array(segment_ids)

    return pairs_text, pairs_mask, pairs_segment



def smi_value(sent,batchs_video_features,batchs_video_mask):

    # pairs_text, pairs_mask, pairs_segment,
    input_ids,token_type_ids, t_attention_mask  =txt_tokenizer(sent)
    input_ids= torch.Tensor(input_ids).to(device).unsqueeze(dim=0).long()
    token_type_ids= torch.Tensor(token_type_ids).to(device).unsqueeze(dim=0)
    t_attention_mask= torch.Tensor(t_attention_mask).to(device).unsqueeze(dim=0)
    
    model.eval()
    with torch.no_grad():
        eot_tokens,text_features= model.get_sequence_output(input_ids, t_attention_mask,token_type_ids, shaped=True)
    
    # text_features = text_features[0].unsqueeze(dim=0)
    # eot_tokens = eot_tokens[0].unsqueeze(dim=0)

    each_row = []
    attention_mask=None
    # for batch_video_features, batch_video_mask in zip(batchs_video_features, batchs_video_mask):
        
        # b1b2_logits, *_tmp = model.get_similarity_logits(sequence_output, seq_features, visual_output, input_mask, video_mask,
        #                                                              loose_type=model.loose_type)
    b1b2_logits, *_tmp = model.get_similarity_logits(eot_tokens,text_features, batchs_video_features, 
                                                         attention_mask, batchs_video_mask, shaped=True,loose_type=model.loose_type)
    b1b2_logits = b1b2_logits.cpu().detach().numpy().squeeze()#1*40
        # each_row.append(b1b2_logits)
    # each_row = np.concatenate(tuple(each_row), axis=-1)
    return b1b2_logits




################loding pre-extract visual feature#######################


import json


#######################################
# test_p  = '/home/caz/VisualSearch/msrvtt10k/TextData/msrvtt1kAtest.caption.txt'
# # test_p='/home/caz/caz/github/toViS/ts2_net-master/Pos_res/msrvttFullsplit/msrvttFullsplit_4Pos.txt'
# test_video_ids = open(test_p).read().strip().split('\n')
# test_video_ids = list(set([x.split('#')[0] for x in test_video_ids]))
# print('-----num of videos is :',len(test_video_ids))
# load video feature
device = next(model.parameters()).device # 与model保持一致

feat_p= args.visual_path
mask_p= args.visual_mask_path
test_id_path = args.test_id_path

visual_feat = pickle.load(open(feat_p,'rb'))
visual_mask = pickle.load(open(mask_p,'rb'))


# test_id_path ='/var/scratch/achen/VisualSearch/vln_uvo/vln_UVO_val.txt'

# test_video_ids = open(test_id_path).read().strip().split('\n')

import  pandas as pd
# test_id_path='/home/caz/VisualSearch/vatex_data/VATEX_test.csv'
# csv = pd.read_csv(test_id_path)
# test_video_ids = list(csv['video_id'].values)
test_video_ids = open(test_id_path).read().strip().split('\n')

## caz 
# frame_feature_path = '/home/caz/caz/github/toUvA/X-CLIP_ag/X-CLIP-feats/XCLIP_features/vatex/xclip_vatex_frame_feat.jsonl'
# video_features = []
# with open(frame_feature_path, 'r') as fin:
#     for line in fin:
#         obj = json.loads(line.strip())
#         video_id = obj['video_id']
#         video_feature = obj['video_feature']
#         video_features.append(video_feature)
# video_features = torch.tensor(video_features).to(device) # (video_number, frame_number, feat_dim)
# video_mask = torch.ones((video_features.shape[0],video_features.shape[1])).to(device) # (video_number, frame_number)

# import  pandas as pd
# csv_path='/home/caz/VisualSearch/vatex_data/VATEX_test.csv'
# csv = pd.read_csv(csv_path)
# temp_video_ids = list(csv['video_id'].values)

# new_id =[]
# for i in temp_video_ids:
#     if i not in new_id:
#         new_id.append(i)
visual_feat = pickle.load(open(feat_p,'rb'))
visual_mask = pickle.load(open(mask_p,'rb'))



video_frame_feat = {}
for video_id ,f_feat in zip(test_video_ids, visual_feat):
        video_frame_feat[video_id]=f_feat


video_frame_mask = {}
for video_id ,f_mask in zip(test_video_ids, visual_mask):
        video_frame_mask[video_id]=f_mask 


#######################################
# csv_path = '/home/caz/VisualSearch/msrvtt10k/TextData/MSRVTT_JSFUSION_test.csv'
# csv = pd.read_csv(csv_path)
# test_video_ids = list(csv['video_id'].values)
# test_sentences = list(csv['sentence'].values)
# video_frame_feat={}
# for f_id,f_feat in zip(test_video_ids, video_features):
#     video_frame_feat[f_id]=f_feat
# visual_frame_mask={}  
# for f_id,f_mask in zip(test_video_ids, video_mask):
#     visual_frame_mask[f_id]=f_mask   


#######################################

test_p = '/var/scratch/achen/VisualSearch/vatex_data/vatex_test1k5_enc.txt'
msrvtt_s = open(test_p).read().strip().split('\n')
raw_caps={}
for line in msrvtt_s:
    tt_sent_id = line.split(' ')[0]
    raw_caps[tt_sent_id]=' '.join(line.split(' ')[1:])


changeS_P = '/var/scratch/achen/VisualSearch/vatex_data/hardnegative/' 
test_list = ['vatex1k5_ChangeADP_RE20_20230920.json','vatex1k5_ChangeVP_RE20_20230920.json','vatex1k5_ChangeVerb_RE20_20230920.json','vatex1k5_ChangeNP_RE20_20230920.json','vatex1k5_ChangeNoun_RE20_20230920.json']


import numpy
from tqdm import tqdm
import torch

save_p=args.save_dir
if not os.path.exists(save_p):
        os.mkdir(save_p)

for i in test_list:
    changejs = json.load(open(os.path.join(changeS_P,i)))
    testname=i.split('.json')[0]
    

    rank_GT=[]# append rank of gt sentence 
    res_list ={}

    for sent_id in tqdm(list(changejs.keys())):
    
        video_id = sent_id.split('#')[0]
        video_feat =  torch.tensor(np.array(video_frame_feat[video_id])).to(device).unsqueeze(dim=0)
        video_mask =  torch.tensor(np.array(video_frame_mask[video_id])).to(device).unsqueeze(dim=0)
        
        r_sent_id= sent_id.replace('#','#enc#')
        raw_sent =raw_caps[r_sent_id] # use the encoded sentence
        
        
        chage_s = list(changejs[sent_id].values())
        temp_test_sent = [raw_sent] +chage_s
        temp_sim_res=[]
        for S in temp_test_sent:
            temp_sim=smi_value(S,video_feat,video_mask)
            temp_sim_res.append(temp_sim)
        temp_sim_res=np.array(temp_sim_res).squeeze()
        temp_sort_res=np.argsort(temp_sim_res)#[-1] is the index of largest one
        
        temp_gt_rank =list(temp_sort_res)[::-1].index(0)+1
        res_list[sent_id]=[int(x) for x in list(temp_sort_res)[::-1]]
        rank_GT.append(int(temp_gt_rank))



    wr = open(os.path.join(save_p,'{}_dump_antonym_v2tres_MRR.txt'.format(testname)),'w')
    print(len(rank_GT))
    meanR_rank_GT = sum(rank_GT)/len(rank_GT)
    mrr = calculate_mrr(rank_GT)
    print('Mean Reciprocal Rank of  {} is:'.format(testname),mrr)
    wr.write('meanR of change {} is:{}\n'.format(testname,mrr))
    avg_rank_GT=rank_GT.count(1)/len(rank_GT)
    print('GT @ R1 count {} is:'.format(testname) ,rank_GT.count(1))
    wr.write('GT @ R1 count {} is:{}\n'.format(testname,rank_GT.count(1)))
    wr.close()


    with open(os.path.join(save_p,'{}_dump_antonym_v2tres.json'.format(testname)),'w') as json_file:
        json_file.write(json.dumps(res_list))





