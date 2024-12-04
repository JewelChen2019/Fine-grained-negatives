from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import pickle
from dataloaders.rawvideo_util import RawVideoExtractor

import json
from gensim.models import KeyedVectors
import re
import nltk
from nltk.corpus import wordnet
import random
# 下载WordNet语料库
nltk.download('wordnet')
import spacy
from spacy import displacy
import textacy
cls = spacy.util.get_lang_class('en')
stop_words = cls.Defaults.stop_words
nlp = spacy.load("en_core_web_sm")

dataset_based_vocab_path=  '/var/scratch/achen/VisualSearch/vatex_data/TextData/vatex_word_dic_20230924_len1.json'
from dataloaders.generated_negative_sentence import Get_Negative_text_samples


get_senteces= Get_Negative_text_samples(dataset_based_vocab_path)
get_neg_word_level_sent_fun = get_senteces.change_word
get_neg_phrase_level_sent_fun = get_senteces.change_phrase
# get_neg_sent_fun = get_senteces.change_random_phrase


class VATEX_TrainDataLoader(Dataset):
    """UVO dataset loader."""
    def __init__(
            self,
            subset,
            data_path,
            features_path,
            tokenizer,
            output_dir,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
    ):
        self.output_dir=output_dir
        self.temp_wr = open('{}/neg_train.txt'.format(self.output_dir),'a')
        self.data_path = data_path
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.subset = subset
        assert self.subset in ["train", "val", "test"]
        video_id_path_dict = {}
        video_id_path_dict["train"] = os.path.join(self.data_path, "vatex_train.txt")
        video_id_path_dict["val"] = os.path.join(self.data_path, "vatex_val1k5.txt")
        video_id_path_dict["test"] = os.path.join(self.data_path, "vatex_test1k5.txt")
        # video_id_path_dict["test"] = os.path.join(self.data_path, "feat_test.txt")
        caption_file = os.path.join(self.data_path, 'en_vatex_data.json')

        with open(video_id_path_dict[self.subset], 'r') as fp:
            video_ids = [itm.strip() for itm in fp.readlines()]

        captions = json.load(open(caption_file))

        video_dict = {}
        for root, dub_dir, video_files in os.walk(self.features_path):
            for video_file in video_files:
                video_id_ = ".".join(video_file.split(".")[:-1])
                if video_id_ not in video_ids:
                    continue
                file_path_ = os.path.join(root, video_file)
                video_dict[video_id_] = file_path_
        self.video_dict = video_dict

        self.sample_len = 0
        self.sentences_dict = {}
        self.cut_off_points = []
        for video_id in video_ids:
            assert video_id in captions
            for cap_txt in captions[video_id]:
                self.sentences_dict[len(self.sentences_dict)] = (video_id, cap_txt)
            self.cut_off_points.append(len(self.sentences_dict))
        print("Total Paire: {} {}".format(self.subset, len(self.sentences_dict)))

        ## below variables are used to multi-sentences retrieval
        # self.cut_off_points: used to tag the label when calculate the metric
        # self.sentence_num: used to cut the sentence representation
        # self.video_num: used to cut the video representation
        self.multi_sentence_per_video = True    # !!! important tag for eval
        if self.subset == "val" or self.subset == "test":
            self.sentence_num = len(self.sentences_dict)
            self.video_num = len(video_ids)
            assert len(self.cut_off_points) == self.video_num
            print("For {}, sentence number: {}".format(self.subset, self.sentence_num))
            print("For {}, video number: {}".format(self.subset, self.video_num))

        print("Video number: {}".format(len(self.video_dict)))
        print("Total Paire: {} {}".format(self.subset, len(self.sentences_dict)))

        self.sample_len = len(self.sentences_dict)
        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id, caption):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.int64)

        for i, video_id in enumerate(choice_video_ids):
            words = self.tokenizer.tokenize(caption)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.int64)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float64)

        for i, video_id in enumerate(choice_video_ids):
            video_path = self.video_dict[video_id]

            raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
            # print('-------',video_path)
            raw_video_data = raw_video_data['video']

            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                # L x T x 3 x H x W
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_indx, ...]
                else:
                    video_slice = raw_video_slice

                video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

                slice_len = video_slice.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def __getitem__(self, idx):
        video_id, caption = self.sentences_dict[idx]

        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption)
        video, video_mask = self._get_rawvideo(choice_video_ids)
        return pairs_text, pairs_mask, pairs_segment, video, video_mask

    def _get_text_wneg(self, video_id, caption,change_num=15):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.int64)
        #产生change_num个 hard negative samples
        # neg_word_level_sents,change_word_pos = get_neg_word_level_sent_fun(caption,change_num=change_num)#generated K sentences but only use K-1 as hard negative samples
        ## 输出 word-level and phrase-level
        word_text_neg = np.zeros((k, change_num, self.max_words), dtype=np.int64)
        word_mask_neg = np.zeros((k, change_num, self.max_words), dtype=np.int64)
        word_segment_neg = np.zeros((k,change_num, self.max_words), dtype=np.int64)

        word_neg_sents,word_change_pos = get_neg_word_level_sent_fun(caption.split(' '),change_num=change_num)#generated K sentences but only use K-1 as hard negative samples

        phrase_text_neg = np.zeros((k, change_num, self.max_words), dtype=np.int64)
        phrase_mask_neg = np.zeros((k, change_num, self.max_words), dtype=np.int64)
        phrase_segment_neg = np.zeros((k,change_num, self.max_words), dtype=np.int64)
        phrase_neg_sents,phrase_change_pos = get_neg_phrase_level_sent_fun(caption.split(' '),change_num=change_num)#generated K sentences but only use K-1 as hard negative samples
        ## because there is no need for the phrase-level negative samples, so we only use word-level negative samples
        ## and use '  ' replace the phrase-level negative samples to save time

        self.temp_wr.write(caption+'\n')
        for i_sent,i_pos in zip(word_neg_sents,word_change_pos):
            # import pdb;pdb.set_trace()
            self.temp_wr.write(i_pos+'##'+i_sent+'\n')
        for i_sent,i_pos in zip(phrase_neg_sents,phrase_change_pos):
            self.temp_wr.write(i_pos+'##'+i_sent+'\n')
        self.temp_wr.write('---------------------------'+'\n')



        for i, video_id in enumerate(choice_video_ids):
            # print('-----------------')
            # print(caption)
            # print('-----------------')
            
            words = self.tokenizer.tokenize(caption)
            # print('---------words--------')
            # print(words)
            # print('-----------------')

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)
            ## negative 

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)
            for j, t_neg_sent in enumerate(word_neg_sents):
                # print('--------word_neg_sents---------')
                # print(word_neg_sents)
                # print('-----------------')
                words = self.tokenizer.tokenize(t_neg_sent)
                words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
                total_length_with_CLS = self.max_words - 1
                if len(words) > total_length_with_CLS:
                    words = words[:total_length_with_CLS]
                words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

                input_ids_aug = self.tokenizer.convert_tokens_to_ids(words)
                input_mask_aug = [1] * len(input_ids_aug)
                segment_ids_aug = [0] * len(input_ids_aug)

                while len(input_ids_aug) < self.max_words:
                    input_ids_aug.append(0)
                    input_mask_aug.append(0)
                    segment_ids_aug.append(0)
                assert len(input_ids_aug) == self.max_words
                assert len(input_mask_aug) == self.max_words
                assert len(segment_ids_aug) == self.max_words
                
                if j==i:# i和j 因为i始终为0 所以，每一个list中，第一个句子为original sentence
                    word_text_neg[i][j]= np.array(input_ids)
                    word_mask_neg[i][j] =  np.array(input_mask)
                    word_segment_neg[i][j] = np.array(segment_ids)
                    

                else:
                    word_text_neg[i][j]= np.array(input_ids_aug)
                    word_mask_neg[i][j] =  np.array(input_mask_aug)
                    word_segment_neg[i][j] = np.array(segment_ids_aug)
                
                
                # pairs_text_neg[i][j]= np.array(input_ids_aug)
                # pairs_mask_neg[i][j] =  np.array(input_mask_aug)
                # pairs_segment_neg[i][j] = np.array(segment_ids_aug)



            for j, t_neg_sent in enumerate(phrase_neg_sents):
                # print('--------phrase_neg_sents---------')
                # print(phrase_neg_sents)
                # print('-----------------')
                words = self.tokenizer.tokenize(t_neg_sent)
                words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
                total_length_with_CLS = self.max_words - 1
                if len(words) > total_length_with_CLS:
                    words = words[:total_length_with_CLS]
                words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

                input_ids_aug = self.tokenizer.convert_tokens_to_ids(words)
                input_mask_aug = [1] * len(input_ids_aug)
                segment_ids_aug = [0] * len(input_ids_aug)

                while len(input_ids_aug) < self.max_words:
                    input_ids_aug.append(0)
                    input_mask_aug.append(0)
                    segment_ids_aug.append(0)
                assert len(input_ids_aug) == self.max_words
                assert len(input_mask_aug) == self.max_words
                assert len(segment_ids_aug) == self.max_words
                
                if j==i:
                    phrase_text_neg[i][j]= np.array(input_ids)
                    phrase_mask_neg[i][j] =  np.array(input_mask)
                    phrase_segment_neg[i][j] = np.array(segment_ids)
                else:
                    phrase_text_neg[i][j]= np.array(input_ids_aug)
                    phrase_mask_neg[i][j] =  np.array(input_mask_aug)
                    phrase_segment_neg[i][j] = np.array(segment_ids_aug)



        return pairs_text, pairs_mask, pairs_segment, choice_video_ids, word_text_neg,word_mask_neg,word_segment_neg,phrase_text_neg,phrase_mask_neg,phrase_segment_neg

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.int64)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float64)

        for i, video_id in enumerate(choice_video_ids):
            video_path = self.video_dict[video_id]

            raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data['video']

            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                # L x T x 3 x H x W
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_indx, ...]
                else:
                    video_slice = raw_video_slice

                video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

                slice_len = video_slice.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def __getitem__(self, idx):
        video_id, caption = self.sentences_dict[idx]

        do_neg_aug = False
        do_word_phrase_neg_aug = True
        # change_num= self.batch_size*self.n_gpu - 1
        # 设置一个超参数 控制negative sample 和 batch size的 ratio
        change_num= 16
        if do_neg_aug ==True:
            pairs_text, pairs_mask, pairs_segment, choice_video_ids,pairs_text_neg,pairs_neg_mask,pairs_neg_segment = self._get_text_wneg(video_id, caption,change_num=change_num)
            video, video_mask = self._get_rawvideo(choice_video_ids)
            return pairs_text, pairs_mask, pairs_segment, video, video_mask,word_text_neg,word_mask_neg,word_segment_neg
        elif do_word_phrase_neg_aug ==True:

            pairs_text, pairs_mask, pairs_segment, choice_video_ids,word_text_neg,word_mask_neg,word_segment_neg,phrase_text_neg,phrase_mask_neg,phrase_segment_neg = self._get_text_wneg(video_id, caption,change_num=change_num)
            video, video_mask = self._get_rawvideo(choice_video_ids)
            return pairs_text, pairs_mask, pairs_segment, video, video_mask, word_text_neg,word_mask_neg,word_segment_neg,phrase_text_neg,phrase_mask_neg,phrase_segment_neg
    
        else:
            pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption)
            video, video_mask = self._get_rawvideo(choice_video_ids)
            return pairs_text, pairs_mask, pairs_segment, video, video_mask
