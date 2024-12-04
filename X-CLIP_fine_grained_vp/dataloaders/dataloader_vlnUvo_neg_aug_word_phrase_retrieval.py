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
# nltk.download('wordnet')
import spacy
from spacy import displacy
import textacy
cls = spacy.util.get_lang_class('en')
stop_words = cls.Defaults.stop_words
# nlp = spacy.load("en_core_web_sm")

dataset_based_vocab_path=  '/data1/caz/github/accv24/vln_uvo/hardnegative/vln_uvo_word_dic_20230920.json'
from dataloaders.generated_negative_sentence import Get_Negative_text_samples

'''
class Get_Negative_text_samples():
    def __init__(
            self,
        
            dataset_based_vocab_path,
    ):
    
        self.vocab = json.load(open(dataset_based_vocab_path, 'r'))


    def remove_words_with_symbols(self,word_list):
    # 定义正则表达式模式匹配非字母和非数字字符
        pattern = re.compile(r'[^a-zA-Z0-9]')
        word_list = [str(word) for word in word_list]

        # 过滤列表中带有符号的单词
    
        filtered_list = [word for word in word_list if not pattern.search(word)]

        return filtered_list

    
    def find_synonyms_antonyms(self,word):
        # 下载 WordNet 数据（仅首次运行需要）
        # nltk.download('wordnet')

        # 获取输入词的所有词义
        synsets = wordnet.synsets(word)

        antonyms = []

        # 遍历每个词义，提取其中的反义词
        for synset in synsets:
            for lemma in synset.lemmas():
                if lemma.antonyms():
                    antonyms.extend(lemma.antonyms())

        # 提取反义词列表
        antonyms_list = [antonym.name() for antonym in antonyms]

        return list(set(antonyms_list))


    def get_hypernyms(self,word,pos_item='NOUN'):
        if pos_item == 'ADV':
            pos=wordnet.ADV 
        elif pos_item == 'ADJ':
            pos=wordnet.ADJ
        elif pos_item == 'VERB':
            pos=wordnet.VERB
        elif pos_item == 'NOUN':
            pos=wordnet.NOUN
    
        


        synsets = wordnet.synsets(word, pos=pos)
        hypernyms = []
        for synset in synsets:
            for hypernym in synset.hypernyms():
                for lemma in hypernym.lemmas():
                    hypernyms.append(lemma.name())
        return list(set(hypernyms))


    def get_hyponyms(self,word,pos_item='NOUN'):
        if pos_item == 'ADV':
            pos=wordnet.ADV 
        elif pos_item == 'ADJ':
            pos=wordnet.ADJ
        elif pos_item == 'VERB':
            pos=wordnet.VERB
        elif pos_item == 'NOUN':
            pos=wordnet.NOUN
     

        synsets = wordnet.synsets(word, pos=pos)
        hyponyms = []
        for synset in synsets:
            for hyponym in synset.hyponyms():
                for lemma in hyponym.lemmas():
                    hyponyms.append(lemma.name())
        return list(set(hyponyms))


    def find_patterns_VP(self,test_text):
        temp_verb_phrases_chunk=[]
        pos_doc = textacy.make_spacy_doc(test_text, lang="en_core_web_sm")
        patterns_VP =  [{"POS": "ADV"}, {"POS": "VERB"}]
        verb_phrases = textacy.extract.token_matches(pos_doc, patterns=patterns_VP)
        assert verb_phrases !=0
        for chunk in verb_phrases: 
                temp = str(chunk).split(' ')
                temp = [x.strip() for x in temp]
                if len(list(set(temp).intersection(set(stop_words))))==0:
                    temp_verb_phrases_chunk.append(str(chunk))
        return temp_verb_phrases_chunk


    def find_patterns_NP(self,test_text):
        temp_noun_phrases_chunk=[]
        pos_doc = textacy.make_spacy_doc(test_text, lang="en_core_web_sm")
        patterns_NP = [  {"POS": "ADJ"}, {"POS": "NOUN"}]
        noun_phrases = textacy.extract.token_matches(pos_doc, patterns=patterns_NP)
        assert noun_phrases !=0
        for chunk in noun_phrases:    
                temp = str(chunk).split(' ')#trans  spacy.tokens.span.Span to str ->list
                temp = [x.strip() for x in temp]
                if len(list(set(temp).intersection(set(stop_words))))==0:
                    temp_noun_phrases_chunk.append(str(chunk))

        return temp_noun_phrases_chunk


    def get_index(self,lst=None, item=''):
        return [index for (index,value) in enumerate(lst) if value == item]
    
    
    def replace_word_in_sentence(self,sentence, target_word, replacement_word):
        # 使用正则表达式匹配整个单词，并进行替换
        pattern = r'\b' + re.escape(target_word) + r'\b'
        new_sentence = re.sub(pattern, replacement_word, sentence)

        return new_sentence

    def generate_random_number(self,exclude_number):
        numbers = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
        while True:
            random_number = random.choice(numbers)
            if random_number != exclude_number:
                return random_number

    def change_word(self,sent,change_num=16):    
        temp_list =[]
        temp_TAG_list=[]
        #1. pos找到句子中的VP NP verb noun
        sent = ' '.join(sent)
        # print(sent)
        temp_doc = nlp(sent)
        word_list = [w for w in temp_doc]
        temp_tags = [w.pos_ for w in temp_doc]
        #仅需要  'ADJ','NOUN', 'ADV', 'VERB',
        # sent_pos = list(set(temp_tags).intersection(set(['ADJ','NOUN', 'ADV', 'VERB','ADP','NUM'])))
        sent_pos = list(set(temp_tags).intersection(set(['ADJ','NOUN', 'ADV', 'VERB','ADP'])))
        temp_list.append(sent)
        while len(temp_list)!=change_num+1:
            if len(sent_pos)>0:
                t_pos_dic={}
                for i in sent_pos:
                    for j in self.get_index(temp_tags, item=i):
                        t_pos_dic[i]= [word_list[j] for j in self.get_index(temp_tags, item=i)]
                pos_item = random.choices(list(t_pos_dic.keys()))[0]

                if pos_item=='ADP':
                    word = random.choices(self.remove_words_with_symbols(t_pos_dic[pos_item]))
                    if len(self.find_synonyms_antonyms(str(word)))!=0:
                        if len(self.remove_words_with_symbols(self.find_synonyms_antonyms(str(word))))!=0:
    
                            repalce_word = self.remove_words_with_symbols(self.find_synonyms_antonyms(str(word)))#不管传入进去的词是什么最后返回一个词就对了
                            new_sent =  self.replace_word_in_sentence(sent,str(word),random.choice(repalce_word))#去除部分匹配问题
                            if new_sent not in temp_list:
                                temp_list.append(new_sent)
                                temp_TAG_list.append(pos_item)
                      
                    else:
                        repalce_word = random.choice(self.vocab[pos_item.lower()])#根据index找到改词语的pos type,再随机选择一个词替换
                        new_sent =  self.replace_word_in_sentence(sent,str(word),repalce_word)#去除部分匹配问题
                        if new_sent not in temp_list:
                            temp_list.append(new_sent)
                            temp_TAG_list.append(pos_item)

               
                elif  pos_item=='NUM':
                    word = random.choices(self.remove_words_with_symbols(t_pos_dic[pos_item]))
                    repalce_word = self.generate_random_number(word)
                    new_sent =  self.replace_word_in_sentence(sent,str(word),repalce_word)#去除部分匹配问题

                    if new_sent not in temp_list:
                        temp_list.append(new_sent)
                        temp_TAG_list.append(pos_item)
            
                else:
                
                    word = random.choices(self.remove_words_with_symbols(t_pos_dic[pos_item]))

                    if len(self.find_synonyms_antonyms(str(word)))!=0:

                        if len(self.remove_words_with_symbols(self.find_synonyms_antonyms(str(word))))!=0:

                            repalce_word = self.remove_words_with_symbols(self.find_synonyms_antonyms(str(word)))#不管传入进去的词是什么最后返回一个词就对了
                            new_sent =  self.replace_word_in_sentence(sent,str(word),random.choice(repalce_word))#去除部分匹配问题
                            # new_sent = sent.replace(str(word),random.choice(repalce_word))
                            if new_sent not in temp_list:
                                temp_list.append(new_sent)
                                temp_TAG_list.append(pos_item)

                    elif len(self.get_hypernyms(str(word),pos_item=pos_item))!=0:
                        temp_hyponyms=[]
                        for i in self.remove_words_with_symbols(self.get_hypernyms(str(word),pos_item=pos_item)):
                            if len(self.remove_words_with_symbols(self.get_hyponyms(i,pos_item=pos_item)))!=0:
                                temp_hyponyms.extend(self.remove_words_with_symbols(self.get_hyponyms(i,pos_item=pos_item)))
                        if len(temp_hyponyms)!=0: 
                            repalce_word = random.choice(temp_hyponyms)#不管传入进去的词是什么最后返回一个词就对了   
                            new_sent =  self.replace_word_in_sentence(sent,str(word),repalce_word)                
                            # new_sent = sent.replace(str(word),repalce_word)
                            if new_sent not in temp_list:
                                temp_list.append(new_sent)
                                temp_TAG_list.append(pos_item)

                    else:
                        random_choice_pos= random.choice(list(t_pos_dic.keys()))#找到随机替换的词语的index

                        randm_choice_word = word_list[random.choice(self.get_index(temp_tags, item=random_choice_pos))]

                        # repalce_word = random.choice(self.vocab[random_choice_pos.lower()])#根据index找到改词语的pos type,再随机选择一个词替换
                        repalce_word = random.choice(self.vocab[random_choice_pos.lower()])#根据index找到改词语的pos type,再随机选择一个词替换
                        new_sent =  self.replace_word_in_sentence(sent,str(randm_choice_word),repalce_word) 
                        # new_sent = sent.replace(str(randm_choice_word),repalce_word)
                        if new_sent not in temp_list:
                            temp_list.append(new_sent)
                            temp_TAG_list.append(random_choice_pos)
            else:
                random_choice_idx= word_list.index(random.choice(word_list))#找到随机替换的词语的index
                random_choice_pos= random.choice(list(self.vocab.keys()))
                temp_TAG_list.append(random_choice_pos)
                repalce_word = random.choice(self.vocab[random_choice_pos])#根据index找到改词语的pos type,再随机选择一个词替换
                new_sent =  self.replace_word_in_sentence(sent,str(word_list[random_choice_idx]),repalce_word) 
                # new_sent = sent.replace(str(word_list[random_choice_idx]),repalce_word)
                if new_sent not in temp_list:
                    temp_list.append(new_sent)
                    temp_TAG_list.append(random_choice_pos)
                
                            
        return temp_list[1:],temp_TAG_list
'''

get_senteces= Get_Negative_text_samples(dataset_based_vocab_path)
get_neg_word_level_sent_fun = get_senteces.change_word
get_neg_phrase_level_sent_fun = get_senteces.change_phrase
# get_neg_sent_fun = get_senteces.change_random_phrase


class UVO_TrainDataLoader(Dataset):
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
        video_id_path_dict["train"] = os.path.join(self.data_path, "vln_UVO_train.txt")
        video_id_path_dict["val"] = os.path.join(self.data_path, "vln_UVO_val.txt")
        video_id_path_dict["test"] = os.path.join(self.data_path, "vln_UVO_val.txt")
        caption_file = os.path.join(self.data_path, "vln_uvo.pkl")

        with open(video_id_path_dict[self.subset], 'r') as fp:
            video_ids = [itm.strip() for itm in fp.readlines()]

        with open(caption_file, 'rb') as f:
            captions = pickle.load(f)

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
                # cap_txt = " ".join(cap)
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
        neg_word_level_sents,change_word_pos = get_neg_word_level_sent_fun(caption,change_num=change_num)#generated K sentences but only use K-1 as hard negative samples
        ## 输出 word-level and phrase-level
        word_text_neg = np.zeros((k, change_num, self.max_words), dtype=np.int64)
        word_mask_neg = np.zeros((k, change_num, self.max_words), dtype=np.int64)
        word_segment_neg = np.zeros((k,change_num, self.max_words), dtype=np.int64)
        word_neg_sents,word_change_pos = get_neg_word_level_sent_fun(caption,change_num=change_num)#generated K sentences but only use K-1 as hard negative samples

        phrase_text_neg = np.zeros((k, change_num, self.max_words), dtype=np.int64)
        phrase_mask_neg = np.zeros((k, change_num, self.max_words), dtype=np.int64)
        phrase_segment_neg = np.zeros((k,change_num, self.max_words), dtype=np.int64)
        phrase_neg_sents,phrase_change_pos = get_neg_phrase_level_sent_fun(caption,change_num=change_num)#generated K sentences but only use K-1 as hard negative samples

        self.temp_wr.write(' '.join(caption)+'\n')
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
            caption =' '.join(caption)
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
            ## negative 

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)
            for j, t_neg_sent in enumerate(word_neg_sents):
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
