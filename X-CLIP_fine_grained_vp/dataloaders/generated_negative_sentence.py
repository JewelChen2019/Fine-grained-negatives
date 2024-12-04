# -*- coding: utf-8 -*-
import json
from gensim.models import KeyedVectors
import re
import nltk
import time 
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.tokenize import word_tokenize

# 下载 NLTK 数据
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

import random
# 下载WordNet语料库
# nltk.download('wordnet')
import spacy
from spacy import displacy
import textacy
cls = spacy.util.get_lang_class('en')
stop_words = cls.Defaults.stop_words
# nlp = spacy.load("en_core_web_sm")


# # 1.Using VP/NP in other sentences to replace it.
# # 2.Randomly replace the noun and adjective （verb and adverb） respectively

# 将 名词 动词 形容词 副词 都选出来 一个句子形成一个字典，以所选词性 为key ，词为value, 每次改都选择不同的词性中的某个词

# 每次都修改名词和动词以及其前面的单词


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
    
    def remove_backslashes(self,input_string):
    # 使用字符串的 replace 方法将反斜杠替换为空字符串
        # input_string =' '.join(input_string)
        cleaned_string = input_string.replace('\\', '')
        # cleaned_string =cleaned_string.split(' ')
        return cleaned_string
    

    def remove_dashes(self,input_string):
        # input_string =' '.join(input_string)
        # 使用字符串的 replace 方法将反斜杠替换为空字符串
        cleaned_string = input_string.replace('-', '')
        # cleaned_string =cleaned_string.split(' ')
        
        return cleaned_string




    
    def find_synonyms_antonyms(self,word):
        # 下载 WordNet 数据（仅首次运行需要）
        # nltk.download('wordnet')

        # 获取输入词的所有词义
        # print('****')
        # print(type(word))
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

        else:
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
        else:
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
    
    def get_pos(self,word):
        # 对单词进行标记
        tagged_word = pos_tag(word_tokenize(word))
        
        # 获取标记后的词性
        pos = tagged_word[0][1]
        
        # 映射 NLTK 的标签到通用的标签
        pos_mapping = {
            'N': 'NOUN',
            'V': 'VERB',
            'R': 'ADV',
            'J': 'ADJ'
        }
        
        # 获取通用的标签
        general_pos = pos_mapping.get(pos[0], pos)
        
        return general_pos
    
    def replaced_word(self,word):
      
        if len(self.find_synonyms_antonyms(word))!=0:
            new_word = random.choice(self.find_synonyms_antonyms(word))
    #         print('0',new_word )
            
        elif len(self.get_hypernyms(word,pos_item=self.get_pos(word)))!=0:
            temp_hyper = random.choice(self.get_hypernyms(word,pos_item=self.get_pos(word)))
            if len(self.get_hyponyms(temp_hyper,pos_item=(self.get_pos(temp_hyper))))!=0:
                temp_rep_list=self.get_hyponyms(temp_hyper,pos_item=(self.get_pos(temp_hyper)))
                temp_rep_list.pop(word)
                new_word =random.choice(temp_rep_list)
    #             print('1',new_word )
            else:
                temp_rep_list =  random.choice(self.vocab[self.get_pos(word).lower()])
                new_word = random.choice(temp_rep_list)
    #             print('2',new_word )
        else:
                temp_rep_list = self.vocab[self.get_pos(word).lower()]
                new_word = random.choice(temp_rep_list) 
    #             print('3',new_word )
        
        return new_word





    def generate_random_number(self,exclude_number):
        numbers = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
        while True:
            random_number = random.choice(numbers)
            if random_number != exclude_number:
                return random_number
            

    ## 实现一个函数修改句子中的phrase
    ## 这里的find_patterns_NP中识别不了‘-’，所以修改句子中的短横线“-”为下划线"_""
    # def replaced_word(self,word):
        
    #     if len(self.find_synonyms_antonyms(word))!=0:
    #         new_word = random.choice(self.find_synonyms_antonyms(word))
    # #         print('0',new_word )
            
    #     elif len(self.get_hypernyms(word,pos_item=self.get_pos(word)))!=0:
    #         temp_hyper = random.choice(self.get_hypernyms(word,pos_item=self.get_pos(word)))
    #         if len(self.get_hyponyms(temp_hyper,pos_item=(self.get_pos(temp_hyper))))!=0:
    #             temp_rep_list=self.get_hyponyms(temp_hyper,pos_item=(self.get_pos(temp_hyper)))
    #             # temp_rep_list.pop(word)
    #             new_word =random.choice(temp_rep_list)
    # #             print('1',new_word )
    #         else:
    #             temp_rep_list =  random.choice(self.vocab[self.get_pos(word).lower()])
    #             new_word = random.choice(temp_rep_list)
    # #             print('2',new_word )
    #     else:
    #             temp_rep_list = self.vocab[self.get_pos(word).lower()]
    #             new_word = random.choice(temp_rep_list) 
    # #             print('3',new_word )
        
    #     return new_word


    def find_closest_pos_word_by_order(self,sentence, target_word, target_pos='NOUN'):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(sentence)
        target_token = None
        for token in doc:
            if token.text == target_word:
                target_token = token
                break

        if target_token is None:
            return None

        closest_pos_word = None
        min_distance = float('inf')

        for token in doc:
            if token.pos_ == target_pos:
                distance = abs(target_token.i - token.i)
                if distance < min_distance and token.text != target_word:
                    min_distance = distance
                    closest_pos_word = token.text

        return closest_pos_word


    def replaced_word(self,word):
        # 如果每次选的词首先都根据其反义词返回，这样的话，如果其反义词就一个， 就会一直返回一样的词语，因此需要想想怎么设计遍历，BFS or DFS?
     
        # new_words_list=[]
        if len(self.find_synonyms_antonyms(word))!=0:
            new_word = random.choice(self.find_synonyms_antonyms(word))
            # print(new_word)
            # new_words_list.append(new_word)
            # print('0',new_word )

        elif len(self.get_hypernyms(word,pos_item=self.get_pos(word)))!=0:
            temp_hyper = random.choice(self.get_hypernyms(word,pos_item=self.get_pos(word)))
            if len(self.get_hyponyms(temp_hyper,pos_item=(self.get_pos(temp_hyper))))!=0:
                # print(self.get_pos(temp_hyper))
                new_word=random.choice(self.get_hyponyms(temp_hyper,pos_item=(self.get_pos(temp_hyper))))
                if new_word==word:
                    if self.get_pos(word).lower() == 'in':
                        new_word =  random.choice(self.vocab['adp'])
                        
                    elif self.get_pos(word).lower() == 'cd':
                        new_word =  random.choice(self.vocab['num'])
                    else:
                        new_word =  random.choice(self.vocab[self.get_pos(word).lower()])
                        
               
                # print('1',new_word )
            else:
                random_pos = random.choice(['ADJ','NOUN', 'ADV', 'VERB','ADP'])
                new_word =  random.choice(self.vocab[random_pos.lower()])
                
                # new_words_list.append(new_word)
                # print('2',new_word )
        else:
            random_pos = random.choice(['ADJ','NOUN', 'ADV', 'VERB','ADP'])
            new_word  =  random.choice(self.vocab[random_pos.lower()])
            # new_words_list.append( new_word )
            # print('3',new_word)
            
          
        
        # return_word =random.choice(new_words_list)

        return new_word


    def change_random_phrase(self,sent,change_num=16,num_pos=2):    
        # sent = ' '.join(sent)
        sent = self.remove_dashes(self.remove_backslashes(sent))
        temp_list =[]
        temp_TAG_list=[]
        #1. pos找到句子中的VP NP verb noun
        # sent = ' '.join(sent)
        # print('##'*10)
        # print(sent)
        temp_doc = nlp(sent)
        word_list = [w for w in temp_doc]
        temp_tags = [w.pos_ for w in temp_doc]
        #仅需要  'ADJ','NOUN', 'ADV', 'VERB',
        sent_pos = list(set(temp_tags).intersection(set(['ADJ','NOUN', 'ADV', 'VERB','ADP'])))
        temp_list.append(sent)
        if len(sent_pos)>0:
            t_pos_dic={}
            for i in sent_pos:
                for j in self.get_index(temp_tags, item=i):
                    t_pos_dic[i]= [word_list[j] for j in self.get_index(temp_tags, item=i)]
                    
            while len(temp_list)!=change_num+1:
                if len(temp_list)>=change_num+1:
                    break
        #         print(list(t_pos_dic.keys()))
                if len(list(t_pos_dic.keys())) >=2:
                    # print("stop here")
                    # print(sent)
                    # print(len(temp_list),change_num)
                    # print(temp_list)
                    pos_item1,pos_item2 = random.sample(list(t_pos_dic.keys()),k=num_pos)
                    # print("pos_item1,pos_item2")
                    ticks = time.time()
                    # print("当前时间戳为:", ticks)
                    #选定第一个词性的单词，再根据第一个词性选中的单词，选择其最近的'ADJ','NOUN', 'ADV', 'VERB','ADP'
                    temp_words1 = str(random.choice(t_pos_dic[pos_item1]))
                    temp_words2= str(self.find_closest_pos_word_by_order(sent,temp_words1, target_pos=pos_item2))
                    # print("--------")
                    # print('temp_words1,temp_words2',temp_words1,temp_words2)
                    # print("--------")
                    new_w1=self.replaced_word(temp_words1)
                    # print(temp_words2)
                    new_w2=self.replaced_word(temp_words2)
                    # print("--------")
                    # print('temp_words1,new_w1',temp_words1,new_w1)
                    # print('temp_words2,new_w2',temp_words2,new_w2)
                    # print("--------") 
                    new_sent =  self.replace_word_in_sentence(sent,temp_words1 ,self.remove_backslashes(new_w1)) 
                    new_sent =  self.replace_word_in_sentence(new_sent,temp_words2,self.remove_backslashes(new_w2)) 
                    # print(new_sent)
                    random_choice_pos =pos_item1+' ' +pos_item2 
                    if new_sent not in temp_list and len(temp_list)!=change_num+1:
                            temp_list.append(new_sent)
                            temp_TAG_list.append(random_choice_pos)
                    if len(temp_list)==change_num+1:
                        break
                    
                    # if len(sent.split(' '))<=5:
                    if len(temp_list)<change_num+1:
                        random_choice1,random_choice2 = random.sample(word_list,2)
                        random_choice_idx2=  word_list.index(random_choice1)#找到随机替换的词语的index
                        random_choice_idx1 = word_list.index(random_choice2)
                        random_choice_pos1,random_choice_pos2= random.sample(list(self.vocab.keys()),2)
                        random_choice_pos = random_choice_pos1+" "+random_choice_pos2
                        temp_TAG_list.append(random_choice_pos)
                        repalce_word1 = random.choice(self.vocab[random_choice_pos1])#根据index找到改词语的pos type,再随机选择一个词替换
                        repalce_word2 = random.choice(self.vocab[random_choice_pos2])#根据index找到改词语的pos type,再随机选择一个词替换
                        new_sent =  self.replace_word_in_sentence(sent,str(word_list[random_choice_idx1]),repalce_word1) 
                        new_sent =  self.replace_word_in_sentence(new_sent,str(word_list[random_choice_idx2]),repalce_word2) 
                        if new_sent not in temp_list and len(temp_list)!=change_num+1:
                            temp_list.append(new_sent)
                            temp_TAG_list.append(random_choice_pos)
                            if len(temp_list)==change_num+1:
                                break



                else:
                    if len(temp_list)>=change_num+1:
                        break
                    # print("***")
                    # print(sent)
                    pos_item1,pos_item2 = list(t_pos_dic.keys())[0],list(t_pos_dic.keys())[0]
                    temp_words1 = str(random.choice(t_pos_dic[pos_item1]))
                    temp_words2= str(self.find_closest_pos_word_by_order(sent,temp_words1, target_pos=pos_item2))
                    new_w1=self.replaced_word(temp_words1)
                    # print(temp_words2)
                    new_w2=self.replaced_word(temp_words2)
                    new_sent =  self.replace_word_in_sentence(sent,temp_words1 ,self.remove_backslashes(new_w1)) 
                    new_sent =  self.replace_word_in_sentence(new_sent,temp_words2,self.remove_backslashes(new_w2)) 

                    # print(new_sent)
                    random_choice_pos =pos_item1+' ' +pos_item2 
                    if new_sent not in temp_list and len(temp_list)!=change_num+1:
                            temp_list.append(new_sent)
                            temp_TAG_list.append(random_choice_pos)
                            if len(temp_list)>=change_num+1:
                                break

                    if len(temp_list)<change_num+1:
                        random_choice1,random_choice2 = random.sample(word_list,2)
                        random_choice_idx2=  word_list.index(random_choice1)#找到随机替换的词语的index
                        random_choice_idx1 = word_list.index(random_choice2)
                        random_choice_pos1,random_choice_pos2= random.sample(list(self.vocab.keys()),2)
                        random_choice_pos = random_choice_pos1+" "+random_choice_pos2
                        temp_TAG_list.append(random_choice_pos)
                        repalce_word1 = random.choice(self.vocab[random_choice_pos1])#根据index找到改词语的pos type,再随机选择一个词替换
                        repalce_word2 = random.choice(self.vocab[random_choice_pos2])#根据index找到改词语的pos type,再随机选择一个词替换
                        new_sent =  self.replace_word_in_sentence(sent,str(word_list[random_choice_idx1]),repalce_word1) 
                        new_sent =  self.replace_word_in_sentence(new_sent,str(word_list[random_choice_idx2]),repalce_word2) 
                        if new_sent not in temp_list and len(temp_list)!=change_num+1:
                            temp_list.append(new_sent)
                            temp_TAG_list.append(random_choice_pos)
                            if len(temp_list)==change_num+1:
                                break


                if len(temp_list)>=change_num+1:
                    break
            return temp_list[1:],temp_TAG_list
        else:
            while len(temp_list)!=change_num+1:
                if len(temp_list)>=change_num+1:
                    break
                random_choice1,random_choice2 = random.sample(word_list,2)
                random_choice_idx2=  word_list.index(random_choice1)#找到随机替换的词语的index
                random_choice_idx1 = word_list.index(random_choice2)
                random_choice_pos1,random_choice_pos2= random.sample(list(self.vocab.keys()),2)
                random_choice_pos = random_choice_pos1+" "+random_choice_pos2
                temp_TAG_list.append(random_choice_pos)
                repalce_word1 = random.choice(self.vocab[random_choice_pos1])#根据index找到改词语的pos type,再随机选择一个词替换
                repalce_word2 = random.choice(self.vocab[random_choice_pos2])#根据index找到改词语的pos type,再随机选择一个词替换
                new_sent =  self.replace_word_in_sentence(sent,str(word_list[random_choice_idx1]),repalce_word1) 
                new_sent =  self.replace_word_in_sentence(new_sent,str(word_list[random_choice_idx2]),repalce_word2) 
                if new_sent not in temp_list and len(temp_list)!=change_num+1:
                    temp_list.append(new_sent)
                    temp_TAG_list.append(random_choice_pos)
                    if len(temp_list)==change_num+1:
                        break
            return temp_list[1:],temp_TAG_list

        


    def change_phrase(self,sent,change_num=16): 
        
        sent = self.remove_dashes(self.remove_backslashes(sent))

        np_list = self.find_patterns_NP(sent)
        vp_list = self.find_patterns_VP(sent)
    
        temp_list =[]
        temp_TAG_list=[]
        temp_doc = nlp(sent)
        word_list = [w for w in temp_doc]
        temp_tags = [w.pos_ for w in temp_doc]
        sent_pos = list(set(temp_tags).intersection(set(['ADJ','NOUN', 'ADV', 'VERB','ADP'])))
        temp_list.append(sent)

        while len(temp_list)<=change_num+1:
            # print('-----temp_list-----')
            # print(len(temp_list))
            
            
            if len(np_list)!=0:
                counter_np=0
                for t_phrase in np_list:
                    if ' ' in t_phrase:
                        counter_np+=1
                        w1,w2 =t_phrase.split(' ') 
        #                 print(w1,'*',w2)
                        new_w1=self.replaced_word(w1)
        #                 print(new_w1)
                        new_w2=self.replaced_word(w2)
        #                 print(new_w2)
                        new_phrase=new_w1 + ' '+new_w2
                        new_sent =  self.replace_word_in_sentence(sent,str(t_phrase),self.remove_backslashes(new_phrase))  
                        # print('0 is dead')
                        if new_sent not in temp_list and len(temp_list)!=change_num+1:
                            temp_list.append(new_sent)
                            temp_TAG_list.append('NP')
                            if len(temp_list)==change_num+1:
                                break
                        if counter_np == len(np_list):
                            break
                        if len(temp_list)==change_num+1:
                                break
                    
        #                     print('---',temp_list)

            if len(vp_list)!=0:
                counter_vp=0
                for t_phrase in vp_list:
                    if ' ' in t_phrase:
                        counter_vp+=1
                        w1,w2 =t_phrase.split(' ') 
        #                 print(w1,'*',w2)
                        new_w1=self.replaced_word(w1)
                        new_w2=self.replaced_word(w2)
                        new_phrase=new_w1 + ' '+new_w2
                        new_sent =  self.replace_word_in_sentence(sent,str(t_phrase),new_phrase)  
                        # print('1 is dead')
                        if new_sent not in temp_list and len(temp_list)!=change_num+1:
                            temp_list.append(new_sent)
                            temp_TAG_list.append('VP')
                            if len(temp_list)==change_num+1:
                                break
                        if len(temp_list)==change_num+1:
                                break
                        if counter_vp == len(vp_list):
                            break

                if len(temp_list)==change_num+1:
                    break 
            if len(temp_list)==change_num+1:
                    break 

            if len(temp_list)!=change_num+1:
                # print('2 is dead')
               
                random_choice1,random_choice2 = random.sample(word_list,2)
                random_choice_idx2=  word_list.index(random_choice1)#找到随机替换的词语的index
                random_choice_idx1 = word_list.index(random_choice2)
                random_choice_pos1,random_choice_pos2= random.sample(list(self.vocab.keys()),2)
                random_choice_pos = random_choice_pos1+" "+random_choice_pos2
                temp_TAG_list.append(random_choice_pos)
                repalce_word1 = random.choice(self.vocab[random_choice_pos1])#根据index找到改词语的pos type,再随机选择一个词替换
                repalce_word2 = random.choice(self.vocab[random_choice_pos2])#根据index找到改词语的pos type,再随机选择一个词替换
                new_sent =  self.replace_word_in_sentence(sent,str(word_list[random_choice_idx1]),repalce_word1) 
                new_sent =  self.replace_word_in_sentence(new_sent,str(word_list[random_choice_idx2]),repalce_word2) 
                if new_sent not in temp_list and len(temp_list)!=change_num+1:
                    temp_list.append(new_sent)
                    temp_TAG_list.append(random_choice_pos)
                    if len(temp_list)==change_num+1:
                        break
                if len(temp_list)==change_num+1:
                    break 


            if len(temp_list)==change_num+1:
                break
                

        return temp_list[1:],temp_TAG_list


    def change_word(self,sent,change_num=16):    
        temp_list =[]
        temp_TAG_list=[]
        #1. pos找到句子中的VP NP verb noun
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
                    # print("adp")
                    if len(temp_list) ==change_num+1:
                                 break
                    if len(self.remove_words_with_symbols(t_pos_dic[pos_item])) != 0:
                        word = random.choices(self.remove_words_with_symbols(t_pos_dic[pos_item]))
                        if len(self.find_synonyms_antonyms(str(word)))!=0 and len(temp_list) <change_num+1 :                         
                            if len(self.remove_words_with_symbols(self.find_synonyms_antonyms(str(word))))!=0:
        
                                repalce_word = self.remove_words_with_symbols(self.find_synonyms_antonyms(str(word)))#不管传入进去的词是什么最后返回一个词就对了
                                new_sent =  self.replace_word_in_sentence(sent,str(word),random.choice(repalce_word))#去除部分匹配问题
                                if new_sent not in temp_list:
                                    temp_list.append(new_sent)
                                    temp_TAG_list.append(pos_item)
                                
                                
                      
                        else:
                            if len(temp_list) < change_num+1:
                                repalce_word = random.choice(self.vocab[pos_item.lower()])#根据index找到改词语的pos type,再随机选择一个词替换
                                new_sent =  self.replace_word_in_sentence(sent,str(word),repalce_word)#去除部分匹配问题
                                if new_sent not in temp_list:
                                    temp_list.append(new_sent)
                                    temp_TAG_list.append(pos_item)
                            else:
                                pass

                            
                    else:
                        if len(temp_list) <=change_num+1:
                            

                            random_choice_idx= word_list.index(random.choice(word_list))#找到随机替换的词语的index
                            random_choice_pos= random.choice(list(self.vocab.keys()))
                            temp_TAG_list.append(random_choice_pos)
                            repalce_word = random.choice(self.vocab[random_choice_pos])#根据index找到改词语的pos type,再随机选择一个词替换
                            new_sent =  self.replace_word_in_sentence(sent,str(word_list[random_choice_idx]),repalce_word) 
                            # new_sent = sent.replace(str(word_list[random_choice_idx]),repalce_word)
                            if new_sent not in temp_list:
                                temp_list.append(new_sent)
                                temp_TAG_list.append(random_choice_pos)
                        else:
                            pass


                    random_choice_idx= word_list.index(random.choice(word_list))#找到随机替换的词语的index
                    random_choice_pos= random.choice(list(self.vocab.keys()))
                    temp_TAG_list.append(random_choice_pos)
                    repalce_word = random.choice(self.vocab[random_choice_pos])#根据index找到改词语的pos type,再随机选择一个词替换
                    new_sent =  self.replace_word_in_sentence(sent,str(word_list[random_choice_idx]),repalce_word) 
                    # new_sent = sent.replace(str(word_list[random_choice_idx]),repalce_word)
                    if new_sent not in temp_list:
                        temp_list.append(new_sent)
                        temp_TAG_list.append(random_choice_pos)

                    if len(temp_list) ==change_num+1:
                        break

               
                elif  pos_item=='NUM':
                    # print("num")
                    word = random.choices(self.remove_words_with_symbols(t_pos_dic[pos_item]))
                    repalce_word = self.generate_random_number(word)
                    new_sent =  self.replace_word_in_sentence(sent,str(word),repalce_word)#去除部分匹配问题

                    if new_sent not in temp_list:
                        temp_list.append(new_sent)
                        temp_TAG_list.append(pos_item)
            
                else:
                    # print("1")
                    if len(self.remove_words_with_symbols(t_pos_dic[pos_item])) != 0:
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
                            # print("!!!!!!!")
                            random_choice_pos= random.choice(list(t_pos_dic.keys()))#找到随机替换的词语的index
                            randm_choice_word = word_list[random.choice(self.get_index(temp_tags, item=random_choice_pos))]
                            # print("**"*10)
                            # print(self.vocab.keys())
                            # repalce_word = random.choice(self.vocab[random_choice_pos.lower()])#根据index找到改词语的pos type,再随机选择一个词替换
                            repalce_word = random.choice(self.vocab[random_choice_pos.lower()])#根据index找到改词语的pos type,再随机选择一个词替换
                            
                            new_sent =  self.replace_word_in_sentence(sent,str(randm_choice_word),repalce_word) 
                            # new_sent = sent.replace(str(randm_choice_word),repalce_word)
                            if new_sent not in temp_list:
                                temp_list.append(new_sent)
                                temp_TAG_list.append(random_choice_pos)

                        if len(temp_list) ==change_num+1:
                            break


                    else:
                        # print("2")
                        random_choice_idx= word_list.index(random.choice(word_list))#找到随机替换的词语的index
                        random_choice_pos= random.choice(list(self.vocab.keys()))
                        temp_TAG_list.append(random_choice_pos)
                        repalce_word = random.choice(self.vocab[random_choice_pos])#根据index找到改词语的pos type,再随机选择一个词替换
                        new_sent =  self.replace_word_in_sentence(sent,str(word_list[random_choice_idx]),repalce_word) 
                        # new_sent = sent.replace(str(word_list[random_choice_idx]),repalce_word)
                        if new_sent not in temp_list:
                            temp_list.append(new_sent)
                            temp_TAG_list.append(random_choice_pos)

                    if len(temp_list) ==change_num+1:
                        break

                if len(temp_list) ==change_num+1:
                    break

            else:
                # print("3")
                random_choice_idx= word_list.index(random.choice(word_list))#找到随机替换的词语的index
                random_choice_pos= random.choice(list(self.vocab.keys()))
                temp_TAG_list.append(random_choice_pos)
                repalce_word = random.choice(self.vocab[random_choice_pos])#根据index找到改词语的pos type,再随机选择一个词替换
                new_sent =  self.replace_word_in_sentence(sent,str(word_list[random_choice_idx]),repalce_word) 
                # new_sent = sent.replace(str(word_list[random_choice_idx]),repalce_word)
                if new_sent not in temp_list:
                    temp_list.append(new_sent)
                    temp_TAG_list.append(random_choice_pos)

            if len(temp_list) ==change_num+1:
                        break  
                            
        return temp_list[1:],temp_TAG_list


# get_senteces= Get_Negative_text_samples(dataset_based_vocab_path)
# get_neg_sent_fun = get_senteces.change_word




# # -*- coding: utf-8 -*-
# import json
# from gensim.models import KeyedVectors
# import re
# import nltk
# from nltk.corpus import wordnet
# from nltk import pos_tag
# from nltk.tokenize import word_tokenize

# # 下载 NLTK 数据
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

# import random
# # 下载WordNet语料库
# nltk.download('wordnet')
# import spacy
# from spacy import displacy
# import textacy
# cls = spacy.util.get_lang_class('en')
# stop_words = cls.Defaults.stop_words
# nlp = spacy.load("en_core_web_sm")


# # # 1.Using VP/NP in other sentences to replace it.
# # # 2.Randomly replace the noun and adjective （verb and adverb） respectively

# # 将 名词 动词 形容词 副词 都选出来 一个句子形成一个字典，以所选词性 为key ，词为value, 每次改都选择不同的词性中的某个词

# # 每次都修改名词和动词以及其前面的单词


# class Get_Negative_text_samples():
#     def __init__(
#             self,
        
#             dataset_based_vocab_path,
#     ):
    
#         self.vocab = json.load(open(dataset_based_vocab_path, 'r'))


#     def remove_words_with_symbols(self,word_list):
#     # 定义正则表达式模式匹配非字母和非数字字符
#         pattern = re.compile(r'[^a-zA-Z0-9]')
#         word_list = [str(word) for word in word_list]

#         # 过滤列表中带有符号的单词
    
#         filtered_list = [word for word in word_list if not pattern.search(word)]

#         return filtered_list
    
#     def remove_backslashes(self,input_string):
#     # 使用字符串的 replace 方法将反斜杠替换为空字符串
#         # input_string =' '.join(input_string)
#         cleaned_string = input_string.replace('\\', '')
#         # cleaned_string =cleaned_string.split(' ')
#         return cleaned_string
    

#     def remove_dashes(self,input_string):
#         # input_string =' '.join(input_string)
#         # 使用字符串的 replace 方法将反斜杠替换为空字符串
#         cleaned_string = input_string.replace('-', '')
#         # cleaned_string =cleaned_string.split(' ')
        
#         return cleaned_string







    
#     def find_synonyms_antonyms(self,word):
#         # 下载 WordNet 数据（仅首次运行需要）
#         # nltk.download('wordnet')

#         # 获取输入词的所有词义
#         # print('****')
#         # print(type(word))
#         synsets = wordnet.synsets(word)

#         antonyms = []

#         # 遍历每个词义，提取其中的反义词
#         for synset in synsets:
          
#             for lemma in synset.lemmas():
#                 if lemma.antonyms():
#                     antonyms.extend(lemma.antonyms())

#         # 提取反义词列表
#         antonyms_list = [antonym.name() for antonym in antonyms]

#         return list(set(antonyms_list))
    



#     def get_hypernyms(self,word,pos_item='NOUN'):
#         if pos_item == 'ADV':
#             pos=wordnet.ADV 
#         elif pos_item == 'ADJ':
#             pos=wordnet.ADJ
#         elif pos_item == 'VERB':
#             pos=wordnet.VERB
#         elif pos_item == 'NOUN':
#             pos=wordnet.NOUN

#         else:
#             pos=wordnet.NOUN
    
        


#         synsets = wordnet.synsets(word, pos=pos)
#         hypernyms = []
#         for synset in synsets:
#             for hypernym in synset.hypernyms():
#                 for lemma in hypernym.lemmas():
#                     hypernyms.append(lemma.name())
#         return list(set(hypernyms))


#     def get_hyponyms(self,word,pos_item='NOUN'):
#         if pos_item == 'ADV':
#             pos=wordnet.ADV 
#         elif pos_item == 'ADJ':
#             pos=wordnet.ADJ
#         elif pos_item == 'VERB':
#             pos=wordnet.VERB
#         elif pos_item == 'NOUN':
#             pos=wordnet.NOUN
#         else:
#             pos=wordnet.NOUN
     

#         synsets = wordnet.synsets(word, pos=pos)
#         hyponyms = []
#         for synset in synsets:
#             for hyponym in synset.hyponyms():
#                 for lemma in hyponym.lemmas():
#                     hyponyms.append(lemma.name())
#         return list(set(hyponyms))


#     def find_patterns_VP(self,test_text):
#         temp_verb_phrases_chunk=[]
#         pos_doc = textacy.make_spacy_doc(test_text, lang="en_core_web_sm")
#         patterns_VP =  [{"POS": "ADV"}, {"POS": "VERB"}]
#         verb_phrases = textacy.extract.token_matches(pos_doc, patterns=patterns_VP)
#         assert verb_phrases !=0
#         for chunk in verb_phrases: 
#                 temp = str(chunk).split(' ')
#                 temp = [x.strip() for x in temp]
#                 if len(list(set(temp).intersection(set(stop_words))))==0:
#                     temp_verb_phrases_chunk.append(str(chunk))
#         return temp_verb_phrases_chunk


#     def find_patterns_NP(self,test_text):
#         temp_noun_phrases_chunk=[]
#         pos_doc = textacy.make_spacy_doc(test_text, lang="en_core_web_sm")
#         patterns_NP = [  {"POS": "ADJ"}, {"POS": "NOUN"}]
#         noun_phrases = textacy.extract.token_matches(pos_doc, patterns=patterns_NP)
#         assert noun_phrases !=0
#         for chunk in noun_phrases:    
#                 temp = str(chunk).split(' ')#trans  spacy.tokens.span.Span to str ->list
#                 temp = [x.strip() for x in temp]
#                 if len(list(set(temp).intersection(set(stop_words))))==0:
#                     temp_noun_phrases_chunk.append(str(chunk))

#         return temp_noun_phrases_chunk


#     def get_index(self,lst=None, item=''):
#         return [index for (index,value) in enumerate(lst) if value == item]
    
    
#     def replace_word_in_sentence(self,sentence, target_word, replacement_word):
#         # 使用正则表达式匹配整个单词，并进行替换
#         pattern = r'\b' + re.escape(target_word) + r'\b'
#         new_sentence = re.sub(pattern, replacement_word, sentence)

#         return new_sentence
    
#     def get_pos(self,word):
#         # 对单词进行标记
#         tagged_word = pos_tag(word_tokenize(word))
        
#         # 获取标记后的词性
#         pos = tagged_word[0][1]
        
#         # 映射 NLTK 的标签到通用的标签
#         pos_mapping = {
#             'N': 'NOUN',
#             'V': 'VERB',
#             'R': 'ADV',
#             'J': 'ADJ'
#         }
        
#         # 获取通用的标签
#         general_pos = pos_mapping.get(pos[0], pos)
        
#         return general_pos
    
#     def replaced_word(self,word):
      
#         if len(self.find_synonyms_antonyms(word))!=0:
#             new_word = random.choice(self.find_synonyms_antonyms(word))
#     #         print('0',new_word )
            
#         elif len(self.get_hypernyms(word,pos_item=self.get_pos(word)))!=0:
#             temp_hyper = random.choice(self.get_hypernyms(word,pos_item=self.get_pos(word)))
#             if len(self.get_hyponyms(temp_hyper,pos_item=(self.get_pos(temp_hyper))))!=0:
#                 temp_rep_list=self.get_hyponyms(temp_hyper,pos_item=(self.get_pos(temp_hyper)))
#                 temp_rep_list.pop(word)
#                 new_word =random.choice(temp_rep_list)
#     #             print('1',new_word )
#             else:
#                 temp_rep_list =  random.choice(self.vocab[self.get_pos(word).lower()])
#                 new_word = random.choice(temp_rep_list)
#     #             print('2',new_word )
#         else:
#                 temp_rep_list = self.vocab[self.get_pos(word).lower()]
#                 new_word = random.choice(temp_rep_list) 
#     #             print('3',new_word )
        
#         return new_word


#     #####################################for msrvtt#####################################
#     def get_words_with_same_pos(self,pos):
#         """
#         获取具有相同词性的随机单词列表
#         """
#         # 将spaCy的词性标签转换为WordNet的词性标签
#         pos_map = {
#             'NOUN': wordnet.NOUN,
#             'VERB': wordnet.VERB,
#             'ADJ': wordnet.ADJ,
#             'ADV': wordnet.ADV
#         }
#         wn_pos = pos_map.get(pos, wordnet.NOUN)  # 默认为名词
#         words = set()
#         # 从WordNet中获取相同词性的所有词
#         for synset in list(wordnet.all_synsets(pos=wn_pos)):
#             for lemma in synset.lemmas():
#                 words.add(lemma.name().replace('_', ' '))
#         return list(words)

#     def replace_random_word_with_same_pos(self, sentence):
#         """
#         随机选择句子中的一个词，用具有相同词性的任意词替换
#         """
#         doc = nlp(sentence)
#         words = [(token.text, token.pos_) for token in doc]
#         if not words:
#             return sentence  # 如果句子中没有单词，则返回原句子

#         # 随机选择一个词
#         word, pos = random.choice(words)
#         same_pos_words = self.get_words_with_same_pos(pos)
#         if not same_pos_words:
#             return sentence  # 如果没有找到相同词性的词，则返回原句子

#         # 从列表中随机选择一个词进行替换
#         replacement_word = random.choice(same_pos_words)
#         modified_sentence = sentence.replace(word, replacement_word, 1)
#         return modified_sentence


#     #####################################for msrvtt#####################################




#     def generate_random_number(self,exclude_number):
#         numbers = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
#         while True:
#             random_number = random.choice(numbers)
#             if random_number != exclude_number:
#                 return random_number
            

#     ## 实现一个函数修改句子中的phrase
#     ## 这里的find_patterns_NP中识别不了‘-’，所以修改句子中的短横线“-”为下划线"_""
#     # def replaced_word(self,word):
        
#     #     if len(self.find_synonyms_antonyms(word))!=0:
#     #         new_word = random.choice(self.find_synonyms_antonyms(word))
#     # #         print('0',new_word )
            
#     #     elif len(self.get_hypernyms(word,pos_item=self.get_pos(word)))!=0:
#     #         temp_hyper = random.choice(self.get_hypernyms(word,pos_item=self.get_pos(word)))
#     #         if len(self.get_hyponyms(temp_hyper,pos_item=(self.get_pos(temp_hyper))))!=0:
#     #             temp_rep_list=self.get_hyponyms(temp_hyper,pos_item=(self.get_pos(temp_hyper)))
#     #             # temp_rep_list.pop(word)
#     #             new_word =random.choice(temp_rep_list)
#     # #             print('1',new_word )
#     #         else:
#     #             temp_rep_list =  random.choice(self.vocab[self.get_pos(word).lower()])
#     #             new_word = random.choice(temp_rep_list)
#     # #             print('2',new_word )
#     #     else:
#     #             temp_rep_list = self.vocab[self.get_pos(word).lower()]
#     #             new_word = random.choice(temp_rep_list) 
#     # #             print('3',new_word )
        
#     #     return new_word


#     def find_closest_pos_word_by_order(self,sentence, target_word, target_pos='NOUN'):
#         nlp = spacy.load("en_core_web_sm")
#         doc = nlp(sentence)
#         target_token = None
#         for token in doc:
#             if token.text == target_word:
#                 target_token = token
#                 break

#         if target_token is None:
#             return None

#         closest_pos_word = None
#         min_distance = float('inf')

#         for token in doc:
#             if token.pos_ == target_pos:
#                 distance = abs(target_token.i - token.i)
#                 if distance < min_distance and token.text != target_word:
#                     min_distance = distance
#                     closest_pos_word = token.text

#         return closest_pos_word


#     def replaced_word(self,word):
#         # 如果每次选的词首先都根据其反义词返回，这样的话，如果其反义词就一个， 就会一直返回一样的词语，因此需要想想怎么设计遍历，BFS or DFS?
     
#         # new_words_list=[]
#         if len(self.find_synonyms_antonyms(word))!=0:
#             new_word = random.choice(self.find_synonyms_antonyms(word))
#             # print(new_word)
#             # new_words_list.append(new_word)
#             # print('0',new_word )

#         elif len(self.get_hypernyms(word,pos_item=self.get_pos(word)))!=0:
#             temp_hyper = random.choice(self.get_hypernyms(word,pos_item=self.get_pos(word)))
#             if len(self.get_hyponyms(temp_hyper,pos_item=(self.get_pos(temp_hyper))))!=0:
#                 # print(self.get_pos(temp_hyper))
#                 new_word=random.choice(self.get_hyponyms(temp_hyper,pos_item=(self.get_pos(temp_hyper))))
#                 if new_word==word:
#                     if self.get_pos(word).lower() == 'in':
#                         new_word =  random.choice(self.vocab['adp'])
                        
#                     elif self.get_pos(word).lower() == 'cd':
#                         new_word =  random.choice(self.vocab['num'])
#                     else:
#                         new_word =  random.choice(self.vocab[self.get_pos(word).lower()])
                        
               
#                 # print('1',new_word )
#             else:
#                 random_pos = random.choice(['ADJ','NOUN', 'ADV', 'VERB','ADP'])
#                 new_word =  random.choice(self.vocab[random_pos.lower()])
                
#                 # new_words_list.append(new_word)
#                 # print('2',new_word )
#         else:
#             random_pos = random.choice(['ADJ','NOUN', 'ADV', 'VERB','ADP'])
#             new_word  =  random.choice(self.vocab[random_pos.lower()])
#             # new_words_list.append( new_word )
#             # print('3',new_word)
            
          
        
#         # return_word =random.choice(new_words_list)

#         return new_word


#     def change_random_phrase(self,sent,change_num=16,num_pos=2):    
#         sent = ' '.join(sent)
#         sent = self.remove_dashes(self.remove_backslashes(sent))
#         temp_list =[]
#         temp_TAG_list=[]
#         #1. pos找到句子中的VP NP verb noun
#         # sent = ' '.join(sent)
#         # print('##'*10)
#         # print(sent)
#         temp_doc = nlp(sent)
#         word_list = [w for w in temp_doc]
#         temp_tags = [w.pos_ for w in temp_doc]
#         #仅需要  'ADJ','NOUN', 'ADV', 'VERB',
#         sent_pos = list(set(temp_tags).intersection(set(['ADJ','NOUN', 'ADV', 'VERB','ADP'])))
#         temp_list.append(sent)
#         if len(sent_pos)>0:
#             t_pos_dic={}
#             for i in sent_pos:
#                 for j in self.get_index(temp_tags, item=i):
#                     t_pos_dic[i]= [word_list[j] for j in self.get_index(temp_tags, item=i)]
#         while len(temp_list)!=change_num+1:
#     #         print(list(t_pos_dic.keys()))
#             pos_item1,pos_item2 = random.sample(list(t_pos_dic.keys()),k=num_pos)
            
#             #选定第一个词性的单词，再根据第一个词性选中的单词，选择其最近的'ADJ','NOUN', 'ADV', 'VERB','ADP'
#             temp_words1 = str(random.choice(t_pos_dic[pos_item1]))
#             temp_words2= str(self.find_closest_pos_word_by_order(sent,temp_words1, target_pos=pos_item2))
#             # print("--------")
#             # print('temp_words1,temp_words2',temp_words1,temp_words2)
#             # print("--------")
#             new_w1=self.replaced_word(temp_words1)
#             # print(temp_words2)
#             new_w2=self.replaced_word(temp_words2)
#             # print("--------")
#             # print('temp_words1,new_w1',temp_words1,new_w1)
#             # print('temp_words2,new_w2',temp_words2,new_w2)
#             # print("--------") 
#             new_sent =  self.replace_word_in_sentence(sent,temp_words1 ,self.remove_backslashes(new_w1)) 
#             new_sent =  self.replace_word_in_sentence(new_sent,temp_words2,self.remove_backslashes(new_w2)) 
#             # print(new_sent)
#             random_choice_pos =pos_item1+' ' +pos_item2 
#             if new_sent not in temp_list and len(temp_list)!=change_num+1:
#                     temp_list.append(new_sent)
#                     temp_TAG_list.append(random_choice_pos)
#                     if len(temp_list)==change_num+1:
#                         break
#             if len(temp_list)==change_num+1:
#                 break

#         return temp_list[1:],temp_TAG_list


#     def change_phrase(self,sent,change_num=16): 
#         # sent = ' '.join(sent)## msrvtt 不需要
#         # print(sent)
#         sent = self.remove_dashes(self.remove_backslashes(sent))

#         np_list = self.find_patterns_NP(sent)
#         vp_list = self.find_patterns_VP(sent)
    
#         temp_list =[]
#         temp_TAG_list=[]
#         temp_doc = nlp(sent)
#         word_list = [w for w in temp_doc]
#         temp_tags = [w.pos_ for w in temp_doc]
#         sent_pos = list(set(temp_tags).intersection(set(['ADJ','NOUN', 'ADV', 'VERB','ADP'])))
#         temp_list.append(sent)

#         while len(temp_list)<=change_num+1:
#             # print('-----temp_list-----')
#             # print(len(temp_list))
#             if len(temp_list)>=change_num+1:
#                 break
            
#             if len(np_list)!=0:
#                 counter_np=0
#                 for t_phrase in np_list:
#                     if ' ' in t_phrase:
#                         counter_np+=1
#                         w1,w2 =t_phrase.split(' ') 
#         #                 print(w1,'*',w2)
#                         new_w1=self.replaced_word(w1)
#         #                 print(new_w1)
#                         new_w2=self.replaced_word(w2)
#         #                 print(new_w2)
#                         new_phrase=new_w1 + ' '+new_w2
#                         new_sent =  self.replace_word_in_sentence(sent,str(t_phrase),self.remove_backslashes(new_phrase))  
#                         # print('0 is dead')
#                         if new_sent not in temp_list and len(temp_list)!=change_num+1:
#                             temp_list.append(new_sent)
#                             temp_TAG_list.append('NP')
#                             if len(temp_list)>=change_num+1:
#                                 break
#                         if counter_np >= len(np_list):
#                             break
#                         if len(temp_list)>=change_num+1:
#                                 break
#                     else:
#                         random_choice1,random_choice2 = random.sample(word_list,2)
#                         random_choice_idx2=  word_list.index(random_choice1)#找到随机替换的词语的index
#                         random_choice_idx1 = word_list.index(random_choice2)
#                         random_choice_pos1,random_choice_pos2= random.sample(list(self.vocab.keys()),2)
#                         random_choice_pos = random_choice_pos1+" "+random_choice_pos2
#                         temp_TAG_list.append(random_choice_pos)
#                         repalce_word1 = random.choice(self.vocab[random_choice_pos1])#根据index找到改词语的pos type,再随机选择一个词替换
#                         repalce_word2 = random.choice(self.vocab[random_choice_pos2])#根据index找到改词语的pos type,再随机选择一个词替换
#                         new_sent =  self.replace_word_in_sentence(sent,str(word_list[random_choice_idx1]),repalce_word1) 
#                         new_sent =  self.replace_word_in_sentence(new_sent,str(word_list[random_choice_idx2]),repalce_word2) 
#                         if new_sent not in temp_list and len(temp_list)!=change_num+1:
#                             temp_list.append(new_sent)
#                             temp_TAG_list.append(random_choice_pos)
#                             if len(temp_list)>=change_num+1:
#                                 break

#                     if len(temp_list)>=change_num+1:
#                         break


#             else:
#                 random_choice1,random_choice2 = random.sample(word_list,2)
#                 random_choice_idx2=  word_list.index(random_choice1)#找到随机替换的词语的index
#                 random_choice_idx1 = word_list.index(random_choice2)
#                 random_choice_pos1,random_choice_pos2= random.sample(list(self.vocab.keys()),2)
#                 random_choice_pos = random_choice_pos1+" "+random_choice_pos2
#                 temp_TAG_list.append(random_choice_pos)
#                 repalce_word1 = random.choice(self.vocab[random_choice_pos1])#根据index找到改词语的pos type,再随机选择一个词替换
#                 repalce_word2 = random.choice(self.vocab[random_choice_pos2])#根据index找到改词语的pos type,再随机选择一个词替换
#                 new_sent =  self.replace_word_in_sentence(sent,str(word_list[random_choice_idx1]),repalce_word1) 
#                 new_sent =  self.replace_word_in_sentence(new_sent,str(word_list[random_choice_idx2]),repalce_word2) 
#                 if new_sent not in temp_list and len(temp_list)!=change_num+1:
#                     temp_list.append(new_sent)
#                     temp_TAG_list.append(random_choice_pos)
#                     if len(temp_list)>=change_num+1:
#                         break
#                 if len(temp_list)>=change_num+1:
#                     break 

                    
#             if len(temp_list)>=change_num+1:
#                     break 
                    
                    
#         #                     print('---',temp_list)

#             if len(vp_list)!=0:
#                 counter_vp=0
#                 for t_phrase in vp_list:
#                     if ' ' in t_phrase:
#                         counter_vp+=1
#                         w1,w2 =t_phrase.split(' ') 
#         #                 print(w1,'*',w2)
#                         new_w1=self.replaced_word(w1)
#                         new_w2=self.replaced_word(w2)
#                         new_phrase=new_w1 + ' '+new_w2
#                         new_sent =  self.replace_word_in_sentence(sent,str(t_phrase),new_phrase)  
#                         # print('1 is dead')
#                         if new_sent not in temp_list and len(temp_list)!=change_num+1:
#                             temp_list.append(new_sent)
#                             temp_TAG_list.append('VP')
#                             if len(temp_list)>=change_num+1:
#                                 break
#                         if len(temp_list)>=change_num+1:
#                                 break
#                         if counter_vp == len(vp_list):
#                             break
#                     else:
#                         random_choice1,random_choice2 = random.sample(word_list,2)
#                         random_choice_idx2=  word_list.index(random_choice1)#找到随机替换的词语的index
#                         random_choice_idx1 = word_list.index(random_choice2)
#                         random_choice_pos1,random_choice_pos2= random.sample(list(self.vocab.keys()),2)
#                         random_choice_pos = random_choice_pos1+" "+random_choice_pos2
#                         temp_TAG_list.append(random_choice_pos)
#                         repalce_word1 = random.choice(self.vocab[random_choice_pos1])#根据index找到改词语的pos type,再随机选择一个词替换
#                         repalce_word2 = random.choice(self.vocab[random_choice_pos2])#根据index找到改词语的pos type,再随机选择一个词替换
#                         new_sent =  self.replace_word_in_sentence(sent,str(word_list[random_choice_idx1]),repalce_word1) 
#                         new_sent =  self.replace_word_in_sentence(new_sent,str(word_list[random_choice_idx2]),repalce_word2) 
#                         if new_sent not in temp_list and len(temp_list)!=change_num+1:
#                             temp_list.append(new_sent)
#                             temp_TAG_list.append(random_choice_pos)  

#                     if len(temp_list)>=change_num+1:
#                                 break      

                
#             else:
#                 random_choice1,random_choice2 = random.sample(word_list,2)
#                 random_choice_idx2=  word_list.index(random_choice1)#找到随机替换的词语的index
#                 random_choice_idx1 = word_list.index(random_choice2)
#                 random_choice_pos1,random_choice_pos2= random.sample(list(self.vocab.keys()),2)
#                 random_choice_pos = random_choice_pos1+" "+random_choice_pos2
#                 temp_TAG_list.append(random_choice_pos)
#                 repalce_word1 = random.choice(self.vocab[random_choice_pos1])#根据index找到改词语的pos type,再随机选择一个词替换
#                 repalce_word2 = random.choice(self.vocab[random_choice_pos2])#根据index找到改词语的pos type,再随机选择一个词替换
#                 new_sent =  self.replace_word_in_sentence(sent,str(word_list[random_choice_idx1]),repalce_word1) 
#                 new_sent =  self.replace_word_in_sentence(new_sent,str(word_list[random_choice_idx2]),repalce_word2) 
#                 if new_sent not in temp_list and len(temp_list)!=change_num+1:
#                     temp_list.append(new_sent)
#                     temp_TAG_list.append(random_choice_pos)
#                     if len(temp_list)==change_num+1:
#                         break
#                 if len(temp_list)>=change_num+1:
#                     break
#             if len(temp_list) >= change_num+1:
#                     break 

#             if len(temp_list)!=change_num+1:
#                 # print('2 is dead')
               
#                 random_choice1,random_choice2 = random.sample(word_list,2)
#                 random_choice_idx2=  word_list.index(random_choice1)#找到随机替换的词语的index
#                 random_choice_idx1 = word_list.index(random_choice2)
#                 random_choice_pos1,random_choice_pos2= random.sample(list(self.vocab.keys()),2)
#                 random_choice_pos = random_choice_pos1+" "+random_choice_pos2
#                 temp_TAG_list.append(random_choice_pos)
#                 repalce_word1 = random.choice(self.vocab[random_choice_pos1])#根据index找到改词语的pos type,再随机选择一个词替换
#                 repalce_word2 = random.choice(self.vocab[random_choice_pos2])#根据index找到改词语的pos type,再随机选择一个词替换
#                 new_sent =  self.replace_word_in_sentence(sent,str(word_list[random_choice_idx1]),repalce_word1) 
#                 new_sent =  self.replace_word_in_sentence(new_sent,str(word_list[random_choice_idx2]),repalce_word2) 
#                 if new_sent not in temp_list and len(temp_list)!=change_num+1:
#                     temp_list.append(new_sent)
#                     temp_TAG_list.append(random_choice_pos)
#                     if len(temp_list) >= change_num+1:
#                         break
#                 if len(temp_list) >= change_num+1:
#                     break 


            
                

#         return temp_list[1:change_num+1],temp_TAG_list


#     def change_word(self,sent,change_num=16):    
#         temp_list =[]
#         temp_TAG_list=[]
#         #1. pos找到句子中的VP NP verb noun
#         # sent = ' '.join(sent)#msrvtt的时候不能用' '去连接句子，因为已经时候就只了
#         # print(sent)
#         temp_doc = nlp(sent)
#         word_list = [w for w in temp_doc]
#         temp_tags = [w.pos_ for w in temp_doc]
#         #仅需要  'ADJ','NOUN', 'ADV', 'VERB',
#         # sent_pos = list(set(temp_tags).intersection(set(['ADJ','NOUN', 'ADV', 'VERB','ADP','NUM'])))
#         sent_pos = list(set(temp_tags).intersection(set(['ADJ','NOUN', 'ADV', 'VERB','ADP'])))
#         temp_list.append(sent)
#         while len(temp_list)!=change_num+1:
#             if len(sent_pos)>0:
#                 t_pos_dic={}
#                 for i in sent_pos:
#                     for j in self.get_index(temp_tags, item=i):
#                         t_pos_dic[i]= [word_list[j] for j in self.get_index(temp_tags, item=i)]

#                 pos_item = random.choices(list(t_pos_dic.keys()))[0]

#                 if pos_item=='ADP':
#                     if len(self.remove_words_with_symbols(t_pos_dic[pos_item])) != 0:
#                         word = random.choices(self.remove_words_with_symbols(t_pos_dic[pos_item]))
#                         if len(self.find_synonyms_antonyms(str(word)))!=0:
#                             if len(self.remove_words_with_symbols(self.find_synonyms_antonyms(str(word))))!=0:
        
#                                 repalce_word = self.remove_words_with_symbols(self.find_synonyms_antonyms(str(word)))#不管传入进去的词是什么最后返回一个词就对了
#                                 new_sent =  self.replace_word_in_sentence(sent,str(word),random.choice(repalce_word))#去除部分匹配问题
#                                 if new_sent not in temp_list:
#                                     temp_list.append(new_sent)
#                                     temp_TAG_list.append(pos_item)
#                         else:
#                             #
#                             repalce_word = random.choice(self.vocab[pos_item.lower()])#根据index找到改词语的pos type,再随机选择一个词替换
#                             new_sent =  self.replace_word_in_sentence(sent,str(word),repalce_word)#去除部分匹配问题
#                             if new_sent not in temp_list:
#                                 temp_list.append(new_sent)
#                                 temp_TAG_list.append(pos_item)




#                     else:
#                         random_choice_pos= random.choice(list(t_pos_dic.keys()))#找到随机替换的词语的index
#                         randm_choice_word = word_list[random.choice(self.get_index(temp_tags, item=random_choice_pos))]
#                         repalce_word = random.choice(self.vocab[random_choice_pos.lower()])#根据index找到改词语的pos type,再随机选择一个词替换
#                         new_sent =  self.replace_word_in_sentence(sent,str(randm_choice_word),repalce_word) 
#                         if new_sent not in temp_list:
#                             temp_list.append(new_sent)
#                             temp_TAG_list.append(random_choice_pos)
#                         # random_choice_idx= word_list.index(random.choice(word_list))#找到随机替换的词语的index
#                         # random_choice_pos= random.choice(list(self.vocab.keys()))
#                         # temp_TAG_list.append(random_choice_pos)
#                         # repalce_word = random.choice(self.vocab[random_choice_pos])#根据index找到改词语的pos type,再随机选择一个词替换
#                         # new_sent =  self.replace_word_in_sentence(sent,str(word_list[random_choice_idx]),repalce_word) 
#                         # # new_sent = sent.replace(str(word_list[random_choice_idx]),repalce_word)
#                         # if new_sent not in temp_list:
#                         #     temp_list.append(new_sent)
#                         #     temp_TAG_list.append(random_choice_pos)
                    

               
#                 elif  pos_item=='NUM':
#                     print("numnumnumnum")
#                     word = random.choices(self.remove_words_with_symbols(t_pos_dic[pos_item]))
#                     repalce_word = self.generate_random_number(word)
#                     new_sent =  self.replace_word_in_sentence(sent,str(word),repalce_word)#去除部分匹配问题

#                     if new_sent not in temp_list:
#                         temp_list.append(new_sent)
#                         temp_TAG_list.append(pos_item)
#                     else:
#                         print("!!!!!!!!!!!numnumnumnum!!!!!!!!!!!!!!!!!")
#                         random_choice_pos= random.choice(list(t_pos_dic.keys()))#找到随机替换的词语的index

#                         randm_choice_word = word_list[random.choice(self.get_index(temp_tags, item=random_choice_pos))]
#                         # print("**"*10)
#                         # print(self.vocab.keys())
#                         # repalce_word = random.choice(self.vocab[random_choice_pos.lower()])#根据index找到改词语的pos type,再随机选择一个词替换
#                         repalce_word = random.choice(self.vocab[random_choice_pos.lower()])#根据index找到改词语的pos type,再随机选择一个词替换
                        
#                         new_sent =  self.replace_word_in_sentence(sent,str(randm_choice_word),repalce_word) 
#                         # new_sent = sent.replace(str(randm_choice_word),repalce_word)
#                         if new_sent not in temp_list:
#                             temp_list.append(new_sent)
#                             temp_TAG_list.append(random_choice_pos)
        
#                 else:
#                     if len(self.remove_words_with_symbols(t_pos_dic[pos_item])) != 0:
#                         word = random.choices(self.remove_words_with_symbols(t_pos_dic[pos_item]))

#                         if len(self.find_synonyms_antonyms(str(word)))!=0:

#                             if len(self.remove_words_with_symbols(self.find_synonyms_antonyms(str(word))))!=0:

#                                 repalce_word = self.remove_words_with_symbols(self.find_synonyms_antonyms(str(word)))#不管传入进去的词是什么最后返回一个词就对了
#                                 new_sent =  self.replace_word_in_sentence(sent,str(word),random.choice(repalce_word))#去除部分匹配问题
#                                 # new_sent = sent.replace(str(word),random.choice(repalce_word))
#                                 if new_sent not in temp_list:
#                                     temp_list.append(new_sent)
#                                     temp_TAG_list.append(pos_item)
#                                 else:
#                                     print("x"*10)
#                                     random_choice_pos= random.choice(list(t_pos_dic.keys()))#找到随机替换的词语的index
#                                     randm_choice_word = word_list[random.choice(self.get_index(temp_tags, item=random_choice_pos))]
#                                     repalce_word = random.choice(self.vocab[random_choice_pos.lower()])#根据index找到改词语的pos type,再随机选择一个词替换
#                                     new_sent =  self.replace_word_in_sentence(sent,str(randm_choice_word),repalce_word) 
#                                     if new_sent not in temp_list:
#                                         temp_list.append(new_sent)
#                                         temp_TAG_list.append(random_choice_pos)


#                             else:
#                                 print("^"*10)
#                                 random_choice_pos= random.choice(list(t_pos_dic.keys()))#找到随机替换的词语的index
#                                 randm_choice_word = word_list[random.choice(self.get_index(temp_tags, item=random_choice_pos))]
#                                 repalce_word = random.choice(self.vocab[random_choice_pos.lower()])#根据index找到改词语的pos type,再随机选择一个词替换
#                                 new_sent =  self.replace_word_in_sentence(sent,str(randm_choice_word),repalce_word) 
#                                 if new_sent not in temp_list:
#                                     temp_list.append(new_sent)
#                                     temp_TAG_list.append(random_choice_pos)
#                                 # random_choice_idx= word_list.index(random.choice(word_list))#找到随机替换的词语的index
#                                 # random_choice_pos= random.choice(list(self.vocab.keys()))
#                                 # temp_TAG_list.append(random_choice_pos)
#                                 # repalce_word = random.choice(self.vocab[random_choice_pos])#根据index找到改词语的pos type,再随机选择一个词替换
#                                 # new_sent =  self.replace_word_in_sentence(sent,str(word_list[random_choice_idx]),repalce_word) 
#                                 # # new_sent = sent.replace(str(word_list[random_choice_idx]),repalce_word)
#                                 # if new_sent not in temp_list:
#                                 #     temp_list.append(new_sent)
#                                 #     temp_TAG_list.append(random_choice_pos)

#                         elif len(self.get_hypernyms(str(word),pos_item=pos_item))!=0:
#                             temp_hyponyms=[]
#                             for i in self.remove_words_with_symbols(self.get_hypernyms(str(word),pos_item=pos_item)):
#                                 if len(self.remove_words_with_symbols(self.get_hyponyms(i,pos_item=pos_item)))!=0:
#                                     temp_hyponyms.extend(self.remove_words_with_symbols(self.get_hyponyms(i,pos_item=pos_item)))
#                             if len(temp_hyponyms)!=0: 
#                                 repalce_word = random.choice(temp_hyponyms)#不管传入进去的词是什么最后返回一个词就对了   
#                                 new_sent =  self.replace_word_in_sentence(sent,str(word),repalce_word)                
#                                 # new_sent = sent.replace(str(word),repalce_word)
#                                 if new_sent not in temp_list:
#                                     temp_list.append(new_sent)
#                                     temp_TAG_list.append(pos_item)
#                                 else:
#                                     print("*************************")
#                                     random_choice_pos= random.choice(list(t_pos_dic.keys()))#找到随机替换的词语的index
#                                     randm_choice_word = word_list[random.choice(self.get_index(temp_tags, item=random_choice_pos))]
#                                     repalce_word = random.choice(self.vocab[random_choice_pos.lower()])#根据index找到改词语的pos type,再随机选择一个词替换
#                                     new_sent =  self.replace_word_in_sentence(sent,str(randm_choice_word),repalce_word) 
#                                     if new_sent not in temp_list:
#                                         temp_list.append(new_sent)
#                                         temp_TAG_list.append(random_choice_pos)

#                             else:
#                                 # print("*************************")
#                                 random_choice_pos= random.choice(list(t_pos_dic.keys()))#找到随机替换的词语的index
#                                 randm_choice_word = word_list[random.choice(self.get_index(temp_tags, item=random_choice_pos))]
#                                 repalce_word = random.choice(self.vocab[random_choice_pos.lower()])#根据index找到改词语的pos type,再随机选择一个词替换
#                                 new_sent =  self.replace_word_in_sentence(sent,str(randm_choice_word),repalce_word) 
#                                 if new_sent not in temp_list:
#                                     temp_list.append(new_sent)
#                                     temp_TAG_list.append(random_choice_pos)
#                                 # random_choice_idx= word_list.index(random.choice(word_list))#找到随机替换的词语的index
#                                 # random_choice_pos= random.choice(list(self.vocab.keys()))
#                                 # temp_TAG_list.append(random_choice_pos)
#                                 # repalce_word = random.choice(self.vocab[random_choice_pos])#根据index找到改词语的pos type,再随机选择一个词替换
#                                 # new_sent =  self.replace_word_in_sentence(sent,str(word_list[random_choice_idx]),repalce_word) 
#                                 # # new_sent = sent.replace(str(word_list[random_choice_idx]),repalce_word)
#                                 # if new_sent not in temp_list:
#                                 #     temp_list.append(new_sent)
#                                 #     temp_TAG_list.append(random_choice_pos)

#                         else:
#                             # caz2024629
#                             print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#                             random_choice_pos= random.choice(list(t_pos_dic.keys()))#找到随机替换的词语的index

#                             randm_choice_word = word_list[random.choice(self.get_index(temp_tags, item=random_choice_pos))]
#                             # print("**"*10)
#                             # print(self.vocab.keys())
#                             # repalce_word = random.choice(self.vocab[random_choice_pos.lower()])#根据index找到改词语的pos type,再随机选择一个词替换
#                             repalce_word = random.choice(self.vocab[random_choice_pos.lower()])#根据index找到改词语的pos type,再随机选择一个词替换
                            
#                             new_sent =  self.replace_word_in_sentence(sent,str(randm_choice_word),repalce_word) 
#                             # new_sent = sent.replace(str(randm_choice_word),repalce_word)
#                             if new_sent not in temp_list:
#                                 temp_list.append(new_sent)
#                                 temp_TAG_list.append(random_choice_pos)
#                     else:
#                         # print("##########################")
#                         random_choice_pos= random.choice(list(t_pos_dic.keys()))#找到随机替换的词语的index

#                         randm_choice_word = word_list[random.choice(self.get_index(temp_tags, item=random_choice_pos))]
#                         # print("**"*10)
#                         # print(self.vocab.keys())
#                         # repalce_word = random.choice(self.vocab[random_choice_pos.lower()])#根据index找到改词语的pos type,再随机选择一个词替换
#                         repalce_word = random.choice(self.vocab[random_choice_pos.lower()])#根据index找到改词语的pos type,再随机选择一个词替换
#                         new_sent =  self.replace_word_in_sentence(sent,str(randm_choice_word),repalce_word) 
#                         if new_sent not in temp_list:
#                             temp_list.append(new_sent)
#                             temp_TAG_list.append(random_choice_pos)
#                         # random_choice_idx= word_list.index(random.choice(word_list))#找到随机替换的词语的index
#                         # random_choice_pos= random.choice(list(self.vocab.keys()))
#                         # temp_TAG_list.append(random_choice_pos)
#                         # repalce_word = random.choice(self.vocab[random_choice_pos])#根据index找到改词语的pos type,再随机选择一个词替换
#                         # new_sent =  self.replace_word_in_sentence(sent,str(word_list[random_choice_idx]),repalce_word) 
#                         # # new_sent = sent.replace(str(word_list[random_choice_idx]),repalce_word)
#                         # if new_sent not in temp_list:
#                         #     temp_list.append(new_sent)
#                         #     temp_TAG_list.append(random_choice_pos)
#             else:
#                 print("###*###"*5)
#                 random_choice_pos= random.choice(list(t_pos_dic.keys()))#找到随机替换的词语的index

#                 randm_choice_word = word_list[random.choice(self.get_index(temp_tags, item=random_choice_pos))]
#                 # print("**"*10)
#                 # print(self.vocab.keys())
#                 # repalce_word = random.choice(self.vocab[random_choice_pos.lower()])#根据index找到改词语的pos type,再随机选择一个词替换
#                 repalce_word = random.choice(self.vocab[random_choice_pos.lower()])#根据index找到改词语的pos type,再随机选择一个词替换
                
#                 new_sent =  self.replace_word_in_sentence(sent,str(randm_choice_word),repalce_word) 
#                 # new_sent = sent.replace(str(randm_choice_word),repalce_word)
#                 if new_sent not in temp_list:
#                     temp_list.append(new_sent)
#                     temp_TAG_list.append(random_choice_pos)

      
#             # random_choice_idx= word_list.index(random.choice(word_list))#找到随机替换的词语的index
#             # random_choice_pos= random.choice(list(self.vocab.keys()))
#             # temp_TAG_list.append(random_choice_pos)
#             # repalce_word = random.choice(self.vocab[random_choice_pos])#根据index找到改词语的pos type,再随机选择一个词替换
#             # new_sent =  self.replace_word_in_sentence(sent,str(word_list[random_choice_idx]),repalce_word) 
#             # # new_sent = sent.replace(str(word_list[random_choice_idx]),repalce_word)
#             # if new_sent not in temp_list:
#             #     temp_list.append(new_sent)
#             #     temp_TAG_list.append(random_choice_pos)

#         return temp_list[1:],temp_TAG_list


# # get_senteces= Get_Negative_text_samples(dataset_based_vocab_path)
# # get_neg_sent_fun = get_senteces.change_word
