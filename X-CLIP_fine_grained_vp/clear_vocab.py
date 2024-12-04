import json
import re
def remove_backslashes(input_string):
    # 使用字符串的 replace 方法将反斜杠替换为空字符串
        # input_string =' '.join(input_string)
        cleaned_string = input_string.replace('\\', '')
        # cleaned_string =cleaned_string.split(' ')
        return cleaned_string

dataset_based_vocab_path=  '/var/scratch/achen/VisualSearch/msrvtt10k/TextData/msrvtt_word_dic_20230924_len3.json'

vocab = json.load(open(dataset_based_vocab_path, 'r'))

for item in vocab:
    vocab[item] = [remove_backslashes(x) for x in vocab[item]]



with open(dataset_based_vocab_path,'w') as f:
    json.dump(vocab,f) 
