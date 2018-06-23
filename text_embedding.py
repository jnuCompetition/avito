from tqdm import tqdm
import numpy as np
import pandas as pd
import multiprocessing as mlp
from nltk.tokenize.toktok import ToktokTokenizer
import gc
def tokenize_worker(sentences):
    tknzr = ToktokTokenizer()
    sentences = [tknzr.tokenize(seq) for seq in tqdm(sentences)]
    return sentences

def tokenize_word(sentences):
    " 多进程分词"
    results = []
    num_cpu = mlp.cpu_count()
    pool = mlp.Pool(num_cpu)
    aver_t = len(sentences)//num_cpu + 1
    for i in range(num_cpu):
        result = pool.apply_async(tokenize_worker,
                                  args=(sentences[i * aver_t:(i + 1) * aver_t],))
        results.append(result)
    pool.close()
    pool.join()

    tokenized_sentences = []
    for result in results:
        tokenized_sentences.extend(result.get())

    return tokenized_sentences

def tokenize_sentences(sentences):

    def step_cal_frequency(sentences):
        frequency = {}
        for sentence in tqdm(sentences):
            for word in sentence:
                if frequency.get(word) is None:
                    frequency[word] = 0
                frequency[word]+=1
        return frequency

    def step_to_seq(sentences,frequency):
        " 句子转序列 "
        words_dict = { }
        seq_list = []

        for sentence in tqdm(sentences):
            seq = []
            for word in sentence:
                if frequency[word]<= 5 :
                    continue
                if word not in words_dict:
                    words_dict[word] = len(words_dict) + 1
                word_index = words_dict[word]
                seq.append(word_index)
            seq_list.append(seq)
        return seq_list,words_dict

    sentences = tokenize_word(sentences)
    freq = step_cal_frequency(sentences)
    return step_to_seq(sentences,freq)


def get_embedding_matrix(word_index,feat=None):
    print('get embedding matrix')
    from fastText import load_model
    from gensim.models import Word2Vec

    num_words = len(word_index) + 1
    # 停止符用0
    embedding_matrix = np.zeros((num_words,300))

    print('num of word: ',num_words)
    if feat is None:
        ft_model = load_model('./dataset/cc.ru.300.bin')
        for word, i in tqdm(word_index.items()):
            embedding_matrix[i] = ft_model.get_word_vector(word).astype('float32')
    else:
        error = 0
        ft_model = Word2Vec.load('./dataset/2'+feat+'.model')
        for word, i in tqdm(word_index.items()):
            try:
                embedding_matrix[i] = ft_model[word].astype('float32')
            except KeyError as e:
                error +=1
        print('error',error)
    del ft_model

    return embedding_matrix

def clean(sentences):
    from tool import norm_text
    num_cpu = mlp.cpu_count()//2
    pool = mlp.Pool(num_cpu)
    num_task = 1 + len(sentences) // num_cpu
    results = []

    for i in range(num_cpu):
        result = pool.apply_async(norm_text,args=(sentences[i*num_task:(i+1)*num_task],))
        results.append(result)
    pool.close()
    pool.join()

    clean_sent = []
    for result in tqdm(results):
        clean_sent += result.get()
    return clean_sent

def pocess_text(data):
    from keras.preprocessing.sequence import pad_sequences
    from config import desc_len,title_len
    desc,word_idx = tokenize_sentences(clean(data['description'].fillna('unk').values))
    desc = pad_sequences(desc,maxlen=desc_len,truncating='post')
    desc = np.array(desc)
    desc_embed = get_embedding_matrix(word_idx)

    title,word_idx = tokenize_sentences(clean(data['title'].fillna('unk').values))
    title = pad_sequences(title, maxlen=title_len, truncating='post')
    title = np.array(title)
    title_embed = get_embedding_matrix(word_idx)

    return desc,desc_embed,title,title_embed












