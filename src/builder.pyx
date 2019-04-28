import config
import re
import json
import logging
import scipy.sparse
import numpy
import os
import pickle
from tqdm import tqdm

logging.basicConfig(level = logging.DEBUG,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

filtrate = re.compile(u'[^\u4E00-\u9FA5]')
separate = re.compile('[，。？]')

def readCorpus(corpus_type='sina'):
    corpus = []
    if(corpus_type == 'sina'):
        for f in config.CORPUS_SINA:
            f = os.path.join(config.DATA_DIR, f)
            logging.info("Processing file %s, please wait..."%f)
            with open(f, "r", encoding="gbk") as file:
                for line in tqdm(file.readlines()):
                    parsed = json.loads(line)
                    string = filtrate.sub('',parsed['title'])
                    if(len(string)):
                        corpus.append(string)
                    str_list = separate.sub('\n',parsed['html'])
                    str_list = str_list.split('\n')
                    for item in str_list:
                        item = filtrate.sub('', item)
                        if(len(item)):
                            corpus.append(item)
    elif(corpus_type == 'zhihu'):
        f = os.path.join(config.DATA_DIR, config.CORPUS_ZHIHU)
        with open(f, "r", encoding='utf-8') as file:
            for line in tqdm(file.readlines()):
                str_list = separate.sub('',line)
                str_list = str_list.split('\n')
                for item in str_list:
                    item = filtrate.sub('', item)
                    if(len(item)):
                        corpus.append(item)
    else:
        logging.fatal('Invalid corpus selected')
        raise ValueError()
    logging.info("Process %s corpus done with %d entries."%(corpus_type, len(corpus)))
    return corpus

'''
Return value: character to index, index to character
'''
def readCharacterList(charfile):
    ch2idx = {}
    idx2ch = []
    idx = -1
    #logging.info("Processing character list, please wait...")
    with open(charfile, "r", encoding = "gbk") as file:
        for line in file.readlines():
            for ch in line:
                if(ch not in ch2idx.keys()):
                    idx += 1
                    idx2ch.append(ch)
                    ch2idx[ch] = idx
    #logging.info("Process done with %d characters"%idx)
    return ch2idx, idx2ch

def readPinyinList(pinyinfile, charfile):
    pinyin2idx = {}
    ch2idx, idx2ch = readCharacterList(charfile)
    length = 0
    with open(pinyinfile, "r", encoding="gbk") as file:
        for line in file.readlines():
            entry = line[:-1].split(' ')
            processed = []
            for item in entry[1:]:
                if(item in ch2idx.keys()):
                    processed.append(ch2idx[item])
            pinyin2idx[entry[0]] = processed
            length += len(processed)
    logging.info("Read pinyin done with %d valid characters" % length)
    return pinyin2idx, idx2ch

def normalizeVec(vec):
    import math
    tot = numpy.sum(vec) + vec.shape[0]
    new_vec = numpy.zeros(vec.shape[0])
    for i in range(0, vec.shape[0]):
        new_vec[i] = - math.log((vec[i]+1)/tot)
    return new_vec    

def normalizeMat(mat):
    import math
    r = mat.shape[0]
    c = mat.shape[1]
    new_mat = numpy.zeros((r,c))
    for i in range(0,r):
        tot = numpy.sum(mat[i]) + c
        for j in range(0,c):
            new_mat[i][j] = - math.log((mat[i][j] + 1)/tot)
    return new_mat

def normalizeTriplet(mapping):
    import math
    new_mapping = {}
    for term in mapping.items():
        tot = sum(term[1].values())
        new_mapping[term[0]] = {}
        for entry in term[1].items():
            new_mapping[term[0]][entry[0]] = -math.log(entry[1] / tot)
    return new_mapping

def getFrequency(ch2idx, idx2ch, corpus, normalize = True):
    num_chars = len(ch2idx)
    freq = numpy.zeros((num_chars), int)
    freq_gram2 = numpy.zeros((num_chars,num_chars),int)
    for line in tqdm(corpus):
        if(not len(line)):
            continue
        if(line[0] in ch2idx):
            freq[ch2idx[line[0]]] += 1
        for i in range(1, len(line)):
            if(not(line[i] in ch2idx.keys() and line[i-1] in ch2idx.keys())):
                continue
            else:
                freq_gram2[ch2idx[line[i-1]]][ch2idx[line[i]]] += 1
    logging.info("Normalizing gram-1, please wait...")
    freq = normalizeVec(freq)
    logging.info("Normalizing gram-2, please wait...")
    freq_gram2 = normalizeMat(freq_gram2)
    sparse_freq = scipy.sparse.csr_matrix(freq)
    sparse_freq_gram2 = scipy.sparse.csr_matrix(freq_gram2)
    scipy.sparse.save_npz(os.path.join(config.TRAINED_DIR, config.FREQ_GRAM1),sparse_freq)
    scipy.sparse.save_npz(os.path.join(config.TRAINED_DIR, config.FREQ_GRAM2),sparse_freq_gram2)

def getFrequency3(ch2idx, idx2ch, corpus, normalize = True):
    triplet_freq = {}
    char_num = len(ch2idx)
    for line in tqdm(corpus):
        if(len(line) <= 2):
            continue
        for i in range(2, len(line)):
            if(line[i-2] not in ch2idx.keys() or line[i-1] not in ch2idx.keys() or line[i] not in ch2idx.keys()):
                continue
            gram = ch2idx[line[i-2]] * char_num + ch2idx[line[i-1]]
            if(gram not in triplet_freq.keys()):
                triplet_freq[gram] = {ch2idx[line[i]] : 1}
            else:
                if(ch2idx[line[i]] not in triplet_freq[gram].keys()):
                    triplet_freq[gram][ch2idx[line[i]]] = 1
                else:
                    triplet_freq[gram][ch2idx[line[i]]] += 1
    logging.info("Normalizing gram-3, this might take a long time...")
    if(normalize):
        triplet_freq = normalizeTriplet(triplet_freq)
    logging.info("Writing to file %s" % os.path.join(config.TRAINED_DIR, config.FREQ_GRAM3))
    with open(os.path.join(config.TRAINED_DIR, config.FREQ_GRAM3), "wb") as f:
        pickle.dump(triplet_freq, f)

def build(*args, **kwargs):
    ch2idx, idx2ch = readCharacterList(os.path.join(config.DATA_DIR, config.CHARACTER_LIST))
    corpus = readCorpus('sina')
    corpus.extend(readCorpus('zhihu'))
    for func in args:
        if('normalize' in kwargs.keys()):
            func(ch2idx, idx2ch, corpus, normalize=kwargs['normalize'])
        else:
            func(ch2idx, idx2ch, corpus)

def loadFrequency(vecfile, matfile):
    freq = scipy.sparse.load_npz(vecfile).toarray()
    freq_gram2 = scipy.sparse.load_npz(matfile).toarray()
    return freq[0], freq_gram2

def loadFrequency3(mapfile):
    mapping = None
    with open(mapfile, "rb") as f:
        mapping = pickle.load(f)
    return mapping

if(__name__ == "__main__"):
    #loadFrequency3(os.path.join(config.TRAINED_DIR, config.FREQ_GRAM3))
    build(getFrequency, getFrequency3)
