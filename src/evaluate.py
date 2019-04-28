import ime as m_ime
import os.path as path
import config
import argparse
import re
import jieba_fast
import Levenshtein
from pypinyin import lazy_pinyin
from tqdm import tqdm

filtrate = re.compile(u'[^\u4E00-\u9FA5]')
punc = re.compile('[，。？]')

def eval(infile, ime):
    tot = 0
    corr = 0
    tot_sen = 0
    corr_sen = 0
    corpus = []
    with open(infile, "r", encoding = "utf-8") as inf:
        for line in tqdm(inf.readlines()):
        	line = punc.sub('\n', line)
        	line = line.split('\n')
        	for item in line:
        		if(len(item)):
        			corpus.append(item)
    tot_sen = len(corpus)
    for item in tqdm(corpus):
        pinyin = []
        item = filtrate.sub('', item)
        charlist = jieba_fast.cut(item)
        for word in charlist:
        	pinyin.extend(list(lazy_pinyin(word)))
        try:
        	res = ime.predictio(' '.join(pinyin))
        	dis = Levenshtein.distance(res, item)
        	tot += len(item)
        	corr += (len(item) - dis)
        	if(dis == 0):
                    corr_sen += 1
        except:
        	pass
    print("Prediction precision: (word)%f%%, (sentence)%f%%" %( (corr * 100 / tot),(corr_sen * 100 / tot_sen)))
    return corr * 100 / tot

if(__name__ == '__main__'):
    parser = argparse.ArgumentParser(description="Intellipinyin IME Evaluator")
    parser.add_argument('-n', '--n-grams', help='Select to use 2-gram or 3-gram model', default = 2)
    parser.add_argument('-i', '--input-file', help='In-file name')
    parser.add_argument('--lm', help='lambda param', default=1e-3)
    parser.add_argument('--mu', help='mu param', default = 1.03)
    args = parser.parse_args()
    ime = None
    if(args.input_file is None):
        print("Input file should be specified")
        exit(-1)
    if(int(args.n_grams) == 2):
        print("Using 2-gram model")
        w = float(args.lm)
        print("Param lambda %f" % w)
        ime = m_ime.Ime(path.join(config.TRAINED_DIR, config.PINYIN), path.join(config.TRAINED_DIR, config.CHARACTER_LIST), 
        path.join(config.TRAINED_DIR, config.FREQ_GRAM1), path.join(config.TRAINED_DIR, config.FREQ_GRAM2), None, w, None)
        eval(args.input_file, ime)
    elif(int(args.n_grams) == 3):
        print("Using 3-gram model, this will be slower")
        p = float(args.mu)
        w = float(args.lm)
        print("Param mu %f" % p)
        ime = m_ime.Ime(path.join(config.TRAINED_DIR, config.PINYIN), path.join(config.TRAINED_DIR, config.CHARACTER_LIST), 
        path.join(config.TRAINED_DIR, config.FREQ_GRAM1), path.join(config.TRAINED_DIR, config.FREQ_GRAM2),
        path.join(config.TRAINED_DIR, config.FREQ_GRAM3), w, p)
        eval(args.input_file, ime)
    else:
        print("Invalid n_grams. Expected 2 or 3")
        exit(-1)
    
    
    
