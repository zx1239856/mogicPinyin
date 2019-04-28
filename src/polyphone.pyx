import builder
import pickle
import config
import os
import logging

logging.basicConfig(level = logging.DEBUG,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def build():
    from pypinyin import lazy_pinyin
    import jieba_fast
    corpus = builder.readCorpus('sina')
    ch2idx, idx2ch = builder.readCharacterList(os.path.join(config.DATA_DIR, config.CHARACTER_LIST))
    pinyin2idx, idx2ch = builder.readPinyinList(os.path.join(config.DATA_DIR, config.PINYIN), os.path.join(config.DATA_DIR, config.CHARACTER_LIST))
    ch_pinyin_freq = {}
    prev_progress = -1
    for idx, line in enumerate(corpus):
        progress = idx * 100 // len(corpus)
        if(progress > prev_progress):
            print("Progress: %d"%progress)
            prev_progress = progress
        for word in jieba_fast.cut(line):
            inList = True
            for i in word:
                if(i not in ch2idx.keys()):
                    inList = False
                    break
            if(inList):
                pinyin = lazy_pinyin(word)
                for i in range(0, len(word)):
                    if(pinyin[i] in pinyin2idx.keys()):
                        if(word[i] in ch_pinyin_freq.keys()):
                            if(pinyin[i] in ch_pinyin_freq[word[i]].keys()):
                                ch_pinyin_freq[word[i]][pinyin[i]] += 1
                            else:
                                ch_pinyin_freq[word[i]][pinyin[i]] = 1
                        else:
                            ch_pinyin_freq[word[i]] = {}
                            ch_pinyin_freq[word[i]][pinyin[i]] = 1
            else:
                continue
    result = {}
    for i in ch_pinyin_freq.items():
        if(len(i[1]) > 1):
            result[i[0]] = i[1]
    with open(os.path.join(config.TRAINED_DIR, config.POLYPHONE_MAP), "wb") as f:
        pickle.dump(result, f)

def load(pinyin_freq, char_list):
    logging.info("Loading polyphone list...")
    ch_pinyin_freq = {}
    with open(pinyin_freq, "rb") as f:
        ch_pinyin_freq = pickle.load(f)
    ch2idx, idx2ch = builder.readCharacterList(char_list)
    result = [None for i in range(0, len(ch2idx))]
    for entry in ch_pinyin_freq.items():
        result[ch2idx[entry[0]]] = entry[1]
    return result

def normalize(ch_pinyin_freq):
    logging.info("Normalizing polyphone list...")
    import math
    for entry in ch_pinyin_freq:
        if(entry is None):
            continue
        tot = sum(entry.values())
        for key in entry.keys():
            entry[key] = -math.log(entry[key] / tot)
    return ch_pinyin_freq

if(__name__ == "__main__"):
    pass