import builder
import config
import polyphone
import os.path as path
import re

class Vocabulary:
    def __init__(self, pinyinFile, charFile):
        self.pinyin2idx, self.idx2ch = builder.readPinyinList(pinyinFile, charFile)
    def getNumChars(self):
        return len(self.idx2ch)
    def pinyin2index(self, pinyin):
        if(pinyin in self.pinyin2idx.keys()):
            return self.pinyin2idx[pinyin]
        else:
            raise LookupError("Not found!")
    def index2char(self, idx):
        if(idx <0 or idx >= len(self.idx2ch)):
            raise ValueError()
        else:
            return self.idx2ch[idx]
    def pinyinList2idxList(self, pinyinList):
        return [self.pinyin2index(i) for i in pinyinList]

epsilon = 1e-4
import math

class Predictor:
    def __init__(self, pinyinFile, charFile, gramVecFile, gramMatFile, gramMapFile, weight, penalty):
        self.P, self.P_2 = builder.loadFrequency(gramVecFile, gramMatFile)
        self.w1 = weight
        self.w2 = penalty
        self.predictor = self.__predict2
        if(gramMapFile is not None):
            self.P_3 = builder.loadFrequency3(gramMapFile)
            self.predictor = self.__predict3
        self.vocabulary = Vocabulary(pinyinFile, charFile)
        self.numchars = self.vocabulary.getNumChars()
        self.polyphone_map = polyphone.normalize(polyphone.load(path.join(config.TRAINED_DIR, config.POLYPHONE_MAP), path.join(config.DATA_DIR, config.CHARACTER_LIST)))
    def predict(self, pinyinStr):
        pinyin = re.split('[ ]+', pinyinStr)
        pinyinList = [self.vocabulary.pinyin2index(i) for i in pinyin]
        return self.predictor(pinyin, pinyinList)
    def __predict3(self, pinyin, pinyinList):
        if(len(pinyin) < 3):
            return self.__predict2(pinyin, pinyinList)
        path = [[None for j in range(0, len(pinyinList[1]))] for i in range(0, len(pinyinList[0]))]
        cost = [[0 for j in range(0, len(pinyinList[1]))] for i in range(0, len(pinyinList[0]))]
        for idx_i, i in enumerate(pinyinList[0]):
            coeff = 0
            if(self.polyphone_map[i] is not None and pinyin[0] in self.polyphone_map[i].keys()):
                coeff = self.polyphone_map[i][pinyin[0]]
            for idx_j, j in enumerate(pinyinList[1]):
                coeff2 = 0
                if(self.polyphone_map[j] is not None and pinyin[1] in self.polyphone_map[j].keys()):
                    coeff2 = self.polyphone_map[j][pinyin[1]]
                path[idx_i][idx_j] = [i , j]
                cost[idx_i][idx_j] = self.w1 * self.P[j] + (1 - self.w1) * self.P_2[i][j] + coeff + coeff2

        for layer in range(2, len(pinyinList)):
            new_path = [[None for j in range(0, len(pinyinList[layer]))] for i in range(0, len(pinyinList[layer - 1]))]
            new_cost = [[0 for j in range(0, len(pinyinList[layer]))] for i in range(0, len(pinyinList[layer - 1]))]
            for idx_k, k in enumerate(pinyinList[layer]):
                coeff = 0
                if(self.polyphone_map[k] is not None and pinyin[layer] in self.polyphone_map[k].keys()):
                    coeff = self.polyphone_map[k][pinyin[layer]]
                for idx_j, j in enumerate(pinyinList[layer - 1]):
                    best = 0
                    min_cost = float('inf')
                    for idx_i, i in enumerate(pinyinList[layer - 2]):
                        gram_id = i * self.numchars + j
                        p_3 = 0
                        if(gram_id in self.P_3.keys() and k in self.P_3[gram_id].keys()):
                            p_3 = self.P_3[gram_id][k]
                        if(math.fabs(p_3) < epsilon):
                            p_3 = self.w2 * self.P_2[j][k]
                        cc = p_3 + cost[idx_i][idx_j] + coeff
                        if(cc < min_cost):
                            min_cost = cc
                            best = idx_i
                    pp = path[best][idx_j][:]
                    pp.append(k)
                    new_path[idx_j][idx_k] = pp
                    new_cost[idx_j][idx_k] = min_cost
            path = new_path
            cost = new_cost
        best_candidate = [(row.index(min(row)),min(row)) for row in cost]
        best_val = [row[1] for row in best_candidate]
        best_i = best_val.index(min(best_val))
        best_j = best_candidate[best_i][0]
        
        result = []
        for item in path[best_i][best_j]:
            result.append(self.vocabulary.index2char(item))
        return ''.join(result)


    def __predict2(self, pinyin, pinyinList):
        path = [[i] for i in pinyinList[0]]
        cost = []
        for i in pinyinList[0]:
            coeff = 0.
            if(self.polyphone_map[i] is not None and pinyin[0] in self.polyphone_map[i].keys()):
                coeff = self.polyphone_map[i][pinyin[0]]
            cost.append(self.P[i] + coeff)
        for layer in range(1, len(pinyinList)):
            new_cost = []
            new_path = []
            for i in range(0, len(pinyinList[layer])):
                cc = float('inf')
                best = 0
                curr_char_idx = pinyinList[layer][i]
                coeff = 0.
                if(self.polyphone_map[curr_char_idx] is not None and pinyin[layer] in self.polyphone_map[curr_char_idx].keys()):
                    coeff = self.polyphone_map[curr_char_idx][pinyin[layer]]
                for j in range(0, len(pinyinList[layer - 1])):
                    val = cost[j] + (1 - self.w1) * self.P_2[pinyinList[layer-1][j]][pinyinList[layer][i]] + self.w1 * self.P[pinyinList[layer][i]] + coeff
                    if(val < cc):
                        cc = val
                        best = j
                new_cost.append(cc)
                temp_path = path[best][:]
                temp_path.append(curr_char_idx)
                new_path.append(temp_path)
            cost = new_cost
            path = new_path
        best = cost.index(min(cost))
        result = []
        for item in path[best]:
            result.append(self.vocabulary.index2char(item))
        return ''.join(result)
        


class Ime:
    def __init__(self, pinyinFile, charFile, gramVecFile, gramMatFile, gramMapFile, w1, w2):
        self._predictor = Predictor(pinyinFile, charFile, gramVecFile, gramMatFile, gramMapFile, w1, w2)
    def shell(self):
        print("IME Shell Version, powered by zx1239856. Type exit() to exit, or Pinyin string to predict")
        while True:
            try:
                pinyin = input('>>> ')
                if(pinyin == 'exit()'):
                    break
                else:
                    try:
                        print(self._predictor.predict(pinyin))
                    except Exception as e:
                        print(e)
            except:
                print('\n')
                pass
    def fileio(self, infile, outfile = None, silentErrInfo = True):
        import sys
        if(outfile is not None):
            outfile = open(outfile, "w")
        with open(infile, "r") as inf:
            line = inf.readline()
            while line:
                try:
                    line = line.replace('\n','')
                    res = self._predictor.predict(line)
                    if(outfile is not None):
                        outfile.write(res + '\n')
                    print(res)
                except Exception as e:
                    if(not silentErrInfo):
                        print(e)
                finally:
                    line = inf.readline()
    def predictio(self, pinyinStr):
        return self._predictor.predict(pinyinStr)
                    

