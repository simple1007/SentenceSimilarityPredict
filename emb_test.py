from utils.utils import cos_sim
from gensim.models import FastText, Word2Vec
import numpy as np
model = Word2Vec.load('ilbe_word2vec/ilbe_word2vec.model')

if False:
    #성혐오 결과
    seed = ['김치녀'
    # ,'김치남'
    # ,'스시남'
    ,'페미'
    ,'페미니즘'
    ,'동성애'
    ,'성혐오'
    ,'여혐'
    ,'남혐'
    ,'된장녀'
    # ,'된장남'
    ,'혐오'
    ,'걸레갈보개'
    ,'김치맨'
    ,'구멍동서'
    ,'보지'
    ,'자지'
    ,'좆물받이'
    ,'좆물'
    ,'보지'
    ,'보지년'
    ,'김치년'
    ,'후장'
    ,'창녀'
    ,'창놈']

    seed = ['성소수자']

    resultSeedF = open('result1stDict.txt','w',encoding='utf-8')

    for sd in seed:
        re_sim_word = model.wv.most_similar(sd)
        resultSeedF.write('#'+sd+'\n')
        for sim_word in re_sim_word:
            resultSeedF.write(sim_word[0]+'\t'+str(sim_word[1])+'\n')

    resultSeedF.close()

if True:
    wordList = [
        '좌좀'
        ,'좌빨'
        ,'좌좀'
        ,'우파'
        ,'좌파'
        ,'빨갱이'
        ,'친북'
        ,'극우'
        ,'간첩'
        ,'보수'
        ,'진보'
        ,'매국노'
        ,'친일'
        ,'좌빨'
        ,'김정일'
        ,'김정은'
    ]

    words = {}
    resultWords = []
    for word in wordList:
        simwords = model.wv.most_similar(word)

        resultWords.append('#'+word)
        for simword_ in simwords:
            if simword_ in words:
                continue
            words[simword_[0]] = 1
            resultWords.append(simword_[0])

    resultSeedF = open('resultDict_soc.txt','w',encoding='utf-8')
    resultSeedF.write('\n'.join(resultWords)+'\n')
    resultSeedF.close()
    # print(model.wv.most_similar('국민의짐'))
    # print(model.wv.most_similar('좌빨'))

# import sys
# sys.exit()
with open('dictionary.txt',encoding='utf-8') as dic_word:
    word_count = 0
    seed_dic = np.zeros(300)
    _duplicate = {}
    for word in dic_word:
        if word.startswith('//'):
            continue
        
        word = word.replace('#','').strip().split('\t')
        
        if word[0] in _duplicate:
            continue
            
        word_count += 1
        _duplicate[word[0]] = True
        # print(word)
        # print(model.wv[word[0]])
        seed_dic += model.wv[word[0]]
    seed_dic /= word_count
    np.save('gen_seed',seed_dic)

with open('dictionary_soc.txt',encoding='utf-8') as dic_word:
    word_count = 0
    seed_dic = np.zeros(300)
    _duplicate = {}
    for word in dic_word:
        if word.startswith('//'):
            continue
        
        word = word.replace('#','').strip().split('\t')
        
        if word[0] in _duplicate:
            continue
            
        word_count += 1
        _duplicate[word[0]] = True
        # print(word)
        # print(model.wv[word[0]])
        seed_dic += model.wv[word[0]]
    seed_dic /= word_count
    np.save('soc_seed',seed_dic)

line = '나 정말 창녀 질싸 좆물받이'
line = line.split(' ')
_dic = np.zeros(300)
for l in line:
    _dic += model.wv[l]

_dic /= len(line)
print(line)
gen_seed = np.load('gen_seed.npy')
print(cos_sim(_dic,gen_seed))

line = '나 정말 빨갱이 친일'
line = line.split(' ')
_dic = np.zeros(300)
for l in line:
    _dic += model.wv[l]

_dic /= len(line)
print(line)
soc_seed = np.load('soc_seed.npy')
print(cos_sim(_dic,soc_seed))

# import sys
# sys.exit()
word = {}
word['[PAD]'] = 0
word['[UNK]'] = 1
# word['']
wordindex = 2
import csv
csvf = open('dataset.csv','w',encoding='utf-8', newline='')
wr = csv.writer(csvf)
# wr.writerow([1,'림코딩', '부산'])
# wr.writerow([2,'김갑환', '서울'])
 
# f.close()
with open('preprocessing/ht_x.txt',encoding='utf-8') as htx:
    with open('preprocessing/ht_origin.txt',encoding='utf-8') as htori:
        for l, o in zip(htx,htori):
            l = l.strip()
            l = l.split(' ')

            temp_nouns = []
            for noun in l:
                try:
                    if noun not in word:
                        word[noun] = wordindex
                        wordindex += 1

                    temp_nouns.append(model.wv[noun])
                    
                except Exception as ex:
                    a = 1
            _worddic = np.array(temp_nouns)

                # print(_worddic)
            # print(_worddic)
                # print(np.sum(_worddic,axis=0).shape,l,len(temp_nouns))
            _worddic = np.sum(_worddic,axis=0) / len(temp_nouns)
            if len(temp_nouns) == 0:
                # _worddic = np.zeros(300)
                continue
            wr.writerow([' '.join(l),o.strip(),cos_sim(gen_seed,_worddic),cos_sim(soc_seed,_worddic)])
csvf.close()