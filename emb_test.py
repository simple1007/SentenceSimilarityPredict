from utils.utils import cos_sim_normal as cos_sim
from gensim.models import FastText, Word2Vec
import numpy as np
import sys
model = Word2Vec.load('ilbe_word2vec/ilbe_word2vec.model')
# f = open('dictionary_.txt','w',encoding='utf-8')

if len(sys.argv) >= 2 and sys.argv[1] == "emb":
    words = []
    while True:
        try:
            word = input('word: ')
            
            if word == 'exit':
                # f.close()
                exit()
            elif word == 'save':
                filen = input('file name: ')
                f = open('{}.txt'.format(filen),'w',encoding='utf-8')
                for w_ in words:
                    f.write('%s\t%s\n' % (w_[0],str(w_[1])))
                words = []
                f.close()
            sim_w = model.wv.most_similar_cosmul(word,topn=100)
            words.append(['#'+word,0.0])
            print(sim_w)
            for sw in sim_w:
                words.append(sw)
            # f.write(word+'\n')
            # for sw in sim_w:
            #     f.write(sw[0]+'\n')
                
        except Exception as ex:
            print(str(ex))

if len(sys.argv) >= 2 and sys.argv[1] == "2nd":
    f = open('1st.txt',encoding='utf-8')
    tot_score = 0.0
    cnt = 0
    for l in f:
        if l.startswith('#'):
            continue
        l = l.split('\t')[1].strip()
        score = float(l)
        cnt += 1
        tot_score += score
    avg_score = tot_score/cnt
    avg_score = avg_score + (avg_score * 0.2)
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

def word_most_sim(wordList,file_pre):
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

    resultSeedF = open('resultDict_'+file_pre+'.txt','w',encoding='utf-8')
    resultSeedF.write('\n'.join(resultWords)+'\n')
    resultSeedF.close()

def make_dic(file_pre):
    with open(file_pre+'.txt',encoding='utf-8') as dic_word:
        word_count = 0
        seed_dic = np.zeros(300)
        _duplicate = {}
        for word in dic_word:
            if word.startswith('//'):
                continue
            
            word = word.replace('#','').strip().split('\t')
            
            if word[0] in _duplicate:
                continue

            try:
                seed_dic += model.wv[word[0]]
            except:
                continue
            word_count += 1
            _duplicate[word[0]] = True
            # print(word)
            # print(model.wv[word[0]])
            
        seed_dic /= word_count
        np.save(file_pre,seed_dic)

# print(model.wv.most_similar('노키즈존'))
# print(model.wv.most_similar('맘충'))
# print(model.wv.most_similar('맘카페'))
# print(model.wv.most_similar('학식충'))
# print(model.wv.most_similar('노안'))
# print(model.wv.most_similar('패드립'))
# print(model.wv.most_similar('폐드립'))
# print(model.wv.most_similar('니엄마'))
# print(model.wv.most_similar('느금마'))
# print(model.wv.most_similar('니애미'))
# print(model.wv.most_similar('느검마'))
# print(model.wv.most_similar('느그애미'))
# print(model.wv.most_similar('니애비'))
# print(model.wv.most_similar('느그애비'))
# print(model.wv.most_similar('틀딱충'))
# print(model.wv.most_similar('틀딱새끼야'))
# print(model.wv.most_similar('빠돌이'))
# print(model.wv.most_similar('빠순이'))
# print(model.wv.most_similar('느그애미'))
# print(model.wv.most_similar('갓수'))
# print(model.wv.most_similar('히키코모리'))
# print(model.wv.most_similar('히키'))
# print(model.wv.most_similar('아싸'))
# print(model.wv.most_similar('지잡대'))
# print(model.wv.most_similar('수시충'))
# print(model.wv.most_similar('지잡'))
# print(model.wv.most_similar('오타쿠'))
# print(model.wv.most_similar('캣맘'))
# print(model.wv.most_similar('찐따'))
# print(model.wv.most_similar('일진'))
# print(model.wv.most_similar('왕따새끼'))
# print(model.wv.most_similar('따새끼'))
# print(model.wv.most_similar('찐따새끼'))
# print(model.wv.most_similar('빵셔틀'))

ageList = [
    '틀딱'
    ,'박사모'
    ,'꼰대'
    ,'초딩'
    ,'중딩'
    ,'고딩'
    ,'급식충'
    ,'좆고딩'
    ,'노인네'
    ,'노친네'
    ,'늙은이'
    ,'닭사모'
    ,'관짝'
    ,'틀니'
    ,'노인네'
    ,'산송장'
    ,'틀니충'
    ,'할배'
    ,'할매'
    ,'아줌마'
    ,'아재'
    ,'늙은이'
    ,'검버섯'
    ,'애미'
    ,'애비'
    ,'노총각'
    # ,'씹틀딱'
]

# word_most_sim(ageList,'age')
# make_dic('age')
# import sys
# sys.exit()
locationList = [
    '홍어'
    ,'홍어새끼들'
    ,'탈라도'
    ,'홍어들'
    ,'경상'
    ,'경상도'
    ,'지역감정'
    ,'전라디언'
    ,'절라도'
    ,'탈라도'
    ,'국민의당'
    ,'쌍도'
    ,'개쌍도'
    ,'대구놈들'
    ,'홍어천지'
    ,'촌년'
    ,'촌놈'
    ,'탐라국'
    ,'강정마을'
    ,'탐라'
    ,'홍어밭'
    ,'감자국'
    ,'설라디언'
]

# word_most_sim(locationList,'location')
# make_dic('location')
# print(model.wv.most_similar(['쌍도'],topn=20))
if False:
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
word_count = 0
seed_dic = np.zeros(300)
_duplicate = {}
with open('dictionary_gen.txt',encoding='utf-8') as dic_word:
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
    # seed_dic /= word_count
    # np.save('gen_seed',seed_dic)

with open('dictionary_soc.txt',encoding='utf-8') as dic_word:
    # word_count = 0
    # seed_dic = np.zeros(300)
    # _duplicate = {}
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
np.save('total',seed_dic)
total = seed_dic
# total = np.load('total.npy')?

# make_dic(
#     'soc'
# )

# make_dic(
#     'gen'
# )

# soc = np.load('soc.npy')
# gen = np.load('gen.npy')
# make_dic('soc')
# soc_seed = np.load('soc.npy')

# make_dic('age')
# age_seed = np.load('age_seed.npy')

# # make_dic('location')
# location_seed = np.load('location.npy')

# line = '나 정말 창녀 질싸 좆물받이'
# line = line.split(' ')
# _dic = np.zeros(300)
# for l in line:
#     _dic += model.wv[l]

# _dic /= len(line)
# print(line)
# gen_seed = np.load('gen_seed.npy')
# print(cos_sim(_dic,gen_seed))

# line = '나 정말 빨갱이 친일'
# line = line.split(' ')
# _dic = np.zeros(300)
# for l in line:
#     _dic += model.wv[l]

# _dic /= len(line)
# print(line)
# soc_seed = np.load('soc_seed.npy')
# print(cos_sim(_dic,soc_seed))

# import sys
# sys.exit()
word = {}
word['[PAD]'] = 0
word['[UNK]'] = 1
# word['']
wordindex = 2
import csv
import re
csvf = open('dataset.csv','w',encoding='utf-8', newline='')
wr = csv.writer(csvf)
# wr.writerow([1,'림코딩', '부산'])
# wr.writerow([2,'김갑환', '서울'])
# make_dic('total')
# total = np.load('total.npy')
# f.close()
with open('preprocessing/ht_x.txt',encoding='utf-8') as htx:
    with open('preprocessing/ht_origin.txt',encoding='utf-8') as htori:
        for l, o in zip(htx,htori):
            l = l.strip()
            l = l.split(' ')

            o = o.strip()
            o = re.sub(' +',' ',o)
            o = o.split(' ')

            temp_nouns = []
            for noun in o:
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
            # print('test',_worddic.shape)
            if len(temp_nouns) == 0:
                # _worddic = np.zeros(300)
                continue
            wr.writerow([' '.join(l),' '.join(o),cos_sim(total,_worddic)])
csvf.close()