from konlpy.tag import Okt
from utils.utils import normalization

import pandas as pd
tags = ['Noun', 'Adjective', 'Adverb', 'Verb']

df = pd.read_csv('rawdata/ilbe_comments_1.2m.tsv', sep = '\t')
# print(len(df['']))
#print(df)
#import sys
#sys.exit()
o = Okt()
# result = open('preprocessing/ilbe_morphs.txt','w',encoding='utf-8')
fx = open('preprocessing/ilbe_x_morphs.txt','w',encoding='utf-8')
result = open('preprocessing/raw_corpus.txt','w',encoding='utf-8')
#with open('rawdata/ilbe_comments_1.2m.tsv',encoding='utf-8') as f:
# for l in df['댓글']:
lst = ['kcc150.txt','ilbe_x_morphs.txt']
cnt = [1115393,1115393]
from tqdm import tqdm
for i in range(len(lst)):
    with open('rawdata/'+lst[i],encoding='utf-8') as f:
        # for l in f:
        for _ in tqdm(range(cnt[i])):
            l = f.readline()
            l = l.strip()
            l = normalization(l)
            # noun = o.nouns(l)
            l = o.normalize(l)
            temp = o.nouns(l)

            # temp = []
            # for p in pos:
            #     if p[1] in tags:
            #         temp.append(p[0])
            # t = ' '.join(temp)
            # if t.strip() != '':
            temp = ' '.join(temp)
            if temp.strip() != '':
                result.write(temp.strip()+'\n')
                fx.write(l+'\n')

result.close()
fx.close