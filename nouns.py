from konlpy.tag import Okt
from utils.utils import normalization

import pandas as pd

df = pd.read_csv('rawdata/ilbe_comments_1.2m.tsv', sep = '\t')
#print(df)
#import sys
#sys.exit()
o = Okt()
result = open('preprocessing/ilbe_nouns.txt','w',encoding='utf-8')
fx = open('preprocessing/ilbe_x.txt','w',encoding='utf-8')
#with open('rawdata/ilbe_comments_1.2m.tsv',encoding='utf-8') as f:
for l in df['댓글']:
    #print(l)
    l = l.strip()
    l = normalization(l)
    noun = o.nouns(l)
    noun = ' '.join(noun)
    
    if noun.strip() != '':
        result.write(noun.strip()+'\n')
        fx.write(l+'\n')

result.close()
fx.close()