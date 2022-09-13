from konlpy.tag import Okt

o = Okt()
result = open('preprocessing/nouns.txt','w',encoding='utf-8')

with open('rawdata/data.txt',encoding='utf-8') as f:
    for l in f:
        l = l.strip()

        noun = o.nouns(l)
        result.write(' '.join(noun)+'\n')

result.close()