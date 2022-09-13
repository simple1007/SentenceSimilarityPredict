from konlpy.tag import Okt

o = Okt()
result = open('preprocessing/morphs.txt','w',encoding='utf-8')
tags = ['Noun', 'Adjective', 'Adverb', 'Verb']
with open('rawdata/data.txt',encoding='utf-8') as f:
    for l in f:
        l = l.strip()

        # noun = o.nouns(l)
        pos = o.pos(l,stem=True)

        temp = []
        for p in pos:
            if p[1] in tags:
                temp.append(p[0])
        result.write(' '.join(temp)+'\n')

result.close()