import re
from xml.etree.ElementTree import QName

resultRawF = open('preprocessing/ht_x.txt','w',encoding='utf-8')
resultYF = open('preprocessing/ht_y.txt','w',encoding='utf-8')
resultOrigin = open('preprocessing/ht_origin.txt','w',encoding='utf-8')

with open('rawdata/bilstm_bigram_train_hate_pos.txt',encoding='utf-8') as posData:
    with open('rawdata/bilstm_bigram_train_hate_pos_y.txt',encoding='utf-8') as yData:
        for pos, y in zip(posData,yData):
            pos = pos.strip()
            pos = re.sub(' +',' ',pos)
            pos = pos.split(' ')

            temp = []
            origin = []
            
            for p in pos:
                tempword = ''
                if '+' in p:
                    p = p.split('+')
                    pp = p[0].split('/')

                    if pp[1].startswith('N'):
                        temp.append(pp[0])
                    tempword = p[0].split('/')[0] + p[1].split('/')[0]
                else:
                    pp = p.split('/')
                    
                    if pp[1].startswith('N'):
                        temp.append(pp[0])
                    tempword = pp[0]
                origin.append(tempword)
            resultNoun = ' '.join(temp)

            if resultNoun.strip() != '':
                resultRawF.write(resultNoun+'\n')
                resultYF.write(y)
                resultOrigin.write(' '.join(origin)+'\n')

resultRawF.close()
resultYF.close()
resultOrigin.close()