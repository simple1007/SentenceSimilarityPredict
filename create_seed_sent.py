import csv
import tensorflow as tf
import numpy as np
import sentencepiece as spm

class SentEmb:
    def __init__(self,sp,category,model):
        self.sp = sp
        
        self.category = category

        self.model = model

        self.maxlen = 300
        self.vector = np.zeros(128)
        self.cnt = 0
    
    def __dataset(self):
        self.dataset_file = open('dataset.csv',encoding='utf-8')
        self.dataset = csv.reader(self.dataset_file)

    def __sent_cate_file(self):
        self.result_file = open('{}_seed.txt'.format(self.category),'w',encoding='utf-8')
        self.__dataset()
        with open('dictionary_{}.txt'.format(self.category),encoding='utf-8') as f:
            temp = []
            for l in f:
                l = l.strip()
                if l.startswith('//'):
                    continue
                word = l.replace('#','').split('\t')[0]
                temp.append(word)
            
            for data in self.dataset:
                data_word = data[0].strip().split(' ')
                line = data[0].strip()
                for dw in data_word:
                    if dw in temp:
                        self.result_file.write(line+'\n')
                        break

        self.dataset_file.close()
        self.result_file.close()

    def create_sent_emb(self):
        self.__sent_cate_file()
        result_file = open('{}_seed.txt'.format(self.category),encoding='utf-8')
        
        for line in result_file:
            x = line.strip()
            x = self.sp.encode_as_ids(x)
            x = x + [0] * (self.maxlen - len(x))
            x = x[:self.maxlen]

            pred = self.model(np.array([x]))
            self.vector += pred[0]
            self.cnt += 1
            # print("젠더 혐오:",cos_sim(pred[0],gender),"정치 혐오:",cos_sim(pred[0],society),"연령 혐오:",cos_sim(pred[0],age),"지역 혐오:",cos_sim(pred[0],location))
        print("{} nums sent: {}".format(self.category,self.cnt))
        self.vector = self.vector / self.cnt
        np.save('emb/{}_sent'.format(self.category),self.vector)
        result_file.close()

if __name__ == '__main__':
    vocab_file = 'ilbe_ht_spm_model/ilbe_spm.model'
    sp = spm.SentencePieceProcessor()
    sp.load(vocab_file)

    model = tf.keras.models.load_model('emb/embedding_model')
    gen_sent_ds = SentEmb(sp,'gen',model)
    # soc_sent_ds = SentEmb(sp,'soc',model)

    gen_sent_ds.create_sent_emb()
    # soc_sent_ds.create_sent_emb()