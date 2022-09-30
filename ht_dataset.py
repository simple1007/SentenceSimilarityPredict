import csv

import numpy as np
import sentencepiece as spm

encoding_ = 'utf-8'
maxlen = 300

def spm_train(filename,save_path='ilbe_ht_spm_model/ilbe_spm',vocab_size=10000):
    #with open(f'preprocessing/{filename}', encoding='utf-8') as ff:
    templates = '--input={} --model_prefix={} --vocab_size={}'
    cmd = templates.format(filename,save_path,vocab_size)
    spm.SentencePieceTrainer.Train(cmd)

def to_numpy(file,spm,batch_size=32):
    length = []
    with open(file,encoding=encoding_) as f:
        rd = csv.reader(f)
        cnt = 1
        X = []
        Y = []
        for l in rd:
            line = l[1]
            y_gen = l[2]
            y_soc = l[3]
            sbw = spm.encode_as_ids(line)
            sbw = sbw + [0] * (maxlen - len(sbw))
            X.append(sbw)
            Y.append([float(y_gen),float(y_soc)])

            if len(X) == batch_size:
                X = np.array(X)
                Y = np.array(Y)

                np.save('ht/%05d_x' % cnt,X)
                np.save('ht/%05d_y' % cnt,Y)

                cnt += 1
                X = []
                Y = []
            # length.append(len(sbw))
        if len(X) > 0:
            X = np.array(X)
            Y = np.array(Y)

            np.save('ht/%05d_x' % cnt,X)
            np.save('ht/%05d_y' % cnt,Y)

    length = sorted(length,reverse=True)
    print(length[:50])

if False:
    inputf = f'preprocessing/ht_origin.txt'

    spm_train(inputf)

sp = spm.SentencePieceProcessor()
vocab_file = 'ilbe_ht_spm_model/ilbe_spm.model'
sp.load(vocab_file)
to_numpy('dataset.csv',sp)