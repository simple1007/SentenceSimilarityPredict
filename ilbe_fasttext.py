from gensim.models.fasttext import FastText
data = []

with open('preprocessing/ilbe_nouns.txt',encoding='utf-8') as f:
    for l in f:
        l = l.strip()
        data.append(l.split(' '))

ftmodel = FastText(min_count=1,vector_size=300,workers=6)
ftmodel.build_vocab(data)
ftmodel.train(data,total_examples=len(data), epochs=50)

ftmodel.save('ilbe_fasttext.model')