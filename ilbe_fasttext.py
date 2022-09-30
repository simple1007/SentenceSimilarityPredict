from gensim.models.fasttext import Word2Vec,FastText
data = []

with open('preprocessing/ht_x.txt',encoding='utf-8') as f:
    for l in f:
        l = l.strip()
        data.append(l.split(' '))

ftmodel = Word2Vec(min_count=2,vector_size=300,workers=6)
# ftmodel = FastText(min_count=1,vector_size=300,workers=6)
ftmodel.build_vocab(data)
ftmodel.train(data,total_examples=len(data),epochs=50)

ftmodel.save('ilbe_word2vec/ilbe_word2vec.model')