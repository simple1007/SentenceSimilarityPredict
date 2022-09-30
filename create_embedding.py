import tensorflow as tf
import numpy as np
import sentencepiece as spm

from utils.utils import cos_sim

maxlen = 300

w4 = '빨갱이 좌빨 가족 랍시고 데리 살기 싫다 애초 빨리 거르는 이득 임'#['웹툰']
w5 = '북괴 있다 모병제 하자 새끼 짱깨 간첩 새끼'#['마음','소리']
w6 = '총선 짱깨 새끼 선거 거 모르다 병신 있다'
w7 = '살인마 살인 건 타 거지 내 김정은 따다 죽이다 국가 유공 되다 평화조약 첩 때문 아니다'
w8 = '김대중 개새끼 맞다 노무현 소아마비 비서 따먹다 거 다 들다 알 임신 해 딸 낳다 건평 걷다 주다 좌파 본색'
w9 = '개도 유행 끝나다 버리다 씨발 좌좀 개새끼'
w10 = '실제 반일 선동 하다 공부 좌파 쌍것 조선시대 양반 벤치마킹 하다 꼴임 몇 년 뭐 밟다 어쩌 하다 전 양반 짐슴 취급 받다 인간 대접 받다 건 생각 하다 그렇다 조선 왕조 전통 어쩌 팔 누가 보다 최소 사대부 집안 줄 알다 대한민국 역사 불과 년 안되다 신생 국가 전 한반도 땅 이씨 왕조 조선 일 뿐 왜 나라 생기 기도 전의 굴복 수치 대한민국 국민 난리 부르스 치다 오히려 대한민국 주적 건국 수많다 생명 재산 앗 간 북한 괴뢰 정부 공산당 국가 좀 팔 이렇다 것 좀 교육 하다 좋다 막말 옆집 칼부림 나다 문 꼭 잠그다 구경 하다 새끼 전 있다 일로 개 지랄 일제 새끼 들이다'
w11 = '조선 개 같다 나라 짱개 빌붙다 자 국민 노예 삼은 개좆 같다 조선'
w12 = '좃 병신 짱개 색기 토나오다'

words = [w4,w5,w6,w7,w8,w9,w10,w11,w12]

model = tf.keras.models.load_model('embedding.model')

input = tf.keras.layers.Input((maxlen))
emb = model.emb(input)
# input = model.layers[1].input
# emb = model.layers[0].output
output = model.bilstm(emb)

model = tf.keras.models.Model(inputs=input,outputs=output)
# model.save('sentence_embedding.model')

sp = spm.SentencePieceProcessor()
vocab_file = 'ilbe_spm_model/ilbe_spm.model'
sp.load(vocab_file)

soc = np.zeros(128)

for word in words:
    tk = sp.encode_as_ids(word)
    length = len(tk)
    tk = tk + [0] * (maxlen-length)
    pred = model(np.array([tk]))
    pred = pred[0][:length,:]

    sent_emb = np.zeros(128)

    for pred_ in pred:
        sent_emb += pred_

    sent_emb /= length
    soc += sent_emb

soc /= len(words)

np.save('soc_emb',soc)

gender = np.zeros(128)

w4 = '내 씨발 진짜 농담 아니다 저런 기생충 구더기 곱등이 연가시 시구 바퀴벌레 시궁창 쥐 새끼 오원춘 조두순 강호순 이춘 재 김길태 히틀러 모택동 애 밉다 애비 췌장 돌림빵 치다 당한 좆 페미니스트 유사 인류 씹다 쓰레기 남혐충 밉다 뒤지다 걸레 갈다 보 씹다 창년 허다 벌 보지 니미 씹다 창 좆 물통 사시미 토막 내다 도려내다 다음 마체테 찢다 죽이다 버리다 변기 개 갈다 보지 씹다 창년 새끼 사지 전기톱 토막 내다 죽이다 버리다 다음 닭 이통 쑤시다 치다 갈아 버리다 다른 치료 방법 없다 생각 하다 진심'#['웹툰']
w5 = '알다 보빨 한남 썅놈판검 새끼 더 답 없다'#['마음','소리']
w6 = '근대 지장사 하다 전라도 들어가다 정치 우파 인대 자지 좌파 됫더 섹스 존나 밝히다'
w7 = '견적 보다 저능 년 나라 가다 니거 보지 벌리다 살다'
w8 = '인간 지다 부모 섹스 안 어디 다리 밑 줍다 왓 보다 섹스 왜 범죄 임 섹스 범죄 인류 벌써 멸망 햇 생기다'
w9 = '페미 자체 여성 주의 운동 병신 사상'
w10 = '미치다 놈 씨발 여혐 정도 껏 하다'
w11 = '한국 창녀 존나 많다'
w12 = '애 밉다 조선족 명 후 명 보짓구녕 명 입구 녕 명 귓구녕 다발 박다 오르가즘 느끼다 개 씹다 창 허벌쎅년 년 새끼 낳다 니 인생 개 씹다 창 병신 새끼 잖다'

words = [w4,w5,w6,w7,w8,w9,w10,w11,w12]

for word in words:
    tk = sp.encode_as_ids(word)
    length = len(tk)
    tk = tk + [0] * (maxlen-length)
    pred = model(np.array([tk]))
    pred = pred[0][:length,:]

    sent_emb = np.zeros(128)

    for pred_ in pred:
        sent_emb += pred_

    sent_emb /= length
    gender += sent_emb

gender /= len(words)

np.save('gender_emb',gender)

model.save('sentence_embedding.model')