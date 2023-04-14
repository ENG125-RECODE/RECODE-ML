from urllib import parse

from konlpy.tag import Okt
from flask import Flask, jsonify, request
Okt=Okt()



def morph(input_data) : #형태소 분석
    preprocessed = Okt.pos(input_data)
    return preprocessed

def twitter(sentence):
    output = morph(sentence)
    result = [Okt.nouns(i) for i in output]  # 명사만 추출
    final_result = [r for i in result for r in i]
    #단어와 형용사만 반환하기
    return jsonify(final_result)

#@app.route('/api/analysis/summerize', methods=['POST'])
#def sum():
    #args = request.json
    #sentence = args['content']
    #text = parse.unquote(sentence, 'utf8')
    #summarize(text, split=True)

