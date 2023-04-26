from urllib import parse

from konlpy.tag import Okt
from flask import Flask, jsonify, request
from collections import Counter
Okt=Okt()

import csv

with open('../word/stopword.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    stop_words = [word for row in reader for word in row]
stop_words.append("못")
stop_words.append("내")

#stopwords = ['이', '그', '저', '것', '때', '번', '위', '바로', '지금', '또한', '때문', '그렇지만', '하지만', '그러나', '그래서', '그리고', '뿐만', '아니라', '게다가', '다만', '그러므로', '그런데', '따라서', '이러한', '저런', '어떤', '어느', '한', '두', '세', '그것', '이것', '저것', '예를', '들면', '예를', '들자면', '예를', '들어', '예컨대', '이와', '같다', '비슷하다', '같은', '이와', '비슷한', '말이다', '마찬가지', '이러한', '반면에', '반대로', '반대로', '말하자면', '이와', '반대로', '다시', '말하자면', '반드시', '그렇게', '하지', '않으면', '안', '그러니까', '그럼', '때', '문제', '보이', '있', '생각', '합니다', '많', '이제', '하는', '그것', '이것', '저것', '대한', '때문', '어떤', '어느', '없', '모르', '알', '못', '아닌가', '아님', '아니', '있다', '없다', '아래', '위', '그리고', '또한', '이러한', '저러한', '하지만', '그러나', '그리고', '그래서', '따라서', '하지', '않다', '한다', '있다']
def morph(input_data) : #형태소 분석
    preprocessed = Okt.pos(input_data)
    return preprocessed

def twitter(sen):

    output = morph(sen)
    result = []
    print(output)
    for i in output:
        tem = ''.join(i)
        nouns = Okt.nouns(tem)
        result += nouns
    final_result = [r for r in result if r not in stop_words]
    # 단어 빈도수 계산
    counter = Counter(final_result)
    # 가장 빈도가 높은 10개 단어 추출
    most_common_words = counter.most_common(10)

    result = [{"word": word, "count": count} for word, count in most_common_words]

    return jsonify(result)

#@app.route('/api/analysis/summerize', methods=['POST'])
#def sum():
    #args = request.json
    #sentence = args['content']
    #text = parse.unquote(sentence, 'utf8')
    #summarize(text, split=True)
    #result = [Okt.nouns(i) for i in output]  # 명사만 추출
    #final_result = [r for i in result for r in i]
    #단어와 형용사만 반환하기

