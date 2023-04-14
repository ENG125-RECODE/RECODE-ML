from io import BytesIO

import konlpy
import numpy as np
from matplotlib import pyplot as plt
from wordcloud import WordCloud
from flask import Flask, jsonify, request, make_response
from urllib import parse

import PIL
import re


import pandas as pd

kkma = konlpy.tag.Kkma() #형태소 분석기 꼬꼬마(Kkma)

def word_cloud(sentence_list):

    df_word = koko(sentence_list)

    dic_word = df_word.set_index('word').to_dict()['count']

    icon = PIL.Image.open('../word/heart.png')

    img = PIL.Image.new('RGB', icon.size, (255,255,255))
    img.paste(icon, icon)
    img = np.array(img)

    wc = WordCloud(random_state = 123, font_path = '/Users/kwonjiyun/Downloads/GmarketSansTTF/GmarketSansTTFBold', width = 400,
               height = 400, background_color = 'white')

    img_wordcloud = wc.generate_from_frequencies(dic_word)

    return img_wordcloud




def koko(sentence):
    df = pd.DataFrame(sentence, columns=['Keyword'])

    df['Keyword'] = df['Keyword'].str.replace('[^가-힣]', ' ', regex=True)
    nouns = df['Keyword'].apply(kkma.nouns)
    nouns = nouns.explode()
    print(df)
    df_word = pd.DataFrame({'word': nouns})

    df_word['count'] = df_word['word'].str.len()
    df_word = df_word.query('count >= 2')
    return df_word


