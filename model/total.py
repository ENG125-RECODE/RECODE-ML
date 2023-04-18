from io import BytesIO

import torch
import os
from torch import nn
from torch.utils.data import Dataset
import gluonnlp as nlp
from flask import Flask, jsonify, request, make_response
from urllib import parse

from flask import Flask, jsonify

import numpy as np

from util.emotion import Emotion
from word.twitter import twitter
from word.word_visual import word_cloud

Emotion = Emotion()

import json

# kobert
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model


device = torch.device('cpu')
bertmodel, vocab = get_pytorch_kobert_model()

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i],))

    def __len__(self):
        return (len(self.labels))


max_len = 64 # 해당 길이를 초과하는 단어에 대해선 bert가 학습하지 않음
batch_size = 64
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=7,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)



model = torch.load('emotion_model.pt', map_location=torch.device('cpu'))  # 전체 모델을 통째로 불러온다. 클래스 선언 필수
model.load_state_dict(
    torch.load('emotion_model_state_dict.pt', map_location=torch.device('cpu')))  # state_dict를 불러 온 후, 모델에 저장


app = Flask(__name__)

@app.route('/')
def hello():
    return "deep learning server is running 💗"

@app.route('/api/emotion', methods=['POST'])
def predict():
    args = request.json
    sentence = args['content']
    sentence = parse.unquote(sentence, 'utf8')

    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    data = [sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=0)

    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length = valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)

        test_eval = []
        for i in out:
            logits = i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == Emotion.ANXIETY:
                test_eval.append(Emotion.to_string(Emotion.ANXIETY))
            elif np.argmax(logits) == Emotion.SURPRISE:
                test_eval.append(Emotion.to_string(Emotion.SURPRISE))
            elif np.argmax(logits) == Emotion.ANGER:
                test_eval.append(Emotion.to_string(Emotion.ANGER))
            elif np.argmax(logits) == Emotion.SADNESS:
                test_eval.append(Emotion.to_string(Emotion.SADNESS))
            elif np.argmax(logits) == Emotion.NEUTRALITY:
                test_eval.append(Emotion.to_string(Emotion.NEUTRALITY))
            elif np.argmax(logits) == Emotion.HAPPY:
                test_eval.append(Emotion.to_string(Emotion.HAPPY))
            elif np.argmax(logits) == Emotion.AVERSION:
                test_eval.append(Emotion.to_string(Emotion.AVERSION))

        output = test_eval[0]
        return jsonify(output)

@app.route('/api/analysis/keywords', methods=['POST'])
def keyword():
    args = request.json
    sentence = args['content']
    sentence = parse.unquote(sentence, 'utf8')
    print(type(sentence))

    return twitter(sentence)

@app.route('/api/analysis/keywords/images', methods=['POST'])
def word_visual():
    args = request.json
    sentence = args['content']
    sentence = parse.unquote(sentence, 'utf8')
    sentence_list = sentence.split("\n")
    img_wordcloud = word_cloud(sentence_list)
    #img_wordcloud.to_file('test.jpg')

    response = upload(img_wordcloud)
    return response

def upload(img_wordcloud):
    img_binary = BytesIO()
    img_wordcloud.to_image().save(img_binary, format='PNG')
    img_binary.seek(0)
    response = make_response(img_binary.getvalue())
    response.headers['Content-Type'] = 'image/png'
    return response



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))