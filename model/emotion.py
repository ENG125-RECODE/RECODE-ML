import gluonnlp as nlp
import numpy as np
import torch
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer
from torch.utils.data import Dataset



from model import classifier
import pickle

#from model.classifier import BERTClassifier
from util.emotion import Emotion

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
device = torch.device("cuda:0")

bertmodel, vocab = get_pytorch_kobert_model()

kobert_model, vocab = get_pytorch_kobert_model()
model = classifier.BERTClassifier(kobert_model, dr_rate=0.5, num_classes=3)

ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)

# 수정 필요 emotion_clsf_weights_file = "./checkpoint/emotion_pn.pth"
#model.load_state_dict(torch.load(emotion_clsf_weights_file, map_location=device))


model = torch.load('model/emotion_model.pt', map_location=torch.device('cpu')) #전체 모델을 통째로 불러온다. 클래스 선언 필수
#model.load_state_dict(torch.load('model/7emotions_model_state_dict.pt',map_location=torch.device('cpu'))) #state_dict를 불러 온 후, 모델에 저장



tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

import torch
import torch.nn as nn


class BERTDataset(Dataset):  # KoBERT 모델의 입력으로 들어갈 수 있는 형태가 되도록 토큰화, 정수 인코딩, 패딩 등을 해주는 것이다.
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, vocab=vocab, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i],))

    def __len__(self):
        return (len(self.labels))

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=7,  # 7가지 감정으로 분류
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
                              attention_mask=attention_mask.float().to(token_ids.device), return_dict=False)
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

model = BERTClassifier(model,  dr_rate=0.5).to(device)
model.load_state_dict(torch.load('model/emotion_model_state_dict.pt',map_location=device))
model.eval()

def predict(predict_sentence):
    max_len = 64 # 해당 길이를 초과하는 단어에 대해선 bert가 학습하지 않음
    batch_size = 64
    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, vocab, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)

    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length = valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)
        for i in out:
            logits=i
            logits = logits.detach().cpu().numpy()

            return np.argmax(logits)

