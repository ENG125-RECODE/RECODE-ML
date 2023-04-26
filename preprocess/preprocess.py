import pandas as pd

def korean_one_off_conversation():
    data = pd.read_csv('/content/drive/MyDrive/한국어_단발성_대화_데이터셋.csv')
    data.loc[(data['Emotion'] == "공포"), 'Emotion'] = 0  # 공포 => 0
    data.loc[(data['Emotion'] == "놀람"), 'Emotion'] = 1  # 놀람 => 1
    data.loc[(data['Emotion'] == "분노"), 'Emotion'] = 2  # 분노 => 2
    data.loc[(data['Emotion'] == "슬픔"), 'Emotion'] = 3  # 슬픔 => 3
    data.loc[(data['Emotion'] == "중립"), 'Emotion'] = 4  # 중립 => 4
    data.loc[(data['Emotion'] == "행복"), 'Emotion'] = 5  # 행복 => 5
    data.loc[(data['Emotion'] == "혐오"), 'Emotion'] = 6  # 혐오


    # !wget -O .cache/ratings_train.txt http://skt-lsl-nlp-model.s3.amazonaws.com/KoBERT/datasets/nsmc/ratings_train.txt
    # !wget -O .cache/ratings_test.txt http://skt-lsl-nlp-model.s3.amazonaws.com/KoBERT/datasets/nsmc/ratings_test.txt
    small_range = list(range(0, 38594, 50))
    data.iloc[small_range]

if __name__ == '__main__':
    korean_one_off_conversation()
