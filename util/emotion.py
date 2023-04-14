class Emotion:
    def __init__(self):
        pass

    HAPPY = 5
    SURPRISE = 1
    NEUTRALITY = 4
    SADNESS = 3
    ANGER = 2
    ANXIETY = 0
    AVERSION = 6


    def to_string(self, num):
        if num == self.HAPPY:
            return "행복"
        if num == self.SURPRISE:
            return "놀람"
        if num == self.NEUTRALITY:
            return "중립"
        if num == self.SADNESS:
            return "슬픔"
        if num == self.ANGER:
            return "분노"
        if num == self.ANXIETY:
            return "불안"
        if num == self.AVERSION:
            return "혐오"


    def to_num(self, st):
        st = st.strip()
        if st == "행복":
            return self.HAPPY
        if st == "놀람":
            return self.SURPRISE
        if st == "중립":
            return self.NEUTRALITY
        if st == "슬픔":
            return self.SADNESS
        if st == "분노":
            return self.ANGER
        if st == "불안":
            return self.ANXIETY
        if st == "혐오":
            return self.AVERSION
