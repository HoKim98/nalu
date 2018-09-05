from lang.num.ja import Num as _Base


class Num(_Base):

    def __init__(self):
        super().__init__()
        self.m_other = {
            0: '',
            1: '마이너스',
            2: '영',
            3: '일',
            4: '이',
            5: '삼',
            6: '사',
            7: '오',
            8: '육',
            9: '칠',
            10: '팔',
            11: '구',
            12: '십',
            13: '백',
            14: '천',
            15: '만',
            16: '억',
            17: '조',
            18: '경',
        }
