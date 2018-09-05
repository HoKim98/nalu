from lang.num._interface import NumInterface


class Num(NumInterface):

    def __init__(self):
        super(NumInterface, self).__init__()
        self.units_pow = [x for x in range(16, 21)]
        self.units = [x for x in range(13, 16)]
        self.nums = [x for x in range(3, 13)]
        self.m = {
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
            13: '십',
            14: '백',
            15: '천',
            16: '만',
            17: '억',
            18: '조',
            19: '경',
            # 20: '해',
            # 21: '자',
            # 22: '양',
            # 23: '구',
            # 24: '간',
            # 25: '정',
            # 26: '재',
            # 27: '극',
        }

    def generate(self, nums):
        def read_digit(n):
            x = []
            i = -1

            while n > 0:
                n, r = divmod(n, 10)
                if r > 0:
                    if i >= 0:
                        x.append(self.units[i])
                    if i == -1 or r != 1:
                        x.append(self.nums[r - 1])
                i += 1

            x = x[::-1]
            return x

        def read_pow(n):
            x = []

            if n == 0:
                return [2]
            if n < 0:
                n = abs(n)
                pre = [1]
            else:
                pre = []

            i = -1
            while n > 0:
                a = n % 10000
                if a > 0:
                    a = read_digit(a)
                    if i >= 0:
                        x = a + [self.units_pow[i]] + x
                    else:
                        x = a + x
                n //= 10000
                i += 1
            x = pre + x
            return x

        return [read_pow(int(n)) for n in nums]

    def to_string(self, nums):
        return [''.join([self.m[x] for x in n]) for n in nums]

    def __len__(self):
        return len(self.m)
