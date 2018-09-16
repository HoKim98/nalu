class NumInterface:
    EOS = 0

    def __init__(self):
        pass

    def generate(self, nums):
        raise NotImplementedError

    def to_string(self, nums):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
