class BetaIterator:

    def __init__(self, beta_sum, max_ratio, length=3):
        self.beta_sum = beta_sum
        self.max_ratio = max_ratio
        self.length = length

    def __iter__(self):
        # The 0 in the last element because on the first call to next it is incremented and a series of 1s is returned

        self.ratios = [1] * self.length
        self.ratios[-1] = 0

        return self

    def __get_next_ratio(self):
        self.ratios[-1] = self.ratios[-1] + 1

        idx = len(self.ratios) - 1

        while idx != 0 and self.ratios[idx] > self.max_ratio:
            self.ratios[idx - 1] += 1
            self.ratios[idx] = 1

            idx -= 1

    def __next__(self):

        if all(b == self.max_ratio for b in self.ratios[:-1]) and self.ratios[-1] == self.max_ratio-1:
            raise StopIteration

        self.__get_next_ratio()

        # If this isn't 1, 1, 1 and all ratios are the same, then we skip it
        if self.ratios[0] != 1:
            if all(b == self.ratios[0] for b in self.ratios):
                self.__get_next_ratio()

        total = float(sum(self.ratios))

        betas = [(x/total) * self.beta_sum for x in self.ratios]

        return self.ratios, betas