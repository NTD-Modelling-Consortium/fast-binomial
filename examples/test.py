import numpy as np

from fast_binomial import FastBinomial

test = np.array([[0, 3, 0], [0, 1, 3]], dtype=int)

gen = FastBinomial(0.5, 1)
print(gen.generate(test))
