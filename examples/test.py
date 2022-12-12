import numpy as np

from fast_binomial import SFC64, Generator

test = np.array([[0, 3, 0], [0, 1, 3]], dtype=int)
gen = Generator(SFC64(), 0.5)
results = gen.binomial(test)
print(results)
