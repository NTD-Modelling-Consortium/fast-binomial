from fast_binomial import (
    FastBinomial
)
import numpy as np
test = np.array([[0,3,0],[0,1,3]], dtype=int)

gen = FastBinomial(
    p = 0.5
)
gen(test)
