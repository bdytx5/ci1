import numpy as np
import struct

with open('train-images-idx3-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    data = data.reshape((size, nrows, ncols))

import matplotlib.pyplot as plt
im = data[0,:,:]
import cv2
res = cv2.resize(im, dsize=(14, 14), interpolation=cv2.INTER_CUBIC)
x = np.array(res).flatten()


plt.imshow(res, cmap='gray')
plt.show()


# add ones to inputs 
