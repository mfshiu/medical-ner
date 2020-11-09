# coding: utf-8
from common.config import GPU
from common.config import Device as GPU_Device


if GPU:
    import cupy as np

    np.cuda.Device(GPU_Device).use()
    print('Use GPU: %d' % (GPU_Device,))

    pool = np.cuda.MemoryPool(np.cuda.malloc_managed)
    np.cuda.set_allocator(pool.malloc)

    print('\033[92m' + '-' * 60 + '\033[0m')
    print(' ' * 23 + '\033[92mGPU Mode (cupy)\033[0m')
    print('\033[92m' + '-' * 60 + '\033[0m\n')
else:
    import numpy as np
