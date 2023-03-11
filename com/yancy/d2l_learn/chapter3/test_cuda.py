import time
import torch


def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


init0 = time.time()
x_cpu = torch.ones(size=(10000, 10000))
init1 = time.time()
x_gpu = torch.ones(size=(10000, 10000), device=try_gpu())
init2 = time.time()
print('数据初始化速度:{:.4f} s和{:.4f} s'.format(init1 - init0, init2 - init1))

start1 = time.time()
torch.mm(x_cpu, x_cpu)
end1 = time.time()

start2 = time.time()
torch.mm(x_gpu, x_gpu)
end2 = time.time()

print('cpu运行速度:{:.4f} s, gpu运行速度:{:.4f} s'.format(end1 - start1, end2 - start2))
