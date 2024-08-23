import torch


def test_memory():
    x = torch.randn(2, 25, 3, 256, 256)
    # print memory usage before and after putting x on GPU in MB
    print(torch.cuda.memory_allocated() / 1024 / 1024)
    x = x.cuda()
    print(torch.cuda.memory_allocated() / 1024 / 1024)
    del x
    # print memory usage after deleting x in MB


test_memory()
