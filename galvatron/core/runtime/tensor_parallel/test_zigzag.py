import torch
#引入要检查的函数,但我不知道这样引入对不对，因为我不熟悉python import的相关用法
from galvatron.core.runtime.redistribute import _zigzag_transformation,  _reverse_zigzag_transformation


#设置test函数 是不是函数名字一定要是main

def main():
    #设计输入数据
    input_ = torch.arange(8,dtype=torch.float)
    print("input_",input_)
    cp_size = 1
    for i in range(4):
        tranformed_output = _zigzag_tranformation(input_, cp_size)#我不太清楚python的索引机制，让使用函数去变换的时候，他会新创建一个张量还是会在原地改变
        print("cp size", cp_size, "trnaformed output", tranformed_output)
        reverse_tranformed_output = _reverse_zigzag_tranformation(tranformed_output, cp_size)
        print("cp size", cp_size, " reversetrnaformed output", reverse_tranformed_output)
    print("finished")


if __name__ == "main":
    main()