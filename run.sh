#!/bin/bash

# 可执行程序需接收7个参数，分别为：输入顶点特征长度 F0 ，第一层顶点特征长度 F1 ，第二次顶点特征长度 F2 ，图结构文件名，输入顶点特征矩阵文件名
# 均使用相对路径
./example_openmp.out 64 16 8 graph/1024_example_graph.txt embedding/1024.bin weight/W_64_16.bin weight/W_16_8.bin

