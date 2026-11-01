"""Small actual inference fixtures for bounded P34/P35 owned workers."""
import time


def torch_model(mode='normal'):
    import torch
    from torch import nn
    class Model(nn.Module):
        def __init__(self):
            super().__init__();self.linear=nn.Linear(12,2)
        def forward(self,x):
            if mode=='hang':time.sleep(60)
            if mode=='oom':raise torch.cuda.OutOfMemoryError('controlled unit fixture OOM')
            if mode=='cuda_oom':torch.empty(512*1024*1024,device=x.device,dtype=torch.float32)
            if mode=='mutate':self.linear.weight.add_(1)
            return self.linear(x.flatten(1))
    return Model(),torch.ones(2,3,2,2)


def onnx_model(mode='normal'):
    import numpy as np
    import onnx
    from onnx import helper,TensorProto
    node=helper.make_node('Relu',['X'],['Y'],name='relu')
    graph=helper.make_graph([node],'deployment-test',
        [helper.make_tensor_value_info('X',TensorProto.FLOAT,[2,3])],
        [helper.make_tensor_value_info('Y',TensorProto.FLOAT,[2,3])])
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',13)],ir_version=8)
    if mode=='hang':time.sleep(60)
    return model.SerializeToString(),{'X':np.array([[-1,2,3],[4,-5,6]],dtype=np.float32)}
