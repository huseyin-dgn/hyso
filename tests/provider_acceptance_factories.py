"""Tiny self-contained P35 native acceptance graphs; no downloads or models."""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def partition_model():
    nodes = [helper.make_node('Relu',['X'],['R'],name='relu_gpu'),
             helper.make_node('Scaler',['R'],['Y'],name='scaler_cpu',domain='ai.onnx.ml',
                              offset=[.5,1.,1.5],scale=[2.,2.,2.])]
    graph = helper.make_graph(nodes,'p35-controlled-partition',
        [helper.make_tensor_value_info('X',TensorProto.FLOAT,[2,3])],
        [helper.make_tensor_value_info('Y',TensorProto.FLOAT,[2,3])])
    model = helper.make_model(graph,opset_imports=[helper.make_opsetid('',13),helper.make_opsetid('ai.onnx.ml',1)],ir_version=8)
    return model.SerializeToString(), {'X':np.array([[-1,2,3],[4,-5,6]],dtype=np.float32)}


def fusion_model():
    nodes = [helper.make_node('MatMul',['X','W'],['M'],name='matmul'),
             helper.make_node('Add',['M','B'],['A'],name='bias_add'),
             helper.make_node('Relu',['A'],['Y'],name='relu')]
    graph = helper.make_graph(nodes,'p35-real-fusion',
        [helper.make_tensor_value_info('X',TensorProto.FLOAT,[2,3])],
        [helper.make_tensor_value_info('Y',TensorProto.FLOAT,[2,3])],
        initializer=[numpy_helper.from_array(np.eye(3,dtype=np.float32),name='W'),
                     numpy_helper.from_array(np.array([.25,-.5,1.],dtype=np.float32),name='B')])
    model = helper.make_model(graph,opset_imports=[helper.make_opsetid('',13)],ir_version=8)
    return model.SerializeToString(), {'X':np.array([[-1,2,3],[4,-5,6]],dtype=np.float32)}
