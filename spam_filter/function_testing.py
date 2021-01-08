import heterocl as hcl
import numpy as np
import os

from constants import *
from lut import *

def top_dot_product():
  def dot_product_test(dtype = hcl.Float()):  
  
      hcl.init(dtype)
      
      def dot_product(A,B, out):
        with hcl.for_(0, NUM_FEATURES, name = "dot_product_loop") as i:
          out[0] += A[i]*B[i]
          
      A = hcl.placeholder((NUM_FEATURES,), "A")
      B = hcl.placeholder((NUM_FEATURES,), "B")
      out = hcl.placeholder((1,), "out")
      
      s = hcl.create_schedule([A, B, out], dot_product)
      return hcl.build(s)
  
  Tdtype = hcl.Float()
  f = dot_product_test(Tdtype)
  
  np_A =  hcl.cast_np(np.full((NUM_FEATURES,), 1, dtype = int), Tdtype)
  np_B = hcl.cast_np(np.full((NUM_FEATURES,), 1, dtype = int), Tdtype)
  np_dot = hcl.cast_np([0], Tdtype)
  
  hcl_A = hcl.asarray(np_A, dtype = Tdtype)
  hcl_B = hcl.asarray(np_B, dtype = Tdtype)
  hcl_dot = hcl.asarray(np_dot, dtype = Tdtype)
  
  f(hcl_A, hcl_B, hcl_dot)
  
  dot = hcl_dot.asnumpy()
  
  print(dot)

#top_dot_product()
def top_lut():
    def lut_testing(dtype = hcl.Float()):
    
        hcl.init(dtype)
        
        def use_lut(lut, in_, val):
          temp_in = hcl.scalar(in_, dtype = hcl.Int())
          with hcl.if_(in_ < 0):
                #in_ = (int)(-1 * in_)
            index = LUT_SIZE - (temp_in) << (LUTIN_TWIDTH - LUTIN_IWIDTH)
          with hcl.else_():
            index = (temp_in) << (LUTIN_TWIDTH - LUTIN_IWIDTH)
    
          val[0] = lut[index]
        
        # Call: Sigmoid(dot, prob)
        def Sigmoid(lut, exponent, prob):
            
            with hcl.if_(exponent[0] > 4):
                prob.v = 1.0
            with hcl.elif_(exponent[0] < -4):
                prob.v = 0.0
            with hcl.else_():
                use_lut(lut, exponent[0], prob)
        
        lut = hcl.placeholder((LUT_SIZE,), "lut")
        exponent = hcl.placeholder((1,), "exponent")
        prob = hcl.placeholder((1,), "prob")
        
        s = hcl.create_schedule([lut, exponent, prob], Sigmoid)
        
        return hcl.build(s)
    
    Ldtype = hcl.Fixed(10,5)
    
    np_lut = hcl.cast_np(lut, dtype = Ldtype)
    
    hcl_lut = hcl.asarray(np_lut, dtype = Ldtype)
    hcl_exponent = hcl.asarray([2], dtype = Ldtype) 
    hcl_prob = hcl.asarray([0], dtype = Ldtype)
    
    f = lut_testing(Ldtype)
    f(hcl_lut, hcl_exponent, hcl_prob)
    
    prob = hcl_prob.asnumpy()
    print(prob)

def top_update_param():    
    def update_param_testing(dtype = hcl.Float()):
    
        hcl.init(dtype)
        def update_parameter(param_theta, param_grad, param_scale):
          with hcl.for_(0, NUM_FEATURES, name = "update_parameter_loop") as i: 
            param_theta[i] += param_grad[i] * param_scale[0]
            
        param_theta = hcl.placeholder((NUM_FEATURES,), "param_theta")
        param_grad = hcl.placeholder((NUM_FEATURES,), "param_grad")
        param_scale = hcl.placeholder((1,), "param_scale")
        
        s = hcl.create_schedule([param_theta, param_grad, param_scale], update_parameter)
        return hcl.build(s)
        
       
    dtype = hcl.Fixed(40,20) 
    
    np_theta =  hcl.cast_np(np.full((NUM_FEATURES,), 1, dtype = int), dtype)
    np_grad = hcl.cast_np(np.full((NUM_FEATURES,), 1, dtype = int), dtype)
    np_scale = hcl.cast_np([2], dtype)
        
    hcl_theta = hcl.asarray(np_theta, dtype = dtype)
    hcl_grad = hcl.asarray(np_grad, dtype = dtype)
    hcl_scale = hcl.asarray(np_scale, dtype = dtype)
    
        
    f = update_param_testing(dtype)
    
    f(hcl_theta, hcl_grad, hcl_scale)
      
    theta = hcl_theta.asnumpy()
    grad = hcl_grad.asnumpy()
    scale = hcl_scale.asnumpy()
    
    print(theta)
    print(grad)
    print(scale)
 
def top_slicing_test():   
    def slicing_testing(dtype = hcl.Float()):
        hcl.init(dtype)
        
        OUT_NUM = 5
        IN_NUM = 2
        def slicing(data):
            with hcl.for_(0, OUT_NUM, name = "training_loop") as i:
              d = hcl.compute((IN_NUM,), lambda x: data[i*IN_NUM + x], "dot_product")
              hcl.print(d)
            
        data= hcl.placeholder((OUT_NUM*IN_NUM, ), "data")
        
        s = hcl.create_schedule([data], slicing)
        return hcl.build(s)
        
    
    dtype = hcl.Float()
    np_data = hcl.cast_np([0,1,2,3,4,5,6,7,8,9], dtype = dtype)
    
    hcl_data = hcl.asarray(np_data, dtype = dtype)
    
    f = slicing_testing(dtype)
    f(hcl_data)
          
 
LABEL_SIZE = 10 
GRAD_SIZE = 4        
def grad_testing(dtype = hcl.Float()):
    hcl.init(dtype)
    
    def grad(data, prob, label):
        with hcl.for_(0, LABEL_SIZE) as i:
          grad = hcl.compute((GRAD_SIZE,), lambda x: data[x] * (prob[0]-label[i]), "grad")    
          hcl.print(grad)
        
    prob = hcl.placeholder((1,), "prob")
    
    data = hcl.placeholder((GRAD_SIZE,), "data_array")
    label = hcl.placeholder((LABEL_SIZE, ), "label")
    
    s = hcl.create_schedule([data, prob, label], grad)
    return hcl.build(s)    
    
dtype = hcl.Float()
np_d = hcl.cast_np(np.full((GRAD_SIZE,), 1, dtype = float), dtype)
np_prob = hcl.cast_np([2.928376589], dtype = dtype)
np_label = hcl.cast_np(np.full((LABEL_SIZE,), 1, dtype = float), dtype)

hcl_d = hcl.asarray(np_d, dtype = dtype)
hcl_prob = hcl.asarray(np_prob, dtype = dtype)
hcl_label = hcl.asarray(np_label, dtype = dtype)

f = grad_testing(dtype)
f(hcl_d, hcl_prob, hcl_label)