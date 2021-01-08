import heterocl as hcl
import numpy as np
import os

from constants import *
from lut import *

default_dtype = hcl.Float()
default_reg_type = float

def top_function(dtype=default_dtype, target=None):

    hcl.init(dtype)
    
    def dot_product(A,B, out):
      with hcl.for_(0, NUM_FEATURES, name = "dot_product_loop") as i:
        out[0] += A[i]*B[i]
      
  
    def use_lut(lut, in_, val):
      temp_in = hcl.scalar(in_, dtype = hcl.Int())
      with hcl.if_(in_ < 0):
        index = LUT_SIZE - (temp_in) << (LUTIN_TWIDTH - LUTIN_IWIDTH)
      with hcl.else_():
        index = (temp_in) << (LUTIN_TWIDTH - LUTIN_IWIDTH)

      val[0] = lut[index]
    
    
    def Sigmoid(lut, exponent, prob):
        
        with hcl.if_(exponent[0] > 4):
            prob[0] = 1.0
        with hcl.elif_(exponent[0] < -4):
            prob[0] = 0.0
        with hcl.else_():
            use_lut(lut, exponent[0], prob)
    
    
    def update_parameter(param_theta, param_grad, param_scale):
      with hcl.for_(0, NUM_FEATURES, name = "update_parameter_loop") as i: 
        param_theta[i] += param_grad[i] * param_scale[0]
    
    def final_algorithm(lut, data, label, theta):
      
      with hcl.for_(0, NUM_EPOCHS, name = "epoch_loop") as z:
        with hcl.for_(0, NUM_TRAINING, name = "training_loop") as i:
          d = hcl.compute((NUM_FEATURES,), lambda x: data[i*NUM_FEATURES + x], "dot_product")
          dot = hcl.compute((1,),lambda x: 0, "dot")
          dot_product(d, theta, dot)

          prob = hcl.compute((1,), lambda x: 0, "prob")
          Sigmoid(lut, dot, prob)
          
          grad = hcl.compute((NUM_FEATURES,), lambda x: d[x] * (prob[0]-label[i]), "grad")
          
          step_size = hcl.compute((1,), lambda x: -STEP_SIZE, "step_size")
          update_parameter(theta, grad, step_size)
          
          
    lut = hcl.placeholder((LUT_SIZE, ), "lut")
    data = hcl.placeholder((NUM_FEATURES*NUM_TRAINING, ), "data")
    label = hcl.placeholder((NUM_TRAINING,),"label")
    theta = hcl.placeholder((NUM_FEATURES,), "theta")
    
    schedule = hcl.create_schedule([lut, data, label, theta], final_algorithm)
    
    
    f = hcl.build(schedule, target = target)
    
    return f

def codegen():
    os.makedirs('device', exist_ok=True)
    targets = ['vhls', 'aocl']
    #hcl_lut = hcl.asarray(lut, dtype = hcl.Float())
    
    for target in targets:
        f = top_function(dtype=default_dtype, target=target)
        fp = open('device/kernel_' + target + '.cl', 'w')
        fp.write(f)
        fp.close()

def test():

    Ddtype = hcl.Float() #Fixed(40,20)

      
    np_lut = hcl.cast_np(lut, dtype = Ddtype)
    np_data = hcl.cast_np(np.loadtxt("shuffledfeats.dat"), dtype = Ddtype)
    np_label = hcl.cast_np(np.loadtxt("shuffledlabels.dat"), dtype = Ddtype)

    np_theta = hcl.cast_np(np.zeros((NUM_FEATURES,), dtype=default_reg_type), Ddtype)
    
    

    hcl_lut = hcl.asarray(np_lut, dtype = Ddtype)
    hcl_data = hcl.asarray(np_data, dtype = Ddtype)
    hcl_label = hcl.asarray(np_label, dtype = Ddtype)
    hcl_theta_out = hcl.asarray(np_theta, dtype = Ddtype)
    
    f = top_function(Ddtype)
    
   
    
    f(hcl_lut, hcl_data, hcl_label, hcl_theta_out)
    
    np_theta_out = hcl_theta_out.asnumpy()
    np_data = hcl_data.asnumpy()
    np_label = hcl_label.asnumpy()
    
    #np.set_printoptions(threshold=np.inf)

    print(np_theta_out)
    print(np_lut)
    print(np_data)
    print(np_label)
    
    
    
    np_train_data = np_data[:NUM_FEATURES * NUM_TRAINING]
    np_train_label = np_label[:NUM_TRAINING]
    
    train_error = 0.0
    for i in range(NUM_TRAINING):
        data = np_train_data[i * NUM_FEATURES : (i + 1) * NUM_FEATURES]
        dot = 0.0
        for j in range(NUM_FEATURES):
            dot += data[j] * np_theta_out[j]
        
        result = 1.0 if dot > 0 else 0.0
    
        if result != np_train_label[i]:
            train_error += 1.0
            
    print("training error rate")
    #print(train_error)
    print(train_error/NUM_TRAINING * 100)
   
    
    np_test_data = np_data[NUM_FEATURES * NUM_TRAINING:]
    np_test_label = np_label[NUM_TRAINING:]

    test_error = 0.0
    for i in range(NUM_TESTING):
        data = np_test_data[i * NUM_FEATURES : (i + 1) * NUM_FEATURES]
        dot = 0.0
        for j in range(NUM_FEATURES):
            dot += data[j] * np_theta_out[j]
        
        result = 1.0 if dot > 0 else 0.0
    
    
        if result != np_test_label[i]:
            test_error += 1.0
    print("testing error rate")
    #print(test_error)
    print(test_error/NUM_TESTING * 100)


if __name__ == "__main__":
    test()
    #codegen()