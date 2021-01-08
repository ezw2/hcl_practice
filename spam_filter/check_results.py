import heterocl as hcl
import numpy as np
from constants import *

hcl.init(hcl.Float())

def dot_product(A,B):
  out = hcl.compute((1,), lambda x: 0, "out")
  with hcl.for_(0, NUM_FEATURES) as i:
    out[0] += A[i]*B[i]
  return out
  

def get_prediction(num):
  with hcl.if_(num > THRESHOLD):
    hcl.return_(1)
  with hcl.else_():
    hcl.return_(0)


def compute_error(param_vector, data_points, labels, num_data_points):
  true_pos = hcl.compute((1,), lambda x: 0, "true_pos")
  true_neg = hcl.compute((1,), lambda x: 0, "true_neg")
  false_pos = hcl.compute((1,), lambda x: 0, "false_pos")
  false_neg = hcl.compute((1,), lambda x: 0, "false_neg")
  error_rate = hcl.compute((1,), lambda x: 0, "errrrrrr")
  index = hcl.compute((1,), lambda x: 0, "index")
  
  #with hcl.for_(0, num_data_points) as i:
  
  with hcl.while_(index[0] < num_data_points):
    #dat = hcl.compute(param_vector.shape, lambda x: data_points[i* NUM_FEATURES + x], " trying to divide data")
    dat = hcl.compute(param_vector.shape, lambda x: data_points[index[0]* NUM_FEATURES + x], " trying to divide data")
    dot = dot_product(param_vector, dat)
    #pr = hcl.compute((1,), lambda x: get_prediction(dot[0]))
    pr = hcl.scalar(get_prediction(dot[0]))
    #with hcl.if_(pr[0] == labels[i]):
    with hcl.if_(pr.v == labels[index[0]]):
      with hcl.if_(pr.v == 1):
        true_pos[0] += 1
      with hcl.else_():
        true_neg[0] += 1
    with hcl.else_():
     with hcl.if_(pr.v == 1):
       false_pos[0] += 1
     with hcl.else_():
       false_neg[0] += 1
    index[0]+=1
  #in_d[0] = (false_pos[0]+false_neg[0]) /num_data_points
  error_rate[0] = (false_pos[0]+false_neg[0]) /num_data_points
  return error_rate
  

def get_error(theta_out, data, lab):
  out_file = open("results.txt", "w")
  
  param_vector = hcl.placeholder((NUM_FEATURES,),"param_vector for compute error")
  data_points = hcl.placeholder((DATA_SET_SIZE,),"data points")
  labels = hcl.placeholder((NUM_SAMPLES,), "labels") 
  num_data_points = hcl.placeholder((1,), "num data points")
  in_d = hcl.placeholder((1,), "error rate")
  
  t = hcl.create_schedule([param_vector, data_points, labels, num_data_points], compute_error)
  g = hcl.build(t)
  
  np_err = np.zeros((1,)) 
  in_d = hcl.asarray(np_err)
  num = np.array([NUM_TRAINING])  
  #hcl_num = hcl.asarray(num)
  hcl_num = NUM_TRAINING
  hcl_data = hcl.asarray(data)
  hcl_label = hcl.asarray(lab)
  
  g(theta_out, hcl_data, hcl_label, hcl_num, in_d)
  out_file.write("this is what the training error result is: ")
  x = in_d.asnumpy()
  np.savetxt(out_file, x)
  
  num_2 = np.array([NUM_TESTING])  
  np_err_2 = np.zeros((1,))
  in_d_2 = hcl.asarray(np_err_2)
  #hcl_num_2 = hcl.asarray(num_2)
  hcl_num_2 = NUM_TESTING
  data_2 = data[NUM_TRAINING*NUM_FEATURES:]
  label_2 = lab[NUM_TRAINING:]
  hcl_data_2 = hcl.asarray(data_2)
  hcl_label_2 = hcl.asarray(label_2)
  
  
  data_points_testing = hcl.placeholder((NUM_FEATURES*NUM_TESTING,),"data points")
  labels_testing = hcl.placeholder((NUM_TESTING,), "labels") 

  t_testing = hcl.create_schedule([param_vector, data_points_testing, labels_testing, num_data_points], compute_error)
  g_testing = hcl.build(t_testing)
  g_testing(theta_out, hcl_data_2, hcl_label_2, hcl_num_2, in_d_2)
  
  out_file.write("this is what the testing error result is: ")
  y = in_d_2.asnumpy()
  np.savetxt(out_file, y)
  
  out_file.close()
  return [x,y]

#x = execute_the_thing(in_theta_out, in_data, in_lab, in_num)#, in_d)
#
#print_theta = in_data.asnumpy()
#dot_out = x.asnumpy()
#
#print(print_theta)
#print(dot_out)