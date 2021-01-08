import os
import numpy as np
import heterocl as hcl

DIGIT_WIDTH = 64
K_CONST = 3
NUM_TRAINING = 18000
CLASS_SIZE = 1800
NUM_TEST = 2000
#DIGIT_WIDTH = 4
PAR_FACTOR = 40

def popcount_test():
    def top_func():
        
        def popcount64(number):
            count= hcl.scalar(0, "count")
            with hcl.for_(0, 64) as i:
                count.v += number[i]
            hcl.return_(count.v)
            
        def uh(number):
            x = hcl.compute((1,), lambda x: popcount64(number), "uh")
            return x
        
        number = hcl.placeholder((64,), "number")
    
        s = hcl.create_schedule([number], uh)
        return hcl.build(s)
    
    
    
    np_number = np.random.randint(2, size = (DIGIT_WIDTH,))
    np_count = hcl.cast_np(np.zeros((1,)), dtype = hcl.Int())
    
    hcl_count = hcl.asarray(np_count)
    #hcl_count = 0
    hcl_number = hcl.asarray(np_number)
    f = top_func()
    f(hcl_number, hcl_count)
    
    number = hcl_number.asnumpy()
    print(number)
    print(hcl_count)

def random_test():
    def top_func(dtype = hcl.Int()):
        
        def random(number):
            number[0] = 78
            
        number = hcl.placeholder((64,), "number")
    
        s = hcl.create_schedule([number], random)
        return hcl.build(s)
    
    
    
    np_number = np.random.randint(2, size = (64,))
    np_count = hcl.cast_np(np.zeros((1,)), dtype = hcl.Int())
    
    hcl_count = hcl.asarray(np_count)
    #hcl_count = 0
    hcl_number = hcl.asarray(np_number)
    dtype = hcl.Int()
    f = top_func(dtype)
    f(hcl_number)
    
    number = hcl_number.asnumpy()
    print(number)
    print(hcl_count)

#random_test()

def update_knn_top():
    def test_update_knn():
    
        def popcount64(number):
                count= hcl.scalar(0, "count")
                with hcl.for_(0, 64) as i:
                    count.v += number[i]
                hcl.return_(count.v)
                
        def update_knn(train_inst, test_inst, dists, labels, label):
          
            #dist = hcl.scalar(0)
            diff = hcl.compute((DIGIT_WIDTH,), lambda x: test_inst[x] ^ train_inst[x], "diff")
            dist = hcl.compute((1,), lambda x: popcount64(diff), "dist")
        
            #max_dist = hcl.compute((1,), lambda x: 0, "max_dist")
            max_dist = hcl.scalar(0)
            max_dist_id = hcl.scalar(K_CONST + 1)
            
        
            with hcl.for_(0, K_CONST) as k:
                with hcl.if_(dists[k] > max_dist.v):
                    max_dist.v = dists[k]
                    max_dist_id.v = k
                
            with hcl.if_(dist[0] < max_dist.v):
                print("wat")
    #            dists[0] = dist[0]
                dists[max_dist_id.v] = dist[0]
                labels[max_dist_id.v] = label[0]
            hcl.print(dist)
            return dist
       
        test_inst = hcl.placeholder((DIGIT_WIDTH,), "test_inst")
        train_inst = hcl.placeholder((DIGIT_WIDTH,), "train_inst")
        dists = hcl.placeholder((K_CONST,), "dists")
        labels = hcl.placeholder((K_CONST,), "labels")
        label = hcl.placeholder((1,),"label")
        
        s = hcl.create_schedule([test_inst, train_inst, dists, labels, label], update_knn)
        return hcl.build(s)
    
    g = test_update_knn()
    
    #np_test_inst = np.random.randint(2, size = (DIGIT_WIDTH,))
    #np_test_inst = np.full((DIGIT_WIDTH,), 1)
    #np_train_inst = np.random.randint(2, size = (DIGIT_WIDTH,))
#    np_dists = np.array([20, 61, 60])
#    np_labels = np.array([9, 9, 9])
#    np_label = np.array([3])
    np_test_inst = hcl.cast_np(np.zeros((DIGIT_WIDTH,)), dtype = hcl.Int())
    np_train_inst = hcl.cast_np(np.zeros((DIGIT_WIDTH,)), dtype = hcl.Int())
    np_dists = np.array([0,0,0])
    np_labels = np.array([0,0,0])
    np_label = np.array([0])
    
    np_dist = hcl.cast_np(np.zeros((1,)), dtype = hcl.Int())
    
    hcl_test_inst = hcl.asarray(np_test_inst)
    hcl_train_inst = hcl.asarray(np_train_inst)
    hcl_dists = hcl.asarray(np_dists)
    hcl_labels = hcl.asarray(np_labels)
    hcl_label = hcl.asarray(np_label)
    
    hcl_dist = hcl.asarray(np_dist)
    
    g(hcl_test_inst, hcl_train_inst, hcl_dists, hcl_labels, hcl_label, hcl_dist)
      
    test_inst = hcl_test_inst.asnumpy()
    train_inst = hcl_train_inst.asnumpy()
    dists = hcl_dists.asnumpy()
    labels = hcl_labels.asnumpy()
    label = hcl_label.asnumpy()
    dist = hcl_dist.asnumpy()
    
#    print(test_inst)
#    print(train_inst)
#    print(dists)
#    print(labels)
#    print(label)
#    print(dist)
#update_knn_top()

def knn_vote_top():
    def test_knn_vote():
    
        def knn_vote(labels, max_label):
            max_vote = hcl.scalar(0)
            #max_label = hcl.compute((1,), lambda x: 0, "max_label")
              
            votes = hcl.compute((10,), lambda x: 0, "votes")
            with hcl.for_(0, K_CONST) as i:
                votes[labels[i]] += 1
    
            with hcl.for_(0, 10) as i:
                with hcl.if_(votes[i] > max_vote.v):
                    max_vote.v = votes[i]
                    max_label[0] = i
                    
            #return max_label
                
        labels = hcl.placeholder((K_CONST,), "labels")
        max_label = hcl.placeholder((1,), "max_label")
        s = hcl.create_schedule([labels, max_label], knn_vote)
        return hcl.build(s)
        
    np_labels = np.array([2,1,2])
    np_votes = hcl.cast_np(np.zeros((10,), dtype = int), hcl.Int()) 
    np_max_label = np.array([0])
    
    hcl_labels = hcl.asarray(np_labels)
    hcl_votes = hcl.asarray(np_votes)
    hcl_max_label = hcl.asarray(np_max_label)
    hcl_label_out = 0
    
    g = test_knn_vote()
    g(hcl_labels, hcl_max_label)
    
    labels = hcl_labels.asnumpy()
    votes = hcl_votes.asnumpy()
    max_label = hcl_max_label.asnumpy()
    
    print(labels)
    print(max_label)
    print(votes)
    print(hcl_label_out)
    




A = hcl.placeholder((10,))


def top(dtype = hcl.Fixed(10,2)):

  hcl.init(dtype)
  
  def quantization(A):
      return hcl.compute(A.shape, lambda x: hcl.tanh(A[x]), "B")
  
  ##############################################################################
  # First, let's build the application without applying any quantization scheme.
  
  s = hcl.create_schedule([A], quantization)
  f = hcl.build(s)
  return f

import numpy as np

np_A = hcl.cast_np(np.zeros((10,), dtype = float), hcl.Fixed(10,2)) 

np_B = hcl.cast_np(np.zeros((10,), dtype = float), hcl.Fixed(10,2)) 

hcl_A = hcl.asarray(np_A, dtype = hcl.Fixed(10,2))
hcl_B = hcl.asarray(np_B, dtype = hcl.Fixed(10,2))

f = top(hcl.Fixed(10,2))

f(hcl_A, hcl_B)

np_A = hcl_A.asnumpy()
np_B = hcl_B.asnumpy()

print(np_A)
print('Before tanh')
print(np_A)
print('After tanh')
print(np_B)

##############################################################################
# Now let's use ``hcl.create_scheme`` to create a quantization scheme. The
# usage is the same as ``hcl.create_schedule``.

#sm = hcl.create_scheme([A], quantization)
#sm_B = quantization.B

##############################################################################
# After we create the schemes, we have two methods that can be used. First,
# if we are dealing with **integers**, we need to use ``downsize``. Second,
# if we are dealing with **floating points**, we need to use ``quantize``.
# No matter which method we choose, the first parameter is a list of tensors
# we want to quantize/downsize and the second parameter is the target data
# type.


#sm.quantize(sm_B, hcl.Fixed(10, 9))

##############################################################################
# In this example, since we know the output of `tanh` is between 1 and -1,
# we can safely set the integer part to be 2 bits (i.e., 10-8). The larger
# total bitwidth we choose, the more accurate we get. Now we can create the
# schedule by using ``hcl.create_schedule_from_scheme``, build the executable,
# and test it.

#sl = hcl.create_schedule_from_scheme(sm)
#f = hcl.build(sl)

#hcl_BQ = hcl.asarray(np.zeros(10), dtype = hcl.Fixed(10, 2))
#
#f(hcl_A, hcl_BQ)
#
#np_BQ = hcl_BQ.asnumpy()

print('Without quantization')
print(np_B)
print('Quantized to Fixed(10, 8)')
#print(np_BQ)

##############################################################################
# We can double-check this.

#assert np.array_equal(np_BQ, np.trunc(np_B*256)/256)

            