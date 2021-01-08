import os
import numpy as np
import heterocl as hcl

DIGIT_WIDTH = 64
K_CONST = 3
NUM_TRAINING = 18000
CLASS_SIZE = 1800
NUM_TEST = 2000
PAR_FACTOR = 40

def top_digit_rec(dtype=hcl.Int(), target=None):

    hcl.init(dtype)
    
    def  popcount64(number):
        count = hcl.scalar(0, "count")
        with hcl.for_(0, 64) as i:
            count.v += number[i]
        hcl.return_(count.v)
      
    def update_knn(train_inst, test_inst, dists, labels, label):
        diff = hcl.compute((DIGIT_WIDTH,), lambda x: test_inst[x] ^ train_inst[x], "diff")
        dist = hcl.compute((1,), lambda x: popcount64(diff), "dist")
    
        max_dist = hcl.scalar(0)
        max_dist_id = hcl.scalar(K_CONST + 1)
    
        with hcl.for_(0, K_CONST) as k:
            with hcl.if_(dists[k] > max_dist.v):
                max_dist.v = dists[k]
                max_dist_id.v = k
            
        with hcl.if_(dist[0] < max_dist.v):
            dists[max_dist_id.v] = dist[0]
            labels[max_dist_id.v] = label[0]
            
             
        
    def knn_vote(labels, max_label):
        max_vote = hcl.scalar(0)
        
        votes = hcl.compute((10,), lambda x: 0, "votes")
        with hcl.for_(0, K_CONST) as i:
            votes[labels[i]] += 1
        
        with hcl.for_(0, 10) as i:
            with hcl.if_(votes[i] > max_vote.v):
                max_vote.v = votes[i]
                max_label[0] = i
          
          
    def get_data(data_set, i):
        data = hcl.compute((DIGIT_WIDTH,), lambda x: 0, "data")
        with hcl.for_(0, DIGIT_WIDTH) as x:
            data[i] = data_set[DIGIT_WIDTH * i + x]
        return data

    def kernel_digit_rec(training_set, test_set, result):
        #temp_result = hcl.compute((NUM_TEST,), lambda x: 0, "temp_result")
        
        with hcl.for_(0, NUM_TEST) as t:
            dists = hcl.compute((K_CONST,), lambda x: 0, "dists")
            labels = hcl.compute((K_CONST,), lambda x: 0, "labels")
            test = get_data(test_set, t)
           
            with hcl.for_(0, NUM_TRAINING) as i:
                training = get_data(training_set, i)
                label = hcl.compute((1,), lambda x: 0, "label")
                update_knn(training, test, dists, labels, label)

            max_label = hcl.compute((1,), lambda x: 0, "max_label")
            knn_vote(labels, max_label)
            result[t] = max_label[0]
            
    training_set = hcl.placeholder((NUM_TRAINING * DIGIT_WIDTH,), "training_set")
    test_set = hcl.placeholder((NUM_TEST * DIGIT_WIDTH,), "test_set")
    result = hcl.placeholder((NUM_TEST,), "result")
    
    s = hcl.create_schedule([training_set, test_set, result], kernel_digit_rec)
    return hcl.build(s, target=target)
    

        
def codegen():
    os.makedirs('device', exist_ok=True)
    targets = ['vhls', 'aocl']
    
    for target in targets:
        f = top_digit_rec(dtype=hcl.Int(), target=target)
        fp = open('device/kernel_' + target + '.cl', 'w')
        fp.write(f)
        fp.close()

def initial_test():
    Tdtype = hcl.Int()
    Rdtype = hcl.Int()
    
    np_training = hcl.cast_np(np.full((NUM_TRAINING * DIGIT_WIDTH,), 1, dtype = int), Tdtype)
    np_test_set = hcl.cast_np(np.full((NUM_TEST * DIGIT_WIDTH,), 2, dtype = int), Tdtype) 
    np_results = hcl.cast_np(np.full((NUM_TEST,), 7, dtype = int), Tdtype)
    
    hcl_training = hcl.asarray(np_training, dtype = Tdtype)
    hcl_test_set = hcl.asarray(np_test_set, dtype = Tdtype)
    hcl_results = hcl.asarray(np_results, dtype = Tdtype)
    
    dtype = hcl.Int()
    f = top_digit_rec(dtype)

    f(hcl_training, hcl_test_set, hcl_results)
    
    
def test():
    
    Tdtype = hcl.Int()
    Rdtype = hcl.Int()

    dtype = hcl.Int()
    f = top_digit_rec(dtype)

    np_results = hcl.cast_np(np.zeros((NUM_TEST,), dtype=int), Rdtype)
    hcl_results = hcl.asarray(np_results, dtype = Rdtype)
        
    np_training_0 = np.loadtxt("data/training_set_0.dat")
    np_training_1 = np.loadtxt("data/training_set_1.dat")
    np_training_2 = np.loadtxt("data/training_set_2.dat")
    np_training_3 = np.loadtxt("data/training_set_3.dat")
    np_training_4 = np.loadtxt("data/training_set_4.dat")
    np_training_5 = np.loadtxt("data/training_set_5.dat")
    np_training_6 = np.loadtxt("data/training_set_6.dat")
    np_training_7 = np.loadtxt("data/training_set_7.dat")
    np_training_8 = np.loadtxt("data/training_set_8.dat")
    np_training_9 = np.loadtxt("data/training_set_9.dat")
    
    np_training = np.concatenate((np_training_0, np_training_1, np_training_2, np_training_3, np_training_4, np_training_5, np_training_6, np_training_7, np_training_8, np_training_9), axis = 0)
    
    hcl_training_set = hcl.asarray(np_training, dtype = Tdtype)
    np_test_set = hcl.cast_np(np.loadtxt("data/test_set.dat"), Tdtype)
    
    hcl_test_set = hcl.asarray(np_test_set, dtype = Ttype)
    
    f(hcl_training_set, hcl_test_set, hcl_results)

    results = hcl_results.asnumpy()

    correct = 0

    expected = np.loadtxt("data/expected.dat")
    for i in range(NUM_TEST):
      if(results[i] == expected[i]):
        correct += 1
      else:
        print(f"should have been {expected[i]} but was {results[i]} ")
    print(f'number correct: {correct}')
      
if __name__ == "__main__":
    initial_test()
    #codegen()

      
