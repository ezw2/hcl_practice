import heterocl as hcl
import numpy as np

def popcount(num):
  #out = hcl.compute((4,), lambda x: 0, "uhh")
  out = hcl.scalar(0, "hey dude")
  with hcl.while_(out.v < 5):
    out.v += 1
  #hcl.print(out)
  return out
  
num = hcl.placeholder((10,), "hey")

schedule = hcl.create_schedule([num], popcount)
f = hcl.build(schedule)

x = np.array([7,7,7,7,7,7,7,7,7,7])
hcl_x = hcl.asarray(x)

#o = np.array([0,0,0,0])
#hcl_o = hcl.asarray(o)

hcl_o = 0

f(hcl_x, hcl_o)

o_np = hcl_o.asnumpy()

print(o_np)