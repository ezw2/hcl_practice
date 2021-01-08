"""
Getting Started
===============

**Author**: Yi-Hsiang Lai (seanlatias@github)

In this tutorial, we demonstrate the basic usage of HeteroCL.

Import HeteroCL
---------------
We usually use ``hcl`` as the acronym of HeteroCL.
"""

import heterocl as hcl
import numpy as np

##############################################################################
# Initialize the Environment
# --------------------------
# We need to initialize the environment for each HeteroCL application. We can
# do this by calling the API ``hcl.init()``. We can also set the default data
# type for every computation via this API. The default data type is **32-bit**
# integers.
#
# .. note::
#
#    For more information on the data types, please see
#    :ref:`sphx_glr_tutorials_tutorial_05_dtype.py`.

hcl.init()

##############################################################################
# Algorithm Definition
# --------------------
# After we initialize, we define the algorithm by using a Python function
# definition, where the arguments are the input tensors. The function can
# optionally return tensors as outputs. In this example, the two inputs are a
# scalar `a` and a tensor `A`, and the output is also a tensor `B`. The main
# difference between a scalar and a tensor is that *a scalar cannot be updated*.
#
# Within the algorithm definition, we use HeteroCL APIs to describe the
# operations. In this example, we use a tensor-based declarative-style
# operation ``hcl.compute``. We also show the equivalent  Python code.
#
# .. note::
#
#    For more information on the APIs, please see
#    :ref:`sphx_glr_tutorials_tutorial_03_api.py`

hcl_rand_array = hcl.placeholder((20,),"ahh")
#rand_array = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
rand_array = np.random.randint(100, size = hcl_rand_array.shape)

def simple_compute(a, A, B):

    B = hcl.compute(A.shape, lambda x, y: A[x, y] + a + B[0,0], "B")
    """
    The above API is equivalent to the following Python code.

    for x in range(0, 10):
        for y in range(0, 10):
            B[x, y] = A[x, y] + a
    """

    return B

def test_loop(C, D, E):

    def loop(x,y):
        with hcl.if_(D[x,y] < 50):
            E[x,y] = D[x,y] + C[x,y] 
        with hcl.else_():
            E[x,y] = 0
        
    hcl.mutate(E.shape, lambda x,y: loop(x,y),"huh")
    

def maximum(Ax, Bx, Cx, Dx):

    @hcl.def_([Ax.shape, Bx.shape, ()])
    def find_max(Ax, Bx, x):
        with hcl.if_(Ax[x] > Bx[x]):
            hcl.return_(Ax[x])
        with hcl.else_():
            hcl.return_(Bx[x])

    max_1 = hcl.compute(Ax.shape, lambda x: find_max(Ax, Bx, x), "max_1")
    max_2 = hcl.compute(Ax.shape, lambda x: find_max(Cx, Dx, x), "max_2")
    return hcl.compute(Ax.shape, lambda x: find_max(max_1, max_2, x), "max_o")
    
    
##############################################################################
# Inputs/Outputs Definition
# -------------------------
# One of the advantages of such *modularized algorithm definition* is that we
# can reuse the defined function with different input settings. We use
# ``hcl.placeholder`` to set the inputs, where we specify the shape, name,
# and data type. The shape must be specified and should be in the form of a
# **tuple**. If it is empty (i.e., `()`), the returned object is a *scalar*.
# Otherwise, the returned object is a *tensor*. The rest two fields are
# optional. In this example, we define a scalar input `a` and a
# two-dimensional tensor input `A`.
#
# .. note::
#
#    For more information on the interfaces, please see
#    :obj:`heterocl.placeholder`

a = hcl.placeholder((), "a")
B = hcl.placeholder((1,1),"wat")
A = hcl.placeholder((10, 10), "A")


C = hcl.placeholder((10,10), "C")
D = hcl.placeholder((10,10), "D")
E = hcl.placeholder((10,10), "E")

##############################################################################
# Apply Hardware Customization
# ----------------------------
# Usually, our next step is apply various hardware customization techniques to
# the application. In this tutorial, we skip this step which will be discussed
# in the later tutorials. However, we still need to build a default schedule
# by using ``hcl.create_schedule`` whose inputs are a list of inputs and
# the Python function that defines the algorithm.

s = hcl.create_schedule([a, A, B], simple_compute)
t = hcl.create_schedule([C,D,E],test_loop)
##############################################################################
# Inspect the Intermediate Representation (IR)
# --------------------------------------------
# A HeteroCL program will be lowered to an IR before backend code generation.
# HeteroCL provides an API for users to inspect the lowered IR. This could be
# helpful for debugging.

print(hcl.lower(s))
print(hcl.lower(t))
##############################################################################
# Create the Executable
# ---------------------
# The next step is to build the executable by using ``hcl.build``. You can
# define the target of the executable, where the default target is `llvm`.
# Namely, the executable will be run on CPU. The input for this API is the
# schedule we just created.

f = hcl.build(s)
g = hcl.build(t)
##############################################################################
# Prepare the Inputs/Outputs for the Executable
# ---------------------------------------------
# To run the generated executable, we can feed it with Numpy arrays by using
# ``hcl.asarray``. This API transforms a Numpy array to a HeteroCL container
# that is used as inputs/outputs to the executable. In this tutorial, we
# randomly generate the values for our input tensor `A`. Note that since we
# return a new tensor at the end of our algorithm, we also need to prepare
# an input array for tensor `B`.



hcl_a = 10
np_A = np.random.randint(100, size = A.shape)
np_Bin = np.random.randint(100, size = B.shape)
hcl_A = hcl.asarray(np_A)
hcl_Bin = hcl.asarray(np_Bin)
hcl_B = hcl.asarray(np.zeros(A.shape))


np_C = np.random.randint(100, size = C.shape)
np_D = np.random.randint(100, size = D.shape)
np_E = np.random.randint(100, size = E.shape)

hcl_C = hcl.asarray(np_C)
hcl_D = hcl.asarray(np_D)
hcl_E = hcl.asarray(np_E)
##############################################################################
# Run the Executable
# ------------------
# With the prepared inputs/outputs, we can finally feed them to our executable.

f(hcl_a, hcl_A, hcl_Bin, hcl_B)
g(hcl_C, hcl_D, hcl_E)

##############################################################################
# View the Results
# ----------------
# To view the results, we can transform the HeteroCL tensors back to Numpy
# arrays by using ``asnumpy()``.

np_A = hcl_A.asnumpy()
np_Bin = hcl_Bin.asnumpy()
np_B = hcl_B.asnumpy()

np_C = hcl_C.asnumpy()
np_D = hcl_D.asnumpy()
np_E = hcl_E.asnumpy()

print(hcl_a)
print(np_A)
print(np_Bin)
print(np_B)

print("rip")
print(np_C)
print(np_D)
print(np_E)
##############################################################################
# Let's run a test

assert np.array_equal(np_B, np_A + 10+np_Bin)

