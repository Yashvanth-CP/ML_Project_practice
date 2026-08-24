import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import copy  
import math  

# Add these at top
x_train = np.array([[12, 34, 65, 56, 67],
                    [23, 34, 56, 67, 89],
                    [34, 45, 67, 67, 78]])

y_train = np.array([11, 21, 31])

print(f"x_shape : {x_train.shape}, x type : {type(x_train)}")
print(x_train)

print(f"Y_shape : {y_train.shape}, y type : {type(y_train)}")
print(y_train)

b_init = 100.000000000000001
w_init = np.array([0.98765432, 1.1234567890, 2.456789987, 2.5678987,  1.2345678])

print(f"w_init shape : {w_init.shape}, ")

def predict_single_loop(x, w, b): 
    n = x.shape[0]
    p = 0
    for i in range(n):
        p_i = x[i] * w[i]

        p += p_i
    p = p + b

    return p

# getting row from training data 
x_vec = x_train[0, : ]
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")


#make a prediction 

f_wb = predict_single_loop(x_vec, w_init, b_init)
print(f"f_wb shape {f_wb.shape}, prection: {f_wb}")

def prediction(x, w, b):
    p = np.dot(x, w) + b
    return p


# get a row from our training data
x_vec = x_train[0,:]
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")

# make a prediction
f_wb = prediction(x_vec,w_init, b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")

def compute_cost(x, y, w, b):

    m = x.shape[0]
    cost = 0.0

    for i in range(m):
        f_wb_i = np.dot(x[i],w) + b
        cost = cost + (f_wb_i - y[i]) ** 2

    cost = cost/ (2*m)

    return cost

cost = compute_cost(x_train, y_train, w_init, b_init)
print(f"the cpost at optimal w: {cost}")

def compute_gradients(x, y, w, b):
    m,n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        err = (np.dot(x[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * x[i, j]
        dj_db = dj_db + err
    dj_db = dj_db/m
    dj_dw = dj_dw/m

    return dj_db, dj_dw

# Compute and Display gradient 

temp_dj_db, temp_dj_dw = compute_gradients(x_train, y_train, w_init, b_init)
print(f" dj_db at intial w,b : {temp_dj_db}")
print(f" dj_dw at intial w,b : {temp_dj_dw}")

def gradient_descent(X, y,w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    J_history = [] # this array to store the cost j and w's at each iteration
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w, b) # calculate the gradient 

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if (i < 100000):
            J_history.append( cost_function(X, y, w, b))

        if i% math.ceil(num_iters / 10) == 0:
            print(f" Iteration {i:40} : Costv{J_history[-1]:8.2f}")

    return w, b, J_history


# initialize parameters
initial_w = np.zeros_like(w_init)
initial_b = 0.
# some gradient descent settings
iterations = 1000
alpha = 5.0e-7

# run gradient descent 
w_final, b_final, J_hist = gradient_descent(x_train, y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradients, 
                                                    alpha, iterations)


print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = x_train.shape
for i in range(m):
    print(f"prediction: {np.dot(x_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")


# plot cost versus iteration  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
plt.show()


    



