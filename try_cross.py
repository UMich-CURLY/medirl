import numpy as np
import torch 
from torch.autograd import Variable
from IPython import embed

def func_ten (x_ten,y_ten):
    full_sum = 0
    # x = x_ten.detach().numpy()
    # y = y_ten.detach().numpy()
    x = x_ten
    y = y_ten
    for i in range(x.shape[0]):
        print(i)
        a = np.exp(x[i, y[i]])
        print("a is ", a)
        sum = 0
        for j in range(x.shape[1]):
            b = np.exp(x[i, j])
            sum += b
        print("Term in fraction is ", a/sum)
        print("Sum of all terms in the denominator is ", sum)
        full_sum += -np.log(a/sum)
    return full_sum/3

def func (x,y):
    print(range(x.shape[0]))
    full_sum = 0
    for i in range(x.shape[0]):
        print(i)
        a = np.exp(x[i, y[i]])
        print("a is ", a)
        sum = 0
        for j in range(x.shape[1]):
            b = np.exp(x[i, j])
            sum += b
        print("Term in fraction is ", a/sum)
        print("Sum of all terms in the denominator is ", sum)
        full_sum += -np.log(a/sum)
    return full_sum/3
    
def cross_entropy_prob (x,y, weight):
    sum_class = 0
    for c in range(x.shape[1]):
        full_sum = 0
        for n in range(x.shape[0]):
            a = np.exp(x[n, c])
            sum = 0
            for i in range(x.shape[1]):
                b = np.exp(x[n, i])
                sum += b
            full_sum += -np.log(a/sum)*y[n, c]
        sum_class += full_sum * weight[c]
    return sum_class/x.shape[0]

def __main__():
    # x = np.array([[ 0.1639, -1.2095,  0.0496,  1.1746,  0.9474],
    #     [ 1.0429,  1.3255, -1.2967,  0.2183,  0.3562],
    #     [-0.1680,  0.2891,  1.9272,  2.2542,  0.1844]])
    # y = np.array([4,0,3])
    # print(func(x, y))
    # x_tensor = torch.tensor(x, dtype = torch.float32)
    # y_tensor = torch.tensor(y, dtype = torch.int64)
    # loss_fun = func_ten
    # loss = loss_fun(x_tensor, y_tensor)
    # loss = Variable(loss, requires_grad = True)
    # loss.backward()
    # print(func_ten(x_tensor, y_tensor))
    input = np.array([[ 0.1017, -0.2670,  1.9646, -1.7581,  0.4053],
        [ 0.2952, -2.0723,  2.4954, -0.2017, -0.9611],
        [-0.9246, -1.2307,  1.0739,  0.2575,  0.6390]])
    target = np.array([[0.1068, 0.6470, 0.0364, 0.0092, 0.2006],
        [0.1695, 0.0375, 0.4484, 0.1360, 0.2086],
        [0.0660, 0.1529, 0.6689, 0.0575, 0.0546]])
    target = np.zeros((3,5))
    target[:, 4] = 1.0
    # output = func_prob(input, target)
    # print("Array Output is ", output)
    x_tensor = torch.tensor(input, dtype = torch.float32)
    y_tensor = torch.tensor(target, dtype = torch.float32)
    out_tensor = cross_entropy_prob(x_tensor, y_tensor, weight = [0,1,0,0,0])
    print("Tensor Output is ", out_tensor)
    gt_output = 1.8743
    

__main__()
