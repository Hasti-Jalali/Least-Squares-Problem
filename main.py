import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SIZE = 318 #335 * 95% = 318.25

def multiply_matrix(A, B) :
    result = [] #A*B
   # iterating by row of A
    for i in range(len(A)):
        res = []
        # iterating by column by B
        for j in range(len(B[0])):
            data = 0
            # iterating by rows of B
            for k in range(len(B)):
                data += A[i][k] * B[k][j]
            res.append(data)
        result.append(res)
    
    return result
    

data = pd.read_csv('./covid_cases.csv')
y = data.loc[:,'World']
SIZE = int(y.size * 0.95)
y_tst = np.array([y[:SIZE]]).transpose().tolist()
A1 = [[1, i] for i in range(SIZE)] #2D A
A1T = np.array(A1).transpose().tolist() #A1 Transpose
A1T_A1 = multiply_matrix(A1T, A1) 
inv_A1T_A1 = np.linalg.inv(A1T_A1) #(A^T * A) ^ -1
A1T_Y = multiply_matrix(A1T, y_tst) 

X1 = multiply_matrix(inv_A1T_A1, A1T_Y) 
B1_0 = X1[0]
B1_1 = X1[1]


# print(X1)

A2 = [[1,i, i**2] for i in range(SIZE)] #3D A
A2T = np.array(A2).transpose().tolist() #A2 Transpose
A2T_A2 = multiply_matrix(A2T, A2)
inv_A2T_A2 = np.linalg.inv(A2T_A2) #(A^T * A) ^ -1
A2T_Y = multiply_matrix(A2T, y_tst) 

X2 = multiply_matrix(inv_A2T_A2, A2T_Y) 
B2_0 = X2[0]
B2_1 = X2[1]
B2_2 = X2[2]

# PLOT 
SIZE = round(SIZE / 0.95)
x1 = np.linspace(0,SIZE,SIZE)
y1 = B1_0 + x1 * B1_1


x2 = np.linspace(0,SIZE,SIZE)
y2 = B2_0 + x2 * B2_1 + x2 * x2 * B2_2

plt.plot(x1, y, color='r')
plt.plot(x1,y1, color='b')
plt.plot(x2,y2, color='g')

# print(A1T_A1)
# print(A2T_A2)
# y.plot()

# print(data)

plt.show()

# TESTING : X = 20, 46, 110, 192, 274
def calc_y1(x) : 
    return B1_0[0] + x * B1_1[0]
def calc_y2(x) :
    return B2_0[0] + x * B2_1[0] + x * x * B2_2[0]

xTestList = [20, 46, 110, 192, 274]
for i in range(5) : 
    t_x = xTestList[i]
    print('x =', t_x, 'y =', y[t_x])
    t_y1 = calc_y1(t_x)
    error = y[t_x] - calc_y1(t_x)
    print('2D : ' , "y' =", t_y1, 'error =', error)
    error = y[t_x] - calc_y2(t_x)
    t_y2 = calc_y2(t_x)
    print('3D : ' , "y' =", t_y2, 'error =', error, '\n')
