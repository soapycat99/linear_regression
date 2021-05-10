import matplotlib.pyplot as plt
import numpy as np
import matplotlib

#random data
b = [2,5,7,9,11,16,19,23,22,29,29,35,37,40,46,42,39,31,30,28,20,15,10,6]
A = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]

#visualize
plt.plot(A,b,"ro")

#change row vector to column vector
A = np.array([A]).T
b = np.array([b]).T

#Create A Square
x_square = np.array([A[:,0]**2]).T #create x_square with a matrix of the first row of A power two
A = np.concatenate((x_square, A), axis = 1)
print(x_square)
#vector 1
ones = np.ones((A.shape[0],1), dtype = np.int8)

#Combine 1 vs A
A = np.concatenate((A, ones),axis = 1)



#Formula to calculate x
x = np.linalg.inv(A.transpose().dot(A)).dot(A.transpose()).dot(b)
# print(x)

#Test data to draw
x0 = np.linspace(2,25, 10000)
y0 = x[0][0]*x0*x0 + x[1][0]*x0 +x[2][0]

# print(y0)

plt.plot( x0, y0)

x_test = 12
y_test = x_test*x[0][0] + x[1][0]

plt.show()


# x = (A^T * A) ^ -1 * A^T * y
