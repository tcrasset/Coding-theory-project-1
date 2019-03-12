
import math
import numpy as np


# H(X|Y)
x_and_y = np.array(
    [[1/8, 1/16, 1/16, 1/4],
    [1/16, 1/8, 1/16, 0],
    [1/32, 1/32, 1/16, 0],
    [1/32, 1/32, 1/16, 0]]
)

p_y = [0.5, 0.25, 0.125, 0.125]
s = 0
for j in range(4):
    for i in range(4):
        if(x_and_y[j][i] != 0 and p_y[j] != 0):
            s += x_and_y[j][i] * math.log(x_and_y[j][i]/p_y[j],2)

print("H(X|Y): ",-s)


# H(W|X)

p_x = 0.25
s = 0
three = 0
one = 0
for j in range(4):
    three += 3/16* math.log((3/16)/p_x,2)
    one += 1/16* math.log((1/16)/p_x,2)


s = - one - three
print("H(W|X): ",s)


# H(Z|W)

print("H(Z|W): ", - (math.log(0.25/0.25,2) + math.log(0.75/0.75,2)))


# H(Y|Z)

p_z0 = 0.25
p_z1 = 0.75
p_yz0 = [1/8, 1/16,1/32, 1/32]
p_yz1 = [3/8, 3/16,3/32, 3/32]
s = 0
zero = 0
one = 0
for j in range(4):
    zero += p_yz0[j]* math.log(p_yz0[j]/p_z0,2)
    one += p_yz1[j]* math.log(p_yz1[j]/p_z1,2)


s = - one - zero
print("H(Y|Z): ",s)


# H(X,Y|W)
p_xyw0 = [0.0625, 0.03125, 0.03125, 0.0625, 0.03125, 0.03125, 0.0625, 0.0625, 0.0625, 0.25]
p_xyw1 = [0.125, 0.125, 0.0625]

p_w0 = 0.75
p_w1 = 0.25

H_xyw0 = 0
H_xyw1 = 0

for p in p_xyw0:
    H_xyw0 += p * math.log(p/p_w0,2)

for p in p_xyw1:
    H_xyw1 += p * math.log(p/p_w1,2)

print("H(X,Y|W):", -H_xyw0 - H_xyw1)


# H(W,Z|X)

p_wzx= [1/16, 1/16, 1/16, 1/16, 3/16, 3/16, 3/16, 3/16]

H_wzx = 0
for p in p_wzx:
    H_wzx -= p * math.log(p/0.25,2)

print("H(W,Z|X): ", H_wzx)
