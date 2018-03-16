import matplotlib.pyplot as plt
import numpy as np

x1 = np.array([1, 1, 2, 1.5, 3, 3.5])
x2 = np.array([4, 3, 2, 4, 1, 2])
z = 17 - ((x1-1.5)**2+(x2-4)**2)**0.5

fig, ax = plt.subplots()
cm = ax.scatter(x1, x2, c="r") #, c=y)

for i, txt in enumerate(z):
    ax.annotate("{:0.1f}".format(txt), (x1[i]+0.05,x2[i]))
    
ax.scatter(4.5,4.5, c="k", marker="*", s=100)


plt.ylim((0,5))
plt.xlim((0,5))

ax.set_ylabel("X2")
ax.set_xlabel("X1")
ax.set_title("Example")

fig.set_size_inches((10,10))
plt.show()
