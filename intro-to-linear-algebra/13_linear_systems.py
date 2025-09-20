import matplotlib.pyplot as plt
import numpy as np

# create x variable range (x = 2 // y = 6)
x = np.linspace(0, 10, 1000) # start, finish, n points

# y equation variables
y1 = 3 * x
y2 = 1 + (5*x)/2

# create plot
fig, ax = plt.subplots()
plt.xlabel("x")
plt.ylabel("y")
ax.set_xlim([0, 3])
ax.set_ylim([0, 8])
ax.plot(x, y1, c="green")
ax.plot(x, y2, c="pink")
plt.axvline(x=2, color="purple", linestyle="--")
_ = plt.axhline(y=6, color="purple", linestyle="--")

# create x2 variable range x = 6 // y = -1
x2 = np.linspace(-10, 10, 1000)

# y equation variables
y1_2 = -5 + (2*x2)/3
y2_2 = (7-2*x2)/5

# create plot
fig, ax = plt.subplots()
plt.xlabel("x")
plt.ylabel("y")

# create x and y axis line (0, 0)
plt.axvline(x=0, c="lightgrey")
plt.axhline(y=0, c="lightgrey")

ax.set_xlim([-2, 10])
ax.set_ylim([-6, 4])
ax.plot(x2, y1_2, c="green")
ax.plot(x2, y2_2, c="brown")
plt.axvline(x=6, color="purple", linestyle="--")
_ = plt.axhline(y=-1, color="purple", linestyle="--")

# show plot
plt.show()