import numpy as np
import matplotlib.pyplot as plt

# Variable t = time for plot
t = np.linspace(0, 40, 1000) # start, finish, n points

# Distance travelled by robber (d = 2.5t)
d_r = 2.5 * t

# Distance travelled by sherrif (d = 3(t-5))
d_s = 3 * (t - 5)

# Create the plot
fig, ax = plt.subplots()
plt.title("A Sherrif Chases The Robber")
plt.xlabel("time (in minutes)")
plt.ylabel("distance (in km)")
ax.set_xlim([0, 40])
ax.set_ylim([0, 100])
ax.plot(t, d_r, c="green")
ax.plot(t, d_s, c="purple")
plt.axvline(x=30, color="purple", linestyle="--")
_ = plt.axhline(y=75, color="purple", linestyle="--")

# Show plot
plt.show()