import numpy as np
import matplotlib.pyplot as plt

# Variable for days
d = np.linspace(0, 50, 1000)

# Variable for Mark I
e_1 = 1 * (d + 30)

# Variable for Mark II
e_2 = 4 * d

# Create the plot
fig, ax = plt.subplots()
plt.title("Mark I and Mark II Electricity Generated")
plt.xlabel("time (in days)")
plt.ylabel("generated (in kJ)")
ax.set_xlim([0, 20])
ax.set_ylim([0, 50])
ax.plot(d, e_1, c="green")
ax.plot(d, e_2, c="purple")
plt.axvline(x=10, color="purple", linestyle="--")
_ = plt.axhline(y=40, color="purple", linestyle="--")

# Show plot
plt.show()