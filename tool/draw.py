import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.pyplot import MultipleLocator


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 15})
# plt.rc('font',family='Times New Roman', size=12)

# plt.title('title name')  # Title
fig = plt.figure(figsize=(8, 6), dpi=300)
plt.xlabel('Distance to sensor [m]') # x value
plt.ylabel('Mean IoU [%]') # y value
plt.grid(True)

dist = [1, 2, 3, 4, 5, 6]

# Polar_Balance_Random = [56.66, 51.62, 44.07, 32.34, 24.54, 10.16]
Random = [58.73, 52.51, 44.83, 32.10, 23.61, 9.37]

# plt.plot(dist, Polar_Balance_Random, color='r', linestyle='-', marker='o', label='Polar Random Sampling')
plt.plot(dist, Random, color='b', linestyle='-', marker='s', label='Random Sampling')

x = [1, 2, 3, 4, 5, 6]
labels = ['(0, 10]', '(10, 20]', '(20, 30]', '(30, 40]', '(40, 50]', '(50, max)']
plt.xticks(x, labels)

# plt.xlim(0, 7)
plt.ylim(top=60)

# plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(10))

plt.legend()

# **************************************************************************************************************************
fig.set_size_inches(8, 6, forward=True)
fig.tight_layout()

plt.savefig("distance.png") #, bbox_inches='tight'
plt.show()
plt.close()
