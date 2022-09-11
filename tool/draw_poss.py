import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.pyplot import MultipleLocator


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 12})
# plt.rc('font',family='Times New Roman', size=12)

# plt.title('title name')  # Title
fig = plt.figure(figsize=(8, 6), dpi=300)
plt.xlabel('Distance to sensor [m]') # x value
plt.ylabel('Mean IoU [%]') # y value
plt.grid(True)

dist = [1, 2, 3, 4, 5, 6, 7, 8, 9]

Polar_Balance_Random = [62.92, 59.77, 56.52, 48.89, 38.54, 34.00, 27.12, 21.41, 12.35]
Random = [59.58, 60.78, 54.43, 43.71, 32.88, 27.05, 22.14, 18.39, 11.97]

plt.plot(dist, Polar_Balance_Random, color='r', linestyle='-', marker='o', label='Polar Cylinder Balanced Random Sampling')
plt.plot(dist, Random, color='b', linestyle='-', marker='s', label='Random Sampling')

x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
labels = ['(0, 10]', '(10, 20]', '(20, 30]', '(30, 40]', '(40, 50]', '(50, 60]', '(60, 70]', '(70, 80]', '(80, max)']
plt.xticks(x, labels)

# plt.xlim(0, 7)
plt.ylim(top=64)

# plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(10))

plt.legend()

# **************************************************************************************************************************
fig.set_size_inches(8, 6, forward=True)
fig.tight_layout()

plt.savefig("distance_poss.png") #, bbox_inches='tight'
plt.show()
plt.close()
