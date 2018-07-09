import numpy as np
import matplotlib
import matplotlib.pyplot as plt

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 18,
        }
matplotlib.rc('font', **font)


micro=(0.819749, 0.840921, 0.855900, 0.884728, 0.852636, 0.802678, 0.788201, 0.777782, 0.772343, 0.768117, 0.764519, 0.760042, 0.757699, 0.756067)
macro = (0.816476, 0.848821, 0.864833, 0.889410, 0.860570, 0.823873, 0.812380, 0.804540, 0.800219, 0.796091, 0.793383, 0.789956, 0.787777, 0.786303)
lx = (0, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


plt.plot(lx, micro, '--b8',label='Micro Accuracy',linewidth=2)
plt.plot(lx, macro, '-k8',label='Macro Accuracy',linewidth=2)

plt.ylabel('Accuracy')
plt.xlabel('Smoothing Parameter $\gamma$')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()