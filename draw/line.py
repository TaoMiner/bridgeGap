import numpy as np
import matplotlib.pyplot as plt
import matplotlib
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 14,
        }
matplotlib.rc('font', **font)

l1=(0.539350,0.445487,0.418773,0.415884,0.410830,0.410830)
l1x=(0.2427,0.3876,0.4410,0.4863,0.5338,0.5553)


plt.plot(l1x, l1, '-k8',label='Accuracy',linewidth=2)

plt.ylabel('Word Accuracy')
plt.xlabel('Entity Relatedness')

plt.xlim(0.22,0.57)
plt.ylim(0.38,0.57)

plt.tight_layout()
plt.show()