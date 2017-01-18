import numpy as np
import matplotlib.pyplot as plt
import matplotlib
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 18,
        }
matplotlib.rc('font', **font)
N = 3

l1sampleone= (0.555753808,0.566363435,0.569286801)
multisampleone=(0.51613954,0.567974165,0.59175437)
singlesense=(0.603015066,0.612033307,0.621578094)
fcse1=(0.608323725,0.619866673,0.624333575)
fcse2=(0.608722283,0.614529043,0.632379691)

ind = np.arange(N)*2.7  # the x locations for the groups
width = 0.25       # the width of the bars
interval = 0.1

fig, ax = plt.subplots()
npmssg = ax.bar(ind+2*interval, l1sampleone, width, color='#CCCC33')

sgplus = ax.bar(ind+width+3*interval, multisampleone, width, color='#80CC33')

sg = ax.bar(ind+2*width+4*interval, singlesense, width, color='#33CCCC')

fcse1 = ax.bar(ind+3*width+5*interval, fcse1, width, color='#3380CC')

fcse2 = ax.bar(ind+4*width+6*interval, fcse2, width, color='#8033CC')

# add some text for labels, title and axes ticks
ax.set_ylabel('%', fontsize='14')
ax.set_xticks(ind+3*width+3*interval)
ax.set_xticklabels( ('30%', '60%', '100%') , fontsize='20')

ax.legend( (npmssg[0], sgplus[0],sg[0],fcse1[0],fcse2[0]), ('NP-MSSG', 'SG+','Skip-gram','FCSE-1','FCSE-2'), bbox_to_anchor=(0., 1.02, 1., .102),loc=3,fontsize=13,ncol=5, mode="expand", borderaxespad=0.)


plt.xlim(0,7.5)
plt.ylim(0.5,0.7)
plt.tight_layout()
plt.show()
