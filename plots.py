import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#path = 'logdir/2022-10-03_09-45-46/small_instance'
path = 'logdir/ibmq_kolkata/small_instance'

data = dict()
for f in os.listdir(path):
    f_path = path + '/' + f
    df = pd.read_csv(f_path, dtype={'binary': str, 'prob_squared': float, 'energy': int})
    df['prob'] = df['prob_squared']**2
    p = f.split('_')[1]
    if p in data:
        data[p].append(df)
    else:
        data[p] = [df]

data_to_plot = []
for p, l in data.items():
    probs = None
    for i, df in enumerate(l):
        p = df.groupby('energy').agg({'prob': np.sum})
        if probs is None:
            probs = p.rename(columns={'prob': i}).transpose()
        else:
            probs = probs.append(p.rename(columns={'prob': i}).transpose())
    probs = probs.agg([np.mean, np.std])
    data_to_plot.append(probs.transpose())

fig, axes = plt.subplots(1, len(data_to_plot), figsize=(20,5))
plt.tight_layout(w_pad=0, h_pad=0, rect=[0.04, 0.12, 1, 0.95])

prob_maximum = max([max(x['mean']) for x in data_to_plot]) + 0.01

for i, (dtp, ax) in enumerate(zip(data_to_plot, axes)):

    x = dtp['mean'].loc[dtp.index < 9].to_list()\
        + [sum(dtp['mean'].loc[dtp.index >= 9])]

    y = np.arange(9).astype(str).tolist() + ['$9+$']

    ax.bar(y, x, color='darkgreen', align='center', width=0.7)
    ax.set_ylim(0, prob_maximum)
    labels = np.concatenate([[x] + [''] + [''] for x in y[::3]])

    ax.set_xticklabels(labels)
    ax.yaxis.grid()

    #ax.xaxis.grid(True, which='minor')

    ax.tick_params(axis='y', which='minor')
    ax.yaxis.set_ticks(np.arange(0, max(x), 0.01), minor=True)

    if i == 2:
        ax.set_xlabel('Energy', fontsize=24, labelpad=15)
    if i == 0:
        ax.set_ylabel('Probability', fontsize=24, labelpad=15)
    else:
        ax.yaxis.set_ticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.set_title('$p={}$'.format(i+1), fontsize=20)
plt.savefig('kolkata.pdf')
#plt.show()
a = 1
