import numpy as np


with open('scripts/logs/sgemm.nvprof.log', 'r') as f:
    data = [l.strip('\n') for l in f.readlines()]

TS = 8
LINES = len(data)

OUT_nvprof = []
OUT_joules = []

#print(' \\textbf{N} & \\textbf{reads (MB)} & \\textbf{writes (MB)} & \\textbf{GFLOPS} \\\\')
#print(' \\hline')
for ts in range(0, LINES, TS):
    N = int(data[ts].split(' ')[-1])
    R = float(data[ts+5].split(' ')[-1])
    W = float(data[ts+6].split(' ')[-1])
    F = float(data[ts+7].split(' ')[-1])
    
    #print(' {} & {:.2f} & {:.2f} & {:.2f} \\\\'.format(
    #    N,
    #    R * 10**-6,
    #    W * 10**-6,
    #    F * 10**-9))
    OUT_nvprof.append([N, R*10**-6, W*10**-6, F*10**-9])


with open('scripts/logs/sgemm.joules.log', 'r') as f:
    data = [l.strip('\n') for l in f.readlines()]

TS = 10
RS = 3
LINES = len(data)
for ts in range(0, LINES, TS):
    N = int(data[ts].split(' ')[-1])
    T = []
    E = []
    for rs in range(ts, ts+TS-1, RS):
        t = float(data[rs+1].split(' ')[-2])
        e = float(data[rs+2].split(' ')[-2])
        T.append(t)
        E.append(e)
    time = np.mean(T)
    energy = np.mean(E)
    #print('{} & {:.2f} & {:.2f}'.format(N, time, energy))
    OUT_joules.append([time,energy])

print(' \\textbf{N} & \\textbf{reads (MB)} & \\textbf{writes (MB)} & \\textbf{GF} & \\textbf{time (s)} & \\textbf{TF/s} & \\textbf{Joules} \\\\')
print(' \\hline')
for O1,O2 in zip(OUT_nvprof, OUT_joules):
    N, R, W, F = O1
    T, E = O2
    print(' {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\'.format(
        N,
        R,
        W,
        F,
        T,
        F/T/1000,
        E))

