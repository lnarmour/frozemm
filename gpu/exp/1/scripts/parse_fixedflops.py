
data = []
TS = [11, 12]

with open('scripts/logs/transpose.fixedflops.nvprof.2.log', 'r') as f:
    lines = [l.strip('\n') for l in f.readlines()]
    data.append([' '.join([c for c in s.split(' ') if c]) for s in lines])
with open('scripts/logs/transpose.fixedflops.joules.2.log', 'r') as f:
    data.append([l.strip('\n') for l in f.readlines()])
LINES = [len(data[0]), len(data[1])]

results = {}
for ts in range(0, LINES[0], TS[0]):
    N = int(data[0][ts].split(' ')[-3])
    F = int(data[0][ts].split(' ')[-2])
    S = int(data[0][ts].split(' ')[-1])
    r_thruput = float(data[0][ts+5].split(' ')[-1][:-4])
    if data[0][ts+5].split(' ')[-1][-4:-2] == 'MB':
        r_thruput /= 1000
    w_thruput = float(data[0][ts+6].split(' ')[-1][:-4])
    if data[0][ts+6].split(' ')[-1][-4:-2] == 'MB':
        w_thruput /= 1000
    r_bytes = int(data[0][ts+7].split(' ')[-1])
    w_bytes = int(data[0][ts+8].split(' ')[-1])
    flops_sp = float(data[0][ts+9].split(' ')[-1])

    key = '{}_{}_{}'.format(N,F,S)
    results[key] = [r_thruput+w_thruput, r_bytes+w_bytes, flops_sp]

for ts in range(0, LINES[1], TS[1]):
    N = int(data[1][ts].split(' ')[-3])
    F = int(data[1][ts].split(' ')[-2])
    S = int(data[1][ts].split(' ')[-1])
    time = float(data[1][ts+6][12:].split(' ')[0])
    energy = float(data[1][ts+7][12:].split(' ')[0])
    compute = float(data[1][ts+8][12:].split(' ')[0])
    avg_pow = float(data[1][ts+10][12:].split(' ')[0])

    key = '{}_{}_{}'.format(N,F,S)
    results[key] += [time, energy, compute, avg_pow]

#print('N, F, S, TFLOPS, r/w (GB), time (sec), TFLOPS/sec, GB/sec, Joules')
print(' \\textbf{N} & \\textbf{F} & \\textbf{S} & \\textbf{TF} & \\textbf{r/w (GB)} & \\textbf{time (s)} & \\textbf{TF/s} & \\textbf{GB/s} & \\textbf{Joules} \\\\')
print(' \\hline')
for r in results:
    N, F, S = [int(x) for x in r.split('_')]
    thruput = results[r][0]
    bytes_rw = results[r][1]
    flops = results[r][2]
    time = results[r][3]
    energy = results[r][4]
    compute = results[r][5]
    avg_pow = results[r][6]
    #print('{}, {}, {}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(
    print(' {} & {} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\'.format(
            N,
            F,
            S,
            flops * 10**-12 * 1000,
            bytes_rw * 10**-9 * 1000,
            time,
            compute,
            thruput,
            energy))
