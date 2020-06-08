
with open('scripts/logs/transpose.fixedmem.log', 'r') as f:
    data = [l.strip('\n') for l in f.readlines()]

TS = 12
LINES = len(data)

#print('N, F, TFLOPS, r/w (GB), time (sec), TFLOPS/sec, GB/sec, Joules, avg Watts')
print(' \\textbf{N} & \\textbf{F} & \\textbf{TFLOPS} & \\textbf{r/w (GB)} & \\textbf{time (sec)} & \\textbf{TFLOPS/sec} & \\textbf{GB/sec} & \\textbf{Joules} & \\textbf{avg Watts} \\\\')
print(' \\hline')

for ts in range(0, LINES, 2*TS):
    N = int(data[ts].split(' ')[1])
    F = int(data[ts].split(' ')[2])
    gbs_rw = float(data[ts+4][12:].split(' ')[0])
    tflops = float(data[ts+5][12:].split(' ')[0]) / 1000
    time = float(data[ts+6][12:].split(' ')[0])
    energy = float(data[ts+7][12:].split(' ')[0])
    compute = float(data[ts+8][12:].split(' ')[0])
    thruput = float(data[ts+9][12:].split(' ')[0])
    avg_pow = float(data[ts+10][12:].split(' ')[0])
    #print('{}, {}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(
    print(' {} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\'.format(
            N,
            F,
            tflops,
            gbs_rw,
            time,
            compute,
            thruput,
            energy,
            avg_pow))
