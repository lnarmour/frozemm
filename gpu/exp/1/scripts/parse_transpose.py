
with open('scripts/logs/transpose.N.FMAS.log', 'r') as f:
    data = [l.strip('\n') for l in f.readlines()]

TS = 13
LINES = len(data)

print('N,F,time,energy,tflops,thruput,avg_pow')

for ts in range(0, LINES, TS):
    N = data[ts].split(' ')[1]
    F = data[ts].split(' ')[2]
    time = data[ts+7][12:].split(' ')[0]
    energy = data[ts+8][12:].split(' ')[0]
    tflops = data[ts+9][12:].split(' ')[0]
    thruput = data[ts+10][12:].split(' ')[0]
    avg_pow = data[ts+11][12:].split(' ')[0]
    print('{},{},{},{},{},{},{}'.format(N,F,time,energy,tflops,thruput,avg_pow))

