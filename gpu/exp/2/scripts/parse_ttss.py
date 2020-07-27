import numpy as np
import os
from subprocess import Popen
from subprocess import PIPE

def main():
    ##  cmd = "paste <(cat scripts/zzold_logs/ttss.nvprof.log | grep 'lnvprof' | sed 's~.*-ttss \\(.*\\)~\\1~') <(cat scripts/zzold_logs/ttss.nvprof.log | grep '\\--> R' | sed 's~.*: \\(.*\\)~\\1~' ) <(cat scripts/zzold_logs/ttss.nvprof.log | grep '\\--> W'| sed 's~.*: \\(.*\\)~\\1~') | sed 's~   ~ ~g' | sed 's~	~ ~g'\n"
    ##  with open('.tmpcmd.sh','w') as f:
    ##      f.write(cmd)
    ##  
    ##  cmd_pipe = Popen(['bash','.tmpcmd.sh'], stdout=PIPE)
    ##  blob = cmd_pipe.stdout.read()
    ##  lines_nvprof = [l.decode("utf-8") for l in blob.split(b'\n') if l]
    ##  
    ##  VALS = {}
    ##  for l in lines_nvprof:
    ##      key = '_'.join(l.split(' ')[:4])
    ##      VALS[key] = [float(v)*32*10**-9 for v in l.split(' ')[4:]]  # GBs
    
    # ./bin/sgemm-ttss 10000 5000 5000 1000
    # ops:        2000.00 GFLOPs
    # time:       0.20466 sec
    # energy:     49.09 Joules
    # compute:    9.77 TFLOPs/sec
    # avg power:  240.65 W
    # ./bin/sgemm-ttss 10000 5000 5000 1000
    # ops:        2000.00 GFLOPs
    # time:       0.20447 sec
    # energy:     49.71 Joules
    # compute:    9.78 TFLOPs/sec
    # avg power:  243.70 W
    # ./bin/sgemm-ttss 10000 5000 5000 1000
    # ops:        2000.00 GFLOPs
    # time:       0.20446 sec
    # energy:     47.20 Joules
    # compute:    9.78 TFLOPs/sec
    # avg power:  243.29 W
    #

    VALS = {}
    LOG_FILE = os.getenv('LOG_FILE')
    with open(LOG_FILE, 'r') as f:
        data = [l.strip('\n') for l in f.readlines()]
    TS = 19
    LINES = len(data)
    lines_joules = []
    for ts in range(0,LINES,TS):
        nppk = '_'.join(data[ts].split(' ')[-4:])
        times = 0
        energys = 0
        for r in range(3):
            times += float(data[ts+r*6+2][12:].split(' ')[0])
            energys += float(data[ts+r*6+3][12:].split(' ')[0])
        VALS[nppk] = [times/3, energys/3]
   
    keys = [[int(v) for v in k.split('_')] for k in VALS]
    Ns = sorted(list(set([k[0] for k in keys])))
    Ps = sorted(list(set([k[1] for k in keys if k[1] not in Ns])))
    TKs = sorted(list(set([k[3] for k in keys if k[3] not in Ns])))

    VALS = [[int(v) for v in k.split('_')]+VALS[k] for k in VALS]

    def print_metric(ms, name):
        print(',{}'.format(name))
        print(',,raw' + ','*len(TKs) + ',' + 'normalized' + ','*len(TKs))
        print(',,TK' + ','*len(TKs) + ',' + 'TK' + ','*len(TKs))
        tk_str = ','.join([str(tk) for tk in TKs])
        print(',,' + tk_str + ',,' + tk_str + ',')
        baseline = [v for v in VALS if v[0]==v[1]==v[2]==v[3]==Ns[0]][0]
        baseline_val = sum([baseline[m] for m in ms])
        for i,P in enumerate(Ps):
            pVALS = [v for v in VALS if v[1]==P]
            print('{},{},'.format('P' if i==0 else '',P),end='')
            for v in pVALS:
                print('{:.3f},'.format(sum([v[m] for m in ms])), end='')
            print(',', end='')
            for v in pVALS:
                print('{:.3f},'.format(sum([v[m] for m in ms]) / baseline_val), end='')
            print()
        print()

    # baseline
    b = [v for v in VALS if v[0]==v[1]==v[2]==v[3]==Ns[0]][0]
    print('baseline')
    print('N,{}'.format(b[0]))
    #print('r (GBs),{:.3f}'.format(b[4]))
    #print('w (GBs),{:.3f}'.format(b[5]))
    #print('r/w (GBs),{:.3f}'.format(b[4]+b[5]))
    print('time (sec),{:.3f}'.format(b[4]))
    print('energy (joules),{:.3f}'.format(b[5]))
    print()
    print()

    #print_metric([4], 'DRAM reads (GBs)')
    #print_metric([5], 'DRAM writes (GBs)')
    #print_metric([4,5], 'DRAM reads & writes (GBs)')
    print_metric([4], 'time (sec)')
    print_metric([5], 'energy (joules)')

 
    return VALS
    
if __name__ == '__main__':
    main()
