import os
import numpy as np

RESULTS_FILE = os.getenv('RESULTS') if os.getenv('RESULTS') else 'results.log'

def main():
    if not RESULTS_FILE:
        return
    with open(RESULTS_FILE, 'r') as f:
        lines = [l.strip('\n') for l in f.readlines()]

    data_pts = []

    for line in lines:
        if 'PI' in line:
            N = int(line.split(' ')[1])
            PI,PJ,TK = (l.split('=')[-1] for l in line[line.index('(')+1:line.index(')')].split(' '))
            data_pts.append({'N':N, 'PI':int(PI), 'PJ':int(PJ), 'TK':int(TK), 'OCA':None, 'times':[]})
            continue
        if len(line) == 8:
            data_pts[-1]['times'].append(float(line))
            continue
        if 'L3_MISS' in line:
            OCA = line.split('ANY_REQ')[0].replace(' ', '').replace(',', '')
            data_pts[-1]['OCA'] = int(OCA)

    N = data_pts[0]['N']
    for pt in data_pts:
        if pt['PI']==N and pt['PJ']==N and pt['TK']==N and pt['N']==N:
            base = pt
        pt['avg_time'] = np.mean(pt['times'])
    base_time = base['avg_time']
    base_OCA = base['OCA']
    for pt in data_pts:
        pt['norm_time'] = pt['avg_time'] / base_time
        pt['norm_OCA'] = pt['OCA'] / base_OCA

    if not data_pts:
        return

    PIs = [p for p in sorted(list(set([d['PI'] for d in data_pts]))) if p != N] 
    PJs = [p for p in sorted(list(set([d['PJ'] for d in data_pts]))) if p != N] 
    TKs = [p for p in sorted(list(set([d['TK'] for d in data_pts]))) if p != N] 

    print('base')
    print('N,{}'.format(N))
    print('time,{:.6f}'.format(base_time))
    print('OCA,{}'.format(base_OCA))
    print()

    tks_str = ',,,{}'.format(','.join([str(tk) for tk in TKs]))
    print('{}{}{}{}'.format(tks_str, tks_str, tks_str, tks_str))
    print()
    for pi in PIs:
        for pj in PJs:
            pts = [p for p in data_pts if p['PI']==pi and p['PJ']==pj]    
            print('{},{},,'.format(pi, pj), end='')
            for p in pts:
                print('{:.6f},'.format(p['avg_time']), end='')
            print(',,', end='')
            for p in pts:
                print('{},'.format(p['OCA']), end='')
            print(',,', end='')
            for p in pts:
                print('{:.6f},'.format(p['norm_time']), end='')
            print(',,', end='')
            for p in pts:
                print('{:.6f},'.format(p['norm_OCA']), end='')
            print()
        print()


if __name__ == '__main__':
    main()
