import os
import numpy as np

RESULTS_FILE = os.getenv('RESULTS')

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

    for pt in data_pts:
        pt['avg_time'] = np.mean(pt['times'])
        #print(pt)

    PIs = sorted(list(set([d['PI'] for d in data_pts])))
    PJs = sorted(list(set([d['PJ'] for d in data_pts])))
    TKs = sorted(list(set([d['TK'] for d in data_pts])))

    if not data_pts:
        return

    N = data_pts[0]['N']

    print(',,{}'.format(','.join([str(tk) for tk in TKs])))
    for pi in PIs:
        if pi == N:
            continue
        for pj in PJs:
            if pj == N:
                continue
            pts = [p for p in data_pts if p['PI']==pi and p['PJ']==pj]    
            print('{},{},,'.format(pi, pj), end='')
            for p in pts:
                print('{:.6f},'.format(p['avg_time']), end='')
            print(',,', end='')
            for p in pts:
                print('{},'.format(p['OCA']), end='')
            print()
        print()


if __name__ == '__main__':
    main()
