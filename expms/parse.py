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
            data_pts.append({'N':N, 'PI':int(PI), 'PJ':int(PJ), 'TK':int(TK), 'pkg':None, 'ram':None, 'times':[]})
            continue
        if len(line) == 8:
            data_pts[-1]['times'].append(float(line))
            continue
        if 'energy-pkg' in line:
            pkg = line.split('Joules')[0].replace(' ', '').replace(',', '')
            data_pts[-1]['pkg'] = float(pkg)
        if 'energy-ram' in line:
            ram = line.split('Joules')[0].replace(' ', '').replace(',', '')
            data_pts[-1]['ram'] = float(ram)

    N = data_pts[0]['N']
    for pt in data_pts:
        if pt['PI']==N and pt['PJ']==N and pt['TK']==N and pt['N']==N:
            base = pt
            if pt['pkg']:
                base_pkg = pt['pkg']
            if pt['ram']:
                base_ram = pt['ram']
        pt['avg_time'] = np.mean(pt['times'])
    base_time = base['avg_time']
    for pt in data_pts:
        pt['norm_time'] = pt['avg_time'] / base_time
        if pt['pkg']:
            pt['norm_pkg'] = pt['pkg'] / base_pkg
        if pt['ram']:
            pt['norm_ram'] = pt['ram'] / base_ram

    if not data_pts:
        return

    PIs = [p for p in sorted(list(set([d['PI'] for d in data_pts]))) if p != N] 
    PJs = [p for p in sorted(list(set([d['PJ'] for d in data_pts]))) if p != N] 
    TKs = [p for p in sorted(list(set([d['TK'] for d in data_pts]))) if p != N] 

    print('base')
    print('N,{}'.format(N))
    print('time,{:.6f}'.format(base_time))
    print('pkg,{}'.format(base_pkg))
    print('ram,{}'.format(base_ram))
    print()

    tks_str = ',,{},'.format(','.join([str(tk) for tk in TKs]))
    print('{}{}{}'.format(tks_str, tks_str, tks_str))
    print()
    print()
    for pi in PIs:
        pts_pkg = [p for p in data_pts if p['PI']==pi and p['pkg']]
        pts_ram = [p for p in data_pts if p['PI']==pi and p['ram']]

        print('{},,'.format(pi), end='')
        for p in pts_pkg:
            print('{:.6f},'.format(p['avg_time']), end='')
        print(',,', end='')
        for p in pts_pkg:
            print('{},'.format(p['pkg']), end='')
        print(',,', end='')
        for p in pts_ram:
            print('{},'.format(p['ram']), end='')
        print()
    print()
    print()

    for pi in PIs:
        pts_pkg = [p for p in data_pts if p['PI']==pi and p['pkg']]
        pts_ram = [p for p in data_pts if p['PI']==pi and p['ram']]

        print('{},,'.format(pi), end='')
        for p in pts_pkg:
            print('{:.6f},'.format(p['norm_time']), end='')
        print(',,', end='')
        for p in pts_pkg:
            print('{:.6f},'.format(p['norm_pkg']), end='')
        print(',,', end='')
        for p in pts_ram:
            print('{:.6f},'.format(p['norm_ram']), end='')
        print()


if __name__ == '__main__':
    main()
