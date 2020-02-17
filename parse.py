import numpy as np
from matplotlib import pyplot as plt

def main():
    with open('data/8192.log', 'r') as f:
        raw = [l.strip() for l in f.readlines()][1:]

    data = [(int(l.split(':')[0].split(',')[1]),np.mean(sorted([float(v) for v in l.split(':')[1].split(',')[1:-1]]))) for l in raw]

    print(data)

    plt.plot([p[0] for p in control_data], [p[1]*10**6 for p in control_data], 'k--', label='transpose.baseline')

    plt.xlabel('TK size')
    plt.ylabel('gflops/sec')
    plt.title('8192 tiled MM calling MKL with TI=TJ=8192 for diff TK')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
