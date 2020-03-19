import sys
import numpy as np


def main():
    data = sys.stdin.readlines()
    vals = [float(i) for i in data]
    print('{:.6f}'.format(np.mean(vals)))


if __name__ == '__main__':
    main()

