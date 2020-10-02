import os
N = int(os.getenv('N'))
t = float(os.getenv('time'))
print('{}'.format(int(100*(4*N**3*10**-9 / t / 115.6))))
