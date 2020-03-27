
DEBUG = False

def main():
    cases = []
    for x in range(0,2):
        for y in range(0,2):
            for z in range(0,2):
                cases.append((1-x,1-y,1-z))
   
    if DEBUG:
        print('  long cnt = 0;')
    print()
 
    c = 0
    for i,j,k in cases:
        indent='  '
        print('{}// case {} of 8: ti,tj,tk = {},{},{}'.format(indent, c+1, 'full' if i else 'partial', 'full' if j else 'partial', 'full' if k else 'partial'))
        # generate tile loops
        if i:
            print('{}for (ti=0; ti<N-TI; ti+=TI)'.format(indent))
            indent += '  '
        if k:
            print('{}for (tk=0; tk<N-TK; tk+=TK)'.format(indent))
            indent += '  '
        if j:
            print('{}for (tj=0; tj<N-TJ; tj+=TJ)'.format(indent))
            indent += '  '
        # generate point loops
        if i:
            print('{}for (i=ti; i<ti+TI; i++)'.format(indent))
        else:
            print('{}for (i=N-N%TI; i<N; i++)'.format(indent))
        indent += '  '
        
        if k:
            print('{}for (k=tk; k<tk+TK; k++) {{'.format(indent))
        else:
            print('{}for (k=N-N%TK; k<N; k++) {{'.format(indent))
        indent += '  '
        print('{}#pragma vector aligned'.format(indent))
        if j:
            print('{}for (j=tj; j<tj+TJ; j++) {{'.format(indent))
        else:
            print('{}for (j=N-N%TJ; j<N; j++) {{'.format(indent))
        indent += '  '
        if DEBUG:
            print('{}cnt+=2;'.format(indent))
        print('{}R[i*N+j] += A[i*N+k] * B[k*N+j];'.format(indent))
        print('{}}}'.format(indent[:-2]))
        print('{}}}'.format(indent[:-4]))
        if DEBUG:
            print('  printf("ti,tj,tk = {},{},{}  cnt=%ld\\n\\n", cnt);'.format('full' if i else 'partial', 'full' if j else 'partial', 'full' if k else 'partial'))
        print()
        c += 1

    if DEBUG:
        print('  printf("cnt=%ld\\n", cnt);')
    print()
    print('}')
    print()
    


if __name__ == '__main__':
    main()
