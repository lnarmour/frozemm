import itertools

def main():
    
    i_loop = '  for (i=0; i<N; i+={}) {{'
    j_loop = '    for (j=0; j<N; j+={}) {{'
    k_loop = '      for (k=0; k<N; k+={}) {{'

    UNROLL = {'i': [1,2,4], 'j': [1,2,4], 'k': [1,2,4]}
    idxs = [['i.i', 'i.j', 'i.k', 'j.i', 'j.j', 'j.k', 'k.i', 'k.j', 'k.k']]*3
    PERMS = ['.'.join(list(e)) for e in itertools.product(*idxs)]

    b_cnt = 0

    for shared_writes in [False, True]:
        for shared_reads in [False, True]:
            for perm in PERMS:
                if perm != 'i.k.i.j.j.k' and perm != 'i.j.i.k.k.j':  # C=AB and C=AB_transpose
                    continue
                for unroll_i in UNROLL['i']:
                    for unroll_j in UNROLL['j']:
                        for unroll_k in UNROLL['k']:
                            print('  #ifdef version{}'.format(b_cnt))
                            print('  // shared_writes({}) shared_reads({}) unroll({},{},{}) perm({})'.format(shared_writes, shared_reads, unroll_i, unroll_j, unroll_k, perm))

                            print(i_loop.format(unroll_i))
                            print(j_loop.format(unroll_j))
                            print(k_loop.format(unroll_k))

                            gen_body(shared_reads, shared_writes, 
                                     unroll_i, unroll_j, unroll_k, perm) 
                            
                            print('      }')
                            print('    }')
                            print('  }')
                            print('  #endif')
                            print()
                            b_cnt += 1



def gen_body(shared_reads, shared_writes, unroll_i, unroll_j, unroll_k, perm):
    z0, x0, y0 = 'z0', 'x0', 'y0'
    z1, x1, y1 = 'z0' if shared_writes else 'z1', 'x0' if shared_reads else 'x1', 'y1'
    l0, l1, l2, l3, l4, l5 = perm.split('.')

    for ui in range(unroll_i):
        for uj in range(unroll_j):
            for uk in range(unroll_k):
                cX = {'i': ui, 'j': uj, 'k': uk}
                c = [cX[l] for l in perm.split('.')]

                S0_LHS =  '{}[({}+{})*N+({}+{})]'.format(z0, l0, c[0], l1, c[1])
                S0_RHS0 = '{}[({}+{})*N+({}+{})]'.format(x0, l2, c[2], l3, c[3])
                S0_RHS1 = '{}[({}+{})*N+({}+{})]'.format(y0, l4, c[4], l5, c[5])
                S1_LHS = '{}[({}+{})*N+({}+{})]'.format(z1, l0, c[0], l1, c[1])
                S1_RHS0 = '{}[({}+{})*N+({}+{})]'.format(x1, l2, c[2], l3, c[3])
                S1_RHS1 = '{}[({}+{})*N+({}+{})]'.format(y1, l4, c[4], l5, c[5])

                print('        {} += {} * {};'.format(S0_LHS, S0_RHS0, S0_RHS1))
                #print('        {} += {} * {};'.format(S1_LHS, S1_RHS0, S1_RHS1))

if __name__ == '__main__':
    main()

