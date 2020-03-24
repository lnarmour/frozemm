import itertools

def main():
    
    i_loop = '  for (i=0; i<N; i+={}) {{'
    j_loop = '    for (j=0; j<N; j+={}) {{'
    k_loop = '      for (k=0; k<N; k+={}) {{'

    UNROLL = {'i': [1,2,4,8], 'j': [1,2,4,8], 'k': [1,2,4,8]}
    idxs = [['i.i', 'i.j', 'i.k', 'j.i', 'j.j', 'j.k', 'k.i', 'k.j', 'k.k']]*3
    PERMS = ['.'.join(list(e)) for e in itertools.product(*idxs)]

    #UNROLL = {'i': [1,2], 'j': [1], 'k': [1,2]}
    #PERMS = PERMS[:2]

    b_cnt = 0

    for shared_writes in [False, True]:
        for shared_reads in [False, True]:
            for perm in PERMS:
                for unroll_i in UNROLL['i']:
                    for unroll_j in UNROLL['j']:
                        for unroll_k in UNROLL['k']:
                            if perm != 'i.i.i.j.k.j' and perm != 'i.i.i.j.k.j':
                                continue

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
    z0 = 'z0'
    z1 = 'z0' if shared_writes else 'z1'
    x0 = 'x0'
    x1 = 'x0' if shared_reads else 'x1'
    y0 = 'y0'
    y1 = 'y1'

    for ui in range(unroll_i):
        for uj in range(unroll_j):
            for uk in range(unroll_k):
                for idx in ['i', 'j', 'k']:
                    # C = (l0+c0)*d1+(l1+c1)
                    # A = (l2+c2)*d1+(l3+c3)
                    # B = (l4+c4)*d1+(l5+c5)
                    #
                    # for each lX, X is either 'i', 'j', or 'k'
                    # cX goes from range(0,uX)

                    l0,l1,l2,l3,l4,l5 = perm.split('.')
                    cX = {'i': ui, 'j': uj, 'k': uk}
    
                    S0_LHS =  '{}[({}+{{}})*N+({}+{{}})]'.format(z0, l0, l1)
                    S0_RHS0 = '{}[({}+{{}})*N+({}+{{}})]'.format(x0, l2, l3)
                    S0_RHS1 = '{}[({}+{{}})*N+({}+{{}})]'.format(y0, l4, l5)
                    
                    print(idx,'>', S0_LHS)   
                    print(idx,'>', S0_RHS0)   
                    print(idx,'>', S0_RHS1)   

 
                    # S0_LHS = z0[(i+{})*N+(i+{})]

                    S0_LHS = S0_LHS.format(cX[idx] if idx==l0 or idx==l1 else 0, '{}')
                    S0_LHS = S0_LHS.format(cX[idx] if idx==l0 or idx==l1 else 0, '{}')
                    S0_RHS0 = S0_RHS0.format(cX[idx] if idx==l2 or idx==l3 else 0, '{}')
                    S0_RHS0 = S0_RHS0.format(cX[idx] if idx==l2 or idx==l3 else 0, '{}')
                    S0_RHS1 = S0_RHS1.format(cX[idx] if idx==l4 or idx==l5 else 0, '{}')
                    S0_RHS1 = S0_RHS1.format(cX[idx] if idx==l4 or idx==l5 else 0, '{}')


                    if idx in perm: 
                        print(idx, '{}{} += {} * {};'.format(' '*8, S0_LHS, S0_RHS0, S0_RHS1))


if __name__ == '__main__':
    main()
































