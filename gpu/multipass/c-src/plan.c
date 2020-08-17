//
// THIS IS ALL PSEUDO-CODE
//


Option 1:

Ideally, you could just do this to represent the pass [2,2,2].  Assume that the
input problem size and BENCH_RAD is such that these are not boundary cases and this
represents a full pass:

#define PI 100
#define PJ 100
#define PK 100
int pi = 2;
int pj = 2;
int pk = 2;

#pragma scop 
for (int t = 0; t < timestep; t++) 
  for (int i = pi*PI-BENCH_RAD*t; i < (pi+1)*PI-BENCH_RAD*t; i++)
    for (int j = pj*PJ-BENCH_RAD*t; j < (pj+1)*PJ-BENCH_RAD*t; j++)
      for (int k = pk*PK-BENCH_RAD*t; k < (pk+1)*PK-BENCH_RAD*t; k++)
      { 
        A[(t+1)%2][i][j][k] = ...
      }   
#pragma endscop

But AN5D generates incorrect code when this is given as input.






Option 2:

Organize the input C code so that it each spatial iterator traverses a patch
  
#define PI 100
#define PJ 100
#define PK 100
int pi = 2;
int pj = 2;
int pk = 2;
  
#pragma scop
for (int t = 0; t < timestep; t++) 
  for (int i = BENCH_RAD; i < BENCH_RAD + PI; i++)
    for (int j = BENCH_RAD; j < BENCH_RAD + PJ; j++)
      for (int k = BENCH_RAD; k < BENCH_RAD + PK; k++)
      {
        A[(t+1)%2][i][j][k] = ...
      }   
#pragma endscop

AN5D can correctly handle the scop section and generate correct CUDA code

The generated host code has the following structure:

{ cudaMalloc dev_A }
{ cudaMemCopy host A -> device dev_A }
{ timestep/BT number of kernel calls on dev_A }
{ cudaMemCopy device dev_A -> host A }


I want to change the host code to something like this:

{ cudaMalloc dev_A }
{ cudaMemCopy host A -> device dev_A }
{
  timestep = BT;
  for (int bt=0; bt<timestep; bt+=BT)
    dev_A_patch = &(dev_A[f(bt,pi,pj,pk)])
    { one kernel call on dev_A_patch }
}
{ cudaMemCopy device dev_A -> host A }
















}
