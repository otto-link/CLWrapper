R""(
kernel void add_kernel(global float *A,
                       global float *B,
                       global float *C,
                       const int     n)
{
  const uint i = get_global_id(0);

  if (i >= n) return;

  C[i] = A[i] + B[i];
}

kernel void add_kernel_with_args(global float *A,
                                 global float *B,
                                 global float *C,
                                 const int     n,
                                 const float   p1,
                                 const float   p2,
                                 const int     p3)
{
  const uint i = get_global_id(0);

  if (i >= n) return;

  C[i] = A[i] + B[i] + p1 + p2 + p3;
}
)""
