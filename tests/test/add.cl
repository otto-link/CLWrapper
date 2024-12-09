R""(
kernel void add_kernel(global float *A, global float *B, global float *C)
{
  const uint n = get_global_id(0);
  C[n] = A[n] + B[n];
}

kernel void add_kernel_with_args(global float *A,
                                 global float *B,
                                 global float *C,
                                 const float   p1,
                                 const float   p2,
                                 const int     p3)
{
  const uint n = get_global_id(0);
  C[n] = A[n] + B[n] + p1 + p2 + p3;
}
)""
