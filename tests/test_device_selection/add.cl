R""(
kernel void add_kernel(global float *A, global float *B, global float *C)
{
  const uint n = get_global_id(0);
  C[n] = A[n] + B[n];
}
)""
