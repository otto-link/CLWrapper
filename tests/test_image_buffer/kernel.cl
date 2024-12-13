R""(
kernel void img_3x3_avg(read_only image2d_t  img_in,
                        write_only image2d_t img_out,
                        int                  width,
                        int                  height)
{
  const int2 g = {get_global_id(0), get_global_id(1)};

  if (g.x >= width || g.y >= height) return;

  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                            CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

  float sum = 0.f;

  sum += read_imagef(img_in, sampler, (int2)(g.y - 1, g.x - 1)).x;
  sum += read_imagef(img_in, sampler, (int2)(g.y, g.x - 1)).x;
  sum += read_imagef(img_in, sampler, (int2)(g.y + 1, g.x - 1)).x;
  sum += read_imagef(img_in, sampler, (int2)(g.y - 1, g.x)).x;
  sum += read_imagef(img_in, sampler, (int2)(g.y + 1, g.x)).x;
  sum += read_imagef(img_in, sampler, (int2)(g.y - 1, g.x + 1)).x;
  sum += read_imagef(img_in, sampler, (int2)(g.y, g.x + 1)).x;
  sum += read_imagef(img_in, sampler, (int2)(g.y + 1, g.x + 1)).x;

  sum /= 8.f;

  write_imagef(img_out, (int2)(g.x, g.y), sum);
}
)""
