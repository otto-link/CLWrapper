/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <iostream>

#include "cl_wrapper.hpp"

int main()
{
  const std::string code =
#include "add.cl"
      ;

  clwrapper::KernelManager::get_instance().add_kernel(code);

  auto run = clwrapper::Run("add_kernel_with_args");

  int                n = 20;
  std::vector<float> a(n, 1.f);
  std::vector<float> b(n, 2.f);
  std::vector<float> c(n); // output

  run.bind_buffer<float>("a", a, CL_MEM_READ_ONLY);
  run.bind_buffer<float>("b", b, CL_MEM_READ_ONLY);
  run.bind_buffer<float>("c", c, CL_MEM_WRITE_ONLY);

  float p1 = 1.f;
  float p2 = 2.f;
  int   p3 = 1;
  run.bind_arguments(p1, p2, p3);

  run.write_buffer("a");
  run.write_buffer("b");

  run.execute(n);

  run.read_buffer("c");

  for (auto &v : c)
    std::cout << v << "\n";

  return 0;
}
