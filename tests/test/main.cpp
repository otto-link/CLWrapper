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

  int                n = 11;
  std::vector<float> a(n, 1.f);
  std::vector<float> b(n, 2.f);
  std::vector<float> c(n); // output

  run.bind_buffer<float>("a", a);
  run.bind_buffer<float>("b", b);
  run.bind_buffer<float>("c", c);

  float p1 = 1.f;
  float p2 = 2.f;
  int   p3 = 1;
  run.bind_arguments(n, p1, p2, p3);

  run.write_buffer("a");
  run.write_buffer("b");

  run.execute(n);

  run.read_buffer("c");

  for (size_t k = 0; k < c.size(); ++k)
    std::cout << k << ": " << c[k] << "\n";

  return 0;
}
