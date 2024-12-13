/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <iostream>

#include "cl_wrapper.hpp"

int main()
{
  const std::string code =
#include "kernel.cl"
      ;

  clwrapper::KernelManager::get_instance().add_kernel(code);

  auto run = clwrapper::Run("img_3x3_avg");

  int width = 4;
  int height = 3;

  std::vector<float> a(width * height, 2.f);
  std::vector<float> b(width * height); // output

  // NB - data are copied at binding, and one vector host data can be
  // binded to multiple device image buffers
  run.bind_imagef("a", a, width, height, clwrapper::Direction::IN);
  run.bind_imagef("b", b, width, height, clwrapper::Direction::OUT);

  run.bind_arguments(width, height);

  run.execute({width, height});

  run.read_imagef("b");

  for (auto &v : b)
    std::cout << v << "\n";

  return 0;
}
