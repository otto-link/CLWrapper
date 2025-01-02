/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <iostream>

#include "cl_wrapper.hpp"

int main()
{
  std::map<size_t, std::string> cl_device_map =
      clwrapper::DeviceManager::get_instance().get_available_devices();

  std::cout << "Available devices:\n";

  for (auto &[id, name] : cl_device_map)
    std::cout << "device: id = " << id << ", name = " << name << "\n";

  // --- execute the same kernel on each device

  const std::string code =
#include "add.cl"
      ;

  // add the source (will be build for the current or default device)
  clwrapper::KernelManager::get_instance().add_kernel(code);

  for (auto &[id, name] : cl_device_map)
  {
    std::cout << "\n\n--- Running kernel on " << name << " ---\n\n";

    if (clwrapper::DeviceManager::get_instance().set_device(id))
    {
      // program needs to be rebuild for the current device
      clwrapper::KernelManager::get_instance().build_program();

      // reminder is "standard" run execution
      auto run = clwrapper::Run("add_kernel");

      int                n = 9;
      std::vector<float> a(n, 1.f);
      std::vector<float> b(n, 2.f);
      std::vector<float> c(n); // output

      run.bind_buffer<float>("a", a);
      run.bind_buffer<float>("b", b);
      run.bind_buffer<float>("c", c);
      run.write_buffer("a");
      run.write_buffer("b");

      run.execute(n);
      run.read_buffer("c");

      for (auto &v : c)
        std::cout << v << "\n";
    }
  }

  return 0;
}
