/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <iostream>

#include "macrologger.h"

#include "cl_wrapper/device_manager.hpp"
#include "cl_wrapper/kernel_manager.hpp"

namespace clwrapper
{

KernelManager::KernelManager()
{
  this->build_program();
}

void KernelManager::add_kernel(const std::string &kernel_sources)
{
  this->full_sources += kernel_sources;
  this->build_program();
}

void KernelManager::build_program()
{
  LOG_DEBUG("loading kernel sources");

  if (this->full_sources.length() > 0)
  {
    cl::Device cl_device = clwrapper::DeviceManager::device();
    this->cl_context = cl::Context({cl_device});

    cl::Program::Sources sources;

    sources.push_back(
        {this->full_sources.c_str(), this->full_sources.length()});

    LOG_DEBUG("building OpenCL kernels");

    this->cl_program = cl::Program(this->cl_context, sources);
    std::string building_options = "-DBLOCK_SIZE=" +
                                   std::to_string(this->block_size);

    LOG_DEBUG("building options: %s", building_options.c_str());

    if (this->cl_program.build({cl_device}, building_options.c_str()) !=
        CL_SUCCESS)
    {
      LOG_ERROR("build error");
      std::cout << " Error building: "
                << this->cl_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(
                       cl_device)
                << "\n";
      throw std::runtime_error("build error");
    }

    std::string kernel_names = this->cl_program
                                   .getInfo<CL_PROGRAM_KERNEL_NAMES>();
    LOG_DEBUG("available kernels: %s", kernel_names.c_str());
  }
  else
  {
    LOG_ERROR("program building skipped, kernel sources are empty");
  }
}

void KernelManager::set_block_size(int new_block_size)
{
  this->block_size = new_block_size;
  this->build_program();
}

} // namespace clwrapper
