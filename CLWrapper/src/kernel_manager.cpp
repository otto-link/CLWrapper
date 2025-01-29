/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <iostream>

#include "cl_error_lookup.hpp"

#include "cl_wrapper/device_manager.hpp"
#include "cl_wrapper/kernel_manager.hpp"
#include "cl_wrapper/logger.hpp"

#include <iostream>

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
  Logger::log()->trace("loading kernel sources");

  if (this->full_sources.length() > 0)
  {
    cl::Device cl_device = clwrapper::DeviceManager::device();
    this->cl_context = cl::Context({cl_device});

    cl::Program::Sources sources;

    sources.push_back(
        {this->full_sources.c_str(), this->full_sources.length()});

    Logger::log()->trace("building OpenCL kernels");

    this->cl_program = cl::Program(this->cl_context, sources);
    int err = this->cl_program.build({cl_device});

    if (err != 0)
    {
      Logger::log()->critical("build error");
      std::cout << " Error building, compiler says:\n"
                << "----------------------------------------------\n"
                << this->cl_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(
                       cl_device)
                << "----------------------------------------------\n";
      clerror::throw_opencl_error(err);
    }

    std::string kernel_names = this->cl_program
                                   .getInfo<CL_PROGRAM_KERNEL_NAMES>();
    Logger::log()->trace("available kernels: {}", kernel_names.c_str());
  }
  else
  {
    Logger::log()->error("program building skipped, kernel sources are empty");
  }
}

} // namespace clwrapper
