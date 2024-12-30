/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */

/**
 * @file kernel_manager.hpp
 * @author Otto Link (otto.link.bv@gmail.com)
 * @brief
 *
 * @copyright Copyright (c) 2025
 */
#pragma once
#include <CL/opencl.hpp>

namespace clwrapper
{

class KernelManager
{
public:
  // Get the singleton instance
  static KernelManager &get_instance()
  {
    static KernelManager instance;
    return instance;
  }

  // Get the OpenCL context attached to the singleton instance
  static cl::Context context()
  {
    return KernelManager::get_instance().get_context();
  }

  static cl::Program program()
  {
    return KernelManager::get_instance().get_program();
  }

  void add_kernel(const std::string &kernel_sources);

  void build_program();

  int get_block_size() const
  {
    return this->block_size;
  }

  cl::Context get_context() const
  {
    return this->cl_context;
  }

  cl::Program get_program() const
  {
    return this->cl_program;
  }

  void set_block_size(int new_block_size);

private:
  // Private constructor
  KernelManager();

  // Delete copy constructor and assignment operator to enforce singleton
  KernelManager(const KernelManager &) = delete;
  KernelManager &operator=(const KernelManager &) = delete;

  int block_size = 32;

  cl::Program cl_program;

  cl::Context cl_context;

  std::string full_sources = "";
};

} // namespace clwrapper
