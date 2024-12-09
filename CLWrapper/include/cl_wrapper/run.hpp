/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */

/**
 * @file run.hpp
 * @author Otto Link (otto.link.bv@gmail.com)
 * @brief
 *
 * @copyright Copyright (c) 2025
 */
#pragma once
#include <map>

#include <CL/opencl.hpp>

#include "cl_error_lookup.hpp"
#include "macrologger.h"

namespace clwrapper
{

// helper
template <typename T> size_t vector_sizeof(const typename std::vector<T> &v)
{
  return sizeof(T) * v.size();
}

struct Buffer
{
  cl::Buffer cl_buffer;
  void      *vector_ref;
  size_t     size;
};

// class
class Run
{
public:
  Run(const std::string &kernel_name);

  ~Run();

  template <typename T> void bind_arguments(T arg)
  {
    err = this->cl_kernel.setArg(this->arg_count++, arg);
    clerror::throw_opencl_error(err);
  }

  template <typename... Args> void bind_arguments(Args... args)
  {
    // Expand the parameter pack and call fct for each argument
    (this->bind_arguments(args), ...);
  }

  template <typename T>
  void bind_buffer(const std::string &id,
                   std::vector<T>    &vector,
                   cl_mem_flags       flags = CL_MEM_READ_WRITE)
  {
    Buffer buffer;

    buffer.vector_ref = static_cast<void *>(vector.data());
    buffer.size = vector_sizeof<float>(vector);
    buffer.cl_buffer = cl::Buffer(KernelManager::context(),
                                  flags,
                                  buffer.size,
                                  nullptr,
                                  &err);
    clerror::throw_opencl_error(err);

    err = this->cl_kernel.setArg(this->arg_count++, buffer.cl_buffer);
    clerror::throw_opencl_error(err);

    this->buffers[id] = buffer;
  }

  void execute(int total_elements);

  void read_buffer(const std::string &id);

  void write_buffer(const std::string &id);

private:
  std::string kernel_name;

  cl::CommandQueue queue;

  cl::Kernel cl_kernel;

  int arg_count = 0;

  std::map<std::string, Buffer> buffers;

  int err = 0;
};

} // namespace clwrapper