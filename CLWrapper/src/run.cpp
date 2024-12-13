/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "macrologger.h"

#include "cl_error_lookup.hpp"

#include "cl_wrapper/device_manager.hpp"
#include "cl_wrapper/kernel_manager.hpp"
#include "cl_wrapper/run.hpp"

#include <iostream>

namespace clwrapper
{

Run::Run(const std::string &kernel_name) : kernel_name(kernel_name)
{
  LOG_DEBUG("Run::Run [%s]", this->kernel_name.c_str());

  this->queue = cl::CommandQueue(KernelManager::context(),
                                 DeviceManager::device());

  this->cl_kernel = cl::Kernel(KernelManager::program(),
                               this->kernel_name.c_str(),
                               &err);
  clerror::throw_opencl_error(err);
}

Run::~Run()
{
  this->queue.finish();
}

void Run::execute(int total_elements)
{
  LOG_DEBUG("executing... [%s]", this->kernel_name.c_str());

  this->queue.flush();

  // ensure gloabl size is rounded up to the nearest multiple of local
  // group size
  int bsize = KernelManager::get_instance().get_block_size();
  int gsize = ((total_elements + bsize - 1ull) / bsize) * bsize;

  const cl::NDRange global_work_size(gsize);
  const cl::NDRange local_work_size(bsize);

  err = this->queue.enqueueNDRangeKernel(this->cl_kernel,
                                         cl::NullRange,
                                         global_work_size,
                                         local_work_size);
  clerror::throw_opencl_error(err);

  err = this->queue.flush();
  clerror::throw_opencl_error(err);
}

void Run::execute(const std::vector<int> &global_range_2d)
{
  LOG_DEBUG("executing... [%s]", this->kernel_name.c_str());

  this->queue.flush();

  int bsize = KernelManager::get_instance().get_block_size();
  int gsize_x = ((global_range_2d[0] + bsize - 1ull) / bsize) * bsize;
  int gsize_y = ((global_range_2d[0] + bsize - 1ull) / bsize) * bsize;

  const cl::NDRange global_work_size(gsize_x, gsize_y);
  const cl::NDRange local_work_size(bsize, bsize);

  err = this->queue.enqueueNDRangeKernel(this->cl_kernel,
                                         cl::NullRange,
                                         global_work_size,
                                         local_work_size);
  clerror::throw_opencl_error(err);

  err = this->queue.flush();
  clerror::throw_opencl_error(err);
}

void Run::read_buffer(const std::string &id)
{
  if (this->buffers.find(id) != this->buffers.end())
  {
    err = this->queue.enqueueReadBuffer(buffers[id].cl_buffer,
                                        CL_TRUE,
                                        0,
                                        buffers[id].size,
                                        buffers[id].vector_ref);
    clerror::throw_opencl_error(err);
  }
  else
  {
    LOG_ERROR("unknown buffer id: [%s]", id.c_str());
  }
}

void Run::write_buffer(const std::string &id)
{
  if (this->buffers.find(id) != this->buffers.end())
  {
    err = this->queue.enqueueWriteBuffer(buffers[id].cl_buffer,
                                         CL_TRUE,
                                         0,
                                         buffers[id].size,
                                         buffers[id].vector_ref);
    clerror::throw_opencl_error(err);
  }
  else
  {
    LOG_ERROR("unknown buffer id: [%s]", id.c_str());
  }
}

} // namespace clwrapper
