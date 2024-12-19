/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "macrologger.h"

#include "cl_wrapper/device_manager.hpp"

namespace clwrapper
{

DeviceManager::DeviceManager()
{
  LOG_DEBUG("DeviceManager::DeviceManager");

  // initialize the device (example: first GPU)
  LOG_DEBUG("initializing OpenCL devices...");

  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  if (platforms.empty()) throw std::runtime_error("No OpenCL platforms found!");

  std::vector<cl::Device> devices;
  platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

  if (devices.empty()) throw std::runtime_error("No OpenCL devices found!");

  this->cl_device = devices[0]; // Select the first GPU device

  LOG_DEBUG("OpenCL device: %s",
            this->cl_device.getInfo<CL_DEVICE_NAME>().c_str());
}

// Access the OpenCL device
cl::Device DeviceManager::get_device() const
{
  return this->cl_device;
}

void log_device_infos(cl::Device cl_device)
{
  LOG_INFO("- device Name: %s", cl_device.getInfo<CL_DEVICE_NAME>().c_str());
  LOG_INFO(" - device Vendor: %s",
           cl_device.getInfo<CL_DEVICE_VENDOR>().c_str());

  switch (cl_device.getInfo<CL_DEVICE_TYPE>())
  {
  case CL_DEVICE_TYPE_GPU: LOG_INFO(" - device Type: GPU"); break;
  case CL_DEVICE_TYPE_CPU: LOG_INFO(" - device Type: CPU"); break;
  case CL_DEVICE_TYPE_ACCELERATOR:
    LOG_INFO(" - device Type: ACCELERATOR");
    break;
  default: LOG_INFO(" - device Type: unknown");
  }
}

} // namespace clwrapper
