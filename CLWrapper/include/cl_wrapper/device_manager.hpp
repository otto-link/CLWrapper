/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */

/**
 * @file device_manager.hpp
 * @author Otto Link (otto.link.bv@gmail.com)
 * @brief
 *
 * @copyright Copyright (c) 2025
 */
#pragma once
#include <CL/opencl.hpp>

namespace clwrapper
{

class DeviceManager
{
public:
  // Get the singleton instance
  static DeviceManager &get_instance()
  {
    static DeviceManager instance;
    return instance;
  }

  // Get the OpenCL device attached to the singleton instance
  static cl::Device device()
  {
    return DeviceManager::get_instance().get_device();
  }

  // Access the OpenCL device
  cl::Device get_device() const;

private:
  cl::Device cl_device;

  // Private constructor
  DeviceManager();

  // Delete copy constructor and assignment operator to enforce singleton
  DeviceManager(const DeviceManager &) = delete;
  DeviceManager &operator=(const DeviceManager &) = delete;
};

void log_device_infos(cl::Device cl_device);

} // namespace clwrapper