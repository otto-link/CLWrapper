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
#include <map>

#include <CL/opencl.hpp>

#include "macrologger.h"

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

  static bool is_ready()
  {
    try
    {
      DeviceManager::get_instance();
    }
    catch (const std::exception &e)
    {
      LOG_ERROR("Error: %s", e.what());
      return false;
    }
    catch (...)
    {
      LOG_ERROR("Unknown error");
      return false;
    }

    return true;
  }

  // Get the OpenCL device attached to the singleton instance
  static cl::Device device()
  {
    return DeviceManager::get_instance().get_device();
  }

  // list all the available devices
  std::map<size_t, std::string> get_available_devices();

  // Access the OpenCL device
  cl::Device get_device() const;

  bool set_device(size_t platform_id);

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
