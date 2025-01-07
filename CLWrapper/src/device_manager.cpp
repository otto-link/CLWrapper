/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "macrologger.h"

#include "cl_wrapper/device_manager.hpp"

namespace clwrapper
{

bool helper_find_string_insensitive(const std::string &text,
                                    const std::string &word)
{
  // https://stackoverflow.com/questions/3152241
  auto it = std::search(text.begin(),
                        text.end(),
                        word.begin(),
                        word.end(),
                        [](unsigned char ch1, unsigned char ch2)
                        { return std::toupper(ch1) == std::toupper(ch2); });
  return (it != text.end());
}

DeviceManager::DeviceManager()
{
  LOG_DEBUG("DeviceManager::DeviceManager");

  // initialize the device (example: first GPU)
  LOG_DEBUG("initializing OpenCL devices...");

  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  if (platforms.empty()) throw std::runtime_error("No OpenCL platforms found!");

  // select the platform with the most computational resources
  // (assuming one device / per platform)
  int platform_index = 0;
  int flops_best = 0;

  LOG_DEBUG("checking device performances...");

  for (size_t kp = 0; kp < platforms.size(); kp++)
  {
    LOG_DEBUG("checking platform: %s - %s",
              platforms[kp].getInfo<CL_PLATFORM_VENDOR>().c_str(),
              platforms[kp].getInfo<CL_PLATFORM_NAME>().c_str());

    std::vector<cl::Device> devices;
    platforms[kp].getDevices(this->device_type, &devices);

    if (devices.empty())
    {
      LOG_ERROR("No OpenCL devices found for this platform!");
    }
    else
    {
      // estimate of the number of cores per computational unit for the
      // platform
      std::string vendor = devices[0].getInfo<CL_DEVICE_VENDOR>();
      int         cores = 1;

      if (helper_find_string_insensitive(vendor, "nvidia") ||
          helper_find_string_insensitive(vendor, "amd"))
        cores = 128;
      else if (helper_find_string_insensitive(vendor, "intel"))
        cores = 16;

      int flops = (int)devices[0].getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() *
                  (int)devices[0].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() *
                  cores;

      if (flops > flops_best)
      {
        flops_best = flops;
        platform_index = kp;
      }

      LOG_DEBUG("rating - device: %s, vendor: %s, rating: %d",
                devices[0].getInfo<CL_DEVICE_NAME>().c_str(),
                vendor.c_str(),
                flops);
    }
  }

  // eventually assign the platform / device
  std::vector<cl::Device> devices;
  platforms[platform_index].getDevices(this->device_type, &devices);
  this->cl_device = devices[0];
  this->device_id = platform_index;

  LOG_DEBUG("OpenCL device: %s",
            this->cl_device.getInfo<CL_DEVICE_NAME>().c_str());
}

std::map<size_t, std::string> DeviceManager::get_available_devices()
{
  std::map<size_t, std::string> device_map = {};

  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  if (platforms.empty()) throw std::runtime_error("No OpenCL platforms found!");

  for (size_t kp = 0; kp < platforms.size(); kp++)
  {
    std::vector<cl::Device> devices;
    platforms[kp].getDevices(this->device_type, &devices);

    if (devices.empty())
    {
      LOG_ERROR("No OpenCL devices found for this platform!");
    }
    else
    {
      std::string name = platforms[kp].getInfo<CL_PLATFORM_VENDOR>() + "/" +
                         platforms[kp].getInfo<CL_PLATFORM_NAME>() + "/" +
                         devices[0].getInfo<CL_DEVICE_NAME>();

      device_map[kp] = name;
    }
  }

  return device_map;
}

cl::Device DeviceManager::get_device() const
{
  return this->cl_device;
}

bool DeviceManager::set_device(size_t platform_id)
{
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  if (platforms.empty()) throw std::runtime_error("No OpenCL platforms found!");

  std::vector<cl::Device> devices;
  platforms[platform_id].getDevices(this->device_type, &devices);

  if (devices.empty())
  {
    LOG_ERROR("No OpenCL devices found for this platform!");
    return false;
  }
  else
  {
    this->cl_device = devices[0];
    this->device_id = platform_id;

    LOG_DEBUG("OpenCL device: %s",
              this->cl_device.getInfo<CL_DEVICE_NAME>().c_str());
  }

  return true;
}

void log_device_infos(cl::Device cl_device)
{
  LOG_INFO("- device Name: %s", cl_device.getInfo<CL_DEVICE_NAME>().c_str());
  LOG_INFO(" - device Vendor: %s",
           cl_device.getInfo<CL_DEVICE_VENDOR>().c_str());
  LOG_INFO(" - device Version: %s",
           cl_device.getInfo<CL_DEVICE_VERSION>().c_str());

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
