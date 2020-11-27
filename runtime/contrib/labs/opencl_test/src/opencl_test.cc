/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*******************************************************************************
 * Copyright (c) 2008-2015 The Khronos Group Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and/or associated documentation files (the
 * "Materials"), to deal in the Materials without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Materials, and to
 * permit persons to whom the Materials are furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
 ******************************************************************************/

#include "arm_compute/core/CL/OpenCL.h"

#include <iostream>
#include <vector>

void printDeviceInfo(int n, cl::Device &device, cl::Device &default_device)
{
  bool is_default = (device() == default_device());
  std::cout << "\t\t\t#" << n << " Device: (id: " << device() << ") "
            << (is_default ? " -> default" : "") << "\n";

  const auto name = device.getInfo<CL_DEVICE_NAME>();
  std::cout << "\t\t\t\tName: " << name << "\n";

  const auto compute_unit = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
  std::cout << "\t\t\t\tMax Compute Unit: " << compute_unit << "\n";

  const auto max_work_item_size = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
  std::cout << "\t\t\t\tMax Work Item Size: [";
  for (auto size : max_work_item_size)
    std::cout << size << ",";
  std::cout << "]\n";

  const auto max_work_group_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
  std::cout << "\t\t\t\tMax Work Grpup Size: " << max_work_group_size << "\n";

  const auto max_clock_frequency = device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
  std::cout << "\t\t\t\tMax Clock Frequency: " << max_clock_frequency << "\n";

  std::cout << "\n";
}

class OpenCLGpu
{
public:
  cl::Platform platform_;
  cl::Context context_;
  cl::vector<cl::Device> devices_;
  std::vector<cl::CommandQueue *> q_;
  cl::Program program_;

  OpenCLGpu()
  {
    cl_int cl_error;

    platform_ = cl::Platform::getDefault();

    try
    {
      cl_context_properties properties[3] = {CL_CONTEXT_PLATFORM,
                                             (cl_context_properties)platform_(), 0};

      context_ = cl::Context(CL_DEVICE_TYPE_GPU, properties, NULL, NULL, &cl_error);
    }
    catch (cl::Error &err) // thrown when there is no Context for this platform
    {
      std::cout << "\t\t No Context Found\n";
      return;
    }

    devices_ = context_.getInfo<CL_CONTEXT_DEVICES>();

    for (int dev_id = 0; dev_id < devices_.size(); dev_id++)
    {
      cl::CommandQueue *que = new cl::CommandQueue(context_, devices_[dev_id]);
      q_.emplace_back(que);
    }
  }

  ~OpenCLGpu()
  {
    for (auto each_q : q_)
      delete each_q;
  }

  void buildProgram(std::string &kernel_source_code)
  {
    std::vector<std::string> programStrings{kernel_source_code};

    program_ = cl::Program(context_, programStrings);

    try
    {
      program_.build("-cl-std=CL1.2");
    }
    catch (cl::Error &err)
    {
      cl_int buildErr = CL_SUCCESS;
      auto buildInfo = program_.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
      for (auto &pair : buildInfo)
      {
        std::cerr << pair.second << std::endl << std::endl;
      }
    }
  }
};

void checkContextMem()
{
  cl_int cl_error;

  // get context, devices
  //
  std::cout << "\nChecking if devices in GPU shares the same memory address:\n\n";

  OpenCLGpu gpu;

  std::cout << "\nDevices in GPU:\n\n";

  auto &devices = gpu.devices_;
  auto default_device = cl::Device::getDefault();

  int d = 0;
  for (auto device : devices)
    printDeviceInfo(++d, device, default_device);

  if (d < 2)
  {
    std::cout << "\t\t This options works when there are n (>= 2) devices.\n";
    return;
  }

  // allocate and map memory

  typedef cl_int T;
  const int items_per_device = 128;
  const int length = items_per_device * devices.size();

  std::vector<T> input(length);
  std::vector<T> output(length, 0);

  for (int i = 0; i < length; i++)
    input[i] = i;

  cl::Buffer input_buf(gpu.context_, (cl_mem_flags)CL_MEM_USE_HOST_PTR, length * sizeof(T),
                       input.data(), &cl_error);
  cl::Buffer output_buf(gpu.context_, (cl_mem_flags)CL_MEM_USE_HOST_PTR, length * sizeof(T),
                        output.data(), &cl_error);

  // compile test cl code

  std::string kernel_source{"typedef int T;                                                 \n"
                            "kernel void memory_test(                                       \n"
                            "   const int dev_id,                                           \n"
                            "   global T* input,                                            \n"
                            "   global T* output,                                           \n"
                            "   const int start_idx,                                        \n"
                            "   const int count)                                            \n"
                            "{                                                              \n"
                            "   int input_idx = get_global_id(0);                           \n"
                            "   if(input_idx < count)                                       \n"
                            "   {                                                           \n"
                            "       int output_idx = start_idx + input_idx;                 \n"
                            "       output[output_idx] = input[input_idx] + dev_id;         \n"
                            "   }                                                           \n"
                            "}                                                              \n"};

  gpu.buildProgram(kernel_source);

  try
  {
    auto kernel_functor = cl::KernelFunctor<cl_int, cl::Buffer, cl::Buffer, cl_int, cl_int>(
      gpu.program_, "memory_test"); // name should be same as cl function name

    // create a queue per device and queue a kernel job

    for (int dev_id = 0; dev_id < devices.size(); dev_id++)
    {
      kernel_functor(cl::EnqueueArgs(*(gpu.q_[dev_id]), cl::NDRange(items_per_device)),
                     (cl_int)dev_id, // dev id
                     input_buf, output_buf,
                     (cl_int)(items_per_device * dev_id), // start index
                     (cl_int)(items_per_device),          // count
                     cl_error);
    }

    // sync

    for (d = 0; d < devices.size(); d++)
      (gpu.q_[d])->finish();

    // check if memory state changed by all devices

    cl::copy(*(gpu.q_[0]), output_buf, begin(output), end(output));

    bool use_same_memory = true;

    for (int dev_id = 0; dev_id < devices.size(); dev_id++)
    {
      for (int i = 0; i < items_per_device; ++i)
      {
        int output_idx = items_per_device * dev_id + i;
        if (output[output_idx] != input[i] + dev_id)
        {
          std::cout << "Output[" << output_idx << "] : "
                    << "expected = " << input[i] + dev_id << "; actual = " << output[output_idx]
                    << "\n";
          use_same_memory = false;
          break;
        }
      }
    }

    if (use_same_memory)
      std::cout << "\n=> Mapped memory addresses used by devices in GPU are same.\n\n";
    else
      std::cout << "\n=> Mapped memory addresses used by devices in GPU are different.\n\n";
  }
  catch (cl::Error &err)
  {
    std::cerr << "error: code: " << err.err() << ", what: " << err.what() << std::endl;
  }
}

void printHelp()
{
  std::cout << "opencl information: \n\n";
  std::cout << "\t -h : help\n";
  std::cout
    << "\t -g : print if memory map is shared among devices in GPU (in default platform)\n\n";
  std::cout << "\t -s : test for synchronized work by two devices in a GPU\n\n";
}

#include <mutex>
#include <chrono>
#include <thread>
#include <condition_variable>

#define MAX_DEVICE_NUM 8 // just for testing

int kernel_idx[MAX_DEVICE_NUM];
unsigned char kernel_completed = 0x00; // bit 0 = 1 means kernel by device[0] was completed.
unsigned char
  kernel_completed_flag; // if comparing kernel_completed with this var, all kernels are completed
int device_num;
std::mutex kernel_complete_handler_mutex;

std::condition_variable wakeup_main;
std::mutex wakeup_main_mutex;

void notifyKernelFinished(cl_event ev, cl_int ev_info, void *device_idx)
{
  std::cout << "callback from device[" << *((int *)device_idx) << "] : ==> completed.\n";

  std::unique_lock<std::mutex> lock(kernel_complete_handler_mutex);

  kernel_completed |= 0x01 << *((int *)device_idx);
  if (kernel_completed == kernel_completed_flag)
    wakeup_main.notify_one();
}

void testSync()
{
  OpenCLGpu gpu;

  cl_int cl_error;
  typedef cl_int T;
  const int items_per_device = 1024 * 768;
  const int length = items_per_device * gpu.devices_.size();

  std::vector<T> output(length, 0);

  cl::Buffer output_buf(gpu.context_, (cl_mem_flags)CL_MEM_USE_HOST_PTR, length * sizeof(T),
                        output.data(), &cl_error);

  std::string kernel_source{"kernel void test(global float* output, const int count)  \n"
                            "{                                                        \n"
                            "   int idx = get_global_id(0);                           \n"
                            "   if(idx < count)                                       \n"
                            "   {                                                     \n"
                            "       float x = hypot(idx/1.111, idx*1.111);            \n"
                            "       for (int y = 0; y < 200; y++)                     \n"
                            "         x = rootn(log(pown(rootn(log(pown(x, 20)), 5), 20)), 5);  \n"
                            "       output[idx] = x;                                  \n"
                            "   }                                                     \n"
                            "}                                                        \n"};

  gpu.buildProgram(kernel_source);

  try
  {
    auto kernel_functor = cl::KernelFunctor<cl::Buffer, cl_int>(
      gpu.program_, "test"); // name should be same as cl function name

    // variable init
    cl::Event ev[MAX_DEVICE_NUM];

    device_num = gpu.devices_.size();

    kernel_completed = 0;
    kernel_completed_flag = 0;
    for (int i = 0; i < device_num; i++)
    {
      kernel_idx[i] = i;
      kernel_completed_flag |= 0x01 << i;
    }

    // create a queue per device and queue a kernel job
    // queueing with callback function
    for (int dev_id = 0; dev_id < gpu.devices_.size(); dev_id++)
    {
      ev[dev_id] = kernel_functor(cl::EnqueueArgs(*(gpu.q_[dev_id]), cl::NDRange(items_per_device)),
                                  output_buf,
                                  (cl_int)(items_per_device), // count
                                  cl_error);
      ev[dev_id].setCallback(CL_COMPLETE, notifyKernelFinished, (void *)(kernel_idx + dev_id));

      // how to check kernel execution status
      //
      // auto status  = ev[dev_id].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>();
      // std::cout << "Event status = " << (status == CL_QUEUED ? "CL_QUEUED" : status ==
      // CL_SUBMITTED ? "CL_SUBMITTED" : status == CL_COMPLETE ? "CL_COMPLETE" : "unknown")
      //           << std::endl;
      // std::cout << "Event status code = " << status << std::endl;
    }

    // long wait until kernels are over
    {
      std::unique_lock<std::mutex> lk(wakeup_main_mutex);
      wakeup_main.wait(lk, [] { return (kernel_completed == kernel_completed_flag); });

      std::cout << "all devices were completed.\n";
    }
  }
  catch (cl::Error &err)
  {
    std::cerr << "error: code: " << err.err() << ", what: " << err.what() << std::endl;
  }
}

int main(const int argc, char **argv)
{
  if (argc < 2)
    printHelp();
  else
  {
    std::string option = argv[1];

    if (option == "-h") // help
      printHelp();
    else if (option == "-g") // check if devices in GPU uses same memory address
      checkContextMem();
    else if (option == "-s") // check synchronization between devices in GPU
      testSync();
  }
  return 0;
}
