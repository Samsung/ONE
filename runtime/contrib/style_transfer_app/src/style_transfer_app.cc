/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "args.h"
#include "bitmap_helper.h"
#include "nnfw.h"

#ifdef NNFW_ST_APP_JPEG_SUPPORTED
#include "jpeg_helper.h"
#endif

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <math.h>

#define NNPR_ENSURE_STATUS(a)        \
  do                                 \
  {                                  \
    if ((a) != NNFW_STATUS_NO_ERROR) \
    {                                \
      exit(-1);                      \
    }                                \
  } while (0)

enum ImageFormat
{
  JPEG = 0,
  BMP,
  OTHERS
};

uint64_t NowMicros()
{
  auto time_point = std::chrono::high_resolution_clock::now();
  auto since_epoch = time_point.time_since_epoch();
  // default precision of high resolution clock is 10e-9 (nanoseconds)
  return std::chrono::duration_cast<std::chrono::microseconds>(since_epoch).count();
}

uint64_t num_elems(const nnfw_tensorinfo *ti)
{
  uint64_t n = 1;
  for (uint32_t i = 0; i < ti->rank; ++i)
  {
    assert(ti->dims[i] >= 0);
    n *= ti->dims[i];
  }
  return n;
}

NNFW_STATUS resolve_op_backend(nnfw_session *session)
{
  static std::unordered_map<std::string, std::string> operation_map = {
    {"TRANSPOSE_CONV", "OP_BACKEND_TransposeConv"},      {"CONV_2D", "OP_BACKEND_Conv2D"},
    {"DEPTHWISE_CONV_2D", "OP_BACKEND_DepthwiseConv2D"}, {"MEAN", "OP_BACKEND_Mean"},
    {"AVERAGE_POOL_2D", "OP_BACKEND_AvgPool2D"},         {"MAX_POOL_2D", "OP_BACKEND_MaxPool2D"},
    {"INSTANCE_NORM", "OP_BACKEND_InstanceNorm"},        {"ADD", "OP_BACKEND_Add"}};

  for (auto i : operation_map)
  {
    char *default_backend = std::getenv(i.second.c_str());
    if (default_backend)
    {
      NNFW_STATUS return_result = nnfw_set_op_backend(session, i.first.c_str(), default_backend);
      if (return_result == NNFW_STATUS_ERROR)
        return return_result;
    }
  }

  return NNFW_STATUS_NO_ERROR;
}

ImageFormat get_image_format(const std::string &FileName)
{
  std::string ext;
  if (FileName.find_last_of(".") != std::string::npos)
    ext = FileName.substr(FileName.find_last_of(".") + 1);

  if (ext == "jpeg" || ext == "jpg")
    return ImageFormat::JPEG;
  else if (ext == "bmp")
    return ImageFormat::BMP;
  else
    return ImageFormat::OTHERS;
}

static int vector_tanh(std::vector<float> &a)
{
  int size = a.size();

#pragma omp parallel for
  for (int i = 0; i < size; i++)
  {
    float temp = tanh(a[i]) * 150 + 127.5f;
    a[i] = temp > 255 ? 255 : temp < 0 ? 0 : temp;
  }

  return 0;
}

int main(const int argc, char **argv)
{
  StyleTransferApp::Args args(argc, argv);
  auto nnpackage_path = args.getPackageFilename();

  nnfw_session *session = nullptr;
  NNPR_ENSURE_STATUS(nnfw_create_session(&session));
  char *available_backends = std::getenv("BACKENDS");
  if (available_backends)
    NNPR_ENSURE_STATUS(nnfw_set_available_backends(session, available_backends));
  NNPR_ENSURE_STATUS(resolve_op_backend(session));

  NNPR_ENSURE_STATUS(nnfw_load_model_from_file(session, nnpackage_path.c_str()));

  uint32_t num_inputs;
  NNPR_ENSURE_STATUS(nnfw_input_size(session, &num_inputs));

  // verify input and output

  if (num_inputs == 0)
  {
    std::cerr << "[ ERROR ] "
              << "No inputs in model => execution is not possible" << std::endl;
    exit(1);
  }

  auto verifyInputTypes = [session]() {
    uint32_t sz;
    NNPR_ENSURE_STATUS(nnfw_input_size(session, &sz));
    for (uint32_t i = 0; i < sz; ++i)
    {
      nnfw_tensorinfo ti;
      NNPR_ENSURE_STATUS(nnfw_input_tensorinfo(session, i, &ti));
      if (ti.dtype != NNFW_TYPE_TENSOR_FLOAT32)
      {
        std::cerr << "Only float 32bit is supported." << std::endl;
        exit(-1);
      }
    }
  };

  auto verifyOutputTypes = [session]() {
    uint32_t sz;
    NNPR_ENSURE_STATUS(nnfw_output_size(session, &sz));

    for (uint32_t i = 0; i < sz; ++i)
    {
      nnfw_tensorinfo ti;
      NNPR_ENSURE_STATUS(nnfw_output_tensorinfo(session, i, &ti));
      if (ti.dtype != NNFW_TYPE_TENSOR_FLOAT32)
      {
        std::cerr << "Only float 32bit is supported." << std::endl;
        exit(-1);
      }
    }
  };

  verifyInputTypes();
  verifyOutputTypes();

  // prepare execution

  uint64_t prepare_us = NowMicros();
  NNPR_ENSURE_STATUS(nnfw_prepare(session));
  prepare_us = NowMicros() - prepare_us;

  // prepare input

  std::vector<std::vector<float>> inputs(num_inputs);

  auto loadInputs = [session, num_inputs, &inputs](const std::string &filename) {
    for (uint32_t i = 0; i < num_inputs; ++i)
    {
      nnfw_tensorinfo ti;
      NNPR_ENSURE_STATUS(nnfw_input_tensorinfo(session, i, &ti));

      ImageFormat format = get_image_format(filename);
      switch (format)
      {
        case ImageFormat::JPEG:
        {
#ifdef NNFW_ST_APP_JPEG_SUPPORTED
          StyleTransferApp::JpegHelper jpeg_helper;
          jpeg_helper.readJpeg(filename, inputs[i], ti.dims[2], ti.dims[1]);
#else
          std::cerr << "JPEG format not supported. Install libjpeg to read/write jpeg images."
                    << std::endl;
          exit(-1);
#endif
          break;
        }
        case ImageFormat::BMP:
        {
          StyleTransferApp::BitmapHelper bitmap_helper;
          bitmap_helper.read_bmp(filename, inputs[i], ti.dims[2], ti.dims[1]);
          break;
        }
        default:
          std::cerr << "Unsupported image format." << std::endl;
          exit(-1);
          break;
      }

      NNPR_ENSURE_STATUS(nnfw_set_input(session, i, NNFW_TYPE_TENSOR_FLOAT32, inputs[i].data(),
                                        sizeof(float) * num_elems(&ti)));
      NNPR_ENSURE_STATUS(nnfw_set_input_layout(session, i, NNFW_LAYOUT_CHANNELS_LAST));
    }
  };

  if (!args.getInputFilename().empty())
    loadInputs(args.getInputFilename());
  else
    std::exit(-1);

  // prepare output

  uint32_t num_outputs = 0;
  NNPR_ENSURE_STATUS(nnfw_output_size(session, &num_outputs));
  std::vector<std::vector<float>> outputs(num_outputs);

  for (uint32_t i = 0; i < num_outputs; i++)
  {
    nnfw_tensorinfo ti;
    NNPR_ENSURE_STATUS(nnfw_output_tensorinfo(session, i, &ti));
    auto output_num_elements = num_elems(&ti);
    outputs[i].resize(output_num_elements);
    NNPR_ENSURE_STATUS(nnfw_set_output(session, i, NNFW_TYPE_TENSOR_FLOAT32, outputs[i].data(),
                                       sizeof(float) * output_num_elements));
    NNPR_ENSURE_STATUS(nnfw_set_output_layout(session, i, NNFW_LAYOUT_CHANNELS_LAST));
  }

  uint64_t run_us = NowMicros();
  NNPR_ENSURE_STATUS(nnfw_run(session));
  run_us = NowMicros() - run_us;

  // dump output tensors

  auto dumpOutputs = [session, num_outputs, &outputs](const std::string &filename) {
    for (uint32_t i = 0; i < num_outputs; ++i)
    {
      nnfw_tensorinfo ti;
      NNPR_ENSURE_STATUS(nnfw_output_tensorinfo(session, i, &ti));

      vector_tanh(outputs[i]);

      ImageFormat format = get_image_format(filename);
      switch (format)
      {
        case ImageFormat::JPEG:
        {
#ifdef NNFW_ST_APP_JPEG_SUPPORTED
          StyleTransferApp::JpegHelper jpeg_helper;
          jpeg_helper.writeJpeg(filename, outputs[i], ti.dims[2], ti.dims[1]);
#else
          std::cerr << "JPEG format not supported. Install libjpeg to read/write jpeg images."
                    << std::endl;
          exit(-1);
#endif
          break;
        }
        case ImageFormat::BMP:
        {
          StyleTransferApp::BitmapHelper bitmap_helper;
          bitmap_helper.write_bmp(filename, outputs[i], ti.dims[2], ti.dims[1], ti.dims[3]);
          break;
        }
        default:
          std::cerr << "Unsupported image format." << std::endl;
          exit(-1);
          break;
      }
    }
  };

  if (!args.getOutputFilename().empty())
    dumpOutputs(args.getOutputFilename());

  std::cout << "nnfw_prepare takes " << prepare_us / 1e3 << " ms" << std::endl;
  std::cout << "nnfw_run     takes " << run_us / 1e3 << " ms" << std::endl;

  NNPR_ENSURE_STATUS(nnfw_close_session(session));

  return 0;
}
