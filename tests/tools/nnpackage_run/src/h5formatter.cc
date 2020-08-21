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

#include "h5formatter.h"
#include "nnfw.h"
#include "nnfw_util.h"

#include <iostream>
#include <stdexcept>
#include <H5Cpp.h>

namespace nnpkg_run
{
static const char *h5_value_grpname = "value";

void H5Formatter::loadInputs(const std::string &filename, std::vector<Allocation> &inputs)
{
  uint32_t num_inputs;
  NNPR_ENSURE_STATUS(nnfw_input_size(session_, &num_inputs));
  try
  {
    // Turn off the automatic error printing.
    H5::Exception::dontPrint();

    H5::H5File file(filename, H5F_ACC_RDONLY);
    H5::Group value_group = file.openGroup(h5_value_grpname);
    for (uint32_t i = 0; i < num_inputs; ++i)
    {
      nnfw_tensorinfo ti;
      NNPR_ENSURE_STATUS(nnfw_input_tensorinfo(session_, i, &ti));
      // allocate memory for data
      auto bufsz = bufsize_for(&ti);
      inputs[i].alloc(bufsz);

      H5::DataSet data_set = value_group.openDataSet(std::to_string(i));
      H5::DataType type = data_set.getDataType();
      switch (ti.dtype)
      {
        case NNFW_TYPE_TENSOR_FLOAT32:
          if (type == H5::PredType::IEEE_F32BE || type == H5::PredType::IEEE_F32LE)
            data_set.read(inputs[i].data(), H5::PredType::NATIVE_FLOAT);
          else
            throw std::runtime_error("model input type is f32. But h5 data type is different.");
          break;
        case NNFW_TYPE_TENSOR_INT32:
          if (type == H5::PredType::STD_I32BE || type == H5::PredType::STD_I32LE)
            data_set.read(inputs[i].data(), H5::PredType::NATIVE_INT32);
          else
            throw std::runtime_error("model input type is i32. But h5 data type is different.");
          break;
        case NNFW_TYPE_TENSOR_INT64:
          if (type == H5::PredType::STD_I64BE || type == H5::PredType::STD_I64LE)
            data_set.read(inputs[i].data(), H5::PredType::NATIVE_INT64);
          else
            throw std::runtime_error("model input type is i64. But h5 data type is different.");
          break;
        case NNFW_TYPE_TENSOR_QUANT8_ASYMM:
        case NNFW_TYPE_TENSOR_BOOL:
        case NNFW_TYPE_TENSOR_UINT8:
          if (type == H5::PredType::STD_U8BE || type == H5::PredType::STD_U8LE)
            data_set.read(inputs[i].data(), H5::PredType::NATIVE_UINT8);
          else
            throw std::runtime_error(
                "model input type is qasymm8, bool or uint8. But h5 data type is different.");
          break;
        default:
          throw std::runtime_error("nnpkg_run can load f32, i32, qasymm8, bool and uint8.");
      }
      NNPR_ENSURE_STATUS(nnfw_set_input(session_, i, ti.dtype, inputs[i].data(), bufsz));
      NNPR_ENSURE_STATUS(nnfw_set_input_layout(session_, i, NNFW_LAYOUT_CHANNELS_LAST));
    }
  }
  catch (const H5::Exception &e)
  {
    H5::Exception::printErrorStack();
    std::exit(-1);
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << std::endl;
    std::exit(-1);
  }
};

void H5Formatter::dumpOutputs(const std::string &filename, std::vector<Allocation> &outputs)
{
  uint32_t num_outputs;
  NNPR_ENSURE_STATUS(nnfw_output_size(session_, &num_outputs));
  try
  {
    // Turn off the automatic error printing.
    H5::Exception::dontPrint();

    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group value_group = file.createGroup(h5_value_grpname);
    for (uint32_t i = 0; i < num_outputs; i++)
    {
      nnfw_tensorinfo ti;
      NNPR_ENSURE_STATUS(nnfw_output_tensorinfo(session_, i, &ti));
      std::vector<hsize_t> dims(ti.rank);
      for (uint32_t j = 0; j < ti.rank; ++j)
      {
        if (ti.dims[j] >= 0)
          dims[j] = static_cast<hsize_t>(ti.dims[j]);
        else
        {
          std::cerr << "Negative dimension in output tensor" << std::endl;
          exit(-1);
        }
      }
      H5::DataSpace data_space(ti.rank, dims.data());
      switch (ti.dtype)
      {
        case NNFW_TYPE_TENSOR_FLOAT32:
        {
          H5::DataSet data_set =
              value_group.createDataSet(std::to_string(i), H5::PredType::IEEE_F32BE, data_space);
          data_set.write(outputs[i].data(), H5::PredType::NATIVE_FLOAT);
          break;
        }
        case NNFW_TYPE_TENSOR_INT32:
        {
          H5::DataSet data_set =
              value_group.createDataSet(std::to_string(i), H5::PredType::STD_I32LE, data_space);
          data_set.write(outputs[i].data(), H5::PredType::NATIVE_INT32);
          break;
        }
        case NNFW_TYPE_TENSOR_INT64:
        {
          H5::DataSet data_set =
              value_group.createDataSet(std::to_string(i), H5::PredType::STD_I64LE, data_space);
          data_set.write(outputs[i].data(), H5::PredType::NATIVE_INT64);
          break;
        }
        case NNFW_TYPE_TENSOR_UINT8:
        case NNFW_TYPE_TENSOR_QUANT8_ASYMM:
        {
          H5::DataSet data_set =
              value_group.createDataSet(std::to_string(i), H5::PredType::STD_U8BE, data_space);
          data_set.write(outputs[i].data(), H5::PredType::NATIVE_UINT8);
          break;
        }
        case NNFW_TYPE_TENSOR_BOOL:
        {
          H5::DataSet data_set =
              value_group.createDataSet(std::to_string(i), H5::PredType::STD_U8LE, data_space);
          data_set.write(outputs[i].data(), H5::PredType::NATIVE_INT8);
          break;
        }
        default:
          throw std::runtime_error("nnpkg_run can dump f32, i32, qasymm8, bool and uint8.");
      }
    }
  }
  catch (const H5::Exception &e)
  {
    H5::Exception::printErrorStack();
    std::exit(-1);
  }
  catch (const std::runtime_error &e)
  {
    std::cerr << "Error during dumpOutputs on nnpackage_run : " << e.what() << std::endl;
    std::exit(-1);
  }
};

} // end of namespace nnpkg_run
