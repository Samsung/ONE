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

namespace
{
onert_run::TensorShape getShape(H5::DataSet &data_set)
{
  std::vector<hsize_t> h5_shape; // hsize_t is unsigned long long
  H5::DataSpace data_space = data_set.getSpace();
  int rank = data_space.getSimpleExtentNdims();
  h5_shape.resize(rank);

  // read shape info from H5 file
  data_space.getSimpleExtentDims(h5_shape.data(), NULL);

  onert_run::TensorShape shape;
  for (auto dim : h5_shape)
    shape.emplace_back(static_cast<int32_t>(dim));

  return shape;
}
} // namespace

namespace onert_run
{
static const char *h5_value_grpname = "value";

std::vector<TensorShape> H5Formatter::readTensorShapes(const std::string &filename,
                                                       uint32_t num_inputs)
{
  std::vector<TensorShape> tensor_shapes;

  try
  {
    H5::Exception::dontPrint();

    H5::H5File file(filename, H5F_ACC_RDONLY);
    H5::Group value_group = file.openGroup(h5_value_grpname);

    // Constraints: if there are n data set names, they should be unique and
    //              one of [ "0", "1", .. , "n-1" ]
    for (uint32_t i = 0; i < num_inputs; ++i)
    {
      H5::DataSet data_set = value_group.openDataSet(std::to_string(i));
      H5::DataType type = data_set.getDataType();
      auto shape = getShape(data_set);

      tensor_shapes.emplace_back(shape);
    }

    return tensor_shapes;
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
}

void H5Formatter::loadInputs(const std::string &filename, std::vector<Allocation> &inputs)
{
  uint32_t num_inputs = inputs.size();
  try
  {
    // Turn off the automatic error printing.
    H5::Exception::dontPrint();

    H5::H5File file(filename, H5F_ACC_RDONLY);
    H5::Group value_group = file.openGroup(h5_value_grpname);
    for (uint32_t i = 0; i < num_inputs; ++i)
    {
      // TODO Add Assert(nnfw shape, h5 file shape size)
      auto bufsz = inputs[i].size();
      H5::DataSet data_set = value_group.openDataSet(std::to_string(i));
      H5::DataType type = data_set.getDataType();
      switch (inputs[i].type())
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
        case NNFW_TYPE_TENSOR_QUANT8_ASYMM_SIGNED:
          if (type == H5::PredType::STD_I8BE || type == H5::PredType::STD_I8LE)
            data_set.read(inputs[i].data(), H5::PredType::NATIVE_INT8);
          else
            throw std::runtime_error("model input type is int8. But h5 data type is different.");
          break;
        case NNFW_TYPE_TENSOR_QUANT16_SYMM_SIGNED:
          throw std::runtime_error("NYI for NNFW_TYPE_TENSOR_QUANT16_SYMM_SIGNED type");
        default:
          throw std::runtime_error("onert_run can load f32, i32, qasymm8, bool and uint8.");
      }
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

void H5Formatter::dumpOutputs(const std::string &filename, const std::vector<Allocation> &outputs,
                              const std::vector<TensorShape> &shape_map)
{
  uint32_t num_outputs = outputs.size();
  if (num_outputs != shape_map.size())
    throw std::runtime_error("Number of outputs and shape map are not matched");

  try
  {
    // Turn off the automatic error printing.
    H5::Exception::dontPrint();

    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group value_group = file.createGroup(h5_value_grpname);
    for (uint32_t i = 0; i < num_outputs; i++)
    {
      auto shape = shape_map[i];
      std::vector<hsize_t> dims(shape.size());
      for (uint32_t j = 0; j < shape.size(); ++j)
      {
        if (shape[j] >= 0)
          dims[j] = static_cast<hsize_t>(shape[j]);
        else
        {
          std::cerr << "Negative dimension in output tensor" << std::endl;
          exit(-1);
        }
      }
      H5::DataSpace data_space(shape.size(), dims.data());
      switch (outputs[i].type())
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
        case NNFW_TYPE_TENSOR_QUANT8_ASYMM_SIGNED:
        {
          H5::DataSet data_set =
            value_group.createDataSet(std::to_string(i), H5::PredType::STD_I8LE, data_space);
          data_set.write(outputs[i].data(), H5::PredType::NATIVE_INT8);
          break;
        }
        case NNFW_TYPE_TENSOR_QUANT16_SYMM_SIGNED:
          throw std::runtime_error("NYI for NNFW_TYPE_TENSOR_QUANT16_SYMM_SIGNED type");
        default:
          throw std::runtime_error("onert_run can dump f32, i32, qasymm8, bool and uint8.");
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
    std::cerr << "Error during dumpOutputs on onert_run : " << e.what() << std::endl;
    std::exit(-1);
  }
};

} // end of namespace onert_run
