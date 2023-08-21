/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Dump.h"

#include <mio_circle/Reader.h>

#include <H5Cpp.h>

#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace
{

template <typename T>
void print_comma_sepearted(std::ostream &os, const flatbuffers::Vector<T> *vec)
{
  if (vec == nullptr)
    return;
  for (auto iter = vec->begin(); iter != vec->end(); iter++)
  {
    if (iter != vec->begin())
      os << ", ";
    os << *iter;
  }
}

void print_buffer(std::ostream &os, uint32_t buff_idx, const flatbuffers::Vector<uint8_t> *data_ptr,
                  const circle::TensorType &type)
{
  if (data_ptr == nullptr)
    return;

  os << " └── buffer" << std::endl;
  os << "     ├── index : " << buff_idx << std::endl;
  size_t buff_size = data_ptr->size();
  os << "     ├── size  : " << buff_size << std::endl;
  os << "     └── data  : ";
  switch (type)
  {
    case circle::TensorType_UINT8:
    {
      const uint8_t *buff_data_ui8 = reinterpret_cast<const uint8_t *>(data_ptr->data());
      for (uint32_t idx = 0; idx < buff_size / sizeof(uint8_t); idx++)
      {
        os << static_cast<const uint32_t>(buff_data_ui8[idx]) << ", ";
      }
      break;
    }
    case circle::TensorType_INT32:
    {
      const int32_t *buff_data_i32 = reinterpret_cast<const int32_t *>(data_ptr->data());
      for (uint32_t idx = 0; idx < buff_size / sizeof(int32_t); idx++)
      {
        os << buff_data_i32[idx] << ", ";
      }
      break;
    }
    case circle::TensorType_INT64:
    {
      const int64_t *buff_data_i64 = reinterpret_cast<const int64_t *>(data_ptr->data());
      for (uint32_t idx = 0; idx < buff_size / sizeof(int64_t); idx++)
      {
        os << buff_data_i64[idx] << ", ";
      }
      break;
    }
    case circle::TensorType_FLOAT32:
    {
      const float *buff_data_f32 = reinterpret_cast<const float *>(data_ptr->data());
      for (uint32_t idx = 0; idx < buff_size / sizeof(float); idx++)
      {
        os << buff_data_f32[idx] << ", ";
      }
      break;
    }
    default:
      throw std::runtime_error("NYI tensor type : " + std::to_string(type));
  }
  os << std::endl;
}

} // namespace

namespace circletensordump
{

void DumpTensors::run(std::ostream &os, const circle::Model *model, const std::string &)
{
  mio::circle::Reader reader(model);
  uint32_t num_subgraph = reader.num_subgraph();
  auto buffers = reader.buffers();

  for (uint32_t subgraph_idx = 0; subgraph_idx < num_subgraph; subgraph_idx++)
  {
    reader.select_subgraph(subgraph_idx);

    auto tensors = reader.tensors();
    for (const auto &tensor : *tensors)
    {
      const auto tensor_name = tensor->name();
      std::string tensor_name_str = tensor_name ? tensor_name->str() : "no_name";
      os << std::string(70, '-') << std::endl;
      os << "[" << tensor_name_str << "]" << std::endl;
      auto buff_idx = tensor->buffer();
      auto buff_data_ptr = reader.buffers()->Get(buff_idx)->data();
      auto quant_param = tensor->quantization();
      std::string print_format = (!buff_data_ptr && !quant_param) ? "└──" : "├──";

      // shape
      auto shape = tensor->shape();
      os << " " + print_format + " shape : (";
      ::print_comma_sepearted(os, shape);
      os << ")" << std::endl;

      // quantization paramters
      if (quant_param)
      {
        std::string print_format1 = buff_data_ptr ? "├──" : "└──";
        std::string print_format2 = buff_data_ptr ? "│" : " ";
        os << " " + print_format1 + " quantization" << std::endl;
        auto min = quant_param->min();
        auto max = quant_param->max();
        auto scale = quant_param->scale();
        auto zero_point = quant_param->zero_point();
        auto quantized_dimension = quant_param->quantized_dimension();

        os << " " + print_format2 + "   ├── min        : ";
        ::print_comma_sepearted(os, min);
        os << std::endl;
        os << " " + print_format2 + "   ├── max        : ";
        ::print_comma_sepearted(os, max);
        os << std::endl;
        os << " " + print_format2 + "   ├── scale      : ";
        ::print_comma_sepearted(os, scale);
        os << std::endl;
        os << " " + print_format2 + "   ├── zero_point : ";
        ::print_comma_sepearted(os, zero_point);
        os << std::endl;
        os << " " + print_format2 + "   └── quantized_dimension : " << quantized_dimension;
        os << std::endl;
      }

      // buffer
      print_buffer(os, buff_idx, buff_data_ptr, tensor->type());
      os << std::endl;
    }
  }
}

} // namespace circletensordump

namespace
{

// HDF5 forbids the inclusion of '/' in the name.
std::string mangle(const std::string &name)
{
  std::string ret{name};
  std::replace(ret.begin(), ret.end(), '/', '_');
  return ret;
}

H5::PredType hdf5_dtype_cast(const circle::TensorType &circle_type)
{
  switch (circle_type)
  {
    case circle::TensorType_UINT8:
    {
      return H5::PredType::NATIVE_UINT8;
    }
    case circle::TensorType_INT8:
    {
      return H5::PredType::NATIVE_INT8;
    }
    case circle::TensorType_INT16:
    {
      return H5::PredType::NATIVE_INT16;
    }
    case circle::TensorType_INT32:
    {
      return H5::PredType::NATIVE_INT32;
    }
    case circle::TensorType_INT64:
    {
      return H5::PredType::NATIVE_INT64;
    }
    case circle::TensorType_FLOAT32:
    {
      return H5::PredType::NATIVE_FLOAT;
    }
    default:
      throw std::runtime_error("NYI tensor type : " + std::to_string(circle_type));
  }
}

/**
 *  In order to create a dataspace, its rank and dimensions are required as hsize_t type.
 *  This function converts flatbuffers::Vector<T> to std::vector<hsize_t>.
 *
 *  If "dims" parameter is passed, the parameter will be converted. However, if
 *  not passed(nullptr), data is considered as a rank 1 vector.
 */
template <typename T>
std::vector<hsize_t> hdf5_dims_cast(const flatbuffers::Vector<T> *data,
                                    const flatbuffers::Vector<int32_t> *dims = nullptr)
{
  std::vector<hsize_t> ret;
  if (data != nullptr)
  {
    if (dims == nullptr)
    {
      ret.resize(1);
      ret.at(0) = data->size();
    }
    else
    {
      const uint32_t rank = dims->size();
      ret.resize(rank);
      for (uint32_t d = 0; d < rank; d++)
      {
        if (dims->Get(d) < 0)
          throw std::runtime_error("Dimensions shouldn't be negative");
        ret.at(d) = static_cast<hsize_t>(dims->Get(d));
      }
    }
  }
  return ret;
}

/**
 *  This function writes vector data to given hdf5 file like below.
 *
 *  GROUP "group_name"
 *   ㄴDATATYPE "type"
 *   ㄴDATASET "dataset_name"
 *   ㄴDATASPACE "dims"
 *   ㄴDATA "data"
 */
template <typename T>
void write_vector_data_to_hdf5(H5::H5File &file, std::string &group_name, std::string dataset_name,
                               const H5::PredType &type, const flatbuffers::Vector<T> *data,
                               std::vector<hsize_t> dims)
{
  if (data == nullptr)
    return;
  auto dataspace = std::make_unique<H5::DataSpace>(dims.size(), dims.data());
  auto dataset = std::make_unique<H5::DataSet>(
    file.createDataSet(group_name + "/" + dataset_name, type, *dataspace));
  dataset->write(data->data(), type);
}

/// @brief This function writes scalar data to given hdf5 file
template <typename T>
void write_scalar_data_to_hdf5(H5::H5File &file, std::string &group_name, std::string dataset_name,
                               const H5::PredType &type, T data)
{
  auto dataspace = std::make_unique<H5::DataSpace>(H5S_SCALAR);
  auto dataset = std::make_unique<H5::DataSet>(
    file.createDataSet(group_name + "/" + dataset_name, type, *dataspace));
  dataset->write(&data, type);
}

} // namespace

namespace circletensordump
{

/**
 *  HDF5 layout is like below
 *
 *  GROUP "/"
 *   ㄴGROUP "tensor name"
 *     ㄴDATASET "weights"    : Shape (x, y, ...), type(uint8, int16)
 *     ㄴDATASET "min"        : Shape (n)
 *     ㄴDATASET "max"        : Shape (n)
 *     ㄴDATASET "scale"      : Shape (m)
 *     ㄴDATASET "zero_point" : Shape (m)
 *
 *  NOTE All Dataset is optional. It means that if tensor doesn't have the data, it won't be created
 *  as a Dataset
 *
 */
void DumpTensorsToHdf5::run(std::ostream &os, const circle::Model *model,
                            const std::string &output_path)
{
  // loads a circle model
  mio::circle::Reader reader(model);
  uint32_t num_subgraph = reader.num_subgraph();

  // create a hdf5 file
  H5::H5File file{output_path, H5F_ACC_TRUNC};

  for (uint32_t subgraph_idx = 0; subgraph_idx < num_subgraph; subgraph_idx++)
  {
    reader.select_subgraph(subgraph_idx);

    auto tensors = reader.tensors();
    for (const auto &tensor : *tensors)
    {
      // If tensor does not have name, do nothing.
      const auto tensor_name = tensor->name();
      if (tensor_name == nullptr)
      {
        assert(false && "There is no tensor name");
        continue;
      }

      // create a group for each tensor whose name is its tensor name
      std::string group_name = ::mangle(tensor_name->c_str());
      std::unique_ptr<H5::Group> tensor_group =
        std::make_unique<H5::Group>(file.createGroup(group_name));

      // write a buffer data
      uint32_t buff_idx = tensor->buffer();
      auto buff_data_ptr = reader.buffers()->Get(buff_idx)->data();
      if (buff_data_ptr)
      {
        ::write_vector_data_to_hdf5(file, group_name, "weights", ::hdf5_dtype_cast(tensor->type()),
                                    buff_data_ptr,
                                    ::hdf5_dims_cast(buff_data_ptr, tensor->shape()));
      }

      // write quantization parameters
      auto quant_param = tensor->quantization();
      if (quant_param)
      {
        auto min = quant_param->min();
        ::write_vector_data_to_hdf5(file, group_name, "min", H5::PredType::NATIVE_FLOAT, min,
                                    ::hdf5_dims_cast(min));
        auto max = quant_param->max();
        ::write_vector_data_to_hdf5(file, group_name, "max", H5::PredType::NATIVE_FLOAT, max,
                                    ::hdf5_dims_cast(max));
        auto scale = quant_param->scale();
        ::write_vector_data_to_hdf5(file, group_name, "scale", H5::PredType::NATIVE_FLOAT, scale,
                                    ::hdf5_dims_cast(scale));
        auto zero_point = quant_param->zero_point();
        ::write_vector_data_to_hdf5(file, group_name, "zero_point", H5::PredType::NATIVE_INT64,
                                    zero_point, ::hdf5_dims_cast(zero_point));
        auto quantized_dimension = quant_param->quantized_dimension();
        ::write_scalar_data_to_hdf5(file, group_name, "quantized_dimension",
                                    H5::PredType::NATIVE_INT32, quantized_dimension);
      }
    }
  }
}

} // namespace circletensordump
