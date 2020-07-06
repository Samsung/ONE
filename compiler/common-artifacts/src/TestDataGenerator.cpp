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

#include <foder/FileLoader.h>
#include <luci/Importer.h>
#include <luci_interpreter/Interpreter.h>
#include <mio/circle/schema_generated.h>

#include <H5Cpp.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <string>

namespace
{

uint32_t element_num(std::vector<hsize_t> &vec)
{
  return static_cast<uint32_t>(
      std::accumulate(std::begin(vec), std::end(vec), 1, std::multiplies<uint32_t>()));
}

H5::PredType hdf5_dtype_cast(const loco::DataType loco_dtype)
{
  switch (loco_dtype)
  {
    case loco::DataType::U8:
      return H5::PredType::NATIVE_UINT8;
    case loco::DataType::S32:
      return H5::PredType::NATIVE_INT32;
    case loco::DataType::S64:
      return H5::PredType::NATIVE_INT64;
    case loco::DataType::FLOAT32:
      return H5::PredType::NATIVE_FLOAT;
    default:
      throw std::runtime_error("NYI data type.");
  }
}

template <typename T> void geneate_random_data(std::mt19937 &gen, void *data, uint32_t size)
{
  std::normal_distribution<float> distrib(0, 2); // mean(0), stddev(2)
  for (uint32_t i = 0; i < size; i++)
  {
    static_cast<T *>(data)[i] = static_cast<T>(distrib(gen));
  }
}

void fill_random_data(void *data, uint32_t size, loco::DataType dtype)
{
  std::random_device rd;  // used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // standard mersenne_twister_engine seeded with rd()

  switch (dtype)
  {
    case loco::DataType::U8:
      geneate_random_data<uint8_t>(gen, data, size);
      break;
    case loco::DataType::S32:
      geneate_random_data<int32_t>(gen, data, size);
      break;
    case loco::DataType::S64:
      geneate_random_data<int64_t>(gen, data, size);
      break;
    case loco::DataType::FLOAT32:
      geneate_random_data<float>(gen, data, size);
      break;
    default:
      break;
  }
}

} // namespace

int entry(int argc, char **argv)
{
  std::string circle_file{argv[1]};
  size_t last_dot_index = circle_file.find_last_of(".");
  std::string prefix = circle_file.substr(0, last_dot_index);

  // load circle file
  foder::FileLoader file_loader{circle_file};
  std::vector<char> model_data = file_loader.load();
  const circle::Model *circle_model = circle::GetModel(model_data.data());
  if (circle_model == nullptr)
  {
    std::cerr << "ERROR: Failed to load circle '" << circle_file << "'" << std::endl;
    return EXIT_FAILURE;
  }

  // load luci module
  std::unique_ptr<luci::Module> module = luci::Importer().importModule(circle_model);
  luci_interpreter::Interpreter interpreter(module.get());

  /**
   *  HDF5 layout is like below
   *
   *  GROUP "/"
   *   ㄴGROUP "name"
   *     ㄴATTRIBUTE "0"
   *       ㄴDATA (0): "input_01:0"
   *     ㄴATTRIBUTE "1"
   *       ㄴDATA (0): "input_02:0"
   *   ㄴGROUP "value"
   *     ㄴDATASET "0"
   *       ㄴDATA ...
   *     ㄴDATASET "1"
   *       ㄴDATA ...
   */
  // create random data and dump into hdf5 file
  H5::H5File input_file{prefix + ".input.h5", H5F_ACC_TRUNC};
  std::unique_ptr<H5::Group> input_name_group =
      std::make_unique<H5::Group>(input_file.createGroup("name"));
  std::unique_ptr<H5::Group> input_value_group =
      std::make_unique<H5::Group>(input_file.createGroup("value"));

  H5::H5File output_file{prefix + ".expected.h5", H5F_ACC_TRUNC};
  std::unique_ptr<H5::Group> output_name_group =
      std::make_unique<H5::Group>(output_file.createGroup("name"));
  std::unique_ptr<H5::Group> output_value_group =
      std::make_unique<H5::Group>(output_file.createGroup("value"));

  uint32_t input_index = 0;
  for (uint32_t g = 0; g < circle_model->subgraphs()->size(); g++)
  {
    const auto input_nodes = loco::input_nodes(module->graph(g));
    for (const auto &node : input_nodes)
    {
      const auto *input_node = dynamic_cast<const luci::CircleInput *>(node);
      std::string name = input_node->name();
      if (name.find(":") == std::string::npos)
        name += ":";

      // create attribute
      H5::DataSpace name_dataspace(H5S_SCALAR);
      H5::StrType name_datatype(H5::PredType::C_S1, name.size());

      auto name_attr = input_name_group->createAttribute(std::to_string(input_index), name_datatype,
                                                         name_dataspace);

      name_attr.write(name_datatype, name);

      // create value
      std::vector<hsize_t> dims(input_node->rank());
      for (uint32_t d = 0; d < input_node->rank(); d++)
      {
        dims.at(d) = input_node->dim(d).value();
        assert(dims.at(d) >= 0);
      }
      auto dataspace = std::make_unique<H5::DataSpace>(dims.size(), dims.data());
      auto dtype = hdf5_dtype_cast(input_node->dtype());
      auto dataset = std::make_unique<H5::DataSet>(
          input_file.createDataSet("value/" + std::to_string(input_index), dtype, *dataspace));

      auto data_size = ::element_num(dims);
      auto dtype_size = loco::size(input_node->dtype());
      auto byte_size = dtype_size * data_size;
      std::vector<int8_t> data(byte_size);

      // generate random data
      fill_random_data(data.data(), data_size, input_node->dtype());

      dataset->write(data.data(), dtype);

      interpreter.writeInputTensor(input_node, data.data(), byte_size);

      input_index++;
    }
  }

  interpreter.interpret();

  // dump output data into hdf5 file
  uint32_t output_index = 0;
  for (uint32_t g = 0; g < circle_model->subgraphs()->size(); g++)
  {
    const auto output_nodes = loco::output_nodes(module->graph(g));
    for (const auto &node : output_nodes)
    {
      const auto *output_node = dynamic_cast<const luci::CircleOutput *>(node);
      std::string name = output_node->name();
      if (name.find(":") == std::string::npos)
        name += ":";

      // create attribute
      H5::DataSpace name_dataspace(H5S_SCALAR);
      H5::StrType name_datatype(H5::PredType::C_S1, name.size());

      auto name_attr = output_name_group->createAttribute(std::to_string(output_index),
                                                          name_datatype, name_dataspace);

      name_attr.write(name_datatype, name);

      // create value
      std::vector<hsize_t> dims(output_node->rank());
      for (uint32_t d = 0; d < output_node->rank(); d++)
      {
        dims.at(d) = output_node->dim(d).value();
        assert(dims.at(d) >= 0);
      }
      auto dataspace = std::make_unique<H5::DataSpace>(dims.size(), dims.data());
      auto dtype = hdf5_dtype_cast(output_node->dtype());
      auto dataset = std::make_unique<H5::DataSet>(
          output_file.createDataSet("value/" + std::to_string(output_index), dtype, *dataspace));

      uint32_t tensor_bytesize = loco::size(output_node->dtype());
      tensor_bytesize *= ::element_num(dims);
      std::vector<int8_t> output_data(tensor_bytesize);
      interpreter.readOutputTensor(output_node, output_data.data(), output_data.size());

      dataset->write(output_data.data(), dtype);

      output_index++;
    }
  }

  return EXIT_SUCCESS;
}
