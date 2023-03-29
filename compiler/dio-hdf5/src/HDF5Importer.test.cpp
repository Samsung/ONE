/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "dio_hdf5/HDF5Importer.h"

#include <loco.h>

#include <H5Cpp.h>

#include <cstdio>

#include <gtest/gtest.h>

using HDF5Importer = dio::hdf5::HDF5Importer;
using Shape = std::vector<loco::Dimension>;
using DataType = loco::DataType;

namespace
{

const std::string file_name("dio_hdf5_test.h5");

void createFile()
{
  // File already exists. Remove it.
  if (auto f = fopen(file_name.c_str(), "r"))
  {
    fclose(f);
    if (remove(file_name.c_str()) != 0)
      throw std::runtime_error("Error deleting file.");
  }

  const auto rank = 3;
  hsize_t dim[3] = {1, 2, 3};
  H5::DataSpace space(rank, dim);

  float data[] = {0, 1, 2, 3, 4, 5};

  // Create test file in the current directory
  H5::H5File file(file_name, H5F_ACC_TRUNC);
  {
    file.createGroup("/value");
    file.createGroup("/value/0");
    H5::DataSet dataset(file.createDataSet("/value/0/0", H5::PredType::IEEE_F32BE, space));
    dataset.write(data, H5::PredType::IEEE_F32LE);
  }
}

} // namespace

TEST(dio_hdf5_test, read_with_type_shape)
{
  createFile();

  HDF5Importer h5(::file_name);

  h5.importGroup("value");

  std::vector<float> buffer(6);

  DataType dtype;
  Shape shape;
  h5.readTensor(0, 0, &dtype, &shape, buffer.data(), buffer.size() * sizeof(float));

  for (uint32_t i = 0; i < 6; i++)
    EXPECT_EQ(i, buffer[i]);

  EXPECT_EQ(DataType::FLOAT32, dtype);
  EXPECT_EQ(3, shape.size());
  EXPECT_EQ(1, shape[0]);
  EXPECT_EQ(2, shape[1]);
  EXPECT_EQ(3, shape[2]);
}

TEST(dio_hdf5_test, wrong_path_NEG)
{
  const std::string wrong_path = "not_existing_file_for_dio_hdf5_test";

  EXPECT_ANY_THROW(HDF5Importer h5(wrong_path));
}

TEST(dio_hdf5_test, wrong_group_name_NEG)
{
  createFile();

  HDF5Importer h5(::file_name);

  EXPECT_ANY_THROW(h5.importGroup("wrong"));
}

TEST(dio_hdf5_test, data_out_of_index_NEG)
{
  createFile();

  HDF5Importer h5(::file_name);

  h5.importGroup("value");

  std::vector<float> buffer(6);

  DataType dtype;
  Shape shape;
  // Read non-existing data (data_idx = 1)
  EXPECT_ANY_THROW(
    h5.readTensor(1, 0, &dtype, &shape, buffer.data(), buffer.size() * sizeof(float)));
}

TEST(dio_hdf5_test, input_out_of_index_NEG)
{
  createFile();

  HDF5Importer h5(::file_name);

  h5.importGroup("value");

  std::vector<float> buffer(6);

  DataType dtype;
  Shape shape;
  // Read non-existing input (input_idx = 1)
  EXPECT_ANY_THROW(
    h5.readTensor(0, 1, &dtype, &shape, buffer.data(), buffer.size() * sizeof(float)));
}

TEST(dio_hdf5_test, wrong_buffer_size_NEG)
{
  createFile();

  HDF5Importer h5(::file_name);

  h5.importGroup("value");

  std::vector<float> buffer(6);

  DataType dtype;
  Shape shape;
  EXPECT_ANY_THROW(h5.readTensor(0, 0, &dtype, &shape, buffer.data(), 1 /* wrong buffer size */));
}
