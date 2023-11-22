/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __MPQSOLVER_DATA_PROVIDER_H__
#define __MPQSOLVER_DATA_PROVIDER_H__

#include <dio_hdf5/HDF5Importer.h>

#include <luci/IR/Module.h>

#include <string>
#include <vector>

namespace mpqsolver
{
namespace core
{

struct InputData
{
  InputData(size_t size) : _data(size) {}

  const std::vector<char> &data() const { return _data; }

  std::vector<char> &data() { return _data; }

private:
  std::vector<char> _data;
};

class DataProvider
{
public:
  virtual ~DataProvider() = default;
  virtual size_t numSamples() const = 0;
  virtual uint32_t numInputs(uint32_t sample) const = 0;
  virtual void getSampleInput(uint32_t sample, uint32_t input, InputData &data) const = 0;
};

class H5FileDataProvider final : public DataProvider
{
public:
  H5FileDataProvider(const std::string &h5file, const std::string &module_path);
  size_t numSamples() const override;
  uint32_t numInputs(uint32_t sample) const override;
  void getSampleInput(uint32_t sample, uint32_t input, InputData &data) const override;

private:
  std::vector<loco::Node *> _input_nodes;
  std::unique_ptr<luci::Module> _module;
  dio::hdf5::HDF5Importer _importer;
  bool _is_raw_data = false;
};

} // namespace core
} // namespace mpqsolver

#endif //__MPQSOLVER_DATA_PROVIDER_H__
