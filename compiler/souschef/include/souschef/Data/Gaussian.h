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

#ifndef __SOUSCHEF_GAUSSIAN_FILLER_H__
#define __SOUSCHEF_GAUSSIAN_FILLER_H__

#include "souschef/DataChef.h"

namespace souschef
{

/**
 * @brief Generate a sequence of random values according to the gaussian(=normal) distribution
 */
class GaussianFloat32DataChef final : public DataChef
{
public:
  GaussianFloat32DataChef(float mean, float stddev) : _mean{mean}, _stddev{stddev}
  {
    // DO NOTHING
  }

public:
  std::vector<uint8_t> generate(int32_t count) const override;

private:
  float _mean;
  float _stddev;
};

class GaussianInt32DataChef final : public DataChef
{
public:
  GaussianInt32DataChef(float mean, float stddev) : _mean{mean}, _stddev{stddev}
  {
    // DO NOTHING
  }

public:
  std::vector<uint8_t> generate(int32_t count) const override;

private:
  float _mean;
  float _stddev;
};

class GaussianUint8DataChef final : public DataChef
{
public:
  GaussianUint8DataChef(float mean, float stddev) : _mean{mean}, _stddev{stddev}
  {
    // DO NOTHING
  }

public:
  std::vector<uint8_t> generate(int32_t count) const override;

private:
  float _mean;
  float _stddev;
};

struct GaussianFloat32DataChefFactory : public DataChefFactory
{
  std::unique_ptr<DataChef> create(const Arguments &args) const;
};

struct GaussianInt32DataChefFactory : public DataChefFactory
{
  std::unique_ptr<DataChef> create(const Arguments &args) const;
};

struct GaussianUint8DataChefFactory : public DataChefFactory
{
  std::unique_ptr<DataChef> create(const Arguments &args) const;
};

} // namespace souschef

#endif // __SOUSCHEF_GAUSSIAN_FILLER_H__
