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

#include "DumpingHooks.h"
#include <cmath>

using namespace mpqsolver::core;

DumpingHooks::DumpingHooks(const std::string &save_path)
  : _save_path(save_path), _dumper(_save_path)
{
}

void DumpingHooks::onBeginSolver(const std::string &model_path, float q8error, float q16error)
{
  _model_path = model_path;
  _dumper.setModelPath(_model_path);
  if (!std::isnan(q8error) || !std::isnan(q16error))
  {
    _dumper.prepareForErrorDumping();
  }
  if (!std::isnan(q8error))
  {
    _dumper.dumpQ8Error(q8error);
  }
  if (!std::isnan(q16error))
  {
    _dumper.dumpQ16Error(q16error);
  }
}

void DumpingHooks::onBeginIteration()
{
  _in_iterations = true;
  _num_of_iterations += 1;
}

void DumpingHooks::onEndIteration(const LayerParams &layers, const std::string &def_type,
                                  float error)
{
  _dumper.dumpMPQConfiguration(layers, def_type, "channel", _num_of_iterations);
  _dumper.dumpMPQError(error, _num_of_iterations);
  _in_iterations = false;
}

void DumpingHooks::onEndSolver(const LayerParams &layers, const std::string &def_dtype,
                               float qerror)
{
  _dumper.dumpFinalMPQ(layers, def_dtype, "channel");
  if (!std::isnan(qerror))
  {
    _dumper.dumpMPQError(qerror);
  }
}

void DumpingHooks::onQuantized(luci::Module *module) const
{
  if (_in_iterations)
  {
    _dumper.dumpQuantized(module, _num_of_iterations);
  }
}
