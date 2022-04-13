/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __TRIX_TRIX_LOADER_BASE_H__
#define __TRIX_TRIX_LOADER_BASE_H__

#include "ir/Graph.h"
#include <memory>

namespace onert
{
namespace trix_loader
{
class TrixLoaderBase
{
public:
  /**
   * @brief Construct a new Loader object
   *
   * @param graph reference on subgraphs
   */
  explicit TrixLoaderBase(std::unique_ptr<ir::Subgraphs> &subgs) : _subgraphs(subgs) {}

  /**
   * @brief Load a model from file
   * @param file_path
   */
  void loadFromFile(const std::string &file_path);

  // loadFromBuffer() may be added here.

protected:
  /*
   * @brief Load model actually
   * @return true on successful loading, false otherwise
   */
  virtual bool loadModel() { return true; }

protected:
  /** path to model (e.g. tvn) */
  std::string _model_path;
  /** Reference on loadable subgraphs */
  std::unique_ptr<ir::Subgraphs> &_subgraphs;
};

} // namespace trix_loader
} // namespace onert

#endif // __TRIX_TRIX_LOADER_BASE_H__
