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
  explicit TrixLoaderBase(std::unique_ptr<ir::Subgraphs> &subgs)
    : _base(nullptr), _fd(-1), _subgraphs(subgs)
  {
  }

  /**
   * @brief Load a model from file
   *
   * @param file_path
   */
  void loadFromFile(const std::string &file_path);

protected:
  /*
   * @brief Load model actually
   *
   * Used for both loadFromFile and loadFromBuffer
   */
  virtual void loadModel() {}

  /*
   * @brief Verify model
   *
   * @return true if model is valid, false otherwise
   *
   */
  virtual bool verifyModel() const { return true; }

private:
  // Base address for mapped region for loading (if needed)
  uint8_t *_base;
  // size of model in memory in bytes
  size_t _sz;
  // loaded file description
  // -1 means model is in-memory, nonnegative means model is constructed from file
  int _fd;
  // path to model (e.g. tvn)
  std::string _model_path;
  // Reference on loadable subgraphs
  std::unique_ptr<ir::Subgraphs> &_subgraphs;
};

} // namespace trix_loader
} // namespace onert

#endif // __TRIX_TRIX_LOADER_BASE_H__
