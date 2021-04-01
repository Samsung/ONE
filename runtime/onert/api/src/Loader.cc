/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Loader.h"

#include <circle_loader.h>
#include <tflite_loader.h>

namespace onert
{
namespace api
{

std::unique_ptr<ir::Subgraphs> Loader::loadCircleBuffer(uint8_t *buffer, size_t size)
{
  try
  {
    return onert::circle_loader::loadModel(buffer, size);
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during model loading : " << e.what() << std::endl;
    return nullptr;
  }
}

std::unique_ptr<ir::Subgraphs> Loader::loadModelFile(const std::string &path,
                                                     const std::string &type)
{
  try
  {
    if (type == "tflite")
    {
      return onert::tflite_loader::loadModel(path);
    }
    else if (type == "circle")
    {
      return onert::circle_loader::loadModel(path);
    }
    else
    {
      std::cerr << "Unsupported model type" << std::endl;
      return nullptr;
    }
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during model loading : " << e.what() << std::endl;
    return nullptr;
  }
}

} // namespace api
} // namespace onert
