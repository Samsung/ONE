/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ShapeParser.h"
#include <Dim.h>

#include <algorithm>
#include <stdexcept>
#include <sstream>

using namespace circle_resizer;

namespace
{
bool is_blank(const std::string &s)
{
  return !s.empty() && std::find_if(s.begin(), s.end(),
                                    [](unsigned char c) { return !std::isblank(c); }) == s.end();
}

Shape parse_single_shape(const std::string &shape)
{
  if (shape.empty() || is_blank(shape))
  {
    return Shape::scalar();
  }

  std::vector<Dim> result_dims;
  std::stringstream shape_stream(shape);
  std::string token;
  try
  {
    while (std::getline(shape_stream, token, ','))
    {
      result_dims.push_back(Dim{std::stoi(token)});
    }
  }
  catch (...)
  {
    throw std::invalid_argument("Error during shape processing: " + shape);
  }
  if (result_dims.empty())
  {
    throw std::invalid_argument("No shapes found in input string: " + shape);
  }
  return Shape{result_dims};
}

} // namespace

std::vector<Shape> circle_resizer::parse_shapes(const std::string &shapes)
{
  std::vector<Shape> result_shapes;
  auto shapes_tmp = shapes;
  std::string token;
  size_t begin_pos = 0, end_pos = 0;
  while ((begin_pos = shapes_tmp.find_first_of("[")) != std::string::npos &&
         (end_pos = shapes_tmp.find_first_of("]")) != std::string::npos)
  {
    if (begin_pos > end_pos)
    {
      throw std::invalid_argument("Invalid shape format: " + shapes);
    }
    const size_t token_size = end_pos - begin_pos - 1;
    token = shapes_tmp.substr(begin_pos + 1, token_size);
    result_shapes.push_back(parse_single_shape(token));
    shapes_tmp.erase(0, end_pos + 1);
  }

  if (result_shapes.empty())
  {
    throw std::invalid_argument("No shapes found in the input string: " + shapes);
  }

  // the rest of the input not handled by loop above cannot be processed properly
  if (shapes_tmp.size() > 0 && !is_blank(shapes_tmp))
  {
    throw std::invalid_argument("The part of input shapes: " + shapes_tmp + " cannot be processed");
  }

  return result_shapes;
}
