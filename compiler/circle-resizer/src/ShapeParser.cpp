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

Shape parse_shape(std::string shape_str)
{
  Shape result_shape;
  std::stringstream shape_stream(shape_str);
  std::string token;
  try
  {
    while (std::getline(shape_stream, token, ','))
    {
      result_shape.push_back(Dim{std::stoi(token)});
    }
  }
  catch (...)
  {
    throw std::invalid_argument("Error during shape processing: " + shape_str);
  }
  if (result_shape.empty())
  {
    throw std::invalid_argument("No shapes found in input string: " + shape_str);
  }
  return result_shape;
}
} // namespace

std::vector<Shape> circle_resizer::parse_shapes(std::string shapes_str)
{
  std::vector<Shape> result_shapes;
  std::stringstream shapes_stream(shapes_str);
  std::string token;
  size_t begin_pos = 0, end_pos = 0;
  while ((begin_pos = shapes_str.find("[")) != std::string::npos &&
         (end_pos = shapes_str.find("]")) != std::string::npos)
  {
    token = shapes_str.substr(begin_pos + 1, end_pos);
    result_shapes.push_back(parse_shape(token));
    shapes_str.erase(0, end_pos + 1);
  }

  if (result_shapes.empty())
  {
    throw std::invalid_argument("No shapes found in input string: " + shapes_str);
  }

  // the rest of input not processed by loop above cannot be processed properly
  if (shapes_str.size() > 0 && !is_blank(shapes_str))
  {
    throw std::invalid_argument("The part of input shape: " + shapes_str + " cannot be processed");
  }

  return result_shapes;
}
