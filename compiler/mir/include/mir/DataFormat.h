/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef _MIR_DATA_FORMAT_H_
#define _MIR_DATA_FORMAT_H_

#include <cassert>
#include <string>

namespace mir
{

enum class DataFormat
{
  NCHW,
  NHWC
};

inline int getDataBatchDimIndex(DataFormat data_format)
{
  switch (data_format)
  {
    case DataFormat::NCHW:
    case DataFormat::NHWC:
      return 0;
    default:
      assert(false);
      return -1; // Dummy value to silence compiler warning.
  }
}

inline int getDataChannelDimIndex(DataFormat data_format)
{
  switch (data_format)
  {
    case DataFormat::NCHW:
      return 1;
    case DataFormat::NHWC:
      return 3;
    default:
      assert(false);
      return -1; // Dummy value to silene compiler warning.
  }
}

inline int getDataSpatialDimIndex(DataFormat data_format, int dim)
{
  assert(dim >= 0 && dim <= 1);
  switch (data_format)
  {
    case DataFormat::NCHW:
      return 2 + dim;
    case DataFormat::NHWC:
      return 1 + dim;
    default:
      assert(false);
      return -1; // Dummy value to silence compiler warning.
  }
}

inline std::string toString(DataFormat data_format)
{
  switch (data_format)
  {
    case DataFormat::NCHW:
      return "NCHW";
    case DataFormat::NHWC:
      return "NHWC";
    default:
      assert(false);
      return ""; // Dummy value to silence compiler warning.
  }
}

} // namespace mir

#endif //_MIR_DATA_FORMAT_H_
