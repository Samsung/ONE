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

#ifndef _MIR_QUANTIZATION_H_
#define _MIR_QUANTIZATION_H_

namespace mir
{

class AffineQuantization
{
public:
  AffineQuantization() = default;

  AffineQuantization(float scale, int zero_point)
    : _scale(scale), _zero_point(zero_point), _empty(false)
  {
  }

  float getScale() const { return _scale; }

  int getZeroPoint() const { return _zero_point; }

  bool empty() const { return _empty; }

private:
  float _scale = 0.f;
  int _zero_point = 0;
  bool _empty = true;
};

} // namespace mir

#endif //_MIR_QUANTIZATION_H_
