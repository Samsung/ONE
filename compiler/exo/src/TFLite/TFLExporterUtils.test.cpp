/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "TFLExporterUtils.h"

#include <gtest/gtest.h>

using namespace exo::tflite_detail;

TEST(ExporterUtilsTests, getOpPadding)
{
  loco::Padding2D pad;
  loco::Stride<2> stride;
  exo::ShapeDescription ifm;
  exo::ShapeDescription ofm;

  ifm._dims.resize(4);
  ofm._dims.resize(4);

  // VALID padding
  {
    pad.top(0);
    pad.bottom(0);
    pad.left(0);
    pad.right(0);

    stride.vertical(2);
    stride.horizontal(2);

    ifm._dims[1] = 5;
    ifm._dims[2] = 5;

    ofm._dims[1] = 2;
    ofm._dims[2] = 2;

    ASSERT_EQ(getOpPadding(&pad, &stride, ifm, ofm), tflite::Padding_VALID);
  }

  // SAME padding
  {
    pad.top(1);
    pad.bottom(1);
    pad.left(1);
    pad.right(1);

    stride.vertical(2);
    stride.horizontal(2);

    ifm._dims[1] = 5;
    ifm._dims[2] = 5;

    ofm._dims[1] = 3;
    ofm._dims[2] = 3;

    ASSERT_EQ(getOpPadding(&pad, &stride, ifm, ofm), tflite::Padding_SAME);
  }

  // Custom padding 1 - Not supported by tflite
  {
    pad.top(2);
    pad.bottom(0);
    pad.left(1);
    pad.right(1);

    stride.vertical(2);
    stride.horizontal(2);

    ifm._dims[1] = 5;
    ifm._dims[2] = 5;

    ofm._dims[1] = 3;
    ofm._dims[2] = 3;

    ASSERT_ANY_THROW(getOpPadding(&pad, &stride, ifm, ofm));
  }

  // Custom padding 2 - Not supported by tflite
  {
    pad.top(2);
    pad.bottom(2);
    pad.left(2);
    pad.right(2);

    stride.vertical(2);
    stride.horizontal(2);

    ifm._dims[1] = 5;
    ifm._dims[2] = 5;

    ofm._dims[1] = 4;
    ofm._dims[2] = 4;

    ASSERT_ANY_THROW(getOpPadding(&pad, &stride, ifm, ofm));
  }
}
