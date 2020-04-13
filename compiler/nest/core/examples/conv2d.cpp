/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <nest/Module.h>

int main(int, char **)
{
  // This example shows how to specify convolution with IFM(1x3x3) and Kernel(1x1x3x3) with nest
  // - STRIDE is 1, and there is no padding
  //
  // The below code corresponds to the following nest DSL code:
  // ----------------------------------------------------------------------------------------------
  // Domain ofm(1, 1, 1)
  // Domain ifm(1, 3, 3)
  // Domain ker(1, 1, 3, 3)
  //
  // Var ofm_ch  : { min = 0, max = 1 }
  // Var ofm_row : { min = 0, max = 1 }
  // Var ofm_col : { min = 0, max = 1 }
  // Var ker_ch  : { min = 0, max = 1 }
  // Var ker_row : { min = 0, max = 3 }
  // Var ker_col : { min = 0, max = 3 }
  //
  // PUSH ifm(ker_ch, ker_row, ker_col) * ker(ofm_ch, ker_ch, ofm_row + ker_row, ofm_col + ker_col)
  // RET  ofm(ofm_ch, ofm_row, ofm_col)
  // ----------------------------------------------------------------------------------------------
  //
  // The first part declares Domain(s) which corresponds to a multi-dimensional array in C-style
  // (without type). For example, 'Domain ofm(1, 3, 3)' corresponds to the
  // following C array declaration.
  //   float ofm[1][3][3];
  // (Here we assume that domain type is 'float')
  //
  // The second part declares Var(s) which serves as a loop iteration variable. Basically, each
  // variable emits one for loop and these loops are nested. As there are 6 variables in the above
  // example, there will be 6 nested-loops.
  //
  // Each variable has a corresponding bound, and the bound of each variable states the starting /
  // termination condition. For example, 'Var ofm_ch : { min = 0, max = 1 }' will introduce the
  // following for loop:
  // ----------------------------------------------------------------------------------------------
  //   for (int ofm_ch = 0; ofm_ch < 1; ++ofm_ch) { ... }
  // ----------------------------------------------------------------------------------------------
  //
  // The last part declares statement(s) which state the computation performed inside these nested
  // loops. Nest is stack-based. There is a virtual stack inside nested loop, and the evaluation of
  // each statement will update this stack.
  //
  // Each nest code has one return statement (RET). This return statement specifies where to write
  // the computed result.
  //
  // PUSH 'expr' statement evaluates an arithmetic expression (specified by 'expr') and pushes the
  // numeric result to the stack. When PUSH statement evaluates an arithmetic expression, variables
  // that do not appear in RET statement are treated as reduction variables. For example,
  // ker_ch, ker_row, and ker_col do not appear in RET statement. So, PUSH '...' statement in the
  // above example corresponds to the following nested loops:
  // ----------------------------------------------------------------------------------------------
  // float value = 0.0f;
  //
  // for (int ker_ch = 0; ker_ch < 1; ++ker_ch) {
  //   for (int ker_row = 0; ker_row < 3; ++ker_row) {
  //     for (int ker_col = 0; ker_col < 3; ++ker_col) {
  //       float ifm_value = ifm[ker_ch][ofm_row + ker_row][ofm_col + ker_col];
  //       float ker_value = ker[ofm_ch][ker_ch][ker_row][ker_col];
  //       value += ifm_value * ker_value;
  //    }
  //  }
  // }
  // ----------------------------------------------------------------------------------------------
  //
  // In summary, the above nest example corresponds to the following 2D convolution:
  // ----------------------------------------------------------------------------------------------
  // float ofm[1][1][1];
  // float ifm[1][3][3];
  // float ker[1][1][3][3];
  //
  // for (int ofm_ch = 0; ofm_ch < 1; ++ofm_ch) {
  //   for (int ofm_row = 0; ofm_row < 1; ++ofm_row) {
  //     for (int ofm_col = 0; ofm_col < 1; ++ofm_col) {
  //       float value = 0.0f;
  //
  //       for (int ker_ch = 0; ker_ch < 1; ++ker_ch) {
  //         for (int ker_row = 0; ker_row < 3; ++ker_row) {
  //           for (int ker_col = 0; ker_col < 3; ++ker_col) {
  //             float ifm_value = ifm[ker_ch][ofm_row + ker_row][ofm_col + ker_col];
  //             float ker_value = ker[ofm_ch][ker_ch][ker_row][ker_col];
  //             value += ifm_value * ker_value;
  //           }
  //         }
  //       }
  //
  //       ofm[ofm_ch][ofm_col][ofm_row] = value;
  //     }
  //   }
  // }
  // ----------------------------------------------------------------------------------------------
  //
  nest::Module m;

  //
  // Domains
  //
  auto ofm = m.domain().make({1 /*C*/, 1 /*H*/, 1 /*W*/});
  auto ifm = m.domain().make({1 /*C*/, 3 /*H*/, 3 /*W*/});
  auto ker = m.domain().make({1 /*N*/, 1 /*C*/, 3 /*H*/, 3 /*W*/});

  //
  // Variables
  //
  auto ofm_ch = m.var().make();
  auto ofm_row = m.var().make();
  auto ofm_col = m.var().make();

  auto ker_ch = m.var().make();
  auto ker_row = m.var().make();
  auto ker_col = m.var().make();

  // Declare the bound of each variables
  using nest::Bound;

  m.var().bound(ofm_ch) = Bound{0, 1};
  m.var().bound(ofm_row) = Bound{0, 1};
  m.var().bound(ofm_col) = Bound{0, 1};

  m.var().bound(ker_ch) = Bound{0, 1};
  m.var().bound(ker_row) = Bound{0, 3};
  m.var().bound(ker_col) = Bound{0, 3};

  //
  // Statement
  //
  auto ifm_value = ifm(ker_ch, ofm_row + ker_row, ofm_col + ker_col);
  auto ker_value = ker(ofm_ch, ker_ch, ker_row, ker_col);

  m.push(ifm_value * ker_value);
  m.ret(ofm(ofm_ch, ofm_row, ofm_col));

  return 0;
}
