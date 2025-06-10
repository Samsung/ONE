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

/**
 * @file  NNAPIConvert.h
 * @brief This file contains convereter(s)\n
 *        from NNAPI frontend's struct to onert's internal struct
 */
#ifndef __ONERT_NNAPI_CONVERT_H__
#define __ONERT_NNAPI_CONVERT_H__

#include <NeuralNetworks.h>

#include <ir/TypeInfo.h>
#include <ir/Shape.h>
#include <ir/Padding.h>
#include <ir/InternalType.h>

class NNAPIConvert
{

public:
  /**
   * @brief     Convert data type from NNAPI to internal data type
   * @param[in] type  NNAPI's data type
   * @return    onert's internal data type
   */
  static onert::ir::DataType getDataType(OperandCode type);

  /**
   * @brief     Convert operand type info from NNAPI to interanl operand type info
   * @param[in] type  NNAPI's operand type
   * @return    onert's internal operand type info
   */
  static onert::ir::TypeInfo getTypeInfo(const ANeuralNetworksOperandType *type);

  /**
   * @brief     Convert operand shape info from NNAPI to internal operand shape
   * @param[in] type  NNAPI's operand type
   * @return    onert's internal operand shape
   */
  static onert::ir::Shape getShape(const ANeuralNetworksOperandType *type);

  /**
   * @brief     Calcaulate operand size from NNAPI type
   * @param[in] type  NNAPI's operand type
   * @return    Operand size
   */
  static size_t calculateSizeFromType(const ANeuralNetworksOperandType *type);

  /**
   * @brief     Convert NNAPI FuseCode to internal activation type
   * @param[in] act NNAPI's FuseCode type
   * @return    onert's internal activation type
   */
  static onert::ir::Activation getFusedActivation(FuseCode act);

  /**
   * @brief     Convert NNAPI PaddingCode to internal padding type
   * @param[in] type NNAPI's PaddingCode type
   * @return    onert's internal padding type
   */
  static onert::ir::PaddingType getPaddingType(PaddingCode type);
};

#endif // __ONERT_NNAPI_CONVERT_H__
