/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright (C) 2017 The Android Open Source Project
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

#ifndef __VALIDATION_H__
#define __VALIDATION_H__

#include "OperationType.h"
#include "Model.h"
#include "Request.h"
#include "NeuralNetworks.h"

int validateOperationType(const OperationType &);
int validateOperandType(const ANeuralNetworksOperandType &type, const char *tag, bool allowPartial);
int validateOperandList(uint32_t count, const uint32_t *list, uint32_t operandCount,
                        const char *tag);

bool validateModel(const Model &model);
bool validateRequest(const Request &request, const Model &model);

#endif // __VALIDATION_H__
