/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NNFW_EXPERIMENTAL_TRAIN_H__
#define __NNFW_EXPERIMENTAL_TRAIN_H__

#include "nnfw.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Training session to query with runtime
 */
typedef struct nnfw_train_session nnfw_train_session;

/**
 *  Training C APIs
 *
 * Training APIs are designed to be used in the following order for training
 * 1. nnfw_prepare_train
 * 2. nnfw_train
 *
 * If you want to inference with the same session, you can use the following API
 * - nnfw_inference
 *
 */

//////////////////////////////////////////////
// Essential APIs for training
//////////////////////////////////////////////

/**
 * @brief Training information to prepare training
 */
typedef struct nnfw_train_inf
{
  /** Learning rate */
  float learning_rate = 0.001;
  /** Batch size */
  uint32_t batch_size = 1;
} nnfw_train_inf;

NNFW_STATUS nnfw_create_train_session(nnfw_train_session **session);

/**
 * @brief Prepare session to be ready for training
 *
 * @note  If we can get training information from model, info may be deprecated.
 *        Or we will change type to pointer and overwrite training information if not null.
 *        (Policy is not decided yet)
 *
 * @param[in] session The session to be prepared for training
 * @param[in] info    Training information.
 *                    Default value is {learning_rate: 0.001, batch_size: 1}
 * @return  @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_prepare_train(nnfw_train_session *session, nnfw_train_inf info);

/**
 * @brief Train the model
 *
 * @param[in]   session         The session to train the model
 * @param[in]   inputs          The input buffers for training
 * @param[in]   input_infos     The shape and type of input buffers
 * @param[in]   expecteds       The expected buffers for training
 * @param[in]   expected_infos  The shape and type of expected buffers
 * @param[out]  loss            The loss value after training
 * @return @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_train(nnfw_train_session *session, void **inputs, nnfw_tensorinfo *input_infos,
                       void **expecteds, nnfw_tensorinfo *expected_infos, float *loss);

/**
 * @brief Inference with the trained model
 *
 * @param[in] session       The session to inference with the trained model
 * @param[in] inputs        The input buffers for inference
 * @param[in] input_infos   The shape and type of input buffers
 * @param[in] outputs       The output buffers for inference
 * @param[in] output_infos  The shape and type of output buffers
 * @return @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_inference(nnfw_train_session *session, void **inputs, nnfw_tensorinfo *input_infos,
                           void **outputs, nnfw_tensorinfo *output_infos);

//////////////////////////////////////////////
// Optional APIs for training
//////////////////////////////////////////////

/**
 * @brief Get the number of training model inputs
 *        This function should be called after {@link nnfw_prepare_train}
 *
 * @param[in]   session The session to get the number of training model inputs
 * @param[out]  count   The number of training model inputs
 * @return @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_get_input_count(nnfw_train_session *session, uint32_t *count);

/**
 * @brief Get the training model input information
 *
 * @param[in]   session The session to get the training model input information
 * @param[in]   index   The index of training model input
 * @param[out]  info    The shape and type of training model input
 * @return @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_get_input_tensorinfo(nnfw_train_session *session, uint32_t index,
                                      nnfw_tensorinfo *info);

/**
 * @brief Get the number of training model expected outputs
 *        This function should be called after {@link nnfw_prepare_train}
 *
 * @param[in]   session The session to get the number of training model expected outputs
 * @param[out]  count   The number of training model expected outputs
 */
NNFW_STATUS nnfw_get_expected_count(nnfw_train_session *session, uint32_t *count);

/**
 * @brief Get the training model expected output information
 *
 * @param[in]   session The session to get the training model expected output information
 * @param[in]   index   The index of training model expected output
 * @param[out]  info    The shape and type of training model expected output
 * @return @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_get_expected_tensorinfo(nnfw_train_session *session, uint32_t index,
                                         nnfw_tensorinfo *info);

/**
 * @brief Export inference model
 *        This function should be called after {@link nnfw_train}
 *
 * @param[in] session The session to export inference model
 * @param[in] path    The path to export inference model
 * @return @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_export_inference_model(nnfw_train_session *session, const char *path);

/**
 * @brief Convert training session to inference session
 *        This function will be used to use inference session without exporting inference model
 *
 * @param[in]   train_session     The training session to be converted
 * @param[out]  inference_session The inference session converted from training session
 * @return @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_convert_to_inference_session(nnfw_train_session *train_session,
                                              nnfw_session **inference_session);

/**
 * @brief Set training information before prepare training
 * @note  This function may be used after {@link nnfw_prepare_pipeline}
 *        to support dynamic training information (not planned)
 *
 * @param[in] session The session to be prepared for training
 * @param[in] info    Training information
 * @return  @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_set_traininfo(nnfw_train_session *session, nnfw_train_inf info);

#ifdef __cplusplus
}
#endif

#endif // __NNFW_EXPERIMENTAL_TRAIN_H__
