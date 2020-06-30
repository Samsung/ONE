// This file is included from AddOpsAndParams defined in nnapi_delegate.cc
// and contains lambda for extened implementation to original Tensorflow Lite.
    auto add_scalar_bool8 = [&nn_model, &augmented_inputs,
                             &next_id](bool value) {
      // Fix to use strict build option
      int8_t casted_value = (value ? 1 : 0);
      ANeuralNetworksOperandType operand_type{}; operand_type.type = ANEURALNETWORKS_BOOL;
      CHECK_NN(ANeuralNetworksModel_addOperand(nn_model, &operand_type))
      CHECK_NN(ANeuralNetworksModel_setOperandValue(nn_model, next_id, &casted_value,
                                                    sizeof(int8_t)))
      augmented_inputs.push_back(next_id++);
    };

    auto add_resize_bilinear_params = [&add_scalar_int32, &subgraph, &augmented_inputs](void* data) {
      auto builtin = reinterpret_cast<TfLiteResizeBilinearParams*>(data);
      if  (builtin->align_corners) {
        FATAL("Resize bilinear does not support align corners in NNAPI");
      }

      TfLiteTensor* tensor = subgraph->tensor(augmented_inputs.back());
      assert(tensor->type == kTfLiteInt32);
      assert(tensor->bytes == sizeof(int)*2);
      augmented_inputs.pop_back();

      int height = ((int*)(tensor->data.raw))[1];
      int width = ((int*)(tensor->data.raw))[0];
      add_scalar_int32(height);
      add_scalar_int32(width);
    };

    auto add_transpose_conv_params = [&add_scalar_int32](void* data) {
      auto builtin = reinterpret_cast<TfLiteTransposeConvParams*>(data);
      add_scalar_int32(builtin->padding);
      add_scalar_int32(builtin->stride_width);
      add_scalar_int32(builtin->stride_height);
    };

    auto add_lrn_params = [&add_scalar_int32,
                            &add_scalar_float32](void* data) {
      auto builtin = reinterpret_cast<TfLiteLocalResponseNormParams*>(data);
      add_scalar_int32(builtin->radius);
      add_scalar_float32(builtin->bias);
      add_scalar_float32(builtin->alpha);
      add_scalar_float32(builtin->beta);
    };

    auto add_strided_slice_params = [&add_scalar_int32](void* data) {
      auto builtin = reinterpret_cast<TfLiteStridedSliceParams*>(data);
      add_scalar_int32(builtin->begin_mask);
      add_scalar_int32(builtin->end_mask);
      // ellipsis_mask and new_axis_mask are not supported on nn runtime
      // cf) tflite interpreter supports both operations
      if (builtin->ellipsis_mask) {
        FATAL("STRIDE_SLICE does not support ellipsis_mask in NNAPI");
      }
      if (builtin->new_axis_mask) {
        FATAL("STRIDE_SLICE does not support new_axis_mask in NNAPI");
      }
      add_scalar_int32(builtin->shrink_axis_mask);
    };

    auto add_gather_params = [&add_scalar_int32, &augmented_inputs](void* data) {
      auto builtin = reinterpret_cast<TfLiteGatherParams*>(data);
      if (builtin->axis != 0) {
        FATAL("GATHER does not support axis>0 in NNAPI");
      }

      auto indices_index = augmented_inputs.back();
      augmented_inputs.pop_back();
      add_scalar_int32(builtin->axis);
      augmented_inputs.push_back(indices_index);
    };

    auto add_pack_ex_params = [&add_scalar_int32](void* data) {
      auto builtin = reinterpret_cast<TfLitePackParams*>(data);
      add_scalar_int32(builtin->values_count);
      add_scalar_int32(builtin->axis);
    };

    auto add_unpack_ex_params = [&add_scalar_int32](void* data) {
      auto builtin = reinterpret_cast<TfLiteUnpackParams*>(data);
      add_scalar_int32(builtin->num);
      add_scalar_int32(builtin->axis);
    };

    auto check_batch_to_space_params = [subgraph, &node, &augmented_inputs]() {

    //If there are 3 inputs, check if crops is having default values {0, 0, 0, 0}
    //Else unsupported by NNAPI

      if(augmented_inputs.size() == 3)
      {
        const uint32_t crops_buffer_index = node.inputs->data[2];
        const TfLiteTensor* crops = subgraph->tensor(crops_buffer_index);
        const int *crops_value = crops->data.i32;

        //Check if crops is having default values {0, 0, 0, 0}
        if(crops_value[0] != 0 || crops_value[1] != 0 || crops_value[2] != 0 || crops_value[3] != 0)
        {
          FATAL("BATCH_TO_SPACE_ND does not support Explicit crops in NNAPI");
        }
        else
        {
          //Restrict crops input and pass only other two inputs
          augmented_inputs.pop_back();
        }
      }
    };

    auto add_split_params = [&add_scalar_int32, &augmented_inputs](void* data) {
      // swap 1st and 2nd operand order
      auto input_tensor = augmented_inputs[1];
      auto axis = augmented_inputs[0];
      augmented_inputs[0] = input_tensor;
      augmented_inputs[1] = axis;

      auto builtin = reinterpret_cast<TfLiteSplitParams*>(data);
      add_scalar_int32(builtin->num_splits);
    };

    auto check_arg_max_input = [&subgraph, &augmented_inputs](void *data) {
      auto params = reinterpret_cast<TfLiteArgMaxParams*>(data);
      if (params->output_type != kTfLiteInt32)
      {
        FATAL("Cannot handle output type in NNAPI");
      }

      TfLiteTensor* axis_tensor = subgraph->tensor(augmented_inputs.back());
      assert(axis_tensor->type == kTfLiteInt32);

      int64_t count = 1;
      for (int i = 0; i < axis_tensor->dims->size; ++i) {
        count *= axis_tensor->dims->data[i];
      }
      assert(count == 1);
    };

    auto add_reducer_params = [&add_scalar_bool8](void* data) {
      auto builtin = reinterpret_cast<TfLiteReducerParams*>(data);
      if (builtin == nullptr)
      {
        add_scalar_bool8(0);
      }
      else
      {
        add_scalar_bool8(builtin->keep_dims);
      }
    };

    auto add_one_hot_params = [&add_scalar_int32](void* data) {
      const auto* builtin = reinterpret_cast<TfLiteOneHotParams*>(data);
      add_scalar_int32(builtin->axis);
    };
