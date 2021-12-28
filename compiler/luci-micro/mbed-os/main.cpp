/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "mbed.h"
#undef ARG_MAX
#define LUCI_LOG 0
#include <luci_interpreter/Interpreter.h>
#include <luci_interpreter/StaticMemoryManager.h>
#include <luci/Importer.h>
#include <luci/IR/Module.h>
#include <loco/IR/DataTypeTraits.h>
#include <circlemodel.h>
#include <cstdlib>
#include <iostream>
#include <luci/Log.h>
#include "ov7670.h"
#include <stm32h7xx_hal.h>

static DCMI_HandleTypeDef hdcmi;
volatile uint8_t OV7670_buffer[IMG_ROWS * IMG_COLUMNS];
static I2C i2c(PB_7, PB_6);

void OV7670_update(void){
  HAL_DCMI_Start_DMA(&hdcmi, DCMI_MODE_SNAPSHOT, (uint32_t)&OV7670_buffer, IMG_ROWS * IMG_COLUMNS * 2 / 4);
}

static void MX_DCMI_Init(void)
{

  /* USER CODE BEGIN DCMI_Init 0 */

  /* USER CODE END DCMI_Init 0 */

  /* USER CODE BEGIN DCMI_Init 1 */

  /* USER CODE END DCMI_Init 1 */
  hdcmi.Instance = DCMI;
  hdcmi.Init.SynchroMode = DCMI_SYNCHRO_HARDWARE;
  hdcmi.Init.PCKPolarity = DCMI_PCKPOLARITY_RISING;
  hdcmi.Init.VSPolarity = DCMI_VSPOLARITY_HIGH;
  hdcmi.Init.HSPolarity = DCMI_HSPOLARITY_LOW;
  hdcmi.Init.CaptureRate = DCMI_CR_ALL_FRAME;
  hdcmi.Init.ExtendedDataMode = DCMI_EXTEND_DATA_8B;
  hdcmi.Init.JPEGMode = DCMI_JPEG_DISABLE;
  hdcmi.Init.ByteSelectMode = DCMI_BSM_ALL;
  hdcmi.Init.ByteSelectStart = DCMI_OEBS_ODD;
  hdcmi.Init.LineSelectMode = DCMI_LSM_ALL;
  hdcmi.Init.LineSelectStart = DCMI_OELS_ODD;
  if (HAL_DCMI_Init(&hdcmi) != HAL_OK)
  {
    std::cout << "Failed to init DCMI\r\n";
  }
  /* USER CODE BEGIN DCMI_Init 2 */

  /* USER CODE END DCMI_Init 2 */

}

void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Supply configuration update enable
  */
  HAL_PWREx_ConfigSupply(PWR_LDO_SUPPLY);
  /** Configure the main internal regulator output voltage
  */
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);

  while(!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}
  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_DIV1;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_NONE;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    std::cout << "Failed to init RCC\r\n";
  }
  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                                |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2
                                |RCC_CLOCKTYPE_D3PCLK1|RCC_CLOCKTYPE_D1PCLK1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_HSI;
  RCC_ClkInitStruct.SYSCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB3CLKDivider = RCC_APB3_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV1;
  RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_1) != HAL_OK)
  {
    std::cout << "Failed to configure FLASH_LATENCY\r\n";
  }
}

static void MX_DMA_Init(void)
{

  /* DMA controller clock enable */
  __HAL_RCC_DMA2_CLK_ENABLE();

  /* DMA interrupt init */
  /* DMA2_Stream1_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA2_Stream1_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA2_Stream1_IRQn);
  /* DMAMUX1_OVR_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMAMUX1_OVR_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMAMUX1_OVR_IRQn);

}

void fill_in_tensor(std::vector<char> &data, loco::DataType dtype)
{
  switch (dtype)
  {
    case loco::DataType::FLOAT32:
      for (int i = 0; i < data.size() / sizeof(float); ++i)
      {
        reinterpret_cast<float *>(data.data())[i] = 123.f;
      }
      break;
    case loco::DataType::S8:
      for (int i = 0; i < data.size() / sizeof(int8_t); ++i)
      {
        reinterpret_cast<int8_t *>(data.data())[i] = 123;
      }
      break;
    case loco::DataType::U8:
      for (int i = 0; i < data.size() / sizeof(uint8_t); ++i)
      {
        reinterpret_cast<uint8_t *>(data.data())[i] = 123;
      }
      break;
    default:
      assert(false);
  }
}
int main()
{
  SystemClock_Config();
  pin_mode(PB_7, OpenDrainPullUp);
  pin_mode(PB_6, OpenDrainPullUp);
  MX_DCMI_Init();
  MX_DMA_Init();
  for (int i = 0; i < OV7670_REG_NUM; ++i) {
    auto err = i2c.write(OV7670_WRITE_ADDR, &(OV7670_reg[i][0]), 2);

    std::cout << "Writing register: " << i << "\r\n";
    if (err == true) {
      std::cout << "Failed to update register\r\n";
      break;
    }
    ThisThread::sleep_for(50ms);
  }
  OV7670_update();

  setenv("ONE_HERMES_COLOR", "ON", 1);
  // Verify flatbuffers
  flatbuffers::Verifier verifier{reinterpret_cast<const uint8_t *>(circle_model_raw), sizeof(circle_model_raw) / sizeof(circle_model_raw[0])};

  std::cout << "circle::VerifyModelBuffer\n";
  if (!circle::VerifyModelBuffer(verifier))
  {
    std::cout << "ERROR: Failed to verify circle\n";
  }
  std::cout << "OK\n";
  std::cout << "circle::GetModel(circle_model_raw)\n";
  auto model = circle::GetModel(circle_model_raw);
  std::cout << "luci::Importer().importModule\n";
  auto module = luci::Importer().importModule(model);
  std::cout << "OK\n";
  std::cout << "std::make_unique<luci_interpreter::Interpreter>(module.get())\n";
  auto interpreter = std::make_unique<luci_interpreter::Interpreter>(module.get());

  std::cout << "OK\n";
  auto nodes = module->graph()->nodes();
  auto nodes_count = nodes->size();
  // Fill input tensors with some garbage
  while (true)
  {
    Timer t;
    for (int i = 0; i < nodes_count; ++i)
    {
      auto *node = dynamic_cast<luci::CircleNode *>(nodes->at(i));
      assert(node);
      if (node->opcode() == luci::CircleOpcode::CIRCLEINPUT)
      {
        auto *input_node = static_cast<luci::CircleInput *>(node);
        loco::GraphInput *g_input = module->graph()->inputs()->at(input_node->index());
        const loco::TensorShape *shape = g_input->shape();
        size_t data_size = 1;
        for (int d = 0; d < shape->rank(); ++d)
        {
          assert(shape->dim(d).known());
          data_size *= shape->dim(d).value();
        }
        data_size *= loco::size(g_input->dtype());
        std::vector<char> data(data_size);
        fill_in_tensor(data, g_input->dtype());

        interpreter->writeInputTensor(static_cast<luci::CircleInput *>(node), data.data(),
                                      data_size);
      }
    }
    t.start();

    interpreter->interpret();
    t.stop();
    std::cout << "\rFinished in " << t.read_us();
    ThisThread::sleep_for(10);
  }
}
