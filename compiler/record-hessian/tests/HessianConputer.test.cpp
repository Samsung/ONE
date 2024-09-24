#include "record-hessian/HessianComputer.h"
#include <gtest/gtest.h>
#include <vector>
#include <luci/IR/CircleNode.h>
#include <luci_interpreter/Interpreter.h>

using namespace record_hessian;

TEST(HessianComputerTest, recordHessianValidInput)
{

    luci::CircleFullyConnected node;

    std::vector<float> input_data = {1.0, 2.0, 3.0, 4.0};

    luci_interpreter::DataType data_type = luci_interpreter::DataType::FLOAT32;
    luci_interpreter::Shape shape({1, 4}); 
    luci_interpreter::AffineQuantization quantization;
    quantization.scale = {1.0};
    quantization.zero_point = {0};

    std::string tensor_name = "input_tensor";

    luci_interpreter::Tensor input_tensor(data_type, shape, quantization, tensor_name);

    size_t data_size = input_data.size() * sizeof(float);
    std::vector<uint8_t> buffer(data_size);  

    input_tensor.set_data_buffer(buffer.data());
    input_tensor.writeData(input_data.data(), data_size);

    HessianComputer computer;

    EXPECT_NO_THROW(computer.recordHessian(&node, &input_tensor));
}

TEST(HessianComputerTest, recordHessianValidInput_NEG)
{
    luci::CircleAdd node;

    std::vector<float> input_data = {1.0, 2.0, 3.0, 4.0};
    
    luci_interpreter::DataType data_type = luci_interpreter::DataType::FLOAT32;
    luci_interpreter::Shape shape({1, 2, 2, 1});
    luci_interpreter::AffineQuantization quantization;
    quantization.scale = {1.0};
    quantization.zero_point = {0};

    std::string tensor_name = "input_tensor";

    luci_interpreter::Tensor input_tensor(data_type, shape, quantization, tensor_name);

    size_t data_size = input_data.size() * sizeof(float);
    std::vector<uint8_t> buffer(data_size);

    input_tensor.set_data_buffer(buffer.data());
    input_tensor.writeData(input_data.data(), data_size);

    HessianComputer computer;

    EXPECT_ANY_THROW(computer.recordHessian(&node, &input_tensor));
}

TEST(HessianComputerTest, recordHessianNullTensor_NEG)
{
    luci::CircleAdd node;
    HessianComputer computer;
    EXPECT_ANY_THROW(computer.recordHessian(&node, nullptr));
}

TEST(HessianComputerTest, unfoldValidInput)
{
    std::vector<float> buf = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    uint32_t input_n = 1, input_h = 2, input_w = 2, input_c = 2;
    uint32_t stride_h = 1, stride_w = 1, dilation_h = 1, dilation_w = 1;
    uint32_t kernel_oc = 1, kernel_h = 2, kernel_w = 2, kernel_ic = 2;

    HessianComputer computer;
    computer.unfold(buf, input_n, input_h, input_w, input_c, stride_h, stride_w, dilation_h, dilation_w, kernel_oc, kernel_h, kernel_w, kernel_ic);
    std::vector<float> expected_output = {1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0};

    EXPECT_EQ(buf, expected_output);
}

TEST(HessianComputerTest, unfoldInvalidInput_NEG)
{
    std::vector<float> buf = {1.0, 2.0, 3.0, 4.0};
    uint32_t input_n = 1, input_h = 2, input_w = 2, input_c = 1;
    uint32_t stride_h = 1, stride_w = 1, dilation_h = 1, dilation_w = 1;
    uint32_t kernel_oc = 1, kernel_h = 2, kernel_w = 2, kernel_ic = 2;

    HessianComputer computer;
    EXPECT_ANY_THROW(computer.unfold(buf, input_n, input_h, input_w, input_c, stride_h, stride_w, dilation_h, dilation_w, kernel_oc, kernel_h, kernel_w, kernel_ic));
}