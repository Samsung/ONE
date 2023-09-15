#include <pybind11.h>
#include <stl.h>
#include "nnfw.h" // 주어진 코드에 맞게 헤더 파일 경로를 수정해야 합니다.
#include "nnfw_api_internal.h"

namespace py = pybind11;

uint64_t num_elems(const nnfw_tensorinfo *ti)
{
    uint64_t n = 1;
    for (uint32_t i = 0; i < ti->rank; ++i)
    {
        n *= ti->dims[i];
    }
    return n;
}

class NNFW_SESSION
{

private:
    nnfw_session *session;
    nnfw_tensorinfo ti;
    std::vector<float> input;
    std::vector<float> output;

public:
    NNFW_STATUS create()
    {
        return nnfw_create_session(&session);
    }
    NNFW_STATUS close()
    {
        NNFW_STATUS state = nnfw_close_session(session);
        session = nullptr;
        return state;
    }
    NNFW_STATUS await()
    {
        return nnfw_await(session);
    }
    NNFW_STATUS apply_tensorinfo(uint32_t index, const nnfw_tensorinfo &tensor_info)
    {
        return nnfw_apply_tensorinfo(session, index, tensor_info);
    }
    NNFW_STATUS input_size(uint32_t *number)
    {
        return nnfw_input_size(session, number);
    }
    NNFW_STATUS load_model_from_file(const char *package_file_path)
    {
        return nnfw_load_model_from_file(session, package_file_path);
    }
    NNFW_STATUS output_size(uint32_t *number)
    {
        return nnfw_output_size(session, number);
    }
    NNFW_STATUS output_tensorinfo(uint32_t index)
    {
        return nnfw_output_tensorinfo(session, index, &ti);
    }
    NNFW_STATUS prepare()
    {
        return nnfw_prepare(session);
    }
    NNFW_STATUS query_info_u32(NNFW_INFO_ID id, uint32_t *val)
    {
        return nnfw_query_info_u32(session, id, val);
    }
    NNFW_STATUS run()
    {
        return nnfw_run(session);
    }
    NNFW_STATUS run_async()
    {
        return nnfw_run_async(session);
    }
    NNFW_STATUS set_available_backends(const char *backends)
    {
        return nnfw_set_available_backends(session, backends);
    }
    NNFW_STATUS set_input(uint32_t index)
    {
        NNFW_SESSION::set_input_tensorinfo(index);
        uint32_t input_elements = num_elems(&ti);
        input.resize(input_elements);
        return nnfw_set_input(session, index, ti.dtype, input.data(), sizeof(float) * input_elements);
    }
    NNFW_STATUS set_input_layout(uint32_t index, NNFW_LAYOUT layout)
    {
        return nnfw_set_input_layout(session, index, layout);
    }
    NNFW_STATUS set_input_tensorinfo(uint32_t index)
    {
        return nnfw_set_input_tensorinfo(session, index, &ti);
    }
    NNFW_STATUS set_op_backend(const char *op, const char *backend)
    {
        return nnfw_set_op_backend(session, op, backend);
    }
    NNFW_STATUS set_output(uint32_t index)
    {
        NNFW_SESSION::output_tensorinfo(index);
        uint32_t output_elements = num_elems(&ti);
        output.resize(output_elements);
        return nnfw_set_output(session, index, ti.dtype, output.data(), sizeof(float) * output_elements);
    }
    NNFW_STATUS set_output_layout(uint32_t index, NNFW_LAYOUT layout)
    {
        return nnfw_set_output_layout(session, index, layout);
    }
    NNFW_STATUS input_tensorinfo(uint32_t index)
    {
        return nnfw_input_tensorinfo(session, index, &ti);
    }
    void print_session()
    {
        std::cout << session << std::endl;
    }
    std::vector<float> get_input()
    {
        return input;
    }
    std::vector<float> get_output()
    {
        return output;
    }
    // Other methods can be added as needed
};

// Function to get the values from nnfw_tensorinfo array and convert to a Python list
py::list get_dims(const nnfw_tensorinfo &tensorinfo)
{
    py::list dims_list;
    for (int i = 0; i < tensorinfo.rank; ++i)
    {
        dims_list.append(tensorinfo.dims[i]);
    }
    return dims_list;
}

// Function to set the values of nnfw_tensorinfo array from a Python list
void set_dims(nnfw_tensorinfo &tensorinfo, const py::list &dims_list)
{
    tensorinfo.rank = py::len(dims_list);
    for (int i = 0; i < tensorinfo.rank; ++i)
    {
        tensorinfo.dims[i] = py::cast<int>(dims_list[i]);
    }
}

// 파이썬 모듈의 이름은 파일명에서 확장자를 제외한 이름으로 지정합니다.
PYBIND11_MODULE(nnfw_api_pybind, m)
{
    m.doc() = "Python binding for nnfw"; // 모듈에 대한 설명

    // 모듈 내에 C++ 클래스, 함수, 변수를 추가하는 부분
    // 예시: py::class_<YourCppClass>(m, "YourCppClass")
    //        .def(py::init<>());

    py::enum_<NNFW_TYPE>(m, "NNFW_TYPE")
        .value("NNFW_TYPE_TENSOR_FLOAT32", NNFW_TYPE_TENSOR_FLOAT32)
        .value("NNFW_TYPE_TENSOR_INT32", NNFW_TYPE_TENSOR_INT32)
        .value("NNFW_TYPE_TENSOR_QUANT8_ASYMM", NNFW_TYPE_TENSOR_QUANT8_ASYMM)
        .value("NNFW_TYPE_TENSOR_BOOL", NNFW_TYPE_TENSOR_BOOL)
        .value("NNFW_TYPE_TENSOR_UINT8", NNFW_TYPE_TENSOR_UINT8)
        .value("NNFW_TYPE_TENSOR_INT64", NNFW_TYPE_TENSOR_INT64)
        .value("NNFW_TYPE_TENSOR_QUANT8_ASYMM_SIGNED", NNFW_TYPE_TENSOR_QUANT8_ASYMM_SIGNED)
        .value("NNFW_TYPE_TENSOR_QUANT16_SYMM_SIGNED", NNFW_TYPE_TENSOR_QUANT16_SYMM_SIGNED)
        .export_values();

    py::enum_<NNFW_STATUS>(m, "NNFW_STATUS")
        .value("NNFW_STATUS_NO_ERROR", NNFW_STATUS_NO_ERROR)
        .value("NNFW_STATUS_ERROR", NNFW_STATUS_ERROR)
        .value("NNFW_STATUS_UNEXPECTED_NULL", NNFW_STATUS_UNEXPECTED_NULL)
        .value("NNFW_STATUS_INVALID_STATE", NNFW_STATUS_INVALID_STATE)
        .value("NNFW_STATUS_OUT_OF_MEMORY", NNFW_STATUS_OUT_OF_MEMORY)
        .value("NNFW_STATUS_INSUFFICIENT_OUTPUT_SIZE", NNFW_STATUS_INSUFFICIENT_OUTPUT_SIZE)
        .export_values();

    py::enum_<NNFW_LAYOUT>(m, "NNFW_LAYOUT")
        .value("NNFW_LAYOUT_NONE", NNFW_LAYOUT_NONE)
        .value("NNFW_LAYOUT_CHANNELS_LAST", NNFW_LAYOUT_CHANNELS_LAST)
        .value("NNFW_LAYOUT_CHANNELS_FIRST", NNFW_LAYOUT_CHANNELS_FIRST)
        .export_values();

    py::enum_<NNFW_INFO_ID>(m, "NNFW_INFO_ID")
        .value("NNFW_INFO_ID_VERSION", NNFW_INFO_ID_VERSION)
        .export_values();

    py::class_<nnfw_tensorinfo>(m, "nnfw_tensorinfo")
        .def(py::init<>())
        .def_readwrite("dtype", &nnfw_tensorinfo::dtype)
        .def_readwrite("rank", &nnfw_tensorinfo::rank)
        .def_property(
            "dims",
            [](const nnfw_tensorinfo &tensorinfo)
            {
                return get_dims(tensorinfo);
            },
            [](nnfw_tensorinfo &tensorinfo, const py::list &dims_list)
            {
                set_dims(tensorinfo, dims_list);
            });

    py::class_<NNFW_SESSION>(m, "nnfw_session")
        .def(py::init<>())
        .def("create", &NNFW_SESSION::create)
        .def("close", &NNFW_SESSION::close)
        .def("await", &NNFW_SESSION::await)
        .def("apply_tensorinfo", &NNFW_SESSION::apply_tensorinfo)
        .def("input_size", &NNFW_SESSION::input_size)
        .def("load_model_from_file", &NNFW_SESSION::load_model_from_file)
        .def("output_size", &NNFW_SESSION::output_size)
        .def("output_tensorinfo", &NNFW_SESSION::output_tensorinfo)
        .def("prepare", &NNFW_SESSION::prepare)
        .def("query_info_u32", &NNFW_SESSION::query_info_u32)
        .def("run", &NNFW_SESSION::run)
        .def("run_async", &NNFW_SESSION::run_async)
        .def("set_available_backends", &NNFW_SESSION::set_available_backends)
        .def("set_input", &NNFW_SESSION::set_input)
        .def("set_input_layout", &NNFW_SESSION::set_input_layout)
        .def("set_input_tensorinfo", &NNFW_SESSION::set_input_tensorinfo)
        .def("set_op_backend", &NNFW_SESSION::set_op_backend)
        .def("set_output", &NNFW_SESSION::set_output)
        .def("input_tensorinfo", &NNFW_SESSION::input_tensorinfo)
        .def("set_output_layout", &NNFW_SESSION::set_output_layout)
        .def("print_session", &NNFW_SESSION::print_session)
        .def("get_input", &NNFW_SESSION::get_input)
        .def("get_output", &NNFW_SESSION::get_output);
}