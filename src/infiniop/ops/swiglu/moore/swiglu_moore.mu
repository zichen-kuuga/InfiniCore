#include "swiglu_moore.h"

#include "../../../elementwise/moore/elementwise_moore.h"

#include "siwglu_moore_kernel.h"
#include "torch_musa_swiglu.h"

namespace op::swiglu::moore {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::moore::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    const auto &up_desc = input_desc_vec.at(0);
    const auto &gate_desc = input_desc_vec.at(1);
    const auto &out_shape = out_desc->shape();
    const auto &up_shape = up_desc->shape();
    const auto &gate_shape = gate_desc->shape();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);
    // CHECK_SAME_SHAPE(out_shape, up_shape, gate_shape);

    // create MOORE elementwise descriptor
    CREATE_ELEMENTWISE_MOORE_DESCRIPTOR(handle, dtype, out_desc, input_desc_vec)

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    //switch (_dtype) {
    //case INFINI_DTYPE_F16:
    //    return _device_info->calculate<256, cuda::SwiGLUOp, half>(_info, workspace, output, inputs, stream);
    //case INFINI_DTYPE_BF16:
    //    return _device_info->calculate<256, cuda::SwiGLUOp, cuda_bfloat16>(_info, workspace, output, inputs, stream);
    //case INFINI_DTYPE_F32:
    //    return _device_info->calculate<256, cuda::SwiGLUOp, float>(_info, workspace, output, inputs, stream);
    //case INFINI_DTYPE_F64:
    //    return _device_info->calculate<256, cuda::SwiGLUOp, double>(_info, workspace, output, inputs, stream);
    //default:
    //    return INFINI_STATUS_BAD_TENSOR_DTYPE;
    //}
    auto dim = _info.getNdim();
    auto input_shape =_info.getInputShape(0);
    auto output_shape =_info.getOutputShape();
    auto input_stride = _info.getInputStrides(0);
    auto output_stride = _info.getOutputStrides();
    musaStream_t mstream = reinterpret_cast<musaStream_t>(stream);
    
    SwishGlu(inputs.at(0), static_cast<const void*>(output), _dtype, mstream, static_cast<int64_t>(dim), 
        reinterpret_cast<const int64_t*>(input_shape), reinterpret_cast<const int64_t*>(output_shape), 
        reinterpret_cast<const int64_t*>(input_stride), reinterpret_cast<const int64_t*>(output_stride));

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::swiglu::moore
