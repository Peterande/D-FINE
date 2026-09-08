#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstring>
#include <string>
#include <vector>

namespace plugin_ge1 {

constexpr const char* kPLUGIN_NAME{"GatherElementsAxis1Plugin"};
constexpr const char* kPLUGIN_VERSION{"1"};

__global__ void gatherAxis1KernelI32(
    const float* data, const int32_t* indices, float* out, int B, int N, int M, int C) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;
    if (b >= B || m >= M || c >= C) {
        return;
    }
    int outOff = ((b * M + m) * C + c);
    int idxOff = outOff;
    int idx = indices[idxOff];
    if (idx < 0) {
        idx += N;
    }
    idx = idx < 0 ? 0 : idx;
    idx = idx >= N ? (N - 1) : idx;
    int inOff = ((b * N + idx) * C + c);
    out[outOff] = data[inOff];
}

__global__ void gatherAxis1KernelI64(
    const float* data, const int64_t* indices, float* out, int B, int N, int M, int C) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;
    if (b >= B || m >= M || c >= C) {
        return;
    }
    int outOff = ((b * M + m) * C + c);
    int idxOff = outOff;
    int idx = static_cast<int>(indices[idxOff]);
    if (idx < 0) {
        idx += N;
    }
    idx = idx < 0 ? 0 : idx;
    idx = idx >= N ? (N - 1) : idx;
    int inOff = ((b * N + idx) * C + c);
    out[outOff] = data[inOff];
}

class GatherElementsAxis1Plugin final : public nvinfer1::IPluginV2DynamicExt {
public:
    GatherElementsAxis1Plugin() = default;
    GatherElementsAxis1Plugin(const void* data, size_t length) {
        (void)data;
        (void)length;
    }

    // IPluginV2
    const char* getPluginType() const noexcept override { return kPLUGIN_NAME; }
    const char* getPluginVersion() const noexcept override { return kPLUGIN_VERSION; }
    int getNbOutputs() const noexcept override { return 1; }
    int initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    size_t getSerializationSize() const noexcept override { return 0; }
    void serialize(void* buffer) const noexcept override { (void)buffer; }
    void destroy() noexcept override { delete this; }
    IPluginV2DynamicExt* clone() const noexcept override { return new GatherElementsAxis1Plugin(); }
    void setPluginNamespace(const char* pluginNamespace) noexcept override {
        mNamespace = pluginNamespace ? pluginNamespace : "";
    }
    const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

    // IPluginV2Ext
    nvinfer1::DataType getOutputDataType(
        int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override {
        (void)index;
        (void)nbInputs;
        return inputTypes[0];
    }
    void configurePlugin(
        const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs, const nvinfer1::DynamicPluginTensorDesc* out,
        int nbOutputs) noexcept override {
        (void)in;
        (void)nbInputs;
        (void)out;
        (void)nbOutputs;
    }

    // IPluginV2DynamicExt
    nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override {
        (void)outputIndex;
        (void)nbInputs;
        (void)exprBuilder;
        // output shape equals indices shape
        return inputs[1];
    }

    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override {
        (void)nbInputs;
        (void)nbOutputs;
        const nvinfer1::PluginTensorDesc& desc = inOut[pos];
        if (desc.format != nvinfer1::TensorFormat::kLINEAR) {
            return false;
        }
        if (pos == 0) {
            return desc.type == nvinfer1::DataType::kFLOAT;
        }
        if (pos == 1) {
            return desc.type == nvinfer1::DataType::kINT32 || desc.type == nvinfer1::DataType::kINT64;
        }
        if (pos == 2) {
            return desc.type == inOut[0].type;
        }
        return false;
    }

    size_t getWorkspaceSize(
        const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs,
        int nbOutputs) const noexcept override {
        (void)inputs;
        (void)nbInputs;
        (void)outputs;
        (void)nbOutputs;
        return 0;
    }

    int enqueue(
        const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override {
        (void)outputDesc;
        (void)workspace;
        // expected shapes: data [B,N,C], indices [B,M,C], out [B,M,C]
        const nvinfer1::Dims& d0 = inputDesc[0].dims;
        const nvinfer1::Dims& d1 = inputDesc[1].dims;
        if (d0.nbDims != 3 || d1.nbDims != 3) {
            return 1;
        }
        int B = d0.d[0];
        int N = d0.d[1];
        int C = d0.d[2];
        int M = d1.d[1];
        if (B <= 0 || N <= 0 || M <= 0 || C <= 0) {
            return 1;
        }

        dim3 block(16, 16, 1);
        dim3 grid((C + block.x - 1) / block.x, (M + block.y - 1) / block.y, B);

        const float* data = static_cast<const float*>(inputs[0]);
        float* out = static_cast<float*>(outputs[0]);

        if (inputDesc[1].type == nvinfer1::DataType::kINT32) {
            const int32_t* idx = static_cast<const int32_t*>(inputs[1]);
            gatherAxis1KernelI32<<<grid, block, 0, stream>>>(data, idx, out, B, N, M, C);
        } else if (inputDesc[1].type == nvinfer1::DataType::kINT64) {
            const int64_t* idx = static_cast<const int64_t*>(inputs[1]);
            gatherAxis1KernelI64<<<grid, block, 0, stream>>>(data, idx, out, B, N, M, C);
        } else {
            return 1;
        }
        return cudaGetLastError() == cudaSuccess ? 0 : 1;
    }

    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, nvinfer1::IGpuAllocator* gpuAllocator) noexcept override {
        (void)cudnnContext;
        (void)cublasContext;
        (void)gpuAllocator;
    }
    void detachFromContext() noexcept override {}

private:
    std::string mNamespace;
};

class GatherElementsAxis1PluginCreator final : public nvinfer1::IPluginCreator {
public:
    GatherElementsAxis1PluginCreator() {
        mFC.nbFields = 0;
        mFC.fields = nullptr;
    }

    const char* getPluginName() const noexcept override { return kPLUGIN_NAME; }
    const char* getPluginVersion() const noexcept override { return kPLUGIN_VERSION; }
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override { return &mFC; }
    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override {
        (void)name;
        (void)fc;
        return new GatherElementsAxis1Plugin();
    }
    nvinfer1::IPluginV2* deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept override {
        (void)name;
        return new GatherElementsAxis1Plugin(serialData, serialLength);
    }
    void setPluginNamespace(const char* pluginNamespace) noexcept override {
        mNamespace = pluginNamespace ? pluginNamespace : "";
    }
    const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

private:
    std::string mNamespace;
    nvinfer1::PluginFieldCollection mFC{};
};

}  // namespace plugin_ge1

using GatherElementsAxis1PluginCreatorGlobal = plugin_ge1::GatherElementsAxis1PluginCreator;
REGISTER_TENSORRT_PLUGIN(GatherElementsAxis1PluginCreatorGlobal);
