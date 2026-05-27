#include "../include/deepx_backend.hpp"

#include "../include/logging.hpp"

#include <cstring>
#include <stdexcept>

namespace autoware_pov::vision
{

DeepXBackend::DeepXBackend(const std::string & dxnn_path,
                           const std::string & /*precision*/, int /*gpu_id*/)
{
  LOG_INFO("DeepX: loading DXNN %s", dxnn_path.c_str());
  engine_ = std::make_unique<dxrt::InferenceEngine>(dxnn_path);

  // Probe input. DX-COM rewrites PyTorch NCHW source to NHWC at the runtime
  // boundary and bakes Div(/255)+Normalize+Transpose+ExpandDim into the input
  // wrapper, so the host feeds raw uint8 NHWC [1, H, W, C].
  auto in_tensors = engine_->GetInputs();
  if (in_tensors.size() != 1) {
    throw std::runtime_error(
      "DeepXBackend: only single-input models are supported (got " +
      std::to_string(in_tensors.size()) + ")");
  }
  const auto & in_shape = in_tensors[0].shape();
  if (in_shape.size() != 4) {
    throw std::runtime_error("DeepXBackend: expected 4D input shape");
  }
  // NHWC: [N, H, W, C]
  model_input_height_ = static_cast<int>(in_shape[1]);
  model_input_width_ = static_cast<int>(in_shape[2]);
  model_input_channels_ = static_cast<int>(in_shape[3]);

  if (in_tensors[0].type() != dxrt::DataType::UINT8) {
    throw std::runtime_error("DeepXBackend: expected UINT8 NHWC input from DX-COM");
  }
  input_bytes_ = static_cast<size_t>(model_input_height_) *
                 model_input_width_ * model_input_channels_;
  input_buffer_.resize(input_bytes_);

  // Probe output. SceneSegLite emits float NCHW [1, classes, H, W].
  auto out_tensors = engine_->GetOutputs();
  if (out_tensors.size() != 1) {
    LOG_WARN("DeepX: model has %zu output tensors; backend exposes only the first",
             out_tensors.size());
  }
  output_shape_.assign(out_tensors[0].shape().begin(),
                       out_tensors[0].shape().end());

  LOG_INFO("DeepX: input  uint8 NHWC %dx%dx%d (%zu bytes)",
           model_input_height_, model_input_width_, model_input_channels_,
           input_bytes_);
}

bool DeepXBackend::doInference(const cv::Mat & input_image)
{
  // Host preprocessing: resize, BGR2RGB, into a contiguous uint8 NHWC buffer.
  // The /255 + ImageNet normalize + transpose + expand are baked into the
  // .dxnn input wrapper at compile time (see DX-COM compiler.log).
  cv::Mat resized;
  cv::resize(input_image, resized, cv::Size(model_input_width_, model_input_height_));
  cv::Mat rgb;
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

  if (!rgb.isContinuous()) {
    rgb = rgb.clone();
  }
  if (rgb.total() * rgb.elemSize() != input_bytes_) {
    LOG_ERROR("DeepX: preprocessed buffer size mismatch (got %zu, expected %zu)",
              rgb.total() * rgb.elemSize(), input_bytes_);
    return false;
  }
  std::memcpy(input_buffer_.data(), rgb.data, input_bytes_);

  try {
    last_outputs_ = engine_->Run(input_buffer_.data());
  } catch (const std::exception & e) {
    LOG_ERROR("DeepX: Run() threw: %s", e.what());
    return false;
  }
  if (last_outputs_.empty()) {
    LOG_ERROR("DeepX: Run() returned no outputs");
    return false;
  }
  return true;
}

const float * DeepXBackend::getRawTensorData() const
{
  if (last_outputs_.empty()) {
    throw std::runtime_error("DeepXBackend: getRawTensorData() before doInference()");
  }
  return static_cast<const float *>(last_outputs_[0]->data());
}

std::vector<int64_t> DeepXBackend::getTensorShape() const
{
  if (last_outputs_.empty()) {
    return output_shape_;
  }
  // Prefer the runtime-reported shape (covers dynamic-shape models).
  return last_outputs_[0]->shape();
}

}  // namespace autoware_pov::vision
