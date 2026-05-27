#include "../include/axelera_backend.hpp"

#include "../include/logging.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace autoware_pov::vision
{

namespace
{
// Helper: stringify an axrResult for log messages.
inline std::string axr_err(axrResult r)
{
  return std::string(axr_error_string(r));
}

// Compute the strides (in elements) for an NHWC tensor described by `dims`.
inline void nhwc_strides(const size_t * dims, size_t ndims, size_t * out_strides)
{
  size_t s = 1;
  for (size_t i = ndims; i-- > 0; ) {
    out_strides[i] = s;
    s *= dims[i];
  }
}
}  // namespace

template <typename T>
void AxeleraBackend::AxrDeleter<T>::operator()(T * p) const noexcept
{
  if (p) {
    axr_destroy(reinterpret_cast<const axrObject *>(p));
  }
}
template struct AxeleraBackend::AxrDeleter<axrContext>;
template struct AxeleraBackend::AxrDeleter<axrConnection>;
template struct AxeleraBackend::AxrDeleter<axrModel>;
template struct AxeleraBackend::AxrDeleter<axrModelInstance>;

AxeleraBackend::AxeleraBackend(const std::string & model_dir,
                               const std::string & /*precision*/, int /*gpu_id*/)
{
  LOG_INFO("Axelera: loading compiled-model dir %s", model_dir.c_str());

  // 1. Context.
  context_.reset(axr_create_context());
  if (!context_) {
    throw std::runtime_error("AxeleraBackend: axr_create_context() failed");
  }

  // 2. Pick the first available device.
  axrDeviceInfo * devices_arr = nullptr;
  const size_t n_devices = axr_list_devices(context_.get(), &devices_arr);
  if (n_devices == 0 || devices_arr == nullptr) {
    throw std::runtime_error("AxeleraBackend: no Metis devices found "
                             "(check that axdevice can see one)");
  }
  const axrDeviceInfo & device = devices_arr[0];
  LOG_INFO("Axelera: using device %s (subdevices=%zu)",
           device.name, device.subdevice_count);

  // 3. Connect with 1 sub-device (sized for a single Lite-family model;
  //    multi-model cascades that share a Metis must pass each model its own
  //    Connection with num_sub_devices set from the model's L2 const size).
  connection_.reset(
    axr_device_connect(context_.get(), &device, 1, nullptr));
  if (!connection_) {
    throw std::runtime_error("AxeleraBackend: axr_device_connect failed: " +
                             axr_err(axr_last_error(reinterpret_cast<const axrObject *>(context_.get()))));
  }

  // 4. Load the model from the compiled-model directory.
  model_.reset(axr_load_model(context_.get(), model_dir.c_str()));
  if (!model_) {
    throw std::runtime_error("AxeleraBackend: axr_load_model failed for " + model_dir);
  }

  // 5. Probe IO.  Lite-family + AutoSeg umbrella models are single-input,
  //    single-output.  Multi-output (dual/triple head) needs an interface
  //    extension (see claude_plans plan §7.2).
  const size_t n_in = axr_num_model_inputs(model_.get());
  const size_t n_out = axr_num_model_outputs(model_.get());
  if (n_in != 1) {
    throw std::runtime_error("AxeleraBackend: only single-input models supported (got " +
                             std::to_string(n_in) + ")");
  }
  if (n_out < 1) {
    throw std::runtime_error("AxeleraBackend: model has no outputs");
  }
  if (n_out > 1) {
    LOG_WARN("Axelera: model has %zu outputs; backend exposes only output[0]", n_out);
  }
  input_info_ = axr_get_model_input(model_.get(), 0);
  output_info_ = axr_get_model_output(model_.get(), 0);

  // 6. Decode the input shape (Voyager works in NHWC at the runtime
  //    boundary; PyTorch NCHW input is rewritten by the compiler).
  if (input_info_.ndims != 4) {
    throw std::runtime_error("AxeleraBackend: expected 4D input tensor, got ndims=" +
                             std::to_string(input_info_.ndims));
  }
  // NHWC: [N, H, W, C]
  model_input_height_ = static_cast<int>(input_info_.dims[1]);
  model_input_width_ = static_cast<int>(input_info_.dims[2]);
  model_input_channels_ = static_cast<int>(input_info_.dims[3]);
  input_zero_point_ = static_cast<uint8_t>(
    std::clamp(input_info_.zero_point, 0, 255));

  input_padded_bytes_ = axr_tensor_size(&input_info_);
  input_padded_.assign(input_padded_bytes_, input_zero_point_);

  // 7. Decode output shape into NCHW for downstream consumers.  The
  //    compiled artifact stores it as NHWC + per-axis padding; unpad before
  //    transpose so the public shape matches the source ONNX (1, C, H, W).
  if (output_info_.ndims != 4) {
    throw std::runtime_error("AxeleraBackend: expected 4D output tensor, got ndims=" +
                             std::to_string(output_info_.ndims));
  }
  const size_t out_n = output_info_.dims[0] -
    output_info_.padding[0][0] - output_info_.padding[0][1];
  const size_t out_h = output_info_.dims[1] -
    output_info_.padding[1][0] - output_info_.padding[1][1];
  const size_t out_w = output_info_.dims[2] -
    output_info_.padding[2][0] - output_info_.padding[2][1];
  const size_t out_c = output_info_.dims[3] -
    output_info_.padding[3][0] - output_info_.padding[3][1];
  output_shape_ = {static_cast<int64_t>(out_n), static_cast<int64_t>(out_c),
                   static_cast<int64_t>(out_h), static_cast<int64_t>(out_w)};

  output_padded_bytes_ = axr_tensor_size(&output_info_);
  output_padded_.assign(output_padded_bytes_, 0);
  output_dequant_.assign(out_n * out_c * out_h * out_w, 0.0f);

  LOG_INFO("Axelera: input  NHWC [%zu,%d,%d,%d] (padded %zu bytes, zp=%d)",
           input_info_.dims[0], model_input_height_, model_input_width_,
           model_input_channels_, input_padded_bytes_, input_zero_point_);
  LOG_INFO("Axelera: output NCHW [%lld,%lld,%lld,%lld] (padded %zu bytes)",
           static_cast<long long>(output_shape_[0]),
           static_cast<long long>(output_shape_[1]),
           static_cast<long long>(output_shape_[2]),
           static_cast<long long>(output_shape_[3]),
           output_padded_bytes_);

  // 8. Bind the model to this connection.
  instance_.reset(
    axr_load_model_instance(connection_.get(), model_.get(), nullptr));
  if (!instance_) {
    throw std::runtime_error("AxeleraBackend: axr_load_model_instance failed");
  }
}

AxeleraBackend::~AxeleraBackend() = default;

bool AxeleraBackend::doInference(const cv::Mat & input_image)
{
  // Per the mb-axelera skill (Rule 1): quantize → pad → run → unpad → dequant.
  //
  // The Lite-family + original-AutoSeg ONNX exports do NOT bake in resize
  // or ImageNet normalization — both have to happen host-side.  The
  // calibration transform.py used at compile time defines the data
  // distribution the int8 scales were fit to, so we MUST reproduce it
  // exactly: resize → BGR2RGB → /255 → ImageNet mean/std → CHW (or NHWC
  // depending on the runtime's preferred layout).
  cv::Mat resized;
  cv::resize(input_image, resized,
             cv::Size(model_input_width_, model_input_height_));
  cv::Mat rgb;
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
  cv::Mat floatm;
  rgb.convertTo(floatm, CV_32FC3, 1.0 / 255.0);
  cv::subtract(floatm, cv::Scalar(0.485, 0.456, 0.406), floatm);
  cv::divide(floatm, cv::Scalar(0.229, 0.224, 0.225), floatm);

  // Quantize float → int8: q = round(x / scale + zp).
  const double scale = input_info_.scale;
  const int zp = input_info_.zero_point;
  const size_t unpadded_h = static_cast<size_t>(model_input_height_);
  const size_t unpadded_w = static_cast<size_t>(model_input_width_);
  const size_t unpadded_c = static_cast<size_t>(model_input_channels_);

  // Re-fill the padded buffer with the zero-point so the unwritten padding
  // region stays at quant-zero (idempotent across calls).
  std::fill(input_padded_.begin(), input_padded_.end(), input_zero_point_);

  // Per-axis padding for NHWC: padding[0]=N, padding[1]=H, padding[2]=W,
  // padding[3]=C. The runtime expects values in [zp - 128, zp + 127] mapped
  // through the type (AXR_SIGNED → int8).
  const size_t pad_n_before = input_info_.padding[0][0];
  const size_t pad_h_before = input_info_.padding[1][0];
  const size_t pad_w_before = input_info_.padding[2][0];
  const size_t pad_c_before = input_info_.padding[3][0];

  size_t strides[AXR_MAX_TENSOR_DIMS];
  nhwc_strides(input_info_.dims, input_info_.ndims, strides);

  const float * src = floatm.ptr<float>();  // HWC float
  uint8_t * dst = input_padded_.data();
  for (size_t h = 0; h < unpadded_h; ++h) {
    for (size_t w = 0; w < unpadded_w; ++w) {
      for (size_t c = 0; c < unpadded_c; ++c) {
        const float v = src[(h * unpadded_w + w) * unpadded_c + c];
        const int q = static_cast<int>(std::lround(v / scale + zp));
        const int qclip = std::clamp(q, 0, 255);
        const size_t off =
          (pad_n_before) * strides[0] +
          (pad_h_before + h) * strides[1] +
          (pad_w_before + w) * strides[2] +
          (pad_c_before + c) * strides[3];
        dst[off] = static_cast<uint8_t>(qclip);
      }
    }
  }

  axrArgument in_arg{input_padded_.data(), {}, 0, input_padded_bytes_};
  axrArgument out_arg{output_padded_.data(), {}, 0, output_padded_bytes_};

  const axrResult r = axr_run_model_instance(instance_.get(),
                                             &in_arg, 1, &out_arg, 1);
  if (r != AXR_SUCCESS) {
    LOG_ERROR("Axelera: axr_run_model_instance failed: %s",
              axr_error_string(r));
    return false;
  }

  // Unpad + dequant + transpose: int8 NHWC (padded) → float NCHW (unpadded).
  const double oscale = output_info_.scale;
  const int ozp = output_info_.zero_point;
  size_t ostrides[AXR_MAX_TENSOR_DIMS];
  nhwc_strides(output_info_.dims, output_info_.ndims, ostrides);

  const size_t out_n = static_cast<size_t>(output_shape_[0]);
  const size_t out_c = static_cast<size_t>(output_shape_[1]);
  const size_t out_h = static_cast<size_t>(output_shape_[2]);
  const size_t out_w = static_cast<size_t>(output_shape_[3]);
  const size_t opad_n_before = output_info_.padding[0][0];
  const size_t opad_h_before = output_info_.padding[1][0];
  const size_t opad_w_before = output_info_.padding[2][0];
  const size_t opad_c_before = output_info_.padding[3][0];

  // NHWC int8 → NCHW float
  const int8_t * osrc = output_padded_.data();
  float * odst = output_dequant_.data();
  for (size_t n = 0; n < out_n; ++n) {
    for (size_t c = 0; c < out_c; ++c) {
      for (size_t h = 0; h < out_h; ++h) {
        for (size_t w = 0; w < out_w; ++w) {
          const size_t off =
            (opad_n_before + n) * ostrides[0] +
            (opad_h_before + h) * ostrides[1] +
            (opad_w_before + w) * ostrides[2] +
            (opad_c_before + c) * ostrides[3];
          const int8_t q = osrc[off];
          odst[((n * out_c + c) * out_h + h) * out_w + w] =
            static_cast<float>(static_cast<int>(q) - ozp) * static_cast<float>(oscale);
        }
      }
    }
  }
  return true;
}

const float * AxeleraBackend::getRawTensorData() const
{
  if (output_dequant_.empty()) {
    throw std::runtime_error("AxeleraBackend: getRawTensorData() before doInference()");
  }
  return output_dequant_.data();
}

std::vector<int64_t> AxeleraBackend::getTensorShape() const
{
  return output_shape_;
}

}  // namespace autoware_pov::vision
