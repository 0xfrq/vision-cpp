#include "yolo_onnx.hpp"

using namespace std;
using namespace Ort;

YoloONNX::YoloONNX(const string& model_path)
    : env(ORT_LOGGING_LEVEL_WARNING, "yolo"),
      session(nullptr)
{
    // optimasi multi-thread untuk cpu
    session_options.SetIntraOpNumThreads(4);
    session_options.SetInterOpNumThreads(2);
    
    // enable semua optimasi graph
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    // pre-allocate buffers
    padded_buffer = cv::Mat::zeros(input_height, input_width, CV_8UC3);
    float_buffer = cv::Mat(input_height, input_width, CV_32FC3);
    input_tensor_values.resize(3 * input_width * input_height);
    
    session = Session(env, model_path.c_str(), session_options);
    printModelInfo();
}

void YoloONNX::printModelInfo() {
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_inputs = session.GetInputCount();
    for (size_t i = 0; i < num_inputs; i++) {
        char* input_name = session.GetInputName(i, allocator);
        cout << "Input " << i << " name: " << input_name << endl;
        allocator.Free(input_name);
    }
    size_t num_outputs = session.GetOutputCount();
    for (size_t i = 0; i < num_outputs; i++) {
        char* output_name = session.GetOutputName(i, allocator);
        cout << "Output " << i << " name: " << output_name << endl;
        allocator.Free(output_name);
    }
}

vector<Detection> YoloONNX::infer(const cv::Mat& image)
{
    int orig_w = image.cols;
    int orig_h = image.rows;
    
    // hitung scale dan padding sekali saja jika blobsize berubah
    if(blob_size != last_blob_size) {
        cached_scale = (float)blob_size / max(orig_w, orig_h);
        int new_w = (int)(orig_w * cached_scale);
        int new_h = (int)(orig_h * cached_scale);
        cached_pad_x = (input_width - new_w) / 2;
        cached_pad_y = (input_height - new_h) / 2;
        last_blob_size = blob_size;
        
        // reset padded buffer ke hitam
        padded_buffer.setTo(cv::Scalar(0,0,0));
    }
    
    // resize langsung ke roi di padded buffer (hindari alokasi)
    int new_w = (int)(orig_w * cached_scale);
    int new_h = (int)(orig_h * cached_scale);
    cv::Mat roi = padded_buffer(cv::Rect(cached_pad_x, cached_pad_y, new_w, new_h));
    cv::resize(image, roi, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    
    // convert bgr to rgb dan normalize dalam satu loop
    const int total = input_width * input_height;
    const uchar* src = padded_buffer.ptr<uchar>();
    float* r_ptr = input_tensor_values.data();
    float* g_ptr = r_ptr + total;
    float* b_ptr = g_ptr + total;
    
    const float inv255 = 1.0f / 255.0f;
    for(int i = 0; i < total; i++) {
        int idx = i * 3;
        r_ptr[i] = src[idx + 2] * inv255;  // R
        g_ptr[i] = src[idx + 1] * inv255;  // G
        b_ptr[i] = src[idx + 0] * inv255;  // B
    }
    
    // run inference
    array<int64_t, 4> input_shape{1, 3, input_height, input_width};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, input_tensor_values.data(), input_tensor_values.size(),
        input_shape.data(), input_shape.size());
    
    const char* input_names[] = {"images"};
    const char* output_names[] = {"output0"};
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    
    float* out = outputs[0].GetTensorMutableData<float>();
    auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int num_boxes = shape[1];
    int elements = shape[2];
    
    // pre-compute inverse scale
    float inv_scale = 1.0f / cached_scale;
    
    vector<Detection> detections;
    detections.reserve(10);
    
    for(int i = 0; i < num_boxes; i++) {
        float conf = out[i * elements + 4];
        if(conf < 0.35f) continue;  // lower threshold untuk deteksi lebih cepat
        
        float cx = out[i * elements + 0] - cached_pad_x;
        float cy = out[i * elements + 1] - cached_pad_y;
        float w = out[i * elements + 2];
        float h = out[i * elements + 3];
        
        int x = (int)((cx - w*0.5f) * inv_scale);
        int y = (int)((cy - h*0.5f) * inv_scale);
        int width = (int)(w * inv_scale);
        int height = (int)(h * inv_scale);
        
        // clamp
        x = max(0, min(x, orig_w - 1));
        y = max(0, min(y, orig_h - 1));
        width = min(width, orig_w - x);
        height = min(height, orig_h - y);
        
        if(width > 5 && height > 5) {
            Detection det;
            det.box = cv::Rect(x, y, width, height);
            det.conf = conf;
            det.class_id = 0;
            detections.push_back(det);
        }
    }
    
    return detections;
}

void YoloONNX::setInputSize(int size) {
    blob_size = size;
}
