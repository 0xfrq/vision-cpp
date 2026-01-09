#include "yolo_onnx.hpp"

using namespace std;
using namespace Ort;

YoloONNX::YoloONNX(const string& model_path)
    : env(ORT_LOGGING_LEVEL_WARNING, "yolo"),
      session(nullptr)
{
    // multi-thread untuk inferensi lebih cepat
    session_options.SetIntraOpNumThreads(2);
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    // pre-allocate buffers untuk hindari alokasi berulang
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
    
    // hitung scale dan padding hanya jika blobsize berubah (hindari frame drop)
    if(blob_size != last_blob_size) {
        cached_scale = (float)blob_size / max(orig_w, orig_h);
        cached_new_w = (int)(orig_w * cached_scale);
        cached_new_h = (int)(orig_h * cached_scale);
        cached_pad_x = (input_width - cached_new_w) / 2;
        cached_pad_y = (input_height - cached_new_h) / 2;
        last_blob_size = blob_size;
        
        // reset padded buffer ke hitam
        padded_buffer.setTo(cv::Scalar(0, 0, 0));
    }
    
    // resize langsung ke roi dalam padded buffer
    cv::Mat roi = padded_buffer(cv::Rect(cached_pad_x, cached_pad_y, cached_new_w, cached_new_h));
    cv::resize(image, roi, cv::Size(cached_new_w, cached_new_h), 0, 0, cv::INTER_LINEAR);
    
    // normalize dan convert BGR->RGB dalam satu pass
    const uchar* src = padded_buffer.ptr<uchar>();
    float* r_ptr = input_tensor_values.data();
    float* g_ptr = r_ptr + input_width * input_height;
    float* b_ptr = g_ptr + input_width * input_height;
    
    const float inv255 = 1.0f / 255.0f;
    int total = input_width * input_height;
    for(int i = 0; i < total; i++) {
        int idx = i * 3;
        r_ptr[i] = src[idx + 2] * inv255;
        g_ptr[i] = src[idx + 1] * inv255;
        b_ptr[i] = src[idx + 0] * inv255;
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
    
    float inv_scale = 1.0f / cached_scale;
    
    vector<Detection> detections;
    detections.reserve(20);
    
    // threshold rendah di sini, filter final dilakukan di main.cpp (0.5f seperti Python)
    const float conf_threshold = 0.25f;
    
    for(int i = 0; i < num_boxes; i++) {
        float conf = out[i * elements + 4];
        if(conf < conf_threshold) continue;
        
        float cx = out[i * elements + 0] - cached_pad_x;
        float cy = out[i * elements + 1] - cached_pad_y;
        float w = out[i * elements + 2];
        float h = out[i * elements + 3];
        
        // expand box sedikit untuk capture full ball (10% padding)
        float expand = 0.1f;
        w *= (1.0f + expand);
        h *= (1.0f + expand);
        
        int x = (int)((cx - w * 0.5f) * inv_scale);
        int y = (int)((cy - h * 0.5f) * inv_scale);
        int width = (int)(w * inv_scale);
        int height = (int)(h * inv_scale);
        
        // clamp ke batas frame
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
    
    // simple NMS untuk gabungkan deteksi overlapping
    if(detections.size() > 1) {
        vector<Detection> nms_result;
        vector<bool> suppressed(detections.size(), false);
        
        // sort by confidence
        sort(detections.begin(), detections.end(), 
             [](const Detection& a, const Detection& b) { return a.conf > b.conf; });
        
        for(size_t i = 0; i < detections.size(); i++) {
            if(suppressed[i]) continue;
            nms_result.push_back(detections[i]);
            
            for(size_t j = i + 1; j < detections.size(); j++) {
                if(suppressed[j]) continue;
                
                // hitung IoU
                cv::Rect inter = detections[i].box & detections[j].box;
                float inter_area = inter.area();
                float union_area = detections[i].box.area() + detections[j].box.area() - inter_area;
                float iou = inter_area / union_area;
                
                if(iou > 0.4f) suppressed[j] = true;
            }
        }
        return nms_result;
    }
    
    return detections;
}

void YoloONNX::setInputSize(int size) {
    blob_size = size;
}
