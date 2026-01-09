// deteksi bola yolov5 + hsv tracking untuk robot sepak bola
#include <ros/ros.h>
#include <ros/package.h>
#include <v2_detection/BallState.h>
#include <v2_detection/BallCoordinate.h>
#include <v2_detection/Ballarea.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <algorithm>
#include "yolo_onnx.hpp"

using namespace cv;
using namespace std;

// fungsi utilitas
template<typename T>
inline T clamp(T v, T lo, T hi) { return (v < lo) ? lo : (v > hi) ? hi : v; }

inline double map_value(double v, double smin, double smax, double tmin, double tmax) {
    return tmin + (clamp(v, smin, smax) - smin) * (tmax - tmin) / (smax - smin);
}

// status deteksi
enum DetectState { NOTFOUND = 0, FOUND = 1 };
DetectState detect_status = NOTFOUND;

// hsv range untuk tracking
int min_h=0, min_s=0, min_v=0;
int max_h=0, max_s=0, max_v=0;

// data bola
int x_ball=0, y_ball=0, w_ball=0, h_ball=0;
int ball_area = 0;
Point2f center_ball(0,0);
int scan_x1=0, scan_x2=0;
Rect last_box;

// timing dan fps
double waktu_sebelum = 0.0;
int frame_counter = 0;
double fps_start_time = 0.0;
double fps = 0.0;

// ukuran frame dan blobsize dinamis
constexpr int FRAME_W = 320;
constexpr int FRAME_H = 240;
int blobsize = 416;

// pre-allocated buffers untuk hsv tracking
Mat hsv_buffer, mask_buffer, field_mask;
Mat kernel_morph;
bool buffers_initialized = false;

// ros messages pre-allocated
v2_detection::BallCoordinate bc_msg;
v2_detection::BallState bs_msg;
v2_detection::Ballarea ba_msg;

// hitung fps setiap 0.2 detik
inline void calculate_fps() {
    frame_counter++;
    double now = ros::Time::now().toSec();
    double elapsed = now - fps_start_time;
    if(elapsed >= 0.2) {
        fps = frame_counter / elapsed;
        frame_counter = 0;
        fps_start_time = now;
    }
}

// ambil nilai hsv dari titik-titik di dalam bounding box bola (optimized)
inline void get_hsv_val(const Mat& img) {
    int x1 = x_ball, y1 = y_ball;
    int x2 = x_ball + w_ball, y2 = y_ball + h_ball;
    int cx = (x1+x2)/2, cy = (y1+y2)/2;
    
    // 6 titik sampling
    int px[6] = {cx, (x1+cx)/2, (x2+cx)/2, (cx+x2)/2, (x1+cx)/2, cx};
    int py[6] = {cy, (y1+cy)/2, (y2+cy)/2, (cy+y1)/2, (y1+(y2-y1)+cy)/2, (cy+y2)/2};
    
    int H[6], S[6], V[6];
    
    for(int i = 0; i < 6; i++) {
        int x = clamp(px[i], 0, img.cols-1);
        int y = clamp(py[i], 0, img.rows-1);
        Vec3b bgr = img.at<Vec3b>(y, x);
        
        // konversi bgr ke hsv manual (lebih cepat dari cvtColor untuk 1 pixel)
        int b = bgr[0], g = bgr[1], r = bgr[2];
        int vmax = max({r, g, b});
        int vmin = min({r, g, b});
        int delta = vmax - vmin;
        
        V[i] = vmax;
        S[i] = (vmax == 0) ? 0 : (delta * 255 / vmax);
        
        if(delta == 0) H[i] = 0;
        else if(vmax == r) H[i] = 30 * (g - b) / delta;
        else if(vmax == g) H[i] = 60 + 30 * (b - r) / delta;
        else H[i] = 120 + 30 * (r - g) / delta;
        if(H[i] < 0) H[i] += 180;
        H[i] /= 2;  // opencv uses 0-179
    }
    
    min_h = *min_element(H, H+6);
    max_h = min(*max_element(H, H+6), 33);
    min_s = max(*min_element(S, S+6), 160);
    max_s = *max_element(S, S+6);
    min_v = *min_element(V, V+6);
    max_v = *max_element(V, V+6);
}

// threaded video capture (optimized)
class ThreadedCapture {
private:
    VideoCapture cap;
    Mat frame;
    Mat frame_buffer;
    bool stopped;
    mutex frameMutex;
    thread captureThread;
    
    void update() {
        while(!stopped) {
            cap >> frame_buffer;
            if(!frame_buffer.empty()) {
                lock_guard<mutex> lock(frameMutex);
                swap(frame, frame_buffer);
            }
        }
    }
    
public:
    ThreadedCapture(int src) : stopped(false) {
        cap.open(src);
        cap.set(CAP_PROP_FPS, 60);
        cap.set(CAP_PROP_FRAME_WIDTH, FRAME_W);
        cap.set(CAP_PROP_FRAME_HEIGHT, FRAME_H);
        cap.set(CAP_PROP_BUFFERSIZE, 1);  // minimal buffer untuk latency rendah
        if(!cap.isOpened()) {
            ROS_ERROR("kamera gagal dibuka");
            return;
        }
        cap >> frame;
        captureThread = thread(&ThreadedCapture::update, this);
    }
    
    Mat read() {
        lock_guard<mutex> lock(frameMutex);
        return frame;
    }
    
    void stop() {
        stopped = true;
        if(captureThread.joinable()) captureThread.join();
        cap.release();
    }
    
    ~ThreadedCapture() { stop(); }
};

// init buffers untuk hsv tracking
void init_buffers() {
    if(!buffers_initialized) {
        hsv_buffer = Mat(FRAME_H, FRAME_W, CV_8UC3);
        mask_buffer = Mat(FRAME_H, FRAME_W, CV_8UC1);
        field_mask = Mat(FRAME_H, FRAME_W, CV_8UC1);
        kernel_morph = Mat::ones(3, 3, CV_8U);  // smaller kernel = faster
        buffers_initialized = true;
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "vision_yolo_cpp");
    ros::NodeHandle nh;

    // publisher ros
    auto pub_state = nh.advertise<v2_detection::BallState>("/DEWO/image_processing/deteksi_bola/ball_state", 1);
    auto pub_coord = nh.advertise<v2_detection::BallCoordinate>("/DEWO/image_processing/deteksi_bola/coordinate", 1);
    auto pub_area = nh.advertise<v2_detection::Ballarea>("/DEWO/image_processing/deteksi_bola/ball_area", 1);

    // load model yolo
    string pkg = ros::package::getPath("vision_cpp");
    YoloONNX yolo(pkg + "/src/best.onnx");

    // inisialisasi kamera dan buffers
    ROS_INFO("memulai kamera...");
    ThreadedCapture capture(0);
    init_buffers();
    
    fps_start_time = ros::Time::now().toSec();
    waktu_sebelum = ros::Time::now().toSec();
    
    ROS_INFO("sistem siap");
    
    // pre-compute constants
    const float inv_w = 2.0f / FRAME_W;
    const float inv_h = 2.0f / FRAME_H;

    while(ros::ok()) {
        Mat img = capture.read();
        if(img.empty()) { ros::spinOnce(); continue; }

        // mode tracking hsv (lebih cepat dari yolo)
        if(detect_status == FOUND) {
            // langsung konversi ke hsv tanpa field extraction (lebih cepat)
            cvtColor(img, hsv_buffer, COLOR_BGR2HSV);
            inRange(hsv_buffer, Scalar(min_h, min_s, min_v), Scalar(max_h, max_s, max_v), mask_buffer);
            
            // morphology minimal
            morphologyEx(mask_buffer, mask_buffer, MORPH_CLOSE, kernel_morph);
            morphologyEx(mask_buffer, mask_buffer, MORPH_OPEN, kernel_morph);
            
            // cari kontur
            vector<vector<Point>> contours;
            findContours(mask_buffer, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            
            bool found = false;
            int best_area = 0;
            Rect best_rect;
            
            // cari kontur terbaik dalam scan area
            for(auto& c : contours) {
                Rect r = boundingRect(c);
                int area = r.width * r.height;
                int cx = r.x + r.width/2;
                
                // filter: dalam scan area, ukuran reasonable, lebih besar dari sebelumnya
                if(cx >= scan_x1 && cx <= scan_x2 && 
                   area > ball_area/5 && area < ball_area*2 && 
                   area > 2000 && area > best_area) {
                    best_area = area;
                    best_rect = r;
                    found = true;
                }
            }
            
            if(found) {
                float cx = best_rect.x + best_rect.width * 0.5f;
                float cy = best_rect.y + best_rect.height * 0.5f;
                center_ball = Point2f(cx, cy);
                last_box = best_rect;
                
                // publish langsung tanpa clone message
                bc_msg.pos_x = cx * inv_w - 1.0f;
                bc_msg.pos_y = cy * inv_h - 1.0f;
                bc_msg.obj_size = best_area;
                pub_coord.publish(bc_msg);
                
                ba_msg.ballarea = best_area;
                pub_area.publish(ba_msg);
                
                bs_msg.ball_status = "FOUND";
                pub_state.publish(bs_msg);
                
                // cek waktu reset ke yolo
                double delta = ros::Time::now().toSec() - waktu_sebelum;
                double waktu_detect = map_value(best_area, 0, 76800, 0.5, 80);
                
                if(delta >= waktu_detect) {
                    waktu_sebelum = ros::Time::now().toSec();
                    detect_status = NOTFOUND;
                }
            } else {
                detect_status = NOTFOUND;
            }
        }

        // mode pencarian yolo
        if(detect_status == NOTFOUND) {
            yolo.setInputSize(blobsize);
            auto dets = yolo.infer(img);
            
            if(dets.empty()) {
                bs_msg.ball_status = "NOTFOUND";
                pub_state.publish(bs_msg);
                blobsize = 416;  // reset untuk search
            } else {
                bool yolo_found = false;
                for(auto& d : dets) {
                    if(d.class_id != 0 || d.conf < 0.4f) continue;
                    
                    Rect b = d.box;
                    x_ball = b.x; y_ball = b.y;
                    w_ball = b.width; h_ball = b.height;
                    ball_area = w_ball * h_ball;
                    
                    float cx = x_ball + w_ball * 0.5f;
                    float cy = y_ball + h_ball * 0.5f;
                    center_ball = Point2f(cx, cy);
                    last_box = b;
                    
                    // hitung scan area
                    int in_area = (ball_area <= 5000) ? w_ball * 3 : w_ball + 35;
                    scan_x1 = cx - in_area;
                    scan_x2 = cx + in_area;
                    
                    // publish
                    bc_msg.pos_x = cx * inv_w - 1.0f;
                    bc_msg.pos_y = cy * inv_h - 1.0f;
                    bc_msg.obj_size = ball_area;
                    pub_coord.publish(bc_msg);
                    
                    ba_msg.ballarea = ball_area;
                    pub_area.publish(ba_msg);
                    
                    bs_msg.ball_status = "FOUND";
                    pub_state.publish(bs_msg);
                    
                    // ambil hsv dan switch ke tracking
                    get_hsv_val(img);
                    detect_status = FOUND;
                    waktu_sebelum = ros::Time::now().toSec();
                    
                    // update blobsize
                    blobsize = (ball_area <= 2800) ? 320 : 224;
                    yolo_found = true;
                    break;
                }
                
                if(!yolo_found) {
                    bs_msg.ball_status = "NOTFOUND";
                    pub_state.publish(bs_msg);
                }
            }
        }

        // fps dan display (minimal untuk speed)
        calculate_fps();
        
        // gambar box dan info
        if(detect_status == FOUND) {
            rectangle(img, last_box, Scalar(0,255,255), 2);
        }
        
        char text[32];
        snprintf(text, sizeof(text), "%.0f %d", fps, blobsize);
        putText(img, text, Point(5,15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,0), 1);
        
        imshow("VISION_CPP", img);
        waitKey(1);
        
        ros::spinOnce();
    }
    
    capture.stop();
    return 0;
}
