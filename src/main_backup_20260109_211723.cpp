// deteksi bola yolov5 + hsv tracking untuk robot sepak bola
// EXACT port dari python gandamana_vision.py ball_detect()
#include <ros/ros.h>
#include <ros/package.h>
#include <v2_detection/BallState.h>
#include <v2_detection/BallCoordinate.h>
#include <v2_detection/Ballarea.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <algorithm>
#include <cmath>
#include "yolo_onnx.hpp"

using namespace cv;
using namespace std;

// ============ UTILITY FUNCTIONS ============
template<typename T>
inline T clamp(T v, T lo, T hi) { return (v < lo) ? lo : (v > hi) ? hi : v; }

int min_val(int a, int b) { return (a <= b) ? a : b; }
int max_val(int a, int b) { return (a >= b) ? a : b; }

double map_value(double source_val, double source_min, double source_max, double target_min, double target_max) {
    source_val = min_val(source_val, max_val(source_min, source_max));
    source_val = max_val(source_val, min_val(source_min, source_max));
    return target_min + ((source_val - source_min) * ((target_max - target_min) / (source_max - source_min)));
}

// ============ GLOBAL STATE (SEPERTI PYTHON) ============
enum DetectState { NOTFOUND = 0, FOUND = 1 };
DetectState detect_status = NOTFOUND;

// HSV range
int min_h=0, min_s=0, min_v=0;
int max_h=0, max_s=0, max_v=0;

// ball data
int x_ball=0, y_ball=0, w_ball=0, h_ball=0;
int ball_area = 0;
Point2f center_ball(0, 0);
vector<int> scan_area = {0, 0};

// timing
double waktu_sebelum = 0.0;
int frame_counter = 0;
double fps_start_time = 0.0;
double fps = 0.0;

// frame & blob
const int framesize[2] = {320, 240};
int blobsize = 416;
const float fisheye = 1.0f;  // 0.34 jika pakai fisheye, 1.0 sebaliknya

// ============ FPS CALCULATION ============
void calculate_fps() {
    frame_counter++;
    double current_time = ros::Time::now().toSec();
    double elapsed = current_time - fps_start_time;
    if(elapsed >= 0.2) {
        fps = frame_counter / elapsed;
        frame_counter = 0;
        fps_start_time = current_time;
    }
}

// ============ GET HSV VALUE DARI BOLA (EXACT PYTHON PORT) ============
void get_hsv_val(const Mat& img) {
    int x1 = x_ball;
    int y1 = y_ball;
    int x2 = x_ball + w_ball;
    int y2 = y_ball + h_ball;
    
    ball_area = w_ball * h_ball;
    ROS_INFO("LUAS BOLA : %d", ball_area);
    
    // compute sampling points
    vector<pair<int,int>> dots;
    int dot_tengah_x = (x1 + x2) / 2;
    int dot_tengah_y = (y1 + y2) / 2;
    
    dots.push_back({dot_tengah_x, dot_tengah_y});                                                    // dot_tengah
    dots.push_back({(x1 + dot_tengah_x) / 2, (y1 + dot_tengah_y) / 2});                             // dot_sepertiga_kanan
    dots.push_back({(x2 + dot_tengah_x) / 2, (y2 + dot_tengah_y) / 2});                             // dot_duapertiga_kanan
    dots.push_back({(dot_tengah_x + (x1 + (x2-x1))) / 2, (dot_tengah_y + y1) / 2});                 // dot_sepertiga_kiri
    dots.push_back({(x1 + dot_tengah_x) / 2, (y1 + (y2-y1) + dot_tengah_y) / 2});                   // dot_duapertiga_kiri
    dots.push_back({dot_tengah_x, (dot_tengah_y + y1 + w_ball/5) / 2});                             // dot_atas_tengah
    
    // sample HSV di setiap dot
    vector<int> H, S, V;
    for(auto& dot : dots) {
        int px = clamp(dot.first, 0, img.cols-1);
        int py = clamp(dot.second, 0, img.rows-1);
        Vec3b bgr = img.at<Vec3b>(py, px);
        
        Mat hsv_mat;
        cvtColor(Mat(1, 1, CV_8UC3, Scalar(bgr[0], bgr[1], bgr[2])), hsv_mat, COLOR_BGR2HSV);
        Vec3b hsv_val = hsv_mat.at<Vec3b>(0, 0);
        
        H.push_back(hsv_val[0]);
        S.push_back(hsv_val[1]);
        V.push_back(hsv_val[2]);
    }
    
    // compute min/max
    min_h = *min_element(H.begin(), H.end());
    min_s = *min_element(S.begin(), S.end());
    min_v = *min_element(V.begin(), V.end());
    max_h = *max_element(H.begin(), H.end());
    max_s = *max_element(S.begin(), S.end());
    max_v = *max_element(V.begin(), V.end());
    
    // apply threshold limits seperti python
    if(max_h >= 33) max_h = 33;
    if(min_s <= 160) min_s = 160;
    
    ROS_INFO("min hue %d min sat %d min val %d max hue %d max sat %d max val %d", 
             min_h, min_s, min_v, max_h, max_s, max_v);
}

// ============ EXTRACT FIELD (GREEN AREA) ============
Mat extractField(const Mat& img) {
    Mat hsv, mask, result;
    cvtColor(img, hsv, COLOR_BGR2HSV);
    
    inRange(hsv, Scalar(35, 40, 40), Scalar(85, 255, 255), mask);
    
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5,5));
    erode(mask, mask, kernel, Point(-1,-1), 2);
    dilate(mask, mask, kernel, Point(-1,-1), 5);
    
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_TREE, CHAIN_APPROX_NONE);
    
    Mat fieldMask = Mat::zeros(img.size(), CV_8UC1);
    if(!contours.empty()) {
        auto maxContour = *max_element(contours.begin(), contours.end(),
            [](const vector<Point>&a, const vector<Point>&b){ return contourArea(a) < contourArea(b); });
        vector<Point> hull;
        convexHull(maxContour, hull);
        fillConvexPoly(fieldMask, hull, Scalar(255));
    }
    
    bitwise_and(img, img, result, fieldMask);
    return result;
}

// ============ THREADED CAMERA CAPTURE ============
class ThreadedCapture {
private:
    VideoCapture cap;
    Mat frame;
    bool stopped;
    mutex frameMutex;
    thread captureThread;
    
    void update() {
        while(!stopped) {
            Mat temp;
            cap >> temp;
            if(!temp.empty()) {
                lock_guard<mutex> lock(frameMutex);
                frame = temp.clone();
            }
        }
    }
    
public:
    ThreadedCapture(int src) : stopped(false) {
        cap.open(src, CAP_V4L2);
        cap.set(CAP_PROP_FPS, 60);
        cap.set(CAP_PROP_FRAME_WIDTH, framesize[0]);
        cap.set(CAP_PROP_FRAME_HEIGHT, framesize[1]);
        if(!cap.isOpened()) {
            ROS_ERROR("kamera gagal dibuka");
            return;
        }
        cap >> frame;
        captureThread = thread(&ThreadedCapture::update, this);
    }
    
    Mat read() {
        lock_guard<mutex> lock(frameMutex);
        return frame.clone();
    }
    
    void stop() {
        stopped = true;
        if(captureThread.joinable()) captureThread.join();
        cap.release();
    }
    
    ~ThreadedCapture() { stop(); }
};

// ============ MAIN ============
int main(int argc, char** argv) {
    ros::init(argc, argv, "vision_yolo_cpp");
    ros::NodeHandle nh;

    auto pub_state = nh.advertise<v2_detection::BallState>("/DEWO/image_processing/deteksi_bola/ball_state", 10);
    auto pub_coord = nh.advertise<v2_detection::BallCoordinate>("/DEWO/image_processing/deteksi_bola/coordinate", 10);
    auto pub_area = nh.advertise<v2_detection::Ballarea>("/DEWO/image_processing/deteksi_bola/ball_area", 10);

    string pkg = ros::package::getPath("vision_cpp");
    YoloONNX yolo(pkg + "/src/best.onnx");

    ROS_INFO("memulai kamera...");
    ThreadedCapture capture(0);
    
    fps_start_time = ros::Time::now().toSec();
    waktu_sebelum = ros::Time::now().toSec();
    
    ROS_INFO("sistem siap");

    while(ros::ok()) {
        Mat img = capture.read();
        if(img.empty()) { ros::spinOnce(); continue; }
        
        Mat img_result = img.clone();

        // ============ MODE TRACKING HSV (JIKA BALL FOUND) ============
        if(detect_status == FOUND) {
            Mat field_img = extractField(img);
            
            Mat hsv, binary_ball;
            cvtColor(field_img, hsv, COLOR_BGR2HSV);
            inRange(hsv, Scalar(min_h, min_s, min_v), Scalar(max_h, max_s, max_v), binary_ball);
            
            Mat kernel = Mat::ones(5, 5, CV_8U);
            morphologyEx(binary_ball, binary_ball, MORPH_CLOSE, kernel);
            morphologyEx(binary_ball, binary_ball, MORPH_OPEN, kernel);
            
            vector<vector<Point>> contours;
            findContours(binary_ball, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            
            bool objek_ditemukan = false;
            
            for(auto& contour : contours) {
                double area = contourArea(contour);
                
                // filter area: area > ball_area/5 && area < ball_area*1.1
                if(area > ball_area/5 && area < ball_area*1.1 && contours.size() > 0) {
                    Rect r = boundingRect(contour);
                    float x_center = r.x + r.width/2.0f;
                    
                    // check scan area dan minimum area
                    if(scan_area[0] <= x_center && x_center <= scan_area[1] && (r.width*r.height) > (3000*fisheye)) {
                        int area_bola = r.width * r.height;
                        float x_center_rect = r.x + r.width/2.0f;
                        float y_center_rect = r.y + r.height/2.0f;
                        center_ball = Point2f(x_center_rect, y_center_rect);
                        
                        rectangle(img_result, r, Scalar(0, 255, 255), 2);
                        
                        // send ROS message
                        v2_detection::BallCoordinate bc;
                        bc.pos_x = clamp(x_center_rect/framesize[0]*2-1, -1.0f, 1.0f);
                        bc.pos_y = clamp(y_center_rect/framesize[1]*2-1, -1.0f, 1.0f);
                        bc.obj_size = area_bola;
                        pub_coord.publish(bc);
                        
                        v2_detection::Ballarea ba;
                        ba.ballarea = area_bola;
                        pub_area.publish(ba);
                        
                        v2_detection::BallState bs;
                        bs.ball_status = "FOUND";
                        pub_state.publish(bs);
                        
                        objek_ditemukan = true;
                        
                        // check timing untuk reset ke yolo
                        double waktu_detect = map_value(area_bola, 0, 76800, 0.5, 80);
                        double waktu_sesudah = ros::Time::now().toSec();
                        double delta = waktu_sesudah - waktu_sebelum;
                        
                        ROS_INFO("ball area result: %d", area_bola);
                        
                        if(delta >= waktu_detect) {
                            waktu_sebelum = ros::Time::now().toSec();
                            detect_status = NOTFOUND;
                        }
                        break;
                    }
                    break;
                }
            }
            
            if(!objek_ditemukan) {
                detect_status = NOTFOUND;
            }
        }

        // ============ MODE PENCARIAN YOLO (JIKA BALL NOTFOUND) ============
        if(detect_status == NOTFOUND) {
            yolo.setInputSize(blobsize);
            auto dets = yolo.infer(img);
            
            if(dets.empty()) {
                ROS_INFO("YOLO NOT FOUND");
                v2_detection::BallState bs;
                bs.ball_status = "NOTFOUND";
                pub_state.publish(bs);
                blobsize = 416;
                ROS_INFO("Did not detect ball, setting blobsize to %d", blobsize);
            } else {
                // ITERASI DETEKSI - CARI CLASS 0 DENGAN CONFIDENCE >= 0.5 (SEPERTI PYTHON)
                bool found_ball = false;
                
                for(auto& det : dets) {
                    float conf = det.conf;
                    int cls = det.class_id;
                    
                    // skip jika confidence < 0.5
                    if(conf < 0.5f) continue;
                    
                    int x = det.box.x;
                    int y = det.box.y;
                    int w = det.box.width;
                    int h = det.box.height;
                    
                    // cari class 0 (bola)
                    if(cls == 0) {
                        x_ball = x;
                        y_ball = y;
                        w_ball = w;
                        h_ball = h;
                        
                        float x_center_ball = x_ball + w_ball/2.0f;
                        float y_center_ball = y_ball + h_ball/2.0f;
                        center_ball = Point2f(x_center_ball, y_center_ball);
                        int obj_size_ball = w_ball * h_ball;
                        ball_area = obj_size_ball;
                        
                        // compute scan area seperti python
                        int in_area_ball;
                        if(ball_area <= (5000*fisheye)) {
                            in_area_ball = w_ball * 3;
                        } else {
                            in_area_ball = w_ball + (int)(35*fisheye);
                        }
                        scan_area = {(int)(x_center_ball - in_area_ball), (int)(x_center_ball + in_area_ball)};
                        
                        // draw bounding box
                        rectangle(img_result, det.box, Scalar(255, 0, 0), 1);
                        char text[50];
                        snprintf(text, sizeof(text), "bola: %.0f%%", conf * 100);
                        putText(img_result, text, Point(x, y-5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,0,0), 1);
                        
                        // send ROS message
                        v2_detection::BallCoordinate bc;
                        bc.pos_x = clamp(x_center_ball/framesize[0]*2-1, -1.0f, 1.0f);
                        bc.pos_y = clamp(y_center_ball/framesize[1]*2-1, -1.0f, 1.0f);
                        bc.obj_size = obj_size_ball;
                        pub_coord.publish(bc);
                        
                        v2_detection::Ballarea ba;
                        ba.ballarea = obj_size_ball;
                        pub_area.publish(ba);
                        
                        v2_detection::BallState bs;
                        bs.ball_status = "FOUND";
                        pub_state.publish(bs);
                        
                        ROS_INFO("bola [%.0f%%] area:%d", conf * 100, obj_size_ball);
                        
                        // extract HSV value
                        get_hsv_val(img);
                        detect_status = FOUND;
                        waktu_sebelum = ros::Time::now().toSec();
                        
                        // update blobsize berdasarkan ball area
                        if(obj_size_ball <= (2800*fisheye)) {
                            blobsize = 320;
                        } else if(obj_size_ball > (2800*fisheye)) {
                            blobsize = 224;
                        }
                        
                        found_ball = true;
                        break;  // PENTING: BREAK SETELAH MENEMUKAN BALL
                    }
                }
                
                // jika tidak menemukan ball
                if(!found_ball) {
                    v2_detection::BallCoordinate bc;
                    bc.pos_x = 0;
                    bc.pos_y = 0;
                    bc.obj_size = 0;
                    pub_coord.publish(bc);
                    
                    v2_detection::BallState bs;
                    bs.ball_status = "NOTFOUND";
                    pub_state.publish(bs);
                    ROS_INFO("bola not found (after yolov5)");
                }
            }
        }

        // ============ DISPLAY ============
        calculate_fps();
        
        char fps_text[30];
        snprintf(fps_text, sizeof(fps_text), "FPS: %.1f", fps);
        putText(img_result, fps_text, Point(5, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
        
        char blob_text[30];
        snprintf(blob_text, sizeof(blob_text), "blobsize: %d", blobsize);
        putText(img_result, blob_text, Point(250, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
        
        char state_text[20];
        snprintf(state_text, sizeof(state_text), "%s", (detect_status == FOUND) ? "FOUND" : "NOTFOUND");
        putText(img_result, state_text, Point(5, 35), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
        
        imshow("VISION_CPP", img_result);
        waitKey(1);
        
        ROS_DEBUG("state: %s", state_text);
        ros::spinOnce();
    }
    
    capture.stop();
    ROS_INFO("sistem berhenti");
    return 0;
}
