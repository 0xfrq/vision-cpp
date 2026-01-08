#include <ros/ros.h>
#include <ros/package.h>

#include <v2_detection/BallState.h>
#include <v2_detection/BallCoordinate.h>
#include <v2_detection/Ballarea.h>

#include <opencv2/opencv.hpp>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>
#include <thread>
#include <mutex>

#include "yolo_onnx.hpp"

using namespace cv;
using namespace std;

/* =========================
   UTIL
   ========================= */
template<typename T>
inline T clamp(T v, T lo, T hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

double map_value(double v, double smin, double smax, double tmin, double tmax) {
    v = clamp(v, min(smin, smax), max(smin, smax));
    return tmin + (v - smin) * (tmax - tmin) / (smax - smin);
}

/* =========================
   STATE
   ========================= */
enum DetectState { NOTFOUND = 0, FOUND = 1 };
DetectState state = NOTFOUND;

/* HSV */
int min_h=0, min_s=0, min_v=0;
int max_h=0, max_s=0, max_v=0;

/* Tracking */
Point2f center(0,0);
Point2f smooth_center(0,0);
Point2f velocity(0,0);
bool initialized = false;

int ball_area = 0;
int smooth_area = 0;
Rect last_box;
vector<int> scan_x(2,0);

/* Timing */
double last_seen = 0.0;
int hsv_fail = 0;
int yolo_skip_counter = 0;  // Skip YOLO frames when tracking
Point2f last_velocity(0,0);  // For prediction
int consecutive_found = 0;   // Lock-in counter
int consecutive_lost = 0;    // Loss counter

/* Params - Matching Python's direct tracking behavior */
constexpr int HSV_FAIL_MAX = 10;        // Increased tolerance
constexpr float POS_ALPHA = 0.0f;       // Direct position (no smoothing like Python)
constexpr float AREA_ALPHA = 0.0f;      // Direct area (no smoothing like Python)  
constexpr float VEL_ALPHA = 0.0f;       // No velocity prediction (match Python)
constexpr int YOLO_SKIP_FRAMES = 3;     // Skip YOLO when tracking well
constexpr int LOCK_IN_THRESHOLD = 3;    // Frames needed to lock tracking
constexpr int LOCK_OUT_THRESHOLD =20;  // Frames needed to lose tracking

/* FPS Tracking */
int frame_counter = 0;
double fps_start_time = 0.0;
double fps = 0.0;
constexpr double FPS_DISPLAY_INTERVAL = 0.2;

/* =========================
   FPS CALCULATOR
   ========================= */
void calculate_fps() {
    frame_counter++;
    double current_time = ros::Time::now().toSec();
    double elapsed = current_time - fps_start_time;
    
    if(elapsed >= FPS_DISPLAY_INTERVAL) {
        fps = frame_counter / elapsed;
        frame_counter = 0;
        fps_start_time = current_time;
    }
}

/* =========================
   HSV SAMPLING (PY PORT)
   ========================= */
void extractHSV(const Mat& img, const Rect& b) {

    vector<Point> pts;
    int x1=b.x, y1=b.y, x2=b.x+b.width, y2=b.y+b.height;
    Point mid((x1+x2)/2, (y1+y2)/2);

    pts = {
        mid,
        {(x1+mid.x)/2,(y1+mid.y)/2},
        {(x2+mid.x)/2,(y2+mid.y)/2},
        {(mid.x+x2)/2,(mid.y+y1)/2},
        {(x1+mid.x)/2,(y1+y2+mid.y)/2},
        {mid.x,(mid.y+y1+b.height/5)/2},
        {mid.x,(mid.y+y2)/2}
    };

    vector<int> H,S,V;

    for (auto&p:pts){
        int px=clamp(p.x,0,img.cols-1);
        int py=clamp(p.y,0,img.rows-1);
        Vec3b bgr=img.at<Vec3b>(py,px);
        Mat hsv;
        cvtColor(Mat(1,1,CV_8UC3,Scalar(bgr[0],bgr[1],bgr[2])),hsv,COLOR_BGR2HSV);
        Vec3b hv=hsv.at<Vec3b>(0,0);
        H.push_back(hv[0]); S.push_back(hv[1]); V.push_back(hv[2]);
    }

    min_h=*min_element(H.begin(),H.end());
    max_h=min(*max_element(H.begin(),H.end()),33);

    min_s=max(*min_element(S.begin(),S.end()),160);
    max_s=*max_element(S.begin(),S.end());

    min_v=*min_element(V.begin(),V.end());
    max_v=*max_element(V.begin(),V.end());
    
    // Debug output (matching Python)
    ROS_INFO_THROTTLE(2.0, "HSV Range - H:[%d,%d] S:[%d,%d] V:[%d,%d]", 
                     min_h, max_h, min_s, max_s, min_v, max_v);
}

/* =========================
   THREADED VIDEO CAPTURE (matching Python WebcamVideoStream)
   ========================= */
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
        cap.open(src);
        cap.set(CAP_PROP_FPS, 60);
        cap.set(CAP_PROP_FRAME_WIDTH, 320);
        cap.set(CAP_PROP_FRAME_HEIGHT, 240);
        
        if(!cap.isOpened()) {
            ROS_ERROR("Camera failed to open");
            return;
        }
        
        cap >> frame;  // grab first frame
        captureThread = thread(&ThreadedCapture::update, this);
    }
    
    Mat read() {
        lock_guard<mutex> lock(frameMutex);
        return frame.clone();
    }
    
    void stop() {
        stopped = true;
        if(captureThread.joinable()) {
            captureThread.join();
        }
        cap.release();
    }
    
    ~ThreadedCapture() {
        stop();
    }
};

/* =========================
   FIELD MASKING
   ========================= */
Mat extractField(const Mat& img) {
    // Green field detection (adjust HSV values as needed)
    Mat hsv, mask, result;
    cvtColor(img, hsv, COLOR_BGR2HSV);
    
    // Green field HSV range - adjust these values for your field
    Scalar lower(35, 40, 40);
    Scalar upper(85, 255, 255);
    
    inRange(hsv, lower, upper, mask);
    
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5,5));
    erode(mask, mask, kernel, Point(-1,-1), 2);
    dilate(mask, mask, kernel, Point(-1,-1), 5);
    
    // Find largest contour (field)
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_TREE, CHAIN_APPROX_NONE);
    
    Mat fieldMask = Mat::zeros(img.size(), CV_8UC1);
    if(!contours.empty()){
        auto maxContour = *max_element(contours.begin(), contours.end(),
            [](const vector<Point>&a, const vector<Point>&b){
                return contourArea(a) < contourArea(b);
            });
        vector<Point> hull;
        convexHull(maxContour, hull);
        fillConvexPoly(fieldMask, hull, Scalar(255));
    }
    
    bitwise_and(img, img, result, fieldMask);
    return result;
}

/* =========================
   MAIN
   ========================= */
int main(int argc,char**argv){

    ros::init(argc,argv,"vision_yolo_cpp");
    ros::NodeHandle nh;

    auto pub_state = nh.advertise<v2_detection::BallState>(
        "/DEWO/image_processing/deteksi_bola/ball_state",10);
    auto pub_coord = nh.advertise<v2_detection::BallCoordinate>(
        "/DEWO/image_processing/deteksi_bola/coordinate",10);
    auto pub_area  = nh.advertise<v2_detection::Ballarea>(
        "/DEWO/image_processing/deteksi_bola/ball_area",10);

    string pkg=ros::package::getPath("vision_cpp");
    YoloONNX yolo(pkg+"/src/best.onnx");

    // Use threaded capture (matching Python WebcamVideoStream)
    ROS_INFO("Initializing threaded camera capture...");
    ThreadedCapture capture(0);
    
    fps_start_time = ros::Time::now().toSec();
    last_seen=ros::Time::now().toSec();
    
    ROS_INFO("Vision system ready - starting main loop");

    while(ros::ok()){

        Mat frame = capture.read();
        if(frame.empty()) {
            ros::spinOnce();
            continue;
        }
        
        Mat display_frame = frame.clone();
        bool detected = false;
        Point2f det_center;
        int det_area = 0;
        Rect det_box;

        /* ===== DUAL MODE: YOLO + HSV PARALLEL ===== */
        
        // ALWAYS run YOLO (it's fast!) for continuous detection
        auto dets=yolo.infer(frame);
        bool yolo_found = false;
        Point2f yolo_center;
        int yolo_area = 0;
        Rect yolo_box;
        float yolo_conf = 0;
        
        for(auto&d:dets){
            if(d.class_id!=0 || d.conf<0.3f) continue;  // Very low threshold
            yolo_box=d.box;
            yolo_center=Point2f(yolo_box.x+yolo_box.width/2.f,yolo_box.y+yolo_box.height/2.f);
            yolo_area=yolo_box.area();
            yolo_conf=d.conf;
            yolo_found=true;
            break;
        }

        // If tracking, try HSV first (faster than YOLO)
        bool hsv_found = false;
        Point2f hsv_center;
        int hsv_area = 0;
        Rect hsv_box;
        
        if(initialized && state==FOUND){
            // FAST HSV tracking - NO field masking
            Mat hsv;
            cvtColor(frame,hsv,COLOR_BGR2HSV);
            
            Mat mask;
            inRange(hsv,Scalar(min_h,min_s,min_v),Scalar(max_h,max_s,max_v),mask);

            // Fast morphology
            Mat kernel = Mat::ones(3,3,CV_8U);  // Smaller kernel = faster
            morphologyEx(mask,mask,MORPH_CLOSE,kernel);

            vector<vector<Point>> contours;
            findContours(mask,contours,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);

            double best_score = -1;

            for(auto&c:contours){
                double a=contourArea(c);
                if(a<ball_area*0.2||a>ball_area*1.3) continue;  // Reasonable range
                if(a<1800) continue;

                Rect r=boundingRect(c);
                Point2f nc(r.x+r.width/2.f,r.y+r.height/2.f);
                
                // Quick checks only
                int scan_range = ball_area < 5000 ? 120 : 80;
                if(abs(nc.x - center.x) > scan_range) continue;
                
                // Simple scoring: distance + area
                float dist = norm(nc - center);
                float area_ratio = (float)a / ball_area;
                float score = 1.0f / (1.0f + dist/60.0f + abs(area_ratio-1.0f)*3.0f);
                
                if(score > best_score) {
                    best_score = score;
                    hsv_center = nc;
                    hsv_area = (int)a;
                    hsv_box = r;
                    hsv_found = (score > 0.4);  // Minimum threshold
                }
            }
        }

        /* ===== DECISION LOGIC ===== */
        if(hsv_found && state==FOUND) {
            // HSV tracking successful - use it
            detected = true;
            det_center = hsv_center;
            det_area = hsv_area;
            det_box = hsv_box;
            consecutive_found++;
            consecutive_lost=0;
            
            // Update tracking
            center=det_center;
            ball_area=det_area;
            last_box=det_box;
            
            ROS_INFO_THROTTLE(0.3, "HSV Track: Area=%d", det_area);
            
        } else if(yolo_found) {
            // YOLO found ball - use it (either lost HSV or new detection)
            detected = true;
            det_center = yolo_center;
            det_area = yolo_area;
            det_box = yolo_box;
            
            // Update tracking state
            center=det_center;
            ball_area=det_area;
            last_box=det_box;
            initialized=true;
            
            // Re-extract HSV for new ball
            extractHSV(frame,det_box);
            
            consecutive_found++;
            consecutive_lost=0;
            state=FOUND;
            
            ROS_INFO("YOLO Lock: Area=%d Conf=%.2f", det_area, yolo_conf);
            
        } else {
            // Nothing found
            consecutive_lost++;
            consecutive_found=0;
            
            if(consecutive_lost > 8) {  // Quick timeout
                state=NOTFOUND;
                initialized=false;
            }
        }

        /* ===== PUBLISH ===== */
        v2_detection::BallState bs;
        v2_detection::BallCoordinate bc;
        v2_detection::Ballarea ba;
        
        if(detected) {
            bs.ball_status="FOUND";
            bc.pos_x=clamp(det_center.x/frame.cols*2-1,-1.f,1.f);
            bc.pos_y=clamp(det_center.y/frame.rows*2-1,-1.f,1.f);
            bc.obj_size=det_area;
            ba.ballarea=det_area;
            
            pub_state.publish(bs);
            pub_coord.publish(bc);
            pub_area.publish(ba);
        
        /* ===== VISUALIZATION ===== */
        if(detected) {
            rectangle(display_frame, det_box, Scalar(0,255,0), 2);
            circle(display_frame, Point(det_center), 4, Scalar(0,0,255), -1);
        }

        calculate_fps();
        char fps_text[64];
        snprintf(fps_text, sizeof(fps_text), "FPS:%.1f | State:%s", 
                 fps, state==NOTFOUND ? "SEARCH" : "TRACK");
        putText(display_frame, fps_text, Point(5, 15),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);

        imshow("VISION_CPP", display_frame);
        waitKey(1);

        ros::spinOnce();
    }
    
    capture.stop();
    return 0;
}
            pub_state.publish(bs);
        }
        
        // Calculate and display FPS
        calculate_fps();
        
        // Draw FPS and status on display
        char fps_text[50];
        snprintf(fps_text, sizeof(fps_text), "FPS: %.1f", fps);
        putText(display_frame, fps_text, Point(5, 15), 
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
        
        char status_text[50];
        snprintf(status_text, sizeof(status_text), "%s", 
                 state==FOUND ? "TRACKING" : "SEARCHING");
        putText(display_frame, status_text, Point(5, 35), 
                FONT_HERSHEY_SIMPLEX, 0.5, 
                state==FOUND ? Scalar(0, 255, 0) : Scalar(0, 0, 255), 2);
        
        imshow("VISION_CPP", display_frame);
        waitKey(1);

        ros::spinOnce();
    }
    
    capture.stop();
    ROS_INFO("Vision system shutdown");
    return 0;
}
