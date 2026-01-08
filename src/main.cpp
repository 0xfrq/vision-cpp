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

/* Params */
constexpr int HSV_FAIL_MAX = 6;
constexpr float POS_ALPHA = 0.35f;
constexpr float AREA_ALPHA = 0.25f;
constexpr float VEL_ALPHA = 0.4f;

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
}

/* =========================
   MAIN
   ========================= */
int main(int argc,char**argv){

    ros::init(argc,argv,"vision_yolo_cpp");
    ros::NodeHandle nh;

    auto pub_state = nh.advertise<v2_detection::BallState>(
        "/DEWO/image_processing/deteksi_bola/ball_state",1);
    auto pub_coord = nh.advertise<v2_detection::BallCoordinate>(
        "/DEWO/image_processing/deteksi_bola/coordinate",1);
    auto pub_area  = nh.advertise<v2_detection::Ballarea>(
        "/DEWO/image_processing/deteksi_bola/ball_area",1);

    string pkg=ros::package::getPath("vision_cpp");
    YoloONNX yolo(pkg+"/src/best.onnx");

    VideoCapture cap(0);
    cap.set(CAP_PROP_FRAME_WIDTH,320);
    cap.set(CAP_PROP_FRAME_HEIGHT,240);

    if(!cap.isOpened()){
        ROS_ERROR("Camera failed");
        return -1;
    }

    auto last_time=chrono::steady_clock::now();
    last_seen=ros::Time::now().toSec();

    while(ros::ok()){

        Mat frame;
        cap>>frame;
        if(frame.empty()) continue;

        /* ===== YOLO ===== */
        if(state==NOTFOUND){

            auto dets=yolo.infer(frame);
            for(auto&d:dets){
                if(d.class_id!=0 || d.conf<0.5f) continue;

                Rect b=d.box;
                Point2f nc(b.x+b.width/2.f,b.y+b.height/2.f);
                int na=b.area();

                if(!initialized){
                    smooth_center=nc;
                    velocity=Point2f(0,0);
                    smooth_area=na;
                    initialized=true;
                }else{
                    Point2f delta=nc-smooth_center;
                    velocity=(1-VEL_ALPHA)*velocity + VEL_ALPHA*delta;
                    smooth_center+=velocity;
                    smooth_area=int(AREA_ALPHA*na+(1-AREA_ALPHA)*smooth_area);
                }

                center=smooth_center;
                ball_area=smooth_area;
                last_box=b;

                int in_area = (ball_area<=5000)? int(sqrt(ball_area)*2.5f) : b.width+35;
                scan_x={int(center.x-in_area),int(center.x+in_area)};

                extractHSV(frame,b);
                last_seen=ros::Time::now().toSec();
                hsv_fail=0;
                state=FOUND;
                break;
            }
        }

        /* ===== HSV TRACK ===== */
        else{

            Mat hsv,mask;
            cvtColor(frame,hsv,COLOR_BGR2HSV);
            inRange(hsv,Scalar(min_h,min_s,min_v),Scalar(max_h,max_s,max_v),mask);

            morphologyEx(mask,mask,MORPH_CLOSE,Mat::ones(5,5,CV_8U));
            morphologyEx(mask,mask,MORPH_OPEN ,Mat::ones(5,5,CV_8U));

            vector<vector<Point>> contours;
            findContours(mask,contours,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);

            bool found=false;

            for(auto&c:contours){
                double a=contourArea(c);
                if(a<ball_area*0.2||a>ball_area*1.3) continue;

                Rect r=boundingRect(c);
                int cx=r.x+r.width/2;
                if(cx<scan_x[0]||cx>scan_x[1]) continue;
                if(a<3000) continue;

                Point2f nc(cx,r.y+r.height/2);
                Point2f delta=nc-smooth_center;
                velocity=(1-VEL_ALPHA)*velocity+VEL_ALPHA*delta;
                smooth_center+=velocity;

                smooth_area=int(AREA_ALPHA*a+(1-AREA_ALPHA)*smooth_area);

                center=smooth_center;
                ball_area=smooth_area;
                last_box=r;

                double timeout=map_value(ball_area,0,76800,0.5,80);
                double now=ros::Time::now().toSec();

                if(now-last_seen<=timeout){
                    last_seen=now;
                    found=true;
                }else{
                    state=NOTFOUND;
                    initialized=false;
                }
                break;
            }

            if(!found){
                hsv_fail++;
                if(hsv_fail>HSV_FAIL_MAX){
                    state=NOTFOUND;
                    initialized=false;
                }
            }else hsv_fail=0;
        }

        /* ===== ROS OUTPUT ===== */
        v2_detection::BallState bs;
        v2_detection::BallCoordinate bc;
        v2_detection::Ballarea ba;

        if(state==FOUND){
            bs.ball_status="FOUND";
            bc.pos_x=clamp(center.x/frame.cols*2-1,-1.f,1.f);
            bc.pos_y=clamp(center.y/frame.rows*2-1,-1.f,1.f);
            bc.obj_size=ball_area;
            ba.ballarea=ball_area;
            pub_coord.publish(bc);
            pub_area.publish(ba);
            rectangle(frame,last_box,Scalar(0,255,255),2);
        }else{
            bs.ball_status="NOTFOUND";
        }

        pub_state.publish(bs);
        imshow("VISION_CPP",frame);
        waitKey(1);

        ros::spinOnce();
    }
    return 0;
}
