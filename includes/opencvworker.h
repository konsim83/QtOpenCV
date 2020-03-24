#ifndef OPENCVWORKER_H
#define OPENCVWORKER_H

#include <QObject>
#include <QImage>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

#include <iostream>


class OpenCvWorker : public QObject
{
    Q_OBJECT

private:
    cv::Mat _frameOriginal_old, _frameOriginal;
    cv::Mat _frameGray_old, _frameGray;
    cv::Mat _frameProcessed_old, _frameProcessed;

    cv::VideoCapture *cap;

    cv::CascadeClassifier cascade, nestedCascade;
    const double scale = 0.9;

    std::vector<unsigned char> status_opt_flow;
    std::vector<float> err_opt_flow;

    std::vector<cv::Scalar> random_colors;

    const int N = 100;
    std::vector<cv::Point2f> points_old, points, points_good_new;

    bool status;
    bool toggleStream;

    bool binaryThresholdEnable;
    int binaryThreshold;
    bool opticalFlowEnable;
    bool videoRestarted;
    bool faceDetectorEnable;

    void checkIfDeviceAlreadyOpened(const int device);
    void process_stream_to_optical_flow();
    void process_image_cv_to_qt();

    void detectFace(cv::Mat& img, cv::Mat& mask);
    cv::Mat mask;

public:
    explicit OpenCvWorker(QObject *parent = nullptr);
    ~OpenCvWorker();

signals:
    void sendFrame(QImage frameProcessed);
//    void sendStatus(QString msg, int code);

public slots:
    void receiveGrabFrame();
    void receiveSetup(const int device);
    void receiveToggleStream();

    void receiveEnableBinaryThreshold();
    void receiveEnableOpticalFlow();
    void receiveEnableFaceDetector();
    void receiveBinaryThreshold(int threshold);
};

#endif // OPENCVWORKER_H
