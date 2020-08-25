#include "opencvworker.h"
#include <opencv4/opencv2/imgproc/imgproc.hpp>


OpenCvWorker::OpenCvWorker(QObject *parent)
    : QObject(parent)
    , status(false)
    , toggleStream(false)
    , binaryThresholdEnable(false)
    , binaryThreshold(127)
    , opticalFlowEnable(false)
    , videoRestarted(false)
    , faceDetectorEnable(false)
{
    cap = new cv::VideoCapture();

    cv::RNG rng;
    for(int i = 0; i < N; i++)
    {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        random_colors.push_back(cv::Scalar(r,g,b));
    }

//    cascade.load("/home/ksimon/Code/Qt/QtOpenCV/data/haarcascades/haarcascade_frontalface_default.xml");
    cascade.load("/home/ksimon/Code/Qt/QtOpenCV/data/haarcascades/haarcascade_frontalface_alt.xml");

    nestedCascade.load("/home/ksimon/Code/Qt/QtOpenCV/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml");
}


OpenCvWorker::~OpenCvWorker()
{
    if(cap->isOpened())
        cap->release();

    delete cap;
}


void OpenCvWorker::checkIfDeviceAlreadyOpened(const int device)
{
    if(cap->isOpened())
        cap->release();

    cap->open(device);
}


void OpenCvWorker::process_image_cv_to_qt()
{
    (*cap) >> _frameOriginal_old;

    if(_frameOriginal_old.empty())
        return;

    cv::cvtColor(_frameOriginal_old,
                 _frameGray_old,
                 cv::COLOR_BGR2GRAY);

    if(binaryThresholdEnable)
    {
        _frameProcessed_old = _frameGray_old.clone();
    }
    else {
        cv::cvtColor(_frameOriginal_old,
                     _frameProcessed_old,
                     cv::COLOR_BGR2RGB);
    }

    // If video is restarted we need new good feature points.
    if(videoRestarted)
    {
        points_old.clear();
        points.clear();

        goodFeaturesToTrack(_frameGray_old,
                            points_old,
                            N,
                            0.3,
                            7,
                            cv::Mat(),
                            7,
                            false,
                            0.04);

        videoRestarted = false;
    }

    if(opticalFlowEnable)
    {
        (*cap) >> _frameOriginal;

        cv::cvtColor(_frameOriginal,
                     _frameGray,
                     cv::COLOR_BGR2GRAY);

        if(binaryThresholdEnable)
        {
            _frameProcessed = _frameGray.clone();
        }
        else
        {
            cv::cvtColor(_frameOriginal,
                         _frameProcessed,
                         cv::COLOR_BGR2RGB);
        }

        status_opt_flow.clear();
        err_opt_flow.clear();
        // Set termination criteria
        cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS),
                                                     10,
                                                     0.03);
        // Estimate the optical flow
        cv::calcOpticalFlowPyrLK(_frameGray_old,
                                 _frameGray,
                                 points_old,
                                 points,
                                 status_opt_flow,
                                 err_opt_flow,
                                 cv::Size(15,15),
                                 2,
                                 criteria);

        points_good_new.clear();
        for(unsigned int i = 0; i < points_old.size(); i++)
        {
            // Select good points
            if(status_opt_flow[i] == 1)
            {
                points_good_new.push_back(points[i]);
                // draw the tracks
                cv::line(_frameProcessed, points[i], points_old[i], random_colors[i], 2);
                cv::circle(_frameProcessed, points[i], 5, random_colors[i], -1);
            }
        }

        // Now update the previous frame and previous points
        _frameProcessed_old = _frameProcessed.clone();
        _frameGray_old = _frameGray.clone();
        points_old = points_good_new;
    }

    if(binaryThresholdEnable)
    {
        cv::threshold(_frameProcessed_old,
                      _frameProcessed_old,
                      binaryThreshold, 255,
                      cv::THRESH_BINARY);
    }

    if(faceDetectorEnable)
    {
        cv::Mat _frameCopy = _frameOriginal_old.clone();

        if(!mask.empty())
            mask.release();

        mask = cv::Mat::zeros(_frameProcessed_old.size(),
                              _frameProcessed_old.type());

        detectFace(_frameCopy, mask);

        cv::add(_frameProcessed_old, mask, _frameProcessed_old);
    }
}


void OpenCvWorker::detectFace(cv::Mat& img, cv::Mat& mask)
{
    std::vector<cv::Rect> faces, faces2;
    cv::Mat gray, smallImg;

    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY); // Convert to Gray Scale
    double fx = 1 / scale;

    cv::resize(gray, smallImg, cv::Size(), fx, fx, cv::INTER_LINEAR);
    cv::equalizeHist(smallImg, smallImg);

    cascade.detectMultiScale(smallImg,
                             faces,
                             1.1,
                             2,
                             0|cv::CASCADE_SCALE_IMAGE,
                             cv::Size(30, 30));

    std::cout << "Faces detected:   " << faces.size() << std::endl;

    // Draw circles around the faces
    for (std::size_t i = 0; i < faces.size(); i++)
    {
        cv::Rect r = faces[i];
        cv::Mat smallImgROI;
        std::vector<cv::Rect> nestedObjects;
        cv::Point center;
        cv::Scalar color_face = cv::Scalar(255, 0, 0);
        cv::Scalar color_nested = cv::Scalar(0, 255, 0);
        int radius;

        double aspect_ratio = static_cast<double> (r.width/r.height);
        if(0.75 < aspect_ratio && aspect_ratio < 1.3)
        {
            center.x = cvRound((r.x + r.width*0.5)*scale);
            center.y = cvRound((r.y + r.height*0.5)*scale);
            radius = cvRound((r.width + r.height)*0.25*scale);

            circle(mask,
                   center,
                   radius,
                   color_face,
                   3, 8, 0);
        }
        else
            rectangle(mask,
                      cv::Point(cvRound(r.x*scale),
                              cvRound(r.y*scale)),
                      cv::Point(cvRound((r.x + r.width-1)*scale),
                              cvRound((r.y + r.height-1)*scale)),
                      color_face,
                      3, 8, 0);
        if(nestedCascade.empty())
            continue;
        smallImgROI = smallImg(r);

        // Detection of eyes int the input image
        nestedCascade.detectMultiScale(smallImgROI,
                                       nestedObjects,
                                       1.1,
                                       2,
                                       0|cv::CASCADE_SCALE_IMAGE,
                                       cv::Size(30, 30));

        std::cout << "    Structures in face detected:   " << nestedObjects.size() << std::endl;

        // Draw circles around eyes
        for (std::size_t j = 0; j < nestedObjects.size(); j++)
        {
            cv::Rect nr = nestedObjects[j];
            center.x = cvRound((r.x + nr.x + nr.width*0.5)*scale);
            center.y = cvRound((r.y + nr.y + nr.height*0.5)*scale);
            radius = cvRound((nr.width + nr.height)*0.25*scale);

            circle(mask,
                   center,
                   radius,
                   color_nested,
                   3, 8, 0);
        }
    }
}


void OpenCvWorker::receiveGrabFrame()
{
    if(!toggleStream)
        return;

    /*
     * Now process the stream to also get an optical flow field.
     */
    process_image_cv_to_qt();

    if(binaryThresholdEnable)
    {
        QImage output(static_cast<const unsigned char *>(_frameProcessed_old.data),
                      _frameProcessed_old.cols,
                      _frameProcessed_old.rows,
                      QImage::Format_Grayscale8);

        emit sendFrame(output);
    }
    else
    {
        QImage output(static_cast<const unsigned char *>(_frameProcessed_old.data),
                      _frameProcessed_old.cols,
                      _frameProcessed_old.rows,
                      QImage::Format_RGB888);

        emit sendFrame(output);
    }
}


void OpenCvWorker::receiveSetup(const int device)
{
    checkIfDeviceAlreadyOpened(device);
    if(!cap->isOpened()) {
        status = false;
        return;
    }

    status = true;
}


void OpenCvWorker::receiveToggleStream()
{
    toggleStream = !toggleStream;
}


void OpenCvWorker::receiveEnableBinaryThreshold()
{
    binaryThresholdEnable = !binaryThresholdEnable;
}


void OpenCvWorker::receiveBinaryThreshold(int threshold)
{
    binaryThreshold = threshold;
}


void OpenCvWorker::receiveEnableOpticalFlow()
{
    opticalFlowEnable = !opticalFlowEnable;
    videoRestarted = opticalFlowEnable;
}


void OpenCvWorker::receiveEnableFaceDetector()
{
    faceDetectorEnable = !faceDetectorEnable;
}
