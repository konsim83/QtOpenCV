#include "opencvworker.h"
#include <opencv2/imgproc/imgproc.hpp>


OpenCvWorker::OpenCvWorker(QObject *parent)
    : QObject(parent)
    , status(false)
    , toggleStream(false)
    , binaryThresholdEnable(false)
    , binaryThreshold(127)
    , opticalFlowEnable(false)
    , videoRestarted(false)
{
    cap = new cv::VideoCapture();

    cv::RNG rng;
    for(int i = 0; i < 100; i++)
    {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        random_colors.push_back(cv::Scalar(r,g,b));
    }
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
                 _frameProcessed_old,
                 cv::COLOR_BGR2GRAY);

    /*
     * If video is restarted we need new good feature points.
     */
    if(videoRestarted)
    {
        points_old.clear();
        points.clear();

        goodFeaturesToTrack(_frameProcessed_old,
                            points_old,
                            N,
                            0.3,
                            7,
                            cv::Mat(),
                            7,
                            false,
                            0.04);

        if(!mask.empty())
            mask.release();

        mask = cv::Mat::zeros(_frameProcessed_old.size(),
                              _frameProcessed_old.type());

        videoRestarted = false;
    }

    if(opticalFlowEnable)
    {
        (*cap) >> _frameOriginal;

        cv::cvtColor(_frameOriginal,
                     _frameProcessed,
                     cv::COLOR_BGR2GRAY);

        status_opt_flow.clear();
        err_opt_flow.clear();
        // Set termination criteria
        cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS),
                                                     10,
                                                     0.03);
        // Estimate the optical flow
        cv::calcOpticalFlowPyrLK(_frameProcessed_old,
                                 _frameProcessed,
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
                cv::line(mask, points[i], points_old[i], random_colors[i], 2);
                cv::circle(_frameProcessed, points[i], 5, random_colors[i], -1);
            }
        }
        cv::Mat img;
        cv::add(_frameProcessed, mask, img);

        // Now update the previous frame and previous points
        _frameProcessed_old = _frameProcessed.clone();
        points_old = points_good_new;
    }

    if(binaryThresholdEnable)
    {
        cv::threshold(_frameProcessed_old,
                      _frameProcessed_old,
                      binaryThreshold, 255,
                      cv::THRESH_BINARY);
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

    QImage output(static_cast<const unsigned char *>(_frameProcessed_old.data),
                  _frameProcessed_old.cols,
                  _frameProcessed_old.rows,
                  QImage::Format_Grayscale8);

    emit sendFrame(output);
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
