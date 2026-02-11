// LAB1.cpp : Defines the entry point for the console application.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


cv::Mat getHist(const cv::Mat& image)
{
	cv::Mat hist = cv::Mat::zeros(1, 256, CV_64FC1);

	double m = 0, M = 0;
	cv::minMaxLoc(hist, &m, &M);
	hist = hist / M;
	cv::Mat hist_img = cv::Mat::zeros(100, 256, CV_8U);
	for (int i = 0; i < 256; i++)
		for (int j = 0; j < 100; j++) {
			if (hist.at<double>(0, i) * 100 > j) {
				hist_img.at<unsigned char>(99 - j, i) = 255;
			}
		}
		cv::bitwise_not(hist_img, hist_img);
		return hist_img;
}

cv::Mat downsample(const cv::Mat& in, size_t q)
{
	size_t inWidth = in.cols, inHeight = in.rows;
	size_t outWidth = inWidth / q, outHeight = inHeight / q;

	cv::Mat out = cv::Mat::zeros(inHeight, inWidth, CV_8U);
	for (size_t i = 0; i < inHeight; ++i) {
		for (size_t j = 0; j < inWidth; ++j) {
			out.at<uchar>(i, j) = in.at<uchar>(i / q * q, j / q * q);
		}
	}
	return out;
}

cv::Mat quantise(const cv::Mat& in, size_t q)
{
	size_t inWidth = in.cols, inHeight = in.rows;
	size_t outWidth = inWidth / q, outHeight = inHeight / q;

	size_t qStep = UINT8_MAX / q;

	cv::Mat out = cv::Mat::zeros(inHeight, inWidth, CV_8U);
	for (size_t i = 0; i < inHeight; ++i) {
		for (size_t j = 0; j < inWidth; ++j) {
			out.at<uchar>(i, j) = in.at<uchar>(i, j) / qStep * qStep;
		}
	}
	return out;
}

int main()
{
	cv::Mat image = cv::imread(R"(resources/mordor.png)", 0);
	if (image.empty())
		return -1;


	cv::Mat hist = getHist(image);
    // image = downsample(image, 10);
	imshow("Picture", image);
	imshow("Histogram", hist);

	std::cout << "Hello World\n";
	std::cout << "OpenCV ver: " << CV_VERSION << "\n";
	cv::waitKey(0);
	cv::moveWindow("Picture", 0, 0);
	cv::moveWindow("Histogram", image.cols, 0);
	return 0;
}

