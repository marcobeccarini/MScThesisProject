// CaliCamCalibration.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#define _CRT_SECURE_NO_WARNINGS

//EM sensor headers
#include <ATC3DG.h>
#include <Sample.h>
//#include <StdAfx.h>

//OpenCV headers
#include <tuple>        
#include <iostream>
#include <string> 
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/stereo.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/bgsegm.hpp>
#include <opencv2/bioinspired.hpp>
#include <opencv2/ccalib.hpp>
#include <opencv2/core_detect.hpp>
#include <opencv2/cvconfig.h>
#include <opencv2/dnn.hpp>
#include <opencv2/dpm.hpp>
#include <opencv2/face.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/fuzzy.hpp>
#include <opencv2/gapi.hpp>
#include <opencv2/hdf.hpp>
#include <opencv2/hfs.hpp>
#include <opencv2/img_hash.hpp>
#include <opencv2/line_descriptor.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/phase_unwrapping.hpp>
#include <opencv2/optflow.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/plot.hpp>
#include <opencv2/rgbd.hpp>
#include <opencv2/saliency.hpp>
#include <opencv2/shape.hpp>
#include <opencv2/stereo.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/structured_light.hpp>
#include <opencv2/superres.hpp>
#include <opencv2/surface_matching.hpp>
#include <opencv2/text.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videostab.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xobjdetect.hpp>
#include <opencv2/xphoto.hpp>
#include <opencv2/ccalib/omnidir.hpp>
#include <algorithm>
#include <filesystem>

//PCL Headers
#include <iostream>
#include <pcl/console/parse.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/keypoints/narf_keypoint.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/keypoints/harris_6d.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/joint_icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/console/parse.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/common_headers.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/ia_ransac.h>

#include <math.h>
#include <cmath>
#include <sstream>
#include <string>


using namespace cv;
using namespace cv::ximgproc;
using namespace std;
std::ostringstream os;

////////////////////////////////////////////////////////////////////////////////

//Field of view, width, height parameters 
int       vfov_bar = 0, width_bar = 0, height_bar = 0, zoom_bar = 0;
int       vfov_max = 60, width_max = 480, height_max = 360, zoom_max = 10;
int       vfov_now = 60, width_now = 480, height_now = 360; //Initializing with these parameters
float zoom_now = 0;

//cols rows width, sgbm, changed (to rectify with new dimensions), matrixes to store cameras parameters
int       cap_cols = 2560, cap_rows = 960 , img_width = cap_cols/2;
bool      changed = false;

////////////////////////////////////////////////////////////////////////////////

//Calibration variables
bool pictures = false, registration = false, allign = false, remapCoords = false;
int i = 0;
const int num_pictures = 100;
const int numPntsRegistration = 200;
vector< vector< Point3f > > objPoints;
vector< vector< Point2f > > imgPointsL, imgPointsR;

int photo_counter = 0;
int total_photos = num_pictures;
int divisor = 2;
cv::Size CHECKERBOARD = { 6, 9 };
bool drawCorners = false;
int minDisp, numDisp, bSize, P1, P2, speckleRange, speckleWindowSize, disp12MaxDiff, preFilterCap, uniquenessRatio, mode;
int lambda, LRC;
float sigma;

struct Matrixes {
	cv::Size DIM;
	cv::Mat KL, KR;
	cv::Mat DL, DR;
	cv::Mat R1, R2, P1, P2, T, R, Q, rmap[2][2];
	vector< vector< Point2f > > left_img_points, right_img_points;
	vector< vector< Point3f > > objPoints;
	cv::Mat Knew;
	cv::Mat rvec, tvec;

};

Matrixes CalibMat;
Matrixes* calibPointer;
float square_size = 0.024;
bool ReMap = false;
string wnd = "Control Window";
string wnd2 = "Control Window Registration";

cv::Mat dispFilteredWLSLeft;
bool threeD = false;

Point pt(-1, -1);
Point ptClick(-1, -1);
Point3f Clicked(-1, -1, -1);
bool newCoords = false;
static double fx;
static double fy;
static double cx;
static double cy;
static double bl;

Eigen::Matrix4f TransformationMatrix;
cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
pcl::IterativeClosestPointNonLinear<pcl::PointXYZ, pcl::PointXYZ> icp;
Ptr<aruco::DetectorParameters> dp = aruco::DetectorParameters::create();
Point Arucho1, Arucho2, Arucho3, Arucho4, centerPoint;

float xRel, yRel, zRel;
std::ofstream myFile("CloudsFile.csv");
std::ofstream myFile2("TransformedCloudFile.csv");
std::ofstream myFile3("Stats.csv");
std::ofstream myFile4("ErrorPrediction.csv");
std::ofstream myFile5("TestDisparityMap.csv");
std::ofstream myFile6("TestRegistration.csv");
std::ofstream myFile7("RealTimePerformance.csv");
int dataFlag = 0;
int picCount = 1;
int trialNumber = 0;
float RealDistance = 0.5;

int statsCounter0 = 0, statsCounter1 = 0;
std::vector<cv::Point2f> projectedPointsLast0(2);
std::vector<cv::Point2f> projectedPointsLast1(2);
std::vector<cv::Point2f> projectedPointsLast2(2);

bool firstTime = 1;
float distanceSensorCamera = 0.0, distanceSensorCamera2 = 0.0;
float distanceTrackerCamera = 0.0, distanceClicked = 0.0;
float distance3D = 0.0, distance3D2 = 0.0;
float distanceDiff = 0.0, distanceDiff2 = 0.0;
float xRelI=0.0, yRelI=0.0, zRelI=0.0;
double overallRegErrI = 0;
int counter = 0;
string res11;
string res12;
string res13;
double focal;

std::vector<cv::Mat> mask(3), masked_data1(3), masked_data2(3) ;

bool testingDisp = false;
int test = 0;
bool Sensor1Saved = false;
bool offSet = false;
bool offSetFlag1 = false;
bool offSet3 = false;

/////////////////////////////////////////////////////////////////////////////////

void OnTrackAngle(int, void*) {
	vfov_now = 60 + vfov_bar;
	changed = true;
}

////////////////////////////////////////////////////////////////////////////////

void OnTrackWidth(int, void*) {
	width_now = (480 + width_bar);
	if (width_now % 2 == 1)
		width_now--;
	changed = true;
}

////////////////////////////////////////////////////////////////////////////////

void OnTrackHeight(int, void*) {
	height_now = (360 + height_bar);
	if (height_now % 2 == 1)
		height_now--;
	changed = true;
}


/////////////////////////////////////////////////////////////////////////////////

float distance(float x1, float y1,
	float z1, float x2,
	float y2, float z2)
{
	float d = sqrt(pow(x2 - x1, 2) +
		pow(y2 - y1, 2) +
		pow(z2 - z1, 2) * 1.0);
	return d;
}

/////////////////////////////////////////////////////////////////////////////////

bool  SingleCameraCalib(vector< vector< Point3f > > objPoints,
	vector< vector< Point2f > > imgPointsL, vector< vector< Point2f > > imgPointsR, Matrixes* calibPointer) {

	cv::Mat imgL, imgR, grayL, grayR;
	int h, w;
	photo_counter = 0;
	vector< Point2f > cornersL, cornersR;
	bool retR = false, retL = false;
	vector< cv::Point3f > objp;
	vector< vector< Point2f > > left_img_points, right_img_points;

	for (int i = 0; i < CHECKERBOARD.height; ++i)
		for (int j = 0; j < CHECKERBOARD.width; ++j)
			objp.push_back(Point3f(float((float)j * square_size), float((float)i * square_size), 0)); 

	while (photo_counter != total_photos) {
		cout << "Import pair No " + std::to_string(photo_counter + 1) << endl;
		string leftName = "LeftImages/left" + std::to_string(photo_counter) + ".jpg";
		string rightName = "RightImages/right" + std::to_string(photo_counter) + ".jpg";
		photo_counter = photo_counter + 1;

		imgL = cv::imread(leftName, 1);
		h = imgL.rows;
		w = imgL.cols;
		cv::cvtColor(imgL, grayL, cv::COLOR_BGR2GRAY);

		imgR = cv::imread(rightName, 1);
		cv::cvtColor(imgR, grayR, cv::COLOR_BGR2GRAY);

		retL = cv::findChessboardCorners(grayL, CHECKERBOARD, cornersL, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_FAST_CHECK + cv::CALIB_CB_NORMALIZE_IMAGE);
		retR = cv::findChessboardCorners(grayR, CHECKERBOARD, cornersR, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_FAST_CHECK + cv::CALIB_CB_NORMALIZE_IMAGE);

		if (drawCorners) {
			cv::drawChessboardCorners(grayL, CHECKERBOARD, cornersL, retL);
			cv::imshow("Corners Left", grayL);
			cv::drawChessboardCorners(grayR, CHECKERBOARD, cornersR, retR);
			cv::imshow("Corners Right", grayR);
			char key = cv::waitKey(0);
			if (key == 'q')
				exit(0);
		}


		if ((retL == true) && (retR == true)) {
			cv::cornerSubPix(grayL, cornersL, cv::Size(3, 3), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
			cv::cornerSubPix(grayR, cornersR, cv::Size(3, 3), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
			imgPointsL.push_back(cornersL);
			imgPointsR.push_back(cornersR);
			objPoints.push_back(objp);
		}
		else {
			cout << "Pair No" + to_string(photo_counter) + " ignored, as no chessboard found" << endl;
			continue;
		}

	}

	for (int i = 0; i < imgPointsL.size(); i++) {
		vector< Point2f > v1, v2;
		for (int j = 0; j < imgPointsL[i].size(); j++) {
			v1.push_back(Point2d((double)imgPointsL[i][j].x, (double)imgPointsL[i][j].y));
			v2.push_back(Point2d((double)imgPointsR[i][j].x, (double)imgPointsR[i][j].y));
		}
		left_img_points.push_back(v1);
		right_img_points.push_back(v2);
	}

	int N_ok = objPoints.size();
	cv::Mat KL, KR;
	vector<cv::Mat> TvecsR, TvecsL, RvecsL, RvecsR;
	cv::Mat DL, DR;
	cv::Size DIM = grayL.size();

	int calibFlags = fisheye::CALIB_RECOMPUTE_EXTRINSIC + fisheye::CALIB_FIX_SKEW;
	cv::TermCriteria CalibCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 200, 0.000001);

	if (!objPoints.empty()) {

		double rmsL = cv::fisheye::calibrate(objPoints, left_img_points, DIM, KL, DL, RvecsL, TvecsL, calibFlags, CalibCriteria);

		cout << "Found " << N_ok << " images for calibration" << endl;
		cout << "DIM: " << DIM << endl;
		cout << "KL: " << KL << endl;
		cout << "DL: " << DL << endl;
		cout << "RMS Left: " << rmsL << endl;

		double rmsR = cv::fisheye::calibrate(objPoints, right_img_points, DIM, KR, DR, RvecsR, TvecsR, calibFlags, CalibCriteria);

		cout << "Found " << N_ok << " images for calibration" << endl;
		cout << "DIM: " << DIM << endl;
		cout << "KR: " << KR << endl;
		cout << "DR: " << DR << endl;
		cout << "RMS Right: " << rmsR << endl;


		myFile4 << " " << ", " << "Left" << ", " << rmsL << ", " << square_size << "\n";
		myFile4 << trialNumber << ", " << "Right" << ", " << rmsR << ", " << square_size << "\n";

		(*calibPointer).DIM = DIM;
		(*calibPointer).DL = DL;
		(*calibPointer).DR = DR;
		(*calibPointer).KR = KR;
		(*calibPointer).KL = KL;
		(*calibPointer).left_img_points = left_img_points;
		(*calibPointer).right_img_points = right_img_points;
		(*calibPointer).objPoints = objPoints;
		(*calibPointer).tvec = TvecsL[TvecsL.size()-1];

		return  true;
	}

	else {
		cout << "Chessboard not found in any image, re perform calibration" << endl;
		return false;
	}
}

/////////////////////////////////////////////////////////////////////////////////

void StereoCameraCalibration(vector< vector< Point2f > > left_img_points, vector< vector< Point2f > > right_img_points,
	vector< vector< Point3f > > objPoints, cv::Mat KL, cv::Mat DL, cv::Mat KR, cv::Mat DR, cv::Size DIM, cv::Mat R, cv::Mat T) {

	cout << "Running Stereo Calibration..." << endl;

	int stereoCalibFlags = cv::CALIB_FIX_INTRINSIC;
	cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 200, 0.000001);

	int N_ok = left_img_points.size();

	double rmsStereo = cv::fisheye::stereoCalibrate(objPoints, left_img_points, right_img_points, KL, DL, KR, DL, DIM, R, T, stereoCalibFlags, criteria);

	cout << "Found " << N_ok << " images for calibration" << endl;
	cout << "DIM: " << DIM << endl;
	cout << "KL: " << KL << endl;
	cout << "KR: " << KR << endl;
	cout << "DL: " << DL << endl;
	cout << "DR: " << DR << endl;
	cout << "RMS Stereo: " << rmsStereo << endl;
	(*calibPointer).R = R;
	(*calibPointer).T = T;
	myFile4 << " " << ", " << "Stereo" << ", " << rmsStereo << ", " << square_size << "\n";

}

/////////////////////////////////////////////////////////////////////////////////

static void mouseHandler(int event, int x, int y, int flags, void* param) {
	if (event == EVENT_LBUTTONDBLCLK) {
		std::cout << "Clicked, position : (" << x << " , " << y << ")" << endl;
		ptClick.x = x;
		ptClick.y = y;
	}
}

/////////////////////////////////////////////////////////////////////////////////

void Rectify() {

	double vfov_rad = vfov_now * CV_PI / 180.;
	focal = height_now / 2. / tan(vfov_rad / 2.);

	CalibMat.Knew = (cv::Mat_<double>(3, 3) << focal, 0., width_now / 2. - 0.5,
		0., focal, height_now / 2. - 0.5,
		0., 0., 1.);

	// How to get the depth map
	fx = CalibMat.Knew.at<double>(0, 0);
	fy = CalibMat.Knew.at<double>(1, 1);
	cx = CalibMat.Knew.at<double>(0, 2);
	cy = CalibMat.Knew.at<double>(1, 2);
	bl = -CalibMat.T.at<double>(0, 0);

	Mat rmap[2][2];

	(*calibPointer).rmap[0][0] = rmap[0][0];
	(*calibPointer).rmap[0][1] = rmap[0][1];
	(*calibPointer).rmap[1][0] = rmap[1][0];
	(*calibPointer).rmap[1][1] = rmap[1][1];

	cv::Size img_size(480, 360);

	cv::Mat1f NullVector;

	//Precompute maps for cv::remap()
	fisheye::initUndistortRectifyMap(CalibMat.KL, CalibMat.DL /*NullVector*/, Mat::eye(3, 3, CV_64F), CalibMat.Knew, img_size, CV_16SC2, CalibMat.rmap[0][0], CalibMat.rmap[0][1]);
	fisheye::initUndistortRectifyMap(CalibMat.KR, CalibMat.DR /*NullVector*/, Mat::eye(3, 3, CV_64F), CalibMat.Knew, img_size, CV_16SC2, CalibMat.rmap[1][0], CalibMat.rmap[1][1]);
	//fisheye::initUndistortRectifyMap(CalibMat.KL, CalibMat.DL /*NullVector*/, CalibMat.R1, CalibMat.P1, img_size, CV_16SC2, CalibMat.rmap[0][0], CalibMat.rmap[0][1]);
	//fisheye::initUndistortRectifyMap(CalibMat.KR, CalibMat.DR /*NullVector*/, CalibMat.R2, CalibMat.P2, img_size, CV_16SC2, CalibMat.rmap[1][0], CalibMat.rmap[1][1]);

	std::cout << "Width: " << width_now << "\t"
		<< "Height: " << height_now << "\t"
		<< "V.Fov: " << vfov_now << "\n";
	std::cout << "K Matrix: \n" << CalibMat.Knew << std::endl;

	ReMap = true;
}

/////////////////////////////////////////////////////////////////////////////////

void Calibrate() {

	bool worked = false;
	calibPointer = &CalibMat;

	cv::Mat undistortedL, undistortedR;

	cout << "Main cycle start" << endl;
	cout << "Left and Right Camera Calibration..." << endl;
	worked = SingleCameraCalib(objPoints, imgPointsL, imgPointsR, calibPointer);

	if (worked) {

		StereoCameraCalibration(CalibMat.left_img_points, CalibMat.right_img_points, CalibMat.objPoints, CalibMat.KL, CalibMat.DL, CalibMat.KR, CalibMat.DR, CalibMat.DIM, CalibMat.R, CalibMat.T);

		FileStorage fs("intrinsics.yml", FileStorage::WRITE);
		if (fs.isOpened())
		{
			fs << "KL" << CalibMat.KL << "DL" << CalibMat.DL <<
				"KR" << CalibMat.KR << "DR" << CalibMat.DR;
			fs.release();
		}
		else
			cout << "Error: can not save the intrinsic parameters\n";

		cv::fisheye::stereoRectify(CalibMat.KL, CalibMat.DL,
			CalibMat.KR, CalibMat.DR,
			CalibMat.DIM, CalibMat.R, CalibMat.T,
			CalibMat.R1, CalibMat.R2, CalibMat.P1, CalibMat.P2,
			CalibMat.Q, cv::CALIB_ZERO_DISPARITY, CalibMat.DIM, 1);


		fs.open("extrinsics.yml", FileStorage::WRITE);
		if (fs.isOpened())
		{
			fs << "R" << CalibMat.R << "T" << CalibMat.T << "R1" << CalibMat.R1 << "R2" << CalibMat.R2 << "P1" << CalibMat.P1 << "P2" << CalibMat.P2 << "Q" << CalibMat.Q;
			fs.release();
		}
		else
			cout << "Error: can not save the extrinsic parameters\n";

		Rectify();

	}

	else {

		cout << "Single Camera Calibration didn't work, retry" << endl;
	}

}	

/////////////////////////////////////////////////////////////////////////////////

void CreateTrackbars() {

	/*cv::namedWindow(wnd);
	cv::resizeWindow(wnd, 600, 600);

	cv::createTrackbar("minDisp", wnd, 0, 200);
	cv::createTrackbar("numDisp 16*", wnd, 0, 30);
	cv::createTrackbar("block Size", wnd, 0, 20);
	cv::createTrackbar("Speckle Range", wnd, 0, 20);
	cv::createTrackbar("Speckle Window Size", wnd, 0, 20);
	cv::createTrackbar("disp12MaxDiff", wnd, 0, 150);
	cv::createTrackbar("P1", wnd, 0, 100);
	cv::createTrackbar("P2", wnd, 0, 100);
	cv::createTrackbar("PreFilterCap", wnd, 0, 100);
	cv::createTrackbar("UniquenessRatio", wnd, 0, 30);
	cv::createTrackbar("lambda", wnd, 0, 100000);
	cv::createTrackbar("mode", wnd, 0, 3);
	cv::createTrackbar("lambda", wnd, 0, 100000);
	cv::createTrackbar("sigma", wnd, 0, 50);
	cv::createTrackbar("LRC", wnd, 0, 200);
	


	cv::setTrackbarPos("minDisp", wnd, 105);
	cv::setTrackbarPos("numDisp 16*", wnd, 4);
	cv::setTrackbarPos("block Size", wnd, 0);
	cv::setTrackbarPos("Speckle Range", wnd, 20);
	cv::setTrackbarPos("Speckle Window Size", wnd, 0);
	cv::setTrackbarPos("disp12MaxDiff", wnd, 0);
	cv::setTrackbarPos("P1", wnd, 48);
	cv::setTrackbarPos("P2", wnd, 100);
	cv::setTrackbarPos("PreFilterCap", wnd, 100);
	cv::setTrackbarPos("uniquenessRatio", wnd, 0);
	cv::setTrackbarPos("mode", wnd, 2);
	cv::setTrackbarPos("lambda", wnd, 80000);
	cv::setTrackbarPos("sigma", wnd, 9);
	cv::setTrackbarPos("LRC", wnd, 200);*/
	

	minDisp = -100 + 105;
	numDisp = (16 + 16 * (4));
	bSize = 0;
	P1 = 8 * 48;
	P2 = 32 * 100;
	speckleRange = 20;
	speckleWindowSize = 0;
	disp12MaxDiff = 0;
	preFilterCap = 100;
	uniquenessRatio = 0;
	mode = 2;
	lambda = 80000;
	sigma = (float)(9/float(10));
	LRC = 200;


}

/////////////////////////////////////////////////////////////////////////////////

void CreateTrackbars2() {
	cv::namedWindow(wnd2);
	cv::resizeWindow(wnd2, 1000, 1000);

	cv::createTrackbar("adaptiveThreshWinSizeMin", wnd2, 0, 30);
	cv::createTrackbar("adaptiveThreshWinSizeMax", wnd2, 0, 30);
	cv::createTrackbar("adaptiveThreshWinSizeStep", wnd2, 0, 30);
	cv::createTrackbar("adaptiveThreshConstant", wnd2, 0, 20);
	cv::createTrackbar("minMarkerPerimeterRate", wnd2, 0, 10);
	cv::createTrackbar("maxMarkerPerimeterRate", wnd2, 0, 20);
	cv::createTrackbar("polygonalApproxAccuracyRate", wnd2, 0, 20);
	cv::createTrackbar("minCornerDistanceRate", wnd2, 0, 20);
	cv::createTrackbar("minDistanceToBorder", wnd2, 0, 20);
	cv::createTrackbar("minMarkerDistanceRate", wnd2, 0, 20);
	cv::createTrackbar("doCornerRefinement", wnd2, 0, 1);
	cv::createTrackbar("cornerRefinementWinSize", wnd2, 0, 20);
	cv::createTrackbar("cornerRefinementMaxIterations", wnd2, 0, 50);
	cv::createTrackbar("cornerRefinementMinAccuracy", wnd2, 0, 10);
	cv::createTrackbar("markerBorderBits", wnd2, 0, 10);
	cv::createTrackbar("perpectiveRemovePixelPerCell", wnd2, 0, 20);
	cv::createTrackbar("perspectiveRemoveIgnoredMarginPerCell", wnd2, 0, 20);
	cv::createTrackbar("maxErroneousBitsInBorderRate", wnd2, 0, 100);
	cv::createTrackbar("minOtsuStdDev", wnd2, 0, 20);
	cv::createTrackbar("errorCorrectionRate", wnd2, 0, 10);


	cv::setTrackbarPos("adaptiveThreshWinSizeMin", wnd2, 3);
	cv::setTrackbarPos("adaptiveThreshWinSizeMax", wnd2, 23);
	cv::setTrackbarPos("adaptiveThreshWinSizeStep", wnd2, 10);
	cv::setTrackbarPos("adaptiveThreshConstant", wnd2, 7);
	cv::setTrackbarPos("minMarkerPerimeterRate", wnd2, 3);
	cv::setTrackbarPos("maxMarkerPerimeterRate", wnd2, 4);
	cv::setTrackbarPos("polygonalApproxAccuracyRate", wnd2, 5);
	cv::setTrackbarPos("minCornerDistanceRate", wnd2, 5);
	cv::setTrackbarPos("minDistanceToBorder", wnd2, 3);
	cv::setTrackbarPos("minMarkerDistanceRate", wnd2, 5);
	cv::setTrackbarPos("doCornerRefinement", wnd2, 1);
	cv::setTrackbarPos("cornerRefinementWinSize", wnd2, 5);
	cv::setTrackbarPos("cornerRefinementMaxIterations", wnd2, 30);
	cv::setTrackbarPos("cornerRefinementMinAccuracy", wnd2, 1);
	cv::setTrackbarPos("markerBorderBits", wnd2, 1);
	cv::setTrackbarPos("perpectiveRemovePixelPerCell", wnd2, 8);
	cv::setTrackbarPos("perspectiveRemoveIgnoredMarginPerCell", wnd2, 13);
	cv::setTrackbarPos("maxErroneousBitsInBorderRate", wnd2, 35);
	cv::setTrackbarPos("minOtsuStdDev", wnd2, 5);
	cv::setTrackbarPos("errorCorrectionRate", wnd2, 6);

	dp->adaptiveThreshWinSizeMin = (double)cv::getTrackbarPos("adaptiveThreshWinSizeMin", wnd2);
	dp->adaptiveThreshWinSizeMax = cv::getTrackbarPos("adaptiveThreshWinSizeMax", wnd2);
	dp->adaptiveThreshWinSizeStep = cv::getTrackbarPos("adaptiveThreshWinSizeStep", wnd2);
	dp->adaptiveThreshConstant = cv::getTrackbarPos("adaptiveThreshConstant", wnd2);
	dp->minMarkerPerimeterRate = (float)cv::getTrackbarPos("minMarkerPerimeterRate", wnd2)/100;
	dp->polygonalApproxAccuracyRate = (float)cv::getTrackbarPos("polygonalApproxAccuracyRate", wnd2);
	dp->minCornerDistanceRate = (float)cv::getTrackbarPos("minCornerDistanceRate", wnd2)/100;
	dp->minDistanceToBorder = cv::getTrackbarPos("minDistanceToBorder", wnd2);
	dp->minMarkerDistanceRate = (float)cv::getTrackbarPos("minMarkerDistanceRate", wnd2)/100;
	//dp->doCornerRefinement = cv::getTrackbarPos("doCornerRefinement", wnd2);
	dp->cornerRefinementWinSize = cv::getTrackbarPos("cornerRefinementWinSize", wnd2);
	dp->cornerRefinementMaxIterations = cv::getTrackbarPos("cornerRefinementMaxIterations", wnd2);
	dp->cornerRefinementMinAccuracy = (float)cv::getTrackbarPos("cornerRefinementMinAccuracy", wnd2)/10;
	dp->markerBorderBits = cv::getTrackbarPos("markerBorderBits", wnd2);
	dp->perspectiveRemovePixelPerCell = cv::getTrackbarPos("perpectiveRemovePixelPerCell", wnd2);
	dp->perspectiveRemoveIgnoredMarginPerCell = (float)cv::getTrackbarPos("perspectiveRemoveIgnoredMarginPerCell", wnd2)/100;
	dp->maxErroneousBitsInBorderRate = (float)cv::getTrackbarPos("maxErroneousBitsInBorderRate", wnd2)/100;
	dp->minOtsuStdDev = (float)cv::getTrackbarPos("minOtsuStdDev", wnd2);
	dp->errorCorrectionRate = (float)cv::getTrackbarPos("errorCorrectionRate", wnd2)/10;
	
}

/////////////////////////////////////////////////////////////////////////////////

void DisparityImage(const cv::Mat& recl, const cv::Mat& recr, cv::Mat& dispFilteredLeft) {

	cv::Mat disp16sL, disp16sR, reclGRAY, recrGRAY;
	Ptr<StereoSGBM> left_matcher;

	/*minDisp = -100 + getTrackbarPos("minDisp", wnd);
	numDisp = (16 + 16 * getTrackbarPos("numDisp 16*", wnd));
	bSize = getTrackbarPos("block Size", wnd);
	P1 = 8 * cv::getTrackbarPos("P1", wnd);
	P2 = 32 * cv::getTrackbarPos("P2", wnd);
	speckleRange = getTrackbarPos("Speckle Range", wnd);
	speckleWindowSize = cv::getTrackbarPos("Speckle Window Size", wnd);
	disp12MaxDiff = cv::getTrackbarPos("disp12MaxDiff", wnd);
	preFilterCap = cv::getTrackbarPos("PreFilterCap", wnd);
	uniquenessRatio = cv::getTrackbarPos("uniquenessRatio", wnd);
	mode = cv::getTrackbarPos("mode", wnd);
	lambda = cv::getTrackbarPos("lambda", wnd);
	sigma = (float)(cv::getTrackbarPos("sigma", wnd) / float(10));
	LRC = cv::getTrackbarPos("LRC", wnd);*/

	left_matcher = cv::StereoSGBM::create(minDisp, numDisp, bSize,
		P1, P2, disp12MaxDiff, preFilterCap,
		uniquenessRatio, speckleWindowSize, speckleRange, mode);
	
	int h, w;
	h = recl.rows;
	w = recl.cols;
	
	cvtColor(recl, reclGRAY, COLOR_BGR2GRAY);
	cvtColor(recr, recrGRAY, COLOR_BGR2GRAY);

	left_matcher->compute(reclGRAY, recrGRAY, disp16sL);

	Ptr<StereoMatcher> right_matcher;
	right_matcher = createRightMatcher(left_matcher);
	
	right_matcher->compute(recrGRAY, reclGRAY, disp16sR);

	//WLS Filter
	Ptr<DisparityWLSFilter> wls_filter;

	wls_filter = createDisparityWLSFilter(left_matcher);
	wls_filter->setLambda(lambda);
	wls_filter->setSigmaColor(sigma);
	wls_filter->setLRCthresh(LRC); 

	imshow("disparityImageUnnormalized", disp16sL);

	double minVal, maxVal;
	minMaxLoc(disp16sR, &minVal, &maxVal);
	cv::Mat dispBO;
	disp16sL.copyTo(dispBO);
	dispBO.convertTo(dispBO, CV_8UC1, 255 / (maxVal - minVal));
	imshow("disparityImageNormalized", dispBO);

	wls_filter->filter(disp16sL, recl, dispFilteredLeft, disp16sR, Rect(), recr);


	cv::Mat conf_map;

	conf_map = wls_filter->getConfidenceMap();
	imshow("conf MAP", conf_map);

	
	dispFilteredLeft.convertTo(dispFilteredWLSLeft, CV_32F, 1.f / 16.f);
	cv::setMouseCallback("Color disp LEFT", mouseHandler, 0);

}

/////////////////////////////////////////////////////////////////////////////////

void errorHandler(int error);

//////////////////////////////////////////////////////////////////////////////////
int main() {

	CSystem			ATC3DG;
	CSensor* pSensor;
	CXmtr* pXmtr;
	int				errorCode;
	int				i;
	int				sensorID;
	short			id;
	int				records = 100;
	char			output[256];
	int				numberBytes;
	BOOL metric = true; // metric reporting enabled
	BOOL reset = true; // metric reporting enabled
	BYTE autoconfig = 4; // metric reporting enabled
	double rate = 80.0; // 80 Hz
	int sample = 0;

	printf("Creating CSV File to store data\n");
	myFile3 << "Disparity Map" << ", " << " " << ", " << " " << "," << " " << "," << "Sensor" << ", " << " " << ", " << " " << "\n";

	myFile5 << "X1" << ", " << "Y1" << ", " << "Z1" << ", " << "X2" << ", " << "Y2" << ", " << "Z2" << ", " << "Detected Distance" << ", " << "Real Distance" << "\n";
	myFile6 << "X1DISP" << ", " << "Y1DISP" << ", " << "Z1DISP" << ", " << "X2DISP" << ", " << "Y2DISP" << ", " << "Z2DISP"  << ", " << "X1SENS" << ", " << "Y1SENS" << ", " << "Z1SENS" << ", " << "X2SENS" << ", " << "Y2SENS" << ", " << "Z2SENS" << ", " << "Detected Distance DISP 1-2" << ", " << "Detected Distance SENS 1-2" << ", " << "Real Distance" << "\n";
	
	myFile7 << "Sample" << ", " << "FPS" << endl;


	projectedPointsLast0.push_back(cv::Point(0, 0));
	projectedPointsLast0.push_back(cv::Point(0, 0));
	projectedPointsLast1.push_back(cv::Point(0, 0));
	projectedPointsLast1.push_back(cv::Point(0, 0));
	projectedPointsLast2.push_back(cv::Point(0, 0));
	projectedPointsLast2.push_back(cv::Point(0, 0));



	printf("Initializing ATC3DG system...\n");

	errorCode = SetSystemParameter(RESET, &reset, sizeof(reset));
	if (errorCode != BIRD_ERROR_SUCCESS) errorHandler(errorCode);

	errorCode = SetSystemParameter(AUTOCONFIG, &autoconfig, sizeof(autoconfig));
	if (errorCode != BIRD_ERROR_SUCCESS) errorHandler(errorCode);

	errorCode = InitializeBIRDSystem();
	if (errorCode != BIRD_ERROR_SUCCESS) errorHandler(errorCode);

	errorCode = SetSystemParameter(METRIC, &metric, sizeof(metric));
	if (errorCode != BIRD_ERROR_SUCCESS) errorHandler(errorCode);

	errorCode = SetSystemParameter(MEASUREMENT_RATE, &rate, sizeof(rate));
	if (errorCode != BIRD_ERROR_SUCCESS) errorHandler(errorCode);
	
	errorCode = GetBIRDSystemConfiguration(&ATC3DG.m_config);
	if (errorCode != BIRD_ERROR_SUCCESS) errorHandler(errorCode);
	

	pSensor = new CSensor[ATC3DG.m_config.numberSensors];
	for (i = 0; i < ATC3DG.m_config.numberSensors; i++)
	{
		errorCode = GetSensorConfiguration(i, &(pSensor + i)->m_config);
		if (errorCode != BIRD_ERROR_SUCCESS) errorHandler(errorCode);
	}

	cout << "Number of sensors: " << ATC3DG.m_config.numberSensors << endl;
	
	
	

	pXmtr = new CXmtr[ATC3DG.m_config.numberTransmitters];
	for (i = 0; i < ATC3DG.m_config.numberTransmitters; i++)
	{
		errorCode = GetTransmitterConfiguration(i, &(pXmtr + i)->m_config);
		if (errorCode != BIRD_ERROR_SUCCESS) errorHandler(errorCode);
	}

	for (id = 0; id < ATC3DG.m_config.numberTransmitters; id++)
	{
		if ((pXmtr + id)->m_config.attached)
		{
			// Transmitter selection is a system function.
			// Using the SELECT_TRANSMITTER parameter we send the id of the
			// transmitter that we want to run with the SetSystemParameter() call
			errorCode = SetSystemParameter(SELECT_TRANSMITTER, &id, sizeof(id));
			if (errorCode != BIRD_ERROR_SUCCESS) errorHandler(errorCode);
			break;
		}
	}

	DOUBLE_POSITION_ANGLES_RECORD record, * pRecord = &record;

	cv::Mat raw_img;
	cv::VideoCapture vcapture;
	bool retR = false, retL = false, retL2 = false, retR2 = false;
	cv::Mat resizedL, resizedR, resizedLshow, resizedRshow;
	vector< Point2f > cornersL, cornersR, cornersL2, cornersR2;
	int h, w;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>);

	vcapture.open(1);

	if (!vcapture.isOpened()) {
		std::cout << "Camera doesn't work" << std::endl;
		exit(-1);
	}

	vcapture.set(cv::CAP_PROP_FRAME_WIDTH, cap_cols);
	vcapture.set(cv::CAP_PROP_FRAME_HEIGHT, cap_rows);
	cout << ATC3DG.m_config.numberSensors << endl;

	char win_name[256];
	char win_name2[256];
	char win_name3[256];

	sprintf(win_name, "Raw Image: %d x %d", img_width, cap_rows);
	sprintf(win_name2, "Rectified Left Image: %d x %d", img_width, cap_rows);
	sprintf(win_name3, "Rectified Right Image : %d x %d", img_width, cap_rows);

	std::string param_win_name(win_name);

	cv::namedWindow(param_win_name);


	cv::createTrackbar("Zoom", param_win_name,
		&zoom_bar, zoom_max);
	cv::createTrackbar("V. FoV:60 +", param_win_name,
		&vfov_bar, vfov_max, OnTrackAngle);
	cv::createTrackbar("Width:480 +", param_win_name,
		&width_bar, width_max, OnTrackWidth);
	cv::createTrackbar("Height:360 +", param_win_name,
		&height_bar, height_max, OnTrackHeight);

	cv::Mat raw_imgl, raw_imgr, rect_imgl, rect_imgr;
	CreateTrackbars();
	//CreateTrackbars2();
	double fps;
	
	// Number of frames to capture
	int num_frames = 1;
	// Start and end times
	time_t start, end;

	// Variable for storing video frames
	Mat frame;

	// Start time
	time(&start);
	i = 0;

	while (1) {

		num_frames = num_frames++;

		if (changed) {
			if (ReMap) {
				Rectify();
				changed = false;
			}

		}
		
		vcapture >> raw_img;

		if (raw_img.total() == 0) {
			std::cout << "Image capture error" << std::endl;
			exit(-1);
		}


		raw_img(cv::Rect(0, 0, img_width, cap_rows)).copyTo(raw_imgl);
		raw_img(cv::Rect(img_width, 0, img_width, cap_rows)).copyTo(raw_imgr);

		cv::Mat small_imgL, small_imgR, small_imgRectL, small_imgRectR;
		cv::resize(raw_imgl, small_imgL, cv::Size(), 0.35, 0.35);
		cv::resize(raw_imgr, small_imgR, cv::Size(), 0.35, 0.35);

		char key = cv::waitKey(1);
		if (key == 'q' || key == 'Q' || key == 27)
			break;

		if (key == 'c') {
			printf("Creating CSV File to store Prediction Results\n");
			myFile4 << "Trial Number" << ", " << "Calibration" << ", " << "RMS" << ", " << "Size of Square " << "\n";

			pictures = true;
			i = 0;
			//Calibrate();
			time(&start);
			num_frames = 1;
			trialNumber++;
			
		}

		if (pictures) {

			h = raw_imgl.rows;
			w = raw_imgl.cols;
			
			retL = cv::findChessboardCorners(raw_imgl, CHECKERBOARD, cornersL, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_FAST_CHECK + cv::CALIB_CB_NORMALIZE_IMAGE);
			retR = cv::findChessboardCorners(raw_imgr, CHECKERBOARD, cornersR, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_FAST_CHECK + cv::CALIB_CB_NORMALIZE_IMAGE);
			cv::drawChessboardCorners(raw_imgl, CHECKERBOARD, cornersL, retL);
			cv::drawChessboardCorners(raw_imgr, CHECKERBOARD, cornersR, retR);
			cv::resize(raw_imgl, resizedLshow, cv::Size(), 0.4, 0.4);
			cv::resize(raw_imgr, resizedRshow, cv::Size(), 0.4, 0.4);

			vconcat(resizedLshow, resizedRshow, resizedLshow);

			cv::imshow("Corners", resizedLshow);

			if (retL && retR) {
				string pathL = "LeftImages/left" + std::to_string(i) + ".jpg";
				string pathR = "RightImages/right" + std::to_string(i) + ".jpg";
				i++;
				cv::imwrite(pathL, raw_imgl);
				cv::imwrite(pathR, raw_imgr);
			}

			
			

			if (i == num_pictures) {

				pictures = false;
				ReMap = true;
				Calibrate();
			}
		}

		if (ReMap) {
			cv::Mat image(120, 700, CV_8UC3, cv::Scalar(0, 0, 0)), dispFilteredColorLeft, dispFilteredColorRight;
			cv::Mat image2(120, 700, CV_8UC3, cv::Scalar(0, 0, 0));
			cv::Mat image3(120, 700, CV_8UC3, cv::Scalar(0, 0, 0));
			vector<vector<Point> > contours;
			
			cv::remap(raw_imgl, rect_imgl, CalibMat.rmap[0][0], CalibMat.rmap[0][1], INTER_LINEAR, BORDER_CONSTANT);
			cv::remap(raw_imgr, rect_imgr, CalibMat.rmap[1][0], CalibMat.rmap[1][1], INTER_LINEAR, BORDER_CONSTANT);

			Mat dispFilteredLeft;

			DisparityImage(rect_imgl, rect_imgr, dispFilteredLeft);

			double minVal, maxVal;
			minMaxLoc(dispFilteredLeft, &minVal, &maxVal);
			dispFilteredLeft.convertTo(dispFilteredLeft, CV_8UC1, 255 / (maxVal - minVal));
			imshow("FilteredDispMap", dispFilteredLeft);
			applyColorMap(dispFilteredLeft, dispFilteredColorLeft, COLORMAP_JET);

			Mat out;
			rect_imgl.copyTo(out);
			cv::setMouseCallback("out", mouseHandler, 0);
			std::vector<int> ids;
			std::vector<std::vector<cv::Point2f> > corners;
			cv::aruco::detectMarkers(rect_imgl, dictionary, corners, ids);
			

			// if at least one marker detected
			if (ids.size() > 0) {
				//Point Arucho1, Arucho2, centerPoint;
				Arucho1.x = (corners[0][0].x + corners[0][2].x) / 2;
				Arucho1.y = (corners[0][0].y + corners[0][2].y) / 2;
				circle(out, Arucho1, 8, Scalar(255, 0, 0), -1, 1, 0);
				

				if (ids.size() > 1) {
					Arucho2.x = (corners[1][0].x + corners[1][2].x) / 2;
					Arucho2.y = (corners[1][0].y + corners[1][2].y) / 2;
					circle(out, Arucho2, 8, Scalar(255, 0, 0), -1, 1, 0);


					if (ids.size() > 2) {
						Arucho3.x = (corners[2][0].x + corners[2][2].x) / 2;
						Arucho3.y = (corners[2][0].y + corners[2][2].y) / 2;
						circle(out, Arucho3, 8, Scalar(255, 0, 0), -1, 1, 0);
						

						if (ids.size() > 3) {
							Arucho4.x = (corners[3][0].x + corners[3][2].x) / 2;
							Arucho4.y = (corners[3][0].y + corners[3][2].y) / 2;
							circle(out, Arucho3, 8, Scalar(255, 0, 0), -1, 1, 0);
							line(out, Arucho1, Arucho2, Scalar(255, 0, 0), 1, 1, 0);
							line(out, Arucho2, Arucho3, Scalar(255, 0, 0), 1, 1, 0);
							line(out, Arucho3, Arucho4, Scalar(255, 0, 0), 1, 1, 0);
							line(out, Arucho4, Arucho1, Scalar(255, 0, 0), 1, 1, 0);
							line(out, Arucho1, Arucho3, Scalar(255, 0, 0), 1, 1, 0);
							line(out, Arucho4, Arucho2, Scalar(255, 0, 0), 1, 1, 0);
							centerPoint.x = (Arucho1.x + Arucho2.x + Arucho3.x + Arucho4.x) / 4;
							centerPoint.y = (Arucho1.y + Arucho2.y + Arucho3.y + Arucho4.y) / 4;
							circle(out, centerPoint, 8, Scalar(255, 0, 0), -1, 1, 0);
							circle(dispFilteredColorLeft, centerPoint, 8, Scalar(255, 0, 0), -1, 1, 0);
							pt.x = centerPoint.x;
							pt.y = centerPoint.y;

						}

						else {
							pt.x = -1;
							pt.y = -1;

						}

					}
					else {
						pt.x = -1;
						pt.y = -1;

					}
				}
				else {
					pt.x = -1;
					pt.y = -1;

				}
			}
			else {
				pt.x = -1;
				pt.y = -1;

			}

 

			char key = cv::waitKey(1);

			/*if (key == '3') {
				threeD = true;
			}

			if (threeD == true) {
				cv::Mat output3D;
				reprojectImageTo3D(dispFilteredLeft, output3D, CalibMat.Q, false, -1);
				//This will be your reference camera
				viz::Viz3d myWindow("Coordinate Frame");
				myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem());
				//Store your point clouds here
				//vector<cv::Point3d> pts3d1, pts3d2;
				viz::WCloud cloud_widget1(output3D, viz::Color::green());
				//viz::WCloud cloud_widget2(pts3d2, viz::Color::red());

				myWindow.showWidget("cloud 1", cloud_widget1);
				//myWindow.showWidget("cloud 2", cloud_widget2);

				myWindow.spin();
				threeD == false;
			}*/
			/*if (key == 't') {
				testingDisp = true;
				cout << "SAVING STATS FROM DISP MAP " << endl;
			}

			if (testingDisp) {

				if (ids.size() > 0) {
					//Point Arucho1, Arucho2, centerPoint;
					Arucho1.x = (corners[0][0].x + corners[0][2].x) / 2;
					Arucho1.y = (corners[0][0].y + corners[0][2].y) / 2;
					circle(out, Arucho1, 8, Scalar(255, 0, 0), -1, 1, 0);

					double dispWLS1 = dispFilteredWLSLeft.at<float>(Arucho1.y, Arucho1.x);
					double e1 = (Arucho1.x - cx) / fx;
					double f1 = (Arucho1.y - cy) / fy;
					double depthWLS1 = fx * bl / dispWLS1;
					double X1 = e1 * depthWLS1;
					double Y1 = f1 * depthWLS1;
					double Z1 = depthWLS1;



					if (ids.size() > 1) {
						Arucho2.x = (corners[1][0].x + corners[1][2].x) / 2;
						Arucho2.y = (corners[1][0].y + corners[1][2].y) / 2;
						circle(out, Arucho2, 8, Scalar(255, 0, 0), -1, 1, 0);
						line(out, Arucho1, Arucho2, Scalar(255, 0, 0), 1, 1, 0);
						line(dispFilteredColorLeft, Arucho1, Arucho2, Scalar(255, 0, 0), 1, 1, 0);


						double dispWLS2 = dispFilteredWLSLeft.at<float>(Arucho2.y, Arucho2.x);
						double e2 = (Arucho2.x - cx) / fx;
						double f2 = (Arucho2.y - cy) / fy;
						double depthWLS2 = fx * bl / dispWLS2;
						double X2 = e2 * depthWLS2;
						double Y2 = f2 * depthWLS2;
						double Z2 = depthWLS2;

						float distanceTestCalc = distance(X1, Y1, Z1, X2, Y2, Z2);
						float realDistTest = 0.15;

						myFile5 << X1 << ", " << Y1 << ", " << Z1 << ", " << X2 << ", " << Y2 << ", " << Z2 << ", " << distanceTestCalc << ", " << realDistTest << "\n";
						test++;
						if (test == 100) {
							test = 0;
							testingDisp = false;
							cout << " FINISHED SAVING STATS FROM DISP MAP " << endl;

						}

					}
				}
			}*/
			// sensor attached so get record
			/*errorCode = GetAsynchronousRecord(sensorID, pRecord, sizeof(record));
			if (errorCode != BIRD_ERROR_SUCCESS) { errorHandler(errorCode); }


			// get the status of the last data record
			// only report the data if everything is okay
			unsigned int status = GetSensorStatus(sensorID);
			if (status == VALID_STATUS)
			{
				cout << "X: " << record.x << " Y: "<< record.y << " Z: " << record.z << endl;

			}*/

			
			DOUBLE_POSITION_RECORD offsetSens1;
			offsetSens1.x = 0.0;
			offsetSens1.y = 0.0;
			offsetSens1.z = 0.23622;
			errorCode = SetSensorParameter(0, SENSOR_OFFSET, &offsetSens1, sizeof(offsetSens1));
			if (errorCode != BIRD_ERROR_SUCCESS) errorHandler(errorCode);
			
			/*DOUBLE_POSITION_RECORD offsetSens2;
			offsetSens2.x = 7.87402;
			offsetSens2.y = 0.0;
			offsetSens2.z = 0.0590551;
			errorCode = SetSensorParameter(1, SENSOR_OFFSET, &offsetSens2, sizeof(offsetSens2));
			if (errorCode != BIRD_ERROR_SUCCESS) errorHandler(errorCode);*/
			
			/*DOUBLE_POSITION_RECORD offsetSens2;
			offsetSens2.x = 7.87402;
			offsetSens2.y = 0.0;
			offsetSens2.z = 0.0590551;
			errorCode = SetSensorParameter(1, SENSOR_OFFSET, &offsetSens2, sizeof(offsetSens2));
			if (errorCode != BIRD_ERROR_SUCCESS) errorHandler(errorCode);

			DOUBLE_POSITION_RECORD offsetSens3;
			offsetSens3.x = 7.87402;
			offsetSens3.y = 0.0;
			offsetSens3.z = 0.0590551;
			errorCode = SetSensorParameter(1, SENSOR_OFFSET, &offsetSens2, sizeof(offsetSens2));
			if (errorCode != BIRD_ERROR_SUCCESS) errorHandler(errorCode);*/



			if (key == 'r') {

				myFile.close();
				myFile2.close();

				// Create an output filestream object
				printf("Creating CSV File to store data\n");
				myFile.open("CloudsFile.csv", ios::trunc);
				myFile2.open("TransformedCloudFile.csv", ios::trunc);
				myFile << "Sensor Cloud" << ", " << " " << ", " << " "<< "," << "Sensor Cloud" <<  "\n";
				myFile << "X" << ", " << "Y" << ", " << "Z" << "," << "X" << "," << "Y" << "," << "Z" << "\n";
				myFile2 << "Transformed Sensor Cloud" << "\n";
				myFile2 << "X" << ", " << "Y" << ", " << "Z" << "\n";

				xRel = 0;
				yRel = 0;
				zRel = 0;

				registration = true;
				remapCoords = false;
				cout << "Registration active!" << endl;
				i = 0;
				time(&start);
				num_frames = 1;

			}

			if (registration) {
								
				if (ids.size() > 3) {
					newCoords = true;
				}
				if (i == numPntsRegistration) {
					registration = false;
					allign = true;

				}
					
			}

			if (newCoords) //Generation of the two point clouds
			{
				double dispWLS = dispFilteredWLSLeft.at<float>(pt.y, pt.x);
				double e = (pt.x - cx) / fx;
				double f = (pt.y - cy) / fy;
				double depthWLS = fx * bl / dispWLS;
				double xWLS = e * depthWLS;
				double yWLS = f * depthWLS;
				double zWLS = depthWLS;

				std::string textWLS, textPC;
	
				if (dispWLS <= 0.f)
					textWLS = std::string("Coordinates WLS:") + std::string("Non detected");
				else {

					textWLS = std::string("Tracked X: ") + std::to_string(xWLS) + std::string("   Y: ") + std::to_string(yWLS) + std::string("   Z: ") + std::to_string(zWLS);

					i++;

					

					pcl::PointXYZ pointDisp;

					pointDisp.x = (xWLS);
					pointDisp.y = (yWLS);
					pointDisp.z = (depthWLS);

					sensorID = 0;

					// sensor attached so get record
					errorCode = GetAsynchronousRecord(sensorID, pRecord, sizeof(record));
					if (errorCode != BIRD_ERROR_SUCCESS) { errorHandler(errorCode); }

					// get the status of the last data record
					// only report the data if everything is okay
					unsigned int status = GetSensorStatus(sensorID);

					if (status == VALID_STATUS)
					{
						pcl::PointXYZ pointSens;


						myFile << (record.y) / 1000 << ", " << (record.z) / 1000 << ", " << (record.x) / 1000 << ",";

						//Here I already apply the rotation of the sensor coordinate system w/ respect to the camera coordinate system
						pointSens.z = (record.x) / 1000;
						pointSens.x = (record.y) / 1000;
						pointSens.y = (record.z) / 1000;

						if (pt.x != -1) {
							cloud_in->push_back(pointSens);
							myFile << pointDisp.x << ", " << pointDisp.y << ", " << pointDisp.z << "\n";
							textPC = std::string("Num points in point cloud: ") + std::to_string(i);
							cloud_out->push_back(pointDisp);
						}
						
					}

				
				}
				putText(image, textPC, cvPoint(10, 20), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, cv::LINE_AA);
				putText(image, textWLS, cvPoint(10, 40), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, cv::LINE_AA);
				newCoords = false;
			}   


			if (allign)  //Allignment of the two point clouds
			{ 
				
				cout << "Cloud IN (Sens): " << endl;
				for (auto& pointSens : *cloud_in)
					std::cout << pointSens << std::endl;

				pcl::PLYWriter writer;
				writer.write("SensorCloud.ply", *cloud_in);


				cout << "Cloud OUT (Disp): " << endl;

				for (auto& pointDisp : *cloud_out)
					std::cout << pointDisp << std::endl;

				writer.write("DispCloud.ply", *cloud_out);
				
				// ------------------------------------
				// -----Compute the normals------------
				// ------------------------------------
				
				/*pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
				ne.setInputCloud(cloud_in);
				pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new
					pcl::search::KdTree<pcl::PointXYZ>());
				ne.setSearchMethod(tree);

				// Output datasets
				pcl::PointCloud<pcl::Normal>::Ptr normals_input(new pcl::PointCloud<pcl::Normal>);
				ne.setRadiusSearch(0.05);
				ne.compute(*normals_input);

				// Copy the xyz info from cloud_xyz and add it to cloud_normals as the xyz field in PointNormals estimation is zero
				/*for (size_t i = 0; i < normals_input->points.size(); ++i)
				{
					normals_input->points[i].x = cloud_in->points[i].x;
					normals_input->points[i].y = cloud_in->points[i].y;
					normals_input->points[i].z = cloud_in->points[i].z;
				}*/
				
				/*pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne2;
				ne2.setInputCloud(cloud_out);
				ne2.setSearchMethod(tree);
				pcl::PointCloud<pcl::Normal>::Ptr normals_output(new pcl::PointCloud<pcl::Normal>);
				ne2.setRadiusSearch(0.05);
				ne2.compute(*normals_output);
				// Copy the xyz info from cloud_xyz and add it to cloud_normals as the xyz field in PointNormals estimation is zero
				/*for (size_t i = 0; i < normals_output->points.size(); ++i)
				{
					normals_output->points[i].x = cloud_out->points[i].x;
					normals_output->points[i].y = cloud_out->points[i].y;
					normals_output->points[i].z = cloud_out->points[i].z;
				}*/

				/*for (auto& pointDisp : *normals_output)
					std::cout << pointDisp << std::endl;*/
				// ------------------------------------
				 // -----Compute the keypoints----------
				 // ------------------------------------
				/*const float min_scale = 0.00001f;
				const int n_octaves = 20;
				const int n_scales_per_octave = 10;
				const float min_contrast = 0.00001f;

				pcl::SIFTKeypoint<pcl::PointNormal, pcl::PointXYZ> detector_source;
				pcl::PointCloud<pcl::PointXYZ>::Ptr resultSource;
				pcl::search::KdTree<pcl::PointNormal>::Ptr tree2(new pcl::search::KdTree<pcl::PointNormal>());
				detector_source.setSearchMethod(tree2);
				detector_source.setScales(min_scale, n_octaves, n_scales_per_octave);
				detector_source.setMinimumContrast(min_contrast);
				detector_source.setInputCloud(normals_input);

				detector_source.compute(*resultSource);
				cout << "CIAOOOO: " << resultSource <<endl;

				std::cout << "No of SIFT points in the result are " << resultSource->points.size() << std::endl;
				/*for (size_t i = 0; i < resultSource->points.size(); ++i)
				{
					resultSource->points[i].x = cloud_in->points[i].x;
					resultSource->points[i].y = cloud_in->points[i].y;
					resultSource->points[i].z = cloud_in->points[i].z;
				}

				pcl::SIFTKeypoint<pcl::PointNormal, pcl::PointXYZ> detector_target;
				pcl::PointCloud<pcl::PointXYZ>::Ptr resultTarget;
				detector_target.setSearchMethod(tree2);
				detector_target.setScales(min_scale, n_octaves, n_scales_per_octave);
				detector_target.setMinimumContrast(min_contrast);
				detector_target.setInputCloud(normals_output);
				detector_target.compute(*resultTarget);
				
				std::cout << "No of SIFT points in the result are " << resultTarget->points.size() << std::endl;
				/*for (size_t i = 0; i < resultTarget->points.size(); ++i)
				{
					resultTarget->points[i].x = cloud_out->points[i].x;
					resultTarget->points[i].y = cloud_out->points[i].y;
					resultTarget->points[i].z = cloud_out->points[i].z;
				}*/

				

				// ------------------------------------
				// -----Compute the features ----------
				// ------------------------------------
				

				/*pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh_source;
				pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_source(new
					pcl::search::KdTree<pcl::PointXYZ>);
				fpfh_source.setInputCloud(cloud_in);
				fpfh_source.setInputNormals(normals_input);
				fpfh_source.setSearchMethod(tree_source);

				pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs_source(new
					pcl::PointCloud<pcl::FPFHSignature33>());
				fpfh_source.setRadiusSearch(0.5);
				fpfh_source.compute(*fpfhs_source);

				cout << "FINO A QUI TUTTO OK   1" << endl;


				pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh_target;
				pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_target(new
					pcl::search::KdTree<pcl::PointXYZ>);
				fpfh_target.setInputCloud(cloud_out);
				fpfh_target.setInputNormals(normals_output);
				fpfh_target.setSearchMethod(tree_target);

				pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs_target(new
					pcl::PointCloud<pcl::FPFHSignature33>());
				fpfh_source.setRadiusSearch(0.5);
				fpfh_source.compute(*fpfhs_target);

			
				pcl::SampleConsensusInitialAlignment < pcl::PointXYZ, pcl::PointXYZ,
					pcl::FPFHSignature33> sac_ia;

				sac_ia.setMinSampleDistance(0.1);
				sac_ia.setMaxCorrespondenceDistance(1);
				sac_ia.setMaximumIterations(1000);

				sac_ia.setInputCloud(cloud_in);
				sac_ia.setInputTarget(cloud_out);

				sac_ia.setSourceFeatures(fpfhs_source);
				sac_ia.setTargetFeatures(fpfhs_target);
				pcl::PointCloud<pcl::PointXYZ> Final;

				sac_ia.align(Final);
				Eigen::Matrix4f matrix = sac_ia.getFinalTransformation();
				TransformationMatrix = sac_ia.getFinalTransformation();
				std::cout << matrix;*/

				icp.setInputSource(cloud_in);
				icp.setInputTarget(cloud_out);
				icp.setMaxCorrespondenceDistance(0.5);
				icp.setTransformationEpsilon(1e-10);
				icp.setEuclideanFitnessEpsilon(0.000000001);
				icp.setMaximumIterations(100);
				icp.setRANSACOutlierRejectionThreshold(1);
				//icp.setUseReciprocalCorrespondences(true);
				pcl::PointCloud<pcl::PointXYZ> Final2;

				icp.align(Final2);

				if (icp.hasConverged()) {
					

					writer.write("TransformedCloud2.ply", Final2);

					std::cout << "Cloud in has " << cloud_in->size() << " points" << endl;
					std::cout << "Cloud out has " << cloud_out->size() << " points" << endl;

					TransformationMatrix = icp.getFinalTransformation();

					std::cout << "has converged:" << icp.hasConverged() << " score: " <<
						icp.getFitnessScore() << std::endl;
					std::cout << icp.getFinalTransformation() << std::endl;

					pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>());

					pcl::transformPointCloud(*cloud_in, *transformed_cloud, TransformationMatrix);
					for (auto& pointSensTrans : *transformed_cloud)
						myFile2 << pointSensTrans.x << ", " << pointSensTrans.y << ", " << pointSensTrans.z << "\n";

					for (int i = 0; i < transformed_cloud->size(); i++) {
						xRel = xRel + (double)(abs(cloud_out->at(i).x - transformed_cloud->at(i).x) / abs(cloud_out->at(i).x)) * 100;
						yRel = yRel + (double)(abs(cloud_out->at(i).y - transformed_cloud->at(i).y) / abs(cloud_out->at(i).y)) * 100;
						zRel = zRel + (double)(abs(cloud_out->at(i).z - transformed_cloud->at(i).z) / abs(cloud_out->at(i).z)) * 100;

					}

					xRel = xRel / numPntsRegistration;
					yRel = yRel / numPntsRegistration;
					zRel = zRel / numPntsRegistration;

					double overallRegErr = 0;

					overallRegErr = (xRel + yRel + zRel) / 3;

					std::cout << "Last Registration error percentages: " << std::endl;
					std::cout << "Overall registration: " << overallRegErr << "%" << std::endl;
					std::cout << "X Axis: " << xRel << "%" << std::endl;
					std::cout << "Y Axis: " << yRel << "%" << std::endl;
					std::cout << "Z Axis: " << zRel << "%" << std::endl;;

					xRel = 0;
					yRel = 0;
					zRel = 0;

					cloud_in->clear();
					cloud_out->clear();

					allign = false;
					remapCoords = true;
					time(&start);
					num_frames = 1;
					myFile4 << " " << ", " << "Registration Overall Error" << ", " << overallRegErr << " %" <<  "\n";

				}
				else {
					cout << "ICP has not converged" << endl;
					allign = false;
				}
					

				
			}

			if (ptClick.x != -1 && ptClick.y != -1) {

				string textClicked;
				double dispWLS = dispFilteredWLSLeft.at<float>(ptClick.y, ptClick.x);
				double e = (ptClick.x - cx) / fx;
				double f = (ptClick.y - cy) / fy;
				double depthWLS = fx * bl / dispWLS;
				double xWLS = e * depthWLS;
				double yWLS = f * depthWLS;
				double zWLS = depthWLS;
				Clicked.x = xWLS;
				Clicked.y = yWLS;
				Clicked.z = zWLS;

				float distance3DClicked = distance(0, 0, 0, xWLS, yWLS, zWLS);
				if (dispWLS <= 0.f)
					textClicked = std::string("Coordinates Clicked:") + std::string("Non detected");
				else {
					//text = std::string("Depth:") + std::to_string(z) + std::string("meters");
					textClicked = std::string("CLICKED X: ") + std::to_string(xWLS) + std::string("  Y: ") + std::to_string(yWLS) + std::string("  Z: ") + std::to_string(zWLS) + std::string("  3D dist: ") + std::to_string(distance3DClicked);
					


					/*if (statsCounter < 10) {
						myFile3 << xWLS << ", " << yWLS << ", " << zWLS << "\n";
						statsCounter++;
						printf("Saving stats\n");
					}
					else {
						statsCounter = 0;
						myFile3 << "X" << ", " << "Y" << ", " << "Z" << "\n";
						ptClick.x = -1;
						ptClick.y = -1;

					}*/
						
				}

				putText(image3, textClicked, cvPoint(10, 80), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, cv::LINE_AA);
				cv::circle(dispFilteredColorLeft, ptClick, 10, (255, 0, 0), -1, 8, 0);


			}


			if (remapCoords) {

				mask[0] = cv::Mat::zeros(out.size(), CV_8UC1);
				mask[1] = cv::Mat::zeros(out.size(), CV_8UC1);
				mask[2] = cv::Mat::zeros(out.size(), CV_8UC1);
				masked_data1[0] = cv::Mat::zeros(out.size(), CV_8UC1);
				masked_data1[1] = cv::Mat::zeros(out.size(), CV_8UC1);
				masked_data1[2] = cv::Mat::zeros(out.size(), CV_8UC1);
				masked_data2[0] = cv::Mat::zeros(out.size(), CV_8UC1);
				masked_data2[1] = cv::Mat::zeros(out.size(), CV_8UC1);
				masked_data2[2] = cv::Mat::zeros(out.size(), CV_8UC1);

				pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud0(new pcl::PointCloud<pcl::PointXYZ>());
				pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud1(new pcl::PointCloud<pcl::PointXYZ>());
				pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud2(new pcl::PointCloud<pcl::PointXYZ>());


				for (sensorID = 0; sensorID < ATC3DG.m_config.numberSensors - 1; sensorID++)
				{
					pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>());
					pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloudTEMP(new pcl::PointCloud<pcl::PointXYZ>());

					// sensor attached so get record
					errorCode = GetAsynchronousRecord(sensorID, pRecord, sizeof(record));
					if (errorCode != BIRD_ERROR_SUCCESS) { errorHandler(errorCode); }


					// get the status of the last data record
					// only report the data if everything is okay
					unsigned int status = GetSensorStatus(sensorID);
					if (status == VALID_STATUS)
					{
						pcl::PointXYZ pointSens;

						pointSens.z = record.x / 1000;
						pointSens.x = record.y / 1000;
						pointSens.y = record.z / 1000;


						source_cloud->push_back(pointSens);
						pcl::transformPointCloud(*source_cloud, *transformed_cloudTEMP, TransformationMatrix);


					}
					

					if (sensorID == 0) {
						*transformed_cloud0 = *transformed_cloudTEMP;
					}
					if (sensorID == 1) {
						*transformed_cloud1 = *transformed_cloudTEMP;
					}
					if (sensorID == 2) {
						*transformed_cloud2 = *transformed_cloudTEMP;
					}

				}



				char key = cv::waitKey(1);
				if (key == 'd') {
					dataFlag = 1;
					cout << "Data flag activated" << endl;

					myFile3 << "X" << ", " << "Y" << ", " << "Z" << "," << "3dPosition"
						<< "," << "X" << ", " << "Y" << ", " << "Z" << "," << "3dPosition"
						<< "," << "Distance" << "," << "xRel" << "," << "yRel" << "," << "zRel" << "," << "Overall Point Error"
						<< "," << "Picture" << "\n";

					myFile4 << " " << ", " << "Sample" << ", "
						<< "Mock-up optic disk Distance" << ", " << "Tooltip distance" << ", "
						<< ", " << "Distance detected" << ", "
						<< "Real Distance Observed" << ", " <<"Dimensional error" << ", " << "Final Mock-up Macroscale error" << ", "
						<< "Microscale Square Size" << ", " << "Microscale error" << ", "
						<< "Mock up scale" << ", " << "Distance in Scale" << ", "
						<< "Dimensional Error1" << ", " << "Dimensional Error 2"  << ", " << "Avarage Dimensional Error" << "\n";
					RealDistance = RealDistance - 0.1;
							
				}

				if (key == 't') {
				testingDisp = true;
				cout << "SAVING STATS FROM DISP MAP And Registration " << endl;
				}

				if (key == 'o') {

					offSet = true;
					cout << "OFFSET SENS 1 FLAG" << endl;
				}

				if (key == 'p') {

					offSet3 = true;
					cout << "OFFSET SENS 2 FLAG" << endl;
				}
				
				if (offSet) {

					if (ids.size() > 0 && ids[0] == 23) {

						//Point Arucho1, Arucho2, centerPoint;
						Arucho1.x = (corners[0][0].x + corners[0][2].x) / 2;
						Arucho1.y = (corners[0][0].y + corners[0][2].y) / 2;
						circle(out, Arucho1, 8, Scalar(0, 255, 0), -1, 1, 0);

						double dispWLS1 = dispFilteredWLSLeft.at<float>(Arucho1.y, Arucho1.x);
						double e1 = (Arucho1.x - cx) / fx;
						double f1 = (Arucho1.y - cy) / fy;
						double depthWLS1 = fx * bl / dispWLS1;
						double X1 = e1 * depthWLS1;
						double Y1 = f1 * depthWLS1;
						double Z1 = depthWLS1;

						cout << "COORD ARUCO 23: " << X1 << ", " << Y1 << ", " << Z1 << endl;

						if (transformed_cloud1->size() == 1) {
							double X1SENS;
							double Y1SENS;
							double Z1SENS;
							DOUBLE_POSITION_RECORD offsetSens2;

							X1SENS = transformed_cloud1->at(0).x;
							Y1SENS = transformed_cloud1->at(0).y;
							Z1SENS = transformed_cloud1->at(0).z;


							cout << "COORD SENS 1: " << X1SENS << ", " << Y1SENS << ", " << Z1SENS << endl;
							

							cout <<  "OFFSET SENS 1 : " <<  (X1 - X1SENS) << ", " << (Y1 - Y1SENS) << ", " << (Z1 - Z1SENS) << endl;

							offsetSens2.y =  (X1 - X1SENS) * 39.37;
							offsetSens2.x =  (Y1 - Y1SENS)* 39.37;
							offsetSens2.z = (Z1 - Z1SENS) * 39.37;



							errorCode = SetSensorParameter(1, SENSOR_OFFSET, &offsetSens2, sizeof(offsetSens2));
							if (errorCode != BIRD_ERROR_SUCCESS) errorHandler(errorCode);

							// sensor attached so get record
							errorCode = GetAsynchronousRecord(1, pRecord, sizeof(record));
							if (errorCode != BIRD_ERROR_SUCCESS) { errorHandler(errorCode); }
							pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>());
							pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloudTEMP(new pcl::PointCloud<pcl::PointXYZ>());

							// get the status of the last data record
							// only report the data if everything is okay
							unsigned int status = GetSensorStatus(1);
							if (status == VALID_STATUS)
							{
								pcl::PointXYZ pointSens;

								pointSens.z = record.x / 1000;
								pointSens.x = record.y / 1000;
								pointSens.y = record.z / 1000;


								source_cloud->push_back(pointSens);
								pcl::transformPointCloud(*source_cloud, *transformed_cloudTEMP, TransformationMatrix);


							}
							
							cout << "AFTER OFFSETTING SENS 1 COORD: " << transformed_cloudTEMP->at(0).x << ", " << transformed_cloudTEMP->at(0).y << ", " << transformed_cloudTEMP->at(0).z << "\n";

						}
					}
					offSet = false;
				}				

				if (offSet3) {
					if (ids.size() > 0 && ids[0] == 24) {

						Arucho2.x = (corners[0][0].x + corners[0][2].x) / 2;
						Arucho2.y = (corners[0][0].y + corners[0][2].y) / 2;
						circle(out, Arucho2, 8, Scalar(255, 0, 0), -1, 1, 0);
						line(out, Arucho1, Arucho2, Scalar(255, 0, 0), 1, 1, 0);
						line(dispFilteredColorLeft, Arucho1, Arucho2, Scalar(255, 0, 0), 1, 1, 0);


						double dispWLS2 = dispFilteredWLSLeft.at<float>(Arucho2.y, Arucho2.x);
						double e2 = (Arucho2.x - cx) / fx;
						double f2 = (Arucho2.y - cy) / fy;
						double depthWLS2 = fx * bl / dispWLS2;
						double X2 = e2 * depthWLS2;
						double Y2 = f2 * depthWLS2;
						double Z2 = depthWLS2;
						cout << "COORD ARUCO 24: " << X2 << ", " << Y2 << ", " << Z2 << endl;

						if (transformed_cloud2->size() == 1) {
							DOUBLE_POSITION_RECORD offsetSens3;


							double X2SENS = transformed_cloud2->at(0).x;
							double Y2SENS = transformed_cloud2->at(0).y;
							double Z2SENS = transformed_cloud2->at(0).z;
							cout << "COORD SENS 2: " << X2SENS << ", " << Y2SENS << ", " << Z2SENS << endl;
							cout << "OFFSET SENS 2 : " << (X2 - X2SENS) << ", " << (Y2 - Y2SENS) << ", " << (Z2 - Z2SENS) << endl;

							offsetSens3.y = (X2 - X2SENS) * 39.37;
							offsetSens3.x = (Y2 - Y2SENS) * 39.37;
							offsetSens3.z = (Z2 - Z2SENS) * 39.37;
							errorCode = SetSensorParameter(2, SENSOR_OFFSET, &offsetSens3, sizeof(offsetSens3));
							if (errorCode != BIRD_ERROR_SUCCESS) errorHandler(errorCode);

							// sensor attached so get record
							errorCode = GetAsynchronousRecord(2, pRecord, sizeof(record));
							if (errorCode != BIRD_ERROR_SUCCESS) { errorHandler(errorCode); }
							pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>());
							pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloudTEMP(new pcl::PointCloud<pcl::PointXYZ>());

							// get the status of the last data record
							// only report the data if everything is okay
							unsigned int status = GetSensorStatus(1);
							if (status == VALID_STATUS)
							{
								pcl::PointXYZ pointSens;

								pointSens.z = record.x / 1000;
								pointSens.x = record.y / 1000;
								pointSens.y = record.z / 1000;


								source_cloud->push_back(pointSens);
								pcl::transformPointCloud(*source_cloud, *transformed_cloudTEMP, TransformationMatrix);


							}

							cout << "AFTER OFFSETTING SENS 2 COORD: " << transformed_cloudTEMP->at(0).x << ", " << transformed_cloudTEMP->at(0).y << ", " << transformed_cloudTEMP->at(0).z << "\n";



						}

					}
					offSet3 = false;
				}

				if (testingDisp) {

					if (ids.size() > 0) {
						Point Arucho1, Arucho2, centerPoint;
						Arucho1.x = (corners[0][0].x + corners[0][2].x) / 2;
						Arucho1.y = (corners[0][0].y + corners[0][2].y) / 2;
						circle(out, Arucho1, 8, Scalar(255, 0, 0), -1, 1, 0);

						double dispWLS1 = dispFilteredWLSLeft.at<float>(Arucho1.y, Arucho1.x);
						double e1 = (Arucho1.x - cx) / fx;
						double f1 = (Arucho1.y - cy) / fy;
						double depthWLS1 = fx * bl / dispWLS1;
						double X1 = e1 * depthWLS1;
						double Y1 = f1 * depthWLS1;
						double Z1 = depthWLS1;

							//if (ids.size() > 1) {
							//	Arucho2.x = (corners[1][0].x + corners[1][2].x) / 2;
							//	Arucho2.y = (corners[1][0].y + corners[1][2].y) / 2;
							//	circle(out, Arucho2, 8, Scalar(255, 0, 0), -1, 1, 0);
							//	line(out, Arucho1, Arucho2, Scalar(255, 0, 0), 1, 1, 0);
							//	line(dispFilteredColorLeft, Arucho1, Arucho2, Scalar(255, 0, 0), 1, 1, 0);


							//	double dispWLS2 = dispFilteredWLSLeft.at<float>(Arucho2.y, Arucho2.x);
							//	double e2 = (Arucho2.x - cx) / fx;
							//	double f2 = (Arucho2.y - cy) / fy;
							//	double depthWLS2 = fx * bl / dispWLS2;
								double X2 = /*e2 * depthWLS2*/0.0;
								double Y2 = /*f2 * depthWLS2*/0.0;
								double Z2 = /*depthWLS2*/0.0;

								float distanceTestCalc = /*distance(X1, Y1, Z1, X2, Y2, Z2)*/0.0;
								float realDistTest = 0.15;

								double X1SENS = 0.0;
								double Y1SENS = 0.0;
								double Z1SENS = 0.0;
									
							/*	if (transformed_cloud1->size() == 1) {
									
									
									cout << "Saving Sample: " << test << endl;

									X1SENS = transformed_cloud1->at(0).x;
									Y1SENS = transformed_cloud1->at(0).y;
									Z1SENS = transformed_cloud1->at(0).z;
											
									Sensor1Saved = true;
								}*/

								if (transformed_cloud2->size() == 1 /*&& Sensor1Saved*/) {



									cout << "Saving Sample: " << test << endl;
									//Sensor1Saved = false;
									double X2SENS = transformed_cloud2->at(0).x;
									double Y2SENS = transformed_cloud2->at(0).y;
									double Z2SENS = transformed_cloud2->at(0).z;

									//float distanceTestCalcSens = distance(X1SENS, Y1SENS, Z1SENS, X2SENS, Y2SENS, Z2SENS);
									float distanceTestCalcSens = distance(X1, Y1, Z1, X2SENS, Y2SENS, Z2SENS);
									line(out, Point( (X1, Y1, Z1) ), cv::Point((X2SENS, Y2SENS, Z2SENS)), Scalar(255, 0, 0), 1, 1, 0);
									line(dispFilteredColorLeft, Point( (X1, Y1, Z1) ), cv::Point((X2SENS, Y2SENS, Z2SENS)), Scalar(255, 0, 0), 1, 1, 0);

									myFile6 << X1 << ", " << Y1 << ", " << Z1 << ", " << X2 << ", " << Y2 << ", " << Z2 << ", ";
									test++;
									myFile6 << X1SENS << ", " << Y1SENS << ", " << Z1SENS << ", ";
									myFile6 << X2SENS << ", " << Y2SENS << ", " << Z2SENS << ", "<< distanceTestCalc << ", "<< distanceTestCalcSens << ", " << realDistTest << " \n";

								}
								

								if (test == 400) {
									test = 0;
									testingDisp = false;
									cout << " FINISHED SAVING STATS FROM DISP MAP " << endl;

								}

							//}
						}
					}

				if (transformed_cloud0->size() == 1 && pt.x != -1) {

					double dispWLS = dispFilteredWLSLeft.at<float>(pt.y, pt.x);
					double e = (pt.x - cx) / fx;
					double f = (pt.y - cy) / fy;
					double depthWLS = fx * bl / dispWLS;
					double xWLS = e * depthWLS;
					double yWLS = f * depthWLS;
					double zWLS = depthWLS;

					distanceSensorCamera = distance(0, 0, 0, transformed_cloud0->at(0).x, transformed_cloud0->at(0).y, transformed_cloud0->at(0).z);
					distanceTrackerCamera = distance(0, 0, 0, xWLS, yWLS, zWLS);
					distance3D = distance(xWLS, yWLS, zWLS, transformed_cloud0->at(0).x, transformed_cloud0->at(0).y, transformed_cloud0->at(0).z);
					distanceDiff = abs(distanceSensorCamera - distanceTrackerCamera);


					xRelI = (float)(abs(xWLS - transformed_cloud0->at(0).x) / abs(xWLS)) * 100;
					yRelI = (float)(abs(yWLS - transformed_cloud0->at(0).y) / abs(yWLS)) * 100;
					zRelI = (float)(abs(zWLS - (transformed_cloud0->at(0).z - 0.005)) / abs(zWLS)) * 100;
					overallRegErrI = (xRelI + yRelI + zRelI) / 3;


					/*cout << "X: " << transformed_cloud2->at(0).x << " Y: " << transformed_cloud2->at(0).y << " Z: " << transformed_cloud2->at(0).z << endl;
					cout << "DISP X: " << xWLS << " Y: " << yWLS << " Z: " << zWLS << endl;*/

					if (dataFlag) {
						if (statsCounter0 <  11) {
							myFile3 << xWLS << ", " << yWLS << ", " << zWLS << ", " << distanceTrackerCamera
								<< ", " << transformed_cloud0->at(0).x << ", " << transformed_cloud0->at(0).y << ", " << transformed_cloud0->at(0).z << ", " << distanceSensorCamera
								<< ", " << distance3D << ", " << xRelI << ", " << yRelI << ", " << zRelI << ", " << overallRegErrI << "\n";
							statsCounter0++;
							printf("Saving stats\n");
							string FileName = "Data" + std::to_string(picCount) + ".jpg";
							cv::imwrite(FileName, out);
						}
						else {
							picCount++;
							statsCounter0 = 0;
							dataFlag = 0;
						}

					}

				}

				if (transformed_cloud1->size() == 1 && ptClick.x != -1) {
					counter++;

					distanceSensorCamera2 = distance(0, 0, 0, transformed_cloud1->at(0).x, transformed_cloud1->at(0).y, transformed_cloud1->at(0).z);
					distanceClicked = distance(0, 0, 0, Clicked.x, Clicked.y, Clicked.z);
					distanceDiff2 = abs(distanceSensorCamera2 - distanceClicked);
					distance3D2 = distance(transformed_cloud1->at(0).x, transformed_cloud1->at(0).y, transformed_cloud1->at(0).z, Clicked.x, Clicked.y, Clicked.z);
					float ErrorMacro = (abs(RealDistance - distance3D2) / RealDistance) * 100;
					float microSquareSize = 0.001;
					float mockupScale = 11.29;
					float inScaleDist = distance3D2 / mockupScale;
					float  microError = (ErrorMacro) / (square_size / microSquareSize);
					float microDimError = microError * (inScaleDist / 100);
					float microDimError2 = ErrorMacro * (inScaleDist / 100);
					if (dataFlag) {
						if (statsCounter1 < 11) {
							myFile4 << " " << ", " << statsCounter1 << ", " << distanceClicked << ", " << distanceSensorCamera2 << ", " <<
									", " << distance3D2 << ", " << RealDistance << ", " << distance3D2 - RealDistance << ", " << ErrorMacro << ", " << microSquareSize << ", " <<
								microError << ", " << mockupScale << ", " << inScaleDist << ", " << microDimError <<", " << microDimError2 << "\n";
							statsCounter1++;
						}
						else {
							picCount++;
							statsCounter1 = 0;
							dataFlag = 0;
						}
					}
				}

						
				string res = "Distance Sensor 1 from camera: " + std::to_string(roundf((distanceSensorCamera) * 1000 * 100) / 100) + "mm";
				string res2 = "Distance Tracker from camera: " + std::to_string(roundf((distanceTrackerCamera) * 1000 * 100) / 100) + "mm";
				string res3 = "Distance Between them" + std::to_string((roundf((distanceDiff) * 1000) * 100) / 100) + " Distance coordinates: " + std::to_string((roundf((distance3D) * 1000) * 100) / 100);
				string res4 = "Err X: " + std::to_string(roundf(xRelI * 100) / 100) + "  Err Y: " + std::to_string(roundf(yRelI * 100) / 100) + "  Err Z: " + std::to_string(roundf(zRelI * 100) / 100);
				string res5 = "Overall instant error: " + std::to_string((roundf(overallRegErrI * 100) / 100));

				if (counter > 3) {
					res11 = "Distance Sensor 2 from camera: " + std::to_string(roundf((distanceSensorCamera2) * 1000 * 100) / 100) + "mm";
					res12 = "Distance Clicked from camera: " + std::to_string(roundf((distanceClicked) * 1000 * 100) / 100) + "mm";
					res13 = "Distance between them: " + std::to_string((roundf((distanceDiff2) * 1000) * 100) / 100) + " Distance coordinates: " + std::to_string((roundf((distance3D2) * 1000) * 100) / 100);
					counter = 0;
				}
						
				putText(image, res, cvPoint(10, 20), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 255, 0), 1, cv::LINE_AA);
				putText(image, res2, cvPoint(10, 40), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 255, 0), 1, cv::LINE_AA);
				putText(image, res3, cvPoint(10, 60), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 255, 0), 1, cv::LINE_AA);
				putText(image, res4, cvPoint(10, 80), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 255, 0), 1, cv::LINE_AA);
				putText(image, res5, cvPoint(10, 100), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 255, 0), 1, cv::LINE_AA);
						
						
				putText(image2, res11, cvPoint(10, 20), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, cv::LINE_AA);
				putText(image2, res12, cvPoint(10, 40), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, cv::LINE_AA);
				putText(image2, res13, cvPoint(10, 60), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, cv::LINE_AA);

				Rodrigues(Mat::eye(3, 3, CV_64F), CalibMat.rvec);
				Mat tvec = (Mat_<double>(3, 1) << (0, 0, 0));
				cv::Mat1f NullVector;

				if (transformed_cloud0->size() == 1) {
					std::vector<cv::Point3f> ObjectPoint0;
					std::vector<cv::Point2f> projectedPoints0;
					cv::Point3f object_point0(transformed_cloud0->at(0).x, transformed_cloud0->at(0).y, transformed_cloud0->at(0).z);
					ObjectPoint0.push_back(object_point0);
					cv::Point centerCircle0;

					projectPoints(ObjectPoint0, CalibMat.rvec, tvec, CalibMat.Knew, /*CalibMat.DL*/NullVector, projectedPoints0);

					if ((projectedPoints0[0].x != 239.5) && (projectedPoints0[0].y != 179.5) && !(isnan(projectedPoints0[0].x)) && !(isnan(projectedPoints0[0].y))) {
						circle(out, projectedPoints0[0], 10, Scalar(0, 255, 0), 3, 8, 0);
						circle(dispFilteredColorLeft, projectedPoints0[0], 10, Scalar(0, 255, 0), 3, 8, 0);
						projectedPointsLast0[0] = projectedPoints0[0];
						centerCircle0 = projectedPoints0[0];
					}

					else {
						circle(out, projectedPointsLast0[0], 10, Scalar(0, 255, 0), 3, 8, 0);
						circle(dispFilteredColorLeft, projectedPointsLast0[0], 10, Scalar(0, 255, 0), 3, 8, 0);
						centerCircle0 = projectedPointsLast0[0];
					}

					if (centerCircle0.x < out.cols && centerCircle0.y < out.cols && centerCircle0.y != INT_MIN && centerCircle0.x > -1 && centerCircle0.y > -1) {

						circle(mask[0], centerCircle0, 50, Scalar(255), -1);
						bitwise_and(dispFilteredColorLeft, dispFilteredColorLeft, masked_data1[0], mask[0]);
						bitwise_and(dispFilteredWLSLeft, dispFilteredWLSLeft, masked_data2[0], mask[0]);

						double minDisp, maxDisp;
						cv::Point minLoc, maxLoc;
						minMaxLoc(masked_data2[0], &minDisp, &maxDisp, &minLoc, &maxLoc);
						circle(masked_data1[0], maxLoc, 5, Scalar(0), -1);
						//line(out, centerCircle0, maxLoc, Scalar(255, 0, 0), 3);

					}

				}

				if (transformed_cloud1->size() == 1) {
					std::vector<cv::Point3f> ObjectPoint1;
					std::vector<cv::Point2f> projectedPoints1;
					cv::Point3f object_point1(transformed_cloud1->at(0).x, transformed_cloud1->at(0).y, transformed_cloud1->at(0).z);
					ObjectPoint1.push_back(object_point1);
					projectPoints(ObjectPoint1, CalibMat.rvec, tvec, CalibMat.Knew, /*CalibMat.DL*/NullVector, projectedPoints1);
					cv::Point centerCircle1;

					if ((projectedPoints1[0].x != 239.5) && (projectedPoints1[0].y != 179.5) && !(isnan(projectedPoints1[0].x)) && !(isnan(projectedPoints1[0].y))) {
						//circle(out, projectedPoints1[0], 10, Scalar(0, 255, 0), 3, 8, 0);
						//circle(dispFilteredColorLeft, projectedPoints1[0], 10, Scalar(0, 255, 0), 3, 8, 0);
						projectedPointsLast1[0] = projectedPoints1[0];
						centerCircle1 = projectedPoints1[0];
					}

					else {
						//circle(out, projectedPointsLast1[0], 10, Scalar(0, 255, 0), 3, 8, 0);
						//circle(dispFilteredColorLeft, projectedPointsLast1[0], 10, Scalar(0, 255, 0), 3, 8, 0);
						centerCircle1 = projectedPointsLast1[0];
					}

					if (centerCircle1.x < out.cols && centerCircle1.y < out.cols && centerCircle1.y != INT_MIN && centerCircle1.x > -1 && centerCircle1.y > -1) {

						circle(mask[1], centerCircle1, 50, Scalar(255), -1);
						bitwise_and(dispFilteredColorLeft, dispFilteredColorLeft, masked_data1[1], mask[1]);
						bitwise_and(dispFilteredWLSLeft, dispFilteredWLSLeft, masked_data2[1], mask[1]);

						double minDisp, maxDisp;
						cv::Point minLoc, maxLoc;
						minMaxLoc(masked_data2[1], &minDisp, &maxDisp, &minLoc, &maxLoc);
						circle(masked_data1[1], maxLoc, 5, Scalar(0), -1);
						//line(out, centerCircle1, maxLoc, Scalar(255, 0, 0), 3);

					}
				}

				if (transformed_cloud2->size() == 1) {
					std::vector<cv::Point3f> ObjectPoint2;
					std::vector<cv::Point2f> projectedPoints2;
					cv::Point3f object_point2(transformed_cloud2->at(0).x, transformed_cloud2->at(0).y, transformed_cloud2->at(0).z);
					ObjectPoint2.push_back(object_point2);
					cv::Point centerCircle2;
					projectPoints(ObjectPoint2, CalibMat.rvec, tvec, CalibMat.Knew, /*CalibMat.DL*/NullVector, projectedPoints2);

					if ((projectedPoints2[0].x != 239.5) && (projectedPoints2[0].y != 179.5) && !(isnan(projectedPoints2[0].x)) && !(isnan(projectedPoints2[0].y))) {
						circle(out, projectedPoints2[0], 10, Scalar(0, 0, 255), 3, 8, 0);
						circle(dispFilteredColorLeft, projectedPoints2[0], 10, Scalar(0, 0, 255), 3, 8, 0);
						projectedPointsLast2[0] = projectedPoints2[0];
						centerCircle2 = projectedPoints2[0];
					}

					else {
						circle(out, projectedPointsLast2[0], 10, Scalar(0, 0, 255), 3, 8, 0);
						circle(dispFilteredColorLeft, projectedPointsLast2[0], 10, Scalar(0, 0, 255), 3, 8, 0);
						centerCircle2 = projectedPointsLast2[0];
					}

					if (centerCircle2.x < out.cols && centerCircle2.y < out.cols && centerCircle2.y != INT_MIN && centerCircle2.x > -1 && centerCircle2.y > -1) {

						circle(mask[2], centerCircle2, 50, Scalar(255), -1);
						bitwise_and(dispFilteredColorLeft, dispFilteredColorLeft, masked_data1[2], mask[2]);
						bitwise_and(dispFilteredWLSLeft, dispFilteredWLSLeft, masked_data2[2], mask[2]);

						double minDisp, maxDisp;
						cv::Point minLoc, maxLoc;
						minMaxLoc(masked_data2[2], &minDisp, &maxDisp, &minLoc, &maxLoc);
						circle(masked_data1[2], maxLoc, 5, Scalar(0), -1);
						//line(out, centerCircle2, maxLoc, Scalar(255, 0, 0), 3);

					}

				}

				transformed_cloud0->clear();
				transformed_cloud1->clear();
				transformed_cloud2->clear();
				
				bitwise_or(mask[0], mask[1], mask[2]);
				

				//imshow("mask1", mask[0]);
				imshow("Masked disp1", masked_data1[0]);
				imshow("Masked dispREAL1", masked_data2[0]);
				//imshow("mask2", mask[1]);
				imshow("Masked disp2", masked_data1[1]);
				imshow("Masked dispREAL2", masked_data2[1]);

				imshow("mask", mask[2]);
				

			}
			zoom_now = 1 + (float)getTrackbarPos("Zoom", param_win_name)/10;
			resize(out, out, Size((out.size().width* zoom_now), (out.size().height* zoom_now)));
			resize(dispFilteredColorLeft, dispFilteredColorLeft, Size((dispFilteredColorLeft.size().width* zoom_now), (dispFilteredColorLeft.size().height* zoom_now)));
			cv::Rect myRoi(out.size().width / 2 - 480 / 2, out.size().height / 2 - 360 / 2, 480, 360);
			out = out(myRoi);
			dispFilteredColorLeft = dispFilteredColorLeft(myRoi);
			hconcat(rect_imgl, rect_imgr, rect_imgl);
			imshow("Sensor1", image);
			imshow("Sensor2", image2);
			imshow("Clicked", image3);
			imshow("Color disp LEFT", dispFilteredColorLeft);
			if (dataFlag) {
				string FileName = "Data" + std::to_string(picCount) + ".jpg";
				cv::imwrite(FileName, out);
			}
			cv::imshow("out", out);


		}


		string picNumber = "Pictures Taken: " + std::to_string(i);
		putText(small_imgL, picNumber, cvPoint(30, 30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, cv::LINE_AA);

		hconcat(small_imgL, small_imgR, small_imgL);


		if (ReMap) {
			imshow("Rectified Images", rect_imgl);
		}

		// End Time
		time(&end);

		// Time elapsed
		double seconds = difftime(end, start);

		// Calculate frames per second
		fps = num_frames / seconds;
		putText(small_imgL, to_string(fps), cvPoint(450, 30), FONT_HERSHEY_COMPLEX_SMALL, 1, cvScalar(0, 255, 0), 1, cv::LINE_AA);
		myFile7 << sample << ", " << fps << endl;
		sample++;
		imshow(param_win_name, small_imgL);

	}

	

	id = -1;
	errorCode = SetSystemParameter(SELECT_TRANSMITTER, &id, sizeof(id));
	if (errorCode != BIRD_ERROR_SUCCESS) errorHandler(errorCode);

	//////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////
	//
	//  Free memory allocations before exiting
	//
	delete[] pSensor;
	delete[] pXmtr;
	myFile.close();
	myFile2.close();
	myFile3.close();
	myFile4.close();
	myFile5.close();
	myFile6.close();
	myFile7.close();
	return 0;
}
			


void errorHandler(int error)
{
	char			buffer[1024];
	char* pBuffer = &buffer[0];
	int				numberBytes;

	while (error != BIRD_ERROR_SUCCESS)
	{
		error = GetErrorText(error, pBuffer, sizeof(buffer), SIMPLE_MESSAGE);
		numberBytes = strlen(buffer);
		buffer[numberBytes] = '\n';		// append a newline to buffer
		printf("%s", buffer);
	}
	exit(0);
}