#include <ImageProc.h>
#include <opencv2/core/core.hpp>
#include <string>
#include "vector"
#include <stddef.h>

#include <opencv2/highgui/highgui.hpp>
#include "opencv2/ml.hpp"
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <iomanip>
#include <sstream>
#include <stdlib.h>

#include "android/log.h"

#define LOG_TAG "JNI_DEBUG"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG ,__VA_ARGS__) // 定义LOGD类型
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,LOG_TAG ,__VA_ARGS__) // 定义LOGI类型
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,LOG_TAG ,__VA_ARGS__) // 定义LOGW类型
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,LOG_TAG ,__VA_ARGS__) // 定义LOGE类型
#define LOGF(...) __android_log_print(ANDROID_LOG_FATAL,LOG_TAG ,__VA_ARGS__) // 定义LOGF类型
using namespace cv;
using namespace std;
using namespace ml;

Mat pSrcImg;                //原图
Mat pGrayImg;               //灰度图
Mat pDst;                   //二值化
Mat lunkuo;
vector<vector<Point> > contours;
vector<Vec4i> hierarchy;
int type = 3;

CvMemStorage *stor;
CvSeq *cont;
CvRect DrawRec(IplImage* pImgFrame, IplImage* pImgProcessed, int MaxArea);
CvRect main1(IplImage *img, float * angle);
IplImage* rotateImage2(IplImage* img, double angle);
void number(IplImage* img);
int mythresholdvalue(Mat src);

void generateRandom(int n, int test_num, int min, int max, int* mark_samples);
void annTrain(Mat TrainData, Mat classes, int nNeruns);
float ann_test(Mat samples_set, Mat sample_labels);
int recog(Mat features);

Ptr<ANN_MLP> ann = ANN_MLP::create();
int numCharacter = 10;

JNIEXPORT jintArray JNICALL Java_com_example_grayprocess2_ImageProc_grayProc(
		JNIEnv* env, jclass obj, jintArray buf, jint w, jint h) {
	jint *cbuf;
	cbuf = env->GetIntArrayElements(buf, false);
	if (cbuf == NULL) {
		return 0;
	}

	char s[200];

	Mat imgData(h, w, CV_8UC4, (unsigned char*) cbuf);
	LOGI("imgData.rows:%d",imgData.rows);
	LOGI("imgData.cols:%d",imgData.cols);
	LOGI("imgData.size:%d",imgData.size);
	LOGI("imgData.depth:%d",imgData.depth());
	LOGI("imgData.type:%d",imgData.type());
	uchar* ptr = imgData.ptr(0);
	for (int i = 0; i < w * h; i++) {
		//计算公式：Y(亮度) = 0.299*R + 0.587*G + 0.114*B
		//对于一个int四字节，其彩色值存储方式为：BGRA
		int grayScale = (int) (ptr[4 * i + 2] * 0.299 + ptr[4 * i + 1] * 0.587
				+ ptr[4 * i + 0] * 0.114);
		ptr[4 * i + 1] = grayScale;	//G
		ptr[4 * i + 2] = grayScale;	 //R
		ptr[4 * i + 0] = grayScale;  //B
	}



	pGrayImg = imgData;
	int t1 = mythresholdvalue(imgData);
	IplImage a = _IplImage(imgData);
	IplImage* pGrayImg1 = &a;
	IplImage b = _IplImage(imgData);
	IplImage* pDst1 = &b;
	cvThreshold(pGrayImg1, pDst1, t1, 255, type);
	LOGI("cvThreshold ok");

	IplImage *src = pDst1;
	IplImage *img_erode =  cvCloneImage( src );

	cvErode(src, img_erode, NULL, 10); //腐蚀
	LOGI("cvTcvErod ok");

	Mat resultMat = cv::cvarrToMat(img_erode);
	resultMat.copyTo(imgData);

	LOGI("cvarrToMat ok");
	int size = w * h;

	jintArray result = env->NewIntArray(size);
	env->SetIntArrayRegion(result, 0, size, cbuf);
	env->ReleaseIntArrayElements(buf, cbuf, 0);

	return result;
}

JNIEXPORT jintArray JNICALL Java_com_example_grayprocess2_ImageProc_annProc(
		JNIEnv* env, jclass obj, jfloatArray arrayData, jint row, jint col) {

	jfloat *cbuf;
	cbuf = env->GetFloatArrayElements(arrayData, false);
	if (cbuf == NULL) {
		return 0;
	}

	char s[200];

	LOGI("step1!");

	LOGI("step2!");
	const int ROW = row;
	const int COL = col;
	const int COUNT_OUT = 10;
	const int COUNT_FEATURE = col - 1;

	float dataSet[ROW][COL];
	int classSet[ROW];

	for (int i = 0; i < ROW; ++i) {
		for (int j = 0; j < COL; ++j) {
			if (j < COUNT_FEATURE) {
				dataSet[i][j] = (float) *(cbuf + i * col + j);
				sprintf(s, "%s,%f", s, dataSet[i][j]);
			} else {
				classSet[i] = (float) *(cbuf + i * col + j);
				sprintf(s, "%s,label:%d", s, classSet[i]);
			}
		}
		LOGI("[%d]:%s ", i, s);
		sprintf(s, "");
	}

	Mat dataSetMat(ROW, COUNT_FEATURE, CV_32FC1, dataSet);
	Mat train_set, train_labels;
	Mat sample_set, sample_labels;

	int markSample[ROW];
	generateRandom(0, 100, 0, ROW - 1, markSample);

	for (int i = 0; i < dataSetMat.rows; i++) {
		if (markSample[i] == 1) {
			sample_set.push_back(dataSetMat.row(i));
			sample_labels.push_back(classSet[i]);
		} else {
			train_set.push_back(dataSetMat.row(i));
			train_labels.push_back(classSet[i]);
		}
	}

	LOGI("step3!");
	annTrain(train_set, train_labels, 3);

	LOGI("step4!");

	float rightRate = ann_test(sample_set, sample_labels);

	int size = 1;
	jint *predictResult;
	predictResult[0] = (int) rightRate * 100;

	jintArray result = env->NewIntArray(size);
	env->SetIntArrayRegion(result, 0, size, predictResult);
	return result;
}

void annTrain(Mat train_set, Mat train_labels, int nNeruns) {
	LOGI("annTrain->step1");
	//Prepare trainClases
	//Create a mat with n trained data by m classes
	Mat trainClasses(train_labels.rows, numCharacter, CV_32FC1);
	for (int i = 0; i < trainClasses.rows; i++) {
		for (int k = 0; k < trainClasses.cols; k++) {
			//If class of data i is same than a k class
			if (k == train_labels.at<int>(i))
				trainClasses.at<float>(i, k) = 1;
			else
				trainClasses.at<float>(i, k) = 0;
		}

	}
	LOGI("train_set.rows:%d,trainClasses.rows:%d", train_set.rows,
			trainClasses.rows);
	LOGI("train_set.cols:%d,trainClasses.cols:%d", train_set.cols,
			trainClasses.cols);

	LOGI("annTrain->step2");
	//setting the NN layer size
	int ar[3] = { train_set.cols, nNeruns, numCharacter };
	Mat layerSizes(1, 3, CV_32S, ar);

	ann->setLayerSizes(layerSizes);
	ann->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM);
	ann->setTrainMethod(cv::ml::ANN_MLP::BACKPROP);
	ann->setBackpropMomentumScale(0.3);
	ann->setBackpropWeightScale(0.3);
	ann->setTermCriteria(
			TermCriteria(TermCriteria::MAX_ITER, (int) 5000, 0.01));
//	ann->setRpropDW0(0.1);
//	ann->setRpropDWPlus(1.2);
//	ann->setRpropDWMinus(0.5);
//	ann->setRpropDWMin(FLT_EPSILON);
//	ann->setRpropDWMax(50.0);
	LOGI("annTrain->step3");
	ann->train(train_set, ROW_SAMPLE, trainClasses);
}

float ann_test(Mat samples_set, Mat sample_labels) {
	int correctNum = 0;
	float accurate = 0;

	//Prepare trainClases
	//Create a mat with n trained data by m classes
	Mat trainClasses(samples_set.rows, numCharacter, CV_32FC1);
	for (int i = 0; i < trainClasses.rows; i++) {
		for (int k = 0; k < trainClasses.cols; k++) {
			//If class of data i is same than a k class
			if (k == sample_labels.at<int>(i))
				trainClasses.at<float>(i, k) = 1;
			else
				trainClasses.at<float>(i, k) = 0;
		}

	}

	for (int i = 0; i < samples_set.rows; i++) {
		int result = recog(samples_set.row(i));
		if (result == sample_labels.at<int>(i))
			correctNum++;
	}
	accurate = (float) correctNum / samples_set.rows;
	return accurate;
}

int recog(Mat features) {
	int result = -1;
	Mat Predict_result(1, numCharacter, CV_32FC1);
	ann->predict(features, Predict_result);
	Point maxLoc;
	double maxVal;

	minMaxLoc(Predict_result, 0, &maxVal, 0, &maxLoc);
	return maxLoc.x;
}

void generateRandom(int n, int test_num, int min, int max, int* mark_samples) {
	int range = max - min;
	int index = rand() % range + min;
	if (mark_samples[index] == 0) {
		mark_samples[index] = 1;
		n++;
	}

	if (n < test_num)
		generateRandom(n, test_num, min, max, mark_samples);
}

CvRect DrawRec(IplImage* pImgFrame, IplImage* pImgProcessed, int MaxArea) {
	//pImgFrame:初始未处理的帧，用于最后标出检测结果的输出;
	//pImgProcessed:处理完的帧,用于找运动物体的轮廓
	vector<int> vec;

	stor = cvCreateMemStorage(0);  //创建动态结构和序列
	cont = cvCreateSeq(CV_SEQ_ELTYPE_POINT, sizeof(CvSeq), sizeof(CvPoint),
			stor);

	// 找到所有轮廓
	cvFindContours(pImgProcessed, stor, &cont, sizeof(CvContour), CV_RETR_LIST,
			CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));

	// 直接使用CONTOUR中的矩形来画轮廓
	int max = 0;
	int max_2 = 0;
	CvRect r_max;
	CvRect r_max2;
	for (; cont; cont = cont->h_next) {

		CvRect r = ((CvContour*) cont)->rect;
		if (r.height * r.width > MaxArea) // 面积小的方形抛弃掉
				{
			cvRectangle(pImgFrame, cvPoint(r.x, r.y),
					cvPoint(r.x + r.width, r.y + r.height), CV_RGB(255,0,0), 1,
					CV_AA, 0);
			if (max < r.width) {
				if (max != 0)
					r_max2 = r_max;
				max_2 = max;
				max = r.width;
				r_max = r;
			} else if (max_2 < r.width) {
				max_2 = r.width;
				r_max2 = r;
			}
		}

	}

	cvShowImage("video", pImgFrame);
	printf("!!!!%d %d", (int) r_max2.height, (int) r_max2.width);
	return r_max2;
}

//旋转图像内容不变，尺寸相应变大
IplImage* rotateImage2(IplImage* img, double degree) {
	if (degree < -45)
		degree = 270 - degree;
	else
		degree = -degree;
	double angle = degree * CV_PI / 180.;
	double a = sin(angle), b = cos(angle);
	int width = img->width, height = img->height;
	//旋转后的新图尺寸
	int width_rotate = int(height * fabs(a) + width * fabs(b));
	int height_rotate = int(width * fabs(a) + height * fabs(b));
	IplImage* img_rotate = cvCreateImage(cvSize(width_rotate, height_rotate),
			img->depth, img->nChannels);
	cvZero(img_rotate);
	//保证原图可以任意角度旋转的最小尺寸
	int tempLength = sqrt((double) width * width + (double) height * height)
			+ 10;
	int tempX = (tempLength + 1) / 2 - width / 2;
	int tempY = (tempLength + 1) / 2 - height / 2;
	IplImage* temp = cvCreateImage(cvSize(tempLength, tempLength), img->depth,
			img->nChannels);
	cvZero(temp);
	//将原图复制到临时图像tmp中心
	cvSetImageROI(temp, cvRect(tempX, tempY, width, height));
	cvCopy(img, temp, NULL);
	cvResetImageROI(temp);
	//旋转数组map
	// [ m0  m1  m2 ] ===>  [ A11  A12   b1 ]
	// [ m3  m4  m5 ] ===>  [ A21  A22   b2 ]
	float m[6];
	int w = temp->width;
	int h = temp->height;
	m[0] = b;
	m[1] = a;
	m[3] = -m[1];
	m[4] = m[0];
	// 将旋转中心移至图像中间
	m[2] = w * 0.5f;
	m[5] = h * 0.5f;
	CvMat M = cvMat(2, 3, CV_32F, m);
	cvGetQuadrangleSubPix(temp, img_rotate, &M);
	cvReleaseImage(&temp);
	return img_rotate;
}

void number(IplImage* img) {
	Mat imgSrc = cv::cvarrToMat(img);

	int idx = 0;
	char szName[56] = { 0 };
	float width = (float) (imgSrc.cols - 10) / 18;
	float height = (float) imgSrc.rows;
	while (idx < 18) {
		CvRect rc = cvRect(7 + width * idx, 0, width, height);
		Mat imgNo;

		Mat a(imgSrc, rc);
		a.copyTo(imgNo);
		sprintf(szName, "wnd_%d", idx++);
		cvNamedWindow(szName);
		imshow(szName, imgNo); //如果想切割出来的图像从左到右排序，或从上到下，可以比较rc.x,rc.y;
		IplImage qImg;
		qImg = IplImage(imgNo); // cv::Mat -> IplImage
		sprintf(szName, "2\\%s_%d.jpg", "22", idx);
		cvSaveImage(szName, &qImg);
	}
	cvNamedWindow("src");
	imshow("src", imgSrc);
	cvWaitKey(0);
	cvDestroyAllWindows();
	return;
}

int mythresholdvalue(Mat src) {
	int t1 = 100;
	int t2 = 0;                        //t2保存旧值 用于和t1比较

	int nr = pGrayImg.rows; // number of rowslkl
	int nc = pGrayImg.cols * pGrayImg.channels(); // total number of elements per line

	while (abs(t1 - t2) >= 2) {
		int max = 0;
		int min = 0;
		int max_count = 0;
		int min_count = 0;
		for (int j = 0; j < nr; j++) {
			uchar* data = pGrayImg.ptr<uchar>(j);
			for (int i = 0; i < nc; i++) {
				if (data[i] > t1) {
					max_count++;
					max += data[i];
				} else {
					min_count++;
					min += data[i];
				}

			}
		}
		t2 = t1;
		t1 = (max / max_count + min / min_count) / 2;
	}

	return t1;
}

CvRect main1(IplImage *img, float *angle) {
	int i = 0;
	int mode = CV_RETR_EXTERNAL; //提取轮廓的模式
	int contours_num = 0; //图像中提取轮廓的数目
	IplImage *pContourImg =  cvCloneImage( img );
//	LOGI("IplImage *pContourImg =  cvCloneImage( img ); ok");
//	Mat imgSrc = cv::cvarrToMat(img);
//	imgSrc.convertTo(imgSrc,1,1,0);
//	*pContourImg  = _IplImage(imgSrc);
//	LOGI("*pContourImg  = _IplImage(imgSrc); ok");
	pContourImg = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
	CvMemStorage *storage = cvCreateMemStorage(0); //设置轮廓时需要的存储容器
	CvSeq *contour = 0; //设置存储提取的序列指针
	/*cvFindContours查找物体轮廓*/
//	mode = CV_RETR_EXTERNAL; //提取物体最外层轮廓
	LOGI("CvMemStorage *storage = cvCreateMemStorage(0);  ok");
	contours_num = cvFindContours(pContourImg, storage, &contour, sizeof(CvContour),
			mode, CV_CHAIN_APPROX_NONE);
//	cout << "检测出的轮廓数目为：" << contours_num << " " << endl;
	LOGI("cvFindContours;  ok");
	int MaxArea = 0;
	CvSeq *contour_max = 0;
	for (; contour; contour = contour->h_next) {
		CvRect r = ((CvContour*) contour)->rect;
		if (r.height * r.width > MaxArea) {
			MaxArea = r.height * r.width;
			contour_max = contour;
		}

	}
	LOGI("cvMinAreaRect2;  ok");
	CvBox2D box = cvMinAreaRect2(contour_max);
	LOGI("cvMinAreaRect2;  ok");
	CvPoint2D32f pt[4];
	cvBoxPoints(box, pt);
	LOGI("cvBoxPoints;  ok");
//	IplImage *dst = cvCloneImage( img );
//	cvZero(dst);
//
//	for (int i = 0; i < 4; ++i) {
//		cvLine(dst, cvPointFrom32f(pt[i]),
//				cvPointFrom32f(pt[((i + 1) % 4) ? (i + 1) : 0]),
//				CV_RGB(0,0,255));
////		printf("%d %d\n", (int) pt[i].x, (int) pt[i].y);
//	}
//	cvShowImage("dst", dst);

	int x = (int) pt[1].x < 0 ? 0 : (int) pt[1].x;
	int y = (int) pt[2].y < 0 ? 0 : (int) pt[2].y;
	int weight =
			int(pt[3].x - pt[1].x)+x > img->width ?
					img->width-x : int(pt[3].x - pt[1].x);
	int height =
			int(pt[0].y - pt[2].y)+y > img->height ?
					img->height-y : int(pt[0].y - pt[2].y);
//	printf("%d %d %d %d", x, y, weight, height);

	CvRect r = cvRect(x, y, weight, height);
//	printf("\n%f", box.angle);
	*angle = box.angle;
	return r;
}
