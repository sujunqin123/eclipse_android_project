package com.example.grayprocess2;

import java.util.Arrays;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.ANN_MLP;

import com.example.dataprocess.TextDataRead;
import com.example.machinelearn.MyAnnMlp;
import com.example.utils.LogUtil;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.hardware.Camera;
import android.os.Bundle;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceView;
import android.view.View;
import android.view.View.OnClickListener;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

public class MainActivity extends Activity {

	private static final int ROW = TextDataRead.ROW;
	private static final int COL = TextDataRead.COL;

	private static final String TAG = "MainActivity";
	private CameraBridgeViewBase mOpenCvCameraView;
	private Button btnTrain, btnTest, btnTakePhoto;
	private TextView showRaTextView, resultTextView;
	private CheckBox trainedFlagCheckBox;
	private ImageView photoView;

	private Bitmap bmp;
	private Mat rgba;
	private Mat photoMat;
	private TextDataRead mTextDataRead;
	private float[][] dataSet;
	private ANN_MLP ann_MLP;
	private MyAnnMlp ann;
	private boolean trainedFlag = false;
	private boolean trainingFlag = false;
	private boolean testedFlag = false;

	// OpenCV类库加载并初始化成功后的回调函数，在此我们不进行任何操作
	private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
		@Override
		public void onManagerConnected(int status) {
			switch (status) {
			case LoaderCallbackInterface.SUCCESS: {
				System.loadLibrary("image_proc");
				Log.i(TAG, "OpenCV loaded successfully");
				mOpenCvCameraView.enableView();
			}
				break;
			default: {
				super.onManagerConnected(status);
			}
				break;
			}
		}
	};

	@Override
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);
		getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

		btnTrain = (Button) findViewById(R.id.btn_gray_process);
		btnTest = (Button) findViewById(R.id.btn_test);
		showRaTextView = (TextView) findViewById(R.id.correctRate);
		resultTextView = (TextView) findViewById(R.id.result);
		trainedFlagCheckBox = (CheckBox) findViewById(R.id.trained_model_flag);
		btnTakePhoto = (Button) findViewById(R.id.takephoto);
		photoView = (ImageView) findViewById(R.id.photoImageView);

		btnTrain.setOnClickListener(new OnClickListener() {

			@Override
			public void onClick(View arg0) {

				if (trainedFlag == false && trainingFlag == false) {
					showRaTextView.setText("tringing");
					ann_MLP = ANN_MLP.create();
					ann = new MyAnnMlp(dataSet, ann_MLP);
					ann.annTrain(ann.getTrain_set(), ann.getTrain_labels(), 10);
					float result = ann.ann_test(ann.getSample_set(), ann.getSample_labels());
					LogUtil.info("result:" + result);
					showRaTextView.setText("模型准确率:" + result);
					trainingFlag = false;
					trainedFlag = true;
					trainedFlagCheckBox.setChecked(true);
					trainingFlag = true;
				} else if (trainingFlag == true) {
					LogUtil.info("模型正在训练!");
					Toast.makeText(getApplication(), "模型正在训练!", Toast.LENGTH_SHORT);
				} else {
					LogUtil.info("模型已训练完成!");
					Toast.makeText(getApplication(), "模型已训练完成!", Toast.LENGTH_SHORT);
				}
			}
		});
		btnTest.setOnClickListener(new OnClickListener() {

			@Override
			public void onClick(View arg0) {
				if (ann != null && trainedFlag == true) {
					String result = ann.testOneSample();
					resultTextView.setText("测试结果:" + result);
					testedFlag = true;
					
			        int w = bmp.getWidth();  
			        int h = bmp.getHeight();  
			        int[] pixels = new int[w*h];       
			        bmp.getPixels(pixels, 0, w, 0, 0, w, h);  
			        int[] resultInt = ImageProc.grayProc(pixels, w, h);  
			        Bitmap resultImg = Bitmap.createBitmap(w, h, Config.ARGB_8888);  
			        resultImg.setPixels(resultInt, 0, w, 0, 0, w, h);  
			        photoView.setImageBitmap(resultImg);      
					

					mOpenCvCameraView.setVisibility(View.GONE);
					photoView.setVisibility(View.VISIBLE);
					btnTakePhoto.setVisibility(View.GONE);
					btnTest.setVisibility(View.VISIBLE);


				} else {
					LogUtil.info("请先训练模型!");
					Toast.makeText(getApplication(), "请先训练模型!", Toast.LENGTH_SHORT);
				}
			}
		});
		btnTakePhoto.setOnClickListener(new OnClickListener() {

			@Override
			public void onClick(View arg0) {
				photoMat = new Mat();
				rgba.copyTo(photoMat);
				Mat photoMat_T = photoMat.t();
				Core.flip(photoMat_T, photoMat_T, 0);
				bmp = Bitmap.createBitmap(photoMat_T.cols(), photoMat_T.rows(), Config.RGB_565);
				Utils.matToBitmap(photoMat_T, bmp);
				photoView.setImageBitmap(bmp);

				mOpenCvCameraView.setVisibility(View.GONE);
				photoView.setVisibility(View.VISIBLE);
				btnTakePhoto.setVisibility(View.GONE);
				btnTest.setVisibility(View.VISIBLE);
			}
		});
		photoView.setOnClickListener(new OnClickListener() {

			@Override
			public void onClick(View arg0) {
				testedFlag = false;
				mOpenCvCameraView.setVisibility(View.VISIBLE);
				photoView.setVisibility(View.GONE);
				btnTakePhoto.setVisibility(View.VISIBLE);
				btnTest.setVisibility(View.GONE);
			}
		});
		mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.java_surface_view);
		mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
		mOpenCvCameraView.setCvCameraViewListener(new CvCameraViewListener2() {

			@Override
			public void onCameraViewStopped() {
				// TODO Auto-generated method stub
			}

			@Override
			public void onCameraViewStarted(int width, int height) {
				// TODO Auto-generated method stub
			}

			@Override
			public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

				rgba = inputFrame.rgba();// color frame
				Core.flip(rgba, rgba, -1);
				return rgba;
			}
		});
		mTextDataRead = new TextDataRead(getApplicationContext(), "trainingDataf5.txt");
		dataSet = mTextDataRead.generateDataSet();
		LogUtil.info("dataSet read over");
	}

	@Override
	protected void onPause() {
		super.onPause();
		if (mOpenCvCameraView != null)
			mOpenCvCameraView.disableView();
	}

	@Override
	public void onResume() {
		super.onResume();
		// 通过OpenCV引擎服务加载并初始化OpenCV类库，所谓OpenCV引擎服务即是
		// OpenCV_2.4.3.2_Manager_2.4_*.apk程序包，存在于OpenCV安装包的apk目录中
		super.onResume();
		if (!OpenCVLoader.initDebug()) {
			Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
			OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);

		} else {
			Log.d(TAG, "OpenCV library found inside package. Using it!");
			mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
		}
	}

	@Override
	protected void onDestroy() {
		super.onDestroy();
		if (mOpenCvCameraView != null)
			mOpenCvCameraView.disableView();
	}

}
