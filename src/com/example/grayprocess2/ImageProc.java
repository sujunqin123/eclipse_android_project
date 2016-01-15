package com.example.grayprocess2;

import org.opencv.core.Mat;

public class ImageProc {
	public static native int[] grayProc(int[] pixels, int w, int h);
	
	public static native int[] annProc(float[] data, int row, int col);
}
