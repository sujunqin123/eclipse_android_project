package com.example.dataprocess;

import android.R.integer;
import android.R.string;
import android.annotation.SuppressLint;
import android.content.Context;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import com.example.utils.LogUtil;

public class TextDataRead {
	public static final int ROW = 4608;
	public static final int COL = 14;
	private static final String SPLIT_STRING = ",";
	
	private Context context;
	private String path;
	private float[][] dataSet;
	private List<String> texts;
	public TextDataRead(Context context, String path){
		this.context = context;
		this.path = path;
		readText();
	}
	
	private void readText(){
		try {
            InputStream inputStream = context.getResources().getAssets().open(path);
            InputStreamReader inputStreamReader = new InputStreamReader(inputStream, "UTF-8");
            BufferedReader bufferedReader = new BufferedReader(inputStreamReader);
            String info = "";
            texts = new ArrayList<String>();
            while ((info = bufferedReader.readLine()) != null) {
				texts.add(info);
//				LogUtil.info(info);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
	}
	
	@SuppressLint("NewApi") 
	public float[][] generateDataSet(){
		String[] lineString;
		dataSet = new float[ROW][COL];
		int i = 0;
		
		for (String string : texts) {
			if (!string.isEmpty()) {
				
				lineString = string.split(SPLIT_STRING);
				for (int j = 0; j < lineString.length; j++) {
					if (!lineString[j].isEmpty()) {
						dataSet[i][j] = Float.valueOf(lineString[j]);
					}
				}
				i++;
			}
		}
		return dataSet;
	}

	public float[][] getDataSet() {
		return dataSet;
	}

	public List<String> getTexts() {
		return texts;
	}
	
}
