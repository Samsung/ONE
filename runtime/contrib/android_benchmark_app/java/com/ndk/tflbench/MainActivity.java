package com.ndk.tflbench;

import android.app.Activity;
import android.os.Bundle;
import android.content.Intent;
import android.view.View;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.TextView;
import android.widget.Button;
import android.net.Uri;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.os.Trace;
import android.util.Log;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Vector;

public class MainActivity extends Activity {

  static {
    System.loadLibrary("android_benchmark_native");
  }

  private void setModel(final String message) {
    final TextView textView = (TextView)findViewById(R.id.model_label);
    runOnUiThread(new Runnable() {
      @Override
      public void run() { textView.setText(message); }
    });
  }

  private void setTitle(final String message) {
    final TextView textView = (TextView)findViewById(R.id.title_label);
    runOnUiThread(new Runnable() {
      @Override
      public void run() { textView.setText(message); }
    });
  }

  private void setText(final String message) {
    final TextView textView = (TextView)findViewById(R.id.message_label);
    runOnUiThread(new Runnable() {
      @Override
      public void run() { textView.setText(message); }
    });
  }

  private MappedByteBuffer buffer;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    setModel(getModelName());

    // Load Tensorflow Lite model
    try
    {
      AssetManager assets = getAssets();
      AssetFileDescriptor fileDescriptor = assets.openFd("model.tflite");
      FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
      FileChannel fileChannel = inputStream.getChannel();
      final long startOffset = fileDescriptor.getStartOffset();
      final long declaredLength = fileDescriptor.getDeclaredLength();

      buffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    } catch (IOException e) {
      Log.e("MYAPP", "exception", e);
    }

    Button btn_interp = (Button)findViewById(R.id.button_interp);
    btn_interp.setOnClickListener(new Button.OnClickListener() {
      @Override public void onClick(View view) {
        new Thread(new Runnable() {
          @Override
          public void run() { runInterpreterBenchmark(buffer); }
        }).start();
      }
    });

    Button btn_nnapi = (Button)findViewById(R.id.button_nnapi);
    btn_nnapi.setOnClickListener(new Button.OnClickListener() {
      @Override public void onClick(View view) {
        new Thread(new Runnable() {
          @Override
          public void run() { runNNAPIBenchmark(buffer); }
        }).start();
      }
    });
  }

  public native String getModelName();
  public native void runInterpreterBenchmark(MappedByteBuffer buffer);
  public native void runNNAPIBenchmark(MappedByteBuffer buffer);
}
