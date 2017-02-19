package hch.opencv.resident;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.imgproc.Imgproc;

//import android.support.v7.app.AppCompatActivity;
import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.util.Log;
import android.view.WindowManager;
import android.view.SurfaceView;
import android.app.Activity;

import java.io.IOException;
import java.io.InputStream;

public class ScannerActivity extends Activity implements CvCameraViewListener2{

    private static final String TAG = "ScannerActivity";

    private CameraBridgeViewBase mOpenCvCameraView;
    private Mat                    mRgba;
    private Mat                    mSubRgba;
    private ResidentDetector       mDetector;
    private boolean               mIsContinue=true;//是否继续扫描
    private String                 mScanResult;//身份证号扫描结果
    private int                   mMargin;//内容框留白大小
    private WorkThread             mThread;

    private static final int SUCCESSED = 1;
    private static final int FAILED=0;

    private Handler mHandler = new Handler() {
        @Override
        public void handleMessage(Message msg) {
            if (msg.what == SUCCESSED) {
                Bundle data=msg.getData();
                mScanResult=data.getString("ScanResult");
                //数据是使用Intent返回
                Intent intent = new Intent();
                //把返回数据存入Intent
                intent.putExtra("ScanResult",mScanResult);
                //设置返回数据
                setResult(RESULT_OK, intent);
                //关闭Activity
                finish();
                Log.i(TAG, mScanResult);
            }else if(msg.what==FAILED){
                mIsContinue=true;
            }
        }
    };

    //工作线程,用于识别身份证号
    private class WorkThread extends Thread {
        public Handler mWorkerHandler = null;

        public WorkThread(){
            super();
        }

        @Override
        public void run() {
            Looper.prepare();
            mWorkerHandler = new Handler() {
                public void handleMessage(Message msg) {
                    if(msg.what==1){
                        if(mDetector.recognize(mSubRgba)){
                            SendMessage(mHandler,SUCCESSED,mDetector.getScanResult());
                            this.getLooper().quit();
                        }else{
                            SendMessage(mHandler,FAILED,mDetector.getScanResult());
                        }
                    }
                }
            };
            Looper.myLooper().loop();
        }
    }

    private void SendMessage(Handler handler,int what, String scanResult){
        Message msg = new Message();
        msg.what = what;
        Bundle data=new Bundle();
        data.putString("ScanResult",scanResult);
        msg.setData(data);
        handler.sendMessage(msg);
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                    AssetManager assetManager=getResources().getAssets();
                    try{
                        InputStream trainStream=assetManager.open("train-images.idx3-ubyte");
                        InputStream labelStream=assetManager.open("train-labels.idx1-ubyte");
                        mDetector=new ResidentDetector(trainStream,labelStream);
                    }catch(IOException e){
                        Log.i(TAG,e.getMessage());
                    }

                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_scanner);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.opencv_java_surface_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mMargin=((ResidentJavaCameraView)mOpenCvCameraView).getMargin();
        mThread=new WorkThread();
        mThread.start();
    }

   @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        //mDetector=new ResidentDetector();
        /*try{
            String file_path = Environment.getExternalStorageDirectory().getAbsolutePath()+"/sample.jpg";
            Bitmap myBitmap = BitmapFactory.decodeFile(file_path);
            mSubRgba=new Mat(myBitmap.getHeight(),myBitmap.getWidth(), CvType.CV_8UC4);
            Imgproc.cvtColor(mSubRgba,mSubRgba,Imgproc.COLOR_BGR2RGB);
            Utils.bitmapToMat(myBitmap,mSubRgba);
        }catch(Exception e){
            Log.e(TAG,e.getMessage());
        }*/
    }

    public void onCameraViewStopped() {
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        mRgba= inputFrame.rgba();
        mSubRgba = mRgba.submat(mMargin, mRgba.rows()-2*mMargin, mMargin, mRgba.cols()-2*mMargin);
        //ResidentDetector.saveMatToBmp(mSubRgba,"原始输入图像");
        if(mIsContinue==true){
            Message msg = mHandler.obtainMessage();
            msg.what = 1;
            mThread.mWorkerHandler.sendMessage(msg);
            mIsContinue=false;
        }
        return mRgba;
    }
}
