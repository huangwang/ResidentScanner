package hch.opencv.resident;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Environment;
import android.util.Log;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.UUID;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.ANN_MLP;
import org.opencv.ml.Ml;
import org.opencv.utils.Converters;
import org.opencv.ml.KNearest;
import org.opencv.ml.SVM;

import static org.opencv.imgproc.Imgproc.MORPH_RECT;
import static org.opencv.imgproc.Imgproc.THRESH_BINARY_INV;

/**
 * Created by Administrator on 2016/12/16.
 */

public class ResidentDetector {
    private static final String TAG = "ResidentDetector";
    private String mScanResult;

    private InputStream images_reader;
    private InputStream labels_reader;

    //Dataset parameters
    private int total_images = 0;
    private int width = 0;
    private int height = 0;

    private ANN_MLP ann=null;

    public String getScanResult() {
        return mScanResult;
    }

    public ResidentDetector(InputStream imageStream, InputStream labelStream){
        images_reader = imageStream;
        labels_reader = labelStream;

        try{
            ReadMNISTData();
        }
        catch (IOException e)
        {
            Log.i("Read error:", "" + e.getMessage());
        }
    }

    private void ReadMNISTData() throws FileNotFoundException {
        Mat training_images=null;
        try{
            //Read the file headers which contain the total number of images and dimensions. First 16 bytes hold the header
            /*
            byte 0 -3 : Magic Number (Not to be used)
            byte 4 - 7: Total number of images in the dataset
            byte 8 - 11: width of each image in the dataset
            byte 12 - 15: height of each image in the dataset
            */

            byte [] header = new byte[16];
            images_reader.read(header, 0, 16);

            //Combining the bytes to form an integer
            ByteBuffer temp = ByteBuffer.wrap(header, 4, 12);
            total_images = temp.getInt();
            width = temp.getInt();
            height = temp.getInt();

            //Total number of pixels in each image
            int px_count = width * height;
            training_images = new Mat(total_images, px_count, CvType.CV_32F);

            //Read each image and store it in an array.

            for (int i = 0 ; i < total_images ; i++)
            {
                byte[] image = new byte[px_count*4];
                images_reader.read(image, 0, px_count*4);
                ByteBuffer temp1 = ByteBuffer.wrap(image, 0, px_count*4);
                for(int j=0;j<px_count;j++){
                    training_images.put(i,j,temp1.getFloat());
                }
            }
            training_images.convertTo(training_images, CvType.CV_32FC1);
            images_reader.close();
        }
        catch (IOException e)
        {
            Log.i("MNIST Read Error:", "" + e.getMessage());
        }

        Mat ann_labels=null;
        try{
            byte[] labels_data = new byte[total_images];

            Mat training_labels = new Mat(total_images, 1, CvType.CV_8U);
            Mat temp_labels = new Mat(1, total_images, CvType.CV_8U);
            byte[] header = new byte[8];
            //Read the header
            labels_reader.read(header, 0, 8);
            //Read all the labels at once
            labels_reader.read(labels_data,0,total_images);
            temp_labels.put(0,0, labels_data);

            //Take a transpose of the image
            Core.transpose(temp_labels, training_labels);
            training_labels.convertTo(training_labels, CvType.CV_32FC1);
            labels_reader.close();
            //产生ANN所需的标签矩阵
            ann_labels=Mat.zeros(training_labels.rows(),10,CvType.CV_32FC1);
            for(int i=0;i<training_labels.rows();i++){
                double[] value=training_labels.get(i,0);
                ann_labels.put(i,(int)value[0],1);
            }
        }
        catch (IOException e)
        {
            Log.i("MNIST Read Error:", "" + e.getMessage());
        }

        try{
            //Ann Classifier
            ann=ANN_MLP.create();
            Mat laySizes=new Mat(1,3,CvType.CV_32SC1);
            int[] layer={width*height};
            laySizes.put(0,0,layer);
            layer[0]=24;
            laySizes.put(0,1,layer);
            layer[0]=10;
            laySizes.put(0,2,layer);
            ann.setLayerSizes(laySizes);
            ann.setActivationFunction(ANN_MLP.SIGMOID_SYM);
            ann.setTermCriteria(new TermCriteria(TermCriteria.MAX_ITER+TermCriteria.EPS,5000,0.01));
            ann.train(training_images, Ml.ROW_SAMPLE, ann_labels);
        }catch(Exception e){
            Log.i(TAG,e.getMessage());
        }
    }

    //将mat保存成png图片用于分析
    public static void saveMatToBmp(Mat src,String name){
        Bitmap bitmap = Bitmap.createBitmap(src.cols(), src.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(src, bitmap);
        try{
            String file_path = Environment.getExternalStorageDirectory().getAbsolutePath()+"/debug";
            String  uniqueID = UUID.randomUUID().toString();
            File file = new File(file_path,name+uniqueID+".png");
            FileOutputStream fOut = new FileOutputStream(file);
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, fOut);
            fOut.flush();
            fOut.close();
        } catch(CvException e){
            Log.d("Exception",e.getMessage());
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    //判断矩形轮廓是否符合要求
    private boolean isEligible(RotatedRect candidate){
        float error = 0.2f;
        float aspect = (float)(4.5/0.3); //长宽比
        float min = 10*aspect*10; //最小区域
        float max = 50*aspect*50;  //最大区域
        float rmin = aspect - aspect*error; //考虑误差后的最小长宽比
        float rmax = aspect + aspect*error; //考虑误差后的最大长宽比

        float area = (float)candidate.size.height * (float)candidate.size.width;
        float r = (float)candidate.size.width/(float)candidate.size.height;
        if(r <1)
            r = 1/r;

        if( (area < min || area > max) || (r< rmin || r > rmax)  ) //满足该条件才认为该candidate为身份证区域
            return false;
        else
            return true;
    }

    //二值化
    private void OstuBeresenThreshold(Mat in, Mat out) //输入为单通道
    {
        double ostu_T = Imgproc.threshold(in , out, 0,255 ,Imgproc.THRESH_OTSU); //otsu获得全局阈值

        Core.MinMaxLocResult minMaxResult=Core.minMaxLoc(in);
        double CI = 0.12;
        double beta = CI*(minMaxResult.maxVal - minMaxResult.minVal +1)/128;
        double beta_lowT = (1-beta)*ostu_T;
        double beta_highT = (1+beta)*ostu_T;

        Mat doubleMatIn=in.clone();
        int rows = doubleMatIn.rows();
        int cols = doubleMatIn.cols();
        double Tbn;
        for( int i = 0; i < rows; ++i)
        {
            //对第i 行的每个像素(byte)操作
            for( int j = 0; j < cols; ++j )
            {

                if(i <2 | i>rows - 3 | j<2 | j>rows - 3)
                {

                    if( doubleMatIn.get(i,j)[0] <= beta_lowT )
                        out.put(i,j,0);
                    else
                        out.put(i,j,255);
                }
                else
                {
                    Mat temp=doubleMatIn.submat(new Rect(i-2,j-2,5,5));
                    Tbn=Core.sumElems(temp).val[0]/25;//窗口大小25*25
                    if( doubleMatIn.get(i,j)[0] < beta_lowT | (doubleMatIn.get(i,j)[0] < Tbn &&  (beta_lowT <= doubleMatIn.get(i,j)[0] && doubleMatIn.get(i,j)[0] >= beta_highT)))
                        out.put(i,j,0);
                    if( doubleMatIn.get(i,j)[0]> beta_highT | (doubleMatIn.get(i,j)[0] >= Tbn &&  (beta_lowT <=doubleMatIn.get(i,j)[0] && doubleMatIn.get(i,j)[0] >= beta_highT)))
                        out.put(i,j,255);
                }
            }
        }

    }

    //位置探测
    private void posDetect(Mat input, ArrayList<RotatedRect> rects) throws Exception {
        Mat digit=input.clone();
        Mat threshold_R=new Mat();
        OstuBeresenThreshold(digit ,threshold_R ); //二值化
        Mat imgInv=new Mat(digit.size(),digit.type(),new Scalar(255));
        Mat threshold_Inv=new Mat();
        Core.subtract(imgInv,threshold_R,threshold_Inv);  //黑白色反转，即背景为黑色
        //saveMatToBmp(threshold_Inv,"二值化处理");
        //Imgproc.adaptiveThreshold(digit,digit,255,Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY_INV,15, 30);
        Mat element = Imgproc.getStructuringElement(MORPH_RECT ,new Size(15 ,3));  //闭形态学的结构元素
        Imgproc.morphologyEx(threshold_Inv ,threshold_Inv,Imgproc.MORPH_CLOSE,element);
        //saveMatToBmp(threshold_Inv,"灰度化闭运算处理");
        List<MatOfPoint> contours1=new ArrayList<>();
        Mat hierarchy1=new Mat();
        Imgproc.findContours(threshold_Inv, contours1, hierarchy1, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_NONE);

        for (int contourIdx = 0; contourIdx < contours1.size(); contourIdx++) {
            MatOfPoint2f mp2f=new MatOfPoint2f(contours1.get(contourIdx).toArray());
            RotatedRect mr=Imgproc.minAreaRect(mp2f);
            if(isEligible(mr))  //判断矩形轮廓是否符合要求
            {
                rects.add(mr);
                break;
            }
        }
        if(rects.size()==0){
            throw new Exception("Cannot get the target area!");
        }
        //测试是否找到了号码区域
        /*Mat output=input.clone();
        Point[] vertices=new Point[4];
        rects.get(0).points(vertices);
        for (int i = 0; i < 4; i++)
            Imgproc.line(output, vertices[i], vertices[(i+1)%4], new Scalar(0,0,0));//画黑色线条

        saveMatToBmp(output,"查找目标区域");*/
    }

    //获得身份证号码字符矩阵
    private void normalPosArea(Mat intputImg,RotatedRect rects_optimal,Mat output_area){
        float r,angle;

        angle = (float)rects_optimal.angle;
        r = (float)rects_optimal.size.width / (float) (float)rects_optimal.size.height;
        if(r<1)
            angle = 90 + angle;
        Mat rotmat = Imgproc.getRotationMatrix2D(rects_optimal.center , angle,1);//获得变形矩阵对象
        Mat img_rotated=new Mat();
        Imgproc.warpAffine(intputImg ,img_rotated,rotmat, intputImg.size(),Imgproc.INTER_CUBIC);

        //裁剪图像
        Size rect_size = rects_optimal.size;

        if(r<1){
            double temp=rect_size.width;
            rect_size.width=rect_size.height;
            rect_size.height=temp;
        }
        Mat  img_crop=new Mat();
        Imgproc.getRectSubPix(img_rotated ,rect_size,rects_optimal.center , img_crop );

        //用光照直方图调整所有裁剪得到的图像，使具有相同宽度和高度，适用于训练和分类
        Mat resultResized=new Mat(20,300,CvType.CV_8UC1);
        //resultResized.create(20,300,CV_8UC1);
        Imgproc.resize(img_crop , resultResized,resultResized.size() , 0,0,Imgproc.INTER_CUBIC);

        resultResized.copyTo(output_area);
        //saveMatToBmp(output_area,"身份证区域");
    }

    private void char_segment(Mat inputImg, ArrayList<Mat> dst_mat){
        Mat img_threshold=new Mat();

        Mat whiteImg=new Mat(inputImg.size(),inputImg.type(),new Scalar(255));
        Mat in_Inv = new Mat();
        Core.subtract(whiteImg,inputImg,in_Inv);

        // threshold(in_Inv ,img_threshold , 140,255 ,CV_THRESH_BINARY ); //反转黑白色
        Imgproc.threshold(in_Inv ,img_threshold , 0,255 ,Imgproc.THRESH_OTSU ); //大津法二值化

        int[] x_char=new int[19];
        short counter = 1;
        short num = 0;
        boolean[] flag= new boolean[img_threshold.cols()];

        for(int j = 0 ; j < img_threshold.cols();++j)
        {
            flag[j] = true;
            for(int i = 0 ; i < img_threshold.rows() ;++i)
            {

                if(img_threshold.get(i,j)[0] != 0 )
                {
                    flag[j] = false;
                    break;
                }

            }
        }

        for(int i = 0;i < img_threshold.cols()-2;++i)
        {
            if(flag[i] == true)
            {
                x_char[counter] += i;
                num++;
                if(flag[i+1] ==false && flag[i+2] ==false )
                {
                    x_char[counter] = x_char[counter]/num;
                    num = 0;
                    counter++;
                }
            }
        }
        x_char[18] = img_threshold.cols();

        for(int i = 0;i < 18;i++)
        {
            dst_mat.add(in_Inv.submat(new Rect(x_char[i],0, x_char[i+1] - x_char[i] ,img_threshold.rows())));
        }
    }

    private float  sumMatValue(Mat image)
    {
        return (float)Core.sumElems(image).val[0];
    }

    private Mat projectHistogram(Mat img, int t)
    {
        Mat lowData=new Mat();
        Imgproc.resize(img , lowData ,new Size(8 ,16 )); //缩放到8*16

        int sz = t>0? lowData.rows(): lowData.cols();
        Mat mhist = Mat.zeros(1, sz ,CvType.CV_32F);

        for(int j = 0 ;j < sz; j++ )
        {
            Mat data = t>0?lowData.row(j):lowData.col(j);
            mhist.put(0,j,Core.countNonZero(data));
        }

        Core.MinMaxLocResult minMaxResult=Core.minMaxLoc(mhist);

        if(minMaxResult.maxVal > 0)
            mhist.convertTo(mhist ,-1,1.0f/minMaxResult.maxVal , 0);

        return mhist;
    }

    private void calcGradientFeat(Mat imgSrc, Mat out)
    {
        List<Float> feat=new ArrayList<>() ;
        Mat image=new Mat();

        //cvtColor(imgSrc,image,CV_BGR2GRAY);
        Imgproc.resize(imgSrc,image,new Size(8,16));

        // 计算x方向和y方向上的滤波
        float[][] mask = { { 1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } };

        Mat y_mask =new Mat(3, 3, CvType.CV_32F);
        for(int i=0;i<3;i++)
            for(int j=0;j<3;j++)
                y_mask.put(i,j,mask[i][j]);
        Core.divide(y_mask,Scalar.all(8), y_mask);
        Mat x_mask = y_mask.t(); // 转置
        Mat sobelX=new Mat();
        Mat sobelY=new Mat();

        Imgproc.filter2D(image, sobelX, CvType.CV_32F, x_mask);
        Imgproc.filter2D(image, sobelY, CvType.CV_32F, y_mask);

        Core.absdiff(sobelX,Scalar.all(0),sobelX);
        Core.absdiff(sobelY,Scalar.all(0),sobelY);

        float totleValueX = sumMatValue(sobelX);
        float totleValueY = sumMatValue(sobelY);

        // 将图像划分为4*2共8个格子，计算每个格子里灰度值总和的百分比
        for (int i = 0; i < image.rows(); i = i + 4)
        {
            for (int j = 0; j < image.cols(); j = j + 4)
            {
                Mat subImageX = sobelX.submat(new Rect(j, i, 4, 4));
                feat.add(sumMatValue(subImageX) / totleValueX);
                Mat subImageY= sobelY.submat(new Rect(j, i, 4, 4));
                feat.add(sumMatValue(subImageY) / totleValueY);
            }
        }

        //计算第2个特征
        Mat imageGray=new Mat();
        //cvtColor(imgSrc,imageGray,CV_BGR2GRAY);
        Imgproc.resize(imgSrc,imageGray,new Size(4,8));
        Mat p = imageGray.reshape(1,1);
        p.convertTo(p,CvType.CV_32FC1);
        for (int i = 0;i<p.cols();i++ )
        {
            feat.add((float)p.get(0,i)[0]);
        }

        //增加水平直方图和垂直直方图
        Mat vhist = projectHistogram(imgSrc , 1); //水平直方图
        Mat hhist = projectHistogram(imgSrc , 0);  //垂直直方图
        for (int i = 0;i<vhist.cols();i++ )
        {
            feat.add((float)vhist.get(0,i)[0]);
        }
        for (int i = 0;i<hhist.cols();i++ )
        {
            feat.add((float)hhist.get(0,i)[0]);
        }


        Mat out1 = Mat.zeros(1, feat.size() , CvType.CV_32F);
        for (int i = 0;i<feat.size();i++ )
        {
            out1.put(0,i,feat.get(i));
        }
        out1.copyTo(out);
    }


    private void classify(ArrayList<Mat> char_Mat,int[] char_result)
    {
        //char_result.resize(char_Mat.size());
        for (int i=0;i<char_Mat.size(); ++i)
        {
            //saveMatToBmp(char_Mat.get(i),"切割后数字");
            Mat output=new Mat(1 ,10 , CvType.CV_32FC1); //1*10矩阵

            Mat char_feature=new Mat();
            calcGradientFeat(char_Mat.get(i) ,char_feature);
            ann.predict(char_feature ,output,0);
            Point maxLoc;
            Core.MinMaxLocResult minMaxResult=Core.minMaxLoc(output);

            char_result[i] = (int)minMaxResult.maxLoc.x;

        }
    }

    private void getParityBit(int[] char_result)
    {
        int mod = 0;
        int[] wights=new int[]{ 7,9,10,5,8,4 ,2,1,6,3,7,9,10,5,8,4,2};
        for(int i =0; i < 17 ;++i)
            mod += char_result[i]*wights[i];//乘相应系数求和

        mod = mod%11; //对11求余

        int[] value=new int[] {1,0,10,9,8,7,6,5,4,3,2};
        char_result[17] = value[mod];
    }

    private Mat getRplane(Mat in)
    {
        ArrayList<Mat> splitBGR=new ArrayList<>(in.channels()); //容器大小为通道数3
        Core.split(in,splitBGR);
        //return splitBGR[2];  //R分量

        if(in.cols() > 700 |in.rows() >600)
        {
            Mat resizeR=new Mat( 450,600 , CvType.CV_8UC1);
            Imgproc.resize( splitBGR.get(2) ,resizeR ,resizeR.size());

            return resizeR;
        }
        else
            return splitBGR.get(2);

    }

    public boolean recognize(Mat digit){
        try{
            Mat imgRplane = getRplane(digit); //获得原始图像R分量，并缩放位图到合适大小

            ArrayList<RotatedRect> rects=new ArrayList<>();
            posDetect(imgRplane,rects);//获得身份证号码区域

            Mat outputMat=new Mat();
            normalPosArea(imgRplane ,rects.get(0),outputMat); //获得身份证号码字符矩阵

            ArrayList<Mat> char_mat=new ArrayList<>();  //获得切割得的字符矩阵
            char_segment(outputMat , char_mat);

            int[] char_result=new int[18];
            classify( char_mat ,char_result);

            getParityBit(char_result); //最后一位易出错，直接由前17位计算最后一位

            String id="";
            for(int i = 0; i < char_result.length;++i)
            {
                if (char_result[i] == 10)
                    id += "X";
                else
                {
                    id+=Integer.toString(char_result[i]);
                }
            }
            mScanResult=id;
            return true;
        }catch (Exception e){
            Log.d(TAG,e.getMessage());
            return false;
        }

    }

}
