package hch.opencv.resident;

import org.opencv.android.JavaCameraView;
import android.content.Context;
import android.content.res.Resources;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.util.AttributeSet;

/**
 * Created by Administrator on 2016/12/29.
 */

public class ResidentJavaCameraView extends JavaCameraView {
    private Paint mLinePaint;

    private int mMargin=20;

    public int getMargin() {
        return mMargin;
    }

    public void setMargin(int margin) {
        mMargin = margin;
    }

    public Paint getLinePaint() {
        return mLinePaint;
    }

    public void setLinePaint(Paint linePaint) {
        mLinePaint = linePaint;
    }

    protected void init() {
        Resources r = this.getResources();
        mLinePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        mLinePaint.setAlpha(200);
        mLinePaint.setStrokeWidth(1);
        mLinePaint.setStyle(Paint.Style.STROKE);
        mLinePaint.setColor(r.getColor(R.color.marker_color));
        mLinePaint.setShadowLayer(2, 1, 1, r.getColor(R.color.shadow_color));
        mLinePaint.setStrokeWidth(5);
    }

    public ResidentJavaCameraView(Context context, AttributeSet attrs) {
        super(context, attrs);
        setWillNotDraw(false);
        init();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        canvas.drawRect((getMeasuredWidth()-mFrameWidth)/2+mMargin,(getMeasuredHeight()-mFrameHeight)/2+mMargin,(getMeasuredWidth()+mFrameWidth)/2-mMargin,(getMeasuredHeight()+mFrameHeight)/2-mMargin,mLinePaint);
    }
}
