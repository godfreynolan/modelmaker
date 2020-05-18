package com.riis.modelmaker

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import com.riis.modelmaker.ml.Model
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.TensorProcessor
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.label.TensorLabel
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.w3c.dom.Text


class MainActivity : AppCompatActivity() {
    private var mPicture: String = "Nothing matched"
    private var mConfidence: Float = 0.0f
    private val THRESHOLD = 0.8f

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        val bitmap = BitmapFactory.decodeResource(applicationContext.resources, R.drawable.sunflowers)
        val labelText :TextView = findViewById(R.id.textView) as TextView
        labelText.setText(processImage(bitmap))
    }

    private  fun processImage(bitmap: Bitmap) :String {
        try {
            var tfImage = TensorImage(DataType.FLOAT32)
            tfImage.load(bitmap)

            val imageProcessor = ImageProcessor.Builder()
                    .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                    .build()
            tfImage = imageProcessor.process(tfImage)

            val model = Model.newInstance(this@MainActivity)
            val outputs = model.process(tfImage)
            val outputBuffer = outputs.probabilityAsTensorLabel.categoryList

            for (i in outputBuffer.indices) {
                if (outputBuffer[i].score >= THRESHOLD) {
                    mConfidence = outputBuffer[i].score
                    mPicture = outputBuffer[i].label
                    Log.d("ModelMaker", outputBuffer[i].label + " " + outputBuffer[i].score)
                }
            }
            Log.d("ModelMaker", mPicture + " " + mConfidence)
        } catch (e: Exception) {
            Log.d("ModelMaker", "image wasn't loaded")
        }
        return (mPicture + " " + mConfidence)
    }
}