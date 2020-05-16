package com.riis.modelmaker

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import com.riis.modelmaker.ml.Model
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.TensorProcessor
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.label.TensorLabel
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer


class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        val bitmap = BitmapFactory.decodeResource(applicationContext.resources, R.drawable.sunflowers)
        processImage(bitmap)

    }

    private fun processImage(bitmap: Bitmap) {
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
            Log.d("ModelMaker", outputBuffer[3].label + " " + outputBuffer[3].score)
        } catch (e: Exception) {
            // TO DO
        }
    }
}