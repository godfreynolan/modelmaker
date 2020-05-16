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
            // Tensorflow representation of the image
            var tfImage = TensorImage(DataType.FLOAT32)
            // Loading the original android image to the tensorflow image
            tfImage.load(bitmap)

            // Processing the image
            // The model we build uses 224x224 images as an input so we need so resize the image
            val imageProcessor = ImageProcessor.Builder()
                    .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                    .build()
            tfImage = imageProcessor.process(tfImage)

            // Load "model.tflite" "Model" refers to the name before ".tflite". So if model was named "mymodel.tflite" we would write MyModel.newInstance(...)
            val model = Model.newInstance(this@MainActivity)

            // Apply normalization operator for image classification (a necessary step)
            val probabilityProcessor =
                    TensorProcessor.Builder().add(NormalizeOp(0f, 255f)).build()

            // running classification
            /*
            val outputs =
                    model.process(probabilityProcessor.process(tfImage.tensorBuffer))
            */
            val outputs = model.process(tfImage)
            // getting the output
            val outputBuffer = outputs.probabilityAsTensorLabel.categoryList
            //val outputBuffer = outputs.probabilityAsTensorLabel.outputFeature0AsTensorBuffer

            // adding labels to the output
            /*
            val tensorLabel =
                    TensorLabel(arrayListOf("daisy", "dandelion", "roses", "sunflowers", "tulips"), outputBuffer)

            // getting the first label (hot dog) probability
            // if 80 (you can change that) then we are pretty sure it is a hotdog -> update UI
            val probability = tensorLabel.mapWithFloatValue["sunflowers"]

            probability?.let {
                if (it > 0.80) {
                    Log.d("sdf", "Sunflower : " + probability)
                } else {
                    Log.d("sdf", "Other : " + probability)
                }
            }
            // Logs for debugging

             */
            Log.d("sdf", "HOT DOG : ")
        } catch (e: Exception) {
            Log.d("sdf", "Exception is " + e.localizedMessage)
        }
    }
}