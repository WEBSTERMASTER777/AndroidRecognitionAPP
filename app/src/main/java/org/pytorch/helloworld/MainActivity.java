package org.pytorch.helloworld;


import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.preference.PreferenceManager;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;
import org.pytorch.MemoryFormat;

import java.io.File;
import java.util.Arrays;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {


  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);



    Button gallery = findViewById(R.id.gallery);
    gallery.setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View view) {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(intent,3);
      }
    });
  }

  /**
   * Copies specified asset to the file in /files app directory and returns this file absolute path.
   *
   * @return absolute file path
   */
  public static String assetFilePath(Context context, String assetName) throws IOException {
    File file = new File(context.getFilesDir(), assetName);
    if (file.exists() && file.length() > 0) {
      return file.getAbsolutePath();
    }

    try (InputStream is = context.getAssets().open(assetName)) {
      try (OutputStream os = new FileOutputStream(file)) {
        byte[] buffer = new byte[4 * 1024];
        int read;
        while ((read = is.read(buffer)) != -1) {
          os.write(buffer, 0, read);
        }
        os.flush();
      }
      return file.getAbsolutePath();
    }
  }


  @Override
  protected void onActivityResult(int requestCode, int resultCode, Intent data) {
    super.onActivityResult(requestCode, resultCode, data);
    Bitmap bitmap = null;
    Module module = null;
    try {
      module = LiteModuleLoader.load(assetFilePath(this, "flower_mobile.pt"));
    } catch (IOException e) {
      e.printStackTrace();
    }
    if (resultCode==RESULT_OK && data != null) {
      Uri selectedImage = data.getData();
      ImageView imageView = findViewById(R.id.image);
      imageView.setImageURI(selectedImage);
      bitmap = BitmapFactory.decodeResource(getResources(), R.id.image);
      try {
        bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), selectedImage);
      } catch (IOException e) {
        e.printStackTrace();
      }
      imageView.setImageBitmap(bitmap);
      Log.i("PytorchHelloWorld", "Read");
    }
    else
      {
        Log.i("PytorchHelloWorld", "Not"+requestCode);
      }
    final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB, MemoryFormat.CHANNELS_LAST);

//     running the model
    final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

    // getting tensor content as java array of floats
    final float[] scores = outputTensor.getDataAsFloatArray();

    // searching for the index with maximum score
    float maxScore = -Float.MAX_VALUE;
    int maxScoreIdx = -1;
    for (int i = 0; i < scores.length; i++) {
      if (scores[i] > maxScore) {
        maxScore = scores[i];
        maxScoreIdx = i;
      }
    }
//    Arrays.sort(scores);


//    String className = ImageNetClasses.IMAGENET_CLASSES[maxScoreIdx]+scores[maxScoreIdx];
    String className = ImageNetClasses.IMAGENET_CLASSES[maxScoreIdx];

    // showing className on UI
    TextView textView = findViewById(R.id.text);
    textView.setText(className);
    Log.i("PytorchHelloWorld", className);



  }
}
