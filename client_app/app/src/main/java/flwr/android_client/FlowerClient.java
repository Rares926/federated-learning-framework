package flwr.android_client;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.ConditionVariable;
import android.util.Log;
import android.util.Pair;

import androidx.lifecycle.MutableLiveData;

import org.tensorflow.lite.examples.transfer.api.TransferLearningModel.Prediction;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.util.Map;
import java.util.concurrent.ExecutionException;

public class FlowerClient {

    private TransferLearningModelWrapper tlModel;
    private static final int LOWER_BYTE_MASK = 0xFF;
    private MutableLiveData<Float> lastLoss = new MutableLiveData<>();
    private Context context;
    private final ConditionVariable isTraining = new ConditionVariable();
    private static String TAG = "Flower";
    private int local_epochs = 1;
    private float[][][] inferenceImage;

    public FlowerClient(Context context) {
        this.tlModel = new TransferLearningModelWrapper(context);
        this.context = context;
    }

    public Map<String, Object> getWeights() {
        return tlModel.getParameters();
    }


    public Pair<Map<String, Object>, Integer> fit( Map<String, Object> weights, int epochs) {

        this.local_epochs = epochs;
        tlModel.updateParameters(weights);
        isTraining.close();
        tlModel.train(this.local_epochs);
        tlModel.enableTraining((epoch, loss) -> setLastLoss(epoch, loss));
        Log.e(TAG ,  "Training enabled. Local Epochs = " + this.local_epochs);
        isTraining.block();

        return Pair.create(getWeights(), tlModel.getSize_Training());
    }

    public Pair<Pair<Float, Float>, Integer> evaluate(Map<String, Object> weights) {
        tlModel.updateParameters(weights);
        tlModel.disableTraining();
        return Pair.create(tlModel.calculateTestStatistics(), tlModel.getSize_Testing());
    }

    public Prediction[] predict(){
        return tlModel.predict(this.inferenceImage);
    }


    public void setLastLoss(int epoch, float newLoss) {
        if (epoch == this.local_epochs - 1) {
            Log.e(TAG, "Training finished after epoch = " + epoch);
            lastLoss.postValue(newLoss);
            tlModel.disableTraining();
            isTraining.open();
        }
    }

    public void loadData(int device_id) {
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(this.context.getAssets().open("data/partition_" + (device_id - 1) + "_train.txt")));
            String line;
            int i = 0;
            while ((line = reader.readLine()) != null) {
                i++;
                Log.e(TAG, i + "th training image loaded");
                addSample("data/" + line, true);
            }
            reader.close();

            i = 0;
            reader = new BufferedReader(new InputStreamReader(this.context.getAssets().open("data/partition_" +  (device_id - 1)  + "_test.txt")));
            while ((line = reader.readLine()) != null) {
                i++;
                Log.e(TAG, i + "th test image loaded");
                addSample("data/" + line, false);
            }
            reader.close();

        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    private void addSample(String photoPath, Boolean isTraining) throws IOException {
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;
        Bitmap bitmap =  BitmapFactory.decodeStream(this.context.getAssets().open(photoPath), null, options);
        String sampleClass = get_class(photoPath);

        // get rgb equivalent
        float[][][] rgbImage = prepareImage(bitmap);

        try {
            this.tlModel.addSample(rgbImage, sampleClass, isTraining).get();
        } catch (ExecutionException e) {
            throw new RuntimeException("Failed to add sample to model", e.getCause());
        } catch (InterruptedException e) {
            // no-op
        }

    }

    public void addInferenceImage(Bitmap bitmap){
        this.inferenceImage = prepareImage(bitmap);
    }

    public int getNumClasses(){
        return this.tlModel.getNumClasses();
    }
    public String get_class(String path) {
        String label = path.split("/")[2];
        return label;
    }

    /**
     * Normalizes a camera image to [0; 1]
     * @return
     */

    private static float[][][] prepareImage(Bitmap bitmap)  {
        int modelImageSize = TransferLearningModelWrapper.IMAGE_SIZE;

        float[][][] normalizedRgb = new float[modelImageSize][modelImageSize][3];

        for (int y = 0; y < modelImageSize; y++) {
            for (int x = 0; x < modelImageSize; x++) {
                int rgb = bitmap.getPixel(x, y);

                float r = ((rgb >> 16) & LOWER_BYTE_MASK) * (1 / 255.0f);
                float g = ((rgb >> 8) & LOWER_BYTE_MASK) * (1 / 255.0f);
                float b = (rgb & LOWER_BYTE_MASK) * (1 / 255.0f);

                normalizedRgb[y][x][0] = r;
                normalizedRgb[y][x][1] = g;
                normalizedRgb[y][x][2] = b;
            }
        }

        return normalizedRgb;
    }
}