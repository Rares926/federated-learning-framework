package flwr.android_client;

import android.app.Activity;
import android.app.Dialog;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.icu.text.SimpleDateFormat;
import android.net.Uri;
import android.os.AsyncTask;

import android.os.Bundle;

import androidx.annotation.NonNull;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.os.Environment;
import android.os.Handler;
import android.provider.MediaStore;
import android.text.TextUtils;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.util.Pair;
import android.util.Patterns;
import android.view.View;
import android.view.ViewGroup;
import android.view.inputmethod.InputMethodManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TableLayout;
import android.widget.TableRow;
import android.widget.TextView;
import android.widget.Toast;

import flwr.android_client.utils.Decompress;
import flwr.android_client.utils.RetrofitClient;
import flwr.android_client.utils.Utils;
import flwr.android_client.utils.Helper;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;

import  flwr.android_client.FlowerServiceGrpc.FlowerServiceBlockingStub;
import  flwr.android_client.FlowerServiceGrpc.FlowerServiceStub;

import com.github.dhaval2404.imagepicker.ImagePicker;
import com.google.protobuf.ByteString;

import org.tensorflow.lite.examples.transfer.api.TransferLearningModel.Prediction;
import io.grpc.stub.StreamObserver;
import okhttp3.ResponseBody;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

import java.io.File;

import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;


public class MainActivity extends AppCompatActivity {

    // Uncomment after flask is up and running
    public static File rootFolder = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "Models-Rares");

    //Dialog params
    private Button dialogButtonTrain;
    Dialog dialogTrain;
    private Button dialogButtonInfer;
    Dialog dialogInfer;

    // Inference params
    private Button loadImageButton;
    private Button inference;

    //Inference results params
    Dialog dialogInferenceResults;
    TableLayout inferenceResults;


    // Training And connection params
    private EditText ip;
    private EditText port;
    private Button loadDataButton;
    private Button connectButton;
    private Button trainButton;
    private TextView resultText;
    private EditText device_id;
    private ManagedChannel channel;
    public FlowerClient fc;
    private static String TAG = "Flower";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Uncomment after flask is up and running
        // downloadModels();

        // Training and connection params
        resultText = (TextView) findViewById(R.id.grpc_response_text);
        resultText.setMovementMethod(new ScrollingMovementMethod());
        device_id = (EditText) findViewById(R.id.device_id_edit_text);
        ip = (EditText) findViewById(R.id.serverIP);
        port = (EditText) findViewById(R.id.serverPort);
        loadDataButton = (Button) findViewById(R.id.load_data) ;
        connectButton = (Button) findViewById(R.id.connect);
        trainButton = (Button) findViewById(R.id.trainFederated);


        //Dialog Train params
        dialogTrain = new Dialog(MainActivity.this);
        dialogTrain.setContentView(R.layout.custom_dialog);
        dialogTrain.getWindow().setLayout(ViewGroup.LayoutParams.MATCH_PARENT,ViewGroup.LayoutParams.WRAP_CONTENT);
        //If this is false when we click the outside of the box it will not dissapear
        dialogTrain.setCancelable(false);

        Button ok =dialogTrain.findViewById(R.id.btn_okay);
        ok.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                dialogTrain.dismiss();
            }
        });

        dialogButtonTrain = (Button) findViewById(R.id.dialog_btn);
        dialogButtonTrain.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                dialogTrain.show();
            }
        });

        //Dialog Infer params
        dialogInfer = new Dialog(MainActivity.this);
        dialogInfer.setContentView(R.layout.custom_dialog_infer);
        dialogInfer.getWindow().setLayout(ViewGroup.LayoutParams.MATCH_PARENT,ViewGroup.LayoutParams.WRAP_CONTENT);
        //If this is false when we click the outside of the box it will not dissapear
        dialogInfer.setCancelable(false);

        Button okInfer =dialogInfer.findViewById(R.id.btn_okay_infer);
        okInfer.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                dialogInfer.dismiss();
            }
        });

        dialogButtonInfer = (Button) findViewById(R.id.dialog_btn_infer);
        dialogButtonInfer.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                dialogInfer.show();
            }
        });

        // Inference params
        loadImageButton = (Button) findViewById(R.id.load_image);
        inference = (Button) findViewById(R.id.inference);

        fc = new FlowerClient(this);
    }

    public static void hideKeyboard(Activity activity) {
        InputMethodManager imm = (InputMethodManager) activity.getSystemService(Activity.INPUT_METHOD_SERVICE);
        View view = activity.getCurrentFocus();
        if (view == null) {
            view = new View(activity);
        }
        imm.hideSoftInputFromWindow(view.getWindowToken(), 0);
    }


    public void setResultText(String text) {
        runOnUiThread(()->{
            SimpleDateFormat dateFormat = new SimpleDateFormat("HH:mm:ss");
            String time = dateFormat.format(new Date());
            resultText.append("\n" + time + "   " + text);
        });

    }



    public void loadImage(View view) throws InterruptedException {
        ImagePicker.with(MainActivity.this)
                        .galleryOnly()                       // For now functionality covers only gallery
                        .cropSquare()	    			     //Crop image(Optional), Check Customization for more option
                        .compress(1024)			             //Final image size will be less than 1 MB(Optional)
                        .maxResultSize(224, 224) //Final image resolution will be less than 1080 x 1080(Optional)
                .start();
        TimeUnit.SECONDS.sleep(1);
        inference.setEnabled(true);

    }

    public void predict(View view){

        dialogInferenceResults = new Dialog(MainActivity.this);
        dialogInferenceResults.setContentView(R.layout.inference_results);
        dialogInferenceResults.getWindow().setLayout(ViewGroup.LayoutParams.MATCH_PARENT,ViewGroup.LayoutParams.WRAP_CONTENT);
        //If this is false when we click the outside of the box it will not dissapear
        dialogInferenceResults.setCancelable(true);

        inferenceResults = (TableLayout) dialogInferenceResults.findViewById(R.id.inferenceTableLayout);

        inference.setEnabled(false);
        Prediction[] inferencePrediction = fc.predict();
        setResultText("Prediction done");
        setResultText("Reccomended class: "+inferencePrediction[0].getClassName()+" confidence: "+inferencePrediction[0].getConfidence());

        TextView[] predictions = new TextView[inferencePrediction.length];
        TextView[] accurracy = new TextView[inferencePrediction.length];
        TableRow[] tr_head = new TableRow[inferencePrediction.length];

        for(int i=0;i<inferencePrediction.length;i++){
            //Create the table rows
            tr_head[i] = new TableRow(this);
            tr_head[i].setId(i+1);
            tr_head[i].setBackgroundColor(Color.GRAY);
            tr_head[i].setLayoutParams(new TableRow.LayoutParams(TableRow.LayoutParams.MATCH_PARENT,
                                                                TableRow.LayoutParams.WRAP_CONTENT));

            predictions[i]=new TextView(this);
            predictions[i].setText(inferencePrediction[i].getClassName());
            predictions[i].setTextColor(Color.WHITE);
            predictions[i].setPadding(5,5,5,5);

            accurracy[i]=new TextView(this);
            accurracy[i].setText(String.valueOf(inferencePrediction[i].getConfidence()));
            accurracy[i].setTextColor(Color.WHITE);
            accurracy[i].setPadding(5,5,5,5);

            tr_head[i].addView(predictions[i]);
            tr_head[i].addView(accurracy[i]);

            inferenceResults.addView(tr_head[i],new TableLayout.LayoutParams(TableLayout.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        }


        dialogInferenceResults.show();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == Activity.RESULT_OK) {

            // Image Uri will not be null for RESULT_OK
            Uri imageUri = data.getData();

            try {
                Bitmap bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageUri);
                fc.addInferenceImage(bitmap);
            } catch (IOException e) {
                e.printStackTrace();
            }

            setResultText("Inference image loaded");

        } else if (resultCode == ImagePicker.RESULT_ERROR) {
            Toast.makeText(this, ImagePicker.getError(data), Toast.LENGTH_SHORT).show();
        } else {
            Toast.makeText(this, "Task Cancelled", Toast.LENGTH_SHORT).show();
        }
    }


    public void loadData(View view){
        if (TextUtils.isEmpty(device_id.getText().toString())) {
            Toast.makeText(this, "Please enter a client partition ID between 1 and 4 (inclusive)", Toast.LENGTH_LONG).show();
        }
        else if (Integer.parseInt(device_id.getText().toString()) > 4 ||  Integer.parseInt(device_id.getText().toString()) < 1)
        {
            Toast.makeText(this, "Please enter a client partition ID between 1 and 4 (inclusive)", Toast.LENGTH_LONG).show();
        }
        else{
            hideKeyboard(this);
            setResultText("Loading the local training dataset in memory. It will take several seconds.");
            loadDataButton.setEnabled(false);
            final Handler handler = new Handler();
            handler.postDelayed(new Runnable() {
                @Override
                public void run() {
                    fc.loadData(Integer.parseInt(device_id.getText().toString()));
                    setResultText("Training dataset is loaded in memory.");
                    connectButton.setEnabled(true);
                }
            }, 1000);
        }
    }

    public void connect(View view) {
        String host = ip.getText().toString();
        String portStr = port.getText().toString();
        if (TextUtils.isEmpty(host) || TextUtils.isEmpty(portStr) || !Patterns.IP_ADDRESS.matcher(host).matches()) {
            Toast.makeText(this, "Please enter the correct IP and port of the FL server", Toast.LENGTH_LONG).show();
        }
        else {
            int port = TextUtils.isEmpty(portStr) ? 0 : Integer.valueOf(portStr);
            // HERE WE INCREASE THE MAXIMUM SIZE OF THE GRPC MESSAGE TO CONCLUDE WITH OUR NEEDS
            //channel = ManagedChannelBuilder.forAddress(host, port).maxInboundMessageSize(10 * 1024 * 1024).usePlaintext().build();
            channel = ManagedChannelBuilder.forAddress(host, port).maxInboundMessageSize(32118900).usePlaintext().build();
            hideKeyboard(this);
            trainButton.setEnabled(true);
            connectButton.setEnabled(false);
            setResultText("Channel object created. Ready to train!");
        }
    }

    //This must be uncomment later
    private void downloadModels(){
        RetrofitClient.getClient().downloadFile().enqueue(new Callback<ResponseBody>() {
            @Override
            public void onResponse(@NonNull Call<ResponseBody> call, @NonNull Response<ResponseBody> response) {
                if(response.isSuccessful()) {
                    Toast.makeText(MainActivity.this, "Downloaded models successfully", Toast.LENGTH_SHORT).show();
                    onModelsDownloadedSuccessfully(response);
                }
            }

            @Override
            public void onFailure(@NonNull Call<ResponseBody> call,@NonNull Throwable t) {
                t.printStackTrace();
            }
        });
    }

    // This must be uncomment later
    private void onModelsDownloadedSuccessfully(Response<ResponseBody> response){
        String filename = "models.zip";
        if (!rootFolder.exists() && !rootFolder.mkdirs()) {
            return;
        }

        //save zip file to disk
        if(response != null && response.body() != null){
            File zipFile = Utils.writeContentToAppFolder(response.body(), filename);

            if(zipFile != null){  //decompress zip file
                new Decompress(zipFile).unzip();
            }
        }
    }

    public void runGRCP(View view){
        new GrpcTask(new FlowerServiceRunnable(), channel, this).execute();
    }

    private static class GrpcTask extends AsyncTask<Void, Void, String> {
        private final GrpcRunnable grpcRunnable;
        private final ManagedChannel channel;
        private final MainActivity activityReference;

        GrpcTask(GrpcRunnable grpcRunnable, ManagedChannel channel, MainActivity activity) {
            this.grpcRunnable = grpcRunnable;
            this.channel = channel;
            this.activityReference = activity;
        }

        @Override
        protected String doInBackground(Void... nothing) {
            try {
                grpcRunnable.run(FlowerServiceGrpc.newBlockingStub(channel), FlowerServiceGrpc.newStub(channel), this.activityReference);
                return "Connection to the FL server successful \n";
            } catch (Exception e) {
                StringWriter sw = new StringWriter();
                PrintWriter pw = new PrintWriter(sw);
                e.printStackTrace(pw);
                pw.flush();
                return "Failed to connect to the FL server \n" + sw;
            }
        }

        @Override
        protected void onPostExecute(String result) {
            MainActivity activity = activityReference;
            if (activity == null) {
                return;
            }
            activity.setResultText(result);
            activity.trainButton.setEnabled(false);
        }
    }

    private interface GrpcRunnable {
        void run(FlowerServiceBlockingStub blockingStub, FlowerServiceStub asyncStub, MainActivity activity) throws Exception;
    }

    private static class FlowerServiceRunnable implements GrpcRunnable {
        private Throwable failed;
        private StreamObserver<ClientMessage> requestObserver;
        @Override
        public void run(FlowerServiceBlockingStub blockingStub, FlowerServiceStub asyncStub, MainActivity activity)
                throws Exception {
            join(asyncStub, activity);
        }

        private void join(FlowerServiceStub asyncStub, MainActivity activity)
                throws InterruptedException, RuntimeException {

            final CountDownLatch finishLatch = new CountDownLatch(1);
            requestObserver = asyncStub.join(
                    new StreamObserver<ServerMessage>() {
                        @Override
                        public void onNext(ServerMessage msg) {
                            handleMessage(msg, activity);
                        }

                        @Override
                        public void onError(Throwable t) {
                            failed = t;
                            finishLatch.countDown();
                            Log.e(TAG, t.getMessage());
                        }

                        @Override
                        public void onCompleted() {
                            finishLatch.countDown();
                            Log.e(TAG, "Done");
                        }
                    });
        }

        private void handleMessage(ServerMessage message, MainActivity activity) {

            try {
                Map<String, Object> weights;
                ClientMessage c = null;

                if (message.hasGetParameters()) {
                    Log.e(TAG, "Handling GetParameters");

                    activity.setResultText("Handling GetParameters message from the server.");

                    weights = activity.fc.getWeights();
                    c = weightsAsProto(weights);
                } else if (message.hasFitIns()) {
                    Log.e(TAG, "Handling FitIns");

                    activity.setResultText("Handling Fit request from the server.");

                    //GET THE MODEL WEIGHTS AS A LIST OF ByteString from the server
                    List<ByteString> layers = message.getFitIns().getParameters().getTensorsList();

                    // GET THE LOCAL EPOCHS FROM THE SERVER
                    Scalar epoch_config = message.getFitIns().getConfigMap().getOrDefault("local_epochs", Scalar.newBuilder().setSint64(1).build());
                    int local_epochs = (int) epoch_config.getSint64();

                    //CONVERT THE WEIGHTS GOTTEN FROM THE SERVER TO A MAP<String,Object>
                    int numClasses =  activity.fc.getNumClasses();
                    Map<String,Object> convertedNewWeights=Helper.convertToParameters(layers,numClasses);

                    //TESTING THE CONVERSION TO List<ByteString>
                    List<ByteString> finalConverted=Helper.convertToByteString(convertedNewWeights);

                    //CALL FLOWER FIT FUNCTION
                    Pair<Map<String, Object>, Integer> outputs = activity.fc.fit(convertedNewWeights, local_epochs);

                    c = fitResAsProto(outputs.first, outputs.second);

                } else if (message.hasEvaluateIns()) {
                    Log.e(TAG, "Handling EvaluateIns");

                    activity.setResultText("Handling Evaluate request from the server");

                    //GET THE MODEL WEIGHTS AS A LIST OF ByteString from the server
                    List<ByteString> layers = message.getEvaluateIns().getParameters().getTensorsList();

                    //CONVERT THE WEIGHTS GOTTEN FROM THE SERVER TO A MAP<String,Object>
                    int numClasses =  activity.fc.getNumClasses();
                    Map<String,Object> convertedNewWeights=Helper.convertToParameters(layers, numClasses);

                    //CALL FLOWER EVALUATE FUNCTION
                    Pair<Pair<Float, Float>, Integer> inference = activity.fc.evaluate(convertedNewWeights);

                    //GET LOSS AND ACCURACY METRICS
                    float loss = inference.first.first;
                    float accuracy = inference.first.second;

                    activity.setResultText("Test Accuracy after this round = " + accuracy);

                    int test_size = inference.second;
                    c = evaluateResAsProto(loss, test_size);
                }
                requestObserver.onNext(c);

                activity.setResultText("Response sent to the server");
                c = null;
            }
            catch (Exception e){
                Log.e(TAG, e.getMessage());
            }
        }
    }

    private static ClientMessage weightsAsProto(Map<String, Object> weights){

        List<ByteString> layers = new ArrayList<ByteString>();

        layers=Helper.convertToByteString(weights);

        Parameters p = Parameters.newBuilder().addAllTensors(layers).setTensorType("ND").build();
        ClientMessage.ParametersRes res = ClientMessage.ParametersRes.newBuilder().setParameters(p).build();
        return ClientMessage.newBuilder().setParametersRes(res).build();
    }

    private static ClientMessage fitResAsProto(Map<String, Object> weights, int training_size){
        List<ByteString> layers = new ArrayList<ByteString>();

        layers=Helper.convertToByteString(weights);

        Parameters p = Parameters.newBuilder().addAllTensors(layers).setTensorType("ND").build();
        ClientMessage.FitRes res = ClientMessage.FitRes.newBuilder().setParameters(p).setNumExamples(training_size).build();
        return ClientMessage.newBuilder().setFitRes(res).build();
    }

    private static ClientMessage evaluateResAsProto(float accuracy, int testing_size){
        ClientMessage.EvaluateRes res = ClientMessage.EvaluateRes.newBuilder().setLoss(accuracy).setNumExamples(testing_size).build();
        return ClientMessage.newBuilder().setEvaluateRes(res).build();
    }
}
