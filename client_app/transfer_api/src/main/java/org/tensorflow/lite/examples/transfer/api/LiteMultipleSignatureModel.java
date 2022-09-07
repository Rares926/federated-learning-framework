/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.transfer.api;

import java.io.Closeable;
import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

import org.tensorflow.lite.Interpreter;

/** A wrapper for TFLite model with multiple signature runner. */
public class LiteMultipleSignatureModel implements Closeable {
  private static final int BOTTLENECK_SIZE = 7 * 7 * 1280;
  private static final int EXPECTED_BATCH_SIZE = 20;
  private static final int FLOAT_BYTES = 4;
  private final Interpreter interpreter;
  public final int numClasses;

  /**
   * Constructor for the multiple signature model wrapper.
   *
   * @param tfLiteModel byte buffer of the saved flatbuffer model
   * @param numClasses number of class labels
   */
  LiteMultipleSignatureModel(ByteBuffer tfLiteModel, int numClasses) {
    this.interpreter = new Interpreter(tfLiteModel);
    this.numClasses = numClasses;
  }

  /**
   * Loads the bottleneck feature from the given image array.
   *
   * @param image 3-D float array of size (IMG_SIZE, IMG_SIZE, 3)
   * @return 1-D float array containing bottleneck features
   */
  float[] loadBottleneck(float[][][] image) {

    Map<String, Object> inputs = new HashMap<>();
    inputs.put("feature", new float[][][][]{image});
    Map<String, Object> outputs = new HashMap<>();
    float[][] bottleneck = new float[1][BOTTLENECK_SIZE];
    outputs.put("bottleneck", bottleneck);
    this.interpreter.runSignature(inputs, outputs, "load");
    return bottleneck[0];
  }

  /**
   * Runs one training step with the given bottleneck batches and labels.
   *
   * @param bottlenecks 2-D float array of bottleneck batches of size (BATCH_SIZE, BOTTLENECK_SIZE)
   * @param labels 2-D float array of label batches of size (BATCH_SIZE, NUM_CLASSES)
   * @return the training loss
   */

  float runTraining(float[][] bottlenecks, float[][] labels) {
    Map<String, Object> inputs = new HashMap<>();
    inputs.put("bottleneck", bottlenecks);
    inputs.put("label", labels);

    Map<String, Object> outputs = new HashMap<>();
    FloatBuffer loss = FloatBuffer.allocate(1);
    outputs.put("loss", loss);

    this.interpreter.runSignature(inputs, outputs, "train");

    return loss.get(0);
  }

  /**
   * Invokes inference on the given image batches.
   *
   * @param testImage 3-D float array of image of size (IMG_SIZE, IMG_SIZE, 3)
   * @return 1-D float array of softmax output of prediction
   */
  float[] runInference(float[][][] testImage) {
    Map<String, Object> inputs = new HashMap<>();
    inputs.put("image", new float[][][][] {testImage});

    Map<String, Object> outputs = new HashMap<>();
    float[][] output = new float[1][numClasses];
    outputs.put("output", output);
    this.interpreter.runSignature(inputs, outputs, "infer");
    
    return output[0];
  }

  Map<String, Object> extractWeights(){
    // INPUT GETS A STRING VARIABLE THAT WILL NOT BE USED
    // THIS IS A WORKAROUND USED BECAUSE RUN_SIGNATURE REQUIRES
    // A NOT NULL INPUT EVEN IF THE METHOD DOESN'T T REALLY NEEDS AN INPUT VALUE
    Map<String, Object> inputs = new HashMap<>();
    File outputFile=new File("workaround");
    inputs.put("checkpoint_path",outputFile.getAbsolutePath());

    Map<String,Object> outputs=new HashMap<>();
      // OUTPUT WILL GET A VALUE FOR EACH LAYER
      // This might be rewritten with the getInputTensorFromSignature method after it will be pushed to main

    float[][] layer1_kernel=new float[BOTTLENECK_SIZE][128];
    float[] layer1_bias  =new float[128]; //  IN JAVA THE RIGHT FORMAT FOR [128,] IS [128]
    float[][] layer2_kernel=new float[128][numClasses];
    float[] layer2_bias  =new float[numClasses];

    outputs.put("0/kernel:0", layer1_kernel);
    outputs.put("0/bias:0", layer1_bias);
    outputs.put("1/kernel:0", layer2_kernel);
    outputs.put("1/bias:0", layer2_bias);

    this.interpreter.runSignature(inputs,outputs,"extract");

    return outputs;
  }

  void initializeWeights(Map<String,Object> weights){

    Map<String, Object> inputs = new HashMap<>();
    inputs.put("weights1", weights.get("0/kernel:0"));
    inputs.put("bias1", weights.get("0/bias:0"));
    inputs.put("weights2", weights.get("1/kernel:0"));
    inputs.put("bias2", weights.get("1/bias:0"));

    Map<String,Object> outputs=new HashMap<>();

    this.interpreter.runSignature(inputs,outputs,"initialize");

  }

  int getExpectedBatchSize() {
    return EXPECTED_BATCH_SIZE;
  }

  int getNumBottleneckFeatures() {
    return this.interpreter.getInputTensorFromSignature("bottleneck", "train").shape()[1];
  }

  @Override
  public void close() {
    this.interpreter.close();
  }
}
