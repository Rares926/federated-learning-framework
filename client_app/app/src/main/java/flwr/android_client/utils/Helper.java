package flwr.android_client.utils;


import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.google.protobuf.ByteString;

public class Helper {

    final static int BYTES_IN_FLOAT = Float.SIZE / Byte.SIZE;

    final static List<String> layerNames= Arrays.asList("0/kernel:0","0/bias:0","1/kernel:0","1/bias:0");

    public static Map<String, Object> convertToParameters(List<ByteString> layers, int numClasses){

        //EXTRACT BYTES IN A BYTEBUFFER AND ORDER IT NATIVELY
        ByteBuffer[] parameters = new ByteBuffer[4];
        for (int i = 0; i < layers.size(); i++) {
            parameters[i] = ByteBuffer.wrap(layers.get(i).toByteArray());
            parameters[i].order(ByteOrder.nativeOrder());
        }

        Map<String, Object> newParameters = new HashMap<>();
        for (int i=0; i<layers.size(); i++){
            float[] tmpArray=new float[layers.get(i).size()/BYTES_IN_FLOAT];

            for(int it = 0; it< layers.get(i).size()/BYTES_IN_FLOAT; it++)
            {
                tmpArray[it]=parameters[i].getFloat();
            }

            if (layerNames.get(i).contains("bias"))
            {
                newParameters.put(layerNames.get(i),tmpArray);
            }
            else
            {
                if(i==0){
                    float[][] tmpLayer=new float[7 * 7 * 1280][128];
                    int from=0;
                    int to=128;
                    for(int w=0; w<tmpLayer.length;w++){
                        tmpLayer[w]=Arrays.copyOfRange(tmpArray,from,to);
                        from=from+128;
                        to=to+128;
                    }

                    newParameters.put(layerNames.get(i),tmpLayer);
                }
                else{
                    float[][] tmpLayer=new float[128][numClasses];
                    int from=0;
                    int to=numClasses;
                    for(int w=0; w<tmpLayer.length;w++){
                        tmpLayer[w]=Arrays.copyOfRange(tmpArray,from,to);
                        from=from+numClasses;
                        to=to+numClasses;
                    }
                    newParameters.put(layerNames.get(i),tmpLayer);
                }
            }

        }

        return newParameters;

    }

    public static List<ByteString> convertToByteString(Map<String ,Object> weights ){

        List<ByteString> convertedWeights=new ArrayList<>();

        for(int i=0;i<weights.size();i++)
        {
            if(layerNames.get(i).contains("bias"))
            {
               float[] layer= (float[]) weights.get(layerNames.get(i));
               byte[] convertedLayer=Helper.toByteArray(layer);
               convertedWeights.add(ByteString.copyFrom(convertedLayer));

            }
            else{
                float[][] layer =(float[][]) weights.get(layerNames.get(i));
                int dim1=layer.length;
                int dim2=layer[0].length;
                float[] resizedLayer=new float[dim1*dim2];

                int weightDim=0;

                for (int w=0;w<dim1;w++)
                {
                    //COPY ELEMENTS IN THE RESIZE LAYER
                    for (int w2=0;w2<dim2;w2++)
                    {
                        resizedLayer[weightDim+w2]=layer[w][w2];
                    }
                    weightDim=weightDim+dim2;

                }

                byte[] convertedLayer=Helper.toByteArray(resizedLayer);
                convertedWeights.add(ByteString.copyFrom(convertedLayer));
            }
        }

        return convertedWeights;
    }


    public static byte[] toByteArray(float[] floatArray) {
        ByteBuffer buffer = ByteBuffer.allocate(floatArray.length * BYTES_IN_FLOAT);
        buffer.order(ByteOrder.nativeOrder());
        buffer.asFloatBuffer().put(floatArray);
        return buffer.array();
    }

}
