package flwr.android_client.utils;

import flwr.android_client.api.INetworkApi;
import retrofit2.Retrofit;
import retrofit2.converter.jackson.JacksonConverterFactory;


public class RetrofitClient {

    private static Retrofit retrofit;
    private static INetworkApi networkApi;
    private static final String baseUrl = "https://dropbox.com";

    private RetrofitClient() { }

    public static INetworkApi getClient() {
        if (retrofit == null) {
            setupRetrofitClient();
            networkApi = retrofit.create(INetworkApi.class);
        }
        if (networkApi == null) {
            networkApi = retrofit.create(INetworkApi.class);
        }
        return networkApi;
    }

    private static void setupRetrofitClient() {
        retrofit = new Retrofit.Builder()
                .baseUrl(baseUrl)
                .addConverterFactory(JacksonConverterFactory.create())
                .build();
    }


}
