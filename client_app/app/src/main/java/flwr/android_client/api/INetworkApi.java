package flwr.android_client.api;


import okhttp3.ResponseBody;
import retrofit2.Call;
import retrofit2.http.GET;

public interface INetworkApi {
    @GET("/s/tubgpepk2q6xiny/models.zip?dl=1")
    Call<ResponseBody> downloadFile();
}
