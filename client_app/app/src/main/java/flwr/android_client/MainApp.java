package flwr.android_client;

import android.app.Application;
import android.os.Build;

public class MainApp extends Application {
    private static MainApp mInstance;

    @Override
    public void onCreate() {
        super.onCreate();
        mInstance = this;
    }

    public static synchronized MainApp getInstance() {
        return mInstance;
    }

    public static boolean sdk29AndUp(){
        return Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q;
    }
}

