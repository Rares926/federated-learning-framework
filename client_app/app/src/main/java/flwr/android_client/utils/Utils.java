package flwr.android_client.utils;

import android.content.ContentValues;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;

import flwr.android_client.MainActivity;
import flwr.android_client.MainApp;
import okhttp3.ResponseBody;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class Utils {
    public static File writeContentToAppFolder(ResponseBody rs, String filename) {
        if (!MainActivity.rootFolder.exists() && !MainActivity.rootFolder.mkdirs()) {
            return null;
        }
        File contentFile = new File(MainActivity.rootFolder, filename);
        if (contentFile.exists()) {
            return contentFile;
        }

        InputStream is = rs.byteStream();
        try {
            if (MainApp.sdk29AndUp()) {
                ContentValues contentValues = new ContentValues();
                contentValues.put(MediaStore.MediaColumns.DISPLAY_NAME, filename);
                contentValues.put(MediaStore.MediaColumns.MIME_TYPE, "application/zip");
                contentValues.put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS + File.separator + "Models-Rares");

                FileOutputStream outputStream = new FileOutputStream(contentFile);

                try {
                    byte[] fileReader = new byte[4096];
                    while (true) {
                        int read = is.read(fileReader);
                        if (read == -1) {
                            break;
                        }
                        outputStream.write(fileReader, 0, read);
                    }

                    outputStream.flush();
                } catch (IOException e) {
                    e.printStackTrace();
                } finally {
                    is.close();
                    outputStream.close();
                }
            } else {
                contentFile = new File(MainActivity.rootFolder, filename);

                DataInputStream dis = new DataInputStream(is);
                byte[] buffer = new byte[1024];
                int length;

                FileOutputStream fos = new FileOutputStream(contentFile);
                while ((length = dis.read(buffer)) > 0) {
                    fos.write(buffer, 0, length);
                }
            }
        } catch (IOException ex) {
            ex.printStackTrace();
        }

        return contentFile;
    }

}
