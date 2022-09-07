package flwr.android_client.utils;

import android.widget.Toast;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

import flwr.android_client.MainActivity;
import flwr.android_client.MainApp;

public class Decompress {
    private File _zipFile;
    private InputStream _zipFileStream;

    public Decompress(File zipFile) {
        _zipFile = zipFile;
    }

    public Decompress(InputStream zipFile) {
        _zipFileStream = zipFile;
    }

    public void unzip() {
        try  {
            InputStream fin = _zipFileStream;
            if(fin == null) {
                fin = new FileInputStream(_zipFile);
            }
            ZipInputStream zin = new ZipInputStream(fin);
            ZipEntry ze;
            while ((ze = zin.getNextEntry()) != null) {
                File modelsFolder = new File(MainActivity.rootFolder, "models");
                modelsFolder.mkdirs();

                String destinationName = ze.getName();
                FileOutputStream fout = new FileOutputStream(new File(modelsFolder, destinationName));
                ByteArrayOutputStream baos = new ByteArrayOutputStream();
                byte[] buffer = new byte[1024];
                int count;

                // reading and writing
                while((count = zin.read(buffer)) != -1)
                {
                    baos.write(buffer, 0, count);
                    byte[] bytes = baos.toByteArray();
                    fout.write(bytes);
                    baos.reset();
                }

                fout.close();
                zin.closeEntry();
            }
            zin.close();
            Toast.makeText(MainApp.getInstance(), "Unzip models successfully", Toast.LENGTH_SHORT).show();
        } catch(Exception e) {
            e.printStackTrace();
        }
    }
}
