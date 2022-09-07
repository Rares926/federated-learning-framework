import zipfile
import os
from flask import send_file, Flask

app = Flask(__name__)

@app.route('/download_files')
def download_all():

    # Zip file Initialization
    # The compression type can also be changed

    # ZIP_STORED : It’s just archiving the Files and it’s a Lossless Compression one
    # ZIP_DEFLATED : This is usual ZIP Compression Method
    # ZIP_BZIP2: This method uses BZIP2 compression technique
    # ZIP_LZMA: This method uses LZMA compression technique

    zipfolder = zipfile.ZipFile('models.zip', 'w', compression=zipfile.ZIP_STORED)

    # zip all the files which are inside in the folder
    for root, _, files in os.walk('Z:/Federated Learning Projects/Flower working android/android/tflite_model'):
        for file in files:
            zipfolder.write(os.path.join(root, file), file)
    zipfolder.close()

    return send_file('models.zip',
                     mimetype='zip',
                     as_attachment=True)

    # Delete the zip file if not needed
    os.remove("models.zip")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)
