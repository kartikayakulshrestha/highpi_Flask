'''import cv2
from flask import Flask, request, jsonify, render_template
import os

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def process_image(file_path):
    image = cv2.imread(file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None  # No faces found

    mask = image.copy()
    mask[:,:,:] = 0

    for (x, y, w, h) in faces:
        cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), thickness=cv2.FILLED)

    result = cv2.bitwise_and(image, mask)
    return result

@app.route('/faceCutOut', methods=['POST'])
def face_cut_out():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the uploaded file
    file_path = 'uploads/' + file.filename
    file.save(file_path)

    # Process the image
    result_image = process_image(file_path)

    if result_image is None:
        return jsonify({'error': 'No faces found'})

    # Save the result
    result_path = 'results/' + file.filename
    cv2.imwrite(result_path, result_image)

    return jsonify({'result': 'Face cropped successfully', 'result_path': result_path})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    app.run(debug=True)'''

'''
from flask import Flask, jsonify, send_file
import os

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/api/greet/<name>')
def greet(name):
    return jsonify({'message': f'Hello, {name}!'})

@app.route('/get_code')
def get_code():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return jsonify({'error': 'No selected file'})
    # Save the uploaded file


    file_name = secure_filename(uploaded_file.filename)
    file_name= os.path.join("E:/react project/highpi/api/",file_name)
    uploaded_file.save(file_name)
    return jsonify({"filename":str(file_name)
                  ,"uploaded":str(uploaded_file) })
    #return send_file(file_name, as_attachment=True),200

if __name__ == '__main__':
    app.run(debug=True)'''

from flask import Flask, jsonify, send_file, request
import os
import numpy as np
from werkzeug.utils import secure_filename
import cv2
from flask_cors import CORS
app = Flask(__name__)
CORS(app) 
@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/api/greet/<name>')
def greet(name):
    return jsonify({'message': f'Hello, {name}!'})

@app.route('/facecutout', methods=['POST'])
def facecutout():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return jsonify({'error': 'No selected file'})
    # Save the uploaded file
    # Load the pre-trained face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read the uploaded image
    image_data = np.fromstring(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    
    # Convert the image to grayscale (face detection works better on grayscale images)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Create a mask for the face regions
    mask = image.copy()
    mask[:,:,:] = 0  # Set the mask to black initially
    for (x, y, w, h) in faces:
        cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), thickness=cv2.FILLED)

    # Remove everything except the face from the original image
    result = cv2.bitwise_and(image, mask)
    #x=cv2.imwrite("E:/react project/highpi/api/"+str(uploaded_file)+".png",result)

    #file_name = secure_filename(uploaded_file.filename)
    #file_name= os.path.join("E:/react project/highpi/api/",file_name)
    #uploaded_file.save(file_name)
    
    #return jsonify({
    #              "uploaded":str(uploaded_file)
    #              ,"result":str(result) })
    #return send_file(x, as_attachment=True),200

    # Save the processed image with a secure filename
    filename, extension = os.path.splitext(uploaded_file.filename)

    # Secure the filename and add the desired extension
    secure_file_name = secure_filename(filename) + extension

    # Save the processed image with the secured filename
    file_path = os.path.join("E:/react project/highpi/api/saveImage/", secure_file_name)
    cv2.imwrite(file_path, result)

    # Send the result as a file attachment
    return send_file(file_path, as_attachment=True), 200
    
@app.route("/pencilsketch",methods=['POST'])
def pencilsketch():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    image_data = np.fromstring(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Invert the grayscale image
    inverted_gray = cv2.bitwise_not(gray_image)
    
    # Blur the inverted image
    blurred_image = cv2.GaussianBlur(inverted_gray, (111,111),0)
    
    # Invert the blurred image
    inverted_blurred = cv2.bitwise_not(blurred_image)
    
    # Blend the original image with the inverted blurred image
    pencil_sketch = cv2.divide(gray_image, inverted_blurred, scale=256.0)

    filename, extension = os.path.splitext(uploaded_file.filename)
    secure_file_name = secure_filename(filename) + extension

    file_path = os.path.join("E:/react project/highpi/api/saveImage/", secure_file_name)
    cv2.imwrite(file_path, pencil_sketch)

    return send_file(file_path, as_attachment=True), 200


@app.route("/backgroundremoving",methods=['POST'])
def backgroundremoving():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return jsonify({'error': 'No selected file'})

    image_data = np.fromstring(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Define a rectangle around the object to help the algorithm
    rect = (50, 50, image.shape[1] - 50, image.shape[0] - 50)

    # Apply grabCut algorithm
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Modify the mask to create a binary mask for the foreground
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    background_removed = image * mask2[:, :, np.newaxis]

    filename, extension = os.path.splitext(uploaded_file.filename)
    secure_file_name = secure_filename(filename) + extension

    file_path = os.path.join("E:/react project/highpi/api/saveImage/", secure_file_name)
    cv2.imwrite(file_path, background_removed)

    return send_file(file_path, as_attachment=True), 200


@app.route("/cartoony",methods=['POST'])
def cartoony():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return jsonify({'error': 'No selected file'})

    image_data = np.fromstring(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to smooth the image while preserving edges
    smooth = cv2.bilateralFilter(gray, 9, 300, 300)

    # Apply edge detection using adaptive thresholding
    edges = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 15, 13)

    cartoon = cv2.bitwise_and(image, image, mask=edges)

    filename, extension = os.path.splitext(uploaded_file.filename)
    secure_file_name = secure_filename(filename) + extension

    file_path = os.path.join("E:/react project/highpi/api/saveImage/", secure_file_name)
    cv2.imwrite(file_path, cartoon)

    return send_file(file_path, as_attachment=True), 200

@app.route("/greyscale",methods=['POST'])
def greyscale():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return jsonify({'error': 'No selected file'})

    image_data = np.fromstring(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    filename, extension = os.path.splitext(uploaded_file.filename)
    secure_file_name = secure_filename(filename) + extension

    file_path = os.path.join("E:/react project/highpi/api/saveImage/", secure_file_name)
    cv2.imwrite(file_path, gray_image)

    return send_file(file_path, as_attachment=True), 200


@app.route("/contrast",methods=['POST'])
def contrast():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    alpha, beta=1.5,0
    image_data = np.fromstring(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    contrast_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    filename, extension = os.path.splitext(uploaded_file.filename)
    secure_file_name = secure_filename(filename) + extension

    file_path = os.path.join("E:/react project/highpi/api/saveImage/", secure_file_name)
    cv2.imwrite(file_path, contrast_image)

    return send_file(file_path, as_attachment=True), 200

@app.route("/brightnesss",methods=['POST'])
def brightnesss():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    alpha, beta=1.5,20
    image_data = np.fromstring(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)


    brightness_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    filename, extension = os.path.splitext(uploaded_file.filename)
    secure_file_name = secure_filename(filename) + extension

    file_path = os.path.join("E:/react project/highpi/api/saveImage/", secure_file_name)
    cv2.imwrite(file_path, brightness_image)

    return send_file(file_path, as_attachment=True), 200
if __name__ == '__main__':
    app.run(debug=True)
