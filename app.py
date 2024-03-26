from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import joblib
import os
import pandas as pd
import pytesseract
import json
from PIL import Image
import re
import time

app = Flask(__name__)

# Function to read coordinates from JSON
def read_coordinates_from_json(predicted_name):
    json_file_path = "crop_coordinates.json"
    try:
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
            for item in data.get("data", []):
                if item["template_name"] == predicted_name:
                    return item["coordinates"]
    except Exception as e:
        print(f"Error reading JSON file: {str(e)}")
    return None

# Function to crop and extract name
def crop_and_extract_name(image_path, predicted_name):
    try:
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

        # Retrieve coordinates for the detected template_name from the JSON file
        coordinates_from_json = read_coordinates_from_json(predicted_name)

        # Check if coordinates_from_json is None
        if coordinates_from_json is None:
            return "Not working for this bank"

        # If coordinates_from_json is a single set of coordinates, convert it to a list containing that set
        if isinstance(coordinates_from_json[0], int):
            coordinates_from_json = [coordinates_from_json]

        # Iterate over each set of coordinates
        for coordinates in coordinates_from_json:
            # Convert coordinates to tuple
            name_coordinates = tuple(map(int, coordinates))

            # Cropping coordinates for Name (x1, y1, x2, y2)
            cropped_image_name = Image.open(image_path).crop(name_coordinates)

            # Extract text using pytesseract
            extracted_name = pytesseract.image_to_string(cropped_image_name, lang='eng')

            # Further processing to clean extracted name
            if extracted_name:
                characters_to_replace = ["For", "for", "FOR"]
                for char in characters_to_replace:
                    extracted_name = extracted_name.replace(char, " ")

                extracted_name = re.sub(r'\b[a-z]+\b', ' ', extracted_name)
                extracted_name = extracted_name.replace("&", "And")
                extracted_name = re.sub(r'[^a-zA-Z ]', ' ', extracted_name)
                
                return extracted_name.strip()  # Return the extracted name if text is found

        return "No Name found"

    except Exception as e:
        print(f"Error processing {image_path} for Name: {str(e)}")
        return "Not working for this bank"


@app.route('/', methods=['GET', 'POST'])
def upload_folder_path_or_files():
    if request.method == 'POST':
        uploaded_files = request.files.getlist('files')
        result_df = process_images(uploaded_files)
        result_filename = 'result.csv'
        result_df.to_csv(result_filename, index=False)
        return render_template('download.html', filename=result_filename)
    return render_template('index.html')


def process_images(uploaded_files):
    start_time = time.time()

    model = joblib.load('rf_model_1.joblib')
    l_e = joblib.load("le_1.joblib")

    result_df = pd.DataFrame(columns=['image_name', 'predicted_name', 'extracted_name', 'status'])

    threshold = 0.3  # Set the threshold

    # Create directory if it doesn't exist
    if not os.path.exists('temp'):
        os.makedirs('temp')

    for file in uploaded_files:
        if file.filename.endswith(".tiff"):
            image_name = file.filename
            temp_image_path = os.path.join('temp', secure_filename(image_name))
            file.save(temp_image_path)

            raw_img = cv2.imread(temp_image_path)
            raw_img_gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

            cropped_img = raw_img[0:80, 0:350]
            resized_image = cv2.resize(cropped_img, (100, 100))
            flattened_image = resized_image.flatten()
            flattened_image = flattened_image.reshape(1, -1)

            predicted_class = model.predict(flattened_image)
            probabilities = model.predict_proba(flattened_image)[0]
            max_probability = probabilities.max()

            # Decode the predicted class using the label encoder
            inverse_transformed_class = l_e.inverse_transform(predicted_class)

            if max_probability >= threshold:
                status = 'Predicted'
            else:
                status = 'Not working for this bank'

            extracted_name = crop_and_extract_name(temp_image_path, inverse_transformed_class[0])

            if status == 'Predicted':
                new_row = pd.DataFrame({'image_name': image_name,
                                        'predicted_name': inverse_transformed_class[0],
                                        'extracted_name': extracted_name,
                                         'status': status}, index=[0])
                result_df = pd.concat([result_df, new_row], ignore_index=True)
            else:
                new_row = pd.DataFrame({'image_name': image_name,
                                        'status': status}, index=[0])
                result_df = pd.concat([result_df, new_row], ignore_index=True)

            os.remove(temp_image_path)

    end_time = time.time()
    time_taken = end_time - start_time
    print("Time taken to run the complete code: {:.2f} seconds".format(time_taken))

    return result_df



@app.route('/download/<filename>')
def download_result(filename):
    return send_from_directory(os.getcwd(), filename)


if __name__ == '__main__':
    app.run(debug=True)