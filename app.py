from flask import Flask, redirect, render_template, request, send_file, send_from_directory, url_for, flash
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from keras.models import Sequential
from keras.layers import Dense, Input
from keras import optimizers
from keras.callbacks import EarlyStopping
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Ensure the 'static' directory exists
if not os.path.exists('static'):
    os.makedirs('static')

# Function to process the uploaded CSV file
def process_uploaded_file(uploaded_file):
    try:
        # Load the CSV file with specified encoding
        data = pd.read_csv(uploaded_file, encoding='utf-8')
        print("CSV file loaded successfully")

        # Assuming the last column contains labels and the rest are gene expression values
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        print("Data shapes - X: {}, y: {}".format(X.shape, y.shape))

        # Handle missing values by imputing with the mean of each column
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
        print("Missing values handled")

        # Standardize the data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        print("Data standardized")

        # Apply PCA to reduce dimensionality based on explained variance ratio
        pca = PCA(n_components=0.99)  # Retain 99% of the variance
        X_pca = pca.fit_transform(X)
        print("PCA applied, X_pca shape: {}".format(X_pca.shape))

        # Stacked Autoencoder model using Keras
        model = Sequential()
        model.add(Input(shape=(X_pca.shape[1],)))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(X_pca.shape[1], activation='linear'))
        print("Model created")

        # Use the Adam optimizer with a lower learning rate
        optimizer = optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        print("Model compiled")

        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        print("Early stopping configured")

        # Train the model
        model.fit(X_pca, X_pca, epochs=200, batch_size=64, shuffle=True, callbacks=[early_stopping])
        print("Model training completed")

        # Use the trained autoencoder to transform the data
        X_encoded = model.predict(X_pca)
        print("Data transformed using the trained autoencoder")

        # Concatenate encoded features with labels
        X_encoded_with_labels = pd.DataFrame(data=np.c_[X_encoded, y], 
                                             columns=[f'feature_{i}' for i in range(X_encoded.shape[1])] + ['label'])
        print("Encoded features concatenated with labels")
        
        return X_encoded_with_labels
    
    except Exception as e:
        print("Error during processing: {}".format(e))
        raise

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route("/upload", methods=['GET', 'POST'])
def upload():
    print("Inside upload route")
    if request.method == 'POST':
        try:
            # Get the uploaded file from the form
            uploaded_file = request.files['file']

            if uploaded_file.filename != '':
                # Process the uploaded file
                processed_data = process_uploaded_file(uploaded_file)

                # Save the processed data to a new CSV file with UTF-8 encoding
                processed_file_path = 'static/processed_data.csv'
                processed_data.to_csv(processed_file_path, index=False, encoding='utf-8')

                # Redirect to the results page with the processed data
                return redirect(url_for('results', file_path=processed_file_path))
            else:
                flash('No file uploaded. Please upload a CSV file.', 'error')
        except Exception as e:
            flash(str(e), 'error')
            return redirect(url_for('upload'))

    # If the request method is GET or the file is not provided, render the upload page
    return render_template("upload.html")

@app.route("/results")
def results():
    # Get the file path from the query parameters
    file_path = request.args.get('file_path', None)
    print(f"File path received: {file_path}")

    # Pass the file path to the template
    return render_template("results.html", processed_file_path=file_path)

# Add a new route for downloading
@app.route("/download_processed/<filename>")
def download_processed(filename):
    return send_from_directory('static', filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
