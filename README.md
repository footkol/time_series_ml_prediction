## Instructions on how to run the Jupyter notebooks 

All preprocessing, exploratory data analysis (EDA), feature engineering, model training, and prediction of target variables were carried out in Jupyter notebooks. The notebooks should be reviewed in a particular sequential order:
1. 'merging_data.ipynb'
2. 'eda.ipynb'
3. 'model.ipynb' 

The first one covers merging datasets into a single file, the second focuses on EDA, and the third contains data preprocessing and model training. This ensures a structured workflow where each following step builds on the previous one.

All three notebooks are stored in the notebooks folder.

All EDA analysis, model selection thought process and model evaluation are covered in the Jupyter notebooks. These steps ensure a comprehensive workflow, from data exploration and feature engineering to choosing the best models and evaluating their performance, all organized within the notebooks for clarity and reproducibility.


## Instructions on how to run the solution

### Prepare Your Environment

Ensure you have the necessary libraries installed. You can use Poetry for this, or install them manually:

**Using Poetry:**
1. Open a terminal or command prompt.
2. Navigate to the directory containing your `pyproject.toml` file.
3. Install dependencies with:
```
poetry install
```
###  Python Scripts

- **`merging_data.py`**: This script handles data merging.
    Located `app/preprocessing/merging_data.py`

- **`eda.py`**: This script performs exploratory data analysis (EDA).
    Located `app/preprocessing/eda.py`

-  **`model.py`**: This script is used for modeling.
    Located `app/models/model.py`

###  Run the Scripts in Sequence

The scripts need to be run in the following order due to dependencies between them:

- **Run `merging_data.py`**
```
python merging_data.py
```
This script processes and merges data to generate an output file that will be used by the next script.
   
- **Run `eda.py`**
```
python eda.py
```
 This script performs exploratory data analysis on the merged data from `merging_data.py` and generates new output files needed for the modeling script.
   
- **Run `model.py`**
```
python model.py
```
  
  This script uses the data processed by `eda.py` to perform modeling.
### Verify Output

After running each script, verify the output files generated:

- **`merging_data.py`**: Check the output file created by this script (e.g., `../data/train_merged.csv`).
- **`eda.py`**: Ensure the output file from this script is generated correctly (e.g., `../data/train_final.csv, ../data/test_final.csv`).
- **`model.py`**: Check the results of the modeling process (e.g., model performance metrics or output files).
### Additional Notes

- **Path Adjustments**: Ensure the paths to your CSV files are correct and accessible from where you are running the script.
- **Error Handling**: If there are any issues during execution, check for error messages in the terminal and address any missing files or incorrect paths.
- **Permissions**: Make sure you have read/write permissions for the directories and files involved.
- **Custom Adjustments:** If you need to adjust parameters or file names, modify the scripts accordingly before running them.


## Running the FastAPI App

You can run your FastAPI app using Uvicorn.

```
`uvicorn app.main:app --reload`
```

This will start your FastAPI application, and the `/process/` endpoint will be accessible. You can send POST requests to this endpoint with some data, and the preprocessing logic you defined will be applied.


## Docker Detailed Instructions

**Generate requirements.txt**
Run the following command in your project directory (where pyproject.toml is located) to create requirements.txt:
```
poetry export --without-hashes --format=requirements.txt > requirements.txt
```

**Create Your Dockerfile**
Create a file named Dockerfile (without any file extension) in your project’s root directory and add the content from the above Dockerfile.

**Build the Docker Image**
Navigate to your project’s root directory (where the Dockerfile is located) and build the Docker image using the following command:
```
docker build -t fastapi-app .
```
This command builds the Docker image and tags it as fastapi-app.

**Run the Docker Container**
Run the Docker container with the following command:
```
docker run -p 8080:80 fastapi-app
```
This maps port 8080 on your local machine to port 80 in the Docker container. You can access your FastAPI application at http://localhost:8080.

**Verify the Application**

Open a web browser to test your FastAPI application. For example, to test the /docs endpoint, navigate to:
```
http://localhost:8080/docs
```


