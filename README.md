# **Midterm Project for Machine Learning Zoomcamp**  
**by Vesna Pop-Dimitrijoska**  

---

## **Project Title**  
**Machine Learning Regression Model for Airline Delay Prediction**

![alt text](picture.webp)

---

## **Project Description**  

This project focuses on building a machine learning regression model to predict airline delays using a publicly available dataset. The project involves a systematic approach including:

- **Exploratory Data Analysis (EDA)** to understand patterns in the data.
- **Data Preprocessing and Engineering** to clean and transform the data.
- **Model Training and Optimization** using baseline models, Random Search, Grid Search, and genetic programming with TPOT.
- **Model Deployment** to serve predictions via a Flask API, with the model serialized in a `.pkl` file.
- **Containerization** of the API using Docker for reproducibility.

The dataset was sourced from Kaggle: [Airline Delay Causes](https://www.kaggle.com/datasets/giovamata/airlinedelaycauses).  
*(Note: The dataset is large and not uploaded with this project. You can download it directly from Kaggle.)*

**Target Variable:**  
The target is a new column derived by summing the five delay-related columns in the dataset. Rows with missing values in any of the delay columns were removed before summing.

---

## **Table of Contents**
1. Problem Description
2. Dataset Overview
3. Data Preprocessing and EDA
4. Model Training
   - Baseline Models
   - Random Search Optimization
   - Grid Search Optimization
   - TPOT Model Optimization
5. Model Evaluation
6. Model Deployment
7. Reproducibility and Containerization
8. Results and Conclusion

---

## **Dataset**
### **Features in the Dataset**
Features in Airline Delay dataset:

1.	Year 2008
2.	Month 1-12
3.	DayofMonth 1-31
4.	DayOfWeek 1 (Monday) - 7 (Sunday)
5.	DepTime actual departure time (local, hhmm)
6.	CRSDepTime scheduled departure time (local, hhmm)
7.	ArrTime actual arrival time (local, hhmm)
8.	CRSArrTime scheduled arrival time (local, hhmm)
9.	UniqueCarrier unique carrier code
10.	FlightNum flight number
11.	TailNum plane tail number: aircraft registration, unique aircraft identifier
12.	ActualElapsedTime in minutes
13.	CRSElapsedTime in minutes
14.	AirTime in minutes
15.	ArrDelay arrival delay, in minutes: A flight is counted as "on time" if it operated less than 15 minutes later the scheduled time shown in the carriers' Computerized Reservations Systems (CRS).
16.	DepDelay departure delay, in minutes
17.	Origin origin IATA airport code
18.	Dest destination IATA airport code
19.	Distance in miles
20.	TaxiIn taxi in time, in minutes
21.	TaxiOut taxi out time in minutes
22.	Cancelled *was the flight cancelled
23.	CancellationCode reason for cancellation (A = carrier, B = weather, C = NAS, D = security)
24.	Diverted 1 = yes, 0 = no
25.	CarrierDelay in minutes: Carrier delay is within the control of the air carrier. Examples of occurrences that may determine carrier delay are: aircraft cleaning, aircraft damage, awaiting the arrival of connecting passengers or crew, baggage, bird strike, cargo loading, catering, computer, outage-carrier equipment, crew legality (pilot or attendant rest), damage by hazardous goods, engineering inspection, fueling, handling disabled passengers, late crew, lavatory servicing, maintenance, oversales, potable water servicing, removal of unruly passenger, slow boarding or seating, stowing carry-on baggage, weight and balance delays.
26.	WeatherDelay in minutes: Weather delay is caused by extreme or hazardous weather conditions that are forecasted or manifest themselves on point of departure, enroute, or on point of arrival.
27.	NASDelay in minutes: Delay that is within the control of the National Airspace System (NAS) may include: non-extreme weather conditions, airport operations, heavy traffic volume, air traffic control, etc.
28.	SecurityDelay in minutes: Security delay is caused by evacuation of a terminal or concourse, re-boarding of aircraft because of security breach, inoperative screening equipment and/or long lines in excess of 29 minutes at screening areas.
29.	LateAircraftDelay in minutes: Arrival delay at an airport due to the late arrival of the same aircraft at a previous airport. The ripple effect of an earlier delay at downstream airports is referred to as delay propagation.

**Dataset Shape:**  
The dataset consists of 1,936,758 rows and 29 columns.  

### **Target Variable**  
The target variable, `Delay`, is the sum of the following delay columns:  
- CarrierDelay
- WeatherDelay
- NASDelay
- SecurityDelay
- LateAircraftDelay  

Missing values in these columns were handled by dropping rows with NaNs.

---

## **Project Workflow**
The project is divided into distinct steps, with each step documented in corresponding notebooks or scripts.

### **1. Data Preprocessing and EDA**  
Documented in the notebook: **`data_preprocessing_and_EDA.ipynb`**.  
- Handled missing values and irrelevant features.
- Explored correlations between features and the target variable.
- Visualized feature distributions and transformed skewed features.

### **2. Baseline Model Testing**  
Documented in the notebook: **`baseline_models.ipynb`**.  
- Evaluated multiple baseline regression models (e.g., Dummy Regressor, Linear Regression, Decision Tree, Random Forest).
- Identified Random Forest as the best-performing baseline model.

### **3. Model Optimization with Grid Search**  
Documented in the notebook: **`grid_search.ipynb`**.  
- Performed hyperparameter tuning using Grid Search.
- Limited dataset size due to computational constraints.
- Grid Search performed worse than baseline models due to insufficient data.

### **4. Model Optimization with Random Search**  
Documented in the notebook: **`random_search.ipynb`**.  
- Conducted Random Search for hyperparameter tuning.
- Random Forest Regressor achieved the best performance.

### **5. TPOT Regressor**  
Documented in the notebook: **`TPOT.ipynb`**.  
- Applied TPOT for automated machine learning using genetic programming.
- Performance was not satisfactory due to limited data usage (10% of the dataset) and high computational costs.

### **6. Final Model Training**  
Implemented in the script: **`train.py`**.  
- Trained the best-performing Random Forest model with optimal hyperparameters identified via Random Search.
- Exported the trained model to a file: **`final_model.pkl`**.

### **7. Model Deployment**  
Implemented in the script: **`predict.py`**.  
- Served predictions using a Flask API.
- The API accepts a JSON payload containing feature values and returns the predicted delay.

---

## **Note on Missing Files**

### **1. Newly Generated File**
During the preprocessing and EDA steps, a cleaned and transformed version of the dataset was generated. This file was not uploaded as part of the project because it is very large. You can generate it by running the preprocessing notebook: **`data_preprocessing_and_EDA.ipynb`**.

### **2. Pickle File (`final_model.pkl`)**
The serialized Random Forest model, saved as **`final_model.pkl`**, was not uploaded due to its size (approximately 1.8 GB). You can regenerate this file by running the training script: **`train.py`**. The model will be saved in the project directory upon execution.

---

If you want to work with these files, you can:
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/giovamata/airlinedelaycauses).
2. Run the notebooks and scripts provided in this project to regenerate the necessary files.

---

## **Reproducibility**
### **Dependencies**
All dependencies are listed in the file **`requirements.txt`**.  
To set up the environment, run:
```bash
pip install -r requirements.txt
```


### **Docker Containerization**
The project includes a Dockerfile for containerization.

**Build the Docker Image**:

```bash
docker build -t airline-delay-api .
```

**Run the Docker Container**:

```bash
docker run -p 5000:5000 airline-delay-api
```

**Test the API**:
Use curl or Postman to test the running API:

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{
    "Unnamed: 0": 0,
    "Month": 1,
    "DayOfWeek": 4,
    "DepTime": 1829,
    "CRSDepTime": 1755,
    "ArrTime": 1959,
    "CRSArrTime": 1925,
    "UniqueCarrier": "WN",
    "TailNum": "N464WN",
    "ActualElapsedTime": 9.48683298050514,
    "Origin": "IND",
    "Dest": "BWI",
    "TaxiIn": 1.73205080756888,
    "TaxiOut": 3.16227766016838
}'
```
---

## **Results**      
### **Baseline Models**
Random Forest Regressor outperformed other baseline models.
       
### **Hyperparameter Optimization**
**Random Search**: Achieved best R² score with optimized Random Forest model.      
**Grid Search**: Underperformed due to dataset size limitations.     
**TPOT**: High computational cost limited its utility.      
       
### **Final Model**
The best-performing model is a Random Forest Regressor trained with Random Search-optimized hyperparameters.

---       
           
## **Conclusion**
This project demonstrates the end-to-end process of building, optimizing, and deploying a machine learning model for airline delay prediction. The Random Forest Regressor proved to be the most effective model, balancing accuracy and computational efficiency.