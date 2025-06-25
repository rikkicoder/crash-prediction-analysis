```markdown
# Vehicle Crash Severity Prediction

A data science project to analyze and predict the severity of vehicle crashes using machine learning. This repository contains the dataset, analysis code, and trained models to identify key factors contributing to crash outcomes.

---

### Table of Contents
1.  [About The Project](#about-the-project)
    *   [Built With](#built-with)
2.  [Getting Started](#getting-started)
    *   [Prerequisites](#prerequisites)
    *   [Installation](#installation)
3.  [Usage](#usage)
4.  [Dataset](#dataset)
5.  [Methodology](#methodology)
6.  [Authors](#authors)

---

## About The Project

The goal of this project is to understand the factors that influence the severity of a vehicle crash. By analyzing a dataset of crash incidents, we aim to build a machine learning model capable of predicting whether a crash will result in a minor injury, major injury, or be fatal.

This can help in developing better road safety policies, designing safer vehicles, and informing emergency response protocols. The core of the project is the Jupyter Notebook (`ML_project_final.ipynb`), which walks through the data exploration, preprocessing, model training, and evaluation steps .

### Built With
This project is built using Python and several key data science libraries:
*   [Pandas](https://pandas.pydata.org/)
*   [NumPy](https://numpy.org/)
*   [Scikit-learn](https://scikit-learn.org/stable/)
*   [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/)
*   [XGBoost](https://xgboost.ai/)
*   [CatBoost](https://catboost.ai/)

---

## Getting Started

Follow these instructions to set up the project on your local machine for development and testing purposes [5, 7].

### Prerequisites

You need Python 3.x installed on your system. You will also need to install the necessary packages. It is recommended to use a virtual environment.

Install all the requirements using `pip install -r requirements.txt`



1.  **Clone the repository**
    ```
    git clone https://github.com/your_username/your_repository.git
    ```

2.  **Navigate to the project directory**
    ```
    cd your_repository
    ```

3.  **Install the required packages**
    ```
    pip install -r requirements.txt
    ```

---


To run the analysis and train the models, launch Jupyter Lab or Jupyter Notebook and open the `ML_project_final.ipynb` file [2].

```
jupyter lab
```
You can then run the cells in the notebook sequentially to see the entire process, from data loading to model evaluation [5].

---

## Dataset

The project uses the `Data-Sheet-Sheet1-1.csv` dataset, which contains information about various vehicle crashes [1].

**Key columns in the dataset include** [1, 8]:
*   `Crash_Severity`: The target variable (Minor injury, Major injury, Fatal crash).
*   `Vehicle_Speed`: The speed of the vehicle at the time of the crash.
*   `Crash_Time`: The hour of the day when the crash occurred.
*   `Age` & `Gender`: Demographic information of the driver.
*   `Vehicle_Type`: Type of vehicle involved (e.g., Car, Heavy Vehicle).
*   `Road_Type`: The type of road (Urban or Rural).
*   `Alcohol_Consumption`: Whether alcohol was a factor.
*   `Seatbelt_Usage`: Whether a seatbelt was used.
*   `Road_Surface_Condition`: Condition of the road (e.g., Dry, Wet, Icy).

---

## Methodology

The project follows a standard data science workflow [5]:
1.  **Data Loading and Exploration**: The dataset is loaded using Pandas, and an initial exploratory data analysis (EDA) is performed to understand distributions and relationships.
2.  **Data Preprocessing**: Categorical features like `Gender`, `Road_Type`, and `Crash_Type` are handled using one-hot encoding [2].
3.  **Model Training**: Several classification models are trained on the preprocessed data to predict `Crash_Severity`. The models include:
    *   Random Forest Classifier
    *   Gradient Boosting Classifier
    *   XGBoost Classifier
    *   CatBoost Classifier
4.  **Model Evaluation**: The performance of each model is evaluated using metrics such as accuracy score. The results are compared to identify the best-performing model for this task.

---



## Authors

*   **Thota Rithvik** - *Project Lead* 

```
