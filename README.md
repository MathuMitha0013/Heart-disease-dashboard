# Heart Disease Analysis Dashboard

A comprehensive data visualization and analysis dashboard for heart disease prediction data, built using Streamlit and Plotly.

## 📊 Project Overview

This application provides an interactive dashboard for analyzing heart disease data. It allows users to explore the dataset through various visualizations, statistical summaries, and relationship analyses between different medical features.

## 🛠️ Technologies Used

- **Streamlit** (v1.32.0) - Web application framework for data science projects
- **Pandas** (v2.2.1) - Data manipulation and analysis
- **NumPy** (v1.26.4) - Numerical computing
- **Plotly** (v5.19.0) - Interactive data visualizations
- **Seaborn** (v0.13.2) - Statistical data visualization
- **Matplotlib** (v3.8.4) - Plotting library

## 📁 Project Structure

```
d:\Ai\Project\
├── heart_disease_app.py    # Main Streamlit application
├── heart.csv              # Heart disease dataset
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone or download the project files to your local machine.

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

If you encounter build issues, try installing with pre-built binaries:

```bash
pip install --only-binary=:all: streamlit pandas numpy plotly seaborn matplotlib
```

### Running the Application

Navigate to the project directory and run:

```bash
streamlit run heart_disease_app.py
```

Or alternatively:

```bash
python -m streamlit run heart_disease_app.py
```

The dashboard will open in your default web browser at `http://localhost:8501`.

## 📱 Dashboard Features

### 1. Home
- Overview metrics (Total Patients, Heart Disease Cases, Average Age, Average Cholesterol)
- Sample data preview

### 2. Data Overview
- Dataset statistics (rows, columns, memory usage)
- Data types and missing values
- Statistical summary

### 3. Distributions
- Interactive histograms for numeric features
- Box plots for outlier detection
- Pie charts for categorical features

### 4. Relationships
- Correlation matrix heatmap
- Interactive scatter plots between features

### 5. Target Analysis
- Heart disease distribution pie chart
- Feature comparison by target variable

### 6. Summary
- Key findings and insights from the data

## 📈 Dataset Features

The heart disease dataset includes the following features:

| Feature | Description |
|---------|-------------|
| age | Age of the patient |
| sex | Gender (1 = male, 0 = female) |
| cp | Chest pain type (0-3) |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting electrocardiographic results |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels colored by flourosopy |
| thal | Thalassemia |
| target | Heart disease diagnosis (1 = disease, 0 = no disease) |

## 🔧 Customization

You can customize the dashboard by:
- Adding new visualizations in the ` Distributions` section
- Creating additional analysis pages
- Modifying the preprocessing function to add new derived features
- Adjusting the plot styles and themes

## 📝 License

This project is for educational and demonstration purposes.

## 👤 Author

Created as a heart disease data analysis demonstration using Streamlit.

## 🙏 Acknowledgments

- Dataset source: UCI Machine Learning Repository
- Built with Streamlit and Plotly libraries