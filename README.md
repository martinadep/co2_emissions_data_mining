# CO2 Emissions and GDP Clustering Analysis

A comprehensive data analysis project applying unsupervised machine learning techniques (K-Means and DBSCAN) to analyze the relationship between CO2 emissions and GDP across different countries.

## Project Overview

This project performs clustering analysis on global CO2 emissions and GDP data to identify patterns and groupings among countries based on their environmental and economic indicators. The analysis includes multiple implementations of clustering algorithms, exploratory data analysis, and various visualizations.

## Getting Started

### Prerequisites

- Python 3.11+
- pip
- Virtual environment support

### Installation

1. **Clone the project directory:**
   ```bash
   git clone https://github.com/martinadep/co2_emissions_data_mining
   cd co2_emissions_data_mining
   ```

2. **Create and activate the virtual environment:**
   
   **Create the virtual environment:**
   ```bash
   python -m venv co2_emissions_env
   ```
   
   **Activate the virtual environment:**
   
   On **Windows PowerShell:**
   ```powershell
   .\co2_emissions_env\Scripts\Activate.ps1
   ```
   
   On **Linux/macOS:**
   ```bash
   source co2_emissions_env/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r setup/requirements.txt
   ```

### Key Dependencies

- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn, plotly
- **Machine Learning:** scikit-learn
- **Parallel Processing:** PySpark
- **Notebook Environment:** Jupyter

## Analysis Workflow

### Phase 0: Data Understanding
- **[0a_CO2_data_understanding.ipynb](notebooks/0a_CO2_data_understanding.ipynb)** - Explore CO2 emissions data
- **[0b_GDP_data_understanding.ipynb](notebooks/0b_GDP_data_understanding.ipynb)** - Explore GDP data

### Phase 1: Data Preparation & EDA
- **[1a_data_preparation.ipynb](notebooks/1a_data_preparation.ipynb)** - Merge and process GDP and CO2 datasets
- **[1b_EDA_heatmaps.ipynb](notebooks/1b_EDA_heatmaps.ipynb)** - Correlation and relationship heatmaps
- **[1c_EDA_scatterplots.ipynb](notebooks/1c_EDA_scatterplots.ipynb)** - Scatter plot visualizations

### Phase 2: Clustering Analysis
- **[2_K-Means.ipynb](notebooks/2_K-Means.ipynb)** - K-Means clustering implementation with elbow method
- **[3_DBSCAN.ipynb](notebooks/3_DBSCAN.ipynb)** - DBSCAN density-based clustering

## Clustering Implementations

### K-Means Clustering
Three different implementations are provided:

1. **Parallel (PySpark):** `py_scripts/k-means_parallel.py`
   - Optimized for large datasets
   - Distributed computing with Apache Spark
   
2. **Scikit-learn:** `py_scripts/k-means_scikit.py`
   - Standard library implementation
   - Easy to use and well-documented
   
3. **Sequential:** `py_scripts/k-means_sequential.py`
   - Serial implementation

### DBSCAN Clustering
- Density-based spatial clustering
- Identifies outliers and clusters of arbitrary shape
- Parameters: eps (neighborhood radius) and min_samples

## Data Sources

The project analyzes:
- **CO2 Emissions:** Country-level emissions data
- **GDP:** Gross Domestic Product by country
- **Time Series:** Historical data for trend analysis

## Key Features

- Comprehensive EDA with multiple visualization types
- Multiple clustering algorithm implementations
- Parallel processing support with PySpark
- Elbow method for optimal cluster selection
- Analysis of country groupings by economic and environmental metrics
- Top 40 countries analysis for focused insights


## Project Structure

```
co2_project/
├── data/
│   ├── raw/                    # Original unprocessed data
│   │   ├── co2_emissions/
│   │   ├── gdp/
│   │   └── population/
│   ├── processed/              # Cleaned and preprocessed data
│   │   ├── co2_emissions_continents.csv
│   │   ├── co2_emissions_countries.csv
│   │   └── gdp_countries.csv
│   └── final/                  # Merged datasets ready for analysis
│       ├── gdp_co2_emissions.csv
│       └── gdp_co2_emissions_top40_2024.csv
├── notebooks/                  # Jupyter notebooks for analysis
│   ├── 0a_CO2_data_understanding.ipynb
│   ├── 0b_GDP_data_understanding.ipynb
│   ├── 1a_data_preparation.ipynb
│   ├── 1b_EDA_heatmaps.ipynb
│   ├── 1c_EDA_scatterplots.ipynb
│   ├── 2_K-Means.ipynb
│   └── 3_DBSCAN.ipynb
├── py_scripts/                 
│   ├── k-means_parallel.py     # PySpark parallel implementation
│   ├── k-means_scikit.py       # Scikit-learn implementation
│   └── k-means_sequential.py   # Sequential implementation
├── plots/                      # Generated visualizations
│   ├── dbscan/
│   ├── heatmaps/
│   ├── kmeans/
│   ├── scatterplots/
│   └── single_eda/
├── setup/
│   ├── requirements.txt        # Project dependencies
│   └── setup.sh               # Setup script
└── co2_emissions_env/         # Python virtual environment
```

