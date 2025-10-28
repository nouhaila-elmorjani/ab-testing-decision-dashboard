
# A/B Testing Analysis Dashboard

A professional data science platform for statistical A/B test analysis, machine learning insights, and automated business reporting. This dashboard provides end-to-end experiment analysis with statistical rigor and actionable insights.

## Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ab-testing-decision-dashboard-vlscbmzzkb73fuxhknffkj.streamlit.app/)

## Overview

The A/B Testing Analysis Dashboard transforms raw experiment data into actionable business intelligence through automated statistical testing, machine learning predictions, and professional reporting. Designed for data scientists, product managers, and marketing analysts.

## Features

### Statistical Analysis
- Automated hypothesis testing with Z-tests
- Confidence interval calculations
- Statistical significance determination
- Business recommendations based on results

### Machine Learning
- Conversion probability prediction
- Feature importance analysis
- Random Forest and Logistic Regression models
- User-level conversion scoring

### Experiment Planning
- Sample size calculator
- Statistical power analysis
- Minimum detectable effect estimation
- Confidence level configuration

### Visualization & Reporting
- Interactive charts and graphs
- Automated PDF report generation
- Data export capabilities
- Professional stakeholder reports

## Installation

### Prerequisites
- Python 3.9+
- Conda package manager

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ab-testing-decision-dashboard.git
   cd ab-testing-decision-dashboard
   ```

2. **Create and activate conda environment**
   ```bash
   conda create -n ab-testing python=3.9
   conda activate ab-testing
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the dashboard**
   - Open browser to `http://localhost:8501`
   - Upload A/B test data or use sample data

## Usage

### Data Requirements
Upload CSV files with the following columns:
- `user_id`: Unique user identifier
- `test_group`: Experiment group (e.g., 'control', 'variant')
- `converted`: Binary conversion indicator (0/1)
- Optional: `total_ads`, `most_ads_day`, `most_ads_hour` for ML features

### Workflow
1. **Data Upload**: Upload your A/B test data through the interface
2. **Statistical Analysis**: View conversion rates, significance tests, and confidence intervals
3. **Machine Learning**: Train models to predict conversions and identify key factors
4. **Visualization**: Explore data through interactive charts
5. **Reporting**: Generate PDF reports for stakeholders

## Project Structure

```
ab-testing-decision-dashboard/
├── data/                           # Sample datasets
│   ├── marketing_AB.csv
│   └── marketing_AB_cleaned.csv
├── notebooks/                      # Analysis notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_statistical_analysis.ipynb
│   ├── 03_machine_learning.ipynb
│   └── 04_dashboard_development.ipynb
├── src/                           # Python modules
│   ├── data_loader.py
│   ├── statistical_tests.py
│   ├── power_analysis.py
│   ├── ml_model.py
│   ├── visualization.py
│   ├── pdf_reporter.py
│   └── __init__.py
├── app.py                         # Streamlit application
├── requirements.txt               # Dependencies
├── runtime.txt                    # Python version
└── README.md
```

## Technical Implementation

### Statistical Methods
- Z-test for proportion differences
- 95% confidence intervals using normal approximation
- Power analysis using Cohen's h effect size
- Sample size calculation for experiment planning

### Machine Learning
- Feature engineering and categorical encoding
- Random Forest for non-linear pattern detection
- Logistic Regression for interpretable models
- AUC-ROC evaluation and cross-validation

### Architecture
- Modular Python design with separated concerns
- Streamlit for interactive web interface
- Automated error handling and data validation
- Efficient data processing and caching

## Results

### Case Study: Marketing Campaign
- **Dataset**: 588,101 users across control and variant groups
- **Baseline Conversion**: 1.79%
- **Variant Conversion**: 2.55%
- **Statistical Significance**: p < 0.000001
- **Business Impact**: 43.1% relative improvement

### Model Performance
- **Random Forest AUC**: 0.8575
- **Logistic Regression AUC**: 0.7273
- **Key Insights**: Ad exposure and engagement level are primary conversion drivers

## Technologies Used

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Statistics**: Scipy, Statsmodels
- **Machine Learning**: Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Reporting**: FPDF

## Contributing

This project demonstrates professional data science practices. Contributions that enhance statistical methodologies, machine learning models, or user experience are welcome through standard fork and pull request workflows.


## Author

Developed as a professional data science portfolio project demonstrating end-to-end A/B testing analysis capabilities.

---

*Built with Python and Streamlit for professional data science applications*
```
