# CSV Data Analyzer

A Django web application for analyzing CSV files with automated data cleaning, visualization, and machine learning capabilities.

## Features

- **File Upload**: Upload CSV files with user information
- **Data Cleaning**: Remove null values and download cleaned CSV
- **Basic Analysis**: View data shape, duplicates, null values, and data types
- **Visualizations**: 
  - Heatmap (correlation matrix)
  - Histogram
  - Box plot
  - Bar chart
  - Scatter plot
  - Parallel coordinates plot
- **Machine Learning**:
  - Linear Regression
  - Multiple Linear Regression
  - Polynomial Regression
  - Decision Tree Classification
  - KMeans Clustering
  - Outlier Detection

## Requirements

- Python 3.7+
- Django
- pandas
- matplotlib
- scikit-learn
- seaborn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/csv-analyzer.git
cd csv-analyzer
```

2. Install dependencies:
```bash
pip install django pandas matplotlib scikit-learn seaborn
```

3. Run migrations:
```bash
python manage.py makemigrations
python manage.py migrate
```

4. Start the server:
```bash
python manage.py runserver
```

5. Open browser: `http://localhost:8000`

## How to Use

### Upload CSV
1. Enter your name, email, and phone
2. Choose a CSV file
3. Click "Upload and Analyze"

### View Analysis
- **Basic Info**: See rows, columns, data types, and null values
- **Remove Nulls**: Click button to clean data and download updated CSV
- **Select Columns**: Choose Col 1 and Col 2 for advanced analysis

### Visualizations
- Automatic heatmap generation
- Select numeric columns for histograms, box plots, and scatter plots

### Machine Learning
- **Regression**: Compare Linear, Multiple, and Polynomial models
- **Classification**: Decision tree accuracy
- **Clustering**: KMeans grouping
- **Outliers**: Detect anomalies in data

### Reset
Click "Reset" to clear session and uploaded files

## Project Structure

```
├── analyzer/
│   ├── models.py          # UploadedCSV model
│   ├── views.py           # Main logic
│   ├── data_analysis.py   # Analysis functions
│   ├── templates/
│   │   └── index.html     # Main template
│   └── static/            # Generated plots
├── media/                 # Uploaded CSV files
└── manage.py
```

## Key Functions

- `basic_analysis.read_csv()` - Load CSV file
- `basic_analysis.remove_nulls()` - Clean null values
- `basic_analysis.check_nulls()` - Count null values
- `visualization.heatmap()` - Generate correlation heatmap
- `ml_tasks.linear_regression()` - Perform linear regression
- `ml_tasks.kmeans_clustering()` - Cluster data

## Configuration

- Uploaded files stored in: `MEDIA_ROOT`
- Generated plots saved in: `analyzer/static/`
- Session-based file tracking

## Notes

- Only CSV files are accepted
- Numeric columns required for ML tasks
- Original and cleaned files stored separately
- All plots saved as PNG images

## License

MIT License

---

**Tip**: Use numeric columns for best results with visualizations and ML models!
