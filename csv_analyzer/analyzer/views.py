from django.conf import settings
from django.shortcuts import render
from .models import UploadedCSV
from .data_analysis import basic_analysis, visualization, ml_tasks
import os
import pandas as pd

def index(request):
    context = {}
    file_path = request.session.get('csv_path')
    updated_file_path = request.session.get('updated_csv_path')

    if request.method == 'POST':
        # Upload new CSV
        if request.FILES.get('csv_file'):
            name = request.POST.get('name')
            email = request.POST.get('email')
            phone = request.POST.get('phone')
            csv_file = request.FILES['csv_file']

            if not csv_file.name.endswith('.csv'):
                context['error'] = "Please upload a valid CSV file."
                return render(request, 'index.html', context)

            uploaded = UploadedCSV(name=name, email=email, phone=phone, file=csv_file)
            uploaded.save()
            file_path = uploaded.file.path
            request.session['csv_path'] = file_path

        elif not file_path or not os.path.exists(file_path):
            context['error'] = "No CSV file found. Please upload again."
            return render(request, 'index.html', context)
        
        # Decide which CSV to load (updated or original)
        if updated_file_path and os.path.exists(updated_file_path):
            df = basic_analysis.read_csv(updated_file_path)
            context['using_updated_file'] = True
        else:
            df = basic_analysis.read_csv(file_path)

        
        # Use updated file path if null values are detected
        # Handle remove nulls button
        if request.method == 'POST' and 'remove_nulls' in request.POST:
            if file_path and os.path.exists(file_path):
                original_df = basic_analysis.read_csv(file_path)
                cleaned_df = basic_analysis.remove_nulls(original_df)

                base_name = os.path.basename(file_path)  # original.csv
                name_only, ext = os.path.splitext(base_name)  # ('original', '.csv')
                updated_filename = name_only + '-updated' + ext  # 'original-updated.csv'

                updated_path = os.path.join(settings.MEDIA_ROOT, updated_filename)
                cleaned_df.to_csv(updated_path, index=False)

                request.session['updated_csv_path'] = updated_path
                context['updated_csv_ready'] = True
                context['download_link'] = settings.MEDIA_URL + updated_filename

                # Continue analysis on cleaned data
                df = cleaned_df
            else:
                context['error'] = "Original file not found."
                return render(request, 'index.html', context)


        # Load CSV and perform basic analysis

        all_columns = df.columns.tolist()
        numeric_columns = df.select_dtypes(include='number').columns.tolist()

        context['columns'] = all_columns
        context['numeric_columns'] = numeric_columns
        context['shape'] = f"{df.shape[0]} rows × {df.shape[1]} columns"

        # nulls = basic_analysis.check_nulls(df).to_dict()
        nulls_series = basic_analysis.check_nulls(df)
        nulls = nulls_series.to_dict()

        dtypes = basic_analysis.get_dtypes(df).to_dict()

        # this is for updated csv file
        total_nulls = nulls_series.sum()
        if total_nulls > 0:
            context['show_remove_nulls'] = True
        if updated_file_path and os.path.exists(updated_file_path):
            context['updated_csv_ready'] = True
            context['download_link'] = settings.MEDIA_URL + os.path.basename(updated_file_path)



        combined_info = {}
        for col in df.columns:
            combined_info[col] = {
                'nulls': nulls.get(col, 0),
                'dtype': dtypes.get(col, 'Unknown')
            }
        context['column_info'] = combined_info

        context['duplicates'] = basic_analysis.check_duplicates(df)                                                                     
        context['head'] = df.head().to_html()

        # Always show heatmap
        context['heatmap'] = visualization.heatmap(df)

        # Only perform advanced analysis if columns are selected
        col1 = request.POST.get('col1')
        col2 = request.POST.get('col2')

        if col1 and col1 in df.columns:
            is_col1_numeric = pd.api.types.is_numeric_dtype(df[col1])
            if is_col1_numeric:
                context['histogram'] = visualization.histogram(df, col1)
                context['boxplot'] = visualization.box_plot(df, col1)
                context['bar_chart'] = visualization.bar_chart(df, col1)

            if col2 and col2 in df.columns:
                is_col2_numeric = pd.api.types.is_numeric_dtype(df[col2])
                if is_col1_numeric and is_col2_numeric:
                    context['scatter'] = visualization.scatter_plot(df, col1, col2)
                    # Parallel Coordinates Plot
                    try:
                        context['parallel_plot'] = visualization.parallel_coordinates_plot(df, col2)
                    except Exception as e:
                        context['parallel_plot'] = None

                    try:
                        coef, intercept = ml_tasks.linear_regression(df, [col1], col2)
                        context['regression'] = f"Regression between {col1} and {col2}: Coef={coef[0]:.4f}, Intercept={intercept:.4f}"
                    except Exception as e:
                        context['regression'] = f"Regression error: {str(e)}"
                else:
                    context['regression'] = "Both columns must be numeric for regression."

                # Decision Tree
                try:
                    acc = ml_tasks.decision_tree_classification(df, [col1], col2)
                    context['classification'] = f"Decision Tree Accuracy: {acc:.2%}"
                except ValueError as e:
                    context['classification'] = f"Classification skipped: {str(e)}"
                except Exception as e:
                    context['classification'] = f"Classification error: {str(e)}"

            # KMeans
            if is_col1_numeric:
                try:
                    labels = ml_tasks.kmeans_clustering(df, [col1])
                    context['clustering'] = f"KMeans clustering labels: {labels.tolist()[:10]}..."
                except Exception as e:
                    context['clustering'] = f"KMeans error: {str(e)}"

                # Outliers
                try:
                    outliers = ml_tasks.detect_outliers(df, [col1])
                    outlier_count = list(outliers).count(-1)
                    context['outliers'] = f"Outliers detected in {col1}: {outlier_count}"
                except Exception as e:
                    context['outliers'] = f"Outlier detection error: {str(e)}"

            if col1 and col2 and col1 in df.columns and col2 in df.columns:
                # For comparison, both must be numeric
                if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                    try:
                        lin_metrics = ml_tasks.linear_regression_metrics(df, col1, col2)
                        multi_metrics = ml_tasks.multiple_linear_regression_metrics(df, col2)
                        poly_metrics = ml_tasks.polynomial_regression_metrics(df, col2)

                        comparison = {
                            'linear': lin_metrics,
                            'multiple': multi_metrics,
                            'polynomial': poly_metrics
                        }

                        # Find best model by highest R²
                        best_model = max(comparison, key=lambda k: comparison[k]['r2'])

                        context['regression_comparison'] = {
                            'results': comparison,
                            'best_model': best_model.capitalize()
                        }

                    except Exception as e:
                        context['regression_comparison_error'] = str(e)


    return render(request, 'index.html', context)


from django.shortcuts import redirect
import glob

def reset_view(request):
    # Delete uploaded CSV file

    csv_path = request.session.get('csv_path')
    if csv_path and os.path.exists(csv_path):
        os.remove(csv_path)

    for plot_file in glob.glob("analyzer/static/*.png"):
        os.remove(plot_file)

    updated_file_path = request.session.get('updated_csv_path')
    if updated_file_path and os.path.exists(updated_file_path):
        os.remove(updated_file_path)

    # Clear session
    request.session.flush()

    return redirect('/')

