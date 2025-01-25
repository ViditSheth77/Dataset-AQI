import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import anderson, normaltest, ks_2samp
from prettytable import PrettyTable


# Read the CSV file
df = pd.read_csv('/Users/viditsheth/PycharmProjects/LAB#1/city_hour.csv')

# Preprocess the data
df['Datetime'] = pd.to_datetime(df['Datetime'])
df['Date'] = df['Datetime'].dt.date  # Create a 'Date' column
df.set_index('Datetime', inplace=True)

# Remove outliers
numeric_cols = df.select_dtypes(include='number').columns
cleaned_df_ts = df.copy()
cleaned_df_aqi = df.copy()

for col in numeric_cols:
    data_ts = cleaned_df_ts[col].dropna()  # Drop missing values for Time Series Analysis
    data_aqi = cleaned_df_aqi[col].dropna()  # Drop missing values for Air Quality Indices Analysis
    print(f"{col}: {data_ts.isna().sum()}")


    # Time Series Analysis: Remove outliers
    q1_ts = data_ts.quantile(0.25)
    q3_ts = data_ts.quantile(0.75)
    iqr_ts = q3_ts - q1_ts
    lower_bound_ts = q1_ts - 1.5 * iqr_ts
    upper_bound_ts = q3_ts + 1.5 * iqr_ts
    cleaned_df_ts = cleaned_df_ts[(cleaned_df_ts[col] >= lower_bound_ts) & (cleaned_df_ts[col] <= upper_bound_ts)]

    # Air Quality Indices Analysis: Remove outliers
    q1_aqi = data_aqi.quantile(0.25)
    q3_aqi = data_aqi.quantile(0.75)
    iqr_aqi = q3_aqi - q1_aqi
    lower_bound_aqi = q1_aqi - 1.5 * iqr_aqi
    upper_bound_aqi = q3_aqi + 1.5 * iqr_aqi
    cleaned_df_aqi = cleaned_df_aqi[(cleaned_df_aqi[col] >= lower_bound_aqi) & (cleaned_df_aqi[col] <= upper_bound_aqi)]
    print("First Quantile:", q1_aqi)
    print("Third Quantile:", q3_aqi)

# Define available options for air quality indices and cities
available_indices = list(numeric_cols)
available_cities_ts = cleaned_df_ts['City'].unique()
available_cities_aqi = cleaned_df_aqi['City'].unique()

available_plot_types = ['Line Plot', 'Histogram', 'Box Plot', 'Scatter Plot', 'Pie Chart', 'Area Plot']

# Drop non-numeric columns
numeric_df = df.select_dtypes(include='number')

#################################################
# Pretty Table for data Description.
#################################################
# Describe the data
description = df.describe()

# Extract column names
columns = description.columns

# Create a PrettyTable
table = PrettyTable()

# Add the first column
table.add_column('Index', ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])

# Add columns for each numeric column in the DataFrame
for col in columns:
    table.add_column(col, description[col])

# Print the PrettyTable
print(table)



#%%
################################################
## Box Plot: Before and After outlier removal
################################################

# Create box plot for original DataFrame
fig_original = go.Figure()
for col in numeric_cols:
    fig_original.add_trace(go.Box(y=df[col], name=col + ' (Original)'))
fig_original.update_layout(title="Box Plot: Before Outlier Removal", yaxis_title="Value")

# Create box plot for DataFrame after outlier removal
fig_cleaned = go.Figure()
for col in numeric_cols:
    fig_cleaned.add_trace(go.Box(y=cleaned_df_ts[col], name=col + ' (After)'))
fig_cleaned.update_layout(title="Box Plot: After Outlier Removal", yaxis_title="Value")

# Display the box plots
# fig_original.show()
# fig_cleaned.show()

################################################
## Box-Cox transformation AND SHAPIRO-WILK TESTING
################################################

transformed_data = stats.boxcox(cleaned_df_ts[numeric_cols], lmbda=0.5)
transformed_data_aqi = stats.boxcox(cleaned_df_aqi[numeric_cols], lmbda=0.5)

transformed_df = pd.DataFrame(transformed_data, columns=numeric_cols)
transformed_df_aqi = pd.DataFrame(transformed_data_aqi, columns=numeric_cols)

# Print the transformed data
print("Transformed Data for Time series Analysis:")
print(transformed_df)

print("Transformed Data AQI:")
print(transformed_df_aqi)

# Perform Shapiro-Wilk test for normality on transformed data
normality_test_results = {}
normality_test_results_aqi = {}

for col in numeric_cols:
    # Perform Shapiro-Wilk test
    stat, p_value = stats.shapiro(transformed_df[col])
    stat2, p_value2 = stats.shapiro(transformed_df_aqi[col])

    # Check if the p-value is less than the significance level (e.g., 0.05)
    if p_value < 0.05:
        normality_test_results[col] = {'Normality': False, 'p-value': p_value}
        normality_test_results_aqi[col] = {'Normality': False, 'p-value': p_value2}
    else:
        normality_test_results[col] = {'Normality': False, 'p-value': p_value}
        normality_test_results_aqi[col] = {'Normality': False, 'p-value': p_value2}


# Print the normality test results
print("Normality Test Results:")
for col, result in normality_test_results.items():
    if result['Normality']:
        print(f"{col}: Data is normally distributed (p-value={result['p-value']:.4f})")
    else:
        print(f"{col}: Data is not normally distributed (p-value={result['p-value']:.4f})")

print('------------------------------')
print("\nNormality Test Result of AQI:")
for col, result in normality_test_results_aqi.items():
    if result['Normality']:
        print(f"{col}: Data is normally distributed (p-value={result['p-value']:.4f})")
    else:
        print(f"{col}: Data is not normal distributed(p-value={result['p-value']:.4f})")

#%%
############################################
# Normality testing
############################################
# Initialize dictionaries to store normality test results
normality_test_results_anderson = {}
normality_test_results_dagostino = {}
normality_test_results_ks = {}

# Perform Anderson-Darling, D'Agostino and Pearson's, and Kolmogorov-Smirnov tests
for col in numeric_cols:
    # Anderson-Darling test
    result_anderson = anderson(transformed_df[col])
    normality_test_results_anderson[col] = {'Statistic': result_anderson.statistic, 'Critical Values': result_anderson.critical_values, 'Significance Level': result_anderson.significance_level}

    # D'Agostino and Pearson's test
    result_dagostino = normaltest(transformed_df[col])
    normality_test_results_dagostino[col] = {'Statistic': result_dagostino.statistic, 'p-value': result_dagostino.pvalue}

    # Kolmogorov-Smirnov test
    result_ks = ks_2samp(transformed_df[col], 'norm')
    normality_test_results_ks[col] = {'Statistic': result_ks.statistic, 'p-value': result_ks.pvalue}

# Create a PrettyTable for normality test results
normality_table = PrettyTable()

# Add column headers
normality_table.field_names = ["Index", "Shapiro-Wilk (p-value)", "Anderson-Darling (statistic)", "D'Agostino-Pearson (statistic, p-value)", "Kolmogorov-Smirnov (statistic, p-value)"]

# Add rows to the table
for col in numeric_cols:
    normality_table.add_row([col,
                   normality_test_results[col]['p-value'],
                   f"{normality_test_results_anderson[col]['Statistic']}, {normality_test_results_anderson[col]['Critical Values']}, {normality_test_results_anderson[col]['Significance Level']}",
                   f"{normality_test_results_dagostino[col]['Statistic']}, {normality_test_results_dagostino[col]['p-value']}",
                   f"{normality_test_results_ks[col]['Statistic']}, {normality_test_results_ks[col]['p-value']}"])

# Print the PrettyTable
print(normality_table)

#%%
##############################################
# PCA
##############################################
# Filter out non-numeric columns
numeric_cols = cleaned_df_ts.select_dtypes(include='number').columns

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cleaned_df_ts[numeric_cols])

# Apply PCA
pca = PCA()
pca.fit(scaled_data)
principal_components = pca.transform(scaled_data)

# Calculate cumulative explained variance ratio
cumulative_explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_) * 100

# Find the index where cumulative explained variance ratio crosses or equals 95%
index_95 = np.argmax(cumulative_explained_variance_ratio >= 95) + 1

# Visualize explained variance ratio
plt.figure(figsize=(10, 6))

# Plot explained variance ratio
# plt.plot(range(1, pca.n_components_ + 1), pca.explained_variance_ratio_, marker='o', linestyle='-', label='Explained Variance Ratio')

# Plot cumulative explained variance ratio
plt.plot(range(1, len(cumulative_explained_variance_ratio) + 1), cumulative_explained_variance_ratio, marker='o', linestyle='-', label='Cumulative Explained Variance Ratio')

# Plot the threshold line for 95% explained variance
plt.axhline(y=95, color='black', linestyle='--', label='95% Explained Variance')

# Plot the vertical line for the optimal number of components
plt.axvline(x=index_95, color='red', linestyle='--', label='Optimum Number of Components')

plt.xlabel('Number of Components')
plt.ylabel('Variance Ratio (%)')
plt.title('PCA Explained Variance Ratio')
plt.xticks(range(1, pca.n_components_ + 1))
plt.legend()
plt.grid(True)
plt.show()

# Visualize principal components in a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Principal Components Scatter Plot')
plt.show()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cleaned_df_ts[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Air Quality Indices')
plt.show()

# Calculate the Pearson correlation matrix
pearson_corr_matrix = np.corrcoef(principal_components, rowvar=False)

# Visualize the Pearson correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(pearson_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Pearson Correlation Matrix of Principal Components')
plt.xlabel('Principal Component Index')
plt.ylabel('Principal Component Index')
plt.show()

correlation_table = PrettyTable()

# Add column headers
correlation_table.add_column("Principal Component Index", range(1, pearson_corr_matrix.shape[1] + 1))

# Add correlation coefficients for each principal component pair
for i, row in enumerate(pearson_corr_matrix):
    correlation_table.add_column(f"PC{i+1}", row)

# Print the correlation table
print(correlation_table)

print("Principal Components:")
print(principal_components)
print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)
print("\nCumulative Explained Variance Ratio:")
print(cumulative_explained_variance_ratio)


#%%
# Select specific city and air quality indices
city = 'Amaravati'  # Example city
indices = ['PM2.5', 'NO', 'NH3', 'NOx', 'PM10', 'SO2', 'Xylene', 'Toluene', 'CO']  # Updated air quality indices

# Check if the city and indices are available in the dataset
if city in available_cities_ts and all(index in available_indices for index in indices):
    # Filter the DataFrame for the selected city
    city_df = cleaned_df_ts[cleaned_df_ts['City'] == city]

    # Create subplots
    fig = make_subplots(rows=3, cols=3, subplot_titles=indices)

    # Iterate over each air quality index
    for i, index in enumerate(indices, start=1):
        # Filter the DataFrame for the current index
        index_df = city_df[[index]].dropna()

        # Add histogram to the subplot for the current index
        fig.add_trace(go.Histogram(x=index_df[index], name="", marker_color="royalblue"), row=(i - 1) // 3 + 1, col=(i - 1) % 3 + 1)

        # Add bar plot to the subplot for the current index
        # fig.add_trace(go.Bar(x=index_df.index, y=index_df[index], name=index, marker_color="orange"), row=(i - 1) // 3 + 1, col=(i - 1) % 3 + 1)

    # Update layout
    fig.update_layout(height=800, title_text=f"Histogram Plots of {city}'s Air Quality Indices", showlegend=False)
    fig.show()

    # Bar plot
    fig_line = make_subplots(rows=3, cols=3, subplot_titles=indices)

    # Iterate over each air quality index
    for i, index in enumerate(indices, start=1):
        # Filter the DataFrame for the current index
        index_df = city_df[[index]].dropna()

        fig_line.add_trace(go.Line(x=index_df.index, y=index_df[index], name=index, marker_color="skyblue"), row=(i - 1) // 3 + 1, col=(i - 1) % 3 + 1)

    # Update layout
    fig_line.update_layout(height=800, title_text=f"Line Plot of {city}'s Air Quality Indices", showlegend=False)
    fig_line.show()

    # Pie Chart
    total_values = city_df[indices].sum()

    # Calculate percentage distribution for each index
    percentage_distribution = (total_values / total_values.sum()) * 100

    # Create labels and values for the pie chart
    pie_labels = percentage_distribution.index
    pie_values = percentage_distribution.values

    # Create the pie chart
    fig_pie = go.Figure(data=[go.Pie(labels=pie_labels, values=pie_values, hole=0.3)])

    fig_pie.update_layout(title_text=f"Percentage Distribution of Air Quality Indices in {city}")
    fig_pie.show()


    #DIST PLOT
    fig_dist, axes = plt.subplots(3, 3, figsize=(15, 15))

    # Flatten axes for easy iteration
    axes = axes.flatten()

    # Iterate over each index
    for i, index in enumerate(indices):
        # Filter the DataFrame for the current index
        index_df = city_df[[index]].dropna()

        # Create distribution plot using Seaborn
        sns.histplot(data=index_df, x=index, kde=True, ax=axes[i], color='green')
        axes[i].set_title(f"Distribution of {index} in {city}")
        axes[i].set_xlabel(index)
        axes[i].set_ylabel('Density')

    # Hide empty subplots if there are fewer than 9 indices
    if len(indices) < 9:
        for j in range(len(indices), 9):
            axes[j].axis('off')

    # Adjust layout
    plt.tight_layout()
    plt.title('Distribution of Air Quality Indices in Distribution Plot')
    plt.show(renderer="browser")


    # PAIR PLOT
    city_indices_df = city_df[indices]

    # Add the city column
    city_indices_df['City'] = city

    # Create pair plot using Seaborn
    sns.pairplot(data=city_indices_df, hue='City', palette='husl', diag_kind='kde')

    # Show the pair plot
    plt.title('Distribution of Air Quality Indices in Pair Plot')
    plt.show(rendere='browser')

    #Histogram plot with KDE
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    # Flatten axes for easy iteration
    axes = axes.flatten()

    # Iterate over each index
    for i, index in enumerate(indices):
        # Filter the DataFrame for the current index
        index_df = city_df[[index]].dropna()

        # Create histogram plot with KDE using Seaborn
        sns.histplot(data=index_df, x=index, kde=True, ax=axes[i], color='blue')

        # Set title and labels for each subplot
        axes[i].set_title(f"Distribution of {index} in {city}")
        axes[i].set_xlabel(index)
        axes[i].set_ylabel('Density')

    # Hide empty subplots if there are fewer than 9 indices
    if len(indices) < 9:
        for j in range(len(indices), 9):
            axes[j].axis('off')

    # Adjust layout
    plt.title("Histogram Plot with KDE")
    plt.tight_layout()

    # Show the plots
    plt.show()


    #QQ-Plot
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    # Flatten axes for easy iteration
    axes = axes.flatten()

    # Iterate over each index
    for i, index in enumerate(indices):
        # Filter the DataFrame for the current index
        index_df = city_df[[index]].dropna()

        # Create QQ-plot using scipy.stats.probplot
        stats.probplot(index_df[index], dist="norm", plot=axes[i])

        # Set title for each subplot
        axes[i].set_title(f"QQ-plot for {index} in {city}")

    # Adjust layout
    plt.tight_layout()
    plt.title('Distribution of Air Quality Indices in QQ-Plot')

    # Show the plots
    plt.show()

    #KDE PLOTS
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    # Flatten axes for easy iteration
    axes = axes.flatten()

    # Set color palette
    palette = sns.color_palette("husl", len(indices))

    # Iterate over each index
    for i, index in enumerate(indices):
        # Filter the DataFrame for the current index
        index_df = city_df[[index]].dropna()

        # Create KDE plot using Seaborn
        sns.kdeplot(data=index_df, x=index, fill=True, alpha=0.6, color=palette[i], linewidth=3, ax=axes[i])

        # Set title for each subplot
        axes[i].set_title(f"KDE Plot for {index} in {city}")

    # Adjust layout
    plt.tight_layout()
    plt.title('Distribution of Air Quality Indices in KDE Plots')

    # Show the plots
    plt.show()

    # Im or reg plot with scatter representation and regression line
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    # Flatten axes for easy iteration
    axes = axes.flatten()

    # Iterate over each index
    for i, index in enumerate(indices):
        # Filter the DataFrame for the current index
        index_df = city_df[[index]].dropna()

        # Create scatter plot with regression line using Seaborn
        sns.regplot(data=index_df, x=index, y=index, ax=axes[i], scatter=True, color='blue', line_kws={"color": "red"})

        # Set title for each subplot
        axes[i].set_title(f"Scatter Plot with Regression Line for {index} in {city}")

    # Adjust layout
    plt.tight_layout()
    plt.title('Distribution of Air Quality Indices in Regression line')

    # Show the plots
    plt.show()

    # Multivariate Box or Boxen plot
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    # Flatten axes for easy iteration
    axes = axes.flatten()

    # Iterate over each index
    for i, index in enumerate(indices):
        # Filter the DataFrame for the current index
        index_df = city_df[[index]].dropna()

        # Create multivariate box plot using Seaborn
        sns.boxplot(data=index_df, ax=axes[i], color='skyblue')

        # Set title for each subplot
        axes[i].set_title(f"Multivariate Box Plot for {index} in {city}")

    # Adjust layout
    plt.tight_layout()
    plt.title('Distribution of Air Quality Indices in Boxen Plot')

    # Show the plots
    plt.show()

    # Area plot
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    # Flatten axes for easy iteration
    axes = axes.flatten()

    # Iterate over each index
    for i, index in enumerate(indices):
        # Filter the DataFrame for the current index
        index_df = city_df[[index]].dropna()

        # Create area plot using Seaborn
        sns.lineplot(data=index_df, ax=axes[i], color='skyblue', linewidth=2, alpha=0.7)

        # Set title for each subplot
        axes[i].set_title(f"Area Plot for {index} in {city}")
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)

    # Adjust layout
    plt.tight_layout()
    plt.title('Distribution of Air Quality Indices in Area Plot')

    # Show the plots
    plt.show()

    #Violin plot
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    # Flatten axes for easy iteration
    axes = axes.flatten()

    # Iterate over each index
    for i, index in enumerate(indices):
        # Filter the DataFrame for the current index
        index_df = city_df[[index]].dropna()

        # Create violin plot using Seaborn
        sns.violinplot(data=index_df, ax=axes[i], color='skyblue')

        # Set title for each subplot
        axes[i].set_title(f"Violin Plot for {index} in {city}")

    # Adjust layout
    plt.tight_layout()
    plt.title('Distribution of Air Quality Indices in Violin Plot')

    # Show the plots
    plt.show()

    # # RUG PLOT
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    # Flatten axes for easy iteration
    axes = axes.flatten()

    # Iterate over each index
    for i, index in enumerate(indices):
        # Filter the DataFrame for the current index
        index_df = city_df[[index]].dropna()

        # Create rug plot using Seaborn
        sns.rugplot(data=index_df[index], height=0.5, ax=axes[i], color='purple')

        # Set title for each subplot
        axes[i].set_title(f"Rug Plot for {index} in {city}")

    # Adjust layout
    plt.tight_layout()
    plt.title('Distribution of Air Quality Indices in Rug Plot')

    # Show the plots
    plt.show()
#%%
    #Clustermap
    # Cluster Map
    cluster_grid = sns.clustermap(cleaned_df_ts[numeric_cols].corr(), cmap='coolwarm', annot=True, fmt=".2f",
                                  figsize=(10, 8))
    cluster_grid.ax_heatmap.set_title('Cluster Map of Air Quality Indices', loc='center', fontsize=16,
                                      fontweight='bold', color='black')
    plt.show()


    #Hexbin
    # Hexbin Plot
    plt.figure(figsize=(10, 8))
    hexbin_plot = plt.hexbin(cleaned_df_ts['CO'], cleaned_df_ts['NO2'], gridsize=30, cmap='coolwarm', edgecolors='none')
    plt.colorbar(hexbin_plot)
    plt.title('Hexbin Plot of CO vs NO2', fontsize=16, fontweight='bold', color='black')
    plt.xlabel('CO')
    plt.ylabel('NO2')
    plt.show()

    #Strip Plot
    plt.figure(figsize=(10, 6))
    sns.stripplot(x=cleaned_df_ts['City'], y=cleaned_df_ts['PM2.5'], jitter=True, palette='Set2')
    plt.title('Strip Plot of PM2.5 Levels Across Cities', fontsize=16, fontweight='bold', color='black', loc='center')
    plt.xlabel('City')
    plt.ylabel('PM2.5 Levels')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


else:
    print(f"{city} or one or more selected indices are not found in the dataset.")
