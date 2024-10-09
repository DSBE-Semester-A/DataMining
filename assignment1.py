# %%
# Import all the libraries, numpy and pandas for processing, and seaborn and matplotlib for visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
from sklearn.model_selection import train_test_split 
import sklearn.metrics as metrics
import math
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

# Ignore FutureWarnings
warnings.filterwarnings('ignore', category=FutureWarning)

# %%
# Load the dataframe in
df = pd.read_csv('Ass1Data/California_Houses.csv')
df.head()

# %%
# First, let's see a summary for all the columns
df.describe()

# %%
# And let's check with how many rows and columns we are working with -> 20.640 rows and 14 columns
df.shape

# %%
# Then, let's check the data summary, check for null-values, and check the data-types
df.info()

# %%
# We can double-check the null-values using
df.isnull().sum()

# %% [markdown]
# As can be seen, there are 20.640 entries of houses, and none of the columns contain null-values. That means that most of the data is quite clean already, as we don't need to process missing values or null values.

# %%
# Now, let's check the duplicate values for each column
df.nunique()

# %%
# Now let's check the correlation matrix for the dataset
corr_all_variables = df.corr()
corr_all_variables.style.format(precision=2).background_gradient(cmap='coolwarm')

# %% [markdown]
# Since our target variable is 'median_house_value', and our interest is in whether the proximity to an urban center has influence on it, we'll focus further now on the variables: 'Median_House_Value' and 'Distance_to_coast', 'Distance_to_LA', 'Distance_to_SanDiego', 'Distance_to_SanJose', 'Distance_to_SanFrancisco'

# %%
# Let's see the correlation matrix for only the distance columns again
distance_columns = df[['Median_House_Value', 'Distance_to_coast', 'Distance_to_LA', 'Distance_to_SanDiego', 'Distance_to_SanJose', 'Distance_to_SanFrancisco']]
corr_distances = distance_columns.corr()

# Plot the heatmap and save it as an image
plt.figure(figsize=(10, 8))
sns.heatmap(corr_distances, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix for Distance Columns')
plt.savefig('corr_distances.png')
plt.show()


# %%
# Plot histograms for all distance columns in a matrix of subplots
fig, axes = plt.subplots(3, 2, figsize=(10, 10))
fig.suptitle('Distribution of Median House Value and Distance to Cities and Coast', fontsize=16)

# Plot each column
for ax, column in zip(axes.flatten(), distance_columns):
    sns.histplot(data=distance_columns, x=column, bins=30, kde=True, ax=ax)
    ax.set_title(f'Distribution of {column}')
    ax.set_ylabel('Number of houses')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# %% [markdown]
# From the density plot of the Median House Value, there is a skewness to the left. However, there is a noticeable increase on the highest end of Median_House_Value, so that needs to be expected more closely!

# %%
# Display the table for the histogram of 'Median_House_Value'
median_house_value_counts = df['Median_House_Value'].value_counts().sort_index()
print(median_house_value_counts)

# %% [markdown]
# From the table, it can be seen that there are 965 entries of houses with a median house value of 500001.0. A reason for this could be that this is the limit of median income, and therefore the cut-off value for median_house_value.

# %% [markdown]
# ## Modelling

# %% [markdown]
# ### Splitting the data in test and training data

# %%
#  Select the variables that we want as outcome (y) amd prediciton (x) variables
X = df[['Distance_to_LA', 'Distance_to_SanDiego', 'Distance_to_SanJose', 'Distance_to_SanFrancisco', 'Distance_to_coast']]
y = df[['Median_House_Value']]

# Split the data with 75% train and 25% test.
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=6666)

# %% [markdown]
# ### Linear Regression Trial

# %% Attempt on linear 
mod_poly = LinearRegression()
mod_poly.fit(x_train, y_train)

y_lin_pred = mod_poly.predict(x_test)
y_lin_pred

# %%
# Calculate the residuals
residuals = y_test.values.flatten() - y_lin_pred.flatten()

# Q-Q plot of residuals
plt.figure(figsize=(8, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals for linear regression')
plt.show()


# %%
# Define the predictor columns
predictors = ['Distance_to_LA', 'Distance_to_SanDiego', 'Distance_to_SanJose', 'Distance_to_SanFrancisco', 'Distance_to_coast']

# Set up the figure size
plt.figure(figsize=(15, 10))

# Create a scatter plot for each attribute
for i, predictor in enumerate(predictors, 1):
    plt.subplot(3, 2, i)  # Create a 3x2 grid of plots
    sns.scatterplot(x=x_train[predictor], y=y_train['Median_House_Value'])
    plt.title(f'{predictor} vs Median_House_Value')
    plt.xlabel(predictor)
    plt.ylabel('Median House Value')
    
    # Adding trend line
    sns.regplot(x=x_train[predictor], y=y_train['Median_House_Value'], scatter=False, color='red')

plt.tight_layout()
plt.show()


# %% [markdown]
# ### Decision Tree Regression

# %%

# Create the decision tree regressor
decision_tree_model = DecisionTreeRegressor(random_state=6666)

# Train the model
decision_tree_model.fit(x_train, y_train)

# Make predictions
y_dec_pred = decision_tree_model.predict(x_test)

# %%
MSE_dec = metrics.mean_squared_error(y_test, y_dec_pred)
RMSE_dec = math.sqrt(MSE_dec)
MAE_dec = metrics.mean_absolute_error(y_test, y_dec_pred)
R2_dec = metrics.r2_score(y_test, y_dec_pred)
Med_abs_er_dec = metrics.median_absolute_error(y_test, y_dec_pred)

print(MSE_dec, RMSE_dec, MAE_dec, R2_dec, Med_abs_er_dec)

# %%
sns.histplot(y_test, label='Actual Values')
sns.histplot(y_dec_pred, label='Predicted Values')
plt.legend()


# %% [markdown]
# ### K neighbors Regressor

# %% Find the optimal value for n using gridsearch

# We choose 1 to 21 for a potential value of n
param_grid = {'n_neighbors': range(1, 21)}

# Initialize regressor
knn = KNeighborsRegressor()

# Use GridSearchCV to search for the best value of n_neighbors
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error')

# Fit the model to the training data
grid_search.fit(x_train, y_train)

# Get the best parameters
best_n_neighbors = grid_search.best_params_['n_neighbors']
print(best_n_neighbors)


# %%
# Create model with optimal value for n
knn_model = KNeighborsRegressor(n_neighbors=6)

# Fit the model on the training data
knn_model.fit(x_train, y_train)

# Predict using the test data
y_knn_pred = knn_model.predict(x_test)


# %%
sns.histplot(y_test, label='Actual Values')
sns.histplot(y_knn_pred, label='Predicted Values')
plt.legend()

# %%
MSE_knn = metrics.mean_squared_error(y_test, y_knn_pred)
RMSE_knn = math.sqrt(MSE_knn)
MAE_knn = metrics.mean_absolute_error(y_test, y_knn_pred)
R2_knn = metrics.r2_score(y_test, y_knn_pred)
Med_abs_er_knn = metrics.median_absolute_error(y_test, y_knn_pred)

print(MSE_knn, RMSE_knn, MAE_knn, R2_knn, Med_abs_er_knn)

# %% [markdown]
# ## Evaluation

# %%
# Calculate the knn bias
bias_knn = np.mean((y_test - y_knn_pred) ** 2)
# Calculate the knn variance
variance_knn = np.var(y_knn_pred)


# %%
# Calculate the total variance of the target variable
total_variance = np.var(y_test)

# %%
# Calculate the variance explained by the linear regression model
explained_variance_knn = R2_knn * total_variance

# Calculate the irreducible error for the linear regression model
irreducible_error_knn = total_variance - explained_variance_knn
irreducible_error_knn

# %%
# Calculate the decision tree bias
bias_dec = np.mean((y_test.values.flatten() - y_dec_pred) ** 2)
# Calculate the decision tree variance
variance_dec = np.var(y_dec_pred)


# %%
# Calculate the variance explained by the decision tree model
explained_variance_dec = R2_dec * total_variance

# Calculate the irreducible error for the decision tree model
irreducible_error_dec = total_variance - explained_variance_dec
irreducible_error_dec

# %%
# Create a dictionary with the results
results = {
    'Model': ['KNN', 'Decision Tree'],
    'MSE': [MSE_knn, MSE_dec],
    'RMSE': [RMSE_knn, RMSE_dec],
    'MAE': [MAE_knn, MAE_dec],
    'R2': [R2_knn, R2_dec],
    'Med_abs_er': [Med_abs_er_knn, Med_abs_er_dec],
    'Bias2': [bias_knn, bias_dec],
    'Variance': [variance_knn, variance_dec],
    'Irreducible Error': [irreducible_error_knn.values[0], irreducible_error_dec.values[0]]
}

# %%
# Create the dataframe
results_df = pd.DataFrame(results)

# Display the dataframe
results_df.set_index('Model').T

# %%
results_melted = results_df.melt(id_vars='Model', var_name='Metric', value_name='Value')

# Split the metrics into two categories: positive and negative
large_metrics = results_melted[results_melted['Value'] >= 1e+05]
medium_metrics = results_melted[(results_melted['Value'] >= 1e+04) & (results_melted['Value'] < 1e+05)]
small_metrics = results_melted[results_melted['Value'] < 1e+04]


# %% Plot the large evaluation metrics
sns.barplot(data=large_metrics, x='Model', y='Value', hue='Metric')
plt.title('Comparison of Large Metrics for KNN and Decision Tree Models')
plt.xticks(rotation=45)
plt.legend(loc='upper center')

plt.show()


# %% Plot the medium evaluation metrics
sns.barplot(data=medium_metrics, x='Model', y='Value', hue='Metric')
plt.title('Comparison of Medium Metrics for KNN and Decision Tree Models')
plt.xticks(rotation=45)
plt.legend(loc=0)

plt.show()


# %% Plot the small evaluation metrics
sns.barplot(data=small_metrics, x='Model', y='Value', hue='Metric')
plt.title('Comparison of Small Metrics for KNN and Decision Tree Models')
plt.xticks(rotation=45)
plt.legend(loc='upper left')

plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Plot for KNN
sns.histplot(y_test, label='Actual Values', ax=axes[0], color='blue')
sns.histplot(y_knn_pred.flatten(), label='Predicted Values', ax=axes[0], color='orange')
axes[0].set_title('KNN regression on median house prices on proximity to a big city / urban center', size=10)
axes[0].legend()

# Plot for Decision Tree
sns.histplot(y_test, label='Actual Values', ax=axes[1], color='blue')
sns.histplot(y_dec_pred, label='Predicted Values', ax=axes[1], color='orange')
axes[1].set_title('Decision Tree regression on median house prices on proximity to a big city / urban center', size=10)
axes[1].legend()

plt.tight_layout()
plt.show()


