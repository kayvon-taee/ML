import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multivariate import test_mvmean
import scipy.stats

wine_data_all = pd.read_csv("winequality-all.csv")
# Identify and remove outliers for each numeric column except "colour"
numeric_columns = wine_data_all.select_dtypes(include=['number']).columns
outliers_removed_data = wine_data_all.copy()

for column in numeric_columns:
    if column != 'colour':
        # Calculate interquartile range
        q25, q75 = wine_data_all[column].quantile(0.25), wine_data_all[column].quantile(0.75)
        iqr = q75 - q25

        # Calculate the outlier cutoff
        cut_off = iqr * 2.5
        lower, upper = q25 - cut_off, q75 + cut_off

        # Identify outliers
        outliers = wine_data_all[(wine_data_all[column] < lower) | (wine_data_all[column] > upper)]

        # Remove outliers
        outliers_removed_data = outliers_removed_data[
            (outliers_removed_data[column] >= lower) & (outliers_removed_data[column] <= upper)]

        print(
            f"Column: {column}, Identified outliers: {len(outliers)}, Non-outlier observations: {len(outliers_removed_data)}")

pca = PCA()
scaler = StandardScaler()
pipeline = make_pipeline(scaler, pca)
fit = pipeline.fit(outliers_removed_data)
explained_variance = pca.explained_variance_ratio_

# Create the scree plot
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.grid(True)

# Add a vertical line at the point where the elbow is
num_components_desired = 3
plt.axvline(x=num_components_desired, color='r', linestyle='--', label=f'{num_components_desired} Components')
# Add legend for the red line and save plot
plt.legend()
plt.savefig("Scree_plot.png", dpi=300, bbox_inches='tight')

'''
A function to produce biplots for PCA components. Assumes data has already been fitted with a pipeline and is available as wine_data_all. Values for fit and pca are passed in because different scopes enforce the use of references and not modifying the original fit
'''


def biplot_for_pca(pipeline_fit: object, pca_fit: PCA, pca_number_for_x_axis: int, pca_number_for_y_axis: int,
                   plot_title_fig_1: str, figure_file_name: str, plot_title_fig_2) -> None:
    fig, ax = plt.subplots(figsize=(16, 6), ncols=2)
    ax[0].grid(True)
    ax[1].grid(True)

    for axis_index, pca_number in enumerate([pca_number_for_x_axis, pca_number_for_y_axis]):
        ax[axis_index].axis("equal")
        ax[axis_index].set_xlim(-1, 1)
        ax[axis_index].set_ylim(-1, 1)
        ax[axis_index].set_xlabel(f"PC{pca_number}")
        ax[axis_index].set_ylabel(f"PC{pca_number + 1}")
        if axis_index == 0:
            ax[axis_index].set_title(plot_title_fig_1, fontsize=16, y=1.1)
        else:
            ax[axis_index].set_title(plot_title_fig_2, fontsize=16, y=1.1)
        # Extract transformed values from fit for x and y axes
        pca_data_index_x = pca_number - 1
        pca_data_index_y = pca_number + 1 - 1
        pca_data_x = pipeline_fit.transform(outliers_removed_data)[:, pca_data_index_x]
        pca_data_y = pipeline_fit.transform(outliers_removed_data)[:, pca_data_index_y]
        coeff = np.transpose(pca_fit.components_[pca_data_index_x:pca_data_index_y + 1, :])
        n = wine_data_all.shape[1]
        scale_x_axis = 1.0 / (pca_data_x.max() - pca_data_x.min())
        scale_y_axis = 1.0 / (pca_data_y.max() - pca_data_y.min())

        # Used for differentiating the wine type
        sns.scatterplot(x=pca_data_x * scale_x_axis, y=pca_data_y * scale_y_axis, hue=outliers_removed_data['colour'],
                        alpha=0.8, ax=ax[axis_index])

        # Generate the legend

        legend = ax[axis_index].legend(title='Wine colour', loc='upper right')
        legend.get_texts()[0].set_text("White wine")
        legend.get_texts()[1].set_text("Red wine")

        for column_index in range(n):
            x_position = coeff[column_index, 0]
            y_position = coeff[column_index, 1]

            # Check if the label will overlap with others, and adjust position if needed
            if column_index < n - 1:
                for j in range(column_index + 1, n):
                    if (abs(x_position - coeff[j, 0]) < 0.1) and (abs(y_position - coeff[j, 1]) < 0.1):
                        x_position += 0.1
                        y_position += 0.1

            label = outliers_removed_data.T.index[column_index]

            # Find the last space character and replace it with a newline
            last_space_index = label.rfind(" ")
            if last_space_index >= 0:
                # Add an extra new line for total sulfur dioxide so labels don't overlap
                if pca_data_index_y == 4 and column_index == 6:
                    label = label[:last_space_index] + "\n\n" + label[last_space_index + 1:]
                label = label[:last_space_index] + "\n" + label[last_space_index + 1:]

            ax[axis_index].arrow(0, 0, x_position, y_position, color="r")
            # Add arrowheads to labels
            ax[axis_index].annotate(label, (x_position, y_position), color="black",
                                    arrowprops=dict(arrowstyle="->", lw=1.5),
                                    fontsize=8, ha='center', va='center')

        ax[axis_index].axhline(0, color='black', linewidth=0.5)  # Add horizontal line
        ax[axis_index].axvline(0, color='black', linewidth=0.5)  # Add vertical line

    fig.savefig(figure_file_name, dpi=300, bbox_inches='tight')


# PCA numbers are between 1 and 3, so we just need range 1 to 2, with 3 being 2 + 1.
explained_variance_percentage = ["{:.2%}".format(num) for num in pca.explained_variance_ratio_]
for pca_number in range(1, 2):
    # Create a new PCA object for each plot
    biplot_for_pca(fit, pca, pca_number, pca_number + 1,
                   f"Plot of PC{pca_number + 1} ({explained_variance_percentage[pca_number]} explained variance)\n against PC{pca_number} ({explained_variance_percentage[pca_number - 1]} explained variance)",
                   "biplots_combined.png",
                   f"Plot of PC{pca_number + 2} ({explained_variance_percentage[pca_number + 2]} explained variance)\n against PC{pca_number + 1} ({explained_variance_percentage[pca_number + 1]} explained variance)")


def biplot_for_pca_quality(pipeline_fit: object, pca_fit: PCA, pca_number_for_x_axis: int, pca_number_for_y_axis: int,
                           plot_title: str, figure_file_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # Create subplots with 1 row and 2 columns
    axes = axes.flatten()  # Flatten the 2D array of axes

    for ax, pca_number in zip(axes, range(1, 3)):
        ax.grid(True)
        ax.axis("equal")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel(f"PC{pca_number}")
        ax.set_ylabel(f"PC{pca_number + 1}")
        ax.set_title(f"Plot of PC{pca_number + 1} against PC{pca_number}", fontsize=16, y=1.1)

        # Create a new column based on conditions
        data_copy = deepcopy(outliers_removed_data)
        # Define conditions
        conditions = [
            (data_copy['quality'].ge(3) & data_copy['quality'].lt(6)),
            (data_copy['quality'].ge(6) & data_copy['quality'].le(9)),
        ]

        # Define corresponding categories
        categories = ['3-6', '6-9']

        # Use np.select to assign categories based on conditions
        data_copy['quality_category'] = np.select(conditions, categories, default='Unknown')
        data_copy['quality_categories_by_wine'] = np.where(
            (data_copy['colour'] == 0) & (data_copy['quality_category'] == '3-6'), '3-6 white',
            np.where(
                (data_copy['colour'] == 0) & (data_copy['quality_category'] == '6-9'), '6-9 white',
                np.where(
                    (data_copy['colour'] == 1) & (data_copy['quality_category'] == '3-6'), '3-6 red', '6-9 red'
                )
            )
        )

        # Drop unnecessary columns
        data_copy = data_copy.drop(columns=['quality_category'])

        # Define the features to be scaled (excluding 'quality_categories_by_wine')
        features_to_scale = [col for col in data_copy.columns if col != 'quality_categories_by_wine']

        # Create a ColumnTransformer that applies StandardScaler to specified features
        preprocessor = ColumnTransformer(
            transformers=[
                ('scaler', StandardScaler(), features_to_scale),
            ],
            remainder='passthrough'  # Pass through columns not specified in transformers
        )

        pca_new = PCA(n_components=3)

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('pca', pca_new)
        ])

        # Exclude 'quality_categories_by_wine' during fitting
        pipeline.fit(data_copy.drop(columns=['quality_categories_by_wine']))

        transformed_data = pipeline.transform(data_copy.drop(columns=['quality_categories_by_wine']))

        # Extract transformed values from fit for x and y axes
        pca_data_index_x = pca_number - 1
        pca_data_index_y = pca_number + 1 - 1
        pca_data_x = transformed_data[:, pca_data_index_x]
        pca_data_y = transformed_data[:, pca_data_index_y]
        # Extract the components from the pipeline
        components = pipeline.named_steps['pca'].components_

        if pca_data_index_y >= components.shape[0]:  # Check if index is within bounds
            continue

        coeff = np.transpose(components[pca_data_index_x:pca_data_index_y + 1, :])
        n = data_copy.shape[1]
        scale_x_axis = 1.0 / (pca_data_x.max() - pca_data_x.min())
        scale_y_axis = 1.0 / (pca_data_y.max() - pca_data_y.min())
        # Your existing scatter plot code
        scatter = sns.scatterplot(x=pca_data_x * scale_x_axis, y=pca_data_y * scale_y_axis,
                                  hue=data_copy['quality_categories_by_wine'], alpha=0.8, ax=ax)

        legend_labels = [
            '3-6 (White)',
            '3-6 (Red)',
            '6-9 (White)',
            '6-9 (Red)'
        ]

        ax.legend(title='Quality Categories', loc='upper right')

        for column_index in range(min(n, len(coeff))):
            x_position = coeff[column_index, 0]
            y_position = coeff[column_index, 1]

            # Check if the label will overlap with others, and adjust position if needed
            if column_index < n - 1:
                for j in range(column_index + 1, min(n, len(coeff))):
                    if (abs(x_position - coeff[j, 0]) < 0.1) and (abs(y_position - coeff[j, 1]) < 0.1):
                        x_position += 0.1
                        y_position += 0.1

            label = data_copy.T.index[column_index]

            # Find the last space character and replace it with a newline
            last_space_index = label.rfind(" ")
            if last_space_index >= 0:
                # Add an extra new line for total sulfur dioxide so labels don't overlap
                if pca_data_index_y == 4 and column_index == 6:
                    label = label[:last_space_index] + "\n\n" + label[last_space_index + 1:]
                label = label[:last_space_index] + "\n" + label[last_space_index + 1:]

            ax.arrow(0, 0, x_position, y_position, color="r")
            # Add arrowheads to labels
            ax.annotate(label, (x_position, y_position), color="black",
                        arrowprops=dict(arrowstyle="->", lw=1.5),
                        fontsize=8, ha='center', va='center')

        ax.axhline(0, color='black', linewidth=0.5)  # Add horizontal line
        ax.axvline(0, color='black', linewidth=0.5)  # Add vertical line

    fig.savefig(figure_file_name, dpi=300, bbox_inches='tight')


# Outside the function
for pca_number in range(1, 2):
    # Create a new PCA object for each plot
    pca_new = PCA(n_components=3)

    # Call the function with the specific PCA object
    biplot_for_pca_quality(fit, pca_new, pca_number, pca_number + 1,
                           f"Plot of PC{pca_number + 1} against PC{pca_number}",
                           f"wine_quality_pc{pca_number + 1}_vs_pc{pca_number}_with_quality")

############################################################
# Statistical test part
############################################################

red_wine_acidity = wine_data_all[["volatile acidity", "fixed acidity", "pH"]].loc[wine_data_all["colour"] == 1]
white_wine_acidity = wine_data_all[["volatile acidity", "fixed acidity", "pH"]].loc[wine_data_all["colour"] == 0]
formula = 'Q("volatile acidity") + Q("fixed acidity") + Q("pH") ~ colour'
manova_model = MANOVA.from_formula(formula, data=outliers_removed_data)
print(manova_model.mv_test(skip_intercept_test=True))

# Boxplot for post-manova analysis
# Specify the order of variables
variables = ['volatile acidity', 'fixed acidity', 'pH']

# Create a 1x3 grid of box plots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))

# Iterate through each variable and create a box plot
for i, variable in enumerate(variables):
    ax = axes[i]
    sns.boxplot(x='colour', y=variable, data=outliers_removed_data, ax=ax, hue="colour")
    ax.set_title(f'Distribution of {variable} by wine colour')
    ax.set_xlabel('Wine Type')

    # Exclude units for pH
    if variable.lower() != 'ph':
        units = {
            'fixed acidity': 'g(tartaric acid)/dm³',
            'volatile acidity': 'g(acetic acid)/dm³',
        }
        ax.set_ylabel(f'{variable.capitalize()} ({units[variable]})')
    else:
        ax.set_ylabel(variable)
    legend = ax.legend(loc="upper left")
    legend.get_texts()[0].set_text("White wine")
    legend.get_texts()[1].set_text("Red wine")
    # Customize x-axis labels
    ax.set_xticklabels(['White Wine', 'Red Wine'])

    # Show grid lines
    ax.grid(True, linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()

# Save the figure
fig.savefig("box_plots_for_hotelling_two_sample.png", dpi=300, bbox_inches='tight')

# Function to calculate confidence interval for 1 or 2 sample hotelling t test
from typing import Union
from scipy.stats import f

'''
A function that calculates CIs for hotelling t-test for a given sig level. For two populations, supply the dataframe to group_2.
'''


def hotelling_t_test_ci(group_1: pd.DataFrame, group_2=Union[pd.DataFrame, None], sig_level=0.05) -> dict:
    # Calculate group sizes
    group_1_n = group_1.shape[0]
    # Can use either group 1 or 2 as they have equal number of variables
    number_of_dependent_variables = group_1.shape[1]
    # Calculate means for each variable for each group
    group_1_means = group_1.mean()
    # Degrees of freedom for F statistic
    df1 = number_of_dependent_variables
    df2 = group_1_n - df1
    group_1_cov = group_1.cov()
    if group_2 is not None:
        group_2_n = group_2.shape[0]
        group_2_means = group_2.mean()
        total_n = group_1_n + group_2_n
        df2 = total_n - number_of_dependent_variables - 1
        covariance_matrix = ((group_1_n - 1) * group_1_cov + (group_2_n - 1) * group_2.cov()) / (
                    group_1_n + group_2_n - 2)
        group_mean_diff = group_1_means - group_2_means
        f_critical = f.ppf(1 - sig_level, df1, df2)
        # K = sqrt((np-p)/(n(n-p)) * F_crit_df1_df2_0.95 )
        k_value = np.sqrt((total_n * df1 - df1) / total_n * (total_n - df1) * f_critical)
        std_err = np.sqrt(np.diag(covariance_matrix)) * k_value
        # Construct the dictionary
        result = {col: (mean, std_err[i]) for i, (col, mean) in enumerate(group_mean_diff.items())}

        return result
    f_critical = f.ppf(1 - sig_level, df1, df2)
    k_value = np.sqrt((group_1_n * df1 - df1) / group_1_n * (group_1_n - df1) * f_critical)
    std_err = np.sqrt(np.diag(group_1.cov())) * k_value
    # Construct the dictionary
    result = {col: (mean, std_err[i]) for i, (col, mean) in enumerate(group_1_means.items())}

    return result


# Print the means and standard errors for each variable. CI can be calculated by taking first element
# then +/- for 95% CI with second element

print(hotelling_t_test_ci(white_wine_acidity, red_wine_acidity))

# Sulfate content interesting due to fermentation - Does red have different levels of fermentation to white?
white_wine_sulfates = outliers_removed_data.loc[outliers_removed_data["colour"] == 0]
red_wine_sulfates = outliers_removed_data.loc[outliers_removed_data["colour"] == 1][
    ["free sulfur dioxide", "total sulfur dioxide", "sulphates"]]
wine_wine_sulfates_means = white_wine_sulfates[["free sulfur dioxide", "total sulfur dioxide", "sulphates"]].mean()
one_sample_hotelling_test = test_mvmean(red_wine_sulfates, mean_null=wine_wine_sulfates_means)
print(one_sample_hotelling_test)
# One sample hotelling t test CI
print(hotelling_t_test_ci(group_1=red_wine_sulfates, group_2=None))

# Post-hoc 1-sample hotelling analysis, boxplots
variables = ["free sulfur dioxide", "total sulfur dioxide", "sulphates"]

# Create a 1x3 grid of box plots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))  # Adjusted figsize for a wider plot

# Iterate through each variable and create a box plot
for i, variable in enumerate(variables):
    ax = axes[i]
    sns.boxplot(x=i, y=variable, data=outliers_removed_data, ax=ax)
    ax.set_title(f'Distribution of red wine {variable} ')

    # Add units to y-axis label
    units = {
        'free sulfur dioxide': 'mg/dm³',
        'total sulfur dioxide': 'mg/dm³',
        'sulphates': 'g(potassium sulphate)/dm³'
    }
    ax.set_ylabel(f'{variable.capitalize()} ({units[variable]})')

    # Customize x-axis labels
    ax.set_xticklabels("")

    # Add text annotation for white wine mean
    ax.grid(True, linestyle='--', alpha=0.7)
    white_wine_mean = wine_wine_sulfates_means[i]
    box_props = dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='black')
    ax.annotate(f'White Wine Mean\n{variable.capitalize()}: {white_wine_mean:.2f} {units[variable]}', xy=(0.95, 0.95),
                xycoords='axes fraction', ha='right', va='top', bbox=box_props)
    ax.annotate("", xy=(0.9, 0.9), xytext=(1, 1), xycoords='axes fraction', textcoords='axes fraction')

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()

# Save the figure
fig.savefig("box_plots_for_hotelling_one_sample.png", dpi=300, bbox_inches='tight')

##################################################################################
# One EDA plot produced but may not be used in final report
# Aimed to investigate different levels of citric acid between and within groups
# Citric acid interesting because it adds freshness - is freshness associated with quality?
##################################################################################

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Subplot without hue
sns.barplot(x='quality', y='citric acid', data=outliers_removed_data, ax=axes[0], capsize=0.1)
axes[0].set_title('Effect of citric acid on wine quality - Whole dataset')
axes[0].set_xlabel("Quality")
axes[0].set_ylabel("Citric Acid (g/dm$^3$)")
# Subplot with hue
sns.barplot(x='quality', y='citric acid', data=outliers_removed_data, hue='colour', ax=axes[1], capsize=0.2)
axes[1].set_title('Effect of citric acid on wine quality by wine colour')
axes[1].set_xlabel("Quality")
axes[1].set_ylabel("Citric Acid (g/dm$^3$)")

# Customizing legend labels
legend = axes[1].legend()
legend.get_texts()[0].set_text("White wine")
legend.get_texts()[1].set_text("Red wine")

plt.tight_layout()
plt.show()

# Save the figure
fig.savefig("whole_data_by_colour_citric_acid_vs_quality.png", dpi=300, bbox_inches='tight')

# As a fun bonus, I also wrote my own version of a two-sample hotelling t test to improve my python skills
# I assume a large sample size for both populations and I also consider normality checks with box-m test

import scipy.stats

p_value_rejection_dictionary = {
    0.1: "No evidence against H0",
    0.05: "Weak evidence against H0",
    0.01: "Some evidence against H0",
    0.001: "Strong evidence against H0",
    0.0001: "Very strong evidence against H0"
}
"""
An implementation for the box m test. Returns the p-value which can be used in assumption checking.
"""


def box_m_test(group_1: pd.DataFrame, group_2: pd.DataFrame) -> float:
    if group_1.shape[1] != group_2.shape[1]:
        raise ValueError(f"The number of dependent variables in group_1 ({group_1.shape[1]}) "
                         f"and group_2 ({group_2.shape[1]}) must be equal.")
    # Calculate group sizes
    group_1_n = group_1.shape[0]
    group_2_n = group_2.shape[0]
    # Can use either group 1 or 2 as they have equal number of variables - i.e. k value
    number_of_dependent_variables = group_1.shape[1]
    # Get the covariance matrix for both groups
    group_1_cov = group_1.cov()
    group_2_cov = group_2.cov()
    total_n = group_1_n + group_2_n
    # Calculated the pooled covariance matrix
    pooled_covariance_matrix = ((group_1_n - 1) * group_1_cov + (group_2_n - 1) * group_2_cov) / (
                group_1_n + group_2_n - 2)
    # 2 groups so m (lower case) is 2
    box_m_statistic = (total_n - 2) * np.log(np.linalg.det(pooled_covariance_matrix)) - (
                group_1_n * np.log(np.linalg.det(group_1_cov)) + group_2_n * np.log(np.linalg.det(group_2_cov)))
    # 2k^2 + 3k - 1 / 6(k+1)(m-1)
    critical_region_factor = (2 * number_of_dependent_variables ** 2 + 3 * number_of_dependent_variables - 1) / (
                6 * (number_of_dependent_variables + 1) * (2 - 1))
    # sum_{j=1}^m (where m = 2) 1/(n_j-1) - 1/(n-m), n being total_n
    critical_region_sum = 1 / (group_1_n - 1) - 1 / (total_n - 2) + 1 / (group_2_n - 1)
    # (2k^2 + 3k - 1 / 6(k+1)(m-1))*sum_{j=1}^m (where m = 2) 1/(n_j-1) - 1/(n-m), n being total_n
    critical_region = critical_region_factor * critical_region_sum
    # Calculate chi-square, df and p-value
    chi_square_statistic = box_m_statistic * (1 - critical_region)
    df = 0.5 * number_of_dependent_variables * (number_of_dependent_variables + 1) * (2 - 1)
    p_value = scipy.stats.chi2.sf(chi_square_statistic, df)
    return p_value


'''
A function that takes in data from two groups and calculates the hotelling T-statistic for two groups for a given significance level. Checks for unequal covariances with box m test. Assumes it to be two sided and the data has been parsed in the correct format. Returns a string with the necessary information. Assumes large n for both groups. If significant, will return the confidence intervals

'''
def two_sided_hotelling_test(group_1: pd.DataFrame, group_2: pd.DataFrame, sig_level=0.05) -> str:
    if group_1.shape[1] != group_2.shape[1]:
        raise ValueError(f"The number of dependent variables in group_1 ({group_1.shape[1]}) "
                         f"and group_2 ({group_2.shape[1]}) must be equal.")
    # Calculate group sizes
    group_1_n = group_1.shape[0]
    group_2_n = group_2.shape[0]
    # Can use either group 1 or 2 as they have equal number of variables
    number_of_dependent_variables = group_1.shape[1]
    # Calculate means for each variable for each group
    group_1_means = group_1.mean()
    group_2_means = group_2.mean()
    # Degrees of freedom for F statistic
    df1 = number_of_dependent_variables
    df2 = group_1_n + group_2_n - number_of_dependent_variables - 1
    # Find the difference between group 1 and 2 means
    vector_diff = group_1_means - group_2_means
    # Get the covariance matrix for both groups
    group_1_cov = group_1.cov()
    group_2_cov = group_2.cov()
    # Critical chi square
    critical_chi_square = scipy.stats.chi2.ppf(sig_level, number_of_dependent_variables)
    # Check if we can assume equal covariances - if true used pooled
    box_m_test_result = box_m_test(group_1, group_2)
    if box_m_test_result <= 0.05:
        # (X-Y)^T * (V_x/n_x + V_y/n_y)^-1*(X-Y)
        hotelling_t_statistic = vector_diff @ np.linalg.inv(
            group_1_cov / group_1_n + group_2_cov / group_2_n) @ vector_diff.T
        # Assume n is sufficiently large. Assume equal number of dependent variables
        p_value = scipy.stats.chi2.sf(hotelling_t_statistic, number_of_dependent_variables)
        p_value_conclusion = "No evidence against H0"
        ci_dictionary = None
        if p_value < sig_level:
            ci_dictionary = hotelling_t_test_ci(group_1=group_1, group_2=group_2, sig_level=sig_level)
        for p_value_statement in p_value_rejection_dictionary.keys():
            if p_value < p_value_statement:
                p_value_conclusion = p_value_rejection_dictionary[p_value_statement]
        return f"""
        Hotelling T2 statistic: {hotelling_t_statistic}
        p_value: {p_value}
        Conclusion: {p_value_conclusion}
        Method: Unequal covariance approach
        Means with standard errors (None if not significant): {ci_dictionary}
        """
        # Result was false, used alternative strategy
    # Covariance for groups - used pooled if unbalanced
    covariance_matrix = None
    # Balanced group case
    if group_1_n == group_2_n:
        covariance_matrix = 0.5 * (group_1.cov() + group_2.cov())
    else:
        covariance_matrix = ((group_1_n - 1) * group_1.cov() + (group_2_n - 1) * group_2.cov()) / (
                    group_1_n + group_2_n - 2)
    # Find mahalanobis distance and hotelling T statistic
    mahalanobis_distance_squared = vector_diff @ np.linalg.inv(covariance_matrix) @ vector_diff.T
    hotelling_t_statistic = (group_1_n * group_2_n) / (group_1_n + group_2_n) * mahalanobis_distance_squared
    # Assume n is sufficiently large. Assume equal number of dependent variables
    p_value = scipy.stats.chi2.sf(hotelling_t_statistic, number_of_dependent_variables)
    p_value_conclusion = "No evidence against H0"
    ci_dictionary = None
    if p_value < sig_level:
        ci_dictionary = hotelling_t_test_ci(group_1=group_1, group_2=group_2, sig_level=sig_level)
    for p_value_statement in p_value_rejection_dictionary.keys():
        if p_value < p_value_statement:
            p_value_conclusion = p_value_rejection_dictionary[p_value_statement]
    return f"""
    Hotelling T2 statistic: {hotelling_t_statistic}
    p_value: {p_value}
    Conclusion: {p_value_conclusion}
    Method: Equal covariance approach
    Means with standard errors (None if not significant): {ci_dictionary}
    """

# Here's a version of 2 sample hotelling but with an F-statistic. I've confirmed it works :)
'''
A function that takes in data from two groups and calculates the hotelling T-statistic for two groups for a given significance level. Assumes it to be two sided and the data has been parsed in the correct format. Returns a string with the necessary information.
'''
def two_sided_hotelling_test_f_statistic_version(group_1: pd.DataFrame, group_2: pd.DataFrame) -> str:
    if group_1.shape[1] != group_2.shape[1]:
        raise ValueError(f"The number of dependent variables in group_1 ({group_1.shape[1]}) "
                         f"and group_2 ({group_2.shape[1]}) must be equal.")

    # Calculate group sizes
    group_1_n = group_1.shape[0]
    group_2_n = group_2.shape[0]

    # Can use either group 1 or 2 as they have an equal number of variables
    number_of_dependent_variables = group_1.shape[1]

    # Calculate means for each variable for each group
    group_1_means = group_1.mean()
    group_2_means = group_2.mean()

    # Degrees of freedom for F statistic
    df1 = number_of_dependent_variables
    df2 = group_1_n + group_2_n - number_of_dependent_variables - 1

    # Covariance for groups - used pooled if unbalanced
    covariance_matrix = None

    # Balanced group case
    if group_1_n == group_2_n:
        covariance_matrix = 0.5 * (group_1.cov() + group_2.cov())
    else:
        covariance_matrix = ((group_1_n - 1) * group_1.cov() + (group_2_n - 1) * group_2.cov()) / (group_1_n + group_2_n - 2)

    # Find the difference between group 1 and 2 means
    vector_diff = group_1_means - group_2_means

    # Find mahalanobis distance and hotelling T statistic
    mahalanobis_distance_squared = vector_diff @ np.linalg.inv(covariance_matrix) @ vector_diff.T
    hotelling_t_statistic = (group_1_n * group_2_n) / (group_1_n + group_2_n) * mahalanobis_distance_squared

    # Get F statistic and p-value. Assumes an equal number of dependent variables
    f_statistic = (group_1_n + group_2_n - number_of_dependent_variables - 1) / (number_of_dependent_variables * (group_1_n + group_2_n - 2)) * hotelling_t_statistic
    p_value = 1 - scipy.stats.f.cdf(f_statistic, df1, df2)

    p_value_conclusion = "No evidence against H0"

    for p_value_statement in p_value_rejection_dictionary.keys():
        if p_value < p_value_statement:
            p_value_conclusion = p_value_rejection_dictionary[p_value_statement]

    return f"""
    Hotelling T2 statistic: {hotelling_t_statistic}
    F-statistic: {f_statistic}
    p_value: {p_value}
    Conclusion: {p_value_conclusion}
    """

# If you are reading this and are interested in reaching out to me with some implementations in python for statistical
# tests so you can use it in your lecture notes, please
# say so in your feedback, and I'll reach out to you after feedback has been released!