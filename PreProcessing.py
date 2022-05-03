##############################################
# Import Libraries #
#############################################

# Standard
import pandas as pd
import numpy as np
import numpy.linalg as LA
from scipy import stats
from scipy.stats import boxcox

# Visualization
from matplotlib import pyplot as plt
import seaborn as sns
import pylab

# scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#############################################################
# ------------------ Reading Dataset ------------------ #
############################################################

df = pd.read_csv("train.csv")

print("Original Dataset: \n", df.head())
print("Dataset Statistics: \n", df.describe())

df_2 = df.copy()

#############################################################
# ------------------ Removing id columns ------------------ #
############################################################
print("Columns in original dataset: \n", df_2.columns)

df_2.drop(['id', 'Unnamed: 0'], axis=1, inplace=True)

print("Columns in modified dataset: \n", df_2.columns)

#############################################################
# ------------------ Removing NaN values ------------------ #
############################################################

print("Columns with NA values - Before removal:")
for i in df_2.columns:
    if df_2[i].isna().sum() > 0:
        print(df_2[i].name, ":", df_2[i].isna().sum())
    else:
        continue
print("\n")

df_2 = df_2.dropna(how="any")

print("Columns with NA values - After removal:")
for i in df_2.columns:
    if df_2[i].isna().sum() > 0:
        print(df_2[i].name, ":", df_2[i].isna().sum())
    else:
        continue
print("\n")

# df_2['Departure Delay in Hours'] = df_2['Departure Delay in Minutes'] / 60
# df_2['Arrival Delay in Hours'] = df_2['Arrival Delay in Minutes'] / 60

# df_2.drop(['Departure Delay in Minutes', 'Arrival Delay in Minutes'], axis=1, inplace=True)
#####################################################################
# ------------------ Outlier Detection - Box Plot ------------------ #
####################################################################

col = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
fig = plt.figure(figsize=(12, 8))
for i in range(1, 5):
    ax1 = fig.add_subplot(2, 2, i)
    ax1.boxplot(df_2[col[i - 1]].values)
    ax1.set_title(f"Outlier {col[i - 1]}", fontsize=15)
    ax1.set_xlabel(f"{col[i - 1]}", fontsize=15)
    ax1.set_ylabel("Average", fontsize=15)
    ax1.set_font = 20
    plt.grid(b=True, axis='y')

plt.tight_layout()
plt.show()

#####################################################################
# ------------------ Outlier Detection - Removal ------------------ #
####################################################################

df_outlier = df_2.copy()

for x in ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']:
    iqr = (df_2[x].quantile(0.75)) - (df_2[x].quantile(0.25))

    upper_limit = (df_2[x].quantile(0.75)) + (1.5 * iqr)
    lower_limit = (df_2[x].quantile(0.25)) - (1.5 * iqr)

    df_2[x] = np.where(df_2[x] > upper_limit, upper_limit, np.where(df_2[x] < lower_limit, lower_limit, df_2[x]))

#####################################################################
# ------------------ Dimensionality Reduction ------------------ #
####################################################################

X = df_2.loc[:, ~df_2.columns.isin(['satisfaction'])]
Y = df_2['satisfaction']


# Handling categorical variables

def mapping(xx):
    dict = {}
    count = -1
    for x in xx:
        dict[x] = count + 1
        count = count + 1
    return dict


# for i in ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'Departure Delay', 'Arrival Delay']:
for i in ['Gender', 'Customer Type', 'Type of Travel', 'Class']:
    unique_tag = X[i].value_counts().keys().values
    dict_mapping = mapping(unique_tag)
    X[i] = X[i].map(lambda x: dict_mapping[x] if x in dict_mapping.keys() else -1)

# The StandardScaler
ss = StandardScaler()

# Standardize the training data
X = ss.fit_transform(X)

H = np.matmul(X.T, X)

s, d, v = np.linalg.svd(H, full_matrices=True)

print(f"Singular value of dataframe are {d}")
print(f"Condition number for dataframe is {LA.cond(X)}")

pca = PCA(n_components='mle', svd_solver='full')
# pca = PCA(0.95)
principalComponents = pca.fit_transform(X)

print("Explained Variance Ratio - Original v/s Reduced Feature Space: \n", pca.explained_variance_ratio_)

print("#" * 100)

x = np.arange(1, len(np.cumsum(pca.explained_variance_ratio_)) + 1, 1)
plt.figure()
plt.plot(x, np.cumsum(pca.explained_variance_ratio_))
plt.xticks(x)
plt.title("Cumulative Explained Variance v/s Number of Components")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.grid()
plt.tight_layout()
plt.show()
print("#" * 100)

#####################################################################
# ------------------ Correlation Matrix ------------------ #
####################################################################

df_corr = df_2.copy()

for i in ['satisfaction']:
    unique_tag = df_corr[i].value_counts().keys().values
    dict_mapping = mapping(unique_tag)
    df_corr[i] = df_corr[i].map(lambda x: dict_mapping[x] if x in dict_mapping.keys() else -1)

plt.figure(figsize=(12, 8))
sns.heatmap(df_corr.corr())
plt.title("Heatmap - Correlation")
plt.tight_layout()
plt.show()

# plt.figure(figsize=(22, 19))
# pd.plotting.scatter_matrix(df_corr, diagonal="kde", hist_kwds={'bins': 20}, alpha=0.5)
# plt.title("Scatter Matrix - Correlation")
# plt.tight_layout()
# plt.show()

#####################################################################
# ------------------ Statistics ------------------ #
####################################################################

num_col = df_2._get_numeric_data().columns

describe_num_df = df_2.describe(include=['int64', 'float64'])
describe_num_df.reset_index(inplace=True)
describe_num_df = describe_num_df[describe_num_df['index'] != 'count']

#####################################################################
# ------------------ Normality Test ------------------ #
####################################################################

norm_col = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']

fig = plt.figure(figsize=(12, 8))
for i in range(1, 5):
    ax1 = fig.add_subplot(2, 2, i)
    ax1.hist(df_2[norm_col[i - 1]], bins=100)
    ax1.set_title(f"Distribution {norm_col[i - 1]}", fontsize=15)
    ax1.set_xlabel(f"{norm_col[i - 1]}", fontsize=15)
    ax1.set_ylabel("Values", fontsize=15)
    ax1.set_font = 20
    plt.grid(b=True, axis='y')

plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(12, 8))
for i in range(1, 5):
    ax1 = fig.add_subplot(2, 2, i)
    stats.probplot(df_2[norm_col[i - 1]], sparams=(2, 3), plot=plt, fit=False)
    ax1.set_title(f"Distribution {norm_col[i - 1]}", fontsize=15)
    plt.grid(b=True, axis='y')

plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(12, 8))

for i in range(len(norm_col)):

    print(f"Shapiro-Wilk Test Result - {norm_col[i]}: \n")

    print(
        f"Shapiro Test: statistics={stats.shapiro(df_2[norm_col[i]])[0]}; p-value={stats.shapiro(df_2[norm_col[i]])[1]}")
    if stats.shapiro(df_2[norm_col[i]])[1] <= 0.05:
        print(f"Shapiro Test: {norm_col[i]} is not normal distributed")
    else:
        print(f"Shapiro Test: {norm_col[i]} is normally distributed")

    y_r = stats.rankdata(df[norm_col[i]])

    tgt = stats.norm.ppf(y_r / (len(df[norm_col[i]]) + 1))

    tgt = np.log(tgt)

    print(
        f"Shapiro Test: statistics={stats.shapiro(tgt)[0]}; p-value={stats.shapiro(tgt)[1]}")
    if stats.shapiro(tgt)[1] <= 0.05:
        print(f"Shapiro Test: Transformed {norm_col[i]} is not normal distributed")
    else:
        print(f"Shapiro Test: Transformed {norm_col[i]} is normally distributed")

    # ax1 = fig.add_subplot(2, 2, i+1)
    # ax1.hist(df_2[norm_col[i]], bins=100)
    # ax1.set_title(f"Distribution {norm_col[i]}", fontsize=15)
    # ax1.set_xlabel(f"{norm_col[i]}", fontsize=15)
    # ax1.set_ylabel("Values", fontsize=15)
    # ax1.set_font = 20
    # plt.grid(b=True, axis='y')

    print("\n")

plt.show()

df_2['Departure Delay'] = np.where(df_2['Departure Delay in Minutes'] > 5, "Yes", "No")
df_2['Arrival Delay'] = np.where(df_2['Arrival Delay in Minutes'] > 5, "Yes", "No")
df_2['Age_Cat'] = np.where(df_2['Age'] <= 2, "Child"
                           , np.where(df_2['Age'] <= 19, "Teenager"
                                      , np.where(df_2['Age'] <= 60, "Adult"
                                                 , np.where(df_2['Age'] > 60, 'Senior Citizen'
                                                            , "Adult"))))
