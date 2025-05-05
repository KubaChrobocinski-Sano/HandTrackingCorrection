#%%
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

#%%
def load_error_dataframe(path):
    measurements = pd.read_csv(path, sep = ';', header=0, index_col=False)
    measurements = measurements.dropna(axis=0)

    measurements['Reference Index X'] = 0.0302
    measurements['Reference Index Y'] = 0.1095
    measurements['Reference Wrist X'] = 0.004
    measurements['Reference Wrist Y'] = -0.048
    measurements['Reference Thumb X'] = 0.04629
    measurements['Reference Thumb Y'] = 0.0556

    measurements['Reference Index Position Relative to Leap X'] = measurements['Reference Index X'] - measurements['Leap Controller X']
    measurements['Reference Index Position Relative to Leap Y'] = measurements['Reference Index Y'] - measurements['Leap Controller Y']
    measurements['Reference Index Position Relative to Leap Z'] = measurements['Reference Index Z'] - measurements['Leap Controller Z']

    measurements['Reference Wrist Position Relative to Leap X'] = measurements['Reference Wrist X'] - measurements['Leap Controller X']
    measurements['Reference Wrist Position Relative to Leap Y'] = measurements['Reference Wrist Y'] - measurements['Leap Controller Y']
    measurements['Reference Wrist Position Relative to Leap Z'] = measurements['Reference Wrist Z'] - measurements['Leap Controller Z']

    measurements['Reference Thumb Position Relative to Leap X'] = measurements['Reference Thumb X'] - measurements['Leap Controller X']
    measurements['Reference Thumb Position Relative to Leap Y'] = measurements['Reference Thumb Y'] - measurements['Leap Controller Y']
    measurements['Reference Thumb Position Relative to Leap Z'] = measurements['Reference Thumb Z'] - measurements['Leap Controller Z']

    measurements['Measured Index Position Relative to Leap X'] = measurements['Measured Index X'] - measurements['Leap Controller X']
    measurements['Measured Index Position Relative to Leap Y'] = measurements['Measured Index Y'] - measurements['Leap Controller Y']
    measurements['Measured Index Position Relative to Leap Z'] = measurements['Measured Index Z'] - measurements['Leap Controller Z']

    measurements['Measured Wrist Position Relative to Leap X'] = measurements['Measured Wrist X'] - measurements['Leap Controller X']
    measurements['Measured Wrist Position Relative to Leap Y'] = measurements['Measured Wrist Y'] - measurements['Leap Controller Y']
    measurements['Measured Wrist Position Relative to Leap Z'] = measurements['Measured Wrist Z'] - measurements['Leap Controller Z']

    measurements['Measured Thumb Position Relative to Leap X'] = measurements['Measured Thumb X'] - measurements['Leap Controller X']
    measurements['Measured Thumb Position Relative to Leap Y'] = measurements['Measured Thumb Y'] - measurements['Leap Controller Y']
    measurements['Measured Thumb Position Relative to Leap Z'] = measurements['Measured Thumb Z'] - measurements['Leap Controller Z']
    print("here")
    ground_truth = pd.DataFrame()
    measured_positions = pd.DataFrame()
    print("here")
    ground_truth['Reference Index X'] = measurements['Reference Index Position Relative to Leap X']
    ground_truth['Reference Index Y'] = measurements['Reference Index Position Relative to Leap Y']
    ground_truth['Reference Index Z'] = measurements['Reference Index Position Relative to Leap Z']
    ground_truth['Reference Thumb X'] = measurements['Reference Thumb Position Relative to Leap X']
    ground_truth['Reference Thumb Y'] = measurements['Reference Thumb Position Relative to Leap Y']
    ground_truth['Reference Thumb Z'] = measurements['Reference Thumb Position Relative to Leap Z']
    ground_truth['Reference Wrist X'] = measurements['Reference Wrist Position Relative to Leap X']
    ground_truth['Reference Wrist Y'] = measurements['Reference Wrist Position Relative to Leap Y']
    ground_truth['Reference Wrist Z'] = measurements['Reference Wrist Position Relative to Leap Z']
    print("here")
    measured_positions['Measured Index X'] = measurements['Measured Index Position Relative to Leap X']
    measured_positions['Measured Index Y'] = measurements['Measured Index Position Relative to Leap Y']
    measured_positions['Measured Index Z'] = measurements['Measured Index Position Relative to Leap Z']
    measured_positions['Measured Thumb X'] = measurements['Measured Thumb Position Relative to Leap X']
    measured_positions['Measured Thumb Y'] = measurements['Measured Thumb Position Relative to Leap Y']
    measured_positions['Measured Thumb Z'] = measurements['Measured Thumb Position Relative to Leap Z']
    measured_positions['Measured Wrist X'] = measurements['Measured Wrist Position Relative to Leap X']
    measured_positions['Measured Wrist Y'] = measurements['Measured Wrist Position Relative to Leap Y']
    measured_positions['Measured Wrist Z'] = measurements['Measured Wrist Position Relative to Leap Z']
    print("here")

    return ground_truth, measured_positions

def fit_multivariate_polynomial_correction(measured, ground_truth, degree):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(measured)
    models = []

    # Fit one model per dimension (X, Y, Z)
    for i in range(ground_truth.shape[1]):
        model = LinearRegression()
        model.fit(X_poly, ground_truth[:, i])
        models.append(model)
    
    return poly, models

def apply_multivariate_correction(measured, poly, models):
    X_poly = poly.transform(measured)
    corrected = np.zeros_like(measured)
    for i, model in enumerate(models):
        corrected[:, i] = model.predict(X_poly)
    return corrected

def evaluate_model(ground_truth, pred):
    metrics = []
    for i in range(ground_truth.shape[1]):
        rmse = np.sqrt(mean_squared_error(ground_truth[:, i], pred[:, i]))
        r2 = r2_score(ground_truth[:, i], pred[:, i])
        metrics.append({'rmse': rmse, 'r2': r2})
    return metrics

def plot_results(measured, ground_truth, corrected, pre_metrics, post_metrics, labels=['X', 'Y', 'Z']):
    for i, dim_name in enumerate(labels):
        plt.figure(figsize=(6, 5))
        plt.scatter(measured[:, i], ground_truth[:, i], label="Original Measured", alpha=0.2)
        plt.scatter(corrected[:, i], ground_truth[:, i], label="Corrected", alpha=0.2)
        plt.xlabel(f"{dim_name} (Predicted)")
        plt.ylabel(f"{dim_name} (Ground Truth)")
        plt.legend()
        plt.title(
            f"{dim_name} Correction\n"
            f"Before RMSE: {pre_metrics[i]['rmse']:.4f}, After RMSE: {post_metrics[i]['rmse']:.4f}\n"
            f"Before R²: {pre_metrics[i]['r2']:.4f}, After R²: {post_metrics[i]['r2']:.4f}"
        )
        plt.grid(True)
        plt.show()

def print_coefficients(poly, models, labels=['X', 'Y', 'Z']):
    feature_names = poly.get_feature_names_out(['X', 'Y', 'Z'])
    for i, dim_name in enumerate(labels):
        print(f"\n{dim_name} correction model coefficients:")
        for coef, name in zip(models[i].coef_, feature_names):
            print(f"{name}: {coef:.4f}")
        print(f"Intercept: {models[i].intercept_:.4f}")

#%%
# Loading real data

path = R"C:\Users\Sheff\OneDrive\Desktop\data\data_high.txt"

ground_truth, measured_positions = load_error_dataframe(path)

ground_truth_Index = ground_truth.iloc[:, 0:3].to_numpy()
ground_truth_Wrist = ground_truth.iloc[:, 3:6].to_numpy()
ground_truth_Thumb = ground_truth.iloc[:, 6:9].to_numpy()

measured_positions_Index = measured_positions.iloc[:, 0:3].to_numpy()
measured_positions_Wrist = measured_positions.iloc[:, 3:6].to_numpy()
measured_positions_Thumb = measured_positions.iloc[:, 6:9].to_numpy()
#%%
# Before Correction
pre_correction_metrics = evaluate_model(ground_truth_Index, measured_positions_Index)
#%%
# Fit & Correct
degree = 5  # You can tweak degree here
poly, models = fit_multivariate_polynomial_correction(measured_positions_Index, ground_truth_Index, degree)
corrected = apply_multivariate_correction(measured_positions_Index, poly, models)

# After Correction
post_correction_metrics = evaluate_model(ground_truth_Index, corrected)
#%%
# Visualize
plot_results(measured_positions_Index, ground_truth_Index, corrected, pre_correction_metrics, post_correction_metrics)

#%%
# Coefficients & Metrics
print_coefficients(poly, models)

# %%
#Doing the same but with 80/20 train/test split

measured_train_Index, measured_test_Index, truth_train_Index, truth_test_Index = train_test_split(
    measured_positions_Index, ground_truth_Index, test_size=0.2, random_state=None, shuffle=True)
# %%
pre_correction_metrics = evaluate_model(truth_test_Index, measured_test_Index)
#%%
# Fit & Correct

degrees = [2, 3, 4, 5, 6, 7]
post_correction_metrics = {}


for degree in degrees:
    poly, models = fit_multivariate_polynomial_correction(measured_train_Index, truth_train_Index, degree)
    corrected = apply_multivariate_correction(measured_test_Index, poly, models)
    # After Correction
    post_correction_metrics[degree] = evaluate_model(truth_test_Index, corrected)
    # Visualize
    #plot_results(measured_test_Index, truth_test_Index, corrected, pre_correction_metrics, post_correction_metrics)
    # Coefficients & Metrics
    

# %%
