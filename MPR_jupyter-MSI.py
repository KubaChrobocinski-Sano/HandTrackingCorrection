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
    measurements['Reference Index Position Relative to Leap Z'] = measurements['Reference Index Z'] #- measurements['Leap Controller Z']

    measurements['Reference Wrist Position Relative to Leap X'] = measurements['Reference Wrist X'] - measurements['Leap Controller X']
    measurements['Reference Wrist Position Relative to Leap Y'] = measurements['Reference Wrist Y'] - measurements['Leap Controller Y']
    measurements['Reference Wrist Position Relative to Leap Z'] = measurements['Reference Wrist Z'] #- measurements['Leap Controller Z']

    measurements['Reference Thumb Position Relative to Leap X'] = measurements['Reference Thumb X'] - measurements['Leap Controller X']
    measurements['Reference Thumb Position Relative to Leap Y'] = measurements['Reference Thumb Y'] - measurements['Leap Controller Y']
    measurements['Reference Thumb Position Relative to Leap Z'] = measurements['Reference Thumb Z'] #- measurements['Leap Controller Z']

    measurements['Measured Index Position Relative to Leap X'] = measurements['Measured Index X'] - measurements['Leap Controller X']
    measurements['Measured Index Position Relative to Leap Y'] = measurements['Measured Index Y'] - measurements['Leap Controller Y']
    measurements['Measured Index Position Relative to Leap Z'] = measurements['Measured Index Z'] #- measurements['Leap Controller Z']

    measurements['Measured Wrist Position Relative to Leap X'] = measurements['Measured Wrist X'] - measurements['Leap Controller X']
    measurements['Measured Wrist Position Relative to Leap Y'] = measurements['Measured Wrist Y'] - measurements['Leap Controller Y']
    measurements['Measured Wrist Position Relative to Leap Z'] = measurements['Measured Wrist Z'] #- measurements['Leap Controller Z']

    measurements['Measured Thumb Position Relative to Leap X'] = measurements['Measured Thumb X'] - measurements['Leap Controller X']
    measurements['Measured Thumb Position Relative to Leap Y'] = measurements['Measured Thumb Y'] - measurements['Leap Controller Y']
    measurements['Measured Thumb Position Relative to Leap Z'] = measurements['Measured Thumb Z'] #- measurements['Leap Controller Z']
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

    full_table = measurements

    return ground_truth, measured_positions, full_table

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

def plot_results(measured, ground_truth, corrected, pre_metrics, post_metrics, joint_name, labels=['X', 'Y', 'Z'], path=None):
    for i, dim_name in enumerate(labels):
        plt.figure(figsize=(6, 5))
        plt.scatter(measured[:, i], ground_truth[:, i], label="Original Measured", alpha=0.2)
        plt.scatter(corrected[:, i], ground_truth[:, i], label="Corrected", alpha=0.2)
        plt.xlabel(f"{dim_name} (Predicted)")
        plt.ylabel(f"{dim_name} (Ground Truth)")
        plt.legend()
        plt.title(
            f"{dim_name} Correction - {joint_name}\n"
            f"Before RMSE: {pre_metrics[i]['rmse']:.4f}, After RMSE: {post_metrics[i]['rmse']:.4f}\n"
            f"Before R²: {pre_metrics[i]['r2']:.4f}, After R²: {post_metrics[i]['r2']:.4f}"
        )
        if path:
            plt.savefig(path+R"/"+joint_name+"_"+dim_name+".png")

        plt.grid(True)
        plt.show()

def print_coefficients(poly, models, labels=['X', 'Y', 'Z']):
    feature_names = poly.get_feature_names_out(['X', 'Y', 'Z'])
    for i, dim_name in enumerate(labels):
        print(f"\n{dim_name} correction model coefficients:")
        for coef, name in zip(models[i].coef_, feature_names):
            print(f"{name}: {coef:.4f}")
        print(f"Intercept: {models[i].intercept_:.4f}")

def correct_and_plot(measured, ground_truth, degrees, label, path=None):

    measured_train, measured_test, truth_train, truth_test = train_test_split(
    measured, ground_truth, test_size=0.2, random_state=None, shuffle=True)

    pre_correction_metrics = evaluate_model(truth_test, measured_test)

    polys = {}
    models_dict = {}
    corrected_dict = {}

    post_correction_metrics_X = {}
    post_correction_metrics_Y = {}
    post_correction_metrics_Z = {}

    # Fit & Correct

    for degree in degrees:
        poly, models = fit_multivariate_polynomial_correction(measured_train, truth_train, degree)
        polys[degree] = poly
        models_dict[degree] = models

        corrected = apply_multivariate_correction(measured_test, poly, models)
    
        corrected_dict[degree] = apply_multivariate_correction(measured, poly, models)
        # After Correction

        post_correction_metrics = evaluate_model(truth_test, corrected)

        post_correction_metrics_X[degree] = post_correction_metrics[0]
        post_correction_metrics_Y[degree] = post_correction_metrics[1]
        post_correction_metrics_Z[degree] = post_correction_metrics[2]

        # Visualize
        plot_results(measured_test, truth_test, corrected, pre_correction_metrics, post_correction_metrics, label, path=path)

        # Coefficients & Metrics

        return corrected_dict, pre_correction_metrics, [post_correction_metrics_X, post_correction_metrics_Y, post_correction_metrics_X], polys, models_dict

def save_as_csv(X, Y, Z, U, V, W, name):
    errors = pd.DataFrame()
    errors['AX'] = X
    errors['AY'] = Y
    errors['AZ'] = Z
    errors['EX'] = U
    errors['EY'] = V
    errors['EZ'] = W

    errors = errors.dropna()

    errors.to_csv(R'C:\Users\Sheff\OneDrive\Desktop\data\error_vectors_{}.csv'.format(name), index=False)

def plot_3d_quiver(x, y, z, u, v, w, title, savefig):
    # COMPUTE LENGTH OF VECTOR -> MAGNITUDE
    mag = np.sqrt(np.abs(v)**2 + np.abs(u)**2 + np.abs(w)**2)
  
    c = (mag.ravel() - mag.min())/mag.ptp()
    c = plt.cm.jet(c)

    fig = plt.figure(figsize=(30,20))
    ax = fig.add_subplot(111, projection='3d')
    q = ax.quiver(x, y, z, u, v, w, colors = c, cmap = plt.cm.jet, arrow_length_ratio = 0.5)
    
    ax.set_xlabel('Left - Right [cm]', labelpad=30)
    ax.set_ylabel('Backward-Forward [cm]', labelpad=30)
    ax.set_zlabel('Up - Down [cm]', labelpad=30)
    ax.set_title(title)
    ax.tick_params(axis='both', which='major', pad=10)
    ax.invert_zaxis()
    q.set_array(mag)
    fig.colorbar(q, ax=ax, label = 'Magnitude of error [cm]')
    plt.savefig("C:\\Users\\Sheff\\OneDrive\\Desktop\\data\\"+savefig, format='png')
    plt.show()


#%%
# Loading real data

path = R"C:\Users\Sheff\OneDrive\Desktop\data\data_low.txt"

ground_truth, measured_positions, full_table = load_error_dataframe(path)

ground_truth_Index = ground_truth.iloc[:, 0:3].to_numpy()
ground_truth_Thumb = ground_truth.iloc[:, 3:6].to_numpy()
ground_truth_Wrist = ground_truth.iloc[:, 6:9].to_numpy()

measured_positions_Index = measured_positions.iloc[:, 0:3].to_numpy()
measured_positions_Thumb = measured_positions.iloc[:, 3:6].to_numpy()
measured_positions_Wrist = measured_positions.iloc[:, 6:9].to_numpy()

# %%
#Doing the same but with 80/20 train/test split
degrees = [5]

corrected_index, pre_correction_index, post_correction_index, polys_index, models_index = correct_and_plot(measured_positions_Index, ground_truth_Index, degrees, "Index - low", path=R"C:\Users\Sheff\OneDrive\Desktop\data\Correction Plots")
corrected_thumb, pre_correction_thumb, post_correction_thumb, polys_thumb, models_thumb = correct_and_plot(measured_positions_Thumb, ground_truth_Thumb, degrees, "Thumb - low", path=R"C:\Users\Sheff\OneDrive\Desktop\data\Correction Plots")
corrected_wrist, pre_correction_wrist, post_correction_wrist, polys_wrist, models_wrist = correct_and_plot(measured_positions_Wrist, ground_truth_Wrist, degrees, "Wrist - low", path=R"C:\Users\Sheff\OneDrive\Desktop\data\Correction Plots")

# %%
corrected_pd = pd.DataFrame()

corrected_pd["Measured Index X"] = corrected_index[5][:,0]
corrected_pd["Measured Index Y"] = corrected_index[5][:,1]
corrected_pd["Measured Index Z"] = corrected_index[5][:,2]

corrected_pd["Measured Wrist X"] = corrected_wrist[5][:,0]
corrected_pd["Measured Wrist Y"] = corrected_wrist[5][:,1]
corrected_pd["Measured Wrist Z"] = corrected_wrist[5][:,2]

corrected_pd["Measured Thumb X"] = corrected_thumb[5][:,0]
corrected_pd["Measured Thumb Y"] = corrected_thumb[5][:,1]
corrected_pd["Measured Thumb Z"] = corrected_thumb[5][:,2]

corrected_pd.to_csv(R"C:\Users\Sheff\OneDrive\Desktop\data\Correction Plots\corrected_positions_low.csv")
#%%
errors = pd.DataFrame()

#%%
errors['Index Errors X'] = ground_truth['Reference Index X'].values - corrected_pd['Measured Index X'].values
errors['Index Errors Y'] = ground_truth['Reference Index Y'].values - corrected_pd['Measured Index Y'].values
errors['Index Errors Z'] = ground_truth['Reference Index Z'].values - corrected_pd['Measured Index Z'].values

errors['Wrist Errors X'] = ground_truth['Reference Wrist X'].values - corrected_pd['Measured Wrist X'].values
errors['Wrist Errors Y'] = ground_truth['Reference Wrist Y'].values - corrected_pd['Measured Wrist Y'].values
errors['Wrist Errors Z'] = ground_truth['Reference Wrist Z'].values - corrected_pd['Measured Wrist Z'].values

errors['Thumb Errors X'] = ground_truth['Reference Thumb X'].values - corrected_pd['Measured Thumb X'].values
errors['Thumb Errors Y'] = ground_truth['Reference Thumb Y'].values - corrected_pd['Measured Thumb Y'].values
errors['Thumb Errors Z'] = ground_truth['Reference Thumb Z'].values - corrected_pd['Measured Thumb Z'].values

#%%
errors_arr_index = np.stack((errors['Index Errors X'].values, errors['Index Errors Z'].values, errors['Index Errors Y'].values), axis=1)
errors_arr_wrist = np.stack((errors['Wrist Errors X'].values, errors['Wrist Errors Z'].values, errors['Wrist Errors Y'].values), axis=1)
errors_arr_thumb = np.stack((errors['Thumb Errors X'].values, errors['Thumb Errors Z'].values, errors['Thumb Errors Y'].values), axis=1)

positions_arr_x = ground_truth['Reference Thumb X'].values
positions_arr_y = ground_truth['Reference Thumb Y'].values
positions_arr_z = ground_truth['Reference Thumb Z'].values
positions_arr = np.stack((positions_arr_x, positions_arr_y, positions_arr_z), axis=1)

#%%
soa_index = np.append(positions_arr, errors_arr_index, axis=1)*100
df_soa_index = pd.DataFrame(soa_index, columns=('v', 'w', 'u', 'x', 'y', 'z'))

X, Y, Z, U, V, W = zip(*soa_index)
save_as_csv(X,Z,Y,U,V,W, 'Index lowRes Corrected')

plot_3d_quiver(X,Z,Y,U,V,W, 'Errors for tip of the index finger \n low Resource Usage Mode', 'Index,low,corrected.png')
#%%
soa_thumb = np.append(positions_arr, errors_arr_thumb, axis=1)*100
df_soa_thumb = pd.DataFrame(soa_thumb, columns=('v', 'w', 'u', 'x', 'y', 'z'))

X, Y, Z, U, V, W = zip(*soa_thumb)
save_as_csv(X,Z,Y,U,V,W, 'Thumb lowRes Corrected')

plot_3d_quiver(X,Z,Y,U,V,W, 'Errors for the thumb joint \n low Resource Usage Mode', 'Thumb,low,corrected.png')
#%%
soa_wrist = np.append(positions_arr, errors_arr_wrist, axis=1)*100
df_soa_wrist = pd.DataFrame(soa_wrist, columns=('v', 'w', 'u', 'x', 'y', 'z'))

X, Y, Z, U, V, W = zip(*soa_wrist)
save_as_csv(X,Z,Y,U,V,W, 'Wrist lowRes Corrected')

plot_3d_quiver(X,Z,Y,U,V,W, 'Errors for the wrist joint \n low Resource Usage Mode', 'Wrist,low,corrected.png')
# %%
