#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import numpy as np, matplotlib.pyplot as plt, pandas as pd
from mpl_toolkits.mplot3d import Axes3D

SMALL_SIZE = 15
MEDIUM_SIZE = 30
BIGGER_SIZE = 45

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)
#%%

def save_as_csv(X, Y, Z, U, V, W, name):
    errors = pd.DataFrame()
    errors['AX'] = X
    errors['AY'] = Y
    errors['AZ'] = Z
    errors['EX'] = U
    errors['EY'] = V
    errors['EZ'] = W

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

def plot_3d_quiver_plotly(x, y, z, u, v, w):
    fig = go.Figure(
    data=go.Cone(
        x=X,
        y=Z,
        z=Y,
        u=U,
        v=W,
        w=V,
        sizemode="raw",
        sizeref=1,
        colorscale="Portland",
        hoverinfo="all",
        
    ),
    layout=dict(
        width=900, height=600, scene=dict(camera=dict(eye=dict(x=1.2, y=0, z=0.6)))
    ),
    )
    fig.update_scenes(
    zaxis = dict(autorange="reversed")
    )
    fig.update_layout(
        font=dict(size=15))
    fig.show()  
#%%
measurements = pd.read_csv(R"C:\Users\Sheff\OneDrive\Desktop\data\data_high.txt", sep = ';', header=0, index_col=False)
measurements = measurements.dropna(axis=0)

#%%
measurements['Reference Index X'] = measurements['Reference Index X'].iloc[-1]
measurements['Reference Index Y'] = measurements['Reference Index Y'].iloc[-1]

measurements['Reference Wrist X'] = measurements['Reference Wrist X'].iloc[-1]
measurements['Reference Wrist Y'] = measurements['Reference Wrist Y'].iloc[-1]

measurements['Reference Thumb X'] = measurements['Reference Thumb X'].iloc[-1]
measurements['Reference Thumb Y'] = measurements['Reference Thumb Y'].iloc[-1]

#%%
errors = pd.DataFrame()
#%%
errors['Index Errors X'] = measurements['Reference Index X'] - measurements['Measured Index X']
errors['Index Errors Y'] = measurements['Reference Index Y'] - measurements['Measured Index Y']
errors['Index Errors Z'] = measurements['Reference Index Z'] - measurements['Measured Index Z']

errors['Wrist Errors X'] = measurements['Reference Wrist X'] - measurements['Measured Wrist X']
errors['Wrist Errors Y'] = measurements['Reference Wrist Y'] - measurements['Measured Wrist Y']
errors['Wrist Errors Z'] = measurements['Reference Wrist Z'] - measurements['Measured Wrist Z']

errors['Thumb Errors X'] = measurements['Reference Thumb X'] - measurements['Measured Thumb X']
errors['Thumb Errors Y'] = measurements['Reference Thumb Y'] - measurements['Measured Thumb Y']
errors['Thumb Errors Z'] = measurements['Reference Thumb Z'] - measurements['Measured Thumb Z']

# %%
errors_abs = errors.abs()
errors['Wrist Position Relative to Leap X'] = measurements['Reference Wrist X'] - measurements['Leap Controller X']
errors['Wrist Position Relative to Leap Y'] = measurements['Reference Wrist Y'] - measurements['Leap Controller Y']
errors['Wrist Position Relative to Leap Z'] = measurements['Reference Wrist Z'] - measurements['Leap Controller Z']
#%%
errors_arr_index = np.stack((errors['Index Errors X'].values, errors['Index Errors Z'].values, errors['Index Errors Y'].values), axis=1)
errors_arr_wrist = np.stack((errors['Wrist Errors X'].values, errors['Wrist Errors Z'].values, errors['Wrist Errors Y'].values), axis=1)
errors_arr_thumb = np.stack((errors['Thumb Errors X'].values, errors['Thumb Errors Z'].values, errors['Thumb Errors Y'].values), axis=1)

#%%

errors_arr_index_mag = np.linalg.norm(errors_arr_index, axis=1)
errors_arr_wrist_mag = np.linalg.norm(errors_arr_wrist, axis=1)
errors_arr_thumb_mag = np.linalg.norm(errors_arr_thumb, axis=1)
#%%
error_index_avr = np.mean(errors_arr_index_mag)
error_wrist_avr = np.mean(errors_arr_wrist_mag)
error_thumb_avr = np.mean(errors_arr_thumb_mag)

error_index_std = np.std(errors_arr_index_mag)
error_wrist_std = np.std(errors_arr_wrist_mag)
error_thumb_std = np.std(errors_arr_thumb_mag)
#%%

mean_index_x = np.mean(errors_abs['Index Errors X'])
mean_index_y = np.mean(errors_abs['Index Errors Y'])
mean_index_z = np.mean(errors_abs['Index Errors Z'])

mean_wrist_x = np.mean(errors_abs['Wrist Errors X'])
mean_wrist_y = np.mean(errors_abs['Wrist Errors Y'])
mean_wrist_z = np.mean(errors_abs['Wrist Errors Z'])

mean_thumb_x = np.mean(errors_abs['Thumb Errors X'])
mean_thumb_y = np.mean(errors_abs['Thumb Errors Y'])
mean_thumb_z = np.mean(errors_abs['Thumb Errors Z'])

std_index_x = np.std(errors_abs['Index Errors X'])
std_index_y = np.std(errors_abs['Index Errors Y'])
std_index_z = np.std(errors_abs['Index Errors Z'])

std_wrist_x = np.std(errors_abs['Wrist Errors X'])
std_wrist_y = np.std(errors_abs['Wrist Errors Y'])
std_wrist_z = np.std(errors_abs['Wrist Errors Z'])

std_thumb_x = np.std(errors_abs['Thumb Errors X'])
std_thumb_y = np.std(errors_abs['Thumb Errors Y'])
std_thumb_z = np.std(errors_abs['Thumb Errors Z'])



#%%
print("Number of measurements: {:.3f}".format(errors['Index Errors X'].size))

print("Mean Error for Index Finger [cm]: {:.3f}".format(error_index_avr*100))
print("Mean Error for Wrist Finger [cm]: {:.3f}".format(error_wrist_avr*100))
print("Mean Error for Thumb Finger [cm]: {:.3f}".format(error_thumb_avr*100))

print("Standard deviation of Error for Index Finger [cm]: {:.3f}".format(error_index_std*100))
print("Standard deviation of Error for Wrist Joint [cm]: {:.3f}".format(error_wrist_std*100))
print("Standard deviation of Error for Thumb Finger [cm]: {:.3f}".format(error_thumb_std*100))
#%%

print("Mean Error for Index Finger for X [cm]: {:.3f}".format(mean_index_x*100))
print("Mean Error for Index Finger for Y [cm]: {:.3f}".format(mean_index_y*100))
print("Mean Error for Index Finger for Z [cm]: {:.3f}".format(mean_index_z*100))

print("Mean Error for Wrist for X [cm]: {:.3f}".format(mean_wrist_x*100))
print("Mean Error for Wrist for Y [cm]: {:.3f}".format(mean_wrist_y*100))
print("Mean Error for Wrist for Z [cm]: {:.3f}".format(mean_wrist_z*100))

print("Mean Error for Thumb Finger for X [cm]: {:.3f}".format(mean_thumb_x*100))
print("Mean Error for Thumb Finger for Y [cm]: {:.3f}".format(mean_thumb_y*100))
print("Mean Error for Thumb Finger for Z [cm]: {:.3f}".format(mean_thumb_z*100))

print("Standard deviation for Index Finger for X [cm]: {:.3f}".format(std_index_x*100))
print("Standard deviation for Index Finger for Y [cm]: {:.3f}".format(std_index_y*100))
print("Standard deviation for Index Finger for Z [cm]: {:.3f}".format(std_index_z*100))

print("Stndard deviation for Wrist for X [cm]: {:.3f}".format(std_wrist_x*100))
print("Stndard deviation for Wrist for Y [cm]: {:.3f}".format(std_wrist_y*100))
print("Stndard deviation for Wrist for Z [cm]: {:.3f}".format(std_wrist_z*100))

print("Standard Deviation for Thumb Finger for X [cm]: {:.3f}".format(std_thumb_x*100))
print("Standard Deviation for Thumb Finger for Y [cm]: {:.3f}".format(std_thumb_y*100))
print("Standard Deviation for Thumb Finger for Z [cm]: {:.3f}".format(std_thumb_z*100))

#%%
positions_arr_x = errors['Wrist Position Relative to Leap X'].values
positions_arr_y = errors['Wrist Position Relative to Leap Y'].values
positions_arr_z = errors['Wrist Position Relative to Leap Z'].values
positions_arr = np.stack((positions_arr_x, positions_arr_y, positions_arr_z), axis=1)

#%%
soa_index = np.append(positions_arr, errors_arr_index, axis=1)*100
df_soa_index = pd.DataFrame(soa_index, columns=('v', 'w', 'u', 'x', 'y', 'z'))

X, Y, Z, U, V, W = zip(*soa_index)
 
save_as_csv(X,Z,Y,U,V,W, 'Index HighRes')
#%%
plot_3d_quiver_plotly(X,Z,Y,U,V,W)
plot_3d_quiver(X,Z,Y,U,V,W, 'Errors for tip of the index finger \n Low Resource Usage Mode', 'Index,Low.png')
# %%
soa_wrist = np.append(positions_arr, errors_arr_wrist, axis=1)*100
df_soa_wrist = pd.DataFrame(soa_wrist, columns=('v', 'w', 'u', 'x', 'y', 'z'))

X, Y, Z, U, V, W = zip(*soa_wrist)
save_as_csv(X,Z,Y,U,V,W, 'Wrist HighRes')
#%%
plot_3d_quiver_plotly(X,Z,Y,U,V,W)
plot_3d_quiver(X,Z,Y,U,V,W, 'Errors for the wrist joint \n Low Resource Usage Mode', 'Wrist,Low.png')

soa_thumb = np.append(positions_arr, errors_arr_thumb, axis=1)*100
df_soa_thumb = pd.DataFrame(soa_thumb, columns=('v', 'w', 'u', 'x', 'y', 'z'))

X, Y, Z, U, V, W = zip(*soa_thumb)
save_as_csv(X,Z,Y,U,V,W, 'Thumb HighRes')
#%%
plot_3d_quiver_plotly(X,Z,Y,U,V,W)
plot_3d_quiver(X,Z,Y,U,V,W, 'Errors for the tip of the thumb \n Low Resource Usage Mode', 'Thumb,Low.png')
# %%
