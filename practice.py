#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Calibration variables and Accelerometer setup
accel_range = 16 #What Range in G 2,4,8,16 g
accel_raw_scale_neg = -32767 #raw value range negative values
accel_raw_scale_pos = 32768 #raw value range positive values

#What Accelerometers [data arrays] do you want to review?
AS = 'y' # Accel Set 

#read the data into an array from text file
dataframe = pd.read_csv('C:\\Users\\smailen\\OneDrive - Quality Bicycle Products\\Vibration Analysis\\Waxwing Test\\WAXWING01.txt', sep='\t')

#get info about the array
#type(dataframe)
#dataframe.shape
#dataframe.info()
#print(dataframe)

#Seperate data into column arrays
time = dataframe["Time"]
Sensor1xRaw = dataframe["Sensor 1 X"]
Sensor1yRaw = dataframe["Sensor 1 Y"]
Sensor1zRaw = dataframe["Sensor 1 Z"]
Sensor2xRaw = dataframe["Sensor 2 X"]
Sensor2yRaw = dataframe["Sensor 2 Y"]
Sensor2zRaw = dataframe["Sensor 2 Z"]

#Time and Index review
time = time - time[0]
#print(time)
# Verify sample rate
dt = time[101] - time[100]
print('Time Interval =', dt)


#Calibrate, orientate, and Normalize readings to G's
#Calculate offset values from calibration test at beginning of test runs, ideally done at same time as testing
accel_g_unit = (accel_raw_scale_pos - accel_raw_scale_neg)/accel_range
Sensor1x_offset = 63.5 
Sensor1y_offset = -692 
Sensor1z_offset = -22.6
Sensor2x_offset = 12.6
Sensor2y_offset = -769
Sensor2z_offset = 98
Calibrate_direction = -1 #multiply raw data by this to change trajectory to normal of drivetrain
#Orientate sensors, calibrate, and convert to G's
Sensor1x_g = ((Sensor1xRaw + Sensor1x_offset)*(accel_range/accel_raw_scale_pos))*Calibrate_direction
Sensor1y_g = (Sensor1yRaw + Sensor1y_offset)*(accel_range/accel_raw_scale_pos)
Sensor1z_g = (Sensor1zRaw + Sensor1z_offset)*(accel_range/accel_raw_scale_pos)
Sensor2x_g = (Sensor2xRaw + Sensor2x_offset)*(accel_range/accel_raw_scale_pos)
Sensor2y_g = (Sensor2yRaw + Sensor2y_offset)*(accel_range/accel_raw_scale_pos)
Sensor2z_g = ((Sensor2zRaw + Sensor2z_offset)*(accel_range/accel_raw_scale_pos))*Calibrate_direction

#If needed trim the arrays for bad/false/fault data
trim_beg = 0 #trim beginning of array
trim_end = 300000  #trim end of array
time = time[trim_beg:trim_end]
Sensor1x_g = Sensor1x_g[trim_beg:trim_end]
Sensor1y_g = Sensor1y_g[trim_beg:trim_end]
Sensor1z_g = Sensor1z_g[trim_beg:trim_end]
Sensor2x_g = Sensor2x_g[trim_beg:trim_end]
Sensor2y_g = Sensor2y_g[trim_beg:trim_end]
Sensor2z_g = Sensor2z_g[trim_beg:trim_end]

#Analyze the Arrays ============================================================================================================
# Find Average of sensor data
S1ymean = np.mean(Sensor1y_g)
S2ymean = np.mean(Sensor2y_g)    
S1xmean = np.mean(Sensor1x_g)
S2xmean = np.mean(Sensor2x_g)    
S1zmean = np.mean(Sensor1z_g)
S2zmean = np.mean(Sensor2z_g)  
print('Sensor 1 Y Axis Average = ', S1ymean)
print('Sensor 2 Y Axis Average = ', S2ymean)
print('Sensor 1 X Axis Average = ', S1xmean)
print('Sensor 2 X Axis Average = ', S2xmean)
print('Sensor 1 Z Axis Average = ', S1zmean)
print('Sensor 2 Z Axis Average = ', S2zmean)

# Find MAX of sensor data
S1yMAX = np.amax(Sensor1y_g)
S2yMAX = np.amax(Sensor2y_g)    
S1xMAX = np.amax(Sensor1x_g)
S2xMAX = np.amax(Sensor2x_g)    
S1zMAX = np.amax(Sensor1z_g)
S2zMAX = np.amax(Sensor2z_g)    
print('Sensor 1 Y Axis MAX = ', S1yMAX)
print('Sensor 2 Y Axis MAX = ', S2yMAX)
print('Sensor 1 X Axis MAX = ', S1xMAX)
print('Sensor 2 X Axis MAX = ', S2xMAX)
print('Sensor 1 Z Axis MAX = ', S1zMAX)
print('Sensor 2 Z Axis MAX = ', S2zMAX)

# Find MIN of sensor data
S1yMIN = np.amin(Sensor1y_g)
S2yMIN = np.amin(Sensor2y_g)    
S1xMIN = np.amin(Sensor1x_g)
S2xMIN = np.amin(Sensor2x_g)    
S1zMIN = np.amin(Sensor1z_g)
S2zMIN = np.amin(Sensor2z_g)    
print('Sensor 1 Y Axis MIN = ', S1yMIN)
print('Sensor 2 Y Axis MIN = ', S2yMIN)
print('Sensor 1 X Axis MIN = ', S1xMIN)
print('Sensor 2 X Axis MIN = ', S2xMIN)
print('Sensor 1 Z Axis MIN = ', S1zMIN)
print('Sensor 2 Z Axis MIN = ', S2zMIN)

# Transmissibility and Isolation %
Trans_x = (S1xmean/S2xmean)*100
Trans_y = (S1ymean/S2ymean)*100
Trans_z = (S1zmean/S2zmean)*100
print('X Axis Transmissibility =', Trans_x, '%')
print('Y Axis Transmissibility =', Trans_y, '%')
print('Z Axis Transmissibility =', Trans_z, '%')

#make some plots
plt.plot(time, Sensor2y_g, time, Sensor1y_g)
plt.axis([0, 180000000, -16, 16])
plt.xlabel('Time(Minutes)')
plt.ylabel('Gs')
plt.show()
plt.plot()