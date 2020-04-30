#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Calibration variables and Accelerometer setup
accel_range = 16 #What Range in G 2,4,8,16 g
accel_raw_scale_neg = -32767 #raw value range negative values
accel_raw_scale_pos = 32768 #raw value range positive values


#read the data into an array from text file
dataframe = pd.read_csv('C:\\Users\\smailen\\OneDrive - Quality Bicycle Products\\Vibration Analysis\\Waxwing Test\\WAXWING02.txt', sep='\t')

#get info about the array
#type(dataframe)
#dataframe.shape
#dataframe.info()
#print(dataframe)

#export sheet to Excel for record and review
#dataframe.to_excel('WAXWING00.xlsx', sheet_name='Waxwing00')

#Seperate data into column arrays
time = dataframe["Time"]
Sensor1xRaw = dataframe["Sensor 1 X"]
Sensor1yRaw = dataframe["Sensor 1 Y"]
Sensor1zRaw = dataframe["Sensor 1 Z"]
Sensor2xRaw = dataframe["Sensor 2 X"]
Sensor2yRaw = dataframe["Sensor 2 Y"]
Sensor2zRaw = dataframe["Sensor 2 Z"]

#Calibrate, orientate, and Normalize readings to g's
#Calculate offset values from calibration test at beginning of test run
accel_g_unit = (accel_raw_scale_pos - accel_raw_scale_neg)/accel_range
Sensor1x_offset = -80 #calibration value for Sensor 1x
Sensor1y_offset = -715 
Sensor1z_offset = -34
Sensor2x_offset = -25
Sensor2y_offset = -774
Sensor2z_offset = 98

Sensor1x_g = (Sensor1xRaw + Sensor1x_offset)*(accel_range/accel_raw_scale_pos)
Sensor1y_g = (Sensor1yRaw + Sensor1y_offset)*(accel_range/accel_raw_scale_pos)
Sensor1z_g = (Sensor1zRaw + Sensor1z_offset)*(accel_range/accel_raw_scale_pos)
Sensor2x_g = (Sensor2xRaw + Sensor2x_offset)*(accel_range/accel_raw_scale_pos)
Sensor2y_g = (Sensor2yRaw + Sensor2y_offset)*(accel_range/accel_raw_scale_pos)
Sensor2y_g = (Sensor2yRaw + Sensor2y_offset)*(accel_range/accel_raw_scale_pos)

#Analyze the Arrays
S1ymean = np.mean(Sensor1y_g)
print('Sensor 1 Y Axis Average = ', S1ymean)
S2ymean = np.mean(Sensor2y_g)    
print('Sensor 2 Y Axis Average = ', S2ymean)

#make some plots
plt.plot(time, Sensor2y_g, time, Sensor1y_g)
plt.xlabel('Time(Minutes)')
plt.ylabel('Gs')
plt.show()
plt.plot()