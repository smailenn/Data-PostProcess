#Calibration file, for analyzing two stationary LIS3DH accelerometers and normalizing them 
#Updated:  09 Nov 2020
Version = 2.0
#Produces txt file results for Main data analysis file

#import libraries
import pandas as pd
import numpy as np
import math 
from numpy import savetxt
from scipy import stats
from scipy import signal
import os

#DTS or Arduino Data (Using LIS3dh)
Arduino = 'Arduino'
DTS = 'DTS'
No = None
Yes = None


###############################################
Sensor_select = 'Arduino'  #DTS or Arduino
# Data location for calibration values?
File_path =  'C:\\Users\\smailen\\OneDrive - Quality Bicycle Products\Vibration Analysis\\Warbird V4\\05 Nov 2020\\Data\\Warbird Cal.txt' 
################################################

#Note this is typical layout, but you may have to adjust
#Sensor 1 - upper sensor, at Handlebar/Seatpost/etc.
#Sensor 2 - lower sensor, at Axle/s

#Orientation of calculations, make sure sensors are calibrated to match - X is positive in riding direction (front of rider), Y is positive above rider, Z is normal to driveside

#read the data into an array from text file
if Sensor_select == Arduino:
    dataframe = pd.read_csv(File_path, sep=',')    
    print('Arduino data selected')
    filename = os.path.basename(File_path)
    FN = os.path.splitext(filename)[0]+' values'
    #dataframe.info()
    #print(dataframe)
elif Sensor_select == DTS:
    #dataframe_DTS = pd.read_excel(File_path)
    dataframe_DTS = pd.read_csv(File_path)
    print('DTS data selected')
    filename = os.path.basename(File_path)
    FN = os.path.splitext(filename)[0]
    #dataframe_DTS.info()
    #print(dataframe_DTS)
else:
    print("No Data present")

# Create Text file of results for record
f = open('{}.txt' .format(FN), "w")    

#Seperate data into column arrays
#For Arduino type data logger
if Sensor_select == Arduino:
    time = dataframe["Time"]
    Sensor1xRaw = dataframe["Sensor 1 X"]
    Sensor1yRaw = dataframe["Sensor 1 Y"]
    Sensor1zRaw = dataframe["Sensor 1 Z"]
    Sensor2xRaw = dataframe["Sensor 2 X"]
    Sensor2yRaw = dataframe["Sensor 2 Y"]
    Sensor2zRaw = dataframe["Sensor 2 Z"]

#For DTS data logger
if Sensor_select == DTS:
    time = dataframe_DTS["Time(s)"]
    Sensor1x_g = dataframe_DTS["Seat Collar,X(Corrected)"]
    Sensor1y_g = dataframe_DTS["Seat Collar,Vertical,Normalized"]
    Sensor1z_g = dataframe_DTS["Seat Collar,Z(Corrected)"]
    Sensor2x_g = dataframe_DTS["Rear Axle,X(Corrected)"]
    Sensor2y_g = dataframe_DTS["Rear Axle,Vertical,Normalized"]
    Sensor2z_g = dataframe_DTS["Rear Axle,Z"]


#Time and Index review
if Sensor_select == Arduino: 
    time = time - time[0]  #Normalize time back to starting at 0
    time = time/1000000 #convert uS to S

time_int = np.ediff1d(time)  # Find time intervals - this is array (time between measurements)
#time_int_avg = stats.trim_mean(time_int,0.1) # Trim time intervals of SD lag points, large time intervals during a SD write, this is scalar
time_int_avg = np.mean(time_int)
#print('Time Interval Average =', "%.8f"% time_int_avg,'S')
time_Hz = 1/time_int_avg
print('Sample Rate =', '%.2f'% time_Hz, 'Hz')
f.write('Sample Rate (Hz) ,' '%.2f'% time_Hz) 
f.write('\n')
total_time = len(time)/time_Hz
print('Length of run =', len(time),'seconds')
f.write('Length of run (sec) ,' '%.2f'% total_time)
f.write('\n')

#re-orientate if necessary 
#Orientation here must match orientation used in analysis file or calibration offset values will be off
if Sensor_select == Arduino: 
    Cd = -1 #multiply raw data by this to change trajectory to match bicycle Orientation
    print('data calibrated')
    #Orientate sensors
    Sensor1x_g = Sensor1xRaw
    Sensor1y_g = Sensor1yRaw
    Sensor1z_g = Sensor1zRaw*Cd
    Sensor2x_g = Sensor2xRaw
    Sensor2y_g = Sensor2yRaw
    Sensor2z_g = Sensor2zRaw*Cd
    print('data orientated + converted')

#Calibrate ============================================================================================================
# Find Average of sensor data
S1ymean = np.mean(Sensor1y_g)
S2ymean = np.mean(Sensor2y_g)    
S1xmean = np.mean(Sensor1x_g)
S2xmean = np.mean(Sensor2x_g)    
S1zmean = np.mean(Sensor1z_g)
S2zmean = np.mean(Sensor2z_g)  
print('Sensor 1 X Axis Average , ', '%.5f'% S1xmean, 'm/s^2')
print('Sensor 1 Y Axis Average , ', '%.5f'% S1ymean, 'm/s^2')
print('Sensor 1 Z Axis Average , ', '%.5f'% S1zmean, 'm/s^2')
print('Sensor 2 X Axis Average , ', '%.5f'% S2xmean, 'm/s^2G')
print('Sensor 2 Y Axis Average , ', '%.5f'% S2ymean, 'm/s^2')
print('Sensor 2 Z Axis Average , ', '%.5f'% S2zmean, 'm/s^2')
f.write('Sensor 1 X Axis Average (m/s^2) ,' '%.5f'% S1xmean)
f.write('\n')
f.write('Sensor 1 Y Axis Average (m/s^2) ,' '%.5f'% S1ymean)
f.write('\n')
f.write('Sensor 1 Z Axis Average (m/s^2) ,' '%.5f'% S1zmean)
f.write('\n')
f.write('Sensor 2 X Axis Average (m/s^2) ,' '%.5f'% S2xmean)
f.write('\n')
f.write('Sensor 2 Y Axis Average (m/s^2) ,' '%.5f'% S2ymean)
f.write('\n')
f.write('Sensor 2 Z Axis Average (m/s^2) ,' '%.5f'% S2zmean)
f.write('\n')

# Difference between between Accelerometers 
S1diffS2x = abs(S1xmean - S2xmean)
S1diffS2y = abs(S1ymean - S2ymean)
S1diffS2z = abs(S1zmean - S2zmean)
print('Sensor 1 & 2 X Axis difference , ', '%.5f'% S1diffS2x, 'm/s^2')
print('Sensor 1 & 2 Y Axis difference , ', '%.5f'% S1diffS2y, 'm/s^2')
print('Sensor 1 & 2 Z Axis difference , ', '%.5f'% S1diffS2z, 'm/s^2')
f.write('Sensor 1 & 2 X Axis difference (m/s^2) , ' '%.5f'% S1diffS2x)
f.write('\n')
f.write('Sensor 1 & 2 Y Axis difference (m/s^2), ' '%.5f'% S1diffS2y)
f.write('\n')
f.write('Sensor 1 & 2 Z Axis difference (m/s^2), ' '%.5f'% S1diffS2z)
f.write('\n')

#Acceleration values in m/s^2
#Accels should be measuring with 
Gravity = -9.80665 

#Calibration offsets
S1xCalib = 0 - S1xmean
S1yCalib = Gravity - S1ymean
S1zCalib = 0 - S1zmean
S2xCalib = 0 - S2xmean
S2yCalib = Gravity - S2ymean
S2zCalib = 0 - S2zmean
print('Sensor 1 X Axis Calibration value , ', '%.5f'% S1xCalib, 'm/s^2')
print('Sensor 1 Y Axis Calibration value , ', '%.5f'% S1yCalib, 'm/s^2')
print('Sensor 1 Z Axis Calibration value , ', '%.5f'% S1zCalib, 'm/s^2')
print('Sensor 2 X Axis Calibration value , ', '%.5f'% S2xCalib, 'm/s^2')
print('Sensor 2 Y Axis Calibration value , ', '%.5f'% S2yCalib, 'm/s^2')
print('Sensor 2 Z Axis Calibration value , ', '%.5f'% S2zCalib, 'm/s^2')
f.write('Sensor 1 X Axis Calibration value , ' '%.5f'% S1xCalib)
f.write('\n')
f.write('Sensor 1 Y Axis Calibration value , ' '%.5f'% S1yCalib)
f.write('\n')
f.write('Sensor 1 Z Axis Calibration value , ' '%.5f'% S1zCalib)
f.write('\n')
f.write('Sensor 2 X Axis Calibration value , ' '%.5f'% S2xCalib)
f.write('\n')
f.write('Sensor 2 Y Axis Calibration value , ' '%.5f'% S2yCalib)
f.write('\n')
f.write('Sensor 2 Z Axis Calibration value , ' '%.5f'% S2zCalib)
f.write('\n')

# Check calibration worked properly
S1xCalib_g = Sensor1x_g + S1xCalib
S1yCalib_g = Sensor1y_g + S1yCalib
S1zCalib_g = Sensor1z_g + S1zCalib
S2xCalib_g = Sensor2x_g + S2xCalib
S2yCalib_g = Sensor2y_g + S2yCalib
S2zCalib_g = Sensor2z_g + S2zCalib

print('Sensor 1 X Axis Average value after Calibration = ', '%.5f'% np.mean(S1xCalib_g), 'm/s^2')
print('Sensor 1 Y Axis Average value after Calibration = ', '%.5f'% np.mean(S1yCalib_g), 'm/s^2')
print('Sensor 1 Z Axis Average value after Calibration = ', '%.5f'% np.mean(S1zCalib_g), 'm/s^2')
print('Sensor 2 X Axis Average value after Calibration = ', '%.5f'% np.mean(S2xCalib_g), 'm/s^2')
print('Sensor 2 Y Axis Average value after Calibration = ', '%.5f'% np.mean(S2yCalib_g), 'm/s^2')
print('Sensor 2 Z Axis Average value after Calibration = ', '%.5f'% np.mean(S2zCalib_g), 'm/s^2')

f.close()
