#Main file, for analyzing vibration data from Data logger using two accelerometers from LIS3DH/s
#Updated:  09 Nov 2020
Version = 2.0
#Produces png charts and txt files with results

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math 
from numpy import savetxt
from scipy import stats
from scipy import signal
from scipy import integrate
from scipy.stats import gaussian_kde
import seaborn as sns
import os
import sys 
from datetime import datetime

#DTS or Arduino Data (Using LIS3dh)
Arduino = 'Arduino'
DTS = 'DTS'
No = None
Yes = None

########################################################################################################################
########################################################################################################################
# what file do you want to analyze?
File_path =  'C:\\Users\\smailen\\OneDrive - Quality Bicycle Products\Vibration Analysis\\Warbird V4\\05 Nov 2020\\Data\\Warbird 01.txt'
# Where is the Calibration file?
Cal_path = 'C:\\Users\\smailen\\OneDrive - Quality Bicycle Products\Vibration Analysis\\Warbird V4\\05 Nov 2020\\Warbird Cal values.txt'
Sensor_select = 'Arduino'  #DTS or Arduino
#If data needs to be trimmed, trim here:
Trim_Lower = 10  #Seconds, from start.  if = 0 no trim
Trim_Upper = 40 #Seconds, from start.  if = 0 no trim

########################################################################################################################
########################################################################################################################

#Note this is typical layout, but you may have to adjust
#Sensor 1 - upper sensor, at Handlebar/Seatpost/etc.
#Sensor 2 - lower sensor, at Axle/s

#read the data into an array from text file
if Sensor_select == Arduino:
    dataframe = pd.read_csv(File_path, sep=',')    # TAB=\t
    print('Arduino data selected')
    filename = os.path.basename(File_path)
    FN = os.path.splitext(filename)[0]  # Get file name as variable 
    #dataframe.info()
    #print(dataframe)
    print(FN)
else:
    print("No Data present")
    
# Create Text file of results for record
f = open('{}_Results.txt' .format(FN), "w")
# Add file info to results file
f.write('Analysis Program version =' '%.2f'% Version)
f.write('\n')
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print(now)
f.write(dt_string)
f.write('\n')
###############################################


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
f.write('Sample Rate (Hz) =' '%.2f'% time_Hz) 
f.write('\n')
total_time = len(time)/time_Hz
print('Length of run =', len(time),'seconds')
f.write('Length of run (sec) =' '%.2f'% total_time)
f.write('\n')
#savetxt('time_int.csv', time_int, delimiter=',')  # save Time Interval Array as CSV for review if needed
#print(len(time))

#Calibrate and orientate if necessary
#Calculate offset values from calibration test at beginning of test runs, done at same time as testing
if Sensor_select == Arduino: 
    #Get Calibration values for Cal value
    dataframe_cal = pd.read_csv(Cal_path, sep=',') 
    #Assign values from Cal dataframe
    Sensor1x_offset = dataframe_cal.iat[10,1]
    print(Sensor1x_offset)
    Sensor2x_offset = dataframe_cal.iat[13,1]
    print(Sensor2x_offset)
    Sensor1y_offset = dataframe_cal.iat[11,1]
    print(Sensor1y_offset)
    Sensor2y_offset = dataframe_cal.iat[14,1]
    print(Sensor2y_offset)
    Sensor1z_offset = dataframe_cal.iat[12,1]
    print(Sensor1z_offset)
    Sensor2z_offset = dataframe_cal.iat[15,1]
    print(Sensor2z_offset)

    Cd = -1 #multiply raw data by this to change trajectory to match bicycle Orientation
    Sensor1x_g = Sensor1xRaw + Sensor1x_offset
    Sensor1y_g = Sensor1yRaw + Sensor1y_offset
    Sensor1z_g = Sensor1zRaw*Cd + Sensor1z_offset
    Sensor2x_g = Sensor2xRaw + Sensor2x_offset
    Sensor2y_g = Sensor2yRaw + Sensor2y_offset
    Sensor2z_g = Sensor2zRaw*Cd + Sensor2z_offset
    print('data calibrated')
    print('data orientated + converted')

#If needed trim the arrays for bad/false/faulty data
#Trimming is based on # of samples, not in time array
#if Trim_data == Yes: 
if Trim_Lower > 0 or Trim_Upper > 0: 
    trim_beg = int(Trim_Lower*time_Hz) #trim beginning of array
    trim_end = int(Trim_Upper*time_Hz) #trim end of array
    print(trim_beg)
    print(trim_end)
    time = time[trim_beg:trim_end]
    print( time[::-1])
    Sensor1x_g = Sensor1x_g[trim_beg:trim_end]
    Sensor1y_g = Sensor1y_g[trim_beg:trim_end]
    Sensor1z_g = Sensor1z_g[trim_beg:trim_end]
    Sensor2x_g = Sensor2x_g[trim_beg:trim_end]
    Sensor2y_g = Sensor2y_g[trim_beg:trim_end]
    Sensor2z_g = Sensor2z_g[trim_beg:trim_end]
    trim_size = trim_end - trim_beg
    f.write('Trim begins at (sec) =' '%.2f'%  Trim_Lower)
    f.write('\n')
    f.write('Trim ends at (sec) =' '%.2f'% Trim_Upper)
    f.write('\n')
    print('Size of array after trim =', trim_size,'samples')
    f.write('Size of array after trim (samples) =' '%.2f'% trim_size)
    f.write('\n')
else:
    print('Data not trimmed')
    print('Size of Data Array/s =',len(time))
    f.write('Data not trimmed')
    f.write('\n')
    Length_time = len(time)
    f.write('Size of Data Array/s =' '%.2f'% Length_time)
    f.write('\n')

#Sensor 1 - upper sensor, at Handlebar/Seatpost/etc.
#Sensor 2 - lower sensor, at Axle/s
print('Sensor 1 = upper sensor (Hbar/seatpost/etc.)')
print('Sensor 2 = lower sensor (at Axle)')
f.write('Sensor 1 = upper sensor (Hbar/seatpost/etc.)')
f.write('\n')
f.write('Sensor 2 = lower sensor (at Axle)')
f.write('\n')

#Analyze the Arrays ============================================================================================================
# Find Average of sensor data
S1ymean = np.mean(Sensor1y_g)
S2ymean = np.mean(Sensor2y_g)    
S1xmean = np.mean(Sensor1x_g)
S2xmean = np.mean(Sensor2x_g)    
#S1zmean = np.mean(Sensor1z_g)
#S2zmean = np.mean(Sensor2z_g)  
print('Upper Sensor Y Axis Average = ', '%.5f'% S1ymean, 'm/s^2')
print('Lower Sensor Y Axis Average = ', '%.5f'% S2ymean, 'm/s^2')
print('Upper Sensor X Axis Average = ', '%.5f'% S1xmean, 'm/s^2')
print('Lower Sensor X Axis Average = ', '%.5f'% S2xmean, 'm/s^2')
#print('Sensor 1 Z Axis Average = ', '%.5f'% S1zmean, 'm/s^2')
#print('Sensor 2 Z Axis Average = ', '%.5f'% S2zmean, 'm/s^2')
f.write('Upper Sensor Y Axis Average (m/s^2) = ' '%.5f'% S1ymean)
f.write('\n')
f.write('Lower Sensor Y Axis Average (m/s^2) = ' '%.5f'% S2ymean)
f.write('\n')
f.write('Upper Sensor X Axis Average (m/s^2) = ' '%.5f'% S1xmean)
f.write('\n')
f.write('Lower Sensor X Axis Average (m/s^2) = ' '%.5f'% S2xmean)
f.write('\n')
#f.write('Sensor 1 Z Axis Average (m/s^2) = ' '%.5f'% S1zmean)
#f.write('\n')
#f.write('Sensor 2 Z Axis Average (m/s^2) = ' '%.5f'% S2zmean)
#f.write('\n')

# Find MAX of sensor data
S1yMAX = np.amax(Sensor1y_g)
S2yMAX = np.amax(Sensor2y_g)    
S1xMAX = np.amax(Sensor1x_g)
S2xMAX = np.amax(Sensor2x_g)    
#S1zMAX = np.amax(Sensor1z_g)
#S2zMAX = np.amax(Sensor2z_g)    
print('Upper Sensor Y Axis MAX = ', '%.5f'% S1yMAX, 'm/s^2')
print('Lower Sensor Y Axis MAX = ', '%.5f'% S2yMAX, 'm/s^2')
print('Upper Sensor X Axis MAX = ', '%.5f'% S1xMAX, 'm/s^2')
print('Lower Sensor X Axis MAX = ', '%.5f'% S2xMAX, 'm/s^2')
#print('Sensor 1 Z Axis MAX = ', '%.5f'% S1zMAX, 'm/s^2')
#print('Sensor 2 Z Axis MAX = ', '%.5f'% S2zMAX, 'm/s^2')
f.write('Upper Sensor Y Axis MAX (m/s^2) = ' '%.5f'% S1yMAX)
f.write('\n')
f.write('Lower Sensor Y Axis MAX (m/s^2) = ' '%.5f'% S2yMAX)
f.write('\n')
f.write('Upper Sensor X Axis MAX (m/s^2) = ' '%.5f'% S1xMAX)
f.write('\n')
f.write('Lower Sensor X Axis MAX (m/s^2) = ' '%.5f'% S2xMAX)
f.write('\n')
#f.write('Sensor 1 Z Axis MAX (m/s^2) = ' '%.5f'% S1zMAX)
#f.write('\n')
#f.write('Sensor 2 Z Axis MAX (m/s^2) = ' '%.5f'% S2zMAX)
#f.write('\n')

# Find MIN of sensor data
S1yMIN = np.amin(Sensor1y_g)
S2yMIN = np.amin(Sensor2y_g)    
S1xMIN = np.amin(Sensor1x_g)
S2xMIN = np.amin(Sensor2x_g)    
#S1zMIN = np.amin(Sensor1z_g)
#S2zMIN = np.amin(Sensor2z_g)    
print('Upper Sensor Y Axis MIN = ', '%.5f'% S1yMIN, 'm/s^2')
print('Lower Sensor Y Axis MIN = ', '%.5f'% S2yMIN, 'm/s^2')
print('Upper Sensor X Axis MIN = ', '%.5f'% S1xMIN, 'm/s^2')
print('Lower Sensor X Axis MIN = ', '%.5f'% S2xMIN, 'm/s^2')
#print('Sensor 1 Z Axis MIN = ', '%.5f'% S1zMIN, 'm/s^2')
#print('Sensor 2 Z Axis MIN = ', '%.5f'% S2zMIN, 'm/s^2')
f.write('Upper Sensor Y Axis MIN (m/s^2) = ' '%.5f'% S1yMIN)
f.write('\n')
f.write('Lower Sensor Y Axis MIN (m/s^2) = ' '%.5f'% S2yMIN)
f.write('\n')
f.write('Upper Sensor X Axis MIN (m/s^2) = ' '%.5f'% S1xMIN)
f.write('\n')
f.write('Lower Sensor X Axis MIN (m/s^2) = ' '%.5f'% S2xMIN)
f.write('\n')
#f.write('Sensor 1 Z Axis MIN (m/s^2) = ' '%.5f'% S1zMIN)
#f.write('\n')
#f.write('Sensor 2 Z Axis MIN (m/s^2) = ' '%.5f'% S2zMIN)
#f.write('\n')


# Transmissibility and Isolation %
#Trans_x = (S1xmean/S2xmean)
#Trans_y = (S1ymean/S2ymean)
#Trans_z = (S1zmean/S2zmean)
#print('X Axis Transmissibility =', '%.5f'%  Trans_x,)
#print('Y Axis Transmissibility =', '%.5f'%  Trans_y,)
#print('Z Axis Transmissibility =', '%.5f'%  Trans_z,)

#Additonal statistics
S1yP2P = S1yMAX - S1yMIN
S2yP2P = S2yMAX - S1yMIN
print('Upper Sensor Y Total Range =', '%.5f'%  S1yP2P, 'm/s^2')
print('Lower Sensor Y Total Range =', '%.5f'%  S2yP2P, 'm/s^2')
#S1vs2P2PY = ((S2yP2P-S1yP2P)/S2yP2P)*100
#print('%.3f'% S1vs2P2PY,'% Reduction Y Axis P2P')
S1xP2P = S1xMAX - S1xMIN
S2xP2P = S2xMAX - S1xMIN
print('Upper Sensor X Total Range =', '%.5f'%  S1xP2P, 'm/s^2')
print('Lower Sensor X Total Range =', '%.5f'%  S2xP2P, 'm/s^2')
#S1vs2P2PX = ((S2xP2P-S1xP2P)/S2xP2P)*100
#print('%.3f'% S1vs2P2PX,'% Reduction X Axis P2P')
#S1zP2P = S1zMAX - S1zMIN
#S2zP2P = S2zMAX - S1zMIN
#print('Sensor 1 Z Total Range =', '%.5f'%  S1zP2P, 'm/s^2')
#print('Sensor 2 Z Total Range =', '%.5f'%  S2zP2P, 'm/s^2')
#S1vs2P2PZ = ((S2zP2P-S1zP2P)/S2zP2P)*100
#print('%.3f'% S1vs2P2PZ,'% Reduction Z Axis P2P')
f.write('Upper Sensor Y Total Range (m/s^2) =' '%.5f'%  S1yP2P)
f.write('\n')
f.write('Lower Sensor Y Total Range (m/s^2) =' '%.5f'%  S2yP2P)
f.write('\n')
f.write('Upper Sensor X Total Range (m/s^2) =' '%.5f'%  S1xP2P)
f.write('\n')
f.write('Lower Sensor X Total Range (m/s^2) =' '%.5f'%  S2xP2P)
f.write('\n')

#RMS 
S1xRMS = math.sqrt(np.sum(Sensor1x_g**2)/len(time))
S2xRMS = math.sqrt(np.sum(Sensor2x_g**2)/len(time))
print('Upper Sensor X Axis RMS =', '%.5f'%  S1xRMS, 'm/s^2')
print('Lower Sensor X Axis RMS =', '%.5f'%  S2xRMS, 'm/s^2')
f.write('Upper Sensor X Axis RMS (m/s^2) =' '%.5f'%  S1xRMS)
f.write('\n')
f.write('Lower Sensor X Axis RMS (m/s^2) =' '%.5f'%  S2xRMS)
f.write('\n')
S1vs2RMSX = ((S1xRMS-S2xRMS)/S2xRMS)*100
if S1vs2RMSX > 0:
    print('%.3f'% abs(S1vs2RMSX),'% Increase RMS X Axis')
    f.write('Increase RMS X Axis  =' '%.3f'% abs(S1vs2RMSX))
    f.write('\n')
else: 
    print('%.3f'% abs(S1vs2RMSX),'% Reduction RMS X Axis') 
    f.write('Reduction RMS X Axis  =' '%.3f'% abs(S1vs2RMSX))
    f.write('\n')
S1yRMS = math.sqrt(np.sum(Sensor1y_g**2)/len(time))
S2yRMS = math.sqrt(np.sum(Sensor2y_g**2)/len(time))
print('Upper Sensor Y Axis RMS =', '%.5f'%  S1yRMS, 'm/s^2')
print('Lower Sensor Y Axis RMS =', '%.5f'%  S2yRMS, 'm/s^2')
f.write('Upper Sensor Y Axis RMS (m/s^2) =' '%.5f'%  S1yRMS)
f.write('\n')
f.write('Lower Sensor Y Axis RMS (m/s^2) =' '%.5f'%  S2yRMS)
f.write('\n')
S1vs2RMSY = ((S1yRMS-S2yRMS)/S2yRMS)*100
if S1vs2RMSY > 0:
    print('%.3f'% abs(S1vs2RMSY),'% Increase RMS Y Axis')
    f.write('Increase RMS Y Axis =' '%.3f'% abs(S1vs2RMSY))
    f.write('\n')
else:  
    print('%.3f'% abs(S1vs2RMSY),'% Reduction RMS Y Axis')   
    f.write('Reduction RMS Y Axis =' '%.3f'% abs(S1vs2RMSY))
    f.write('\n')
#S1zRMS = math.sqrt(np.sum(Sensor1z_g**2)/len(time))
#S2zRMS = math.sqrt(np.sum(Sensor2z_g**2)/len(time))
#print('Sensor 1 Z Axis RMS =', '%.5f'%  S1zRMS, 'G')
#print('Sensor 2 Z Axis RMS =', '%.5f'%  S2zRMS, 'G')
#S1vs2RMSZ = ((S2zRMS-S1zRMS)/S2zRMS)*100
#print('%.3f'% S1vs2RMSZ,'% Reduction RMS Z Axis')

# Absolute Values of Axis Data [array]
Sensor1x_abs = np.absolute(Sensor1x_g)
Sensor1y_abs = np.absolute(Sensor1y_g)
Sensor1z_abs = np.absolute(Sensor1z_g)
Sensor2x_abs = np.absolute(Sensor2x_g)
Sensor2y_abs = np.absolute(Sensor2y_g)
Sensor2z_abs = np.absolute(Sensor2z_g)

#Integration of Data using abs values
# Single integration, to velocity
Sensor1x_int1 = integrate.simps(Sensor1x_abs, time)
print('Upper Sensor X Axis Accel. Integral','%.5f'% Sensor1x_int1)
Sensor2x_int1 = np.sum(integrate.trapz(Sensor2x_abs, time))
print('Upper Sensor X Axis Accel. Integral','%.5f'% Sensor2x_int1)
S1vs2INTX = ((Sensor1x_int1-Sensor2x_int1)/Sensor2x_int1)*100 
if S1vs2INTX > 0:
    print('%.3f'% abs(S1vs2INTX),'% Increase Integral X Axis')
else: print('%.3f'% abs(S1vs2INTX),'% Reduction Integral X Axis') 
print('%.5f'% S1vs2INTX,'% Reduction Integral X Axis')
Sensor1y_int1 = integrate.simps(Sensor1y_abs, time)
print('Upper Sensor Y Axis Accel. Integral','%.5f'% Sensor1y_int1)
Sensor2y_int1 = integrate.simps(Sensor2y_abs, time)
print('Lower Sensor Y Axis Accel. Integral','%.5f'% Sensor2y_int1)
S1vs2INTY = ((Sensor1y_int1-Sensor2y_int1)/Sensor2y_int1)*100
if S1vs2INTY > 0:
    print('%.3f'% abs(S1vs2INTY),'% Increase Integral Y Axis')
else: print('%.3f'% abs(S1vs2INTY),'% Reduction Integral Y Axis') 
#Sensor1z_int1 = integrate.simps(Sensor1z_abs, time)
#print('Sensor 1 Z Axis Accel. Integral','%.5f'% Sensor1z_int1)
#Sensor2z_int1 = integrate.simps(Sensor2z_abs, time)
#print('Sensor 2 Z Axis Accel. Integral','%.5f'% Sensor2z_int1)
#S1vs2INTZ = ((Sensor2z_int1-Sensor1z_int1)/Sensor2z_int1)*100 
#print('%.5f'% S1vs2INTZ,'% Reduction Integral Z Axis')

#Acceleration values in m/s^2
Gravity = 9.80665 
Sensor1x_Grav = Sensor1x_g
Sensor1y_Grav = Sensor1y_g
Sensor1z_Grav = Sensor1z_g
Sensor2x_Grav = Sensor2x_g
Sensor2y_Grav = Sensor2y_g
Sensor2z_Grav = Sensor2z_g

#make some plots==========================================================================================
#=========================================================================================================

# Magnitude Spectrum Density =============================================================================
plt.figure()
Fs = int(1/time_int_avg)
plt.magnitude_spectrum(Sensor2y_g, Fs=Fs, label="Lower Sensor (Axle)", color='r')
plt.magnitude_spectrum(Sensor1y_g, Fs=Fs, label='Upper Sensor (Hbar/Seatpost)')
plt.xlim([0, 200])
plt.ylim([0, 4])
plt.title('Magnitude Spectrum - Y Axis of {}'.format(FN))
plt.legend()
plt.grid()
plt.savefig('{}_MSD' .format(FN), dpi=1000, papertype='letter', orientation='landscape', bbox_inches='tight')

# Histogram ==============================================================================================
# Array of ISO 2631-1 Discomfort Scale
Discomfort = ['Not Uncomfortable', 'A little Uncomfortable', 'Fairly Uncomfortable', 'Uncomfortable', 'Very Uncomfortable', 'Extremely Uncomfortable']
Discomfort_Vals = [0.315, 0.5, 0.63, 0.8, 1.0, 1.25, 1.6, 2, 2.5]

Sensor1y_g_norm = (Sensor1y_g + Gravity)
Sensor2y_g_norm = (Sensor2y_g + Gravity) 

Not_Un_1y = 0
AlitUn_1y = 0
Fair_Un_1y = 0
Uncomf_1y = 0
VeryUn_1y = 0
ExtUncom_1y = 0
for i in Sensor1y_g_norm:
    if i < Discomfort_Vals[0]:
        Not_Un_1y = Not_Un_1y + 1
    elif Discomfort_Vals[0] > i < Discomfort_Vals[2]:
        AlitUn_1y = AlitUn_1y + 1
    elif Discomfort_Vals[1] > i < Discomfort_Vals[4]:
        Fair_Un_1y = Fair_Un_1y + 1 
    elif Discomfort_Vals[3] > i < Discomfort_Vals[6]:
        Uncomf_1y = Uncomf_1y + 1 
    elif Discomfort_Vals[5] > i < Discomfort_Vals[8]:
        VeryUn_1y = VeryUn_1y + 1 
    elif Discomfort_Vals[7] < i :
        ExtUncom_1y = ExtUncom_1y + 1 
ISO_Reactions_1y = [Not_Un_1y, AlitUn_1y, Fair_Un_1y, Uncomf_1y, VeryUn_1y, ExtUncom_1y]

Not_Un_2y = 0
AlitUn_2y = 0
Fair_Un_2y = 0
Uncomf_2y = 0
VeryUn_2y = 0
ExtUncom_2y = 0
for i in Sensor2y_g_norm:
    if i < Discomfort_Vals[0]:
        Not_Un_2y = Not_Un_2y + 1
    elif Discomfort_Vals[0] > i < Discomfort_Vals[2]:
        AlitUn_2y = AlitUn_2y + 1
    elif Discomfort_Vals[1] > i < Discomfort_Vals[4]:
        Fair_Un_2y = Fair_Un_2y + 1 
    elif Discomfort_Vals[3] > i < Discomfort_Vals[6]:
        Uncomf_2y = Uncomf_2y + 1 
    elif Discomfort_Vals[5] > i < Discomfort_Vals[8]:
        VeryUn_2y = VeryUn_2y + 1 
    elif Discomfort_Vals[7] < i :
        ExtUncom_2y = ExtUncom_2y + 1 
ISO_Reactions_2y = [Not_Un_2y, AlitUn_2y, Fair_Un_2y, Uncomf_2y, VeryUn_2y, ExtUncom_2y]
plt.figure()
plt.bar(Discomfort, ISO_Reactions_2y, width=0.4, align='center', color='r', label='Axle Sensor(Y) ')  #color= ['b','g','y','c','m','r']
plt.bar(Discomfort, ISO_Reactions_1y, width=0.4, align='edge', color= 'b', label='Handlebar or Seatpost Sensor(Y)')  #color= ['b','g','y','c','m','r']

plt.title('Comfort Reactions to vibrations {}'.format(FN))
plt.yscale('log')
plt.xlabel('Acceleration (m/s^2) in ISO Reaction Bins')
plt.ylabel('Frequency - log scale')
plt.xticks(rotation=30, fontsize=8)
plt.legend()
#plt.grid()
plt.savefig('{}_Histogram' .format(FN), dpi=1000, papertype='letter', orientation='landscape', bbox_inches='tight')

# Values/Count of comfort categories for Sensor 1 and Sensor 2
f.write('Quantity of "A Little Uncomfortable" Upper Sensor =' '%.3f'% AlitUn_1y)
f.write('\n')
f.write('Quantity of "A Little Uncomfortable" Lower Sensor =' '%.3f'% AlitUn_2y)
f.write('\n')
f.write('Quantity of "Not Uncomfortable" Upper Sensor =' '%.3f'% Not_Un_1y)
f.write('\n')
f.write('Quantity of "Not Uncomfortable" Lower Sensor =' '%.3f'% Not_Un_2y)
f.write('\n')
f.write('Quantity of "Fairly Uncomfortable" Upper Sensor =' '%.3f'% Fair_Un_1y)
f.write('\n')
f.write('Quantity of "Fairly Uncomfortable" Lower Sensor =' '%.3f'% Fair_Un_2y)
f.write('\n')
f.write('Quantity of "Uncomfortable" Upper Sensor =' '%.3f'% Uncomf_1y)
f.write('\n')
f.write('Quantity of "Uncomfortable" Lower Sensor =' '%.3f'% Uncomf_2y)
f.write('\n')
f.write('Quantity of "Very Uncomfortable" Upper Sensor =' '%.3f'% VeryUn_1y)
f.write('\n')
f.write('Quantity of "Very Uncomfortable" Lower Sensor =' '%.3f'% VeryUn_2y)
f.write('\n')
f.write('Quantity of "Extremely Uncomfortable" Upper Sensor =' '%.3f'% ExtUncom_1y)
f.write('\n')
f.write('Quantity of "Extremely Uncomfortable" Lower Sensor =' '%.3f'% ExtUncom_2y)
f.write('\n')

# % Difference of comfort categories Sensor 1 vs Sensor 2 
if AlitUn_2y > 0:
    S1vsS2AlitUnY = ((AlitUn_1y - AlitUn_2y)/AlitUn_2y)*100
else:
    S1vsS2AlitUnY = 0
if S1vsS2AlitUnY >= 0:
    print('A little Uncomfortable - Percent Difference = ' '%.3f'% S1vsS2AlitUnY)
    f.write('A little Uncomfortable - Percent Difference = ' '%.3f'% S1vsS2AlitUnY)
    f.write('\n')

if Not_Un_2y > 0:
    S1vsS2Not_UnY = ((Not_Un_1y - Not_Un_2y)/Not_Un_2y)*100
else:
    S1vsS2Not_UnY = 0
print('Not Uncomfortable - Percent Difference = ' '%.3f'% S1vsS2Not_UnY)
f.write('Not Uncomfortable - Percent Difference = ' '%.3f'% S1vsS2Not_UnY)
f.write('\n')

if Fair_Un_2y > 0:
    S1vsS2Fair_UnY = ((Fair_Un_1y - Fair_Un_2y)/Fair_Un_2y)*100
else:
    S1vsS2Fair_UnY = 0
print('Fairly Uncomfortable - Percent Difference = ' '%.3f'% S1vsS2Fair_UnY)
f.write('Fairly Uncomfortable - Percent Difference = ' '%.3f'% S1vsS2Fair_UnY)
f.write('\n')

if Uncomf_2y > 0:
    S1vsS2Uncomf_Y = ((Uncomf_1y - Uncomf_2y)/Uncomf_2y)*100
else:
    S1vsS2Uncomf_Y = 0
print('Uncomfortable - Percent Difference = ' '%.3f'% S1vsS2Uncomf_Y)
f.write('Uncomfortable - Percent Difference = ' '%.3f'% S1vsS2Uncomf_Y)
f.write('\n')

if VeryUn_2y > 0:
    S1vsS2VeryUn_Y = ((VeryUn_1y - VeryUn_2y)/VeryUn_2y)*100
else:
    S1vsS2VeryUn_Y = 0
print('Very Uncomfortable - Percent Difference = ' '%.3f'% S1vsS2VeryUn_Y)
f.write('Very Uncomfortable - Percent Difference = ' '%.3f'% S1vsS2VeryUn_Y)
f.write('\n')

if ExtUncom_2y > 0:
    S1vsS2ExtUncom_Y = ((ExtUncom_1y - ExtUncom_2y)/ExtUncom_2y)*100
else:
    S1vsS2ExtUncom_Y = 0
print('Extremely Uncomfortable - Percent Difference = ' '%.3f'% S1vsS2ExtUncom_Y)
f.write('Extremely Uncomfortable - Percent Difference = ' '%.3f'% S1vsS2ExtUncom_Y)
f.write('\n')

#FFT=====================================================================
plt.figure()
# Set size of array for FFT (time and data arrays must equal)
X = 1
N = 0
while N < len(time):
    N =2**X  #N must be a BASE-2 number less than total array size 
    X += 1
X-= 2
N =2**X # Size of array for FFT
print('X=', X)
print('FFT array size =', N)
f.write('FFT array size =' '%.3f'% N)
f.write('\n')

#FFT of Data
Fs = int(1/time_int_avg)
print(Fs, 'Hz Sample Rate for FFT')
print('N/2 =', int(N/2))
f.write('Nyquist Frequency =' '%.3f'% int(N/2))
f.write('\n')
frequency = np.linspace(0.0, int(Fs/2), int(N/2))  #X Axis Frequency range
freq_data_1 = np.fft.fft(Sensor1y_Grav[0:N])  #FFT of Sensor[x]
freq_data_2 = np.fft.fft(Sensor2y_Grav[0:N])  #FFT of Sensor[y]
fft_1 = 2/N*np.abs(freq_data_1[0:int(N/2)]) # Take abs value of Sensor[x] 
fft_2 = 2/N*np.abs(freq_data_2[0:int(N/2)]) # Take abs value of Sensor[y] 

# plot results on an x-y scatter plot
plt.plot(frequency, fft_2, label = "Lower Sensor (Axle)", color='r')
plt.plot(frequency, fft_1, label = 'Upper Sensor (Hbar/Seatpost)')
plt.title('FFT - Y Axis of {}'.format(FN))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude(m/s^2)')
plt.xlim([0, 100])
plt.ylim([0, 4])
plt.legend()
plt.grid()
plt.savefig('{}_FFT'  .format(FN), dpi=1000, papertype='letter', orientation='landscape', bbox_inches='tight')

# PSD =============================================================================
plt.figure()
#plt.psd(Sensor2y_abs, Fs= Fs, label = "Axle Sensor", color ='r')
#plt.psd(Sensor1y_abs, Fs = Fs, label = 'Handlebar Sensor')
plt.psd(Sensor2y_g, Fs= Fs, label = "Lower Sensor (Axle)", color ='r')
plt.psd(Sensor1y_g, Fs = Fs, label = 'Upper Sensor (Hbar/Seatpost)')
plt.title('Power Spectral Density - Y Axis of {}'.format(FN))
#f, Pxx_den = signal.welch(Sensor1y_g, Fs)
#plt.semilogy(f, Pxx_den)
plt.xlim([0, 200])
#plt.ylim([0.5e-3, 1])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [(m/s^2)^2/Hz]')
plt.legend()
plt.grid()
plt.savefig('{}_PSD'  .format(FN), dpi=1000, papertype='letter', orientation='landscape', bbox_inches='tight')

#Acceleration plots vs. time======================================================
#plt.figure(1)
fig, (x1, y1, z1) = plt.subplots(1,3)  #Create one Figure with 3 plots
fig.subplots_adjust(hspace=0.5) # extra space between plots
fig.suptitle("Acceleration over Time {}".format(FN))  # Give the figure a title
x1 = plt.subplot(311)
x1.plot(time, Sensor2x_g, label = 'Lower Sensor (Axle)', color='r')
x1.plot(time, Sensor1x_g, label = 'Upper Sensor (Hbar/Seatpost)')
x1.set_ylim(-150,150)
#x1.set_xlim(0,)
#x1.axis([0, 230, -10, 10])  #Trim X Axis to show only good ride data
x1.set_title('Acceleration (X Axis)', fontsize = 'small')
#x1.set_xlabel('Time(Sec)')
x1.set_ylabel('m/s^2')
#x1.legend(fontsize = 'x-small')
fig.legend(bbox_to_anchor=(0.2,0.1, 0.5, 1.5), loc='center left',
           ncol=2, borderaxespad=0., fontsize='xx-small')
x1.grid()

y1 = plt.subplot(312)
y1.plot(time, Sensor2y_g, label = 'Lower Sensor (Axle)', color='r')
y1.plot(time, Sensor1y_g, label = 'Upper Sensor (Hbar/Seatpost)')
y1.set_ylim(-150,150)
#y1.axis([0, 230, -10, 10])  #Trim X Axis to show only good ride data
y1.set_title('Acceleration (Y Axis)', fontsize = 'small')
#y1.set_xlabel('Time(Sec)')
y1.set_ylabel('m/s^2')
#y1.legend()
y1.grid()

z1 = plt.subplot(313)
z1.plot(time, Sensor2z_g, label = 'Lower Sensor (Axle)', color='r')
z1.plot(time, Sensor1z_g, label = 'Upper Sensor (Hbar/Seatpost)')
z1.set_ylim(-150,150)
#z1.axis([0, 230, -10, 10])  #Trim X Axis to show only good ride data
z1.set_title('Acceleration (Z Axis)', fontsize = 'small')
z1.set_xlabel('Time(Sec)', fontsize = 'small')
z1.set_ylabel('m/s^2')
#z1.legend()
z1.grid()
plt.savefig('{}_TimeDomain' .format(FN), dpi=1000, papertype='letter', orientation='landscape', bbox_inches='tight')

f.close()
#Show all plots now  
plt.show()


