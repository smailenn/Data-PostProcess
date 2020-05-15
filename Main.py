#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math 
from numpy import savetxt
from scipy import stats
from scipy import signal
import os

# what file do you want?
File_path = 'C:\\Users\\smailen\\OneDrive - Quality Bicycle Products\\Vibration Analysis\\Waxwing test\\WAXWING01.txt'

#DTS or Arduino Data (Using LIS3dh)
Arduino = 'Arduino'
DTS = 'DTS'
No = None
Yes = None
Sensor_select = 'Arduino'
Trim_data = 'Yes'

#Note this is typical layout!!!!!
#Sensor 1 - upper sensor, at Handlebar
#Sensor 2 - lower sensor, at Axle

#read the data into an array from text file
if Sensor_select == Arduino:
    dataframe = pd.read_csv(File_path, sep='\t')
    print('Arduino data selected')
    filename = os.path.basename(File_path)
    FN = os.path.splitext(filename)[0]
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
    accel_range = 16 #What Range in G 2,4,8,16 g
    accel_raw_scale_neg = -32767 #raw value range negative values
    accel_raw_scale_pos = 32768 #raw value range positive values
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
time_int_avg = stats.trim_mean(time_int,0.1) # Trim time intervals of SD lag points, large time intervals during a SD write, this is scalar
print('Time Interval Average =', "%.8f"% time_int_avg,'S')
time_Hz = 1/time_int_avg
print('Sample Rate =', '%.2f'% time_Hz, 'Hz')
#savetxt('time_int.csv', time_int, delimiter=',')  # save Time Interval Array as CSV for review if needed
#print(len(time))

#Calibrate, orientate, and Normalize readings to G's
#Calculate offset values from calibration test at beginning of test runs, ideally done at same time as testing
if Sensor_select == Arduino: 
    accel_g_unit = (accel_raw_scale_pos - accel_raw_scale_neg)/accel_range
    Sensor1x_offset = 63.5 
    Sensor1y_offset = -692 
    Sensor1z_offset = -22.6
    Sensor2x_offset = 12.6
    Sensor2y_offset = -769
    Sensor2z_offset = 98
    Calibrate_direction = -1 #multiply raw data by this to change trajectory to normal of drivetrain
    print('data calibrated')
    #Orientate sensors, calibrate, and convert to G's
    Sensor1x_g = ((Sensor1xRaw + Sensor1x_offset)*(accel_range/accel_raw_scale_pos))*Calibrate_direction
    Sensor1y_g = ((Sensor1yRaw + Sensor1y_offset)*(accel_range/accel_raw_scale_pos))+1
    Sensor1z_g = (Sensor1zRaw + Sensor1z_offset)*(accel_range/accel_raw_scale_pos)
    Sensor2x_g = (Sensor2xRaw + Sensor2x_offset)*(accel_range/accel_raw_scale_pos)
    Sensor2y_g = ((Sensor2yRaw + Sensor2y_offset)*(accel_range/accel_raw_scale_pos))+1
    Sensor2z_g = ((Sensor2zRaw + Sensor2z_offset)*(accel_range/accel_raw_scale_pos))*Calibrate_direction
    print('data orientated + converted')

#If needed trim the arrays for bad/false/faulty data
#Trimming is based on # of samples, not in time array
if Trim_data == 'Yes': 
    trim_beg = 0 #trim beginning of array
    trim_end = 300000  #trim end of array
    time = time[trim_beg:trim_end]
    Sensor1x_g = Sensor1x_g[trim_beg:trim_end]
    Sensor1y_g = Sensor1y_g[trim_beg:trim_end]
    Sensor1z_g = Sensor1z_g[trim_beg:trim_end]
    Sensor2x_g = Sensor2x_g[trim_beg:trim_end]
    Sensor2y_g = Sensor2y_g[trim_beg:trim_end]
    Sensor2z_g = Sensor2z_g[trim_beg:trim_end]
    trim_size = trim_end - trim_beg
    print('Size of array after trim =', trim_size,'samples')
else:
    print('Data not trimmed')
    print('Size of Array =',len(time))


#Analyze the Arrays ============================================================================================================
# Find Average of sensor data
S1ymean = np.mean(Sensor1y_g)
S2ymean = np.mean(Sensor2y_g)    
S1xmean = np.mean(Sensor1x_g)
S2xmean = np.mean(Sensor2x_g)    
S1zmean = np.mean(Sensor1z_g)
S2zmean = np.mean(Sensor2z_g)  
print('Sensor 1 Y Axis Average = ', '%.5f'% S1ymean, 'G')
print('Sensor 2 Y Axis Average = ', '%.5f'% S2ymean, 'G')
print('Sensor 1 X Axis Average = ', '%.5f'% S1xmean, 'G')
print('Sensor 2 X Axis Average = ', '%.5f'% S2xmean, 'G')
print('Sensor 1 Z Axis Average = ', '%.5f'% S1zmean, 'G')
print('Sensor 2 Z Axis Average = ', '%.5f'% S2zmean, 'G')

# Find MAX of sensor data
S1yMAX = np.amax(Sensor1y_g)
S2yMAX = np.amax(Sensor2y_g)    
S1xMAX = np.amax(Sensor1x_g)
S2xMAX = np.amax(Sensor2x_g)    
S1zMAX = np.amax(Sensor1z_g)
S2zMAX = np.amax(Sensor2z_g)    
print('Sensor 1 Y Axis MAX = ', '%.5f'% S1yMAX, 'G')
print('Sensor 2 Y Axis MAX = ', '%.5f'% S2yMAX, 'G')
print('Sensor 1 X Axis MAX = ', '%.5f'% S1xMAX, 'G')
print('Sensor 2 X Axis MAX = ', '%.5f'% S2xMAX, 'G')
print('Sensor 1 Z Axis MAX = ', '%.5f'% S1zMAX, 'G')
print('Sensor 2 Z Axis MAX = ', '%.5f'% S2zMAX, 'G')

# Find MIN of sensor data
S1yMIN = np.amin(Sensor1y_g)
S2yMIN = np.amin(Sensor2y_g)    
S1xMIN = np.amin(Sensor1x_g)
S2xMIN = np.amin(Sensor2x_g)    
S1zMIN = np.amin(Sensor1z_g)
S2zMIN = np.amin(Sensor2z_g)    
print('Sensor 1 Y Axis MIN = ', '%.5f'% S1yMIN, 'G')
print('Sensor 2 Y Axis MIN = ', '%.5f'% S2yMIN, 'G')
print('Sensor 1 X Axis MIN = ', '%.5f'% S1xMIN, 'G')
print('Sensor 2 X Axis MIN = ', '%.5f'% S2xMIN, 'G')
print('Sensor 1 Z Axis MIN = ', '%.5f'% S1zMIN, 'G')
print('Sensor 2 Z Axis MIN = ', '%.5f'% S2zMIN, 'G')

# Transmissibility and Isolation %
Trans_x = (S1xmean/S2xmean)
Trans_y = (S1ymean/S2ymean)
Trans_z = (S1zmean/S2zmean)
print('X Axis Transmissibility =', '%.5f'%  Trans_x,)
print('Y Axis Transmissibility =', '%.5f'%  Trans_y,)
print('Z Axis Transmissibility =', '%.5f'%  Trans_z,)

# Difference in arrays (vector)
# Looking at difference by vector, then averaging 
Xdiff = Sensor2x_g - Sensor1x_g
Ydiff = Sensor2y_g - Sensor1y_g
Zdiff = Sensor2z_g - Sensor1z_g
Xdiff_avg = np.mean(Xdiff)
Ydiff_avg = np.mean(Ydiff)
Zdiff_avg = np.mean(Zdiff)
#print('Avg diff. of X Axis =', '%.5f'% Xdiff_avg,'G')
#print('Avg diff. of Y Axis =', '%.5f'% Ydiff_avg,'G')
#print('Avg diff. of Z Axis =', '%.5f'% Zdiff_avg,'G')
# looking at % difference of vectors
Xpd = np.mean(Xdiff/Sensor2x_g)*100  
Ypd = np.mean(Ydiff/Sensor2y_g)*100
Zpd = np.mean(Zdiff/Sensor2z_g)*100
#print('Avg %\ diff. of X Axis =', '%.5f'% Xpd,'G')
#print('Avg %\ diff. of Y Axis =', '%.5f'% Ypd,'G')
#print('Avg %\ diff. of Z Axis =', '%.5f'% Zpd,'G')

#Additonal statistics
#S1yMedian = np.median(Sensor1y_g)
#S2yMedian = np.median(Sensor2y_g)
#print('Sensor 1 Y Axis Median =', '%.5f'%  S1yMedian, 'G')
#print('Sensor 2 Y Axis Median =', '%.5f'%  S2yMedian, 'G')
S1yP2P = S1yMAX - S1yMIN
S2yP2P = S2yMAX - S1yMIN
print('Sensor 1 Y Total Range =', '%.5f'%  S1yP2P, 'G')
print('Sensor 2 Y Total Range =', '%.5f'%  S2yP2P, 'G')
S1vs2P2P = ((S2yP2P-S1yP2P)/S2yP2P)*100
print('%.3f'% S1vs2P2P,'% Reduction P2P')
#RMS 
S1yRMS = math.sqrt(np.sum(Sensor1y_g**2)/len(time))
S2yRMS = math.sqrt(np.sum(Sensor2y_g**2)/len(time))
print('Sensor 1 RMS =', '%.5f'%  S1yRMS, 'G')
print('Sensor 2 RMS =', '%.5f'%  S2yRMS, 'G')
S1vs2RMS = ((S2yRMS-S1yRMS)/S2yRMS)*100
print('%.3f'% S1vs2RMS,'% Reduction RMS')

#Integration of Data


#Create a Nice Table to see Scalar Values
plt.figure(1)
columns = ('Average(G)', 'MAX(G)', 'MIN(G)', 'Peak to Peak (G)', 'RMS')
rows = ('Sensor 1 X(Upper)', 'Sensor 2 X(Lower)', 'Sensor 1 Y(Upper)', 'Sensor 2 Y(Lower)', 'Sensor 1 Z(Upper)', 'Sensor 2 Z(Lower)')
data = [[S1xmean, S1xMAX, S1xMIN, 'S1xP2P', 'S1xRMS'],
        [S2xmean, S2xMAX, S2xMIN, 'S2xP2P', 'S2xRMS'],
        [S1ymean, S1yMAX, S1yMIN, S1yP2P, S1yRMS],
        [S2ymean, S2yMAX, S2yMIN, S2yP2P, S1yRMS],
        [S1zmean, S1zMAX, S1zMIN, 'S1zP2P', 'S1zRMS'],
        [S2zmean, S2zMAX, S2zMIN, 'S2zP2P', 'S2zRMS'],]
the_table = plt.table(cellText=data, loc='center', rowLabels=rows, rowLoc='center', colLabels=columns, colLoc='center') 
the_table.set_fontsize(14)
the_table.scale(1,4)
#the_table.axis(off)

data.to_csv()



#make some plots========================================================================================
#Acceleration plots vs. time
fig, (x1, y1, z1) = plt.subplots(3,2)  #Create one Figure with 3 plots
fig.subplots_adjust(hspace=0.5) # extra space between plots
fig.suptitle("Acceleration over Time {}".format(FN))  # Give the figure a title
x1 = plt.subplot(311)
x1.plot(time, Sensor2x_g, label = 'Axle Sensor - X', color='r')
x1.plot(time, Sensor1x_g, label = 'Handlebar Sensor - X')
x1.axis([0, 230, -10, 10])  #Trim X Axis to show only good ride data
x1.set_title('Acceleration (X Axis)')
x1.set_xlabel('Time(Sec)')
x1.set_ylabel('G')
x1.legend()
x1.grid()

y1 = plt.subplot(312)
y1.plot(time, Sensor2y_g, label = 'Axle Sensor - Y', color='r')
y1.plot(time, Sensor1y_g, label = 'Handlebar Sensor - Y')
y1.axis([0, 230, -10, 10])  #Trim X Axis to show only good ride data
y1.set_title('Acceleration (Y Axis)')
y1.set_xlabel('Time(Sec)')
y1.set_ylabel('G')
y1.legend()
y1.grid()

z1 = plt.subplot(313)
z1.plot(time, Sensor2z_g, label = 'Axle Sensor - Z', color='r')
z1.plot(time, Sensor1z_g, label = 'Handlebar Sensor - Z')
z1.axis([0, 230, -10, 10])  #Trim X Axis to show only good ride data
z1.set_title('Acceleration (Z Axis)')
z1.set_xlabel('Time(Sec)')
z1.set_ylabel('G')
z1.legend()
z1.grid()

#FFT========================================================================================
plt.figure(3)
# Set size of array for FFT (time and data arrays must equal)
X = 1
N = 0
while N < len(time):
    N =2**X  #N must be a BASE-2 number less than total array size 
    X += 1
X-= 2
N =2**X # Size of array for FFT
print('FFT array size =', N)

#FFT of Data
Fs = int(1/time_int_avg)
print(Fs, 'Hz Sample Rate for FFT')
print('N/2 =', int(N/2))
frequency = np.linspace(0.0, int(Fs/2), int(N/2))  #X Axis Frequency range
freq_data_1 = np.fft.fft(Sensor1y_g[0:N])  #FFT of Sensor[x]
freq_data_2 = np.fft.fft(Sensor2y_g[0:N])  #FFT of Sensor[y]
fft_1 = 2/N*np.abs(freq_data_1[0:int(N/2)]) # Take abs value of Sensor[x] 
fft_2 = 2/N*np.abs(freq_data_2[0:int(N/2)]) # Take abs value of Sensor[y] 

# plot results on an x-y scatter plot
plt.plot(frequency, fft_2, label = 'Axle Sensor(Y) ', color='r')
plt.plot(frequency, fft_1, label = 'Handlebar Sensor(Y)')
plt.title('FFT - Y Axis of {}'.format(FN))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude(G)')
plt.xlim([0, 400])
#plt.ylim([0, 300])
plt.legend()
plt.grid()

#Data in frequency range Acceleration-> Displacement
#Displacement_1Y = (freq_data_1 * 9810)/(2*pi()*)
#plt.plot(frequency, fft_2, label = 'Axle Sensor', color='r')
#plt.plot(frequency, fft_1, label = 'Handlebar Sensor')
#plt.title('FFT')
#plt.xlabel('Frequency (Hz)')
#plt.ylabel('Amplitude')
#plt.xlim([0, 400])
#plt.ylim([0, 300])
#plt.legend()
#plt.grid()

plt.figure(4)
plt.hist(Sensor2y_g, bins='auto', label = "Axle Sensor", color = 'r')
plt.hist(Sensor1y_g, bins='auto', label = 'Handlebar Sensor')
plt.title('Histogram - Y Axis of {}'.format(FN))
plt.xlabel('Acceleration (G)')
plt.ylabel('Quantity')
plt.xlim([-3,3])
plt.legend()
plt.grid()

plt.figure(5)
plt.psd(Sensor2y_g, Fs= Fs, label = "Axle Sensor", color ='r')
plt.psd(Sensor1y_g, Fs = Fs, label = 'Handlebar Sensor')
#plt.xlim([0, 400])
plt.title('Power Spectral Density - Y Axis of {}'.format(FN))
#f, Pxx_den = signal.welch(Sensor1y_g, Fs)
#plt.semilogy(f, Pxx_den)
#plt.ylim([0.5e-3, 1])
#plt.xlabel('frequency [Hz]')
#plt.ylabel('PSD [V**2/Hz]')
plt.legend()
plt.grid()

plt.figure(6)
plt.magnitude_spectrum(Sensor2y_g, Fs= Fs, label = "Axle Sensor", color ='r')
plt.magnitude_spectrum(Sensor1y_g, Fs = Fs, label = 'Handlebar Sensor')
plt.xlim([0, 400])
plt.title('Magnitude Spectrum - Y Axis of {}'.format(FN))
plt.legend()
plt.grid()

plt.show()

