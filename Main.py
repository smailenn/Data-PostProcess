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


# what file do you want to analyze?
#File_path = 'C:\\Users\\smailen\\OneDrive - Quality Bicycle Products\\Vibration Analysis\\Waxwing test\\WAXWING02.txt'  #Waxwing Data
File_path = 'C:\\Users\\smailen\\OneDrive - Quality Bicycle Products\\Vibration Analysis\\Warbird V3\\WARBIRD02.txt'  #Warbird Data
#File_path = 'C:\\Users\\smailen\\OneDrive - Quality Bicycle Products\\Vibration Analysis\\Cutthroat V2\\CUTTHROAT01.csv'  #  Cutthroat Data

#DTS or Arduino Data (Using LIS3dh)
Arduino = 'Arduino'
DTS = 'DTS'
No = None
Yes = None
Sensor_select = 'Arduino'  #DTS or Arduino
Trim_data = 'Yes'  #Yes or No
Magnitude_chart = 'No'
Histogram_chart = Yes

#Note this is typical layout, but you may have to adjust
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
    Cd = -1 #multiply raw data by this to change trajectory to match bicycle Orientation
    print('data calibrated')
    #Orientate sensors, calibrate, and convert to G's
    Sensor1x_g = ((Sensor1xRaw + Sensor1x_offset)*(accel_range/accel_raw_scale_pos))
    Sensor1y_g = (((Sensor1yRaw + Sensor1y_offset)*(accel_range/accel_raw_scale_pos))+1)*Cd
    Sensor1z_g = ((Sensor1zRaw + Sensor1z_offset)*(accel_range/accel_raw_scale_pos))*Cd
    Sensor2x_g = ((Sensor2xRaw + Sensor2x_offset)*(accel_range/accel_raw_scale_pos))
    Sensor2y_g = (((Sensor2yRaw + Sensor2y_offset)*(accel_range/accel_raw_scale_pos))+1)*Cd
    Sensor2z_g = ((Sensor2zRaw + Sensor2z_offset)*(accel_range/accel_raw_scale_pos))*Cd
    print('data orientated + converted')

#If needed trim the arrays for bad/false/faulty data
#Trimming is based on # of samples, not in time array
if Trim_data == 'Yes': 
    trim_beg = 0 #trim beginning of array
    trim_end = 250000  #trim end of array
    time = time[trim_beg:trim_end]
    print( time[::-1])
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
S1vs2P2PY = ((S2yP2P-S1yP2P)/S2yP2P)*100
print('%.3f'% S1vs2P2PY,'% Reduction Y Axis P2P')
S1xP2P = S1xMAX - S1xMIN
S2xP2P = S2xMAX - S1xMIN
print('Sensor 1 X Total Range =', '%.5f'%  S1xP2P, 'G')
print('Sensor 2 X Total Range =', '%.5f'%  S2xP2P, 'G')
S1vs2P2PX = ((S2xP2P-S1xP2P)/S2xP2P)*100
print('%.3f'% S1vs2P2PX,'% Reduction X Axis P2P')
S1zP2P = S1zMAX - S1zMIN
S2zP2P = S2zMAX - S1zMIN
print('Sensor 1 Z Total Range =', '%.5f'%  S1zP2P, 'G')
print('Sensor 2 Z Total Range =', '%.5f'%  S2zP2P, 'G')
S1vs2P2PZ = ((S2zP2P-S1zP2P)/S2zP2P)*100
print('%.3f'% S1vs2P2PZ,'% Reduction Z Axis P2P')

#RMS 
S1xRMS = math.sqrt(np.sum(Sensor1x_g**2)/len(time))
S2xRMS = math.sqrt(np.sum(Sensor2x_g**2)/len(time))
print('Sensor 1 X Axis RMS =', '%.5f'%  S1xRMS, 'G')
print('Sensor 2 X Axis RMS =', '%.5f'%  S2xRMS, 'G')
S1vs2RMSX = ((S2xRMS-S1xRMS)/S2xRMS)*100
print('%.3f'% S1vs2RMSX,'% Reduction RMS X Axis')
S1yRMS = math.sqrt(np.sum(Sensor1y_g**2)/len(time))
S2yRMS = math.sqrt(np.sum(Sensor2y_g**2)/len(time))
print('Sensor 1 Y Axis RMS =', '%.5f'%  S1yRMS, 'G')
print('Sensor 2 Y Axis RMS =', '%.5f'%  S2yRMS, 'G')
S1vs2RMSY = ((S2yRMS-S1yRMS)/S2yRMS)*100
print('%.3f'% S1vs2RMSY,'% Reduction RMS Y Axis')
S1zRMS = math.sqrt(np.sum(Sensor1z_g**2)/len(time))
S2zRMS = math.sqrt(np.sum(Sensor2z_g**2)/len(time))
print('Sensor 1 Z Axis RMS =', '%.5f'%  S1zRMS, 'G')
print('Sensor 2 Z Axis RMS =', '%.5f'%  S2zRMS, 'G')
S1vs2RMSZ = ((S2zRMS-S1zRMS)/S2zRMS)*100
print('%.3f'% S1vs2RMSZ,'% Reduction RMS Z Axis')

# Absolute Values of Axis Data
Sensor1x_abs = np.absolute(Sensor1x_g)
Sensor1y_abs = np.absolute(Sensor1y_g)
Sensor1z_abs = np.absolute(Sensor1z_g)
Sensor2x_abs = np.absolute(Sensor2x_g)
Sensor2y_abs = np.absolute(Sensor2y_g)
Sensor2z_abs = np.absolute(Sensor2z_g)

#Integration of Data using abs values
# Single integration, to velocity
Sensor1x_int1 = integrate.simps(Sensor1x_abs, time)
print('Sensor 1 X Axis Accel. Integral','%.5f'% Sensor1x_int1)
Sensor2x_int1 = np.sum(integrate.trapz(Sensor2x_abs, time))
print('Sensor 2 X Axis Accel. Integral','%.5f'% Sensor2x_int1)
S1vs2INTX = ((Sensor2x_int1-Sensor1x_int1)/Sensor2x_int1)*100 
print('%.5f'% S1vs2INTX,'% Reduction Integral X Axis')
Sensor1y_int1 = integrate.simps(Sensor1y_abs, time)
print('Sensor 1 Y Axis Accel. Integral','%.5f'% Sensor1y_int1)
Sensor2y_int1 = integrate.simps(Sensor2y_abs, time)
print('Sensor 2 Y Axis Accel. Integral','%.5f'% Sensor2y_int1)
S1vs2INTY = ((Sensor2y_int1-Sensor1y_int1)/Sensor2y_int1)*100 
print('%.5f'% S1vs2INTY,'% Reduction Integral Y Axis')
Sensor1z_int1 = integrate.simps(Sensor1z_abs, time)
print('Sensor 1 Z Axis Accel. Integral','%.5f'% Sensor1z_int1)
Sensor2z_int1 = integrate.simps(Sensor2z_abs, time)
print('Sensor 2 Z Axis Accel. Integral','%.5f'% Sensor2z_int1)
S1vs2INTZ = ((Sensor2z_int1-Sensor1z_int1)/Sensor2z_int1)*100 
print('%.5f'% S1vs2INTZ,'% Reduction Integral Z Axis')

#Acceleration values in m/s^2
Gravity = 9.80665 
Sensor1x_Grav = Sensor1x_g*Gravity
Sensor1y_Grav = Sensor1y_g*Gravity
Sensor1z_Grav = Sensor1z_g*Gravity
Sensor2x_Grav = Sensor2x_g*Gravity
Sensor2y_Grav = Sensor2y_g*Gravity
Sensor2z_Grav = Sensor2z_g*Gravity

# Array of ISO 2631-1 Discomfort Scale
Discomfort = ['Not Uncomfortable', 'A little Uncomfortable', 'Fairly Uncomfortable', 'Uncomfortable', 'Very Uncomfortable', 'Extremely Uncomfortable']
Discomfort_Vals = [-10, 0.315, 0.5, 0.8, 1.0, 1.6, 2.5, 5, 20]

S1y_G_Hist = np.histogram(Sensor1y_Grav, bins = Discomfort_Vals)

#make some plots========================================================================================
#===========================================================================================================
#Create a Nice Table to see scalar values
columns = ('Average(G)', 'MAX(G)', 'MIN(G)', 'Peak to Peak (G)', 'RMS', 'Integral')
rows = ('Sensor 1 X(Upper)', 'Sensor 2 X(Lower)', 'Sensor 1 Y(Upper)', 'Sensor 2 Y(Lower)', 'Sensor 1 Z(Upper)', 'Sensor 2 Z(Lower)')
results = np.array([['%.5f'% S1xmean,'%.5f'%  S1xMAX,'%.5f'%  S1xMIN,'%.5f'% S1xP2P,'%.5f'% S1xRMS,'%.5f'% Sensor1x_int1], 
        ['%.5f'% S2xmean,'%.5f'%  S2xMAX,'%.5f'%  S2xMIN,'%.5f'% S2xP2P,'%.5f'% S2xRMS, '%.5f'% Sensor2x_int1 ],
        ['%.5f'% S1ymean,'%.5f'%  S1yMAX,'%.5f'%  S1yMIN,'%.5f'% S1yP2P,'%.5f'% S1yRMS, '%.5f'% Sensor1y_int1],
        ['%.5f'% S2ymean,'%.5f'%  S2yMAX,'%.5f'%  S2yMIN,'%.5f'% S2yP2P,'%.5f'% S2yRMS, '%.5f'% Sensor2y_int1],
        ['%.5f'% S1zmean,'%.5f'%  S1zMAX,'%.5f'%  S1zMIN,'%.5f'% S1zP2P,'%.5f'% S1zRMS, '%.5f'% Sensor1z_int1],
        ['%.5f'% S2zmean,'%.5f'%  S2zMAX,'%.5f'%  S2zMIN,'%.5f'% S2zP2P,'%.5f'% S2zRMS, '%.5f'% Sensor2z_int1]])
df = pd.DataFrame(results, columns = columns, index = rows)
df.to_csv(FN, sep='\t')
fig, tx = plt.subplots()
fig.patch.set_visible(False)
tx.set_title(FN)
tx.axis('off')
tx.axis('tight')
tx.table(cellText=df.values, colLabels=df.columns, rowLabels=rows, loc='center')
fig.tight_layout()


#Acceleration plots vs. time======================================================
fig, (x1, y1, z1) = plt.subplots(3,2)  #Create one Figure with 3 plots
fig.subplots_adjust(hspace=0.5) # extra space between plots
fig.suptitle("Acceleration over Time {}".format(FN))  # Give the figure a title
x1 = plt.subplot(311)
x1.plot(time, Sensor2x_g, label = 'Axle Sensor - X', color='r')
x1.plot(time, Sensor1x_g, label = 'Handlebar or Seatpost Sensor - X')
x1.set_ylim(-10,10)
#x1.set_xlim(0,)
#x1.axis([0, 230, -10, 10])  #Trim X Axis to show only good ride data
x1.set_title('Acceleration (X Axis)')
x1.set_xlabel('Time(Sec)')
x1.set_ylabel('G')
x1.legend()
x1.grid()

y1 = plt.subplot(312)
y1.plot(time, Sensor2y_g, label = 'Axle Sensor - Y', color='r')
y1.plot(time, Sensor1y_g, label = 'Handlebar or Seatpost Sensor - Y')
y1.set_ylim(-10,10)
#y1.axis([0, 230, -10, 10])  #Trim X Axis to show only good ride data
y1.set_title('Acceleration (Y Axis)')
y1.set_xlabel('Time(Sec)')
y1.set_ylabel('G')
y1.legend()
y1.grid()

z1 = plt.subplot(313)
z1.plot(time, Sensor2z_g, label = 'Axle Sensor - Z', color='r')
z1.plot(time, Sensor1z_g, label = 'Handlebar or Seatpost Sensor - Z')
z1.set_ylim(-10,10)
#z1.axis([0, 230, -10, 10])  #Trim X Axis to show only good ride data
z1.set_title('Acceleration (Z Axis)')
z1.set_xlabel('Time(Sec)')
z1.set_ylabel('G')
z1.legend()
z1.grid()

#FFT=====================================================================
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
freq_data_1 = np.fft.fft(Sensor1y_Grav[0:N])  #FFT of Sensor[x]
freq_data_2 = np.fft.fft(Sensor2y_Grav[0:N])  #FFT of Sensor[y]
fft_1 = 2/N*np.abs(freq_data_1[0:int(N/2)]) # Take abs value of Sensor[x] 
fft_2 = 2/N*np.abs(freq_data_2[0:int(N/2)]) # Take abs value of Sensor[y] 

# plot results on an x-y scatter plot
plt.plot(frequency, fft_2, label = 'Axle Sensor(Y) ', color='r')
plt.plot(frequency, fft_1, label = 'Handlebar or Seatpost Sensor(Y)')
plt.title('FFT - Y Axis of {}'.format(FN))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude(G)')
plt.xlim([0, 100])
#plt.ylim([0, 300])
plt.legend()
plt.grid()

#Data in frequency range Acceleration-> Displacement
#Displacement_1Y = (freq_data_1 * 9810)/(2*pi()*)
#plt.plot(frequency, fft_2, label = 'Axle Sensor', color='r')
#plt.plot(frequency, fft_1, label = 'Handlebar or Seatpost Sensor')
#plt.title('FFT')
#plt.xlabel('Frequency (Hz)')
#plt.ylabel('Amplitude')
#plt.xlim([0, 400])
#plt.ylim([0, 300])
#plt.legend()
#plt.grid()

# Histogram =============================================================================
if Histogram_chart == Yes:
    plt.figure(4)
    #sns.set_style('whitegrid')
    #sns.kdeplot(Sensor1y_Grav, bw=0.1)
    #sns.kdeplot(Sensor2y_Grav, bw=0.1)
    
    #plt.plot(xs,Sensor1y_density(xs))
    
    #plt.hist(Sensor1y_Grav, bins= Discomfort_Vals, histtype='step', label = 'Handlebar or Seatpost Sensor')
    #plt.hist(Sensor2y_Grav, bins= Discomfort_Vals, histtype='step', label = "Axle Sensor", color = 'r')
    plt.hist(Sensor1y_Grav, histtype='step', label = 'Handlebar or Seatpost Sensor')
    plt.hist(Sensor2y_Grav, histtype='step', label = "Axle Sensor", color = 'r')
    plt.title('Histogram - Y Axis of {}'.format(FN))
    plt.xlabel('Acceleration (G)')
    plt.ylabel('Quantity')
    #plt.barh(6, performance, xerr=error, align='center')
    #plt.set_xticks(6)
    #plt.set_xticklabels(Discomfort)
    #plt.xticks(Discomfort_Vals, labels=Discomfort, rotation='vertical', text='wrap')
    plt.xticks(Discomfort_Vals)
    plt.xlim([0,20])
    #plt.ylim([0,10])
    plt.legend()
    plt.grid()
else: 
    print('No Histogram')
# PSD =============================================================================
plt.figure(5)
#plt.psd(Sensor2y_abs, Fs= Fs, label = "Axle Sensor", color ='r')
#plt.psd(Sensor1y_abs, Fs = Fs, label = 'Handlebar Sensor')
plt.psd(Sensor2y_g, Fs= Fs, label = "Axle Sensor", color ='r')
plt.psd(Sensor1y_g, Fs = Fs, label = 'Handlebar or Seatpost Sensor')
#plt.xlim([0, 400])
plt.title('Power Spectral Density - Y Axis of {}'.format(FN))
#f, Pxx_den = signal.welch(Sensor1y_g, Fs)
#plt.semilogy(f, Pxx_den)
#plt.ylim([0.5e-3, 1])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [G**2/Hz]')
plt.legend()
plt.grid()

# Absolute values over time =============================================================================
#splt.figure(6)
#plt.plot(time, Sensor2y_abs, label = "Axle Sensor", color ='r')
#plt.plot(time, Sensor1y_abs, label = "Handlebar or Seatpost Sensor")
#plt.xlim([0, 400])
#plt.title('Absolute Values - Y Axis of {}'.format(FN))
#plt.legend()
#plt.grid()

# Magnitude Spectrum Density =============================================================================
if Magnitude_chart == Yes:
    plt.figure(7)
    plt.magnitude_spectrum(Sensor2y_g, Fs= Fs, label = "Axle Sensor", color ='r')
    plt.magnitude_spectrum(Sensor1y_g, Fs = Fs, label = 'Handlebar or Seatpost Sensor')
    plt.xlim([0, 400])
    plt.title('Magnitude Spectrum - Y Axis of {}'.format(FN))
    plt.legend()
    plt.grid()
    plt.show()
else: 
    print('No MSD')    

#Second Integration of Data, Displacement =============================================================================
plt.figure(8)
Sensor1x_int1c = integrate.cumtrapz(Sensor1x_abs, time)
Sensor2x_int1c = integrate.cumtrapz(Sensor2x_abs, time)
Sensor1y_int1c = integrate.cumtrapz(Sensor1y_abs, time)
Sensor2y_int1c = integrate.cumtrapz(Sensor2y_abs, time)
Sensor1z_int1c = integrate.cumtrapz(Sensor1z_abs, time)
Sensor2z_int1c = integrate.cumtrapz(Sensor2z_abs, time)
Sensor1x_int2 = integrate.cumtrapz(Sensor1x_int1c, time[:-1])
Sensor2x_int2 = integrate.cumtrapz(Sensor2x_int1c, time[:-1])
Sensor1y_int2 = integrate.cumtrapz(Sensor1y_int1c, time[:-1])
Sensor2y_int2 = integrate.cumtrapz(Sensor2y_int1c, time[:-1])
Sensor1z_int2 = integrate.cumtrapz(Sensor1z_int1c, time[:-1])
Sensor2z_int2 = integrate.cumtrapz(Sensor2z_int1c, time[:-1])

Sensor1x_abs.to_csv("Sensor1abs.csv")

Sensor1x_abs = Sensor1x_abs*32.2

B = 1
Sensor1x_int1cc = np.empty(len(time))
while B <= (len(time)-1):
    Sensor1x_int1cc[B] = time_int_avg*((Sensor1x_abs[B-1] + Sensor1x_abs[B])/2)
    B = B+1

Sensor1x_int1cc = Sensor1x_int1cc*3600/5280
#print(np.mean(Sensor1x_int1cc))

plt.plot(time, Sensor1x_int1cc, label = 'Axle Sensor(X)', color='r')
#plt.plot(time[:-2], Sensor1y_int2, label = "Handlebar or Seatpost Sensor(X)")
plt.title('Velocity - X Axis of {}'.format(FN))
plt.xlabel('Time (usec)')
plt.ylabel('Velocity (mph)')
#plt.xlim([0, 400])
#plt.ylim([0, 300])
plt.legend()
plt.grid()


SXAV1 = Sensor1x_int1*32.2*time[-1:]/5280*3600
print('SXAV', SXAV1)
SXAV2 = Sensor2x_int1*32.2*time_int_avg*time[-1:]*(1/5280)*60*60
print('SXAV', SXAV2)


#Show all plots now  
plt.show()

