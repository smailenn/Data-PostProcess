#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Calibration variables and Accelerometer setup



#read the data into an array from text file
dataframe = pd.read_csv('C:\\Users\\smailen\\OneDrive - Quality Bicycle Products\\Vibration Analysis\\Waxwing Test\\WAXWING00.txt', sep='\t')

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


#make some plots
plt.plot(time, Sensor2y, time, Sensor1y)
plt.show()

plt.plot()