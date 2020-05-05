import pandas
import numpy as np
import matplotlib.pyplot as plt

#read the data into an array from CSV file
dataframe = pandas.read_csv(open('C:\\Users\\smailen\\OneDrive - Quality Bicycle Products\\Vibration Analysis\\Waxwing Test\\WAXWING00.txt', mode='t'))
# dataframe = pandas.read_csv(open('C:\\Users\\SeanM\\OneDrive - Quality Bicycle Products\\Programs\\Cutthroat Vibration Data Vert .csv', mode='r'))

arr = dataframe['Sensor 1 Y'].to_numpy()
time = dataframe['Time'].to_numpy()

# create data
N = 1048577 #size array to the smallest power of 2 larger than the original array length plus one
arr = np.pad(arr, (0, N - len(arr)), 'constant') # left pad data array with zeros

# Plot data quick for review
plt.plot(time, arr)

# perform Fast Fourier Transform and multiply by dt
dt = arr[1] - arr[0] 
ft = np.fft.fft(arr) * dt    
freq = np.fft.fftfreq(N, dt)
#freq = freq[:N//2+1]

# plot results on an x-y scatter plot
plt.plot(freq, np.abs(ft),',') # comma indicates a pixel marker
plt.xlabel('Frequency')
plt.ylabel('amplitude')
plt.xlim([0, 15])
plt.ylim([0, 300])
plt.show()