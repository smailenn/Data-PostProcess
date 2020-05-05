import pandas
import numpy as np
import matplotlib.pyplot as plt

#read the data into an array from CSV file
dataframe = pandas.read_csv(open('C:\\Users\\smailen\\OneDrive - Quality Bicycle Products\\Programs\\Cutthroat Vibration Data Vert .csv', mode='r'))
arr = dataframe['Rear Axle, Vertical, Normalized'].to_numpy()

#FFT plots
# Set size of array for FFT, time and data arrays must equal
X = 1
N = 0
while N < trim_size:
    N =2**X
    X += 1
    #print(N)
X-= 2
N =2**X
print(N)

# perform Fast Fourier Transform and multiply by dt
dt = arr[1] - arr[0] 
ft = np.fft.fft(arr) * dt    
freq = np.fft.fftfreq(N), dt)
#freq = freq[:N//2+1]

# plot results on an x-y scatter plot
plt.plot(freq, np.abs(ft),',') # comma indicates a pixel marker
plt.xlabel('Frequency')
plt.ylabel('amplitude')
plt.xlim([0, 20])
plt.ylim([0, 300])
plt.show()






#FFT of Data
ft = np.fft.fft(Sensor1y_g[0:N]) * time_int_avg  
freq = np.fft.fftfreq(N, time[0:N])
#freq = freq[:N//2+1]

# plot results on an x-y scatter plot
fig, (fftx, ffty, ffty) = plt.subplots(3,1)
ffty = plt.subplot(312)
ffty.plot(freq, np.abs(ft)) # comma indicates a pixel marker
plt.xlabel('Frequency')
plt.ylabel('amplitude')
#plt.xlim([0, 15])
#plt.ylim([0, 300])


plt.show()
