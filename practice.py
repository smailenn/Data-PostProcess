import pandas
import numpy as np
import matplotlib.pyplot as plt

#read the data into an array from text file
wax = pandas.read_csv('C:\\Users\\smailen\\OneDrive - Quality Bicycle Products\\Vibration Analysis\\Waxwing Test\\WAXWING00.txt', sep='\t')
#with open('C:\\Users\\smailen\\OneDrive - Quality Bicycle Products\\Vibration Analysis\\Waxwing Test\\WAXWING00.txt', mode='r') as vibration: 
    #WAXWING00 = csv.reader(vibration, delimiter='\t')  
print(wax)

#Waxwing = open('C:\\Users\\smailen\\OneDrive - Quality Bicycle Products\\Vibration Analysis\\Waxwing Test\\Waxwing00.txt')


