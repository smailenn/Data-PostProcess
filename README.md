# Data-PostProcess
Python Code to process accel. Data

Last updated Nov. 9

This script was largely created to analyze accelerometer data from on bicycle testing.  It was prefered over using Excel because of its ability to handle large amounts of data 
easily and produce charts with the exact same formatting for in depth comparison.

Typical workflow:
1. Data is collected using the DataLogger ESP-32 S2 system with two accelerometers recording at a typical 1000 Hz. 
2. Data is saved in a .txt file with preformated headers for the Python code to recognize and bring data in properly.
3. During testing a calibration file is created when all accels have been mounted onto bike and bike is stationary to get base values. 
4. The Calibration Data is run through the calibration.py file to produce the values that the "main.py" file will read once a file is analyzed.
5. Once these values have been created and stored in the "file_name Cal values.txt" then the "main.py" script can be told where they are to use them in the file analysis.
6. The "main.py" file is then open and few parameters to need be addressed to run:
  File_path of the file you want to analyze
  Cal_path of the calibration numbers that were derived from the calibration process done before you acquired data from the particular data you are interested in
  Sensor_select is for Arduino (ESP32 system) or for QBP's DTS system
    If using the DTS system data a dummy Cal_path file needs to be created with all values equal to zero since all calibration for that system is done outside this script before
    recieving the data here
  Trim_Lower and Trim_Upper for trimming any unwanted data from the file
7. The file is now ready to be analyzed
8. The script will produce a .txt file of numerical results named after the file_path for easy reference.  
9. The script will produce several charts named after the file_path for easy reference 
