from math import *
import matplotlib
import matplotlib.pyplot as plt
import pandas
import openpyxl
import numpy as np
from datetime import time, date, datetime, timedelta

csv_file = '/mnt/nvme1n2/git/uniovi-simur-wearablepermed-data/input/PMP1011/PMP1011_W1_M.csv'
sheet_file = '/mnt/nvme1n2/git/uniovi-simur-wearablepermed-data/input/PMP1011/PMP1011_RegistroActividades.xlsx'

df = pandas.read_csv(csv_file, usecols=['dateTime', 'acc_x', 'acc_y', 'acc_z'])
data = df.to_numpy()
workbook = openpyxl.load_workbook(sheet_file, data_only=True)
sheet = workbook['Hoja1']

# Fonction for time conversion
def conversion_time(cell_value):
    if isinstance(cell_value, date):
        return cell_value
    if isinstance(cell_value, timedelta):
        total_seconds = cell_value.total_seconds()
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        return time(hours-1, minutes, seconds) # hours -1 : GMT + 1 => GMT
    if isinstance(cell_value, str):
        time_parts = cell_value.split(":")
        if len(time_parts) == 3:
            hours, minutes, seconds = map(int, time_parts)
            return time(hours-1, minutes, seconds) # hours -1 : GMT + 1 => GMT
        
# Fonction to find acceleration data in csv
def search_csv_numpy(data, time_start, time_end):
    mask = (data[:, 0] >= time_start) & (data[:, 0] < time_end)
    selected = data[mask]
    return selected[:, 1], selected[:, 2], selected[:, 3]

#Date of the experience
exp_date = sheet["E13"].value
exp_date_date = conversion_time(exp_date)

#Start of the experience
exp_start = sheet["D72"].value
exp_start_time = conversion_time(exp_start)

#End of the experience
exp_end = sheet["D73"].value
exp_end_time = conversion_time(exp_end)

time_start = datetime.combine(exp_date_date, exp_start_time)
time_end = datetime.combine(exp_date_date, exp_end_time)
time_start_unix = int(time_start.timestamp())
time_end_unix = int(time_end.timestamp())
# timestamp in second => timestamp in milisecond
time_start_unix *= 1000
time_end_unix *= 1000

acceleration = search_csv_numpy(data, time_start_unix, time_end_unix)

acc_x, acc_y, acc_z = acceleration
avg_acc = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

speeds_done = []
for i, row in enumerate(range(72, 79), start=2):  # 2km/h to 8km/h
    cell_value = sheet[f"F{row}"].value
    speeds_done.append(isinstance(cell_value, (int, float)))  # True if value is present

num_speeds = sum(speeds_done)

if num_speeds == 0:
    print(f"⚠️ This person has no treadmill activity data. Skipping.")

if len(avg_acc) < num_speeds:
    print(f"⚠️ This person has insufficient acceleration data ({len(avg_acc)} < {num_speeds}). Skipping.")

x_labels = ['2km/h', '3km/h', '4km/h', '5km/h', '6km/h', '7km/h', '8km/h']
x_positions = list(range(len(x_labels)))  # [0, 1, 2, ..., 6]

group_size = len(avg_acc) // len(x_labels)
pointer = 0

mean_acceleration = []
for speed_done in speeds_done:
    if speed_done:
        group = avg_acc[pointer:pointer + group_size]
        mean = np.mean(group)
        pointer += group_size
    else:
        mean = float('nan')
    mean_acceleration.append(mean)

plt.figure()
matplotlib.use('TkAgg') 
plt.scatter(x_labels,mean_acceleration)
plt.xticks(ticks=x_positions, labels=x_labels, rotation=45)
plt.title('Mean Acceleration base on Treadmill')
plt.xlabel('Treadmill Speed')
plt.ylabel('Mean Acceleration')
plt.grid(True)
plt.show()
input("Appuie sur Entrée pour fermer...")