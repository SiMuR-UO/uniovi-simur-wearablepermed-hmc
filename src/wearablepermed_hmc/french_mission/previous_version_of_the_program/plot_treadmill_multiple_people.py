from math import sqrt
import matplotlib
import matplotlib.pyplot as plt
import pandas
import openpyxl
import numpy as np
from datetime import time, date, datetime, timedelta

matplotlib.use('TkAgg')

participants = [
    {
        'name': 'PMP1011',
        'csv_file': '/mnt/nvme1n2/git/uniovi-simur-wearablepermed-data/input/PMP1011/PMP1011_W1_M.csv',
        'sheet_file': '/mnt/nvme1n2/git/uniovi-simur-wearablepermed-data/input/PMP1011/PMP1011_RegistroActividades.xlsx'
    },
    {
        'name': 'PMP1053',
        'csv_file': '/mnt/nvme1n2/git/uniovi-simur-wearablepermed-data/input/PMP1053/PMP1053_W1_M.csv',
        'sheet_file': '/mnt/nvme1n2/git/uniovi-simur-wearablepermed-data/input/PMP1053/PMP1053_RegistroActividades.xlsx'
    },
    {
        'name': 'PMP1055',
        'csv_file': '/mnt/nvme1n2/git/uniovi-simur-wearablepermed-data/input/PMP1055/PMP1055_W1_M.csv',
        'sheet_file': '/mnt/nvme1n2/git/uniovi-simur-wearablepermed-data/input/PMP1055/PMP1055_RegistroActividades.xlsx'
    },
    {
        'name': 'PMP1056',
        'csv_file': '/mnt/nvme1n2/git/uniovi-simur-wearablepermed-data/input/PMP1056/PMP1056_W1_M.csv',
        'sheet_file': '/mnt/nvme1n2/git/uniovi-simur-wearablepermed-data/input/PMP1056/PMP1056_RegistroActividades.xlsx'
    },
    {
        'name': 'PMP1057',
        'csv_file': '/mnt/nvme1n2/git/uniovi-simur-wearablepermed-data/input/PMP1057/PMP1057_W1_M.csv',
        'sheet_file': '/mnt/nvme1n2/git/uniovi-simur-wearablepermed-data/input/PMP1057/PMP1057_RegistroActividades.xlsx'
    },
    {
        'name': 'PMP1049',
        'csv_file': '/mnt/nvme1n2/git/uniovi-simur-wearablepermed-data/input/PMP1049/PMP1049_W1_M.csv',
        'sheet_file': '/mnt/nvme1n2/git/uniovi-simur-wearablepermed-data/input/PMP1049/PMP1049_RegistroActividades.xlsx'
    },
]

# Fixed treadmill speed labels
x_labels = ['2km/h', '3km/h', '4km/h', '5km/h', '6km/h', '7km/h', '8km/h']
x_positions = list(range(len(x_labels)))  # [0, 1, 2, ..., 6]

# Function to convert Excel time values
def conversion_time(cell_value):
    if isinstance(cell_value, date):
        return cell_value
    if isinstance(cell_value, timedelta):
        total_seconds = cell_value.total_seconds()
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        return time(hours - 1, minutes, seconds)  # Adjust from GMT+1 to GMT
    if isinstance(cell_value, str):
        time_parts = cell_value.split(":")
        if len(time_parts) == 3:
            hours, minutes, seconds = map(int, time_parts)
            return time(hours - 1, minutes, seconds)  # Adjust from GMT+1 to GMT

# Function to extract accelerometer data between two timestamps
def search_csv_numpy(data, time_start, time_end):
    mask = (data[:, 0] >= time_start) & (data[:, 0] < time_end)
    selected = data[mask]
    return selected[:, 1], selected[:, 2], selected[:, 3]

# Start the plot
plt.figure()

for person in participants:
    df = pandas.read_csv(person['csv_file'], usecols=['dateTime', 'acc_x', 'acc_y', 'acc_z'])
    data = df.to_numpy()

    workbook = openpyxl.load_workbook(person['sheet_file'], data_only=True)
    sheet = workbook['Hoja1']

    # Determine which treadmill speeds were actually performed (E72 to E78)
    speeds_done = []
    for i, row in enumerate(range(72, 79), start=2):  # 2km/h to 8km/h
        cell_value = sheet[f"F{row}"].value
        speeds_done.append(isinstance(cell_value, (int, float)))  # True if value is present

    # Get experiment date and time interval
    exp_date = sheet["E13"].value
    exp_date_date = conversion_time(exp_date)
    exp_start = sheet["D72"].value
    exp_start_time = conversion_time(exp_start)
    exp_end = sheet["D73"].value
    exp_end_time = conversion_time(exp_end)

    time_start = datetime.combine(exp_date_date, exp_start_time)
    time_end = datetime.combine(exp_date_date, exp_end_time)
    time_start_unix = int(time_start.timestamp()) * 1000  # Convert to milliseconds
    time_end_unix = int(time_end.timestamp()) * 1000

    # Extract acceleration data
    acceleration = search_csv_numpy(data, time_start_unix, time_end_unix)

    assert len(acceleration[0]) == len(acceleration[1]) == len(acceleration[2]), \
        "acceleration sublists must be of the same length"

    # Compute magnitude of acceleration vector
    acc_x, acc_y, acc_z = acceleration
    avg_acc = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

    # Determine number of speeds actually performed
    num_speeds = sum(speeds_done)

    if num_speeds == 0:
        print(f"⚠️ {person['name']} has no treadmill activity data. Skipping.")
        continue

    if len(avg_acc) < num_speeds:
        print(f"⚠️ {person['name']} has insufficient acceleration data ({len(avg_acc)} < {num_speeds}). Skipping.")
        continue

    group_size = len(avg_acc) // num_speeds
    pointer = 0
    mean_acceleration = []

    # Compute mean acceleration for each of the 7 treadmill speeds (or NaN if not done)
    for speed_done in speeds_done:
        if speed_done:
            group = avg_acc[pointer:pointer + group_size]
            mean = np.mean(group)
            pointer += group_size
        else:
            mean = float('nan')
        mean_acceleration.append(mean)
    # Plot the data for this participant
    plt.scatter(x_positions, mean_acceleration, label=person['name'])

# Final plot formatting
plt.xticks(ticks=x_positions, labels=x_labels, rotation=45)
plt.title('Mean Hand Acceleration Based on Treadmill Speed')
plt.xlabel('Treadmill Speed')
plt.ylabel('Mean Acceleration')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
input("Press Enter to close...")