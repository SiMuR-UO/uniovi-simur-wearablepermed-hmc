from math import sqrt
import matplotlib
import matplotlib.pyplot as plt
import pandas
import openpyxl
import numpy as np
from datetime import time, date, datetime, timedelta

# matplotlib.use('TkAgg')

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

age_groups = {
    '>60': [],
    '50-60': [],
    '40-50': [],
    '30-40': [],
    '<30': []
}

gender_groups = {
    'Hombre': [],
    'Mujer': []
}

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

for person in participants:
    df = pandas.read_csv(person['csv_file'], usecols=['dateTime', 'acc_x', 'acc_y', 'acc_z'])
    data = df.to_numpy()

    workbook = openpyxl.load_workbook(person['sheet_file'], data_only=True)
    sheet = workbook['Hoja1']

    age = sheet['G8'].value  # Assuming G8 contains the age

    if age > 60:
        age_key = '>60'
    elif 50 <= age <= 60:
        age_key = '50-60'
    elif 40 <= age < 50:
        age_key = '40-50'
    elif 30 <= age < 40:
        age_key = '30-40'
    else:
        age_key = '<30'

    gender = sheet['C6'].value  # Assuming C6 contains the gender

    if gender == 'Masculino':
        gender = 'Hombre'
    elif gender == 'Femenino':
        gender = 'Mujer'

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
    
    age_groups[age_key].append(mean_acceleration)
    gender_groups[gender].append(mean_acceleration)

print("Age group sizes:")
for k, v in age_groups.items():
    print(f"{k}: {len(v)}")

print("\nGender group sizes:")
for k, v in gender_groups.items():
    print(f"{k}: {len(v)}")

fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Plot by Age
for age_key in ['<30', '30-40', '40-50', '50-60', '>60']:
    values = age_groups[age_key]
    if not values:
        continue

    x_vals, y_vals = [], []
    for participant_values in values:
        for i, val in enumerate(participant_values):
            if not pandas.isna(val):
                x_vals.append(x_positions[i])
                y_vals.append(val)

    axes[0].scatter(x_vals, y_vals, label=f'Age {age_key}', alpha=0.6)

handles, labels = axes[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
axes[0].legend(by_label.values(), by_label.keys())

axes[0].set_title('Mean Hand Acceleration by Age Group')
axes[0].set_ylabel('Mean Acceleration')
axes[0].grid(True)

# Plot by Gender
for gender in ['Hombre', 'Mujer']:
    values = gender_groups[gender]
    if not values:
        continue

    x_vals, y_vals = [], []
    for participant_values in values:
        for i, val in enumerate(participant_values):
            if not pandas.isna(val):
                x_vals.append(x_positions[i])
                y_vals.append(val)

    axes[1].scatter(x_vals, y_vals, label=f'Gender {gender}', alpha=0.6)

handles, labels = axes[1].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
axes[1].legend(by_label.values(), by_label.keys())

axes[1].set_title('Mean Hand Acceleration by Gender')
axes[1].set_xlabel('Treadmill Speed')
axes[1].set_ylabel('Mean Acceleration')
axes[1].set_xticks(x_positions)
axes[1].set_xticklabels(x_labels, rotation=45)
axes[1].grid(True)

plt.tight_layout()
plt.show(block=True)