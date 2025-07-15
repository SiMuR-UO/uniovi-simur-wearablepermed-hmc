from math import sqrt
import matplotlib
import matplotlib.pyplot as plt
import pandas
import openpyxl
import numpy as np
from datetime import time, date, datetime, timedelta

# matplotlib.use('TkAgg')

# === Load participants from a TXT file ===
base_path = "/mnt/nvme1n2/git/uniovi-simur-wearablepermed-data/input"

with open('participants.txt', 'r') as f:
    content = f.read()

names = [name.strip() for name in content.split(',') if name.strip()]

participants = []
for name in names:
    participants.append({
        'name': name,
        'csv_file': f"{base_path}/{name}/{name}_W1_M.csv",
        'sheet_file': f"{base_path}/{name}/{name}_RegistroActividades.xlsx"
    })

# === Grouping containers ===
age_groups = {'>60': [], '50-60': [], '40-50': [], '30-40': [], '<30': []}
gender_groups = {'Hombre': [], 'Mujer': []}

x_labels = ['2km/h', '3km/h', '4km/h', '5km/h', '6km/h', '7km/h', '8km/h']
x_positions = list(range(len(x_labels)))  # [0–6]

def conversion_time(cell_value):
    if isinstance(cell_value, date):
        return cell_value
    if isinstance(cell_value, timedelta):
        total_seconds = cell_value.total_seconds()
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        return time(hours - 1, minutes, seconds)
    if isinstance(cell_value, str):
        time_parts = cell_value.split(":")
        if len(time_parts) == 3:
            hours, minutes, seconds = map(int, time_parts)
            return time(hours - 1, minutes, seconds)

def search_csv_numpy(data, time_start, time_end):
    mask = (data[:, 0] >= time_start) & (data[:, 0] < time_end)
    selected = data[mask]
    return selected[:, 1], selected[:, 2], selected[:, 3]

for person in participants:
    try:
        df = pandas.read_csv(person['csv_file'], usecols=['dateTime', 'acc_x', 'acc_y', 'acc_z'])
        data = df.to_numpy()

        workbook = openpyxl.load_workbook(person['sheet_file'], data_only=True)
        sheet = workbook['Hoja1']

        age = sheet['G8'].value
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

        gender = sheet['C6'].value
        gender = 'Hombre' if gender == 'Masculino' else 'Mujer'

        speeds_done = [
            isinstance(sheet[f"F{row}"].value, (int, float))
            for row in range(72, 79)
        ]

        exp_date = conversion_time(sheet["E13"].value)
        exp_start_time = conversion_time(sheet["D72"].value)
        exp_end_time = conversion_time(sheet["D73"].value)

        time_start = datetime.combine(exp_date, exp_start_time)
        time_end = datetime.combine(exp_date, exp_end_time)
        time_start_unix = int(time_start.timestamp()) * 1000
        time_end_unix = int(time_end.timestamp()) * 1000

        acc_x, acc_y, acc_z = search_csv_numpy(data, time_start_unix, time_end_unix)

        if not (len(acc_x) == len(acc_y) == len(acc_z)):
            print(f"⚠️ {person['name']} has mismatched acceleration lengths.")
            continue

        avg_acc = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        num_speeds = sum(speeds_done)

        if num_speeds == 0 or len(avg_acc) < num_speeds:
            print(f"⚠️ {person['name']} has insufficient treadmill data. Skipping.")
            continue

        group_size = len(avg_acc) // num_speeds
        pointer = 0
        mean_acceleration = []

        for speed_done in speeds_done:
            if speed_done:
                group = avg_acc[pointer:pointer + group_size]
                mean = np.mean(group) if len(group) > 0 else float('nan')
                pointer += group_size
            else:
                mean = float('nan')
            mean_acceleration.append(mean)

        age_groups[age_key].append(mean_acceleration)
        gender_groups[gender].append(mean_acceleration)

    except Exception as e:
        print(f"❌ Error processing {person['name']}: {e}")

print("Age group sizes:")
for k, v in age_groups.items():
    print(f"{k}: {len(v)}")

print("\nGender group sizes:")
for k, v in gender_groups.items():
    print(f"{k}: {len(v)}")

fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Age scatter
for age_key in age_groups:
    values = age_groups[age_key]
    if not values:
        continue
    x_vals, y_vals = [], []
    for row in values:
        for i, val in enumerate(row):
            if not pandas.isna(val):
                x_vals.append(x_positions[i])
                y_vals.append(val)
    axes[0].scatter(x_vals, y_vals, label=f'Age {age_key}', alpha=0.6)

axes[0].set_title('Mean Hand Acceleration by Age Group')
axes[0].set_ylabel('Mean Acceleration')
axes[0].legend()
axes[0].grid(True)

# Gender scatter
for gender in gender_groups:
    values = gender_groups[gender]
    if not values:
        continue
    x_vals, y_vals = [], []
    for row in values:
        for i, val in enumerate(row):
            if not pandas.isna(val):
                x_vals.append(x_positions[i])
                y_vals.append(val)
    axes[1].scatter(x_vals, y_vals, label=f'Gender {gender}', alpha=0.6)

axes[1].set_title('Mean Hand Acceleration by Gender')
axes[1].set_xlabel('Treadmill Speed')
axes[1].set_ylabel('Mean Acceleration')
axes[1].set_xticks(x_positions)
axes[1].set_xticklabels(x_labels, rotation=45)
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show(block=True)
