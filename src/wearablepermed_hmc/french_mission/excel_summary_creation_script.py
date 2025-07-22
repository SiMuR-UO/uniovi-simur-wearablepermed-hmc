import pandas as pd
import numpy as np
import openpyxl
from datetime import datetime, date, time, timedelta
from math import sqrt
import os

# === Path where participant data is stored ===
base_path = "/mnt/nvme1n2/git/uniovi-simur-wearablepermed-data/input"

# === Read participant names from file ===
with open('/home/simur/git/uniovi-simur-wearablepermed-hmc/src/wearablepermed_hmc/french_mission/participants.txt', 'r') as f:
    names = [name.strip() for name in f.read().split(',') if name.strip()]

summary_data = []  # Will collect processed results for all participants

# === Convert various time formats into Python time object ===
def convert_time(cell_value):
    if isinstance(cell_value, time):
        return time(cell_value.hour - 1, cell_value.minute, cell_value.second)
    if isinstance(cell_value, date):
        return cell_value
    if isinstance(cell_value, timedelta):
        total_seconds = cell_value.total_seconds()
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        return time(hours - 1, minutes, seconds)
    if isinstance(cell_value, str):
        try:
            hours, minutes, seconds = map(int, cell_value.strip().split(":"))
            return time(hours - 1, minutes, seconds)
        except:
            print(f"⛔️ Unknown time format: {cell_value}")
            return None
    return None

# === Compute mean, min, and max of acceleration norm for a segment ===
def compute_stats(segment):
    if len(segment) == 0:
        return float('nan'), float('nan'), float('nan')
    acc_x, acc_y, acc_z = segment[:, 1], segment[:, 2], segment[:, 3]
    norm = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    return np.mean(norm), np.min(norm), np.max(norm)

# === Extract a segment from data between two timestamps ===
def get_segment(data, start_ms, end_ms):
    return data[(data[:, 0] >= start_ms) & (data[:, 0] < end_ms)]

# === Process each participant ===
for name in names:
    try:
        person_csv = f"{base_path}/{name}/{name}_W1_M.csv"
        person_sheet = f"{base_path}/{name}/{name}_RegistroActividades.xlsx"

        # Load CSV sensor data
        df = pd.read_csv(person_csv, usecols=['dateTime', 'acc_x', 'acc_y', 'acc_z'])
        data = df.to_numpy()

        # Load activity record Excel file
        wb = openpyxl.load_workbook(person_sheet, data_only=True)
        sheet = wb['Hoja1']

        # Extract demographic info
        gender_raw = sheet['C6'].value
        gender = 'Male' if gender_raw in ['Masculino', 'Hombre'] else 'Female'

        row = {
            'Name': name,
            'Age': int(sheet['G8'].value) if sheet['G8'].value else None,
            'Gender': gender,
            'Height': sheet['E8'].value,
            'Weight': sheet['C8'].value
        }

        # Get experiment date
        exp_date = convert_time(sheet['E13'].value)
        if not exp_date:
            print(f"⚠️ Missing date for {name}, skipping.")
            continue

        # === Treadmill analysis ===
        start_global = convert_time(sheet["D72"].value)
        end_global = convert_time(sheet["D73"].value)
        speeds = ['2km/h', '3km/h', '4km/h', '5km/h', '6km/h', '7km/h', '8km/h']

        if start_global and end_global:
            start_ms = int(datetime.combine(exp_date, start_global).timestamp() * 1000)
            end_ms = int(datetime.combine(exp_date, end_global).timestamp() * 1000)
            segment = get_segment(data, start_ms, end_ms)

            speeds_done = [isinstance(sheet[f"F{r}"].value, (int, float)) for r in range(72, 79)]
            n_speeds = sum(speeds_done)
            segment_length = len(segment) // n_speeds if n_speeds > 0 else 0

            pointer = 0
            for i, speed in enumerate(speeds):
                if speeds_done[i]:
                    seg = segment[pointer:pointer + segment_length]
                    mean, min_, max_ = compute_stats(seg)
                    pointer += segment_length
                else:
                    mean = min_ = max_ = float('nan')
                row[f'Treadmill {speed} Mean'] = mean
                row[f'Treadmill {speed} Min'] = min_
                row[f'Treadmill {speed} Max'] = max_
        else:
            # Fill with NaNs if no treadmill data available
            for speed in speeds:
                row[f'Treadmill {speed} Mean'] = float('nan')
                row[f'Treadmill {speed} Min'] = float('nan')
                row[f'Treadmill {speed} Max'] = float('nan')

        # === Incremental cycling phases analysis ===
        inc_segments = {
            'Rest': ('D90', 'D91'),
            'Warm-up': ('D91', 'D92'),
            'Start': ('D92', None),
            'Middle': ('D92', 'D93'),
            'End': ('D92', 'D93')
        }

        inc_start = convert_time(sheet['D92'].value)
        inc_end = convert_time(sheet['D93'].value)
        if inc_start and inc_end:
            inc_start_ms = int(datetime.combine(exp_date, inc_start).timestamp() * 1000)
            inc_end_ms = int(datetime.combine(exp_date, inc_end).timestamp() * 1000)
            duration = inc_end_ms - inc_start_ms
            thirds = [inc_start_ms, inc_start_ms + duration // 3, inc_start_ms + 2 * duration // 3, inc_end_ms]
        else:
            thirds = [None] * 4

        for phase, (cell_start, cell_end) in inc_segments.items():
            if phase == 'Start' and all(thirds[:2]):
                segment = get_segment(data, thirds[0], thirds[1])
            elif phase == 'Middle' and all(thirds[1:3]):
                segment = get_segment(data, thirds[1], thirds[2])
            elif phase == 'End' and all(thirds[2:]):
                segment = get_segment(data, thirds[2], thirds[3])
            else:
                s = convert_time(sheet[cell_start].value)
                e = convert_time(sheet[cell_end].value) if cell_end else None
                if s and e:
                    start_ms = int(datetime.combine(exp_date, s).timestamp() * 1000)
                    end_ms = int(datetime.combine(exp_date, e).timestamp() * 1000)
                    segment = get_segment(data, start_ms, end_ms)
                else:
                    segment = np.array([])

            mean, min_, max_ = compute_stats(segment)
            row[f'Incremental {phase} Mean'] = mean
            row[f'Incremental {phase} Min'] = min_
            row[f'Incremental {phase} Max'] = max_

        # === Other functional activities analysis ===
        activity_map = {
            'Sit-to-stand 30s': ('D81', 'D82'),
            'Yoga': ('D144', 'D145'),
            'Sitting TV': ('D153', 'D154'),
            'Sitting reading': ('D162', 'D163'),
            'Sitting computer': ('D172', 'D173'),
            'Standing computer': ('D181', 'D182'),
            'Standing folding towels': ('D190', 'D191'),
            'Standing moving books': ('D199', 'D200'),
            'Standing sweeping': ('D208', 'D209'),
            'Walking normal': ('D219', 'D220'),
            'Walking phone/book': ('D228', 'D229'),
            'Walking shopping bag': ('D237', 'D238'),
            'Walking zigzag': ('D246', 'D247'),
            'Jogging': ('D255', 'D256'),
            'Stairs': ('D264', 'D265')
        }

        for act, (s_cell, e_cell) in activity_map.items():
            s_raw = convert_time(sheet[s_cell].value)
            e_raw = convert_time(sheet[e_cell].value)

            act_date = convert_time(sheet['E112'].value) if act not in ['Sit-to-stand 30s'] else exp_date

            if s_raw and e_raw and act_date:
                s_ms = int(datetime.combine(act_date, s_raw).timestamp() * 1000)
                e_ms = int(datetime.combine(act_date, e_raw).timestamp() * 1000)
                segment = get_segment(data, s_ms, e_ms)
            else:
                segment = np.array([])

            mean, min_, max_ = compute_stats(segment)
            row[f'{act} Mean'] = mean
            row[f'{act} Min'] = min_
            row[f'{act} Max'] = max_

        summary_data.append(row)
        print(f"✅ Processed: {name}")

    except Exception as e:
        print(f"⛔️ Error processing {name}: {e}")

# === Export all data to Excel ===
df_summary = pd.DataFrame(summary_data)
df_summary.to_excel("/home/simur/git/uniovi-simur-wearablepermed-hmc/src/wearablepermed_hmc/french_mission/summary_participants_mean_min_max.xlsx", index=False)
print("✅ Summary saved to summary_participants_mean_min_max.xlsx")