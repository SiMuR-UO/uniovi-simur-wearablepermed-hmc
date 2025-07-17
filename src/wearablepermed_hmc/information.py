import pandas as pd
import numpy as np
import openpyxl
from datetime import datetime, date, time, timedelta
from math import sqrt
import os

base_path = "/mnt/nvme1n2/git/uniovi-simur-wearablepermed-data/input"

with open('participants.txt', 'r') as f:
    names = [name.strip() for name in f.read().split(',') if name.strip()]

# === List of activities ===
activities = [
    'Tapiz 2km/h', 'Tapiz 3km/h', 'Tapiz 4km/h', 'Tapiz 5km/h', 'Tapiz 6km/h', 'Tapiz 7km/h', 'Tapiz 8km/h',
    'Sit-to-stand 30s', 'Incremental - Rest', 'Incremental - Warm-up', 'Incremental - Start', 'Incremental - Middle', 'Incremental - End', 'Yoga',
    'Sentado TV', 'Sentado leyendo', 'Sentado PC',
    'De pie PC', 'De pie doblando toallas', 'De pie libros', 'De pie barriendo',
    'Caminar normal', 'Caminar con móvil/libro', 'Caminar con compra', 'Caminar zigzag',
    'Trotar', 'Subir y bajar escaleras'
]

summary_data = []

def conversion_time(cell_value):
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
            print(f"⛔️ Unknown format for conversion_time: {cell_value}")
            return None
    return None

def get_mean_acceleration(csv_data, start_ms, end_ms):
    mask = (csv_data[:, 0] >= start_ms) & (csv_data[:, 0] < end_ms)
    segment = csv_data[mask]
    if len(segment) == 0:
        return float('nan')
    acc_x, acc_y, acc_z = segment[:, 1], segment[:, 2], segment[:, 3]
    norm = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    return np.mean(norm)

def get_min_acceleration(csv_data, start_ms, end_ms):
    mask = (csv_data[:, 0] >= start_ms) & (csv_data[:, 0] < end_ms)
    segment = csv_data[mask]
    if len(segment) == 0:
        return float('nan')
    acc_x, acc_y, acc_z = segment[:, 1], segment[:, 2], segment[:, 3]
    norm = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    return np.min(norm)

def get_max_acceleration(csv_data, start_ms, end_ms):
    mask = (csv_data[:, 0] >= start_ms) & (csv_data[:, 0] < end_ms)
    segment = csv_data[mask]
    if len(segment) == 0:
        return float('nan')
    acc_x, acc_y, acc_z = segment[:, 1], segment[:, 2], segment[:, 3]
    norm = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    return np.max(norm)

for name in names:
    try:
        person_csv = f"{base_path}/{name}/{name}_W1_M.csv"
        person_sheet = f"{base_path}/{name}/{name}_RegistroActividades.xlsx"

        df = pd.read_csv(person_csv, usecols=['dateTime', 'acc_x', 'acc_y', 'acc_z'])
        data = df.to_numpy()

        workbook = openpyxl.load_workbook(person_sheet, data_only=True)
        sheet = workbook['Hoja1']

        row = {
            'Name': name,
            'Age': int(sheet['G8'].value) if sheet['G8'].value else None,
            'Gender': 'Male' if sheet['C6'].value == 'Masculino' else 'Female',
            'Height': sheet['E8'].value,
            'Weight': sheet['C8'].value
        }

        exp_date = conversion_time(sheet['E13'].value)
        if not exp_date:
            print(f"⚠️ Missing date for {name}, skipping.")
            continue

        # === Treadmill activity ===
        start_time_global = conversion_time(sheet["D72"].value)
        end_time_global = conversion_time(sheet["D73"].value)

        if start_time_global and end_time_global:
            start_dt = datetime.combine(exp_date, start_time_global)
            end_dt = datetime.combine(exp_date, end_time_global)
            start_ms = int(start_dt.timestamp() * 1000)
            end_ms = int(end_dt.timestamp() * 1000)

            speeds_done = [isinstance(sheet[f"F{row_id}"].value, (int, float)) for row_id in range(72, 79)]

            segment_mask = (data[:, 0] >= start_ms) & (data[:, 0] < end_ms)
            segment = data[segment_mask]

            if len(segment) == 0:
                for speed in ['2km/h', '3km/h', '4km/h', '5km/h', '6km/h', '7km/h', '8km/h']:
                    row[f'Tapiz {speed} Mean'] = float('nan')
                    row[f'Tapiz {speed} Min'] = float('nan')
                    row[f'Tapiz {speed} Max'] = float('nan')
            else:
                acc_x, acc_y, acc_z = segment[:, 1], segment[:, 2], segment[:, 3]
                norm = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
                n_speeds = sum(speeds_done)
                segment_length = len(norm) // n_speeds if n_speeds > 0 else 0
                pointer = 0
                speeds = ['2km/h', '3km/h', '4km/h', '5km/h', '6km/h', '7km/h', '8km/h']
                for i, speed in enumerate(speeds):
                    if speeds_done[i]:
                        group = norm[pointer:pointer + segment_length]
                        pointer += segment_length
                        mean_acc = np.mean(group) if len(group) > 0 else float('nan')
                        min_acc = np.min(group) if len(group) > 0 else float('nan')
                        max_acc = np.max(group) if len(group) > 0 else float('nan')
                    else:
                        mean_acc = min_acc = max_acc = float('nan')
                    row[f'Tapiz {speed} Mean'] = mean_acc
                    row[f'Tapiz {speed} Min'] = min_acc
                    row[f'Tapiz {speed} Max'] = max_acc
        else:
            for speed in ['2km/h', '3km/h', '4km/h', '5km/h', '6km/h', '7km/h', '8km/h']:
                row[f'Tapiz {speed} Mean'] = float('nan')
                row[f'Tapiz {speed} Min'] = float('nan')
                row[f'Tapiz {speed} Max'] = float('nan')

        # === Incremental cicloergómetro segments ===
        incremental_segments = {
            'Incremental - Rest': ('D90', 'D91'),
            'Incremental - Warm-up': ('D91', 'D92'),
            'Incremental - Start': ('D92', None),  # Start to be calculated
            'Incremental - Middle': ('D92', 'D93'),  # Start used as base
            'Incremental - End': ('D92', 'D93')
        }

        inc_start_time = conversion_time(sheet['D92'].value)
        inc_end_time = conversion_time(sheet['D93'].value)

        if inc_start_time and inc_end_time:
            start_dt = datetime.combine(exp_date, inc_start_time)
            end_dt = datetime.combine(exp_date, inc_end_time)
            inc_start_ms = int(start_dt.timestamp() * 1000)
            inc_end_ms = int(end_dt.timestamp() * 1000)
            duration = inc_end_ms - inc_start_ms
            third = duration // 3
            starts = [inc_start_ms, inc_start_ms + third, inc_start_ms + 2 * third]
            ends = [inc_start_ms + third, inc_start_ms + 2 * third, inc_end_ms]
        else:
            starts = ends = [None, None, None]

        for key in incremental_segments:
            if key == 'Incremental - Start' and starts[0] and ends[0]:
                acc_mean = get_mean_acceleration(data, starts[0], ends[0])
                acc_min = get_min_acceleration(data, starts[0], ends[0])
                acc_max = get_max_acceleration(data, starts[0], ends[0])
            elif key == 'Incremental - Middle' and starts[1] and ends[1]:
                acc_mean = get_mean_acceleration(data, starts[1], ends[1])
                acc_min = get_min_acceleration(data, starts[1], ends[1])
                acc_max = get_max_acceleration(data, starts[1], ends[1])
            elif key == 'Incremental - End' and starts[2] and ends[2]:
                acc_mean = get_mean_acceleration(data, starts[2], ends[2])
                acc_min = get_min_acceleration(data, starts[2], ends[2])
                acc_max = get_max_acceleration(data, starts[2], ends[2])
            else:
                start_cell, end_cell = incremental_segments[key]
                raw_start = conversion_time(sheet[start_cell].value)
                raw_end = conversion_time(sheet[end_cell].value)
                if raw_start and raw_end:
                    s_dt = datetime.combine(exp_date, raw_start)
                    e_dt = datetime.combine(exp_date, raw_end)
                    s_ms = int(s_dt.timestamp() * 1000)
                    e_ms = int(e_dt.timestamp() * 1000)
                    acc_mean = get_mean_acceleration(data, s_ms, e_ms)
                    acc_min = get_min_acceleration(data, s_ms, e_ms)
                    acc_max = get_max_acceleration(data, s_ms, e_ms)
                else:
                    acc_mean = acc_min = acc_max = float('nan')

            row[f'{key} Mean'] = acc_mean
            row[f'{key} Min'] = acc_min
            row[f'{key} Max'] = acc_max

        # === Other activities ===
        other_activities = {
            'Sit-to-stand 30s': ('D81', 'D82'),
            'Yoga': ('D144', 'D145'),
            'Sentado TV': ('D153', 'D154'),
            'Sentado leyendo': ('D162', 'D163'),
            'Sentado PC': ('D172', 'D173'),
            'De pie PC': ('D181', 'D182'),
            'De pie doblando toallas': ('D190', 'D191'),
            'De pie libros': ('D199', 'D200'),
            'De pie barriendo': ('D208', 'D209'),
            'Caminar normal': ('D219', 'D220'),
            'Caminar con móvil/libro': ('D228', 'D229'),
            'Caminar con compra': ('D237', 'D238'),
            'Caminar zigzag': ('D246', 'D247'),
            'Trotar': ('D255', 'D256'),
            'Subir y bajar escaleras': ('D264', 'D265'),
        }

        for act, (start_cell, end_cell) in other_activities.items():
            raw_start = sheet[start_cell].value
            raw_end = sheet[end_cell].value

            if act in ['Trotar', 'Subir y bajar escaleras']:
                act_date = conversion_time(sheet['E112'].value)
            elif act in ['Yoga', 'Sentado TV', 'Sentado leyendo', 'Sentado PC',
                         'De pie PC', 'De pie doblando toallas', 'De pie libros',
                         'De pie barriendo', 'Caminar normal', 'Caminar con móvil/libro',
                         'Caminar con compra', 'Caminar zigzag']:
                act_date = conversion_time(sheet['E112'].value)
            else:
                act_date = exp_date

            start_time = conversion_time(raw_start)
            end_time = conversion_time(raw_end)

            if start_time and end_time and act_date:
                start_dt = datetime.combine(act_date, start_time)
                end_dt = datetime.combine(act_date, end_time)
                start_ms = int(start_dt.timestamp() * 1000)
                end_ms = int(end_dt.timestamp() * 1000)
                acc_mean = get_mean_acceleration(data, start_ms, end_ms)
                acc_min = get_min_acceleration(data, start_ms, end_ms)
                acc_max = get_max_acceleration(data, start_ms, end_ms)
            else:
                acc_mean = acc_min = acc_max = float('nan')

            row[f'{act} Mean'] = acc_mean
            row[f'{act} Min'] = acc_min
            row[f'{act} Max'] = acc_max

        summary_data.append(row)
        print(f"✅ Processed {name}")

    except Exception as e:
        print(f"⛔️ Error processing {name}: {e}")

# Save summary to Excel
summary_df = pd.DataFrame(summary_data)
summary_df.to_excel("summary_participants_mean_min_max.xlsx", index=False)
print("Résumé enregistré dans summary_participants_mean_min_max.xlsx")
