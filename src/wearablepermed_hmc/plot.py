import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

def age_to_group(age):
    if pd.isna(age):
        return 'Unknown'
    if age < 30:
        return '<30'
    elif 30 <= age < 40:
        return '30-39'
    elif 40 <= age < 50:
        return '40-49'
    elif 50 <= age < 60:
        return '50-59'
    else:
        return '60+'

def height_to_group(height):
    if pd.isna(height):
        return 'Unknown'
    if height < 150:
        return '<150'
    elif 150 <= height < 160:
        return '150-159'
    elif 160 <= height < 170:
        return '160-169'
    elif 170 <= height < 180:
        return '170-179'
    else:
        return '180+'

def weight_to_group(weight):
    if pd.isna(weight):
        return 'Unknown'
    if weight < 50:
        return '<50'
    elif 50 <= weight < 60:
        return '50-59'
    elif 60 <= weight < 70:
        return '60-69'
    elif 70 <= weight < 80:
        return '70-79'
    else:
        return '80+'

# Load summary file
summary_file = 'summary_english.xlsx'
df = pd.read_excel(summary_file)

# Add group columns
df['Age_Group'] = df['Age'].apply(age_to_group)
df['Height_Group'] = df['Height'].apply(height_to_group)
df['Weight_Group'] = df['Weight'].apply(weight_to_group)
df['Gender'] = df['Gender'].fillna('Unknown')

excluded_cols = ['Age', 'Height', 'Weight', 'Gender', 'Age_Group', 'Height_Group', 'Weight_Group', 'Name']

# Ask user which statistic to plot
stat_choice = input("Choose statistic to plot: mean, min, or max (default: mean): ").strip().lower()
if stat_choice not in ['mean', 'min', 'max']:
    stat_choice = 'mean'

# Prepare columns for treadmill and incremental depending on stat_choice
treadmill_cols_all = [col for col in df.columns if col.startswith('Treadmill ')]
treadmill_cols = [col for col in treadmill_cols_all if col.endswith(stat_choice.capitalize())]

incremental_base_cols = ['Incremental - Rest', 'Incremental - Warm-up', 'Incremental - Start', 'Incremental - Middle', 'Incremental - End']
incremental_cols = [f"{base} {stat_choice.capitalize()}" for base in incremental_base_cols if f"{base} {stat_choice.capitalize()}" in df.columns]

# Build list of activities
activities = ['Treadmill', 'Incremental'] + [
    col for col in df.columns
    if col not in excluded_cols and col not in treadmill_cols_all and not any(base in col for base in incremental_base_cols)
]

print("Available activities:")
print("0. No classification")
print("1. Treadmill (all speeds grouped)")
print("2. Incremental (all phases grouped)")
for i, act in enumerate(activities[2:], start=3):
    print(f"{i}. {act}")

choice_str = input("\nChoose the numbers of activities to display (e.g., 1,3,4): ")
selected_indices = [int(x.strip()) - 1 for x in choice_str.split(',') if x.strip().isdigit()]
activities_to_plot = [activities[i] for i in selected_indices]

print("\nPossible classification criteria:")
print("0. No classification")
print("1. Age_Group")
print("2. Gender")
print("3. Height_Group")
print("4. Weight_Group")
class_choice = input("Choose a classification criterion (0-4): ").strip()

class_map = {
    '0': None,
    '1': 'Age_Group',
    '2': 'Gender',
    '3': 'Height_Group',
    '4': 'Weight_Group'
}
class_col = class_map.get(class_choice, None)

print(f"\nChosen classification: {class_col if class_col else 'None'}")

colors = plt.cm.tab10.colors  # 10-color palette

for activity in activities_to_plot:
    plt.figure(figsize=(10,6))
    stat_label = stat_choice.capitalize()

    if class_col is None:
        # No classification
        if activity == 'Treadmill':
            valid_cols = [col for col in treadmill_cols if col in df.columns]
            if not valid_cols:
                print(f"⚠️ No treadmill data available for {stat_choice}")
                continue
            speeds = [col.split(' ', 1)[1].replace(f" {stat_label}", '') for col in valid_cols]
            data = [df[col].dropna() for col in valid_cols]
            positions = np.arange(len(speeds))
        elif activity == 'Incremental':
            if not incremental_cols:
                print(f"⚠️ No incremental data available for {stat_choice}")
                continue
            phases = [col.split(' - ')[1].replace(f" {stat_label}", '') for col in incremental_cols]
            data = [df[col].dropna() for col in incremental_cols]
            positions = np.arange(len(phases))
        else:
            if activity not in df.columns:
                print(f"⚠️ Activity {activity} not found in the data.")
                continue
            data = df[activity].dropna()
            plt.boxplot(data, patch_artist=True, notch=True, boxprops=dict(facecolor=colors[0]))
            plt.ylabel(f"{stat_label} acceleration")
            plt.title(f"{activity} - {stat_label} acceleration (no classification)")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            continue

        bp = plt.boxplot(data, positions=positions, widths=0.6, patch_artist=True, notch=True)
        for patch in bp['boxes']:
            patch.set_facecolor(colors[0])
        for element in ['whiskers', 'caps', 'medians', 'fliers']:
            for item in bp[element]:
                item.set_color('black')

        plt.xticks(positions, speeds if activity == 'Treadmill' else phases, rotation=30)
        plt.xlabel("Speed" if activity == 'Treadmill' else "Phase")
        plt.ylabel(f"{stat_label} acceleration")
        plt.title(f"{activity} - {stat_label} acceleration (no classification)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    else:
        groups = sorted(df[class_col].dropna().unique())
        if activity == 'Treadmill':
            valid_cols = [col for col in treadmill_cols if col in df.columns]
            if not valid_cols:
                print(f"⚠️ No treadmill data available for {stat_choice}")
                continue
            speeds = [col.split(' ', 1)[1].replace(f" {stat_label}", '') for col in valid_cols]
            speeds_positions = np.arange(len(speeds))
            n_groups = len(groups)
            width = 0.8 / n_groups
            for i, group in enumerate(groups):
                group_data = [df[df[class_col] == group][col].dropna() for col in valid_cols]
                pos = speeds_positions - 0.4 + i*width + width/2
                if len(group_data) != len(pos):
                    print(f"⚠️ Skipping {group} for {activity} — data/position mismatch.")
                    continue
                bp = plt.boxplot(group_data, positions=pos, widths=width, patch_artist=True, notch=True)
                for patch in bp['boxes']:
                    patch.set_facecolor(colors[i % len(colors)])
                for element in ['whiskers', 'caps', 'medians', 'fliers']:
                    for item in bp[element]:
                        item.set_color('black')

            plt.xticks(speeds_positions, speeds, rotation=30)
            plt.xlabel("Speed (km/h)")
            plt.ylabel(f"{stat_label} acceleration")
            plt.title(f"Treadmill - {stat_label} acceleration by speed and {class_col}")

        elif activity == 'Incremental':
            if not incremental_cols:
                print(f"⚠️ No incremental data available for {stat_choice}")
                continue
            phases = [col.split(' - ')[1].replace(f" {stat_label}", '') for col in incremental_cols]
            phase_positions = np.arange(len(phases))
            n_groups = len(groups)
            width = 0.8 / n_groups
            for i, group in enumerate(groups):
                group_data = [df[df[class_col] == group][col].dropna() for col in incremental_cols]
                pos = phase_positions - 0.4 + i*width + width/2
                if len(group_data) != len(pos):
                    print(f"⚠️ Skipping {group} for {activity} — data/position mismatch.")
                    continue
                bp = plt.boxplot(group_data, positions=pos, widths=width, patch_artist=True, notch=True)
                for patch in bp['boxes']:
                    patch.set_facecolor(colors[i % len(colors)])
                for element in ['whiskers', 'caps', 'medians', 'fliers']:
                    for item in bp[element]:
                        item.set_color('black')

            plt.xticks(phase_positions, phases, rotation=30)
            plt.xlabel("Incremental phase")
            plt.ylabel(f"{stat_label} acceleration")
            plt.title(f"Incremental - {stat_label} acceleration by phase and {class_col}")

        else:
            if activity not in df.columns:
                print(f"⚠️ Activity {activity} not found in the data.")
                continue
            data = [df[df[class_col] == g][activity].dropna() for g in groups]
            positions = np.arange(len(groups))
            width = 0.6
            bp = plt.boxplot(data, positions=positions, widths=width, patch_artist=True, notch=True)
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(colors[i % len(colors)])
            for element in ['whiskers', 'caps', 'medians', 'fliers']:
                for item in bp[element]:
                    item.set_color('black')

            plt.xticks(positions, groups, rotation=30)
            plt.xlabel(class_col)
            plt.ylabel(f"{stat_label} acceleration")
            plt.title(f"{activity} - {stat_label} acceleration by group {class_col}")

        patches = [mpatches.Patch(color=colors[i % len(colors)], label=groups[i]) for i in range(len(groups))]
        plt.legend(handles=patches, title=class_col)
        plt.grid(True)
        plt.tight_layout()
        plt.show()