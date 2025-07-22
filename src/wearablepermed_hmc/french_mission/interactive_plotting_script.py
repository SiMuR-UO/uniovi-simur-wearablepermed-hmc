import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# Functions to group continuous variables into categorical bins for better classification in plots
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

# Load the summary Excel file with participant data and computed statistics
summary_file = '/home/simur/git/uniovi-simur-wearablepermed-hmc/src/wearablepermed_hmc/french_mission/summary_participants_mean_min_max.xlsx'
df = pd.read_excel(summary_file)

# Add categorical grouping columns for Age, Height, and Weight
df['Age_Group'] = df['Age'].apply(age_to_group)
df['Height_Group'] = df['Height'].apply(height_to_group)
df['Weight_Group'] = df['Weight'].apply(weight_to_group)

# Fill missing gender values with 'Unknown'
df['Gender'] = df['Gender'].fillna('Unknown')

# Columns to exclude from activity data columns
excluded_cols = ['Age', 'Height', 'Weight', 'Gender', 'Age_Group', 'Height_Group', 'Weight_Group', 'Name']

# Ask user for which statistic to plot (mean, min, or max)
stat_choice = input("Choose statistic to plot: mean, min, or max (default: mean): ").strip().lower()
if stat_choice not in ['mean', 'min', 'max']:
    stat_choice = 'mean'

stat_label = stat_choice.capitalize()  # For matching column suffix

# Identify treadmill and incremental columns based on chosen stat
all_treadmill_cols = [col for col in df.columns if col.startswith('Treadmill') and col.endswith(stat_label)]
all_incremental_cols = [col for col in df.columns if col.startswith('Incremental') and col.endswith(stat_label)]

# Identify other activity columns (excluding treadmill and incremental)
activity_cols = [
    col for col in df.columns
    if col.endswith(f" {stat_label}")
    and col not in all_treadmill_cols
    and col not in all_incremental_cols
]

# Build a clean list of activity names
activities = ['Treadmill', 'Incremental']
for col in activity_cols:
    base = col.rsplit(' ', 1)[0]  # Remove stat suffix
    if base not in activities:
        activities.append(base)

# Show available activities to user
print("Available activities:")
for i, act in enumerate(activities, start=1):
    print(f"{i}. {act}")

# Let user choose which activities to plot by number
choice_str = input("\nChoose the numbers of activities to display (e.g., 1,3,4): ")
selected_indices = [int(x.strip()) - 1 for x in choice_str.split(',') if x.strip().isdigit()]
activities_to_plot = [activities[i] for i in selected_indices if 0 <= i < len(activities)]

# Show classification criteria options
print("\nPossible classification criteria:")
print("0. No classification")
print("1. Age_Group")
print("2. Gender")
print("3. Height_Group")
print("4. Weight_Group")

# Get classification choice from user
class_choice = input("Choose a classification criterion (0-4): ").strip()
class_map = {'0': None, '1': 'Age_Group', '2': 'Gender', '3': 'Height_Group', '4': 'Weight_Group'}
class_col = class_map.get(class_choice, None)

print(f"\nChosen classification: {class_col if class_col else 'None'}")

# Predefined colors for groups
colors = plt.cm.tab10.colors

# Plotting loop for each selected activity
for activity in activities_to_plot:
    plt.figure(figsize=(10,6))

    # No classification chosen: single boxplot per speed/phase/activity
    if class_col is None:
        if activity == 'Treadmill':
            cols = all_treadmill_cols
            if not cols:
                print(f"⚠️ No treadmill data available for {stat_choice}")
                continue
            # Extract speed labels from column names (e.g., '2km/h')
            labels = [col.split(' ')[1] for col in cols]
            data = [df[col].dropna() for col in cols]
            pos = np.arange(len(labels))

        elif activity == 'Incremental':
            cols = all_incremental_cols
            if not cols:
                print(f"⚠️ No incremental data available for {stat_choice}")
                continue
            # Extract phase labels from columns like 'Incremental - Rest Mean'
            labels = [col.split(' ')[1].split()[0] for col in cols]
            data = [df[col].dropna() for col in cols]
            pos = np.arange(len(labels))

        else:
            col = f"{activity} {stat_label}"
            if col not in df.columns:
                print(f"⚠️ Column {col} not found.")
                continue
            # Simple boxplot for single activity column without classification
            plt.boxplot(df[col].dropna(), patch_artist=True, notch=True, boxprops=dict(facecolor=colors[0]))
            plt.title(f"{activity} - {stat_label} acceleration (no classification)")
            plt.ylabel(f"{stat_label} acceleration")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            continue

        # Plot boxplots for treadmill speeds or incremental phases
        bp = plt.boxplot(data, positions=pos, widths=0.6, patch_artist=True, notch=True)
        for patch in bp['boxes']:
            patch.set_facecolor(colors[0])
        # Set black color for other boxplot elements
        for element in ['whiskers', 'caps', 'medians', 'fliers']:
            for item in bp[element]:
                item.set_color('black')

        plt.xticks(pos, labels, rotation=30)
        plt.ylabel(f"{stat_label} acceleration")
        plt.xlabel("Speed" if activity == 'Treadmill' else "Phase")
        plt.title(f"{activity} - {stat_label} acceleration")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Classification chosen: grouped boxplots by category
    else:
        # Get sorted unique groups, ignoring NaNs
        groups = sorted(df[class_col].dropna().unique())

        if activity == 'Treadmill':
            cols = all_treadmill_cols
            if not cols:
                print(f"⚠️ No treadmill data available for {stat_choice}")
                continue
            labels = [col.split(' ')[1] for col in cols]
            pos = np.arange(len(labels))
            width = 0.8 / len(groups)  # Width for each group in each position
            # Plot boxplots side by side for each group per speed
            for i, group in enumerate(groups):
                group_data = [df[df[class_col] == group][col].dropna() for col in cols]
                offsets = pos - 0.4 + i * width + width / 2
                bp = plt.boxplot(group_data, positions=offsets, widths=width, patch_artist=True, notch=True)
                for patch in bp['boxes']:
                    patch.set_facecolor(colors[i % len(colors)])
                for element in ['whiskers', 'caps', 'medians', 'fliers']:
                    for item in bp[element]:
                        item.set_color('black')
            plt.xticks(pos, labels, rotation=30)
            plt.xlabel("Speed (km/h)")

        elif activity == 'Incremental':
            cols = all_incremental_cols
            if not cols:
                print(f"⚠️ No incremental data available for {stat_choice}")
                continue
            labels = []
            for col in cols:
                parts = col.split(' - ')
                if len(parts) > 1:
                    label = parts[1].split()[0]  # e.g., 'Rest' from 'Incremental - Rest Mean'
                else:
                    label = col.replace(f" {stat_label}", "")  # fallback label
                labels.append(label)
            pos = np.arange(len(labels))
            width = 0.8 / len(groups)
            # Plot boxplots side by side for each group per phase
            for i, group in enumerate(groups):
                group_data = [df[df[class_col] == group][col].dropna() for col in cols]
                offsets = pos - 0.4 + i * width + width / 2
                bp = plt.boxplot(group_data, positions=offsets, widths=width, patch_artist=True, notch=True)
                for patch in bp['boxes']:
                    patch.set_facecolor(colors[i % len(colors)])
                for element in ['whiskers', 'caps', 'medians', 'fliers']:
                    for item in bp[element]:
                        item.set_color('black')
            plt.xticks(pos, labels, rotation=30)
            plt.xlabel("Phase")

        else:
            # For other activities, boxplot by classification group
            col = f"{activity} {stat_label}"
            if col not in df.columns:
                print(f"⚠️ Column {col} not found.")
                continue
            data = [df[df[class_col] == g][col].dropna() for g in groups]
            pos = np.arange(len(groups))
            bp = plt.boxplot(data, positions=pos, widths=0.6, patch_artist=True, notch=True)
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(colors[i % len(colors)])
            for element in ['whiskers', 'caps', 'medians', 'fliers']:
                for item in bp[element]:
                    item.set_color('black')
            plt.xticks(pos, groups, rotation=30)
            plt.xlabel(class_col)

        plt.ylabel(f"{stat_label} acceleration")
        plt.title(f"{activity} - {stat_label} acceleration by {class_col}")

        # Add legend for groups
        legend_patches = [mpatches.Patch(color=colors[i % len(colors)], label=group) for i, group in enumerate(groups)]
        plt.legend(handles=legend_patches, title=class_col)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
