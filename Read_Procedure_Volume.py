import pandas as pd

# Read the CSV file
file_path = "~/Downloads/doctors_and_clinicians_current_data/Utilization.csv"
df = pd.read_csv(file_path)

# Function to handle ranges and convert to minimum number
def get_min_number(value):
    try:
        # If it's a range (e.g., "1-10"), split and take the first number
        if isinstance(value, str) and '-' in value:
            return int(value.split('-')[0])
        # If it's already a number, return it
        return int(value)
    except:
        return value

# Apply the function to the Count column
df['Count'] = df['Count'].apply(get_min_number)

# Group by Procedure_Category and sum the counts
summary_table = df.groupby('Procedure_Category')['Count'].sum().reset_index()

# Sort by Count in descending order
summary_table = summary_table.sort_values('Count', ascending=False)

# Save to Excel
excel_path = "~/Downloads/procedure_volumes_summary.xlsx"
summary_table.to_excel(excel_path, index=False, sheet_name='Procedure Volumes')