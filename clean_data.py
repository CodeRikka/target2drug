import pandas as pd
import ast
import re
from tqdm.auto import tqdm

# 配置tqdm来美化pandas apply
tqdm.pandas()

def convert_string_to_list(string):
    try:
        # Remove any leading/trailing whitespace and surrounding square brackets
        clean_string = string.strip().lstrip('[').rstrip(']')
        # Ensure that the string has commas between numbers, which is required by ast.literal_eval
        clean_string = re.sub(r'\s+', ', ', clean_string)
        # Convert the clean string back into a list using ast.literal_eval
        return ast.literal_eval(f'[{clean_string}]')
    except Exception as e:
        # If there's an error, we return None
        return None

# Define the file path
file_path = 'data/processed_data.csv'  # Replace with the actual file path
output_file_path = 'data/processed_data_clean.csv'  # Path for the output file

try:
    # Load the dataset
    df = pd.read_csv(file_path)

    # Apply the conversion to the 'Protein Embedding' column
    # and drop rows where conversion fails
    df['Protein Embedding'] = df['Protein Embedding'].progress_apply(convert_string_to_list)
    df.dropna(subset=['Protein Embedding'], inplace=True)

    # Convert the lists to strings for CSV saving
    df['Protein Embedding'] = df['Protein Embedding'].apply(lambda x: ', '.join(map(str, x)))

    # Reset the index of the cleaned DataFrame
    df.reset_index(drop=True, inplace=True)

    # Save the cleaned DataFrame to a new CSV file
    df.to_csv(output_file_path, index=False)

    print(f"Processed data saved to {output_file_path}")
except Exception as e:
    print(f"Error reading or converting the file: {e}")
