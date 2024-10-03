import os
import pandas as pd
from datasets import Dataset

# Function to load and combine all CSV files in the data folder
# def load_data(data_folder):
#     csv_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]
#     df_list = [pd.read_csv(file)[['Question', 'Answer']] for file in csv_files]
#     combined_df = pd.concat(df_list)
    
#     # Convert to Hugging Face Dataset
#     dataset = Dataset.from_pandas(combined_df)
#     dataset = dataset.rename_column("Question", "prompt")
#     dataset = dataset.rename_column("Answer", "response")
    
#     return dataset
csvPth = [
    r"D:\Project\FYP\Final-Chatbot\tinyLama\data\university_about.csv",
    r"D:\Project\FYP\Final-Chatbot\tinyLama\data\greeting.csv",
    r"D:\Project\FYP\Final-Chatbot\tinyLama\data\incomplete_questions.csv",
    r"D:\Project\FYP\Final-Chatbot\tinyLama\data\faculty.csv"
]


def load_data(csv_paths):
    dataframes = []
    
    for file_path in csv_paths:
        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}")
            continue
        
        df = pd.read_csv(file_path)
        print(f"Loaded DataFrame from {file_path} columns:", df.columns.tolist())
        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)
    print("Combined DataFrame columns:", combined_df.columns.tolist())
    
    return Dataset.from_pandas(combined_df)