# from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
# from datasets import DatasetDict
# from dataset_processing import load_data

# # Load the dataset
# data_folder = "D:\\Project\\FYP\\Final-Chatbot\\tinyLama\\data"
# dataset = load_data(data_folder)

# # Load the model and tokenizer
# model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# # Tokenize the dataset
# def tokenize_function(examples):
#     return tokenizer(examples["prompt"], padding="max_length", truncation=True, max_length=512)

# tokenized_dataset = dataset.map(tokenize_function, batched=True)

# # Split the dataset for training and evaluation
# train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
# dataset_dict = DatasetDict({"train": train_test_split["train"], "test": train_test_split["test"]})

# # Fine-tuning parameters
# training_args = TrainingArguments(
#     output_dir="D:\\Project\\FYP\\Final-Chatbot\\tinyLama\\models\\fine_tuned_model",  # Save model here
#     evaluation_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=2,
#     per_device_eval_batch_size=2,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     push_to_hub=False,
# )

# # Initialize Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset_dict["train"],
#     eval_dataset=dataset_dict["test"],
# )

# # Train the model
# trainer.train()

# # Save the fine-tuned model
# trainer.save_model("D:\\Project\\FYP\\Final-Chatbot\\tinyLama\\models\\fine_tuned_model")
# from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
# from datasets import DatasetDict
# from dataset_processing import load_data

# # Load the dataset
# data_folder = "D:\\Project\\FYP\\Final-Chatbot\\tinyLama\\data"
# dataset = load_data(data_folder)

# # Load the model and tokenizer
# model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# # Tokenize the dataset and add labels (which are the same as input_ids)
# def tokenize_function(examples):
#     inputs = tokenizer(examples["prompt"], padding="max_length", truncation=True, max_length=512)
    
#     # Create labels, same as inputs but with shifting for language modeling
#     inputs["labels"] = inputs["input_ids"].copy()  # Copy the input_ids as labels

#     return inputs

# tokenized_dataset = dataset.map(tokenize_function, batched=True)

# # Split the dataset for training and evaluation
# train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
# dataset_dict = DatasetDict({"train": train_test_split["train"], "test": train_test_split["test"]})

# # Fine-tuning parameters
# training_args = TrainingArguments(
#     output_dir="D:\\Project\\FYP\\Final-Chatbot\\tinyLama\\models\\fine_tuned_model",  # Save model here
#     evaluation_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=2,
#     per_device_eval_batch_size=2,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     push_to_hub=False,
# )

# # Initialize Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset_dict["train"],
#     eval_dataset=dataset_dict["test"],
# )

# # Train the model
# trainer.train()

# # Save the fine-tuned model
# trainer.save_model("D:\\Project\\FYP\\Final-Chatbot\\tinyLama\\models\\fine_tuned_model")
# from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
# from datasets import DatasetDict, load_dataset
# import os

# # Load and process dataset
# def load_data(data_folder):
#     # Load the CSV files from the folder and create a Hugging Face Dataset
#     csv_files = [os.path.join(data_folder, file) for file in os.listdir(data_folder) if file.endswith(".csv")]
#     dataset = load_dataset('csv', data_files=csv_files)  # Load all CSV files in the folder
#     return dataset

# # Specify the data folder containing CSV files
# data_folder = "D:\\Project\\FYP\\Final-Chatbot\\tinyLama\\data"
# dataset = load_data(data_folder)

# # Load model and tokenizer
# model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# # Tokenize the dataset, using "Question" as the input and "Answer" as the label
# def tokenize_function(examples):
#     # Combine question and answer for the model input: Question + Answer
#     inputs = tokenizer(examples["Question"], padding="max_length", truncation=True, max_length=512)
#     # Tokenize the Answer to use as labels
#     with tokenizer.as_target_tokenizer():
#         labels = tokenizer(examples["Answer"], padding="max_length", truncation=True, max_length=512)

#     # Set input_ids and labels
#     inputs["labels"] = labels["input_ids"]
    
#     return inputs

# # Tokenize dataset
# tokenized_dataset = dataset.map(tokenize_function, batched=True)

# # Split the dataset for training and evaluation
# train_test_split = tokenized_dataset['train'].train_test_split(test_size=0.2)
# dataset_dict = DatasetDict({"train": train_test_split["train"], "test": train_test_split["test"]})

# # Define training arguments
# training_args = TrainingArguments(
#     output_dir="D:\\Project\\FYP\\Final-Chatbot\\tinyLama\\models\\fine_tuned_model",  # Save directory
#     evaluation_strategy="epoch",   # Evaluate model after each epoch
#     learning_rate=2e-5,            # Set learning rate
#     per_device_train_batch_size=2,  # Batch size for training
#     per_device_eval_batch_size=2,   # Batch size for evaluation
#     num_train_epochs=3,             # Number of epochs
#     weight_decay=0.01,              # Regularization
#     push_to_hub=False,              # Not pushing to Hugging Face Hub
#     logging_dir="./logs",           # Directory for logging
#     logging_steps=10,               # Log every 10 steps
#     save_steps=500,                 # Save every 500 steps
# )

# # Initialize the Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset_dict["train"],
#     eval_dataset=dataset_dict["test"],
# )

# # Train the model
# trainer.train()

# # Save the fine-tuned model
# trainer.save_model("D:\\Project\\FYP\\Final-Chatbot\\tinyLama\\models\\fine_tuned_model")

# # Save the tokenizer as well to ensure it matches the fine-tuned model
# tokenizer.save_pretrained("D:\\Project\\FYP\\Final-Chatbot\\tinyLama\\models\\fine_tuned_model")
# from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
# from datasets import DatasetDict
# from dataset_processing import load_data

# # Load the dataset
# data_folder = "D:\\Project\\FYP\\Final-Chatbot\\tinyLama\\data"
# dataset = load_data(data_folder)

# # Load the model and tokenizer
# model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# # Tokenize the dataset and add labels
# def tokenize_function(examples):
#     inputs = tokenizer(examples["Question"], padding="max_length", truncation=True, max_length=512)
    
#     # Create labels, same as inputs but with shifting for language modeling
#     inputs["labels"] = inputs["input_ids"].copy()  # Copy the input_ids as labels

#     return inputs

# # Tokenizing the dataset
# tokenized_dataset = dataset.map(tokenize_function, batched=True)

# # Split the dataset for training and evaluation
# train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
# dataset_dict = DatasetDict({"train": train_test_split["train"], "test": train_test_split["test"]})

# # Fine-tuning parameters
# training_args = TrainingArguments(
#     output_dir="D:\\Project\\FYP\\Final-Chatbot\\tinyLama\\models\\fine_tuned_model",
#     evaluation_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=2,  # Reduced batch size
#     per_device_eval_batch_size=2,    # Reduced batch size
#     num_train_epochs=2,               # Reduced number of epochs
#     weight_decay=0.01,
#     gradient_accumulation_steps=2,    # Simulating a larger batch size
#     push_to_hub=False,
# )

# # Initialize Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset_dict["train"],
#     eval_dataset=dataset_dict["test"],
# )

# # Train the model
# trainer.train()

# # Save the fine-tuned model
# trainer.save_model("D:\\Project\\FYP\\Final-Chatbot\\tinyLama\\models\\fine_tuned_model")
# from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
# from datasets import DatasetDict
# from dataset_processing import load_data

# # Load the dataset
# # data_folder = "D:\\Project\\FYP\\Final-Chatbot\\tinyLama\\data"
# data_folder = [
#     r"D:\Project\FYP\Final-Chatbot\tinyLama\data\university_about.csv",
#     r"D:\Project\FYP\Final-Chatbot\tinyLama\data\greeting.csv",
#     r"D:\Project\FYP\Final-Chatbot\tinyLama\data\incomplete_questions.csv",
#     r"D:\Project\FYP\Final-Chatbot\tinyLama\data\faculty.csv"
# ]
# dataset = load_data(data_folder)

# # Load the model and tokenizer
# model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# # Tokenize the dataset and add labels
# def tokenize_function(examples):
#     inputs = tokenizer(examples["Question"], padding="max_length", truncation=True, max_length=512)
#     inputs["labels"] = inputs["input_ids"].copy()  # Copy the input_ids as labels
#     return inputs

# # Debugging: print the dataset structure
# print(dataset)  # Inspect dataset structure

# # Tokenizing the dataset
# tokenized_dataset = dataset.map(tokenize_function, batched=True)

# # Split the dataset for training and evaluation
# train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
# dataset_dict = DatasetDict({"train": train_test_split["train"], "test": train_test_split["test"]})

# # Fine-tuning parameters
# training_args = TrainingArguments(
#     output_dir="D:\\Project\\FYP\\Final-Chatbot\\tinyLama\\models\\fine_tuned_model",
#     evaluation_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=2,
#     per_device_eval_batch_size=2,
#     num_train_epochs=2,
#     weight_decay=0.01,
#     gradient_accumulation_steps=2,
#     push_to_hub=False,
# )

# # Initialize Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset_dict["train"],
#     eval_dataset=dataset_dict["test"],
# )

# # Train the model
# trainer.train()

# # Save the fine-tuned model
# trainer.save_model("D:\\Project\\FYP\\Final-Chatbot\\tinyLama\\models\\fine_tuned_model")
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import DatasetDict
from dataset_processing import load_data

# Load the dataset
data_folder = [
    r"D:\Project\FYP\Final-Chatbot\tinyLama\data\university_about.csv",
    r"D:\Project\FYP\Final-Chatbot\tinyLama\data\greeting.csv",
    r"D:\Project\FYP\Final-Chatbot\tinyLama\data\incomplete_questions.csv",
    r"D:\Project\FYP\Final-Chatbot\tinyLama\data\faculty.csv"
]
dataset = load_data(data_folder)

# Load the model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenize the dataset and add labels
def tokenize_function(examples):
    inputs = tokenizer(examples["Question"], padding="max_length", truncation=True, max_length=512)
    inputs["labels"] = inputs["input_ids"].copy()  # Copy the input_ids as labels
    return inputs

# Debugging: print the dataset structure
print("Dataset loaded:", dataset)  # Inspect dataset structure

# Tokenizing the dataset
print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Split the dataset for training and evaluation
print("Splitting dataset into training and testing sets...")
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
dataset_dict = DatasetDict({"train": train_test_split["train"], "test": train_test_split["test"]})

# Fine-tuning parameters
training_args = TrainingArguments(
    output_dir="D:\\Project\\FYP\\Final-Chatbot\\tinyLama\\models\\fine_tuned_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=2,
    weight_decay=0.01,
    gradient_accumulation_steps=2,
    push_to_hub=False,
    save_steps=500,  # Save checkpoint every 500 steps
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["test"],
)

# Train the model
print("Starting training...")
trainer.train()

# Save the fine-tuned model
print("Saving the fine-tuned model...")
trainer.save_model("D:\\Project\\FYP\\Final-Chatbot\\tinyLama\\models\\fine_tuned_model")
print("Model saved successfully!")
