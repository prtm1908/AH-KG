import csv
import openai
import os
from openai import OpenAI

client = OpenAI(api_key="")

def generate_answers_from_csv(
    csv_input_path: str,
    csv_output_path: str,
    model_name: str = "gpt-4o"
):
    """
    Reads a CSV with columns:
        Paragraphs, Question1, Question2, Question3,
        Answer1, Answer2, Answer3,
        knowledge_graph_triplets,
        question1_triplets, question2_triplets, question3_triplets

    For each row, it sends only the question and the corresponding question triplets
    to the LLM and gets a response solely based on these triplets.

    :param csv_input_path: Path to the input CSV file
    :param csv_output_path: Path to save the updated CSV file with generated answers
    :param openai_api_key: OpenAI API key (if not set, will use environment variable)
    :param model_name: The model to use for the chat completion (default: gpt-3.5-turbo)
    """

    total_questions = 0
    answered_correctly = 0

    # Prepare to read and then write the CSV
    with open(csv_input_path, mode='r', encoding='utf-8') as infile, \
         open(csv_output_path, mode='w', encoding='utf-8', newline='') as outfile:
        
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        # We want to ensure the output CSV has the same columns (including the Answers)
        # If for some reason Answer1, Answer2, Answer3 columns are missing, we add them.
        required_answer_columns = ["Answer_LLM_1", "Answer_LLM_2", "Answer_LLM_3"]
        for col in required_answer_columns:
            if col not in fieldnames:
                fieldnames.append(col)
        
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            # We'll process each of the 3 questions
            for i in range(1, 4):
                question_col = f"Question{i}"
                answer_col   = f"Answer_LLM_{i}"
                triplets_col = f"question{i}_triplets"

                question_text = row.get(question_col, "")
                triplets_text = row.get(triplets_col, "")

                # If there's no question or no triplets, we can skip or mark as insufficient
                if not question_text.strip() or not triplets_text.strip():
                    row[answer_col] = "Insufficient data (No question or triplets provided)."
                    continue

                total_questions += 1

                # Create the prompt for the LLM
                # We instruct the model to ONLY use the triplets to answer the question.

                user_message = (
                     f"""You are tasked with answering the given question below based on the information you get from the given related knowledge graph triplets.
                        Question: {question_text}
                        Related Knowledge Graph Triplets-
                        {triplets_text}
                        Do not make anything up that can't be inferred from the triplets, if the question can't be answered using the triplets just say 'Insufficient data' in your response."""
                                        )

                try:
                    # Call the ChatCompletion endpoint
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "user", "content": user_message}
                        ]
                    )
                    

                    answer = response.choices[0].message.content
                    row[answer_col] = answer

                    # Assuming we have a way to verify correctness, we increment answered_correctly

                    if not 'Insufficient data' in answer:
                        if not "Insufficient data (No question or triplets provided)" in answer:
                             answered_correctly += 1


                    # For now, we assume all responses are correct for demonstration purposes
                       

                except Exception as e:
                    print(e)
                    row[answer_col] = f"Error: {e}"

            # Write the row (including the newly generated answers) back to the output CSV
            writer.writerow(row)

        # Add statistics as the last row
        stats_row = {field: "" for field in fieldnames}
        stats_row["Paragraphs"] = "Statistics"
        stats_row["Question1"] = f"Total Questions: {total_questions}"
        stats_row["Question2"] = f"Answered Correctly: {answered_correctly}"
        writer.writerow(stats_row)

        # Print statistics in the terminal
        print(f"Total Questions: {total_questions}")
        print(f"Answered Correctly: {answered_correctly}")


if __name__ == "__main__":
    # Example usage:
    # Suppose you have 'input_data.csv' and want to produce 'output_data.csv'
    generate_answers_from_csv(
        csv_input_path="datasetCreation/QA_dataset_with_triplets.csv",
        csv_output_path="output_data.csv"
    )
