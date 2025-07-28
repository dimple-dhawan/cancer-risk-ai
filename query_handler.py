import pandas as pd
import openai
import ast
import logging

# Load dataset
DATA_PATH = "cancer_data.csv"
df = pd.read_csv(DATA_PATH)

# Setup logging
logging.basicConfig(filename="query_log.txt", level=logging.INFO,
                    format="%(asctime)s - %(message)s")

# Set your OpenAI API key here (replace this before running)
openai.api_key = ""

def answer_question(question):
    """
    Given a natural language question, generate Python code to query the dataframe and return the result.
    """
    prompt = f"""
    You are a data assistant. Convert the following user question into safe Python Pandas code.
    The DataFrame is called 'df'. Your output should be a single expression, not full scripts.
    Avoid any imports, file access, or OS commands.

    Question: {question}
    Python code:
    """

    try:
        # Call GPT to generate code
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        code = response.choices[0].message.content.strip()
        logging.info(f"Question: {question}\nGenerated code: {code}")

        # Security: Check AST to ensure no unsafe operations
        tree = ast.parse(code, mode='eval')
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Exec, ast.Call)):
                return "Rejected: Unsafe code detected."

        # Evaluate the code safely
        result = eval(code, {'df': df})
        return result

    except Exception as e:
        logging.error(f"Error processing question: {question} - {str(e)}")
        return f"Error: {str(e)}"

# Example (remove in production):
if __name__ == "__main__":
    q = "What is the average age of patients with high cancer levels?"
    print("Answer:", answer_question(q))
