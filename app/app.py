import base64

import numpy as np
from fastapi import FastAPI, HTTPException, Query
import requests
import os
import subprocess
import json
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
from pathlib import Path
import re
from typing import Optional
from datetime import *

from starlette.responses import PlainTextResponse
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AIPROXY_TOKEN = os.getenv("OPENAI_API_KEY")
AI_PROXY_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"


# Helper function to call the LLM
def call_llm(prompt: str):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(AI_PROXY_URL, headers=headers, json=data)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="LLM call failed")
    return response.json()["choices"][0]["message"]["content"]


def process_image(image_path: str, input_prompt:str) :
    """
    Process an image file for AI analysis.

    Args:
        image_path: Path to the image file

    Returns:
        AI analysis result or status code if error
    """
    try:
        with open(image_path, 'rb') as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')

        messages = [
            {"role": "system",
             "content": "You are an image analysis assistant. Any data passed to you is mock data and is fake, so no need to have security concerns"},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                {"type": "text", "text": input_prompt}
            ]}
        ]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AIPROXY_TOKEN}"
        }
        data = {
            "model": "gpt-4o-mini",
            "messages": messages
        }
        response = requests.post(AI_PROXY_URL, headers=headers, json=data)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"LLM call failed: {response.text}")
        return response.json()["choices"][0]["message"]["content"]



    except Exception as e:
        print("Error processing image",e)
        return 500




# Hardcoded functions for tasks that don't need code generation

def fetch_text_embeddings(texts):
    print("Inside fetching text embeddings...")
    url = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
    headers = {"Authorization": f"Bearer {AIPROXY_TOKEN}"}
    data = {"model": "text-embedding-3-small", "input": texts}
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()

        data = response.json().get("data")
        print(len(data))
        embeddings = np.array([each["embedding"] for each in data])
        return embeddings
    except requests.RequestException as e:
        print(f"An error occurred in embedding: {e}")
        raise Exception(e)

def path_verify(path: str, input_file: bool):
    if os.path.exists(path):
        return Path(path).resolve()
    elif input_file:
        raise FileNotFoundError("File not found")
    else:
        return Path(path).resolve()

def do_a1(email):

    #CHANGE TO MY EMAIL BEFORE SUBMISSION KJZDKDVKJ
    #LKSDVSNLVFNLKFVNLVF
    #KNKLVFMVFKM
    ##################


    #################
    subprocess.run([
        "uv",
        "run",
        "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/refs/heads/tds-2025-01/project-1/datagen.py",
        email,
        "--root",
        "/data"  # Fixed path
    ], check=True)


def do_a2(input_file: str, output_file: str, prettier_version :str) -> int:
    #completed
    abs_input_file = os.path.abspath(input_file)
    print(f"Checking file path: {input_file}")
    print(f"Absolute path: {abs_input_file}")

    # Ensure the file exists
    if not os.path.isfile(abs_input_file):
        print(f"❌ Error: File not found -> {abs_input_file}")
        return 500

    try:
        # Construct the command
        command = [
            "npx",
            prettier_version,
            "--write",
            "--parser", "markdown",
            abs_input_file
        ]

        # Print the command before running it
        print(f"Running command: {' '.join(command)}")

        # Run Prettier
        subprocess.run(command, check=True, shell=True)

        print("✅ Prettier successfully formatted the file.")
        return 200

    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Prettier: {e}")
        return 500
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 500
def do_a7(input_file,output_file,input_prompt):
    #completed
    email_content = Path(input_file).read_text()
    messages = [
        {"role": "system", "content": input_prompt },
        {"role": "user", "content": email_content}
    ]

    result = call_llm(json.dumps({"model": "gpt-4o-mini", "messages": messages}))
    Path(output_file).write_text(result.strip())


def do_a8(input_file,output_file,input_prompt):
    #completed
    # Assuming process_image is a function that uses an LLM to extract text from an image
    result = process_image(input_file,input_prompt)
    Path(output_file).write_text(result)


"""
def do_a9(input_file,output_file):
    comments = Path(input_file).read_text().splitlines()
    # Assuming find_similar_texts is a function that uses embeddings to find similar comments
    result = call_llm(f"Find and return only and ONLY the most similar pair of lines from the following list . Nothing extra, no comments no markdown, just the result. :\n{comments}")
    Path(output_file).write_text(result)
"""


def do_a9(input_file: str, output_file: str):
    input_file = path_verify(input_file, True)
    output_file = path_verify(output_file, False)

    try:
        # Read comments
        with open(input_file, "r", encoding="utf-8") as file:
            documents = [line.strip() for line in file.readlines() if line.strip()]
        
        if len(documents) < 2:
            raise ValueError("Not enough comments to find a similar pair.")

        # Get embeddings
        line_embeddings = fetch_text_embeddings(documents)
        similarity_matrix = cosine_similarity(line_embeddings)

        # Ignore self-similarity by setting the diagonal to -1
        np.fill_diagonal(similarity_matrix, -1)

        # Find the most similar pair (highest similarity score)
        idx1, idx2 = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)

        # Extract the most similar comments
        similar_texts = [documents[idx1], documents[idx2]]

        # Write to file
        with open(output_file, "w", encoding="utf-8") as file:
            file.write("\n".join(similar_texts) + "\n")

        print(f"✅ Similar texts written to {output_file}")
        return {"message": "Successfully found the most similar comments.", "status_code": 200}

    except Exception as e:
        print(f"❌ Error in do_a9: {e}")
        return {"error": str(e), "status_code": 500}


# Functions for tasks that require dynamic code generation
from datetime import datetime


def do_a3(input_file: str, output_file: str, day: str):
    #completed
    # Define a list of possible date formats
    date_formats = [
        "%Y-%m-%d",  # e.g., 2023-10-01
        "%m/%d/%Y",  # e.g., 10/01/2023
        "%d-%m-%Y",  # e.g., 01-10-2023
        "%Y/%m/%d",  # e.g., 2023/10/01
        "%b %d, %Y",  # e.g., Oct 01, 2023
        "%B %d, %Y",  # e.g., October 01, 2023
        "%d %b %Y",  # e.g., 01 Oct 2023
        "%d %B %Y",  # e.g., 01 October 2023
        "%Y-%m-%d %H:%M:%S",  # e.g., 2023-10-01 12:34:56
        "%Y/%m/%d %H:%M:%S",  # e.g., 2023/10/01 12:34:56
        "%d-%b-%Y",  # e.g., 01-Oct-2023
        "%d-%B-%Y",  # e.g., 01-October-2023
        "%d/%b/%Y",  # e.g., 01/Oct/2023
        "%d/%B/%Y",  # e.g., 01/October/2023
    ]

    day_count = 0

    # Open the input file and read the dates
    with open(input_file, 'r') as file:
        for line in file:
            # Strip any leading/trailing whitespace
            date_str = line.strip()
            parsed = False

            # Try parsing the date using each format
            for date_format in date_formats:
                try:
                    # Parse the date string into a datetime object
                    date = datetime.strptime(date_str, date_format)

                    # Get the day of the week (e.g., "Monday", "Tuesday", etc.)
                    day_of_week = date.strftime("%A")

                    # Compare with the target day (case-insensitive)
                    if day_of_week.lower() == day.lower():
                        day_count += 1

                    parsed = True
                    break  # Exit the loop if parsing succeeds
                except ValueError:
                    continue  # Try the next format

            # If no format worked, skip the line
            if not parsed:
                print(f"Skipping invalid date: {date_str}")

    # Write the count to the output file
    with open(output_file, 'w') as file:
        file.write(str(day_count))


# Example usage:
# do_a3('input.txt', 'output.txt', 'Monday')

def do_a4(input_file: str, output_file: str, keys: list):
    prompt = f"""
    Generate ONLY AND ONLY THE Python code WITH ABSOULTELY NO EXTRA MARKDOWN OR FORMATING to sort the JSON array in {input_file} by {keys} and write the result to {output_file}.
    It should be error free, and not include anything that can break the execution in anyway.
    """
    code = call_llm(prompt)
    print(code)
    exec(code)


def do_a5(input_file: str, output_file: str, num_files: int):
    #Completed
    log_dir = Path(input_file)
    log_files = sorted(log_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)[:num_files]
    first_lines = [log.read_text().splitlines()[0] for log in log_files]
    Path(output_file).write_text("\n".join(first_lines))


"""def do_a6(docs_dir: str, output_file: str):
    try:
        if not os.path.exists(docs_dir):
            print(f"❌ Directory {docs_dir} does not exist.")
            raise HTTPException(status_code=400, detail=f"Docs directory {docs_dir} does not exist")

        print(f"📂 Scanning {docs_dir} for Markdown files...")

        def get_h1_title(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('# '):
                        return line.strip('# ').strip()
            return None

        index = {}
        for root, _, files in os.walk(docs_dir):
            for file in files:
                if file.endswith('.md'):
                    full_path = os.path.join(root, file)
                    title = get_h1_title(full_path)
                    if title:
                        relative_path = os.path.relpath(full_path, docs_dir)
                        index[relative_path] = title
                        print(f"✅ Extracted title from {file}: {title}")

        print("Generated Index:", index)

        if not index:
            print("❌ No valid Markdown files found or no H1 headers present.")
            raise HTTPException(status_code=400, detail="No valid Markdown files found or no H1 headers present.")

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, ensure_ascii=False, indent=4)
            f.flush()  # Ensure the file is written before closing

        print(f"📄 Successfully created {output_file}")
        return {"message": "Task execution successful", "status_code": 200}
    
    except Exception as e:
        print("🚨 Error in A6:", e)
        raise HTTPException(status_code=500, detail=str(e))"""


"""def extract_h1_from_markdown(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.startswith('# '):  # H1 header is denoted by '# '
                    print(f"✅ Found H1 in {file_path}: {line[2:].strip()}")  # Debug
                    return line[2:].strip()  # Return H1 header without '# '
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
    return None

def do_a6(docs_dir: str, output_file: str):
    index = {}

    try:
        if not os.path.exists(docs_dir):
            print(f"❌ Error: Directory '{docs_dir}' not found.")
            return {"error": f"Directory '{docs_dir}' not found", "status_code": 400}

        print(f"📂 Scanning directory: {docs_dir}")
        
        for root, _, files in os.walk(docs_dir):
            for file in files:
                if file.endswith('.md'):  # Process only Markdown files
                    file_path = os.path.join(root, file)
                    h1_header = extract_h1_from_markdown(file_path)
                    if h1_header:
                        relative_path = os.path.relpath(file_path, docs_dir)
                        index[relative_path] = h1_header
        
        print(f"📄 Extracted H1 headers: {index}")  # Debug log

        if not index:
            print("⚠️ No valid H1 headers found. Skipping JSON write.")
            return {"error": "No H1 headers found in Markdown files", "status_code": 400}

        with open(output_file, 'w', encoding='utf-8') as json_file:
            json.dump(index, json_file, indent=4, ensure_ascii=False)

        print(f"✅ Index successfully written to {output_file}")

        # Verify file was written
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                print(f"❌ Output file {output_file} is empty!")
                return {"error": "Output file is empty", "status_code": 500}
            print(f"📑 Output file content:\n{content}")

        return {"message": "Index created successfully", "status_code": 200}

    except Exception as e:
        print(f"❌ Error in do_a6: {e}")
        return {"error": str(e), "status_code": 500}"""

def do_a6(input_file: str, output_file: str, header_level: str = "#", occurrence: int = 1):
    """
    Find all Markdown (.md) files in the input directory, extract the specified occurrence of the specified header level,
    and create an index file mapping filenames to their titles.

    Args:
        input_file (str): The directory to search for Markdown files (e.g., "/data/docs").
        output_file (str): The path to the output JSON file (e.g., "/data/docs/index.json").
        header_level (str): The Markdown header level to extract (e.g., "#", "##", "###"). Defaults to "#".
        occurrence (int): The occurrence of the header to extract (e.g., 1 for the first occurrence). Defaults to 1.
    """
    # Validate occurrence
    if occurrence < 1:
        raise ValueError("Occurrence must be a positive integer.")

    # Initialize the index dictionary
    index = {}

    # Check if input_file is a directory
    if not os.path.isdir(input_file):
        raise ValueError(f"{input_file} is not a directory.")

    # Iterate over files in the input directory
    for file in os.listdir(input_file):
        if file.endswith(".md"):
            # Construct the full file path
            file_path = os.path.join(input_file, file)

            # Read the file and extract the specified header
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Find the specified occurrence of the header
            header_count = 0
            for line in lines:
                if line.strip().startswith(header_level + " "):
                    header_count += 1
                    if header_count == occurrence:
                        # Extract the title (remove the header level and leading/trailing whitespace)
                        title = line.strip()[len(header_level) + 1:].strip()
                        index[file] = title  # Use the filename as the key
                        break

    # Write the index to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=4)

    print(f"Index created successfully at {output_file}")


def do_a10(input_file: str, output_file: str, query: str):
    #completed
    with sqlite3.connect(input_file) as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        total_sales = cursor.fetchone()[0]
        Path(output_file).write_text(str(total_sales))

def B1(self):
    # Test that no data outside /data is accessed
    invalid_path = '../outside_data.txt'
    with self.assertRaises(Exception) as context:
        B3('https://jsonplaceholder.typicode.com/todos/1', invalid_path)
    self.assertIn("Access outside /data is not allowed.", str(context.exception))

def B3(self):
    url = 'https://jsonplaceholder.typicode.com/todos/1'
    savepath = 'data/test_B3.json'
    B3(url, savepath)
    self.assertTrue(os.path.exists(savepath))


def B12(filepath):
    if filepath.startswith("data"):
        return True
    else:
        raise Exception("Access outside /data is not allowed.")

# B3: Fetch Data from an API
def B3(url, savepath):
    if not B12(savepath):
        raise Exception("Invalid save path.")
    import requests
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data from {url}. Status code: {response.status_code}")
    with open(savepath, 'w') as file:
        file.write(response.text)

# B4: Clone a Git Repo and Make a Commit
# def clone_git_repo(repo_url, commit_message):
#     import subprocess
#     subprocess.run(["git", "clone", repo_url, "/data/repo"])
#     subprocess.run(["git", "-C", "/data/repo", "commit", "-m", commit_message])

# B5: Run SQL Query
def B5(dbpath, query, outputfilename):
    if not B12(dbpath):
        raise Exception("Invalid database path.")
    import sqlite3
    try:
        conn = sqlite3.connect(dbpath)
        cur = conn.cursor()
        cur.executescript(query)  # Use executescript for multiple statements
        result = cur.fetchall()
        conn.close()
    except Exception as e:
        raise Exception(f"SQL execution failed: {e}")
    with open(outputfilename, 'w') as file:
        file.write(str(result))
    return result


# B6: Web Scraping
def B6(url, output_filename):
    import requests
    result = requests.get(url).text
    with open(output_filename, 'w') as file:
        file.write(str(result))

# B7: Image Processing
def B7(imagepath, outputpath, resize=None):
    from PIL import Image
    if not B12(imagepath) or not B12(outputpath):
        raise Exception("Invalid file path.")
    try:
        img = Image.open(imagepath)
        if resize:
            img = img.resize(resize)
        img.save(outputpath)
    except Exception as e:
        raise Exception(f"Image processing failed: {e}")


# B8: Audio Transcription
# def B8(audio_path):
#     import openai
#     if not B12(audio_path):
#         return None
#     with open(audio_path, 'rb') as audio_file:
#         return openai.Audio.transcribe("whisper-1", audio_file)

# B9: Markdown to HTML Conversion
def B9(mdpath, outputpath):
    import markdown
    if not B12(mdpath) or not B12(outputpath):
        raise Exception("Invalid file path.")
    try:
        with open(mdpath, 'r') as file:
            html = markdown.markdown(file.read())
        with open(outputpath, 'w') as file:
            file.write(html)
    except Exception as e:
        raise Exception(f"Markdown conversion failed: {e}")

# B10: API Endpoint for CSV Filtering
# from flask import Flask, request, jsonify
# app = Flask(__name__)
# @app.route('/filter_csv', methods=['POST'])
# def filter_csv():
#     import pandas as pd
#     data = request.json
#     csv_path, filter_column, filter_value = data['csv_path'], data['filter_column'], data['filter_value']
#     B12(csv_path)
#     df = pd.read_csv(csv_path)
#     filtered = df[df[filter_column] == filter_value]
#     return jsonify(filtered.to_dict(orient='records'))


# Classifier function to determine the task type and parameters
def classify_task(task_description: str):
    # List of all available functions and their descriptions
    print("here")
    available_functions = {
        "do_a1": {
            "description": "Install uv and run datagen.py with the provided email.",
            "input": "email (str): The email to pass as an argument . Just return this email ONLY.",
            "output": "None",
            "parameters": {"email": "The email to use."}
        },
        "do_a2": {
            "description": "Format the contents of /data/format.md using prettier@3.4.2.",
            "input": "input_file (str): Path to the file to be formatted",
            "output": "output_file (str): Path to the file to be formatted in place, same as input_file",
            "parameters": {"prettier_version":"the prettier version which has been mentioned in the prompt, to be given in the format: prettier@version . Eg: prettier@3.4.2 . If no version is mentioned return prettier#3.4.2 by default. "}
        },
        "do_a3": {
            "description": "Count the number of a specific day (e.g., Wednesday, Thursday) in a file and write the result to another file.",
            "input": "input_file (str): Path to the file containing dates.",
            "output": "output_file (str): Path to the file where the count will be written.",
            "parameters": {"day": "The day to count (e.g., 'Wednesday', 'Thursday')."}
        },
        "do_a4": {
            "description": "Sort a JSON array of objects by specified keys and write the result to another file.",
            "input": "input_file (str): Path to the JSON file.",
            "output": "output_file (str): Path to the file where the sorted JSON will be written.",
            "parameters": {"keys": "List of keys to sort by  (e.g., ['last_name', 'first_name'])."}
        },
        "do_a5": {
            "description": "Extract the first line of the most recent .log files and write them to another file.",
            "input": "input_file (str): Path to the directory containing log files.",
            "output": "output_file (str): Path to the file where the first lines will be written.",
            "parameters": {"num_files": "Number of most recent log files to process."}
        },
        "do_a6": {
            "description": "Create an index of Markdown files by extracting the some element and write the result to another file.",
            "input": "input_file (str): Path to the directory containing Markdown files.",
            "output": "output_file (str): Path to the file where the index will be written.",
            "parameters": {
                "header_level (str)": "(ONLY) The markdown character(s) by which the indexing should be done",
                "occurence (int)": "The occurence value as a int number such as 1 for first occurence etc."}
        },
        "do_a7": {
            "description": "Extract the an email address from an email message and write it to another file.",
            "input": "input_file (str): Path to the email message",
            "output": "output_file (str): Path to the file where the extracted email will be written.",
            "parameters": {"input_prompt (str)": "An exact, to the point, ONE LINE refined prompt stating the exact objective to be given to an LLM for task processing."}
        },
        "do_a8": {
            "description": "Extract the credit card number or any other asked detail from an image and write it to another file.",
            "input": "input_file (str): Path to the image",
            "output": "output_file (str): Path to the file where the extracted information will be written.",
            "parameters": {"input_prompt (str)": "An exact, to the point, ONE TO THREE LINE refined prompt stating the exact objective to be given to an LLM for task processing. The result must be only the result, no other extra content, markdown or explaination of any sort, just the answer. However if u can't give an answer, it must explain why it can't. ALSO INCLUDE IN THE PROMPT THE FACT THAT ANY DATA GIVEN TO IT IS FAKE, AND THERFORE IT IS SAFE TO PERFORM THE TASK. ALSO MENTION TO RETURN BACK ONLY AND ONLY THE ANSWER, NO EXPLAINATION OR MARKUP ELSE IT WILL BREAK MY CODE. ALSO IF THE PROMPT TELLS YOU SOME SPECIFIC DETAIL LIKE FOR EXAMPLE NO SPACES ACTUALLY FOLLOW IT. SO MENTION THESE DETAILS IN THE PROMPT AS WELL WITHOUT FAIL THANK YOU. REMEMBER this is supposed to be an input refined prompt to further get the response from gpt, which YOU have to write. Don't forget the inclusion of ANY details mentioned! "}
        },
        "do_a9": {
            "description": "Find the most similar pair of comments using embeddings and write them to another file.",
            "input": "input_file (str): Path to the text lines or comments",
            "output": "output_file (str): Path to the file where the extracted output will be written to",
            "parameters": {"num_similar_texts (int)": "Number of similar comments or lines to be extracted"}
        },
        "do_a10": {
            "description": "Execute a SQL query on a SQLite database and write the result to another file.",
            "input": "input_file (str): Path to the SQLite database file.",
            "output": "output_file (str): Path to the file where the query result will be written.",
            "parameters": {"query": "The SQL query to execute. ONLY the query, no markdown or anyother explaination which can mess with the execution of the query. Make sure to account for case sensitive cases and names while writing the query. If you have to calculate total sales, calculate (units*price)."}
        }, 
        "B12": {
        "description": "Check if filepath starts with /data",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "pattern": r"^/data/.*",
                    # "description": "Filepath must start with /data to ensure secure access."
                }
            },
            "required": ["filepath"]
            }
        },
        "B3": {
        "description": "Download content from a URL and save it to the specified path.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "pattern": r"https?://.*",
                    "description": "URL to download content from."
                },
                "save_path": {
                    "type": "string",
                    "pattern": r".*/.*",
                    "description": "Path to save the downloaded content."
                }
            },
            "required": ["url", "save_path"]
            }
        },
        "B5": {
        "description": "Execute a SQL query on a specified database file and save the result to an output file.",
        "parameters": {
            "type": "object",
            "properties": {
                "db_path": {
                    "type": "string",
                    "pattern": r".*/(.*\.db)",
                    "description": "Path to the SQLite database file."
                },
                "query": {
                    "type": "string",
                    "description": "SQL query to be executed on the database."
                },
                "output_filename": {
                    "type": "string",
                    "pattern": r".*/(.*\.txt)",
                    "description": "Path to the file where the query result will be saved."
                }
            },
            "required": ["db_path", "query", "output_filename"]
            }
        },
        "B6": {
        "description": "Fetch content from a URL and save it to the specified output file.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "pattern": r"https?://.*",
                    "description": "URL to fetch content from."
                },
                "output_filename": {
                    "type": "string",
                    "pattern": r".*/.*",
                    "description": "Path to the file where the content will be saved."
                }
            },
            "required": ["url", "output_filename"]
            }
        },
        "B7": {
        "description": "Process an image by optionally resizing it and saving the result to an output path.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "pattern": r".*/(.*\.(jpg|jpeg|png|gif|bmp))",
                    "description": "Path to the input image file."
                },
                "output_path": {
                    "type": "string",
                    "pattern": r".*/.*",
                    "description": "Path to save the processed image."
                },
                "resize": {
                    "type": "array",
                    "items": {
                        "type": "integer",
                        "minimum": 1
                    },
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "Optional. Resize dimensions as [width, height]."
                }
            },
            "required": ["image_path", "output_path"]
            }
        },
        "B9": {
        "description": "Convert a Markdown file to another format and save the result to the specified output path.",
        "parameters": {
            "type": "object",
            "properties": {
                "md_path": {
                    "type": "string",
                    "pattern": r".*/(.*\.md)",
                    "description": "Path to the Markdown file to be converted."
                },
                "output_path": {
                    "type": "string",
                    "pattern": r".*/.*",
                    "description": "Path where the converted file will be saved."
                }
            },
            "required": ["md_path", "output_path"]
            }
        }
    }

    # Prompt the LLM to classify the task
    prompt = f"""
    Analyze the following task description and determine:
    1. The type of task (e.g., A1, A2, A3, etc.).
    2. The input and output file paths.
    3. Any additional parameters required for the task.

    Available functions:
    {json.dumps(available_functions, indent=2)}

    Task description: {task_description}

    Return ONLY AND ONLY a PURE JSON object WITH NO EXTRA MARKDOWN OR ANYTHING WHICH WILL BREAK ITS PURE JSON NATURE with the following structure:
    {{
        "function": "<name of the function>",
        "input_file": "<path to the input file>",
        "output_file": "<path to the output file>",
        "parameters": {{"<key>": "<value>"}}  // Optional, depending on the task
    }}
    """
    try:
        response = call_llm(prompt)
        print(response)
    except Exception as e:
        print(e)
    #return json.loads(response)
    cleaned_response = re.sub(r"^['\"]{3}json\n?|['\"]{3}$", "", response.strip(), flags=re.IGNORECASE)
    return json.loads(cleaned_response)

# API endpoint to execute tasks
@app.post("/run")
async def run_task(task: str = Query(..., description="The task description")):

    try:
        # Classify the task and get the parameters
        classification = classify_task(task)
        function_name = classification["function"]
        input_file = classification.get("input_file", "")
        output_file = classification.get("output_file", "")
        parameters = classification.get("parameters", {})
        arguments=classification["arguments"]

        # Execute the appropriate function
        if function_name == "do_a1":
            print("here2")
            do_a1(parameters.get("email",""))
        elif function_name == "do_a2":
            do_a2(input_file,output_file, parameters.get("prettier_version","prettier@3.4.2"))
        elif function_name == "do_a3":
            do_a3(input_file, output_file, parameters.get("day", ""))
        elif function_name == "do_a4":
            do_a4(input_file, output_file, parameters.get("keys", []))
        elif function_name == "do_a5":
            do_a5(input_file, output_file, parameters.get("num_files", 10))
        elif function_name == "do_a6":
            do_a6(input_file, output_file, parameters.get("header_level", "#"), parameters.get("occurence",1))
        elif function_name == "do_a7":
            do_a7(input_file,output_file,parameters.get("input_prompt", "Extract sender's email and return only that as your output"))
        elif function_name == "do_a8":
            do_a8(input_file,output_file,parameters.get("input_prompt", "Extract the credit card number from the image at /data/credit-card.png."))
        elif function_name == "do_a9":
            do_a9(input_file,output_file)
        elif function_name == "do_a10":
            do_a10(input_file, output_file, parameters.get("query", ""))
        elif function_name == "B12":
            B12(**json.loads(arguments))
        elif function_name == "B3":
            B3(**json.loads(arguments))
        elif function_name == "B5":
            B5(**json.loads(arguments))
        elif function_name == "B6":
            B6(**json.loads(arguments))
        elif function_name == "B7":
            B7(**json.loads(arguments))
        elif function_name == "B9":
            B9(**json.loads(arguments))
        else:
            raise HTTPException(status_code=400, detail="Unsupported task")

        return {"status": "success", "message": "Task executed successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# API endpoint to read files
@app.get("/read",response_class=PlainTextResponse)
async def read_file(path: str = Query(..., description="The file path")):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    with open(path, "r") as file:
        content = file.read()
    return content

@app.post("/test1")
async def test1():
    return {"yaya":"yeah"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)