import streamlit as st
import os
import re
from pdfminer.high_level import extract_text
import spacy
from spacy.matcher import Matcher
from groq import Groq

API_KEY = "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"

client = Groq(api_key=API_KEY)

nlp = spacy.load("en_core_web_sm")

def extract_skills_from_resume(text, skills_list):
    skills = []

    for skill in skills_list:
        pattern = r"\b{}\b".format(re.escape(skill))
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            skills.append(skill)

    return skills

def extract_education_from_resume(text):
    education = []

    # List of education keywords to match against
    education_keywords = ['Bsc', 'B. Pharmacy', 'B Pharmacy', 'Msc', 'M. Pharmacy', 'Ph.D', 'Bachelor', 'Master','B.Tech']

    for keyword in education_keywords:
        pattern = r"(?i)\b{}\b".format(re.escape(keyword))
        match = re.search(pattern, text)
        if match:
            education.append(match.group())

    return education

def extract_name(resume_text):
    matcher = Matcher(nlp.vocab)

    # Define name patterns
    patterns = [
        [{'POS': 'PROPN'}, {'POS': 'PROPN'}],  # First name and Last name
        [{'POS': 'PROPN'}, {'POS': 'PROPN'}, {'POS': 'PROPN'}],  # First name, Middle name, and Last name
        [{'POS': 'PROPN'}, {'POS': 'PROPN'}, {'POS': 'PROPN'}, {'POS': 'PROPN'}]  # First name, Middle name, Middle name, and Last name
        # Add more patterns as needed
    ]

    for pattern in patterns:
        matcher.add('NAME', patterns=[pattern])

    doc = nlp(resume_text)
    matches = matcher(doc)

    for match_id, start, end in matches:
        span = doc[start:end]
        return span.text

    return None

def extract_skills(text):
    doc = nlp(text)
    skills = []

    for ent in doc.ents:
        if ent.label_ == "SKILL":
            skills.append(ent.text)

    return skills

def LLM(query):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": query,
            }
        ],
        model="mixtral-8x7b-32768",
    )
    return chat_completion.choices[0].message.content

def check_resume_keywords(resume_text, keywords):
    resume_skills = extract_skills(resume_text)
    query = "Do the following skills: " + ", ".join(resume_skills) + " match the required keywords: " + ", ".join(keywords) + "? (Answer with Yes or No)"
    response = LLM(query)
    print(response)

    if response is not None:
        if "no" in response.lower():
            return False
        elif "yes" in response.lower():
            return True
        else:
            print("Warning: Ambiguous response from the Groq API.")
            return False
    else:
        print("Error: No response received from the Groq API.")
        return False

def get_recent_file_path(directory):
    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return None
    
    # Get a list of files in the directory
    files = os.listdir(directory)
    
    # Filter out directories and get file paths with their modified time
    files_info = [(os.path.join(directory, file), os.path.getmtime(os.path.join(directory, file))) for file in files if os.path.isfile(os.path.join(directory, file))]
    
    # Sort files by modified time (most recent first)
    files_info.sort(key=lambda x: x[1], reverse=True)
    
    # Return the path of the most recent file
    if files_info:
        return files_info[0][0]
    else:
        print(f"No files found in directory '{directory}'.")
        return None

def main():
    st.title("Resume Ranking System")
    
    # Directory for storing uploaded resumes
    upload_dir = "uploads"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    
    # File uploader for resume
    uploaded_files = st.file_uploader("Upload Resume (PDF or DOCX)", accept_multiple_files=True, type=['pdf', 'docx'])
    
    # Submit button
    if st.button("Submit"):
        if uploaded_files:
            st.write("Resumes submitted successfully!")
            directory_path = "/content/uploads/"
            
            # Save each uploaded resume to the upload directory
            for uploaded_file in uploaded_files:
                file_path = os.path.join(directory_path, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            # Retrieve the path of the most recently uploaded resume
            recent_file_path = get_recent_file_path(directory_path)
            if recent_file_path:
                resume_text = extract_text(recent_file_path)  # Extract text from the most recent uploaded resume
                st.write("Name:")
                st.write(extract_name(resume_text))
                st.write()
                st.write("Education:")
                st.write(extract_education_from_resume(resume_text))
                st.write()
                query = str(resume_text) + '''Give me the technical skills in this resume. Do not type anything else. I dont want any extra words/Information. I just want the keywords.'''
                st.write("Skills:")
                st.write(LLM(query))
            else:
                st.write("No resumes found in the directory.")
        else:
            st.write("No resumes uploaded.")

if __name__ == "__main__":
    main()
