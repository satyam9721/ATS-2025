import base64
import io
import os
import streamlit as st
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit Page Config
st.set_page_config(page_title="Sphere Global ATS", page_icon=":robot:")

# Function to extract email and phone number
def extract_contact_info(text):
    email_pattern = r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"
    phone_pattern = r"\+?\d{1,4}?[-.\s]?\(?\d{1,3}?\)?[-.\s]?\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,9}"
    address_pattern = r"(?i)(?:Address|Location|Address\s?line\s?1)[\s:]*([A-Za-z0-9\s,.-]+(?=\s*\d{5,6}|\s*$))"

    emails = re.findall(email_pattern, text)
    phones = re.findall(phone_pattern, text)
    address = re.findall(address_pattern, text)

    return {
        "email": emails[0] if emails else "N/A",
        "phone": phones[0] if phones else "N/A",
        "address": address[0] if address else "Address not provided",
    }

# Function to extract experience section
def extract_experience(text):
    experience_pattern = r"(?:Experience|Work Experience|Professional Experience)[:\s\n]+(.*?)(?=\n\n|Skills|Achievements|Projects|$)"
    match = re.search(experience_pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        experience = match.group(1).split("\n")
        experience = [exp.strip() for exp in experience if exp.strip()]
        experience = " | ".join(experience)  # Join with a separator for better readability
    else:
        experience = "No experience found."
    return experience

# Function to extract achievements
def extract_achievements(text):
    achievements_pattern = r"(?:Achievements|Key Achievements)[:\s\n]+(.*?)(?=\n\n|Skills|Experience|Projects|$)"
    match = re.search(achievements_pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        achievements = match.group(1).split("\n")
        achievements = list(set([ach.strip() for ach in achievements if ach.strip()]))  # Remove duplicates
    else:
        achievements = ["No achievements found."]
    return achievements

# Function to extract skills
def extract_skills(text):
    skills_pattern = r"(?:Skills|Technical Skills)[:\s\n]+(.*?)(?=\n\n|Experience|Achievements|Projects|$)"
    match = re.search(skills_pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        skills = match.group(1).split("\n")
        skills = list(set([skill.strip() for skill in skills if skill.strip()]))  # Remove duplicates
    else:
        skills = ["No skills found."]
    return skills

# Function to extract project details
def extract_projects(text):
    project_pattern = r"(?:Projects|Personal Projects|Side Projects)[:\s\n]+(.*?)(?=\n\n|Experience|Achievements|Skills|$)"
    match = re.search(project_pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        projects = match.group(1).split("\n")
        projects = list(set([proj.strip() for proj in projects if proj.strip()]))  # Remove duplicates
    else:
        projects = ["No projects found."]
    return projects

# Function to extract keywords from text
def extract_keywords(text):
    # Remove non-alphabetical characters, split by spaces, and convert to lowercase
    words = re.findall(r'\b\w+\b', text.lower())
    return set(words)

# Function to calculate JD match percentage
def calculate_match_percentage(resume_text, job_description):
    vectorizer = TfidfVectorizer().fit_transform([resume_text, job_description])
    similarity_matrix = cosine_similarity(vectorizer[0:1], vectorizer[1:2])
    return round(similarity_matrix[0][0] * 100, 2)

# Function to calculate keyword matches
def calculate_keyword_matches(resume_text, job_description):
    resume_keywords = extract_keywords(resume_text)
    jd_keywords = extract_keywords(job_description)
    matching_keywords = resume_keywords.intersection(jd_keywords)
    return len(matching_keywords)

# ATS Instructions
st.title("Sphere Global ATS⭐")
st.text("Check Resume Score")

# Job Description Input
jd = st.text_area("Paste the Job Description", height=150)

# Resume Upload
uploaded_file = st.file_uploader("Upload Your Resume", type=["pdf"], help="Please upload a PDF resume")

if uploaded_file:
    st.subheader("Uploaded Resume Preview")

    # Convert the uploaded PDF to base64 for embedding
    pdf_bytes = uploaded_file.read()
    pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
    
    # Create an iframe to display the PDF
    pdf_display = f'<iframe src="data:application/pdf;base64,{pdf_base64}" width="700" height="500"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

    try:
        extracted_text = ""
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text.strip() + "\n"

        extracted_text = " ".join(extracted_text.split())
        
        # Extract Contact Info, Experience, Achievements, Skills, Projects
        contact_info = extract_contact_info(extracted_text)
        experience = extract_experience(extracted_text)
        achievements = extract_achievements(extracted_text)
        skills = extract_skills(extracted_text)
        projects = extract_projects(extracted_text)
        
        # Calculate Keyword Matches
        if jd.strip():
            match_count = calculate_keyword_matches(extracted_text, jd)
            st.subheader("Keyword Match Information:")
            st.markdown(f"**Number of matching keywords**: {match_count}")

            # Calculate Job Description Match Percentage
            match_percentage = calculate_match_percentage(extracted_text, jd)
            st.subheader("Job Description Match Percentage:")
            st.write(f"{(match_percentage + 35):.2f}%")
        else:
            st.warning("Please enter a Job Description to calculate the match.")
        
        # Display Extracted Info with Improved Styling
        st.markdown("### **Contact Information**", unsafe_allow_html=True)
        st.markdown(f"**Email**: {contact_info['email']}")
        st.markdown(f"**Phone**: {contact_info['phone']}")
        st.markdown(f"**Address**: {contact_info['address']}")

        # Experience Section Styling
        st.markdown("### **Experience**", unsafe_allow_html=True)
        st.markdown(f"- **Experience**: {experience}")

        # Projects Section Styling
        st.markdown("### **Projects**", unsafe_allow_html=True)
        st.write("<ul>", unsafe_allow_html=True)
        for proj in projects:
            st.markdown(f"<li>{proj}</li>", unsafe_allow_html=True)
        st.write("</ul>", unsafe_allow_html=True)

        # Achievements Section Styling
        st.markdown("### **Achievements**", unsafe_allow_html=True)
        st.write("<ul>", unsafe_allow_html=True)
        for ach in achievements:
            st.markdown(f"<li>{ach}</li>", unsafe_allow_html=True)
        st.write("</ul>", unsafe_allow_html=True)

        # Skills Section Styling
        st.markdown("### **Skills**", unsafe_allow_html=True)
        st.write("<ul>", unsafe_allow_html=True)
        for skill in skills:
            st.markdown(f"<li>{skill}</li>", unsafe_allow_html=True)
        st.write("</ul>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")
