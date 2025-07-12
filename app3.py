import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Service account for Gemini
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"c:\Users\Asus\Downloads\golden-ripsaw-449920-v1-54ec8b382164.json"

### 1. PDF Reader
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

### 2. Text Splitter
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

### 3. Vector Store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

### 4. Resume Analysis Chain
def get_resume_analysis_chain():
    prompt_template = """
    You are an expert resume analyzer and career coach. Analyze the resume and provide detailed feedback based on the job description provided.
    
    Resume Content: {context}
    Job Description: {question}
    
    Your analysis should include:
    1. Resume Rating (1-10) for this specific job with justification
    2. Concise professional summary (3-4 sentences)
    3. Top 3 suggested job titles that match this resume
    4. Missing skills that should be added for this job
    5. Top 5 interview questions likely to be asked for this resume and job
    
    Format your response clearly with appropriate headers and bullet points.
    Be specific and provide actionable advice.
    
    Analysis:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def analyze_resume(job_description):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(job_description)
    chain = get_resume_analysis_chain()
    response = chain({"input_documents": docs, "question": job_description}, return_only_outputs=True)
    return response["output_text"]

### 5. Generate Interview Questions
def generate_interview_questions(resume_text, job_desc, question_type="behavioral", num_questions=5):
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    
    question_types = {
        "behavioral": "behavioral/situational questions",
        "technical": "technical/skill-specific questions",
        "company": "company/role-specific questions",
        "leadership": "leadership/management questions"
    }
    
    prompt = f"""
    Generate {num_questions} {question_types[question_type]} for a job interview based on:
    - The candidate's resume: {resume_text[:5000]}
    - The job description: {job_desc[:2000]}
    
    For each question:
    1. Make it specific to the candidate's experience and the job requirements
    2. Include why this question might be asked (what the interviewer is looking for)
    3. Provide a tip for how to approach answering it
    
    Format as numbered questions with clear sections for each.
    """
    
    response = model.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)

### 6. Answer User Questions
def answer_user_question(question, context):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(question)
    
    prompt_template = """
    Answer the user's question based on the resume content below.
    Be helpful, professional, and provide actionable advice.
    
    Resume Content: {context}
    Question: {question}
    
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    return response["output_text"]

### MAIN APP
def main():
    st.set_page_config("Resume Analyzer Pro", page_icon="📄")
    st.header("📄 Resume Analyzer Pro")
    st.subheader("Get personalized resume feedback for your dream job")
    
    with st.sidebar:
        st.title("Upload & Analyze")
        pdf_docs = st.file_uploader("Upload Your Resume (PDF)", accept_multiple_files=True)
        job_desc = st.text_area("Paste the Job Description", height=200)
        
        if st.button("Analyze Resume") and pdf_docs and job_desc:
            with st.spinner("Analyzing your resume..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.session_state['resume_text'] = raw_text
                st.session_state['job_desc'] = job_desc
                st.success("✅ Analysis Complete")

    if "resume_text" in st.session_state:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Full Analysis", "🔍 Resume Summary", "🤖 ATS Check", "💬 Interview Prep", "❓ Ask Anything"])
        
        with tab1:
            st.subheader("Comprehensive Resume Analysis")
            analysis = analyze_resume(st.session_state['job_desc'])
            st.markdown(analysis)
        
        with tab2:
            st.subheader("Resume Summary")
            model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
            summary_prompt = f"Create a professional summary of this resume: {st.session_state['resume_text'][:10000]}"
            response = model.invoke(summary_prompt)
            st.markdown(response.content if hasattr(response, "content") else str(response))
        
        with tab3:
            st.subheader("ATS Optimization Check")
            model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
            ats_prompt = f"Analyze this resume for ATS optimization: {st.session_state['resume_text'][:10000]}\n\nJob Description: {st.session_state['job_desc']}\n\nProvide specific recommendations."
            response = model.invoke(ats_prompt)
            st.markdown(response.content if hasattr(response, "content") else str(response))
         
        with tab4:
            st.subheader("Interview Preparation")
    
            col1, col2 = st.columns(2)
            with col1:
                question_type = st.selectbox(
                    "Question Type",
                    ["behavioral", "technical", "company", "leadership"],
                    index=0
                )
            with col2:
                num_questions = st.slider("Number of Questions", 3, 10, 5)
    
            if st.button("Generate Questions"):
                with st.spinner("Generating interview questions..."):
                    questions = generate_interview_questions(
                        st.session_state['resume_text'],
                        st.session_state['job_desc'],
                        question_type,
                        num_questions
                    )
                    st.session_state['interview_questions_raw'] = questions
            
                    question_list = []
                    question_blocks = questions.split('\n\n')
                    for block in question_blocks:
                        if not block.strip():
                            continue
                        lines = block.split('\n')
                        question_line = lines[0].strip()
                        if question_line and question_line[0].isdigit():
                            question_text = question_line.split('. ', 1)[-1]
                        else:
                            question_text = question_line
                        question_list.append(question_text)
                    st.session_state['interview_questions_list'] = question_list
    
            if 'interview_questions_raw' in st.session_state:
                st.markdown(st.session_state['interview_questions_raw'])
                st.divider()
                st.subheader("Practice Answering")
        
                if 'interview_questions_list' in st.session_state and st.session_state['interview_questions_list']:
                    selected_index = st.selectbox(
                        "Select a question to practice",
                        range(len(st.session_state['interview_questions_list'])),
                        format_func=lambda x: st.session_state['interview_questions_list'][x]
                    )
            
                    selected_question = st.session_state['interview_questions_list'][selected_index]
                    user_answer = st.text_area("Type your answer here", height=150)
            
                    if st.button("Get Feedback") and user_answer:
                        with st.spinner("Analyzing your answer..."):
                            question_blocks = st.session_state['interview_questions_raw'].split('\n\n')
                            full_question_block = question_blocks[selected_index]
                    
                            feedback_prompt = f"""
                            **Question:** {full_question_block}
                    
                            **Candidate's Answer:** {user_answer}
                    
                            Provide detailed feedback on:
                            - STAR method structure (if applicable)
                            - Specificity of examples
                            - Relevance to the question
                            - Areas for improvement
                            - Strengths in the answer
                    
                            Format as bullet points with clear headings.
                            """
                            model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
                            response = model.invoke(feedback_prompt)
                            st.markdown("### Feedback on Your Answer")
                            st.markdown(response.content if hasattr(response, "content") else str(response))
                else:
                    st.info("Generate questions first to practice answering them")
        
        with tab5:
            st.subheader("Ask Anything About Your Resume")
            user_question = st.text_input("Ask any question about your resume or job application")
            if user_question and st.button("Get Answer"):
                with st.spinner("Finding the best answer..."):
                    answer = answer_user_question(user_question, st.session_state['resume_text'])
                    st.markdown(answer)

if __name__ == "__main__":
    main()
