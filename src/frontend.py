import streamlit as st
import asyncio
from logic import getSeperateKeyQuiz,jsonToDataFrame,fromPDFToDocsAndEmbeddings,uploadToMongoDbAtlas
from io import BytesIO
import PyPDF2

# Assuming 'file_id' is the unique identifier for the uploaded file
if 'file_id' not in st.session_state:
    st.session_state['files'] = []

st.header("Quiz generator 📝")

st.write("Generate quiz using your documents, upload pdf documents, we will extract relevant data from it and use it to create your quiz")
with st.form("quiz_form"):
    file=st.file_uploader("Upload a file", type=["pdf"])

    with st.container():
        number=st.number_input("Number of questions", min_value=1, max_value=10, value=5)
        topic=st.text_input("Topic")
        difficulty=st.selectbox("Difficulty", ["easy", "medium", "hard"])
        additional_instructions=st.text_area("Additional instructions",placeholder="each question must have 5 options\nA question can have more than 1 answer")
        question_type=st.selectbox("Question type", ["Multiple choice", "True/False"])
        answer_key=st.selectbox("Answer key",["Seperate","Included"])
    
    submit=st.form_submit_button("Generate quiz")

    if submit and file and topic and difficulty and additional_instructions and question_type and answer_key:
        with st.spinner('... generating quiz'):
            try:
                docs,embeddings = fromPDFToDocsAndEmbeddings(PyPDF2.PdfReader(BytesIO(file.read())))
                uploaded=uploadToMongoDbAtlas(docs,embeddings)
                # do something with the file
                quiz=getSeperateKeyQuiz(topic=topic, number=int(number), question_type=question_type, complexity=difficulty,context="", instructions=additional_instructions, answer_key=answer_key)
                dataframe=jsonToDataFrame(quiz)
                print(dataframe)
                st.success('Quiz generated successfully')

            except Exception as e:
                print(e)
                st.error('Error generating quiz')
            
    else:
        st.warning('Please fill in all fields')