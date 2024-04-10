import streamlit as st
import asyncio
from logic import getQuiz,jsonToDataFrame,fromPDFToTextAndEmbeddings,uploadToMongoDbAtlas,vector_search,similarity_search
from io import BytesIO
import PyPDF2
from uuid import uuid4

st.set_page_config(
    page_title="Make My Quiz",
    page_icon="📝",
)

# Assuming 'file_id' is the unique identifier for the uploaded file
if 'file_id' not in st.session_state:
    st.session_state['files'] = []

st.header("Quiz generator 📝")

st.write("Generate quiz using your documents, upload pdf documents, we will extract relevant data from it and use it to create your quiz")
with st.form("quiz_form"):
    files=st.file_uploader("Upload a file", type=["pdf"],accept_multiple_files=True)

    with st.container():
        number=st.number_input("Number of questions", min_value=1, max_value=10, value=5)
        topic=st.text_input("Topic")
        difficulty=st.selectbox("Difficulty", ["easy", "medium", "hard"])
        temperature=st.slider('Creativity',help="how creative should the questions be")
        additional_instructions=st.text_area("Additional instructions",placeholder="each question must have 5 options\nA question can have more than 1 answer")
        question_type=st.selectbox("Question type", ["Multiple choice", "True/False"])
        answer_key=st.selectbox("Answer key",["Seperate","Included"])
    
    submit=st.form_submit_button("Generate quiz")

    if submit and files and topic and difficulty and additional_instructions and question_type and answer_key:
        with st.spinner('... generating quiz'):
            try:
                context:list[str]=[]
                for file in files:
                    textWithembeddings = fromPDFToTextAndEmbeddings(PyPDF2.PdfReader(BytesIO(file.read())))
                    relevant_docs=similarity_search(topic,textWithembeddings)
                    print(len(relevant_docs)," relevant chunks in ",file.name)
                    context.append("".join(relevant_docs))

                print("Context Found in : ",len(context)," docs\n")


                # file_ids=[file.file_id for file in st.session_state['files']]
                # vector_search(query=topic,file_ids=file_ids)
                # do something with the file
                quiz=getQuiz(topic=topic, number=int(number), question_type=question_type,context="".join(context), complexity=difficulty, instructions=additional_instructions, answer_key=answer_key,temperature=temperature/100)
                if(answer_key=='Seperate'):
                    quiz_dataframe=jsonToDataFrame(quiz['questions'])
                    answer_dataframe=jsonToDataFrame(quiz['answers'])
                    st.write(quiz_dataframe)
                    st.write(answer_dataframe)
                else:
                    dataframe=jsonToDataFrame(quiz)
                    st.write(dataframe)
                st.success('Quiz generated successfully')

            except Exception as e:
                print(e)
                st.error('Error generating quiz')
            
    else:
        st.warning('Please fill in all fields')