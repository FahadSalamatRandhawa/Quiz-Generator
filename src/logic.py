import os
import traceback
import json
import pandas as pd
from dotenv import load_dotenv,find_dotenv
from Model__Embedding_HuggingFace import HuggingFaceEmbeddingsModel

_:bool=load_dotenv(find_dotenv())

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
import os

llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro",google_api_key=os.getenv("GEMINIULTRA_API_KEY"),temperature=0.2)

response_format_withanswer={
    "1":{
        "question":"question text",
        "options":{
            "a":"text for option",
            "b":"text for option",
            "c":"text for option",
            "d":"text for option"
        },
        "complexity":"complexity of the question",
        "answer":"correct option letter"
    },
    "2":{
        "question":"question text",
        "options":{
            "a":"text for option",
            "b":"text for option",
            "c":"text for option",
            "d":"text for option"
        },
        "complexity":"complexity of the question",
        "answer":"correct option letter"
    },
}
response_format_withoutanswer={
    "questions":{
        "1":{
            "question":"question text",
            "options":{
                "a":"text for option",
                "b":"text for option",
                "c":"text for option",
                "d":"text for option"
            }
        },
        "2":{
            "question":"question text",
            "options":{
                "a":"text for option",
                "b":"text for option",
                "c":"text for option",
                "d":"text for option"
            }
        },
    },
    "answers":{
            "1":"correct option letter",
            "2":"correct option letter"
    }
}

TEMPLATE = """
You are an expert MCQ maker, your job is to create {number} {question_type} type questions with {complexity} complexity on the topic "{topic}", once questions are created check each 
MCQ for mistakes and ensure that no question is repeated and every MCQ has a correct option and the correct answer, if you find any mistake, you will correct it, and then check again until there is no mistake.

Additional Instructions to follow : {instructions}.
Use this information to create the MCQ's : {context}

### Response format must be an object like this sample : {response_format}
"""
quiz_prompt=PromptTemplate(
    input_variables=["topic","number","question_type","complexity","context","instructions","response_format"],
    template=TEMPLATE
)
quiz_chain=LLMChain(llm=llm,prompt=quiz_prompt,verbose=True,output_key="quiz")
# mcqs=quiz_chain.run(topic="python", number=10, question_type="True False Unsure", complexity="medium", context="user your own knowledge", instructions="a question can have more than 1 answer,don't provide the answer with the questions, provide asnwers seperately", response_format=response_format)


REVIEW_TEMPLATE = """
You are an expert MCQ reviewer, your job is to review {number} {question_type} type questions with {complexity} complexity on the topic "{topic}", If you find any incorrect question or options or answer, update it properly, quiz:{quiz}. The output must be the same format as quiz input"""
review_prompt=PromptTemplate(input_variables=["topic","number","question_type","complexity","quiz"], template=REVIEW_TEMPLATE)
review_chain=LLMChain(llm=llm, prompt=review_prompt, verbose=True, output_key="review")


sequence_chain=SequentialChain(chains=[quiz_chain, review_chain], input_variables=["topic", "number", "question_type", "complexity","context","instructions","response_format"], output_variables=["review"] , verbose=True)

def getSeperateKeyQuiz(topic:str, number:int, question_type:str, complexity:str, context:str, instructions, answer_key):
    try:
        print("Generating quiz ....")
        responseformat=json.dumps(response_format_withoutanswer) if answer_key=="Seperate" else json.dumps(response_format_withanswer)
        print("response_format : ",responseformat,"\n")
        quiz_chain=LLMChain(llm=llm,prompt=quiz_prompt,verbose=True,output_key="quiz")
        quiz=quiz_chain.run(topic=topic, number=number, question_type=question_type, complexity=complexity, context=context, instructions=instructions,response_format=responseformat)
        print("Quiz generated successfully")
        print(quiz)

        return json.loads(quiz.strip('`'))
    except Exception as e:
        print("Error in generating quiz\n",e)
        print(traceback.format_exc())
        return None

def getQuiz(topic:str,number:int,question_type:str,complexity:str,context:str,instructions,answer_key):
    print(("Generating quiz ..."))
    review=sequence_chain.run(topic, number, question_type, complexity, context, 
    instructions, response_format=json.dumps(response_format_withoutanswer) if answer_key=="Seperate" else json.dumps(response_format_withanswer))
    review = review.strip('`')
    j_format=json.loads(review)

    print("Quiz generated successfully\n")
    return j_format


def jsonToDataFrame(j_format):
    print("Converting to dataframe")
    data_list = {}
    for key, value in j_format.items():
        data_list[key] = []
        for k,v in value.items():
            if type(v)==dict or json:
                value[k]=json.dumps(v)
        
        # row = value
        # row['options'] = json.dumps(value['options'])  # Convert options to a JSON string
        # data_list.append(row)
    quiz_dataframe:pd.DataFrame=pd.DataFrame(v for k,v in j_format.items())
    print("Dataframe converted successfully")
    return quiz_dataframe

def fromPDFToDocsAndEmbeddings(file):
    embedding_model=HuggingFaceEmbeddingsModel()
    print("Getting embeddings")
    textArray=[]; 
    print(file.pages)
    for page_num in range(len(file.pages)):
        page = file.pages[page_num]
        print(page.extract_text())
        textArray.append(page.extract_text())
    print("----------------------------------------------------------------------------\n")
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter=RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    docs=splitter.create_documents(textArray)
    print("Doc splitted : ", len(docs))
    embeddingsArray=[embedding_model.get_embedding(doc.page_content) for doc in docs]
    print("Embeddings generated successfully")

    return docs,embeddingsArray

def uploadToMongoDbAtlas(docs,embeddings):
    print("Uploading to Atlas")
    from pymongo import MongoClient
    client=MongoClient(os.getenv("MONGODBATLAS_CONNECTION_STRING"))
    collection=client["quiz_creation"]["temporary_storage"]
    collection.create_index("expiry", expireAfterSeconds=3600)
    print("Connected to Atlas at ,",collection)
    new_docs = []
    for doc, embedding in zip(docs, embeddings):
        # Create a new document with the original document and its embedding
        print("Creating new doc")
        new_doc = {
            "original_doc": doc.page_content,
            "embedding": embedding 
        }
        new_docs.append(new_doc)

    collection.insert_many(new_docs)
    print("Uploaded to Atlas successfully")
    return True