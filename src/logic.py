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

def getQuiz(topic:str, number:int, question_type:str, complexity:str, context:str, instructions, answer_key):
    try:
        print("Generating quiz ....")
        responseformat=json.dumps(response_format_withoutanswer) if answer_key=="Seperate" else json.dumps(response_format_withanswer)
        print("response_format : ",responseformat,"\n")
        quiz_chain=LLMChain(llm=llm,prompt=quiz_prompt,verbose=True,output_key="quiz")
        quiz=quiz_chain.run(topic=topic, number=number, question_type=question_type, complexity=complexity, context=context, instructions=instructions,response_format=responseformat)
        print("Quiz generated successfully")

        return json.loads(quiz.strip('`'))
    except Exception as e:
        print("Error in generating quiz\n",e)
        print(traceback.format_exc())
        return None

# def getQuiz(topic:str,number:int,question_type:str,complexity:str,context:str,instructions,answer_key):
#     print(("Generating quiz ..."))
#     review=sequence_chain.run(topic, number, question_type, complexity, context, 
#     instructions, response_format=json.dumps(response_format_withoutanswer) if answer_key=="Seperate" else json.dumps(response_format_withanswer))
#     review = review.strip('`')
#     j_format=json.loads(review)

#     print("Quiz generated successfully\n")
#     return j_format


def jsonToDataFrame(j_format):
    print("Converting to dataframe")
    for key, value in j_format.items():
        if isinstance(value, dict):  # Check if value is a dictionary
            for k, v in value.items():
                if isinstance(v, dict):  # Check if v is a dictionary
                    value[k] = json.dumps(v)
        
        # row = value
        # row['options'] = json.dumps(value['options'])  # Convert options to a JSON string
        # data_list.append(row)
    quiz_dataframe:pd.DataFrame=pd.DataFrame(v for k,v in j_format.items())
    print("Dataframe converted successfully")
    return quiz_dataframe

def fromPDFToTextAndEmbeddings(file):
    embedding_model=HuggingFaceEmbeddingsModel()
    print("Getting embeddings")
    DocPages=[]
    print(file.pages)
    for page_num in range(len(file.pages)):
        page = file.pages[page_num]
        DocPages.append(page.extract_text())
    print("----------------------------------------------------------------------------\n")
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter=RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    textChunksArray=splitter.split_text("".join(DocPages))
    print("Doc splitted into : ", len(textChunksArray))

    textWithEmbeddings=[]
    for text in textChunksArray:
        embedding=embedding_model.get_embedding(text)
        textWithEmbeddings.append({'text':text,'embedding':embedding})
    # embeddingsArray=[embedding_model.get_embedding(doc.page_content) for doc in docs]
    print("Embeddings generated successfully")

    # return docs,embeddingsArray
    return textWithEmbeddings

def similarity_search(query:str,textWithEmbeddings:list,threshold=0.4):
    import numpy as np

    print("Calculating similarity .....\n")
    embedding_model=HuggingFaceEmbeddingsModel()
    queryVector=embedding_model.get_embedding(query)
    query2DVector=np.reshape(queryVector,(1,-1))
    print('query vector generated')

    from sklearn.metrics.pairwise import cosine_similarity
    relevantDocs=[]

    for item in textWithEmbeddings:
        doc2DVector=np.reshape(item['embedding'], (1, -1))
        similarity = cosine_similarity(query2DVector, doc2DVector)
        if similarity > threshold:
            relevantDocs.append(item['text'])
            print("text with similarity>{threshold} found ..")
    
    print("Similarity search complete, returning relevant texts array ....")

    return relevantDocs


def uploadToMongoDbAtlas(docs,embeddings,file_id):
    print("Uploading to Atlas")
    from pymongo import MongoClient
    from datetime import datetime, timedelta

    client=MongoClient(os.getenv("MONGODBATLAS_CONNECTION_STRING"))
    collection=client["quiz_creation"]["temporary_storage"]
    collection.create_index("expiry", expireAfterSeconds=120)
    print("Connected to Atlas at ,",collection)
    new_docs = []
    for doc, embedding in zip(docs, embeddings):
        # Create a new document with the original document and its embedding
        print("Creating new doc")
        new_doc = {
            "file_id":file_id,
            "text": doc.page_content,
            "embedding": embedding ,
            "expiry":datetime.utcnow()
        }
        new_docs.append(new_doc)

    collection.insert_many(new_docs)
    print("Uploaded to Atlas successfully")
    return True

def vector_search(query,file_ids):
    print("Seraching for query in files : ")
    from pymongo import MongoClient
    client=MongoClient(os.getenv("MONGODBATLAS_CONNECTION_STRING"))
    collection=client["quiz_creation"]["temporary_storage"]

    embedding_model=HuggingFaceEmbeddingsModel()
    query_embedding=embedding_model.get_embedding(query)
    query = [
        {"$vectorSearch": {
            "index": "default",
            "path":"embeddings",
            "queryVector":query_embedding,
            "limit":10000
            }
        },
        {"$match": {
                "file_id": {
                    "$in": file_ids
                }
            }}
    ]

    documents = collection.aggregate(query)

    for doc in documents:
        print(doc)

    print("Vector search successful")

    return True
