import os
from langchain.llms import GooglePalm
from langchain.chains import RetrievalQA
from langchain.embeddings import GooglePalmEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()
llm=GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"],temperature=0.4)

google_palm_embeddings = GooglePalmEmbeddings(google_api_key=os.environ["GOOGLE_API_KEY"])
# vector_file_path="faiss_index"

def create_vector_db():
    loader = CSVLoader(file_path='data.csv', source_column="prompt")
    data = loader.load()
    vectordb = Chroma.from_documents(data,
                           embedding=google_palm_embeddings,
                           persist_directory='./chromadb')
    vectordb.persist()

def get_qa_chain():
    vectorDatabase = Chroma(persist_directory="./chromadb",embedding_function=google_palm_embeddings)
    retriever = vectorDatabase.as_retriever(score_threshold = 0.7)

    prompt_template = """
    Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section 
    If prompt related to my personal details and the answer is not found in the context,let them know to contact me personally by sharing my linkedin and github handle

    CONTEXT: {context}

    QUESTION: {question}"""


    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}


    modified_chain = RetrievalQA.from_chain_type(llm=llm,
                                chain_type="stuff",
                                retriever=retriever,
                                input_key="query",
                                return_source_documents=True,
                                chain_type_kwargs=chain_type_kwargs)
    # print(modified_chain("Explain me how Garvit Batra would say Tell me something about yourself?"))
    return modified_chain

if __name__ == "__main__":
    # create_vector_db()
    chain=get_qa_chain()
    # print(chain("Tell me about coding skills of Garvit Batra"))
    

