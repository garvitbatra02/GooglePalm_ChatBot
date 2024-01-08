import os
from langchain.llms import GooglePalm
from dotenv import load_dotenv
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

load_dotenv()
llm=GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"],temperature=0)



instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vector_file_path="faiss_index"

def create_vector_db():
    loader = CSVLoader(file_path='data.csv', source_column="prompt")
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data,embedding=instructor_embeddings)
    vectordb.save_local(vector_file_path)

def get_qa_chain():
    vectordb=FAISS.load_local(vector_file_path,instructor_embeddings)
    retriever=vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """
    Always strictly try to get any related information regarding me in the data given to you(numbers specifically), otherwise just tell you dont know and they can reach out to me on my social handles links.
   just let them know my social linkedin or github links or other handles (available in the data) to personally talk to me regarding that. Always give answer as third person for Garvit Batra

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
    # modified_chain("Tell me about coding skills of Garvit Batra")
    return modified_chain

# if __name__ == "__main__":
#     chain=get_qa_chain()
#     print(chain("Tell me about coding skills of Garvit Batra"))
    

