import os
import argparse
import openai
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')  

CHROMA_PATH = "/Users/aryansood/Github/RAG-chatbot/RAG_openai/chroma"

PROMPT_TEMPLATE = """
Answer the question based on only the following context : 
{context}

---

Answer the question based on the above context : {question}

"""

def main():
    # create CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text" , type = str , help = "The query text." )
    args = parser.parse_args()
    query_text = args.query_text

    # prepare the db
    embedding_function = OpenAIEmbeddings()
    query_embedding = embedding_function.embed_query(query_text)

    db = Chroma(persist_directory=CHROMA_PATH,
                embedding_function=embedding_function)
    
    # print(f"Number of documents in Chroma: {db._collection.count()}")

    # search the db
    results = db.similarity_search_by_vector_with_relevance_scores(query_embedding, k=3)
    # print(results)
    if len(results) == 0 or results[0][1] < 0.3 :
        print("Unable to find matching results.")

        return 
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results ])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    prompt = prompt_template.format(context = context_text,
                                    question = query_text)
    print(prompt)

    model = ChatOpenAI()

    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]

    formatted_response = f"Response : \n{response_text}\nSources: \n{sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()