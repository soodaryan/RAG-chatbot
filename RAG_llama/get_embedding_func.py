from langchain_community.embeddings.ollama import OllamaEmbeddings

def get_embedding_function():
    embedddings = OllamaEmbeddings( model = "nomic-embed-text" )
    return embedddings

# from langchain_community.embeddings.bedrock import BedrockEmbeddings

# def get_embedding_function():
#     embedddings = BedrockEmbeddings(
#         credentials_profile_name="default",
#         region_name="us-east-1"
#     )
#     return embedddings