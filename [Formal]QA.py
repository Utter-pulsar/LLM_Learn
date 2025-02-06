# import
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader

# load the document and split it into chunks
loader = TextLoader(".\RAGDataset/sidamingzhu.txt", encoding="utf-8")
documents = loader.load()

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
print(docs)
# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="./models/Text2vec-Base-Chinese")

# load it into Chroma
db = Chroma.from_documents(docs, embedding_function)
# query it
query = "武大郎打的谁？"
docs = db.similarity_search(query, k=4) # default k is 4

print(len(docs))

# print results
for doc in docs:
    print("="*100)
    print(doc.page_content)






