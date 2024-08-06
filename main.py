#Some practice code looking at CrewAI and Ollama

from crewai import Crew, Agent
from langchain.llms import Ollama
from langchain.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import Tool

loader = UnstructuredURLLoader(urls=["enter URL"])
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(data)

#Initialise embeddings
model_name = "all-MiniLM-L6-v2"
embeddings = SentenceTransformerEmbeddings(model_name=model_name)

#create vector
db = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
db.persist()

retriever = db.as_retriever()

researcher = Agent(
    ai = ollama_wrapper,
    name="Researcher",
    role="Research topcis via the data provided on the website",
    tools=[
        Tool(
            name="Website_Search",
            func=retriever.get_relevant_documents,
            description="Researches involved topics on a website."
        )
    ],
    goal="Answer questions about the topic and data provided on the website.",
    backstory="You are an incredible researcher. You pay attention to detail and break things into interesting and understandable paragraphs."

)

editor = Agent(
    ai = ollama_wrapper,
    name="Editor",
    role="review the summaries provided by the researcher. Make sure the information is correct.",
    tools=[
        Tool(
            name="Website_Search",
            func=retriever.get_relevant_documents,
            description="You are a top editor and cheif. Making sure that all the data has been correctly formatted and ready to be put into a newsletter."
        )
    ],
    goal="Double check the facts and ask further questions to validate the summarised data.",
    backstory="You are a super editor. You make sure the summary is interesting, correctly fact checked and quirky."
)



#Tasks
research_task = Task(
    description = (
        "Analyse the website provided ({crewai_url})"
        "Extract information about the latest news in the fitness industry"
        "Find the most interesting stories."
    ),
    expected_output=(
        "A structured list of news stories"
    ),
    agent=researcher,
    async_execution=True
)

editor_task = Task(
    description = (
        "Analise the data provided by ({crewai})"
        "review the summaries and create 5 key topics with a small paragraph about each"
        "include a link to the main article"
        "include a refence to the author and key figures in the article."
    )
    agent=editor_task,
    async_execution=True
)

research_crew = Crew(
    agents=[researcher, editor],
    tasks=[research_task, editor_task],
    verbose=True
)

job_crew_works = {
    'crewai_url':'Enter URL',
    'write up': """A new dietry requirement for your fitness."""
}

result = research_crew.kickoss(inputs=job_crew_works)
print(result)

