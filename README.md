## ENV Variables
GEMINIULTRA_API_KEY="your api key"

### Functionality Logic Overview (Data Retreival and Extraction)
The uploaded documents are divided into smaller chunks,a function create vector embeddings for chunks and stores them temporarily along with data. Another function uses query embedding to retreive relevant documents by comparing document and query embedding vectores [all embeddings are 1Dimensional and converted to 2Dimension before comparision] ( using `sklearn` library for vector comparision) . Relevant data is fed to llm as context along with rest of data.


## LLM structure
Google Gemini Pro LLM, Langchain used to create quiz and review chain. 
QuizChain creates quiz
Review chain reviews the quiz and makes necessary changes
