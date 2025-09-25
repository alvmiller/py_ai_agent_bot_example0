# https://medium.com/@jagadeesan.ganesh/mastering-llm-ai-agents-building-and-using-ai-agents-in-python-with-real-world-use-cases-c578eb640e35
# https://learnwithhasan.com/blog/create-ai-agents-with-python/
# https://python.langchain.com/docs/tutorials/agents/
# https://b-eye.com/blog/how-to-build-an-ai-agent-in-python/
# https://medium.com/@dey.mallika/building-gen-ai-agents-with-python-a-beginners-guide-bc3f842d99e7
# https://python.langchain.com/docs/tutorials/chatbot/
# https://docs.pytorch.org/tutorials/beginner/chatbot_tutorial.html
# https://www.alltius.ai/glossary/complete-guide-to-build-your-ai-chatbot-with-nlp-in-python
# https://dev.to/shreshthgoyal/create-your-own-ai-rag-chatbot-a-python-guide-with-langchain-dfi
# https://www.geeksforgeeks.org/python/build-an-ai-chatbot-in-python-using-cohere-api/
# https://quidget.ai/blog/ai-automation/how-to-build-an-ai-chatbot-with-python-a-beginners-guide/
# https://docs.slack.dev/tools/bolt-python/tutorial/ai-chatbot/
# https://medium.com/@BenjaminExplains/build-an-ai-chatbot-in-python-easy-and-free-8667b7b4232c
# https://learnpython.com/blog/ai-chatbot-in-python/
# https://code-b.dev/blog/ai-chat-bot-with-python
# https://medium.com/@dvasquez.422/building-a-simple-ai-agent-1e2f2b369b25
# https://medium.com/@varunkukade999/part-1-build-your-first-ai-agent-chatbot-with-langchain-and-langgraph-in-python-3b370bb6e7c1
# https://medium.com/@varunkukade999/part-2-build-your-first-ai-agent-chatbot-with-langchain-and-langgraph-in-python-dc9b4cffce06
# https://medium.com/@varunkukade999/part-3-build-your-first-ai-agent-chatbot-with-langchain-and-langgraph-in-python-5d1716098b7d
# https://medium.com/@varunkukade999/i-built-my-own-ai-coding-assistant-in-python-8e9079e8b9fc
# https://www.freecodecamp.org/news/how-to-build-an-ai-study-planner-agent-using-gemini-in-python/
# https://webflow.copilotkit.ai/blog/agents-101-how-to-build-your-first-ai-agent-in-30-minutes
# https://towardsdatascience.com/ai-agents-from-zero-to-hero-part-1/
# https://towardsdatascience.com/ai-agents-from-zero-to-hero-part-2/
# https://towardsdatascience.com/ai-agents-from-zero-to-hero-part-3/

# https://stackoverflow.com/questions/77727695/google-gemini-api-error-defaultcredentialserror-your-default-credentials-were
# https://cloud.google.com/docs/authentication/application-default-credentials
# https://github.com/google-gemini/deprecated-generative-ai-python/issues/114

# https://lukianovihor.medium.com/python-environment-variables-using-dotenv-library-71529ad0e9c3

# https://python.langchain.com/docs/integrations/chat/google_generative_ai/

#########################################################################

# https://medium.com/@dvasquez.422/building-a-simple-ai-agent-1e2f2b369b25

# https://aistudio.google.com/welcome
# python3 -m venv venv
# source venv/bin/activate
# pip install -r requirements.txt
# python3 main.py

#########################################################################

import os

# Load environment variables from a .env file.
from dotenv import load_dotenv

# Define structured output models using Pydantic
from pydantic import BaseModel

# Langchain imports that we will use to interact with Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

# Custom tools that we will use. These are pulled from our tools.py
from tools import scrape_tool, search_tool, save_tool  

# Pulling our Gemini API key from our .env file.
load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
print('---> GEMINI KEY IS: ', GEMINI_API_KEY)
print('---> GOOGLE KEY IS: ', GOOGLE_API_KEY)

# Define the structure of each lead in the output
class LeadResponse(BaseModel):
    company: str
    contact_info: str
    email: str
    summary: str
    outreach_message: str
    tools_used: list[str]

# Define a list structure to hold multiple leads
class LeadResponseList(BaseModel):
    leads: list[LeadResponse]

# Determining which AI model we will use, in this case, Gemini-2.5-flash
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Tell Gemini how to format the response using the Pydantic schema
parser = PydanticOutputParser(pydantic_object=LeadResponseList)

# The main part here. This is our prompt and the instructions we give to Gemini
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a sales enablement assistant.
            1. Use the 'scrape' tool to find exactly 5 local small businesses in Vancouver, British Columbia, from a variety of industries, that might need IT services.
            2. For each company identified by the 'scrape' tool, use the 'search' tool to gather detailed information from DuckDuckGo.
            3. Analyze the searched website content to provide:
                - company: The company name
                - contact_info: Any available contact details
                - summary: A brief qualification based on the scraped website content, focusing on their potential IT needs even if they are not an IT company.
                - email addresses
                - outreach message
                - tools_used: List tools used        

            Do not include extra text beyond the formatted output and the save confirmation message.
            4. Return the output as a list of 5 entries in this format: {format_instructions}
            5. After formatting the list of 5 entries, use the 'save_to_text' tool to send the json format to the text file. 
            6. If the 'save' tool runs, say that you ran it. If you did not run the 'save' tool, say that you could not run it.
            """,
        ),
        ("human", "{query}"),  # The actual user instruction
        ("placeholder", "{agent_scratchpad}"),  # Where the agent's internal reasoning goes
    ]
).partial(format_instructions=parser.get_format_instructions())

# List the tools we are telling our LLM to use from our tools.py file
tools = [scrape_tool, search_tool, save_tool]

# Create the agent with tool-calling abilities and structured reasoning
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

# Wrap the agent in an executor for running it with inputs
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Define the query that kicks off the lead generation process
query = "Find and qualify exactly 5 local leads in Vancouver for IT Services. No more than 5 small businesses."

# Run the agent with the query
raw_response = agent_executor.invoke({"query": query})

# Parse the structured output using the Pydantic schema
try:
    structured_response = parser.parse(raw_response.get('output'))
    print(structured_response)
except Exception as e:
    print("Error parsing response", e, "Raw Response - ", raw_response)
