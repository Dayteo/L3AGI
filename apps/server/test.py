from langchain.smith import RunEvalConfig, run_on_dataset
from langchain_community.chat_models import ChatOpenAI
from langsmith import Client
import requests
from config import Config  # Ensure you have a config module for sensitive data

# Function to handle authentication
def authenticate():
    response = requests.post(
        f"{Config.L3_AUTH_API_URL}/auth/login",
        json={"email": Config.TEST_USER_EMAIL, "password": Config.TEST_USER_PASSWORD},
        timeout=30,
    )
    auth_data = response.json()
    return {
        "authorization": auth_data["access_token"],
        "x-refresh-token": auth_data["refresh_token"],
    }

# Factory function to create and return an agent
def agent_factory():
    # Create an instance of the ChatOpenAI model with specified parameters
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    
    # Initialize tools (e.g., Google Search)
    # tools = get_tools(["SerpGoogleSearch"])

    # Create and return an agent with specified tools and configuration
    # return initialize_agent(
    #     tools,
    #     llm,
    #     agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    #     verbose=True,
    #     handle_parsing_errors="Check your output and make sure it conforms!",
    #     agent_kwargs={
    #         "system_message": system_message,
    #         "output_parser": ConvoOutputParser(),
    #     },
    #     max_iterations=5,
    # )
    pass

# Initialize the client
client = Client()

# Configuration for evaluation
eval_config = RunEvalConfig(
    evaluators=[
        "qa",
        RunEvalConfig.Criteria("helpfulness"),
        RunEvalConfig.Criteria("conciseness"),
    ],
    input_key="input",
    eval_llm=ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo"),
)

# Run evaluation on the specified dataset
chain_results = run_on_dataset(
    client,
    dataset_name="test-dataset",
    llm_or_chain_factory=agent_factory,
    evaluation=eval_config,
    concurrency_level=1,
    verbose=True,
)

# Optionally print or process the results
# print(chain_results)
