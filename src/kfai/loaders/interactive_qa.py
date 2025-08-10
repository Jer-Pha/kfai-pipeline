from langchain_ollama import OllamaLLM
from loaders.agents.query_agent import QueryAgent
from loaders.utils.config import QA_MODEL


def run() -> None:
    # Configure LLM
    llm = OllamaLLM(
        model=QA_MODEL,
        temperature=0.4,
        top_p=0.95,
        top_k=50,
        reasoning=True,
        verbose=False,
        keep_alive=300,
    )

    # Start the interactive session
    query_agent = QueryAgent(llm=llm)
    while True:
        print("\n--- Ask a question, or type 'exit' to quit. ---")
        try:
            user_query = input("\n> ")
        except Exception as e:
            print(f"\nExiting due to unknown error:\n{e}")
            break

        user_query = user_query.strip()
        if not user_query:
            continue
        if user_query.lower() == "exit":
            print("\nExiting...")
            break

        query_agent.process_query(user_query)
