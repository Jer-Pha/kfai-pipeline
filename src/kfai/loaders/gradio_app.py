import gradio as gr
from langchain_ollama import OllamaLLM

from kfai.loaders.agents.query_agent import QueryAgent
from kfai.loaders.utils.config import QA_MODEL, STYLE_CSS_FILE


def run() -> None:
    """
    Initializes the RAG agent and launches the Gradio web interface.
    """
    print("--- Initializing KFAI Query Agent for GUI ---")
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
    print("--- Agent Initialized. Launching Gradio Interface... ---")

    # This is the function that Gradio will call for each user input.
    # It takes the user's message and the chat history.
    def chat_with_agent(
        message: str,
        history: list[str],
    ) -> str:
        # We pass the user's message to the agent's query method.
        # The 'history' argument is available if you wish to add
        # conversational context to your agent in the future.
        print(f"Received query: {message}")
        response = query_agent.process_query(message, True)
        assert response is not None
        return response

    # Configure the Gradio interface
    web_app = gr.ChatInterface(
        fn=chat_with_agent,
        title="KF/AI",
        description="Ask a question about Kinda Funny content.",
        examples=[
            (
                "How did the crew's opinion of Cyberpunk 2077 change from its"
                " problematic launch to the release of the Phantom Liberty"
                " expansion?"
            ),
            (
                "What did Blessing and Tim say about Rocket League on Kinda"
                " Funny Games Daily or the Gamescast?"
            ),
            'Describe Nick\'s "three finger scale".',
        ],
        cache_examples=False,
        css_paths=[STYLE_CSS_FILE],
    )

    # Launch the web server.
    web_app.launch(share=False)
