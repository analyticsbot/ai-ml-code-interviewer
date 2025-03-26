"""
Configuration settings for the AI-ML Code Interviewer application.
"""
import os

from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# LLM API settings
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
# Provider-specific API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GROK_API_KEY = os.getenv("GROK_API_KEY", "")
# For backward compatibility
LLM_API_KEY = os.getenv("LLM_API_KEY", "lm-studio")
LLM_MODEL = os.getenv("LLM_MODEL", "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "lmstudio")

# LLM Provider options
LLM_PROVIDERS = {
    "lmstudio": {
        "name": "LM Studio (Local)",
        "base_url": "http://localhost:1234/v1",
        "api_key": "lm-studio",
        "models": [
            #     "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
            #     "lmstudio-community/Mistral-7B-Instruct-v0.2-GGUF",
            #     "lmstudio-community/Mixtral-8x7B-Instruct-v0.1-GGUF",
        ],
    },
    "openai": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "api_key": "",  # User must provide
        "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o"],
    },
    "anthropic": {
        "name": "Anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "api_key": "",  # User must provide
        "models": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
    },
    "google": {
        "name": "Google AI (Gemini)",
        "base_url": "https://generativelanguage.googleapis.com/v1",
        "api_key": "",  # User must provide
        "models": ["gemini-pro", "gemini-ultra"],
    },
    "grok": {
        "name": "Grok",
        "base_url": "https://api.grok.ai/v1",
        "api_key": "",  # User must provide
        "models": ["grok-1", "grok-2"],
    },
    "perplexity": {
        "name": "Perplexity AI",
        "base_url": "https://api.perplexity.ai",
        "api_key": "",  # User must provide
        "models": ["sonar", "sonar-pro"],  # Example model names
    },
}

# Topic list
ML_DL_TOPICS = [
    "Linear Regression",
    "Logistic Regression",
    "K-Means Clustering",
    "Self Attention",
    "Multi-Headed Attention",
    "Decision Trees",
    "Random Forests",
    "Support Vector Machines",
    "Neural Networks",
    "Convolutional Neural Networks",
    "Recurrent Neural Networks",
    "Long Short-Term Memory Networks",
    "Generative Adversarial Networks",
    "Gradient Boosting Machines",
    "Natural Language Processing",
    "Reinforcement Learning",
    "Feature Engineering",
    "Hyperparameter Tuning",
    "Model Evaluation",
    "Cross-Validation",
    "Ensemble Learning",
    "Dimensionality Reduction",
    "Principal Component Analysis",
    "t-Distributed Stochastic Neighbor Embedding",
    "Transfer Learning",
    "Active Learning",
    "Time Series Analysis",
    "Anomaly Detection",
    "Autoencoders",
    "Variational Autoencoders",
]

# Difficulty levels
DIFFICULTY_LEVELS = ["Easy", "Medium", "Hard"]

# Maximum number of questions
MAX_QUESTIONS = 20

# Code execution settings
ENABLE_CODE_EXECUTION = os.getenv("ENABLE_CODE_EXECUTION", "True").lower() == "true"
