
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    from openai.types import CompletionUsage
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

TOKEN_COSTS = {
    "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.002},
    "gpt-3.5-turbo-0301": {"prompt": 0.0015, "completion": 0.002},
    "gpt-3.5-turbo-0613": {"prompt": 0.0015, "completion": 0.002},
    "gpt-3.5-turbo-16k": {"prompt": 0.003, "completion": 0.004},
    "gpt-3.5-turbo-16k-0613": {"prompt": 0.003, "completion": 0.004},
    "gpt-35-turbo": {"prompt": 0.0015, "completion": 0.002},
    "gpt-35-turbo-16k": {"prompt": 0.003, "completion": 0.004},
    "gpt-3.5-turbo-1106": {"prompt": 0.001, "completion": 0.002},
    "gpt-3.5-turbo-0125": {"prompt": 0.001, "completion": 0.002},
    "gpt-4-0314": {"prompt": 0.03, "completion": 0.06},
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
    "gpt-4-32k": {"prompt": 0.06, "completion": 0.12},
    "gpt-4-32k-0314": {"prompt": 0.06, "completion": 0.12},
    "gpt-4-0613": {"prompt": 0.06, "completion": 0.12},
    "gpt-4-turbo-preview": {"prompt": 0.01, "completion": 0.03},
    "gpt-4-1106-preview": {"prompt": 0.01, "completion": 0.03},
    "gpt-4-0125-preview": {"prompt": 0.01, "completion": 0.03},
    "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
    "gpt-4-turbo-2024-04-09": {"prompt": 0.01, "completion": 0.03},
    "gpt-4-vision-preview": {"prompt": 0.01, "completion": 0.03},
    "gpt-4-1106-vision-preview": {"prompt": 0.01, "completion": 0.03},
    "gpt-4o": {"prompt": 0.005, "completion": 0.015},
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "gpt-4o-mini-2024-07-18": {"prompt": 0.00015, "completion": 0.0006},
    "gpt-4o-2024-05-13": {"prompt": 0.005, "completion": 0.015},
    "gpt-4o-2024-08-06": {"prompt": 0.0025, "completion": 0.01},
    "o1-preview": {"prompt": 0.015, "completion": 0.06},
    "o1-preview-2024-09-12": {"prompt": 0.015, "completion": 0.06},
    "o1-mini": {"prompt": 0.003, "completion": 0.012},
    "o1-mini-2024-09-12": {"prompt": 0.003, "completion": 0.012},
    "text-embedding-ada-002": {"prompt": 0.0004, "completion": 0.0},
    "glm-3-turbo": {"prompt": 0.0007, "completion": 0.0007},
    "glm-4": {"prompt": 0.014, "completion": 0.014},
    "gemini-1.5-flash": {"prompt": 0.000075, "completion": 0.0003},
    "gemini-1.5-pro": {"prompt": 0.0035, "completion": 0.0105},
    "gemini-1.0-pro": {"prompt": 0.0005, "completion": 0.0015},
    "moonshot-v1-8k": {"prompt": 0.012, "completion": 0.012},
    "moonshot-v1-32k": {"prompt": 0.024, "completion": 0.024},
    "moonshot-v1-128k": {"prompt": 0.06, "completion": 0.06},
    "open-mistral-7b": {"prompt": 0.00025, "completion": 0.00025},
    "open-mixtral-8x7b": {"prompt": 0.0007, "completion": 0.0007},
    "mistral-small-latest": {"prompt": 0.002, "completion": 0.006},
    "mistral-medium-latest": {"prompt": 0.0027, "completion": 0.0081},
    "mistral-large-latest": {"prompt": 0.008, "completion": 0.024},
    "claude-instant-1.2": {"prompt": 0.0008, "completion": 0.0024},
    "claude-2.0": {"prompt": 0.008, "completion": 0.024},
    "claude-2.1": {"prompt": 0.008, "completion": 0.024},
    "claude-3-sonnet-20240229": {"prompt": 0.003, "completion": 0.015},
    "claude-3-5-sonnet-20240620": {"prompt": 0.003, "completion": 0.015},
    "claude-3-opus-20240229": {"prompt": 0.015, "completion": 0.075},
    "claude-3-haiku-20240307": {"prompt": 0.00025, "completion": 0.00125},
    "yi-34b-chat-0205": {"prompt": 0.0003, "completion": 0.0003},
    "yi-34b-chat-200k": {"prompt": 0.0017, "completion": 0.0017},
    "yi-large": {"prompt": 0.0028, "completion": 0.0028},
    "microsoft/wizardlm-2-8x22b": {"prompt": 0.00108, "completion": 0.00108},
    "meta-llama/llama-3-70b-instruct": {"prompt": 0.008, "completion": 0.008},
    "llama3-70b-8192": {"prompt": 0.0059, "completion": 0.0079},
    "openai/gpt-3.5-turbo-0125": {"prompt": 0.0005, "completion": 0.0015},
    "openai/gpt-4-turbo-preview": {"prompt": 0.01, "completion": 0.03},
    "openai/o1-preview": {"prompt": 0.015, "completion": 0.06},
    "openai/o1-mini": {"prompt": 0.003, "completion": 0.012},
    "anthropic/claude-3-opus": {"prompt": 0.015, "completion": 0.075},
    "anthropic/claude-3.5-sonnet": {"prompt": 0.003, "completion": 0.015},
    "google/gemini-pro-1.5": {"prompt": 0.0025, "completion": 0.0075},
    "deepseek-chat": {"prompt": 0.00014, "completion": 0.00028},
    "deepseek-coder": {"prompt": 0.00014, "completion": 0.00028},
    "doubao-lite-4k-240515": {"prompt": 0.000043, "completion": 0.000086},
    "doubao-lite-32k-240515": {"prompt": 0.000043, "completion": 0.000086},
    "doubao-lite-128k-240515": {"prompt": 0.00011, "completion": 0.00014},
    "doubao-pro-4k-240515": {"prompt": 0.00011, "completion": 0.00029},
    "doubao-pro-32k-240515": {"prompt": 0.00011, "completion": 0.00029},
    "doubao-pro-128k-240515": {"prompt": 0.0007, "completion": 0.0013},
    "llama3-70b-llama3-70b-instruct": {"prompt": 0.0, "completion": 0.0},
    "llama3-8b-llama3-8b-instruct": {"prompt": 0.0, "completion": 0.0},
    "Qwen/Qwen2.5-72B-Instruct": {"prompt": 0.00057, "completion": 0.0017},
    "Qwen/Qwen2.5-32B-Instruct": {"prompt": 0.0005, "completion": 0.001},
    "Qwen/Qwen2.5-14B-Instruct": {"prompt": 0.00029, "completion": 0.00086},
    "Qwen/Qwen2.5-7B-Instruct": {"prompt": 0.00014, "completion": 0.00029},
    "Qwen/Qwen3-14B": {"prompt": 0.00006, "completion": 0.00024},
    "Qwen/Qwen3-Next-80B-A3B-Instruct": {"prompt": 0.001, "completion": 0.002},
    "meta-llama/Meta-Llama-3-8B-Instruct": {"prompt": 0.0003, "completion": 0.0005}, 
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": {"prompt": 0.0004, "completion": 0.0006},
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": {"prompt": 0.0004, "completion": 0.0006},
}

TOKEN_MAX = {
    "gpt-4o": 128000,
    "gpt-4o-2024-05-13": 128000,
    "gpt-4o-2024-08-06": 128000,
    "gpt-4o-mini-2024-07-18": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4-turbo-2024-04-09": 128000,
    "gpt-4-0125-preview": 128000,
    "gpt-4-turbo-preview": 128000,
    "gpt-4-1106-preview": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4-vision-preview": 128000,
    "gpt-4-1106-vision-preview": 128000,
    "gpt-4": 8192,
    "gpt-4-0613": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0613": 32768,
    "gpt-3.5-turbo-0125": 16385,
    "gpt-3.5-turbo": 16385,
    "gpt-3.5-turbo-1106": 16385,
    "gpt-3.5-turbo-instruct": 4096,
    "gpt-3.5-turbo-16k": 16385,
    "gpt-3.5-turbo-0613": 4096,
    "gpt-3.5-turbo-16k-0613": 16385,
    "text-embedding-ada-002": 8192,
    "glm-3-turbo": 128000,
    "glm-4": 128000,
    "gemini-1.5-flash": 1000000,
    "gemini-1.5-pro": 2000000,
    "gemini-1.0-pro": 32000,
    "moonshot-v1-8k": 8192,
    "moonshot-v1-32k": 32768,
    "moonshot-v1-128k": 128000,
    "open-mistral-7b": 8192,
    "open-mixtral-8x7b": 32768,
    "mistral-small-latest": 32768,
    "mistral-medium-latest": 32768,
    "mistral-large-latest": 32768,
    "claude-instant-1.2": 100000,
    "claude-2.0": 100000,
    "claude-2.1": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-opus-20240229": 200000,
    "claude-3-5-sonnet-20240620": 200000,
    "claude-3-haiku-20240307": 200000,
    "yi-34b-chat-0205": 4000,
    "yi-34b-chat-200k": 200000,
    "yi-large": 16385,
    "microsoft/wizardlm-2-8x22b": 65536,
    "meta-llama/llama-3-70b-instruct": 8192,
    "llama3-70b-8192": 8192,
    "openai/gpt-3.5-turbo-0125": 16385,
    "openai/gpt-4-turbo-preview": 128000,
    "openai/o1-preview": 128000,
    "openai/o1-mini": 128000,
    "anthropic/claude-3-opus": 200000,
    "anthropic/claude-3.5-sonnet": 200000,
    "google/gemini-pro-1.5": 4000000,
    "deepseek-chat": 32768,
    "deepseek-coder": 16385,
    "doubao-lite-4k-240515": 4000,
    "doubao-lite-32k-240515": 32000,
    "doubao-lite-128k-240515": 128000,
    "doubao-pro-4k-240515": 4000,
    "doubao-pro-32k-240515": 32000,
    "doubao-pro-128k-240515": 128000,
    "Qwen/Qwen2.5-72B-Instruct": 131072,
    "Qwen/Qwen2.5-32B-Instruct": 131072,
    "Qwen/Qwen2.5-14B-Instruct": 131072,
    "Qwen/Qwen2.5-7B-Instruct": 131072,
    "Qwen/Qwen3-Next-80B-A3B-Instruct": 131072,
    "Qwen/Qwen3-14B": 40960,
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": 131072,
}


def count_input_tokens(messages, model="gpt-3.5-turbo-0125"):
    """Return the number of tokens used by a list of messages."""
    if "claude" in model and ANTHROPIC_AVAILABLE:
        try:
            vo = anthropic.Client()
            num_tokens = vo.count_tokens(str(messages))
            return num_tokens
        except:
            # Fallback to tiktoken if anthropic client fails
            pass
    
    if not TIKTOKEN_AVAILABLE:
        # Fallback estimation if tiktoken is not available
        total_chars = sum(len(str(msg)) for msg in messages)
        return max(1, total_chars // 4)  # Rough estimation: 4 chars per token
    
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Use cl100k_base encoding as fallback
        encoding = tiktoken.get_encoding("cl100k_base")
    
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-35-turbo",
        "gpt-35-turbo-16k",
        "gpt-3.5-turbo-16k",
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-0125",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4-turbo",
        "gpt-4-turbo-preview",
        "gpt-4-0125-preview",
        "gpt-4-1106-preview",
        "gpt-4-turbo",
        "gpt-4-vision-preview",
        "gpt-4-1106-vision-preview",
        "gpt-4o",
        "gpt-4o-2024-05-13",
        "gpt-4o-2024-08-06",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "o1-preview",
        "o1-preview-2024-09-12",
        "o1-mini",
        "o1-mini-2024-09-12",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4
        tokens_per_name = -1
    elif "gpt-3.5-turbo" == model:
        return count_input_tokens(messages, model="gpt-3.5-turbo-0125")
    elif "gpt-4" == model:
        return count_input_tokens(messages, model="gpt-4-0613")
    elif "open-llm-model" == model:
        tokens_per_message = 0
        tokens_per_name = 0
    else:
        # Default case for other models
        tokens_per_message = 3
        tokens_per_name = 1
    
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            content = value
            if isinstance(value, list):
                # for gpt-4v
                for item in value:
                    if isinstance(item, dict) and item.get("type") in ["text"]:
                        content = item.get("text", "")
            num_tokens += len(encoding.encode(content))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def count_output_tokens(string: str, model: str) -> int:
    """
    Returns the number of tokens in a text string.

    Args:
        string (str): The text string.
        model (str): The name of the encoding to use.

    Returns:
        int: The number of tokens in the text string.
    """
    if "claude" in model and ANTHROPIC_AVAILABLE:
        try:
            vo = anthropic.Client()
            num_tokens = vo.count_tokens(string)
            return num_tokens
        except:
            # Fallback to tiktoken if anthropic client fails
            pass
    
    if not TIKTOKEN_AVAILABLE:
        # Fallback estimation if tiktoken is not available
        return max(1, len(string) // 4)  # Rough estimation: 4 chars per token
    
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(string))


def get_max_completion_tokens(messages: list[dict], model: str, default: int) -> int:
    """Calculate the maximum number of completion tokens for a given model and list of messages."""
    if model not in TOKEN_MAX:
        return default
    return TOKEN_MAX[model] - count_input_tokens(messages) - 1


def get_token_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Calculate the cost for given tokens and model.
    
    Args:
        model: Model name
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        
    Returns:
        float: Total cost in USD
    """
    if model not in TOKEN_COSTS:
        return 0.0
    
    prompt_cost = prompt_tokens * TOKEN_COSTS[model]["prompt"] / 1000
    completion_cost = completion_tokens * TOKEN_COSTS[model]["completion"] / 1000
    return prompt_cost + completion_cost
