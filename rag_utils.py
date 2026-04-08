import json

VECTOR_STORE_FILE = "vector_store.json"


def save_vector_store_id(vs_id: str):
    with open(VECTOR_STORE_FILE, "w") as f:
        json.dump({"vector_store_id": vs_id}, f)


def load_vector_store_id() -> str:
    with open(VECTOR_STORE_FILE, "r") as f:
        data = json.load(f)
    return data["vector_store_id"]


def extract_answer(response):
    """
    Safely extract the assistant's final text answer
    from the OpenAI Responses API object.
    """
    for output in response.output:
        # We only care about assistant messages
        if hasattr(output, "content"):
            for block in output.content:
                if block.type == "output_text":
                    return block.text

    return "I couldn't find that in the provided PDFs."
