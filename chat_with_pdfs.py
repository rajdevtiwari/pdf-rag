from dotenv import load_dotenv
from openai import OpenAI

from rag_utils import load_vector_store_id, extract_answer

SYSTEM_PROMPT = """
Answer only using the provided PDFs.
If the answer is not found, say:
"I couldn't find that in the provided PDFs."
"""

MODEL = "gpt-4o-mini"


def main():
    load_dotenv()
    client = OpenAI()
    vs_id = load_vector_store_id()

    print("PDF Chat Ready. Type 'exit' to quit.")

    while True:
        question = input("You > ")
        if question.lower() == "exit":
            break

        response = client.responses.create(
            model=MODEL,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": [vs_id],
                    "max_num_results": 4,
                }
            ],
        )

        answer = extract_answer(response)
        print("\nAssistant >", answer)
        print("-" * 40)


if __name__ == "__main__":
    main()
