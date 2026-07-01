import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from colorama import init, Fore, Style

# Initialize colorama for colored terminal output
init(autoreset=True)

def main():
    load_dotenv()

    api_key = os.getenv("LLAMA_API_KEY")
    base_url = os.getenv("LLAMA_BASE_URL")
    model_name = os.getenv("LLAMA_MODEL", "llama-3.1-8b-instant")

    # Verify API key
    if not api_key or api_key == "your_api_key_here":
        print(f"\n{Fore.RED}{Style.BRIGHT}[ERROR] API Key is missing in .env!")
        sys.exit(1)

    # Read the knowledge base file
    knowledge_file_path = "knowledge.txt"
    if not os.path.exists(knowledge_file_path):
        print(f"\n{Fore.RED}{Style.BRIGHT}[ERROR] {knowledge_file_path} not found!")
        sys.exit(1)

    try:
        with open(knowledge_file_path, "r", encoding="utf-8") as f:
            knowledge_content = f.read()
    except Exception as e:
        print(f"{Fore.RED}Error reading {knowledge_file_path}: {e}")
        sys.exit(1)

    print(f"{Fore.CYAN}{Style.BRIGHT}================================================")
    print(f"{Fore.CYAN}{Style.BRIGHT}   Llama Document Assistant (Context: {knowledge_file_path})")
    print(f"{Fore.CYAN}{Style.BRIGHT}================================================")
    print(f"{Fore.GREEN}Model: {Fore.WHITE}{model_name}")
    print(f"{Fore.GREEN}Loaded {len(knowledge_content)} characters of custom knowledge.")
    print(f"{Fore.YELLOW}Ask anything about the document or type 'exit' to quit.\n")

    # Initialize client
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
    except Exception as e:
        print(f"{Fore.RED}Failed to initialize client: {e}")
        sys.exit(1)

    # Construct system prompt injecting our knowledge base content
    system_instruction = (
        "You are a helpful, professional internal assistant for Acme Corporation.\n"
        "You have access to the following official company handbook document.\n"
        "Answer the user's questions based STRICTLY on the document context provided below.\n"
        "If the answer cannot be found in the document context, state politely that the "
        "information is not in the handbook.\n\n"
        f"--- DOCUMENT CONTEXT ---\n{knowledge_content}\n------------------------"
    )

    messages = [
        {"role": "system", "content": system_instruction}
    ]

    while True:
        try:
            user_input = input(f"\n{Fore.BLUE}{Style.BRIGHT}Ask: {Fore.WHITE}")
            if user_input.strip().lower() in ["exit", "quit"]:
                print(f"{Fore.YELLOW}\nGoodbye!")
                break
            
            if not user_input.strip():
                continue

            messages.append({"role": "user", "content": user_input})

            print(f"{Fore.GREEN}{Style.BRIGHT}Acme Assistant: {Fore.WHITE}", end="", flush=True)

            # Request streaming completion from Llama
            response_stream = client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True
            )

            full_response = ""
            for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content

            print() # Newline

            messages.append({"role": "assistant", "content": full_response})

        except KeyboardInterrupt:
            print(f"{Fore.YELLOW}\nGoodbye!")
            break
        except Exception as e:
            print(f"\n{Fore.RED}{Style.BRIGHT}[Error occurred]: {e}")
            if messages[-1]["role"] == "user":
                messages.pop()

if __name__ == "__main__":
    main()
