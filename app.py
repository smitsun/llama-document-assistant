import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from colorama import init, Fore, Style

# Initialize colorama for colored terminal output
init(autoreset=True)

def main():
    # Load environment variables from .env file
    load_dotenv()

    api_key = os.getenv("LLAMA_API_KEY")
    base_url = os.getenv("LLAMA_BASE_URL")
    model_name = os.getenv("LLAMA_MODEL", "llama-3.1-8b-instant")

    # Check if user has set the API key
    if not api_key or api_key == "your_api_key_here":
        print(f"\n{Fore.RED}{Style.BRIGHT}[ERROR] API Key is missing!")
        print(f"{Fore.YELLOW}Please open the {Fore.CYAN}.env{Fore.YELLOW} file in this directory and replace:")
        print(f"{Fore.WHITE}LLAMA_API_KEY=your_api_key_here")
        print(f"{Fore.YELLOW}with your actual API key.")
        print(f"\n{Fore.GREEN}Where to get a key:")
        print(f"- Groq (Recommended, super fast): {Fore.CYAN}https://console.groq.com/{Style.RESET_ALL}")
        print(f"- Together AI: {Fore.CYAN}https://api.together.xyz/{Style.RESET_ALL}\n")
        sys.exit(1)

    print(f"{Fore.CYAN}{Style.BRIGHT}==========================================")
    print(f"{Fore.CYAN}{Style.BRIGHT}   Llama API Chatbot (OpenAI Compatible)")
    print(f"{Fore.CYAN}{Style.BRIGHT}==========================================")
    print(f"{Fore.GREEN}Endpoint: {Fore.WHITE}{base_url}")
    print(f"{Fore.GREEN}Model:    {Fore.WHITE}{model_name}")
    print(f"{Fore.YELLOW}Type 'exit' or 'quit' to end the chat.\n")

    # Initialize the OpenAI client pointing to the Llama provider
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
    except Exception as e:
        print(f"{Fore.RED}Failed to initialize client: {e}")
        sys.exit(1)

    # Maintain conversation history
    messages = [
        {"role": "system", "content": "You are a helpful, concise AI assistant powered by Llama."}
    ]

    while True:
        try:
            # Get user input
            user_input = input(f"\n{Fore.BLUE}{Style.BRIGHT}You: {Fore.WHITE}")
            if user_input.strip().lower() in ["exit", "quit"]:
                print(f"{Fore.YELLOW}\nGoodbye!")
                break
            
            if not user_input.strip():
                continue

            # Append user message to history
            messages.append({"role": "user", "content": user_input})

            print(f"{Fore.GREEN}{Style.BRIGHT}Llama: {Fore.WHITE}", end="", flush=True)

            # Request streaming response from Llama
            response_stream = client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True
            )

            full_response = ""
            for chunk in response_stream:
                if chunk.choices[chunk.choices.__len__() - 1].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content

            print() # Print newline after streaming completes

            # Append assistant response to history
            messages.append({"role": "assistant", "content": full_response})

        except KeyboardInterrupt:
            print(f"{Fore.YELLOW}\nGoodbye!")
            break
        except Exception as e:
            print(f"\n{Fore.RED}{Style.BRIGHT}[Error occurred]: {e}")
            # Remove the last user message if it failed so history stays clean
            if messages[-1]["role"] == "user":
                messages.pop()

if __name__ == "__main__":
    main()
