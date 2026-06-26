import os
from dotenv import load_dotenv

from src.pipeline import build_rag_chain

def main():
    load_dotenv()

    print("\n" + "="*40)
    print("      Initializing RAG System...")
    print("="*40)

    rag_chain = build_rag_chain(session_id="langchain", history_aware=False)

    if rag_chain == "NO_DOCS":
        print("No valid documents found in ./data and no existing database found.")
        print("Please add documents to ./data first.")
        return

    print("\n" + "="*40)
    print("      RAG System Ready")
    print("="*40)
    print("Type your questions below. Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("\nQuestion: ")
            if user_input.lower().strip() == 'exit':
                print("Exiting...")
                break
            if not user_input.strip():
                continue
            
            response = rag_chain.invoke({"input": user_input})
            print(f"\nAnswer: {response['answer']}")
            
            if "context" in response and response["context"]:
                print("\n--- Sources Cited ---")
                for i, doc in enumerate(response["context"], 1):
                    source = doc.metadata.get("source", "Unknown Source")
                    page = doc.metadata.get("page")
                    src_str = f"{source} (Page {page})" if page is not None else f"{source}"
                    preview = doc.page_content[:150].replace("\n", " ") + "..."
                    print(f"{i}. {src_str}")
                    print(f"   Preview: {preview}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
