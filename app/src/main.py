import json
import logging
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from dotenv import load_dotenv
import model_augmentation_pipeline as ma
import ingestion_retrieval_pipeline as ir

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

with open("../src/config.json", mode="r", encoding="utf-8") as read_file:
    global config
    config = json.load(read_file)
    read_file.close()


class FileSystemWatcher(FileSystemEventHandler):
    def on_created(self, event: FileSystemEvent):
        logging.info(f"Detected and Ingesting: {event.src_path}")
        docs = handlel_rag.load_document(event.src_path)
        # Delete the file after ingestion if thew option is true in config
        if config["rag_options"]["delete_file_after_ingestion"] and os.path.exists(event.src_path):
            os.remove(event.src_path)
            logging.info(f"Deleted After Ingestion: {event.src_path}")

        # Add the documents to the vector store
        handlel_rag.add_document(docs)
        logging.info(f"Ingested: {event.src_path}")

    def on_deleted(self, event):
        logging.info(f"Deleted: {event.src_path}")


# Start the folder observer // Klasör gözlemcisini başlat
observer = Observer()
observer.schedule(FileSystemWatcher(
), path=config["rag_options"]["data_ingestion_folder"], recursive=True)
observer.start()


handel_model = ma.HandleModelAndQuery(config)
handlel_rag = ir.HandleIngestionAndRetrieval(config)

model = handel_model.load_model()

if not model:
    logging.error(
        "Error loading model. Make sure you have installed the model and Ollama is running. Exiting...")
    exit(1)
if config["rag_options"]["clear_database_on_start"] and handlel_rag.vector_store._collection.count() > 0:
    handlel_rag.vector_store.reset_collection()


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise KeyError(
            "OPEN AI API key not found. Please add the API key to .env file.")
    print("Welcome, type 'help' for help and 'exit' to exit.")
    try:
        while True:
            user_input = input(">> ")
            if user_input == "exit":
                print("Goodbye!")
                logging.info("Exiting...")
                break
            if user_input == "help":
                print("User requesting HELP")
                continue
            if handlel_rag.vector_store._collection.count() > 0:
                related_docs = handlel_rag.get_docs_by_similarity(user_input)
                response = handel_model.get_response(
                    user_input, related_docs, True)
            else:
                response = handel_model.get_response(user_input, None, False)
            print(f"Response: {response.content}")
    except KeyboardInterrupt:
        logging.info("Exiting...")


if __name__ == '__main__':
    main()
