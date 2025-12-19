from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from typing import Dict
import os
from langchain.chat_models import init_chat_model


class HandleModelAndQuery:

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.model = self.load_model()
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or ""

        self.prompt_template = PromptTemplate(
            input_variables=["context", "user_input"],
            template="""Use the following context to answer the question. 
                Context: {context}
                Question: {user_input}
            """
        )

    def load_model(self):
        try:
            return init_chat_model(
                model=self.config["llm_options"]["model"] or "gpt-4o-mini",
                temperature=self.config["llm_options"]["temperature"] or 0.75,
                timeout=self.config["llm_options"]["timeout"] or 30,
                max_tokens=self.config["llm_options"]["tokens_to_generate"] or 256,
            )
        except Exception as e:
            print(f"Error loading model: {e}\n")
            exit(1)

    def combine_context(self, related_docs):
        context = ""
        for result in related_docs:
            doc = result[0]
            context += doc.page_content+"\n"
        return context

    def get_response(self, user_input, related_docs, usesRAG=False):
        if usesRAG:
            context = self.combine_context(related_docs)
            prompt = self.prompt_template.format(
                context=context, user_input=user_input)
            return self.model.invoke([HumanMessage(prompt)])
        return self.model.invoke([HumanMessage(user_input)])
