from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.documents import Document
from typing import Dict, List, Optional, Tuple, Any
import os
import logging
from langchain.chat_models import init_chat_model

logger = logging.getLogger(__name__)


class HandleModelAndQuery:

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.model = self.load_model()
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or ""

        # self.prompt_template = PromptTemplate(
        #     input_variables=["context", "user_input", "system_prompt"],
        #     template=f"""
        #         System Message: {system_prompt}
        #         Use the below context to answer the User queries.
        #         Context: {context}
        #         User Query: {user_input}
        #     """
        # )

        self.prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(self._get_system_prompt()),
            SystemMessage("Context: {context}"),
            HumanMessage("{user_input}")
        ])

    def load_model(self) -> Any:
        """Load and initialize the chat model"""
        try:
            model = init_chat_model(
                model=self.config["llm_options"]["model"] or "gpt-4o-mini",
                temperature=self.config["llm_options"]["temperature"] or 0.75,
                timeout=self.config["llm_options"]["timeout"] or 30,
                max_tokens=self.config["llm_options"]["tokens_to_generate"] or 256,
            )
            logger.info(
                f"Successfully loaded model: {self.config['llm_options']['model']}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def combine_context(self, related_docs: List[Tuple[Document, float]]) -> str:
        """
        Combine retrieved documents into a context string

        :param related_docs: List of tuples (Document, score)
        :return: Combined context string
        """
        context = ""
        for result in related_docs:
            doc = result[0]
            # Include metadata in context for citations if available
            metadata_info = ""
            if doc.metadata:
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page")
                if page is not None:
                    metadata_info = f"[Source: {source}, Page: {page}] "
                else:
                    metadata_info = f"[Source: {source}] "
            context += metadata_info + doc.page_content + "\n\n"
        return context.strip()

    def get_response(
        self,
        user_input: str,
        related_docs: Optional[List[Tuple[Document, float]]] = None,
        usesRAG: bool = False
    ) -> Any:
        """
        Get response from the model with optional RAG context

        :param user_input: User's query
        :param related_docs: List of retrieved documents with scores
        :param usesRAG: Whether to use RAG context
        :return: Model response
        """
        try:
            if usesRAG and related_docs:
                context = self.combine_context(related_docs)
                prompt = self.prompt_template.format_messages(
                    context=context, user_input=user_input)
                logger.debug(
                    f"Generated prompt with {len(related_docs)} context documents")
                return self.model.invoke(prompt)
            return self.model.invoke([HumanMessage(user_input)])
        except Exception as e:
            logger.error(f"Error getting response: {e}")
            raise

    def _get_system_prompt(self) -> str:
        system_prompt = """
        You are a patient-facing assistant for a physiotherapy clinic.
        Your primary goal is to make patients feel heard, supported, and informed while answering questions using the provided clinic context.

        Tone:
        - Empathetic, reassuring, and kind
        - Professional but human
        - Acknowledge discomfort or concern before giving information

        Answer style:
        - Start by briefly acknowledging the user's situation when relevant
        - Use clear, everyday language
        - Avoid medical jargon unless the user uses it first
        - Do not sound robotic or overly scripted

        Length:
        - Short to medium responses (4–8 sentences)
        - Never overwhelm the user with information

        Safety and boundaries:
        - Do NOT diagnose conditions or recommend specific treatments
        - You may explain what services generally help certain issues without claiming certainty
        - If symptoms appear severe, alarming, or unusual, clearly advise contacting the clinic or emergency services

        Behavior:
        - Be honest if information is unavailable or unclear
        - Encourage reaching out to the clinic when human follow-up is better
        - Do not make promises about outcomes or recovery

        You should feel like a caring, attentive clinic staff member—not a medical authority.
        """
        return system_prompt
