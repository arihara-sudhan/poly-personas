from __future__ import annotations
from typing import Any, Dict
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from llm import LLMBotService, PersonaDetails

class PersonaAgent:
    def __init__(self, bot_service: LLMBotService) -> None:
        self.bot_service = bot_service
        self._graph = (
            RunnablePassthrough()
            | RunnableLambda(self._detect_intent)
            | RunnableBranch(
                (lambda state: state["intent"] == "create_persona", RunnableLambda(self._handle_persona_creation)),
                RunnableLambda(self._prepare_existing_persona),
            )
            | RunnableLambda(self._generate_reply)
        )

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the state machine synchronously."""
        return self._graph.invoke(state)

    def _detect_intent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        intent = self.bot_service.detect_intent(state["message"])
        state["intent"] = intent
        return state

    def _handle_persona_creation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        details: PersonaDetails = self.bot_service.generate_persona(state["user_name"], state["message"])
        state["action"] = "create_persona"
        state["persona_details"] = details
        state["persona_prompt"] = details.persona_prompt
        state["context_snippets"] = []
        return state

    def _prepare_existing_persona(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state["action"] = "chat"
        prompt = state.get("active_persona_prompt") or state.get("persona_prompt")
        state["persona_prompt"] = prompt
        return state

    def _generate_reply(self, state: Dict[str, Any]) -> Dict[str, Any]:
        persona_prompt = state["persona_prompt"]
        context = state.get("context_snippets") or []
        reply = self.bot_service.generate_reply(persona_prompt, context, state["message"])
        state["reply"] = reply
        return state