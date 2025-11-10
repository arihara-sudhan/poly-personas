import json
import os
import re
from dataclasses import dataclass
from typing import List, Sequence

import google.generativeai as genai


@dataclass
class PersonaDetails:
    persona_name: str
    persona_prompt: str


class LLMBotService:
    """Wrapper around Gemini for persona workflows."""

    chat_model_name = "gemini-2.5-flash"
    embed_model_name = "models/text-embedding-004"

    def __init__(self) -> None:
        self._model = None
        self._configured = False

    def _ensure_client(self) -> None:
        if self._configured:
            return
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY environment variable.")
        genai.configure(api_key=api_key)
        self._configured = True

    def _get_chat_model(self):
        self._ensure_client()
        if self._model is None:
            self._model = genai.GenerativeModel(self.chat_model_name)
        return self._model

    def embed_text(self, text: str) -> List[float]:
        self._ensure_client()
        if not text:
            return []
        response = genai.embed_content(model=self.embed_model_name, content=text)
        return response["embedding"]

    def detect_intent(self, message: str) -> str:
        lowered = message.lower()
        heuristic_triggers = [
            "act like",
            "be a",
            "be an",
            "pretend to be",
            "roleplay as",
            "become a",
            "you are now",
            "take the role of",
        ]
        if any(trigger in lowered for trigger in heuristic_triggers):
            return "create_persona"

        prompt = (
            "You classify if a user wants to create a new persona.\n"
            "Reply with exactly one word: CREATE or CHAT.\n"
            "CREATE when the user requests a new persona, new role, or new character; CHAT otherwise.\n"
            f"User message: {message}"
        )
        model = self._get_chat_model()
        response = model.generate_content(prompt)
        normalized = (response.text or "").strip().upper()
        return "create_persona" if "CREATE" in normalized else "chat"

    def generate_persona(self, user_name: str, user_message: str) -> PersonaDetails:
        instruction = (
            "You help create persona descriptions.\n"
            "Return a compact JSON object with keys persona_name and persona_prompt.\n"
            "persona_name should be short (max 5 words).\n"
            "persona_prompt should describe how the assistant should behave for this persona.\n"
            "Do not include backticks or explanations."
        )
        prompt = (
            f"{instruction}\n"
            f"User name: {user_name}\n"
            f"User request: {user_message}\n"
            "JSON:"
        )
        model = self._get_chat_model()
        attempts = [
            prompt,
            prompt + "\nRespond with JSON only. Do not include explanations or extra text.",
        ]
        payload = None

        for attempt in attempts:
            response = model.generate_content(attempt)
            text = (response.text or "").strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
                break
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if match:
                    try:
                        payload = json.loads(match.group())
                        break
                    except json.JSONDecodeError:
                        continue

        if payload is None:
            return PersonaDetails(
                persona_name="Custom Persona",
                persona_prompt=f"Adopt the qualities described in the user's request: {user_message}",
            )

        persona_name = payload.get("persona_name") or "New Persona"
        persona_prompt = payload.get("persona_prompt") or "Be a helpful assistant."
        return PersonaDetails(persona_name=persona_name.strip(), persona_prompt=persona_prompt.strip())

    def generate_reply(self, persona_prompt: str, retrieved_messages: Sequence[str], user_message: str) -> str:
        context_block = "\n".join(retrieved_messages) if retrieved_messages else "No previous context."
        system_prompt = (
            "You are chatting inside a persona space.\n"
            "Follow the persona instructions faithfully.\n"
            "Respond concisely unless the user requests detail."
        )
        parts = [
            system_prompt,
            f"Persona Instructions:\n{persona_prompt}",
            "Relevant Past Messages:",
            context_block,
            f"Current User Message:\n{user_message}",
        ]
        prompt = "\n\n".join(parts)
        model = self._get_chat_model()
        response = model.generate_content(prompt)
        return (response.text or "").strip()

