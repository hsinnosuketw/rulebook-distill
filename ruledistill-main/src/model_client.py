"""
LLM Client

LLMClient class: 
Encapsulates llm backend selection, supporting
- nvidia 
- ollama
chat-completion calls. 
All agents should use this instead of creating their own OpenAI clients.
"""

import threading
import queue
import time

from openai import OpenAI
import config


class LLMClient:
    """
    Unified LLM client supporting nvidia and ollama backends.

    Usage::

        llm = LLMClient(backend="ollama", model_name="qwen3-next:latest")
        response_text = llm.chat(system_prompt, user_prompt)
    """

    def __init__(
        self,
        backend: str = None,
        model_name: str = None,
        temperature: float = 0.1,
        max_tokens: int = None,
        top_p: float = 0.95,
        think: bool = False,
    ):
        """
        Initialize the LLM client.

        Args:
            backend: "nvidia" or "ollama" (defaults to config.LLM_BACKEND)
            model_name: Model identifier (defaults to backend-specific config value)
            temperature: Default generation temperature
            max_tokens: Default max tokens
            top_p: Default top-p sampling value
            think: Enable thinking/reasoning mode for Ollama models that support
                   it (e.g. qwen3-next). Set to False for models without thinking
                   mode (e.g. qwen3-coder-next). Ignored for non-Ollama backends.
        """
        self.backend = backend or config.LLM_BACKEND
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.think = think

        if self.backend == "ollama":
            self.model_name = model_name or config.OLLAMA_MODEL
            self.client = None  # uses native ollama library
            self.use_native_ollama = True
            think_label = " (think=True)" if self.think else ""
            print(f"  [LLMClient] Ollama backend → {self.model_name}{think_label}")
        elif self.backend == "nvidia":
            self.model_name = model_name or config.MODEL_NAME
            self.client = OpenAI(
                base_url=config.NVIDIA_BASE_URL,
                api_key=config.NVIDIA_API_KEY,
            )
            self.use_native_ollama = False
            print(f"  [LLMClient] NVIDIA backend → {self.model_name}")
        else:
            raise ValueError(
                f"Unsupported backend: {self.backend}. Use 'nvidia' or 'ollama'"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = None,
        max_tokens: int = None,
        top_p: float = None,
    ) -> str:
        """
        Send a chat-completion request and return the response text.

        Args:
            system_prompt: System message content
            user_prompt: User message content
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            top_p: Override default top_p

        Returns:
            The model's response text (stripped).
        """
        result = self.chat_with_metadata(
            system_prompt, user_prompt, temperature, max_tokens, top_p
        )
        return result["content"]

    def chat_with_metadata(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = None,
        max_tokens: int = None,
        top_p: float = None,
    ) -> dict:
        """
        Send a chat-completion request and return both content and thinking.

        Returns:
            dict with keys:
              - "content": the model's response text
              - "thinking": the model's thinking/reasoning (empty string if N/A)
        """
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        top_p = top_p if top_p is not None else self.top_p

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if self.use_native_ollama:
            return self._chat_ollama(messages, temperature, max_tokens, top_p)
        else:
            return self._chat_openai(messages, temperature, max_tokens, top_p)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _stream_ollama(self, messages, temperature, max_tokens, top_p, chunk_queue, done_event):
        """Stream from Ollama in a background thread, pushing chunks to queue.
        
        Uses raw httpx streaming so we can close the connection on timeout,
        which causes Ollama to abort the in-progress generation.
        """
        import json as _json
        import httpx

        # Filter out None values to prevent Ollama misinterpretation
        opts = {"temperature": temperature, "top_p": top_p}
        if max_tokens is not None:
            opts["num_predict"] = max_tokens

        payload = {
            "model": self.model_name,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "stream": True,
            "options": opts,
        }
        if self.think:
            payload["think"] = True

        try:
            # Use httpx streaming so we hold a reference to the response
            # that can be closed from the main thread on timeout.
            with httpx.stream(
                "POST",
                "http://localhost:11434/api/chat",
                json=payload,
                timeout=None,  # We handle timeout ourselves
            ) as response:
                # Store response so _abort_ollama_generation can close it
                self._active_stream_response = response
                for line in response.iter_lines():
                    if done_event.is_set():
                        break
                    if not line:
                        continue
                    try:
                        obj = _json.loads(line)
                    except _json.JSONDecodeError:
                        continue
                    msg = obj.get("message", {})
                    content = msg.get("content", "")
                    thinking = msg.get("thinking", "")
                    chunk_queue.put(("chunk", content, thinking))
                    if obj.get("done", False):
                        break
                self._active_stream_response = None
            chunk_queue.put(("done", "", ""))
        except Exception as e:
            self._active_stream_response = None
            if not done_event.is_set():
                chunk_queue.put(("error", str(e), ""))
            else:
                chunk_queue.put(("done", "", ""))  # Treat abort as normal completion

    def _stream_openai(self, messages, temperature, max_tokens, top_p, chunk_queue, done_event):
        """Stream from OpenAI-compatible API in a background thread."""
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=True,
            )
            for chunk in stream:
                if done_event.is_set():
                    break
                delta = chunk.choices[0].delta if chunk.choices else None
                content = delta.content if delta and delta.content else ""
                chunk_queue.put(("chunk", content, ""))
            chunk_queue.put(("done", "", ""))
        except Exception as e:
            chunk_queue.put(("error", str(e), ""))

    def chat_with_timeout(
        self,
        system_prompt: str,
        user_prompt: str,
        timeout_s: float = 60.0,
        temperature: float = None,
        max_tokens: int = None,
        top_p: float = None,
    ) -> dict:
        """
        Stream a chat-completion response with a wall-clock timeout.

        Uses streaming internally and accumulates tokens. If the wall-clock
        time exceeds *timeout_s*, generation is stopped and whatever partial
        content has been received is returned.

        Returns:
            dict with keys:
              - "content": accumulated response text (may be partial)
              - "thinking": accumulated thinking text (if available)
              - "timed_out": bool, True if the timeout fired before completion
        """
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        top_p = top_p if top_p is not None else self.top_p

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        chunk_queue = queue.Queue()
        done_event = threading.Event()

        # Choose the right streaming backend
        if self.use_native_ollama:
            target = self._stream_ollama
        else:
            target = self._stream_openai

        thread = threading.Thread(
            target=target,
            args=(messages, temperature, max_tokens, top_p, chunk_queue, done_event),
            daemon=True,
        )
        thread.start()

        # Log prompt size for diagnostics
        prompt_chars = sum(len(m["content"]) for m in messages)
        print(f"[LLMClient] Prompt size: {prompt_chars} chars → streaming with {timeout_s}s timeout")

        # Drain queue until done or timeout
        accumulated_content = []
        accumulated_thinking = []
        timed_out = False
        deadline = time.time() + timeout_s

        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                timed_out = True
                done_event.set()  # Signal thread to stop
                # Abort the Ollama generation to free the server for new requests
                self._abort_ollama_generation()
                break

            try:
                msg_type, content, thinking = chunk_queue.get(timeout=min(remaining, 0.5))
            except queue.Empty:
                continue

            if msg_type == "done":
                break
            elif msg_type == "error":
                if not accumulated_content:
                    raise RuntimeError(f"LLM streaming error: {content}")
                # If we already have partial content, treat as timeout-like
                print(f"[LLMClient] Stream error after partial content: {content}")
                break
            else:
                if content:
                    accumulated_content.append(content)
                if thinking:
                    accumulated_thinking.append(thinking)

        raw = "".join(accumulated_content).strip()
        thinking_text = "".join(accumulated_thinking).strip()

        # timeout debug 
        if timed_out:
            thinking_info = f", thinking: {len(thinking_text)} chars" if thinking_text else ""
            print(f"[LLMClient] ⚠ Timeout after {timeout_s}s — returning partial response (content: {len(raw)} chars{thinking_info})")
            print(f"[LLMClient] raw: {raw}")
            print(f"[LLMClient] thinking_text: {thinking_text[:3000]}")
            if not raw and not thinking_text:
                print(f"[LLMClient] ⚠ No tokens received at all — model likely still processing the prompt (prefill). Try increasing timeout_s or reducing prompt size.")
        else:
            print(f"[LLMClient] content length: {len(raw)} chars")
            if thinking_text:
                print(f"[LLMClient] thinking length: {len(thinking_text)} chars")

        # Fallback: thinking models sometimes put the answer in thinking field
        if not raw and thinking_text:
            print("[LLMClient] ⚠ content is empty — falling back to thinking field")
            raw = thinking_text

        return {"content": raw, "thinking": thinking_text, "timed_out": timed_out}

    def _abort_ollama_generation(self):
        """Abort any in-progress Ollama generation by closing the HTTP connection.
        
        This terminates the TCP connection to the Ollama server, which causes
        it to stop the in-progress generation and free the slot for new requests.
        Only affects this client's active stream — other models/requests are unaffected.
        """
        if not self.use_native_ollama:
            return
        response = getattr(self, "_active_stream_response", None)
        if response is not None:
            try:
                response.close()
                print("[LLMClient] Closed HTTP connection → Ollama generation aborted")
            except Exception as e:
                print(f"[LLMClient] Connection close failed (non-critical): {e}")
            self._active_stream_response = None
        else:
            print("[LLMClient] No active stream to abort (model may still be in prefill)")

    def _chat_ollama(
        self, messages: list, temperature: float, max_tokens: int, top_p: float
    ) -> str:
        """Call the native ollama library (supports optional think mode)."""
        from ollama import chat

        # Filter out None values to prevent Ollama misinterpretation
        opts = {"temperature": temperature, "top_p": top_p}
        if max_tokens is not None:
            opts["num_predict"] = max_tokens
        
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": opts,
        }
        if self.think:
            kwargs["think"] = True
        response = chat(**kwargs)

        raw = response.message.content.strip() if response.message.content else ""
        thinking = ""

        # Debug logging
        if hasattr(response.message, "thinking") and response.message.thinking:
            thinking = response.message.thinking.strip()
            print(
                f"[LLMClient] thinking length: {len(thinking)} chars"
            )
        print(f"[LLMClient] content length: {len(raw)} chars")

        # Fallback: Qwen3 thinking models sometimes put the answer in the
        # thinking field and leave content empty.  When that happens, use
        # the thinking field as the response.
        if not raw and thinking:
            print("[LLMClient] ⚠ content is empty — falling back to thinking field")
            raw = thinking

        return {"content": raw, "thinking": thinking}

    def _chat_openai(
        self, messages: list, temperature: float, max_tokens: int, top_p: float
    ) -> str:
        """Call an OpenAI-compatible API (NVIDIA NIM, etc.)."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        return {"content": response.choices[0].message.content.strip(), "thinking": ""}
