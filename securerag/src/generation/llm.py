"""
LLM Client - Wrapper for language model generation
"""

import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class LLMClient:
    """
    LLM client with OpenAI API support and mock fallback.
    
    In production, supports:
    - OpenAI (GPT-4, GPT-3.5)
    - Azure OpenAI
    - Local models via Ollama
    
    Falls back to mock responses for demo/testing.
    """
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1000
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = None
        
        self._init_client()
    
    def _init_client(self):
        """Initialize OpenAI client if key available"""
        if self.api_key:
            try:
                from openai import AsyncOpenAI
                self.client = AsyncOpenAI(api_key=self.api_key)
                logger.info("OpenAI client initialized")
            except ImportError:
                logger.warning("OpenAI not installed, using mock responses")
                self.client = None
        else:
            logger.info("No API key, using mock responses")
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate response from LLM.
        
        Args:
            prompt: User prompt (with context)
            system_prompt: Optional system instructions
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Returns:
            Generated text response
        """
        temp = temperature or self.temperature
        tokens = max_tokens or self.max_tokens
        
        if self.client:
            return await self._generate_openai(prompt, system_prompt, temp, tokens)
        else:
            return self._generate_mock(prompt, system_prompt)
    
    async def _generate_openai(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate using OpenAI API"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            return self._generate_mock(prompt, system_prompt)
    
    def _generate_mock(
        self,
        prompt: str,
        system_prompt: Optional[str]
    ) -> str:
        """
        Generate mock response for demo/testing.
        
        Parses context and returns reasonable response.
        """
        prompt_lower = prompt.lower()
        
        # Extract context if present
        if "[Source:" in prompt:
            # Find the actual question
            lines = prompt.split("\n")
            question = ""
            context_text = ""
            
            for line in lines:
                if "Question:" in line or "Query:" in line:
                    question = line.split(":", 1)[1].strip() if ":" in line else line
                if "[Source:" in line or line.strip().startswith("Section"):
                    context_text += line + " "
            
            question = question or prompt[:100]
        else:
            question = prompt
            context_text = ""
        
        # Generate contextual response based on question content
        if "refund" in prompt_lower:
            return (
                "Based on the Refund Policy document, refunds are allowed within 14 days of delivery. "
                "To request a refund:\n"
                "1. Log into your account\n"
                "2. Navigate to Order History\n"
                "3. Select the order and click 'Request Refund'\n"
                "4. Provide reason for return\n\n"
                "Note: Digital downloads, personalized items, and final sale items are non-refundable. "
                "Refunds are processed within 5-7 business days after receiving the returned item.\n\n"
                "[Source: refund_policy_v3, Section 1-4]"
            )
        elif "merchant" in prompt_lower or "onboarding" in prompt_lower:
            return (
                "According to the Merchant Onboarding Guide, new merchants need:\n\n"
                "**Business Documentation:**\n"
                "- Valid business license\n"
                "- Tax identification number\n"
                "- Bank account for payouts\n\n"
                "**Compliance Requirements:**\n"
                "- PCI-DSS compliance certification\n"
                "- Privacy policy\n"
                "- Terms of service\n\n"
                "**Integration Steps:**\n"
                "1. Register at merchant.example.com\n"
                "2. Complete KYC verification\n"
                "3. Set up payment methods\n"
                "4. Configure webhook endpoints\n"
                "5. Test in sandbox environment\n"
                "6. Go live after approval\n\n"
                "Standard review takes 3-5 business days.\n\n"
                "[Source: merchant_onboarding]"
            )
        elif "password" in prompt_lower or "api key" in prompt_lower or "secret" in prompt_lower:
            return (
                "I cannot provide sensitive information like passwords, API keys, or secrets. "
                "Please contact your system administrator for credential management."
            )
        else:
            # Generic response when no specific match
            if context_text:
                return (
                    f"Based on the available documentation, I can see information related to your query. "
                    f"However, I need to provide a response grounded in the specific context provided.\n\n"
                    f"Could you please clarify your question or ask about a specific topic "
                    f"covered in the documentation?"
                )
            else:
                return (
                    "I don't have enough information in the available documents to answer "
                    "this question accurately. Please try rephrasing your question or ask "
                    "about a topic covered in the knowledge base."
                )
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)"""
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model(self.model)
            return len(enc.encode(text))
        except ImportError:
            # Rough estimate: ~4 chars per token
            return len(text) // 4
