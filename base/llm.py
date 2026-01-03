from typing import Dict, Any, List, Optional
import openai
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from init.logger import logger
from utils.token_counter import count_input_tokens, count_output_tokens, get_token_cost
from utils.cost_manager import CostManager, TokenCostManager

class LLMManager:
    """Manager for Large Language Model interactions."""

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", max_tokens: int = 32768, 
                 base_url: str = None, enable_concurrent: bool = True, max_concurrent: int = 16, 
                 timeout: int = 600, max_budget: float = 100.0):
        """Initialize LLM manager."""
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.base_url = base_url or "https://api.openai.com/v1"
        self.enable_concurrent = enable_concurrent
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )
        self.semaphore = asyncio.Semaphore(self.max_concurrent) if self.enable_concurrent else None

        if api_key == "demo-key" or "open-llm" in model.lower():
            self.cost_manager = TokenCostManager(max_budget=max_budget)
        else:
            self.cost_manager = CostManager(max_budget=max_budget)

        logger.info(f"LLM Manager initialized with model: {model}, concurrent: {enable_concurrent}, max_concurrent: {max_concurrent}")

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using LLM with concurrent control."""
        try:
            if self.semaphore:
                async with self.semaphore:
                    return await self._generate_internal(prompt, **kwargs)
            else:
                return await self._generate_internal(prompt, **kwargs)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return ""

    async def _generate_internal(self, prompt: str, **kwargs) -> str:
        """Internal generate method without semaphore control."""
        messages = [{"role": "user", "content": prompt}]
        prompt_tokens = count_input_tokens(messages, self.model)
        
        # For demo purposes, return mock response
        if self.api_key == "demo-key":
            logger.info(f"Using mock LLM response (demo mode)")
            mock_response = f"Mock response for: {prompt[:50]}..."
            completion_tokens = count_output_tokens(mock_response, self.model)
            self.cost_manager.update_cost(prompt_tokens, completion_tokens, self.model)
            return mock_response

        # Real LLM call using OpenAI 1.0+ API
        # Extract specific parameters to avoid duplication
        max_tokens = kwargs.pop('max_tokens', 1000)
        temperature = kwargs.pop('temperature', 0.1)
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
        except Exception as api_error:
            logger.error(f"LLM API call failed: {api_error}")
            raise
        response_content = response.choices[0].message.content.strip()
        
        if hasattr(response, 'usage') and response.usage:
            completion_tokens = response.usage.completion_tokens
            actual_prompt_tokens = response.usage.prompt_tokens
        else:
            completion_tokens = count_output_tokens(response_content, self.model)
            actual_prompt_tokens = prompt_tokens
        
        self.cost_manager.update_cost(actual_prompt_tokens, completion_tokens, self.model)
        
        return response_content

    async def extract_entities(self, text: str, session_time: str = "") -> List[Dict[str, Any]]:
        """Extract entities from text using professional prompt.
        
        Args:
            text: The text to extract entities from
            session_time: Optional session/query time for temporal entity processing
        """
        try:
            from ..prompt.entity_prompt import QUERY_ENTITY_EXTRACTION_PROMPT
        except ImportError:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from prompt.entity_prompt import QUERY_ENTITY_EXTRACTION_PROMPT

        prompt = QUERY_ENTITY_EXTRACTION_PROMPT.format(text=text, session_time=session_time)
        logger.info(f"Sending professional entity extraction prompt to LLM (model: {self.model}, session_time: {session_time})")
        response = await self.generate(prompt)
        logger.debug(f"LLM response received (first 200 chars): {response[:200]}...")

        # Parse the custom format from the professional prompt
        entities = self._parse_entity_extraction_response(response)
        logger.info(f"Successfully extracted {len(entities)} entities using professional prompt")
        return entities

    def _parse_entity_extraction_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse the custom format from entity extraction prompt."""
        entities = []
        records = response.split("##")
        for record in records:
            record = record.strip()
            if not record or record == "##END##":
                continue
            if record.startswith('("entity"|'):
                try:
                    content = record.strip()
                    if content.endswith(')'):
                        content = content[10:-1]
                    else:
                        content = content[10:]

                    # Split by the pipe delimiter |
                    parts = content.split('|')

                    if len(parts) >= 2:
                        # Clean up any remaining quotes from each part
                        entity_name = parts[0].strip('"').strip()
                        entity_type = parts[1].strip('"').strip()

                        # Validate that we have both name and type
                        if entity_name and entity_type:
                            entity = {
                                'entity': entity_name,
                                'type': entity_type,
                            }
                            entities.append(entity)
                            logger.debug(f"Successfully parsed entity: {entity_name} ({entity_type})")
                    else:
                        logger.warning(f"Insufficient parts in entity record: {record} (got {len(parts)} parts, expected at least 2)")
                        
                except Exception as e:
                    logger.warning(f"Failed to parse entity record: {record}, error: {e}")
                    logger.debug(f"Content was: {content if 'content' in locals() else 'N/A'}")

        return entities

    async def batch_generate(self, prompts: List[str], progress_bar=None, **kwargs) -> List[str]:
        """Batch generate text with concurrent support.
        
        Args:
            prompts: List of prompts to generate
            progress_bar: Optional tqdm progress bar to update as each request completes
            **kwargs: Additional arguments for generate method
        """
        if not self.enable_concurrent:
            logger.info("Concurrent not enabled, using sequential processing")
            results = []
            for prompt in prompts:
                result = await self.generate(prompt, **kwargs)
                results.append(result)
                if progress_bar:
                    progress_bar.update(1)
            return results
        
        logger.info(f"Batch concurrent processing {len(prompts)} requests, max concurrent: {self.max_concurrent}")
        
        # Create a list to store results
        results_list = [None] * len(prompts)
        
        async def generate_with_progress(prompt, index):
            """Generate with progress tracking."""
            try:
                result = await self.generate(prompt, **kwargs)
                results_list[index] = result
            except Exception as e:
                logger.error(f"Batch processing request {index} failed: {e}")
                results_list[index] = ""
            finally:
                # Update progress bar when each request completes
                if progress_bar:
                    progress_bar.update(1)
        
        tasks = [generate_with_progress(prompt, i) for i, prompt in enumerate(prompts)]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        return results_list

    async def batch_extract_entities(self, texts: List[str], progress_bar=None) -> List[List[Dict[str, Any]]]:
        """Batch extract entities with concurrent support.
        
        Args:
            texts: List of texts to extract entities from
            progress_bar: Optional tqdm progress bar to update as each request completes
        """
        if not self.enable_concurrent:
            logger.info("Concurrent not enabled, using sequential entity extraction")
            results = []
            for text in texts:
                result = await self.extract_entities(text)
                results.append(result)
                if progress_bar:
                    progress_bar.update(1)
            return results

        logger.info(f"Batch concurrent entity extraction {len(texts)} texts, max concurrent: {self.max_concurrent}")
        
        # Create a list to store results
        results_list = [None] * len(texts)
        
        async def extract_with_progress(text, index):
            """Extract entities with progress tracking."""
            try:
                result = await self.extract_entities(text)
                results_list[index] = result
            except Exception as e:
                logger.error(f"Batch entity extraction request {index} failed: {e}")
                results_list[index] = []
            finally:
                # Update progress bar when each request completes
                if progress_bar:
                    progress_bar.update(1)
        
        tasks = [extract_with_progress(text, i) for i, text in enumerate(texts)]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        return results_list

    def get_costs(self):
        """Get current cost statistics."""
        return self.cost_manager.get_costs()

    def get_last_stage_cost(self):
        """Get last stage cost statistics."""
        return self.cost_manager.get_last_stage_cost()

    def get_cost_summary(self):
        """Get cost summary."""
        return self.cost_manager.get_cost_summary()

    def check_budget(self):
        """Check budget."""
        return self.cost_manager.check_budget()

    def set_max_budget(self, budget: float):
        """Set maximum budget."""
        self.cost_manager.max_budget = budget
        logger.info(f"💰 Max budget set to: ${budget:.2f}")
