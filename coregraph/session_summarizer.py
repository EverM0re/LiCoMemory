import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any, Tuple
import json
import asyncio
from init.logger import logger
from prompt.summary_prompt import SUMMARY_PROMPT, ADDITION_PROMPT


class SessionSummarizer:
    
    def __init__(self, llm_manager):
        self.llm = llm_manager
        self.existing_summaries = {}
        logger.info("Session Summarizer initialized")
    
    async def summarize_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            session_id = session_data.get('session_id', 'unknown')
            session_time = str(session_data.get('session_time', 'unknown'))
            context = session_data.get('context', '')
            session_json = {
                "session_id": session_id,
                "session_time": session_time,
                "context": context
            }

            formatted_text = f"Session ID: {session_id}\nSession Time: {session_time}\nContext: {context}"
            prompt = SUMMARY_PROMPT.replace('{text}', formatted_text)
            response = await self.llm.generate(prompt)
            summary = self._parse_summary_response(response, session_id, session_time)
            
            logger.debug(f"Generated summary for session {session_id}")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate summary for session: {e}")
            return {
                'session_id': session_data.get('session_id', 'unknown'),
                'session_time': str(session_data.get('session_time', 'unknown')),  # Convert to string
                'summary_status': 'failed',
                'error': str(e)
            }
    
    def _parse_summary_response(self, response: str, session_id: str, session_time: str) -> Dict[str, Any]:
        try:
            if response.strip().startswith('{'):
                summary_data = json.loads(response)
                if 'session_id' not in summary_data:
                    summary_data['session_id'] = session_id
                if 'session_time' not in summary_data:
                    summary_data['session_time'] = session_time.split(' ')[0] if ' ' in session_time else session_time.split('T')[0] if 'T' in session_time else session_time
                summary_data.pop('summary_status', None)
                summary_data.pop('raw_response', None)
                logger.debug(f"Successfully parsed JSON summary for session {session_id}")
                return summary_data
            else:
                import re
                json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response)
                if json_match:
                    try:
                        summary_data = json.loads(json_match.group(1))
                        if 'session_id' not in summary_data:
                            summary_data['session_id'] = session_id
                        if 'session_time' not in summary_data:
                            summary_data['session_time'] = session_time.split(' ')[0] if ' ' in session_time else session_time.split('T')[0] if 'T' in session_time else session_time
                        summary_data.pop('summary_status', None)
                        summary_data.pop('raw_response', None)
                        logger.debug(f"Successfully extracted embedded JSON summary for session {session_id}")
                        return summary_data
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse embedded JSON in response for session {session_id}")
                return {
                    'session_id': session_id,
                    'session_time': session_time.split(' ')[0] if ' ' in session_time else session_time.split('T')[0] if 'T' in session_time else session_time,
                    'keys': 'session, conversation',
                    'context': {
                        'theme_1': 'General conversation',
                        'summary_1': response.strip()[:200] + '...' if len(response) > 200 else response.strip()
                    }
                }
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse summary response as JSON for session {session_id}")
            import re
            json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response)
            if json_match:
                try:
                    summary_data = json.loads(json_match.group(1))
                    if 'session_id' not in summary_data:
                        summary_data['session_id'] = session_id
                    if 'session_time' not in summary_data:
                        summary_data['session_time'] = session_time.split(' ')[0] if ' ' in session_time else session_time.split('T')[0] if 'T' in session_time else session_time
                    summary_data.pop('summary_status', None)
                    summary_data.pop('raw_response', None)
                    logger.debug(f"Successfully extracted embedded JSON summary for session {session_id}")
                    return summary_data
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse embedded JSON in response for session {session_id}")
            return {
                'session_id': session_id,
                'session_time': session_time.split(' ')[0] if ' ' in session_time else session_time.split('T')[0] if 'T' in session_time else session_time,
                'keys': 'session, conversation',
                'context': {
                    'theme_1': 'General conversation',
                    'summary_1': response.strip()[:200] + '...' if len(response) > 200 else response.strip()
                }
            }
    
    async def summarize_sessions(self, sessions: List[Dict[str, Any]], progress_bar=None) -> List[Dict[str, Any]]:
        """Generate summaries for multiple sessions.
        
        Args:
            sessions: List of sessions to summarize
            progress_bar: Optional tqdm progress bar to update as each request completes
        """
        if not sessions:
            return []

        if hasattr(self.llm, 'enable_concurrent') and self.llm.enable_concurrent:
            prompts = []
            session_metas = []
            
            for session in sessions:
                session_id = session.get('session_id', 'unknown')
                session_time = str(session.get('session_time', 'unknown'))
                context = session.get('context', '')
                session_json = {
                    "session_id": session_id,
                    "session_time": session_time,
                    "context": context
                }
                formatted_text = f"Session ID: {session_id}\nSession Time: {session_time}\nContext: {context}"
                prompt = SUMMARY_PROMPT.replace('{text}', formatted_text)
                prompts.append(prompt)
                session_metas.append(session)
            
            try:
                # Pass progress_bar to batch_generate so it updates as each request completes
                responses = await self.llm.batch_generate(prompts, progress_bar=progress_bar)
            except Exception as e:
                logger.error(f"Failed to batch generate summaries: {e}")
                responses = ["" for _ in prompts]
                # Update progress bar for failed requests
                if progress_bar:
                    progress_bar.update(len(prompts))

            summaries = []
            for i, (response, session) in enumerate(zip(responses, session_metas)):
                try:
                    session_id = session.get('session_id', 'unknown')
                    session_time = str(session.get('session_time', 'unknown'))  # Convert to string
                    summary = self._parse_summary_response(response, session_id, session_time)
                    summaries.append(summary)
                except Exception as e:
                    logger.error(f"Failed to process summary response for session {i}: {e}")
                    summaries.append({
                        'session_id': session.get('session_id', 'unknown'),
                        'session_time': str(session.get('session_time', 'unknown')),  # Convert to string
                        'summary_status': 'failed',
                        'error': str(e)
                    })
        else:
            summaries = []
            
            for session in sessions:
                try:
                    summary = await self.summarize_session(session)
                    summaries.append(summary)
                except Exception as e:
                    logger.error(f"Failed to summarize session {session.get('session_id', 'unknown')}: {e}")
                    summaries.append({
                        'session_id': session.get('session_id', 'unknown'),
                        'session_time': str(session.get('session_time', 'unknown')),
                        'summary_status': 'failed',
                        'error': str(e)
                    })
                finally:
                    if progress_bar:
                        progress_bar.update(1)
        
        logger.info(f"Generated summaries for {len(summaries)} sessions")
        return summaries
    
    def save_summaries(self, summaries: List[Dict[str, Any]], output_path: str) -> None:
        """Save summaries to a JSON file."""
        try:
            def convert_timestamps(obj):
                if isinstance(obj, dict):
                    return {k: convert_timestamps(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_timestamps(item) for item in obj]
                elif hasattr(obj, 'isoformat'):
                    return obj.isoformat()
                return obj

            summaries_serializable = convert_timestamps(summaries)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summaries_serializable, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(summaries)} summaries to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save summaries to {output_path}: {e}")
    
    def load_summaries(self, input_path: str) -> List[Dict[str, Any]]:
        """Load summaries from a JSON file."""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                summaries = json.load(f)
            logger.info(f"Loaded {len(summaries)} summaries from {input_path}")
            self.existing_summaries = {s.get('session_id', ''): s for s in summaries if s.get('session_id')}
            return summaries
        except Exception as e:
            logger.error(f"Failed to load summaries from {input_path}: {e}")
            return []
    
    async def update_summary_with_chunk(self, chunk: Dict[str, Any], existing_summary: Dict[str, Any] = None) -> Dict[str, Any]:
        try:
            session_id = chunk.get('session_id', 'unknown')
            session_time = str(chunk.get('session_time', 'unknown'))
            chunk_text = chunk.get('text', '')
            
            if existing_summary is None:
                existing_summary = self.existing_summaries.get(session_id)
            
            if existing_summary:
                logger.info(f"Updating existing summary for session {session_id}")
                summary = await self._update_existing_summary(existing_summary, chunk_text)
            else:
                logger.info(f"Creating new summary for session {session_id}")
                formatted_text = f"Session ID: {session_id}\nSession Time: {session_time}\nContext: {chunk_text}"
                prompt = SUMMARY_PROMPT.replace('{text}', formatted_text)
                response = await self.llm.generate(prompt)
                summary = self._parse_summary_response(response, session_id, session_time)
            
            self.existing_summaries[session_id] = summary
            logger.debug(f"Summary updated for session {session_id}")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to update summary for chunk: {e}")
            return {
                'session_id': chunk.get('session_id', 'unknown'),
                'session_time': str(chunk.get('session_time', 'unknown')),
                'summary_status': 'failed',
                'error': str(e)
            }
    
    async def _update_existing_summary(self, existing_summary: Dict[str, Any], new_chunk_text: str) -> Dict[str, Any]:
        try:
            existing_summary_str = json.dumps(existing_summary, ensure_ascii=False, indent=2)
            prompt = ADDITION_PROMPT.format(
                summary=existing_summary_str,
                text=new_chunk_text
            )
            
            response = await self.llm.generate(prompt)
            session_id = existing_summary.get('session_id', 'unknown')
            session_time = existing_summary.get('session_time', 'unknown')
            updated_summary = self._parse_summary_response(response, session_id, session_time)
            logger.debug(f"Successfully updated summary for session {session_id}")
            return updated_summary
            
        except Exception as e:
            logger.error(f"Failed to update existing summary: {e}")
            return existing_summary
