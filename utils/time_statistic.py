import time
from typing import Dict, Any, Optional
from init.logger import logger


class TimeStatistic:
    
    def __init__(self):
        self._start_time = {}
        self._count = {}
        self._total_time = {}
        self._stage_time = []
        self._stage_names = []

    def start_stage(self, stage_name: str = None):
        current_time = time.time()
        self._stage_time.append(current_time)
        if stage_name:
            self._stage_names.append(stage_name)
            logger.info(f"⏱️ Starting stage: {stage_name}")
        else:
            self._stage_names.append(f"Stage_{len(self._stage_time)}")
    
    def stop_last_stage(self) -> float:
        if len(self._stage_time) < 2:
            logger.warning("No previous stage to stop.")
            return 0.0
            
        current_time = time.time()
        self._stage_time.append(current_time)
        
        inc_time = self._stage_time[-1] - self._stage_time[-2]
        
        if len(self._stage_names) > 0:
            stage_name = self._stage_names[-1]
            self._add_time(stage_name, inc_time)
            logger.info(f"⏱️ Completed stage '{stage_name}': {inc_time:.2f}s")
        
        return inc_time
    
    def start(self, name: str):
        self._start_time[name] = time.time()
        logger.debug(f"⏱️ Started timing: {name}")

    def end(self, name: str) -> str:
        """End timing for a specific operation and return time string."""
        if name not in self._start_time:
            logger.error(f"TimeStatistic: {name} not started")
            return "0.00"
            
        inc_time = time.time() - self._start_time[name]
        self._add_time(name, inc_time)
        del self._start_time[name]
        
        time_str = f"{inc_time:.2f}"
        logger.debug(f"⏱️ Completed timing: {name} - {time_str}s")
        return time_str

    def _add_time(self, name: str, inc_time: float):
        """Add time to the statistics for a named operation."""
        if name not in self._total_time:
            self._total_time[name] = 0
            self._count[name] = 0
        self._total_time[name] += inc_time
        self._count[name] += 1

    def get_statistics(self, name: str) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        if name not in self._total_time:
            logger.error(f"TimeStatistic: {name} has no statistics")
            return {"Total time(s)": 0, "Count": 0, "Average time (s)": 0}
            
        return {
            "Total time(s)": round(self._total_time[name], 2),
            "Count": self._count[name],
            "Average time (s)": round(self._total_time[name] / self._count[name], 2)
        }

    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all tracked operations."""
        return {name: self.get_statistics(name) for name in self._total_time.keys()}

    def reset(self):
        """Reset all statistics."""
        self._start_time = {}
        self._count = {}
        self._total_time = {}
        self._stage_time = []
        self._stage_names = []
        logger.info("⏱️ Time statistics reset")


class GraphBuildingTimeStatistic(TimeStatistic):
    """Specialized time statistic for graph building operations."""
    
    def __init__(self):
        super().__init__()
        self.chunking_time = 0
        self.entity_extraction_time = 0
        self.relationship_extraction_time = 0
        self.graph_construction_time = 0
        self.summary_generation_time = 0
        self.total_graph_building_start_time = None
        self.total_graph_building_end_time = None
        
    def start_chunking(self):
        """Start timing for chunking operations."""
        self.start("chunking")
        
    def end_chunking(self):
        """End timing for chunking operations."""
        time_str = self.end("chunking")
        self.chunking_time += float(time_str)
        logger.info(f"📄 Chunking completed in {time_str}s")
        
    def start_entity_extraction(self):
        """Start timing for entity extraction operations."""
        self.start("entity_extraction")
        
    def end_entity_extraction(self):
        """End timing for entity extraction operations."""
        time_str = self.end("entity_extraction")
        self.entity_extraction_time += float(time_str)
        logger.info(f"Entity extraction completed in {time_str}s")
        
    def start_relationship_extraction(self):
        """Start timing for relationship extraction operations."""
        self.start("relationship_extraction")
        
    def end_relationship_extraction(self):
        """End timing for relationship extraction operations."""
        time_str = self.end("relationship_extraction")
        self.relationship_extraction_time += float(time_str)
        logger.info(f"🔗 Relationship extraction completed in {time_str}s")
        
    def start_graph_construction(self):
        """Start timing for graph construction operations."""
        self.start("graph_construction")
        
    def end_graph_construction(self):
        """End timing for graph construction operations."""
        time_str = self.end("graph_construction")
        self.graph_construction_time += float(time_str)
        logger.info(f"Graph construction completed in {time_str}s")
        
    def start_summary_generation(self):
        """Start timing for summary generation operations."""
        self.start("summary_generation")
        
    def end_summary_generation(self):
        """End timing for summary generation operations."""
        time_str = self.end("summary_generation")
        self.summary_generation_time += float(time_str)
        logger.info(f"📝 Summary generation completed in {time_str}s")
        
    def start_total_graph_building(self):
        """Start timing for total graph building process."""
        import time
        self.total_graph_building_start_time = time.time()
        logger.info("🏗️ Starting total graph building process")
        
    def end_total_graph_building(self):
        """End timing for total graph building process."""
        import time
        self.total_graph_building_end_time = time.time()
        if self.total_graph_building_start_time:
            total_time = self.total_graph_building_end_time - self.total_graph_building_start_time
            logger.info(f"🏗️ Total graph building completed in {total_time:.2f}s")
        
    def get_graph_building_summary(self) -> Dict[str, Any]:
        """Get detailed time summary for graph building."""
        # Use actual total time if available, otherwise calculate from components
        if self.total_graph_building_start_time and self.total_graph_building_end_time:
            total_time = self.total_graph_building_end_time - self.total_graph_building_start_time
        else:
            total_time = (
                self.chunking_time + 
                self.entity_extraction_time + 
                self.relationship_extraction_time + 
                self.graph_construction_time +
                self.summary_generation_time
            )
        
        return {
            "chunking_time": round(self.chunking_time, 2),
            "entity_extraction_time": round(self.entity_extraction_time, 2),
            "relationship_extraction_time": round(self.relationship_extraction_time, 2),
            "graph_construction_time": round(self.graph_construction_time, 2),
            "summary_generation_time": round(self.summary_generation_time, 2),
            "total_graph_building_time": round(total_time, 2),
            "chunking_percentage": round((self.chunking_time / total_time * 100), 1) if total_time > 0 else 0,
            "entity_extraction_percentage": round((self.entity_extraction_time / total_time * 100), 1) if total_time > 0 else 0,
            "relationship_extraction_percentage": round((self.relationship_extraction_time / total_time * 100), 1) if total_time > 0 else 0,
            "graph_construction_percentage": round((self.graph_construction_time / total_time * 100), 1) if total_time > 0 else 0,
            "summary_generation_percentage": round((self.summary_generation_time / total_time * 100), 1) if total_time > 0 else 0
        }


class QueryTimeStatistic(TimeStatistic):
    """Specialized time statistic for query operations with detailed step tracking."""
    
    def __init__(self):
        super().__init__()
        self.retrieval_time = 0
        self.answer_generation_time = 0

        self.entity_extraction_time = 0
        self.similar_entity_search_time = 0
        self.triple_retrieval_time = 0
        self.summary_retrieval_time = 0
        self.triple_reranking_time = 0
        self.chunk_retrieval_time = 0
        self.prompt_generation_time = 0
        
    def start_retrieval(self):
        """Start timing for retrieval operations."""
        self.start("retrieval")
        
    def end_retrieval(self):
        """End timing for retrieval operations."""
        time_str = self.end("retrieval")
        self.retrieval_time += float(time_str)
        logger.info(f"🔍 Retrieval completed in {time_str}s")
        
    def start_answer_generation(self):
        """Start timing for answer generation operations."""
        self.start("answer_generation")
        
    def end_answer_generation(self):
        """End timing for answer generation operations."""
        time_str = self.end("answer_generation")
        self.answer_generation_time += float(time_str)
        logger.info(f"💬 Answer generation completed in {time_str}s")
    
    def start_entity_extraction(self):
        """Start timing for entity extraction from query."""
        self.start("entity_extraction")
        
    def end_entity_extraction(self):
        """End timing for entity extraction."""
        time_str = self.end("entity_extraction")
        self.entity_extraction_time += float(time_str)
        logger.info(f"Entity extraction completed in {time_str}s")
        
    def start_similar_entity_search(self):
        """Start timing for similar entity search."""
        self.start("similar_entity_search")
        
    def end_similar_entity_search(self):
        """End timing for similar entity search."""
        time_str = self.end("similar_entity_search")
        self.similar_entity_search_time += float(time_str)
        logger.info(f"Similar entity search completed in {time_str}s")
        
    def start_triple_retrieval(self):
        """Start timing for triple retrieval."""
        self.start("triple_retrieval")
        
    def end_triple_retrieval(self):
        """End timing for triple retrieval."""
        time_str = self.end("triple_retrieval")
        self.triple_retrieval_time += float(time_str)
        logger.info(f"Triple retrieval completed in {time_str}s")
        
    def start_summary_retrieval(self):
        """Start timing for summary retrieval."""
        self.start("summary_retrieval")
        
    def end_summary_retrieval(self):
        """End timing for summary retrieval."""
        time_str = self.end("summary_retrieval")
        self.summary_retrieval_time += float(time_str)
        logger.info(f"Summary retrieval completed in {time_str}s")
        
    def start_triple_reranking(self):
        """Start timing for triple reranking."""
        self.start("triple_reranking")
        
    def end_triple_reranking(self):
        """End timing for triple reranking."""
        time_str = self.end("triple_reranking")
        self.triple_reranking_time += float(time_str)
        logger.info(f"Triple reranking completed in {time_str}s")
        
    def start_chunk_retrieval(self):
        """Start timing for chunk retrieval."""
        self.start("chunk_retrieval")
        
    def end_chunk_retrieval(self):
        """End timing for chunk retrieval."""
        time_str = self.end("chunk_retrieval")
        self.chunk_retrieval_time += float(time_str)
        logger.info(f"Chunk retrieval completed in {time_str}s")
        
    def start_prompt_generation(self):
        """Start timing for prompt generation."""
        self.start("prompt_generation")
        
    def end_prompt_generation(self):
        """End timing for prompt generation."""
        time_str = self.end("prompt_generation")
        self.prompt_generation_time += float(time_str)
        logger.info(f"Prompt generation completed in {time_str}s")
        
    def get_query_summary(self) -> Dict[str, Any]:
        """Get detailed time summary for query operations with breakdown."""
        total_time = self.retrieval_time + self.answer_generation_time
        
        detailed_retrieval_time = (
            self.entity_extraction_time +
            self.similar_entity_search_time +
            self.triple_retrieval_time +
            self.summary_retrieval_time +
            self.triple_reranking_time +
            self.chunk_retrieval_time +
            self.prompt_generation_time
        )
        
        return {
            "retrieval_time": round(self.retrieval_time, 2),
            "answer_generation_time": round(self.answer_generation_time, 2),
            "total_query_time": round(total_time, 2),
            
            "detailed_retrieval_breakdown": {
                "entity_extraction_time": round(self.entity_extraction_time, 2),
                "similar_entity_search_time": round(self.similar_entity_search_time, 2),
                "triple_retrieval_time": round(self.triple_retrieval_time, 2),
                "summary_retrieval_time": round(self.summary_retrieval_time, 2),
                "triple_reranking_time": round(self.triple_reranking_time, 2),
                "chunk_retrieval_time": round(self.chunk_retrieval_time, 2),
                "prompt_generation_time": round(self.prompt_generation_time, 2),
                "detailed_total": round(detailed_retrieval_time, 2)
            },
            
            "retrieval_percentage": round((self.retrieval_time / total_time * 100), 1) if total_time > 0 else 0,
            "answer_generation_percentage": round((self.answer_generation_time / total_time * 100), 1) if total_time > 0 else 0,
            
            "retrieval_step_percentages": {
                "entity_extraction": round((self.entity_extraction_time / self.retrieval_time * 100), 1) if self.retrieval_time > 0 else 0,
                "similar_entity_search": round((self.similar_entity_search_time / self.retrieval_time * 100), 1) if self.retrieval_time > 0 else 0,
                "triple_retrieval": round((self.triple_retrieval_time / self.retrieval_time * 100), 1) if self.retrieval_time > 0 else 0,
                "summary_retrieval": round((self.summary_retrieval_time / self.retrieval_time * 100), 1) if self.retrieval_time > 0 else 0,
                "triple_reranking": round((self.triple_reranking_time / self.retrieval_time * 100), 1) if self.retrieval_time > 0 else 0,
                "chunk_retrieval": round((self.chunk_retrieval_time / self.retrieval_time * 100), 1) if self.retrieval_time > 0 else 0,
                "prompt_generation": round((self.prompt_generation_time / self.retrieval_time * 100), 1) if self.retrieval_time > 0 else 0
            }
        }


class OverallTimeStatistic:
    """Overall time statistic that combines graph building and query timing."""
    
    def __init__(self):
        self.graph_time = GraphBuildingTimeStatistic()
        self.query_time = QueryTimeStatistic()
        self.overall_start_time = None
        self.overall_end_time = None
        
    def start_overall(self):
        """Start overall timing."""
        self.overall_start_time = time.time()
        logger.info("🚀 Starting overall timing")
        
    def end_overall(self):
        """End overall timing."""
        self.overall_end_time = time.time()
        total_time = self.overall_end_time - self.overall_start_time
        logger.info(f"🏁 Overall timing completed: {total_time:.2f}s")
        return total_time
        
    def get_comprehensive_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all timing."""
        graph_summary = self.graph_time.get_graph_building_summary()
        query_summary = self.query_time.get_query_summary()
        
        overall_time = 0
        if self.overall_start_time and self.overall_end_time:
            overall_time = self.overall_end_time - self.overall_start_time
        
        return {
            "overall_time": round(overall_time, 2),
            "graph_building": graph_summary,
            "query_processing": query_summary,
            "total_graph_time": graph_summary["total_graph_building_time"],
            "total_query_time": query_summary["total_query_time"],
            "graph_vs_query_ratio": round(
                graph_summary["total_graph_building_time"] / query_summary["total_query_time"], 2
            ) if query_summary["total_query_time"] > 0 else 0
        }
