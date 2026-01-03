from typing import Dict, Any, Optional
from utils.time_statistic import OverallTimeStatistic
from utils.cost_manager import GraphBuildingCostManager, QueryCostManager
from init.logger import logger


class FinalReportGenerator:
    """Generate comprehensive final reports for Dynamic Memory system."""
    
    def __init__(self):
        self.graph_building_stats = None
        self.query_stats = []
        self.evaluation_results = None
        self.overall_stats = None
    
    def set_graph_building_stats(self, time_stats, cost_stats):
        """Set graph building statistics."""
        self.graph_building_stats = {
            'time': time_stats,
            'cost': cost_stats
        }
    
    def add_query_stats(self, time_stats, cost_stats):
        """Add query processing statistics."""
        self.query_stats.append({
            'time': time_stats,
            'cost': cost_stats
        })
    
    def set_evaluation_results(self, evaluation_results: Dict[str, Any]):
        """Set evaluation results."""
        self.evaluation_results = evaluation_results
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive final report."""
        report = {
            'system_overview': self._generate_system_overview(),
            'graph_building_summary': self._generate_graph_building_summary(),
            'query_processing_summary': self._generate_query_processing_summary(),
            'evaluation_summary': self._generate_evaluation_summary(),
            'cost_analysis': self._generate_cost_analysis(),
            'performance_analysis': self._generate_performance_analysis(),
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_system_overview(self) -> Dict[str, Any]:
        """Generate system overview."""
        total_queries = len(self.query_stats)
        total_graph_building_cost = 0
        total_query_cost = 0
        total_graph_building_time = 0
        total_query_time = 0
        
        if self.graph_building_stats:
            total_graph_building_cost = self.graph_building_stats['cost'].get('total_cost_usd', 0)
            total_graph_building_time = self.graph_building_stats['time'].get('total_graph_building_time', 0)
        
        for query_stat in self.query_stats:
            total_query_cost += query_stat['cost'].get('total_cost_usd', 0)
            total_query_time += query_stat['time'].get('total_query_time', 0)
        
        return {
            'total_queries_processed': total_queries,
            'total_system_cost_usd': round(total_graph_building_cost + total_query_cost, 4),
            'total_system_time_seconds': round(total_graph_building_time + total_query_time, 2),
            'average_cost_per_query': round(total_query_cost / total_queries, 4) if total_queries > 0 else 0,
            'average_time_per_query': round(total_query_time / total_queries, 2) if total_queries > 0 else 0
        }
    
    def _generate_graph_building_summary(self) -> Dict[str, Any]:
        """Generate graph building summary."""
        if not self.graph_building_stats:
            return {'status': 'No graph building data available'}
        
        time_stats = self.graph_building_stats['time']
        cost_stats = self.graph_building_stats['cost']
        
        return {
            'time_breakdown': {
                'chunking_time': time_stats.get('chunking_time', 0),
                'entity_extraction_time': time_stats.get('entity_extraction_time', 0),
                'relationship_extraction_time': time_stats.get('relationship_extraction_time', 0),
                'graph_construction_time': time_stats.get('graph_construction_time', 0),
                'total_time': time_stats.get('total_graph_building_time', 0)
            },
            'cost_breakdown': {
                'chunking_tokens': cost_stats.get('chunking_tokens', 0),
                'entity_extraction_tokens': cost_stats.get('entity_extraction_tokens', 0),
                'relationship_extraction_tokens': cost_stats.get('relationship_extraction_tokens', 0),
                'graph_construction_tokens': cost_stats.get('graph_construction_tokens', 0),
                'total_tokens': cost_stats.get('total_graph_building_tokens', 0),
                'total_cost_usd': cost_stats.get('total_cost_usd', 0)
            },
            'efficiency_metrics': {
                'tokens_per_second': round(
                    cost_stats.get('total_graph_building_tokens', 0) / 
                    max(time_stats.get('total_graph_building_time', 1), 0.1), 2
                ),
                'cost_per_1000_tokens': round(
                    (cost_stats.get('total_cost_usd', 0) / 
                     max(cost_stats.get('total_graph_building_tokens', 1), 1)) * 1000, 4
                )
            }
        }
    
    def _generate_query_processing_summary(self) -> Dict[str, Any]:
        """Generate query processing summary."""
        if not self.query_stats:
            return {'status': 'No query processing data available'}
        
        total_queries = len(self.query_stats)
        total_retrieval_time = 0
        total_answer_generation_time = 0
        total_retrieval_tokens = 0
        total_answer_generation_tokens = 0
        total_query_cost = 0
        
        for query_stat in self.query_stats:
            time_stats = query_stat['time']
            cost_stats = query_stat['cost']
            
            total_retrieval_time += time_stats.get('retrieval_time', 0)
            total_answer_generation_time += time_stats.get('answer_generation_time', 0)
            total_retrieval_tokens += cost_stats.get('retrieval_tokens', 0)
            total_answer_generation_tokens += cost_stats.get('answer_generation_tokens', 0)
            total_query_cost += cost_stats.get('total_cost_usd', 0)
        
        return {
            'total_queries': total_queries,
            'time_breakdown': {
                'total_retrieval_time': round(total_retrieval_time, 2),
                'total_answer_generation_time': round(total_answer_generation_time, 2),
                'total_query_time': round(total_retrieval_time + total_answer_generation_time, 2),
                'average_retrieval_time': round(total_retrieval_time / total_queries, 2) if total_queries > 0 else 0,
                'average_answer_generation_time': round(total_answer_generation_time / total_queries, 2) if total_queries > 0 else 0
            },
            'cost_breakdown': {
                'total_retrieval_tokens': total_retrieval_tokens,
                'total_answer_generation_tokens': total_answer_generation_tokens,
                'total_query_tokens': total_retrieval_tokens + total_answer_generation_tokens,
                'total_query_cost_usd': round(total_query_cost, 4),
                'average_tokens_per_query': round((total_retrieval_tokens + total_answer_generation_tokens) / total_queries, 0) if total_queries > 0 else 0,
                'average_cost_per_query': round(total_query_cost / total_queries, 4) if total_queries > 0 else 0
            },
            'efficiency_metrics': {
                'tokens_per_second': round(
                    (total_retrieval_tokens + total_answer_generation_tokens) / 
                    max(total_retrieval_time + total_answer_generation_time, 0.1), 2
                ) if total_queries > 0 else 0,
                'cost_per_1000_tokens': round(
                    (total_query_cost / 
                     max(total_retrieval_tokens + total_answer_generation_tokens, 1)) * 1000, 4
                )
            }
        }
    
    def _generate_evaluation_summary(self) -> Dict[str, Any]:
        """Generate evaluation summary."""
        if not self.evaluation_results:
            return {'status': 'No evaluation results available'}
        
        return {
            'accuracy': self.evaluation_results.get('accuracy', 0),
            'correct_answers': self.evaluation_results.get('correct_answers', 0),
            'total_answers': self.evaluation_results.get('total_answers', 0),
            'answer_rate': self.evaluation_results.get('answer_rate', 0)
        }
    
    def _generate_cost_analysis(self) -> Dict[str, Any]:
        """Generate cost analysis."""
        graph_cost = self.graph_building_stats['cost'].get('total_cost_usd', 0) if self.graph_building_stats else 0
        query_cost = sum(stat['cost'].get('total_cost_usd', 0) for stat in self.query_stats)
        total_cost = graph_cost + query_cost
        
        return {
            'graph_building_cost': round(graph_cost, 4),
            'query_processing_cost': round(query_cost, 4),
            'total_system_cost': round(total_cost, 4),
            'cost_distribution': {
                'graph_building_percentage': round((graph_cost / max(total_cost, 0.001)) * 100, 1),
                'query_processing_percentage': round((query_cost / max(total_cost, 0.001)) * 100, 1)
            },
            'cost_efficiency': {
                'cost_per_query': round(query_cost / max(len(self.query_stats), 1), 4),
                'amortized_graph_cost_per_query': round(graph_cost / max(len(self.query_stats), 1), 4)
            }
        }
    
    def _generate_performance_analysis(self) -> Dict[str, Any]:
        """Generate performance analysis."""
        graph_time = self.graph_building_stats['time'].get('total_graph_building_time', 0) if self.graph_building_stats else 0
        query_time = sum(stat['time'].get('total_query_time', 0) for stat in self.query_stats)
        total_time = graph_time + query_time
        
        return {
            'graph_building_time': round(graph_time, 2),
            'query_processing_time': round(query_time, 2),
            'total_system_time': round(total_time, 2),
            'time_distribution': {
                'graph_building_percentage': round((graph_time / max(total_time, 0.001)) * 100, 1),
                'query_processing_percentage': round((query_time / max(total_time, 0.001)) * 100, 1)
            },
            'throughput_metrics': {
                'queries_per_minute': round(len(self.query_stats) / max(total_time / 60, 0.001), 2),
                'average_query_latency': round(query_time / max(len(self.query_stats), 1), 2)
            }
        }
    
    def _generate_recommendations(self) -> Dict[str, Any]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Cost optimization recommendations
        if self.graph_building_stats and self.graph_building_stats['cost'].get('total_cost_usd', 0) > 0.1:
            recommendations.append({
                'type': 'cost_optimization',
                'category': 'graph_building',
                'message': 'Consider using smaller models for entity/relationship extraction to reduce costs',
                'impact': 'high'
            })
        
        # Performance optimization recommendations
        if self.query_stats:
            avg_query_time = sum(stat['time'].get('total_query_time', 0) for stat in self.query_stats) / len(self.query_stats)
            if avg_query_time > 10:
                recommendations.append({
                    'type': 'performance_optimization',
                    'category': 'query_processing',
                    'message': 'Query processing time is high. Consider optimizing retrieval or using faster models',
                    'impact': 'medium'
                })
        
        # Accuracy recommendations
        if self.evaluation_results and self.evaluation_results.get('accuracy', 0) < 0.8:
            recommendations.append({
                'type': 'accuracy_improvement',
                'category': 'model_performance',
                'message': 'Accuracy is below 80%. Consider improving prompts or using more powerful models',
                'impact': 'high'
            })
        
        return {
            'total_recommendations': len(recommendations),
            'recommendations': recommendations
        }
    
    def print_final_report(self):
        """Print a formatted final report to console."""
        report = self.generate_comprehensive_report()
        
        logger.info("=" * 100)
        logger.info("🏁 FINAL COMPREHENSIVE REPORT - DYNAMIC MEMORY GRAPH RAG SYSTEM")
        logger.info("=" * 100)
        
        # System Overview
        overview = report['system_overview']
        logger.info("📊 SYSTEM OVERVIEW:")
        logger.info(f"   Total Queries Processed: {overview['total_queries_processed']}")
        logger.info(f"   Total System Cost: ${overview['total_system_cost_usd']}")
        logger.info(f"   Total System Time: {overview['total_system_time_seconds']}s")
        logger.info(f"   Average Cost per Query: ${overview['average_cost_per_query']}")
        logger.info(f"   Average Time per Query: {overview['average_time_per_query']}s")
        
        # Graph Building Summary
        graph_summary = report['graph_building_summary']
        if 'status' not in graph_summary:
            logger.info("\n🏗️ GRAPH BUILDING SUMMARY:")
            time_breakdown = graph_summary['time_breakdown']
            cost_breakdown = graph_summary['cost_breakdown']
            logger.info(f"   Time - Chunking: {time_breakdown['chunking_time']}s")
            logger.info(f"   Time - Entity Extraction: {time_breakdown['entity_extraction_time']}s")
            logger.info(f"   Time - Relationship Extraction: {time_breakdown['relationship_extraction_time']}s")
            logger.info(f"   Time - Graph Construction: {time_breakdown['graph_construction_time']}s")
            logger.info(f"   Time - Total: {time_breakdown['total_time']}s")
            logger.info(f"   Tokens - Total: {cost_breakdown['total_tokens']}")
            logger.info(f"   Cost - Total: ${cost_breakdown['total_cost_usd']}")
        
        # Query Processing Summary
        query_summary = report['query_processing_summary']
        if 'status' not in query_summary:
            logger.info("\n🔍 QUERY PROCESSING SUMMARY:")
            time_breakdown = query_summary['time_breakdown']
            cost_breakdown = query_summary['cost_breakdown']
            logger.info(f"   Total Queries: {query_summary['total_queries']}")
            logger.info(f"   Time - Retrieval: {time_breakdown['total_retrieval_time']}s")
            logger.info(f"   Time - Answer Generation: {time_breakdown['total_answer_generation_time']}s")
            logger.info(f"   Time - Total: {time_breakdown['total_query_time']}s")
            logger.info(f"   Tokens - Total: {cost_breakdown['total_query_tokens']}")
            logger.info(f"   Cost - Total: ${cost_breakdown['total_query_cost_usd']}")
        
        # Evaluation Summary
        eval_summary = report['evaluation_summary']
        if 'status' not in eval_summary:
            logger.info("\n📈 EVALUATION SUMMARY:")
            logger.info(f"   Accuracy: {eval_summary['accuracy']:.2%}")
            logger.info(f"   Correct Answers: {eval_summary['correct_answers']}/{eval_summary['total_answers']}")
            logger.info(f"   Answer Rate: {eval_summary['answer_rate']:.2%}")
        
        # Cost Analysis
        cost_analysis = report['cost_analysis']
        logger.info("\n💰 COST ANALYSIS:")
        logger.info(f"   Graph Building Cost: ${cost_analysis['graph_building_cost']}")
        logger.info(f"   Query Processing Cost: ${cost_analysis['query_processing_cost']}")
        logger.info(f"   Total System Cost: ${cost_analysis['total_system_cost']}")
        logger.info(f"   Cost per Query: ${cost_analysis['cost_efficiency']['cost_per_query']}")
        
        # Performance Analysis
        perf_analysis = report['performance_analysis']
        logger.info("\n⚡ PERFORMANCE ANALYSIS:")
        logger.info(f"   Graph Building Time: {perf_analysis['graph_building_time']}s")
        logger.info(f"   Query Processing Time: {perf_analysis['query_processing_time']}s")
        logger.info(f"   Total System Time: {perf_analysis['total_system_time']}s")
        logger.info(f"   Queries per Minute: {perf_analysis['throughput_metrics']['queries_per_minute']}")
        
        # Recommendations
        recommendations = report['recommendations']
        if recommendations['total_recommendations'] > 0:
            logger.info("\n💡 OPTIMIZATION RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations['recommendations'], 1):
                logger.info(f"   {i}. [{rec['type'].upper()}] {rec['message']} (Impact: {rec['impact']})")
        
        logger.info("=" * 100)
        
        return report
