import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from typing import List, Dict, Any, Tuple
import pandas as pd
from init.logger import logger

plt.rcParams['font.family'] = ['Liberation Serif', 'DejaVu Serif', 'serif']
plt.rcParams['font.size'] = 10

class QueryResultVisualizer:
    
    def __init__(self, config):
        self.config = config
        self.output_dir = getattr(config, 'working_dir', './results')
        logger.info("Query Result Visualizer initialized")
    
    def create_visualization(self, question: str, triples: List[Dict[str, Any]], 
                           output_path: str = None) -> str:
        if not triples:
            logger.warning("No triples provided for visualization")
            return None
        
        top_visual = getattr(self.config.retriever, 'top_visual', 10)
        
        visualization_triples = triples[:top_visual]
        
        G = self._create_network_graph(visualization_triples)
        
        fig = self._create_figure(question, G, visualization_triples)
        
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"query_visualization_{len(visualization_triples)}_nodes.png")
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        logger.info(f"Visualization saved to: {output_path}")
        return output_path
    
    def _create_network_graph(self, triples: List[Dict[str, Any]]) -> nx.DiGraph:
        G = nx.DiGraph()
        
        for i, triple in enumerate(triples):
            if not isinstance(triple, dict):
                logger.error(f"Triple {i} is not a dictionary: {type(triple)} - {triple}")
                continue
                
            src = triple.get('src', f'Node_{i}_src')
            tgt = triple.get('tgt', f'Node_{i}_tgt')
            relation = triple.get('relation', 'relates_to')
            score = triple.get('final_score', 0.0)
            
            if hasattr(score, 'item'):
                score = score.item()
            
            if not G.has_node(src):
                G.add_node(src, node_type='entity', score=score)
            if not G.has_node(tgt):
                G.add_node(tgt, node_type='entity', score=score)
            
            G.add_edge(src, tgt, relation=relation, score=score)
        
        return G
    
    def _create_figure(self, question: str, G: nx.DiGraph, triples: List[Dict[str, Any]]) -> plt.Figure:
        fig = plt.figure(figsize=(20, 14))
        
        gs = fig.add_gridspec(3, 2, height_ratios=[0.5, 2, 1], width_ratios=[3, 1],
                             hspace=0.3, wspace=0.2)
        
        # Question text (top, spanning both columns)
        ax_question = fig.add_subplot(gs[0, :])
        self._add_question_text(ax_question, question)
        
        # Main graph (middle left)
        ax_graph = fig.add_subplot(gs[1, 0])
        self._draw_graph(ax_graph, G, triples)
        
        # Legend (middle right)
        ax_legend = fig.add_subplot(gs[1, 1])
        self._add_legend(ax_legend, G)
        
        # Table (bottom, spanning both columns)
        ax_table = fig.add_subplot(gs[2, :])
        self._add_triples_table(ax_table, triples)
        
        return fig
    
    def _add_question_text(self, ax, question: str):
        """Add question text to the top of the figure."""
        ax.axis('off')
        
        # Create a fancy box for the question
        box = FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                           boxstyle="round,pad=0.02",
                           facecolor='#f0f8ff',
                           edgecolor='#4169e1',
                           linewidth=2)
        ax.add_patch(box)
        
        # Add question text
        ax.text(0.5, 0.5, f"Question: {question}",
               ha='center', va='center', fontsize=14, fontweight='bold',
               color='#2c3e50', wrap=True)
    
    def _draw_graph(self, ax, G: nx.DiGraph, triples: List[Dict[str, Any]]):
        """Draw the main network graph."""
        if len(G.nodes()) == 0:
            ax.text(0.5, 0.5, "No nodes to display", ha='center', va='center')
            ax.axis('off')
            return
        
        # Calculate layout
        pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
        
        # Get node scores for coloring
        node_scores = [G.nodes[node].get('score', 0.0) for node in G.nodes()]
        if node_scores:
            min_score, max_score = min(node_scores), max(node_scores)
            if max_score > min_score:
                normalized_scores = [(score - min_score) / (max_score - min_score) 
                                   for score in node_scores]
            else:
                normalized_scores = [0.5] * len(node_scores)
        else:
            normalized_scores = [0.5] * len(G.nodes())
        
        # Draw nodes with custom colors (light blue, light green, light pink)
        node_colors = []
        for score in normalized_scores:
            if score < 0.33:
                node_colors.append('#ADD8E6')  # Light Blue
            elif score < 0.66:
                node_colors.append('#90EE90')  # Light Green
            else:
                node_colors.append('#FFB6C1')  # Light Pink
        
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                              node_size=1500, alpha=0.8, edgecolors='black', linewidths=2)
        
        # Draw edges with relation labels
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#666666',
                              arrows=True, arrowsize=20, arrowstyle='->',
                              connectionstyle="arc3,rad=0.1", alpha=0.7)
        
        # Add edge labels (relations)
        edge_labels = nx.get_edge_attributes(G, 'relation')
        nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=8,
                                   font_color='#2c3e50', font_weight='bold')
        
        # Add node labels
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold',
                              font_color='black')
        
        ax.set_title("Knowledge Graph Visualization", fontsize=16, fontweight='bold', 
                    color='#2c3e50', pad=20)
        ax.axis('off')
    
    def _add_legend(self, ax, G: nx.DiGraph):
        """Add legend for the graph."""
        ax.axis('off')
        
        # Create legend elements
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ADD8E6', 
                      markersize=10, label='Low Relevance'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#90EE90', 
                      markersize=10, label='Medium Relevance'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFB6C1', 
                      markersize=10, label='High Relevance'),
            plt.Line2D([0], [0], marker='>', color='#666666', markersize=8, 
                      label='Relationship')
        ]
        
        ax.legend(handles=legend_elements, loc='center', fontsize=10,
                 title='Legend', title_fontsize=12, frameon=True,
                 fancybox=True, shadow=True)
        
        ax.set_title("Legend", fontsize=14, fontweight='bold', color='#2c3e50')
    
    def _add_triples_table(self, ax, triples: List[Dict[str, Any]]):
        """Add triples table at the bottom."""
        if not triples:
            ax.text(0.5, 0.5, "No triples to display", ha='center', va='center')
            ax.axis('off')
            return
        
        # Limit table to top 5 triples for better readability
        table_triples = triples[:5]
        
        # Prepare data for table
        table_data = []
        for i, triple in enumerate(table_triples, 1):
            # Ensure triple is a dictionary
            if not isinstance(triple, dict):
                logger.error(f"Triple {i} is not a dictionary: {type(triple)} - {triple}")
                continue
                
            src = triple.get('src', 'N/A')
            relation = triple.get('relation', 'N/A')
            tgt = triple.get('tgt', 'N/A')
            score = triple.get('final_score', 0.0)
            
            # Convert numpy types to Python types for display
            if hasattr(score, 'item'):
                score = score.item()
            
            table_data.append([
                f"{i}",
                str(src),
                str(relation),
                str(tgt),
                f"{float(score):.4f}"
            ])
        
        # Create table
        headers = ['Rank', 'Source', 'Relation', 'Target', 'Relevance Score']
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center',
                        bbox=[0, 0, 1, 1])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Color header row
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color data rows alternately
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')
                else:
                    table[(i, j)].set_facecolor('white')
        
        ax.set_title("Triples Ranking by Relevance", fontsize=14, fontweight='bold',
                    color='#2c3e50', pad=20)
        ax.axis('off')

