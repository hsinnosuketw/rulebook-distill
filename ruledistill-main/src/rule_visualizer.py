"""
Rule Visualization & Analysis Tool

Generates comprehensive visualizations for rulebook analysis:
1. Word Cloud - Visual overview of rule keywords
2. Bar Charts - Frequency analysis of rule types and triggers
3. Network Graph - Relationships between rules based on shared concepts
4. Evolution Trend - How rules evolve across batches

Usage:
    python rule_visualizer.py --checkpoint-dir /path/to/checkpoints
    python rule_visualizer.py --rulebook /path/to/rulebook.xml
"""

import os
import re
import argparse
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional
import json

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from wordcloud import WordCloud
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap


def parse_rulebook_file(filepath: str) -> list[dict]:
    """Parse a rulebook XML file and extract rules."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Handle potential XML issues
        if not content.strip():
            return []
        
        root = ET.fromstring(content)
        rules = []
        
        for rule in root.findall('.//Rule'):
            rule_data = {
                'id': rule.get('id', ''),
                'type': rule.get('type', ''),
                'phase': rule.get('phase', ''),
                'confidence': rule.get('confidence', ''),
                'source': rule.get('source', ''),
                'trigger': '',
                'action': ''
            }
            
            trigger = rule.find('Trigger')
            if trigger is not None and trigger.text:
                rule_data['trigger'] = trigger.text.strip()
            
            action = rule.find('Action')
            if action is not None and action.text:
                rule_data['action'] = action.text.strip()
            
            rules.append(rule_data)
        
        return rules
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return []


def extract_keywords(text: str, stopwords: set = None) -> list[str]:
    """Extract meaningful keywords from text."""
    if stopwords is None:
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'if',
            'then', 'else', 'when', 'where', 'what', 'which', 'who', 'how',
            'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them',
            'their', 'using', 'use', 'used'
        }
    
    # Tokenize and clean
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return [w for w in words if w not in stopwords]


def generate_word_cloud(rules: list[dict], output_path: str, title: str = "Rule Keywords"):
    """Generate a word cloud from rule triggers and actions."""
    # Combine all text
    all_text = []
    for rule in rules:
        all_text.extend(extract_keywords(rule['trigger']))
        all_text.extend(extract_keywords(rule['action']))
    
    if not all_text:
        print("No keywords found for word cloud")
        return
    
    word_freq = Counter(all_text)
    
    # Create word cloud
    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color='white',
        colormap='viridis',
        max_words=100,
        relative_scaling=0.5,
        min_font_size=10
    ).generate_from_frequencies(word_freq)
    
    plt.figure(figsize=(14, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Word cloud saved to: {output_path}")


def generate_type_bar_chart(rules: list[dict], output_path: str, title: str = "Rule Types Distribution"):
    """Generate a bar chart showing rule type distribution."""
    type_counts = Counter(rule['type'] for rule in rules if rule['type'])
    
    if not type_counts:
        print("No rule types found")
        return
    
    # Sort by frequency
    sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
    types, counts = zip(*sorted_types)
    
    # Create color gradient
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(types)))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(types)), counts, color=colors)
    
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels(types, fontsize=10)
    ax.set_xlabel('Number of Rules', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add value labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                str(count), va='center', fontsize=10)
    
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Type bar chart saved to: {output_path}")


def generate_keyword_bar_chart(rules: list[dict], output_path: str, top_n: int = 20, 
                                title: str = "Top Keywords in Rules"):
    """Generate a bar chart showing top keywords."""
    all_keywords = []
    for rule in rules:
        all_keywords.extend(extract_keywords(rule['trigger']))
        all_keywords.extend(extract_keywords(rule['action']))
    
    if not all_keywords:
        print("No keywords found")
        return
    
    keyword_counts = Counter(all_keywords).most_common(top_n)
    keywords, counts = zip(*keyword_counts)
    
    # Create gradient colors
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(keywords)))
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(keywords)), counts, color=colors)
    
    ax.set_yticks(range(len(keywords)))
    ax.set_yticklabels(keywords, fontsize=11)
    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add value labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                str(count), va='center', fontsize=10)
    
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Keyword bar chart saved to: {output_path}")


def generate_network_graph(rules: list[dict], output_path: str, 
                           title: str = "Rule Relationship Network"):
    """Generate a network graph showing relationships between rules."""
    G = nx.Graph()
    
    # Create nodes for each rule
    for rule in rules:
        node_label = f"R{rule['id']}"
        G.add_node(node_label, 
                   type=rule['type'],
                   trigger=rule['trigger'][:30] + '...' if len(rule['trigger']) > 30 else rule['trigger'])
    
    # Create edges based on shared keywords
    rule_keywords = {}
    for rule in rules:
        keywords = set(extract_keywords(rule['trigger'] + ' ' + rule['action']))
        rule_keywords[f"R{rule['id']}"] = keywords
    
    # Connect rules that share keywords
    rule_ids = list(rule_keywords.keys())
    for i in range(len(rule_ids)):
        for j in range(i + 1, len(rule_ids)):
            shared = rule_keywords[rule_ids[i]] & rule_keywords[rule_ids[j]]
            if len(shared) >= 2:  # At least 2 shared keywords
                G.add_edge(rule_ids[i], rule_ids[j], weight=len(shared))
    
    if len(G.nodes()) == 0:
        print("No rules to visualize in network")
        return
    
    # Layout and drawing
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Use spring layout for natural clustering
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Color nodes by rule type
    type_colors = {}
    unique_types = list(set(nx.get_node_attributes(G, 'type').values()))
    color_map = plt.cm.Set3(np.linspace(0, 1, len(unique_types)))
    for t, c in zip(unique_types, color_map):
        type_colors[t] = c
    
    node_colors = [type_colors.get(G.nodes[n].get('type', ''), 'gray') for n in G.nodes()]
    
    # Draw edges with varying thickness
    edge_weights = [G[u][v].get('weight', 1) for u, v in G.edges()]
    if edge_weights:
        max_weight = max(edge_weights)
        edge_widths = [w / max_weight * 3 + 0.5 for w in edge_weights]
    else:
        edge_widths = [1]
    
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color='gray')
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
    
    # Add legend
    patches = [mpatches.Patch(color=type_colors[t], label=t) for t in unique_types]
    ax.legend(handles=patches, loc='upper left', fontsize=9)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Network graph saved to: {output_path}")


def generate_treemap(rules: list[dict], output_path: str, 
                     title: str = "Rule Categories Treemap"):
    """Generate a treemap showing rule types by proportion."""
    try:
        import squarify
    except ImportError:
        print("squarify not installed. Run: pip install squarify")
        return
    
    type_counts = Counter(rule['type'] for rule in rules if rule['type'])
    
    if not type_counts:
        print("No rule types found for treemap")
        return
    
    labels = list(type_counts.keys())
    sizes = list(type_counts.values())
    
    # Format labels with counts
    labels_with_counts = [f"{l}\n({c})" for l, c in zip(labels, sizes)]
    
    # Colors
    colors = plt.cm.Spectral(np.linspace(0.1, 0.9, len(labels)))
    
    fig, ax = plt.subplots(figsize=(14, 8))
    squarify.plot(sizes=sizes, label=labels_with_counts, color=colors, alpha=0.8, ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Treemap saved to: {output_path}")


def generate_evolution_chart(checkpoint_dir: str, output_path: str,
                             title: str = "Rulebook Evolution Over Batches"):
    """Generate a chart showing how rules evolve over batches."""
    # Find all rulebook files
    rulebook_files = sorted(Path(checkpoint_dir).glob("rulebook_batch_*.xml"))
    
    if not rulebook_files:
        print("No rulebook checkpoints found")
        return
    
    batch_nums = []
    rule_counts = []
    type_evolution = defaultdict(list)
    
    for rb_file in rulebook_files:
        # Extract batch number
        match = re.search(r'batch_(\d+)', rb_file.name)
        if match:
            batch_num = int(match.group(1))
            batch_nums.append(batch_num)
            
            rules = parse_rulebook_file(str(rb_file))
            rule_counts.append(len(rules))
            
            # Track type distribution
            type_counts = Counter(rule['type'] for rule in rules if rule['type'])
            for t in type_counts:
                type_evolution[t].append((batch_num, type_counts[t]))
    
    if not batch_nums:
        print("No valid batch data found")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Total rule count over time
    ax1.plot(batch_nums, rule_counts, 'b-o', linewidth=2, markersize=8)
    ax1.fill_between(batch_nums, rule_counts, alpha=0.3)
    ax1.set_xlabel('Batch Number', fontsize=12)
    ax1.set_ylabel('Total Rules', fontsize=12)
    ax1.set_title('Rule Count Evolution', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Stacked area chart of rule types
    all_batches = sorted(set(batch_nums))
    types = list(type_evolution.keys())
    
    # Build matrix for stacking
    type_matrix = []
    for t in types:
        batch_dict = dict(type_evolution[t])
        type_matrix.append([batch_dict.get(b, 0) for b in all_batches])
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(types)))
    ax2.stackplot(all_batches, type_matrix, labels=types, colors=colors, alpha=0.8)
    ax2.set_xlabel('Batch Number', fontsize=12)
    ax2.set_ylabel('Rules by Type', fontsize=12)
    ax2.set_title('Rule Type Distribution Over Time', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Evolution chart saved to: {output_path}")


def generate_metrics_trend(checkpoint_dir: str, output_path: str,
                           title: str = "Pipeline Metrics Over Time"):
    """Generate charts showing accuracy and other metrics over batches."""
    metrics_file = os.path.join(checkpoint_dir, "metrics.jsonl")
    
    if not os.path.exists(metrics_file):
        print(f"Metrics file not found: {metrics_file}")
        return
    
    metrics_list = []
    with open(metrics_file, 'r') as f:
        for line in f:
            if line.strip():
                metrics_list.append(json.loads(line))
    
    if not metrics_list:
        print("No metrics data found")
        return
    
    # Extract data
    batches = [m.get('batch_num', i) for i, m in enumerate(metrics_list)]
    accuracies = [m.get('accuracy', 0) for m in metrics_list]
    correct_counts = [m.get('correct_count', 0) for m in metrics_list]
    error_counts = [m.get('error_count', 0) for m in metrics_list]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Accuracy trend
    ax1 = axes[0, 0]
    ax1.plot(batches, accuracies, 'g-o', linewidth=2, markersize=6)
    ax1.fill_between(batches, accuracies, alpha=0.3, color='green')
    ax1.set_xlabel('Batch', fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_title('Accuracy Trend', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Correct vs Errors stacked
    ax2 = axes[0, 1]
    ax2.bar(batches, correct_counts, label='Correct', color='green', alpha=0.7)
    ax2.bar(batches, error_counts, bottom=correct_counts, label='Errors', color='red', alpha=0.7)
    ax2.set_xlabel('Batch', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Correct vs Errors per Batch', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Rolling average accuracy
    ax3 = axes[1, 0]
    window = min(3, len(accuracies))
    rolling_avg = np.convolve(accuracies, np.ones(window)/window, mode='valid')
    rolling_batches = batches[window-1:]
    ax3.plot(batches, accuracies, 'b-', alpha=0.4, label='Raw')
    ax3.plot(rolling_batches, rolling_avg, 'b-', linewidth=2, label=f'{window}-batch MA')
    ax3.set_xlabel('Batch', fontsize=11)
    ax3.set_ylabel('Accuracy', fontsize=11)
    ax3.set_title('Accuracy with Moving Average', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Cumulative performance
    ax4 = axes[1, 1]
    cumulative_correct = np.cumsum(correct_counts)
    cumulative_total = np.cumsum([c + e for c, e in zip(correct_counts, error_counts)])
    cumulative_accuracy = cumulative_correct / cumulative_total
    ax4.plot(batches, cumulative_accuracy, 'purple', linewidth=2)
    ax4.fill_between(batches, cumulative_accuracy, alpha=0.3, color='purple')
    ax4.set_xlabel('Batch', fontsize=11)
    ax4.set_ylabel('Cumulative Accuracy', fontsize=11)
    ax4.set_title('Cumulative Accuracy Over Time', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Metrics trend chart saved to: {output_path}")


def generate_rule_summary_table(rules: list[dict], output_path: str):
    """Generate a summary table of all rules as HTML."""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Rulebook Summary</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
        .stats { background: #fff; padding: 15px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .stat-item { display: inline-block; margin-right: 30px; }
        .stat-value { font-size: 24px; font-weight: bold; color: #4CAF50; }
        .stat-label { color: #666; font-size: 12px; }
        table { border-collapse: collapse; width: 100%; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        th { background: #4CAF50; color: white; padding: 12px; text-align: left; }
        td { padding: 10px; border-bottom: 1px solid #eee; }
        tr:hover { background: #f9f9f9; }
        .type-badge { display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; background: #e3f2fd; color: #1976d2; }
        .source-badge { display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 10px; background: #fff3e0; color: #f57c00; }
    </style>
</head>
<body>
    <h1>📚 Rulebook Summary</h1>
    <div class="stats">
        <div class="stat-item">
            <div class="stat-value">TOTAL_RULES</div>
            <div class="stat-label">Total Rules</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">UNIQUE_TYPES</div>
            <div class="stat-label">Unique Types</div>
        </div>
    </div>
    <table>
        <thead>
            <tr>
                <th>ID</th>
                <th>Type</th>
                <th>Trigger</th>
                <th>Action</th>
                <th>Source</th>
            </tr>
        </thead>
        <tbody>
TABLE_ROWS
        </tbody>
    </table>
</body>
</html>
"""
    
    rows = []
    for rule in rules:
        row = f"""            <tr>
                <td><strong>R{rule['id']}</strong></td>
                <td><span class="type-badge">{rule['type']}</span></td>
                <td>{rule['trigger']}</td>
                <td>{rule['action']}</td>
                <td><span class="source-badge">{rule['source']}</span></td>
            </tr>"""
        rows.append(row)
    
    unique_types = len(set(rule['type'] for rule in rules if rule['type']))
    
    html_content = html_content.replace('TOTAL_RULES', str(len(rules)))
    html_content = html_content.replace('UNIQUE_TYPES', str(unique_types))
    html_content = html_content.replace('TABLE_ROWS', '\n'.join(rows))
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Summary table saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Rule Visualization & Analysis Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--rulebook",
        type=str,
        default=None,
        help="Path to a single rulebook XML file"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="/root/hsin_research/ruledistill-main/data/checkpoints",
        help="Directory containing checkpoint files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/root/hsin_research/ruledistill-main/data/visualizations",
        help="Directory to save visualizations"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all visualizations"
    )
    parser.add_argument(
        "--wordcloud",
        action="store_true",
        help="Generate word cloud"
    )
    parser.add_argument(
        "--barchart",
        action="store_true",
        help="Generate bar charts"
    )
    parser.add_argument(
        "--network",
        action="store_true",
        help="Generate network graph"
    )
    parser.add_argument(
        "--treemap",
        action="store_true",
        help="Generate treemap"
    )
    parser.add_argument(
        "--evolution",
        action="store_true",
        help="Generate evolution charts"
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Generate metrics trend charts"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Generate HTML summary table"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # If no specific options, generate all
    if not any([args.wordcloud, args.barchart, args.network, args.treemap, 
                args.evolution, args.metrics, args.summary]):
        args.all = True
    
    # Load rules
    if args.rulebook:
        rules = parse_rulebook_file(args.rulebook)
        source_name = Path(args.rulebook).stem
    else:
        # Load latest rulebook from checkpoints
        rulebook_files = sorted(Path(args.checkpoint_dir).glob("rulebook_batch_*.xml"))
        if rulebook_files:
            latest_rulebook = rulebook_files[-1]
            rules = parse_rulebook_file(str(latest_rulebook))
            source_name = latest_rulebook.stem
            print(f"Using latest rulebook: {latest_rulebook}")
        else:
            print("No rulebook files found!")
            return
    
    print(f"Loaded {len(rules)} rules from {source_name}")
    
    # Generate visualizations
    if args.all or args.wordcloud:
        generate_word_cloud(
            rules, 
            os.path.join(args.output_dir, "wordcloud.png"),
            f"Rule Keywords - {source_name}"
        )
    
    if args.all or args.barchart:
        generate_type_bar_chart(
            rules,
            os.path.join(args.output_dir, "type_distribution.png"),
            f"Rule Type Distribution - {source_name}"
        )
        generate_keyword_bar_chart(
            rules,
            os.path.join(args.output_dir, "keyword_frequency.png"),
            title=f"Top Keywords - {source_name}"
        )
    
    if args.all or args.network:
        generate_network_graph(
            rules,
            os.path.join(args.output_dir, "rule_network.png"),
            f"Rule Relationships - {source_name}"
        )
    
    if args.all or args.treemap:
        generate_treemap(
            rules,
            os.path.join(args.output_dir, "rule_treemap.png"),
            f"Rule Categories - {source_name}"
        )
    
    if args.all or args.evolution:
        generate_evolution_chart(
            args.checkpoint_dir,
            os.path.join(args.output_dir, "rule_evolution.png")
        )
    
    if args.all or args.metrics:
        generate_metrics_trend(
            args.checkpoint_dir,
            os.path.join(args.output_dir, "metrics_trend.png")
        )
    
    if args.all or args.summary:
        generate_rule_summary_table(
            rules,
            os.path.join(args.output_dir, "rule_summary.html")
        )
    
    print(f"\n✅ All visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
