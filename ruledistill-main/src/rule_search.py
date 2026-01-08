"""
Rule Search Module using BM25.

This module implements a search engine for rulebooks using the BM25 algorithm.
It loads rules from an XML file, indexes them, and allows for querying
to find the most relevant rules based on trigger and action text.

Classes:
    RuleSearch: Handles loading, indexing, and searching of rules.

Usage:
    Run the script directly to search for rules via CLI:
    $ python rule_search.py "query string" --top_k 5
"""

import argparse
import re

from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi

class RuleSearch:
    def __init__(self, xml_path):
        self.rules = []
        self.corpus = []
        self.bm25 = None
        self.load_rules(xml_path)
        self.initialize_bm25()

    def load_rules(self, xml_path):
        """Loads rules from the XML file using BeautifulSoup for robustness."""
        try:
            with open(xml_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Use 'xml' parser if lxml is installed, otherwise 'html.parser' which is also forgiving
            soup = BeautifulSoup(content, 'xml') 
            
            # Find all rules
            rules = soup.find_all('Rule')
            
            for rule in rules:
                trigger = rule.find('Trigger')
                action = rule.find('Action')
                
                rule_data = {
                    'id': rule.get('id'),
                    'phase': rule.get('phase'),
                    'confidence': rule.get('confidence'),
                    'source': rule.get('source'),
                    'type': rule.get('type'),
                    'trigger': trigger.get_text(strip=True) if trigger else "",
                    'action': action.get_text(strip=True) if action else ""
                }
                self.rules.append(rule_data)
                
                # Combine distinct text parts for indexing
                text_content = f"{rule_data['trigger']} {rule_data['action']}"
                self.corpus.append(text_content)
            
            print(f"Loaded {len(self.rules)} rules from {xml_path}")

        except Exception as e:
            print(f"Error loading rules: {e}")

    def tokenize(self, text):
        """Simple whitespace tokenizer. Can be improved with NLP libs if needed."""
        # Remove non-alphanumeric characters and lowercase
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()
        return text.split()

    def initialize_bm25(self):
        """Initializes the BM25 model."""
        if not self.corpus:
            print("Warning: Corpus is empty. BM25 not initialized.")
            return

        tokenized_corpus = [self.tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query, top_k=5):
        """Searches for rules matching the query."""
        if not self.bm25:
            print("BM25 not initialized.")
            return []

        tokenized_query = self.tokenize(query)
        # Get scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top_k indices
        top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for i in top_n:
            if scores[i] > 0: # Only return results with some relevance
                results.append((self.rules[i], scores[i]))
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Search rules using BM25")
    parser.add_argument('query', type=str, help="Search query")
    parser.add_argument('--xml_path', type=str, default='../data/rulebook/all_rules.xml', help="Path to aggregated XML rules")
    parser.add_argument('--top_k', type=int, default=5, help="Number of results to return")
    
    args = parser.parse_args()
    
    searcher = RuleSearch(args.xml_path)
    results = searcher.search(args.query, args.top_k)
    
    print(f"Query: {args.query}\n")
    if not results:
        print("No matching rules found.")
    else:
        for i, (rule, score) in enumerate(results, 1):
            print(f"Rank {i} (Score: {score:.4f}):")
            print(f"  ID: {rule['id']}")
            print(f"  Type: {rule['type']}")
            
            trig = rule['trigger']
            act = rule['action']
            print(f"  Trigger: {trig[:200]}..." if len(trig) > 200 else f"  Trigger: {trig}")
            print(f"  Action: {act[:200]}..." if len(act) > 200 else f"  Action: {act}")
            print("-" * 40)

if __name__ == "__main__":
    main()
