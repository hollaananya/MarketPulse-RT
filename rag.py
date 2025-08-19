# rag.py - RAG implementation with ChromaDB
import os
import uuid
import logging
from typing import List, Dict, Optional, Any

# Suppress ChromaDB telemetry before importing
os.environ.setdefault("CHROMADB_TELEMETRY_ENABLED", "false")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

# Simple TF-IDF fallback if chromadb is not available
from collections import defaultdict, Counter
import math
import re

class SimpleTFIDFRetriever:
    """Fallback retriever using TF-IDF when ChromaDB is not available"""
    
    def __init__(self):
        self.documents = []
        self.vocab = set()
        self.doc_freq = defaultdict(int)
        self.tf_idf = []
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return re.findall(r'\b\w+\b', text.lower())
    
    def _compute_tf(self, doc_tokens: List[str]) -> Dict[str, float]:
        """Compute term frequency"""
        doc_len = len(doc_tokens)
        tf = defaultdict(float)
        for token in doc_tokens:
            tf[token] += 1.0 / doc_len
        return tf
    
    def _compute_idf(self) -> Dict[str, float]:
        """Compute inverse document frequency"""
        num_docs = len(self.documents)
        idf = {}
        for term in self.vocab:
            idf[term] = math.log(num_docs / self.doc_freq[term])
        return idf
    
    def build_index(self, documents: List[Dict]):
        """Build TF-IDF index"""
        self.documents = documents
        self.vocab = set()
        self.doc_freq = defaultdict(int)
        self.tf_idf = []
        
        # Build vocabulary and document frequency
        for doc in documents:
            text = f"{doc.get('title', '')} {doc.get('summary', '')}"
            tokens = self._tokenize(text)
            unique_tokens = set(tokens)
            
            for token in unique_tokens:
                self.vocab.add(token)
                self.doc_freq[token] += 1
        
        # Compute TF-IDF for each document
        idf = self._compute_idf()
        for doc in documents:
            text = f"{doc.get('title', '')} {doc.get('summary', '')}"
            tokens = self._tokenize(text)
            tf = self._compute_tf(tokens)
            
            doc_tfidf = {}
            for token in tf:
                doc_tfidf[token] = tf[token] * idf.get(token, 0)
            
            self.tf_idf.append(doc_tfidf)
    
    def query(self, query_text: str, k: int = 10) -> List[Dict]:
        """Query using cosine similarity"""
        if not self.documents:
            return []
        
        query_tokens = self._tokenize(query_text)
        query_tf = self._compute_tf(query_tokens)
        
        # Compute query TF-IDF
        idf = self._compute_idf()
        query_tfidf = {}
        for token in query_tf:
            query_tfidf[token] = query_tf[token] * idf.get(token, 0)
        
        # Compute similarities
        similarities = []
        for i, doc_tfidf in enumerate(self.tf_idf):
            # Cosine similarity
            dot_product = sum(query_tfidf.get(term, 0) * doc_tfidf.get(term, 0) 
                            for term in set(query_tfidf.keys()) | set(doc_tfidf.keys()))
            
            query_norm = math.sqrt(sum(v*v for v in query_tfidf.values()))
            doc_norm = math.sqrt(sum(v*v for v in doc_tfidf.values()))
            
            if query_norm > 0 and doc_norm > 0:
                similarity = dot_product / (query_norm * doc_norm)
            else:
                similarity = 0
            
            similarities.append((similarity, i))
        
        # Sort by similarity and return top k
        similarities.sort(reverse=True)
        results = []
        for similarity, idx in similarities[:k]:
            doc = self.documents[idx].copy()
            doc['similarity'] = similarity
            results.append(doc)
        
        return results

class LiveRAG:
    """RAG system for live market data with ChromaDB or TF-IDF fallback"""
    
    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir
        self.collection_name = "market_news"
        
        if CHROMADB_AVAILABLE:
            try:
                # Configure ChromaDB with telemetry disabled
                settings = Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
                
                if persist_dir:
                    os.makedirs(persist_dir, exist_ok=True)
                    self.client = chromadb.PersistentClient(path=persist_dir, settings=settings)
                else:
                    self.client = chromadb.Client(settings=settings)
                
                # Create or get collection
                try:
                    self.collection = self.client.get_collection(self.collection_name)
                except:
                    self.collection = self.client.create_collection(
                        name=self.collection_name,
                        metadata={"hnsw:space": "cosine"}
                    )
                
                self.use_chromadb = True
                self.fallback = None
                
            except Exception as e:
                print(f"ChromaDB initialization failed: {e}")
                self.use_chromadb = False
                self.fallback = SimpleTFIDFRetriever()
        else:
            print("ChromaDB not available, using TF-IDF fallback")
            self.use_chromadb = False
            self.fallback = SimpleTFIDFRetriever()
    
    def _doc_to_text(self, doc: Dict) -> str:
        """Convert document to searchable text"""
        parts = []
        if doc.get('title'):
            parts.append(doc['title'])
        if doc.get('summary'):
            parts.append(doc['summary'])
        
        # Add structured fields
        if doc.get('sectors'):
            sectors = doc['sectors'] if isinstance(doc['sectors'], list) else [doc['sectors']]
            parts.extend(sectors)
        
        if doc.get('events'):
            events = doc['events'] if isinstance(doc['events'], list) else [doc['events']]
            parts.extend(events)
        
        if doc.get('region'):
            parts.append(doc['region'])
        
        return ' '.join(parts)
    
    def rebuild(self, documents: List[Dict]):
        """Rebuild the entire index with new documents"""
        if not documents:
            return
        
        if self.use_chromadb:
            try:
                # Clear existing collection
                self.client.delete_collection(self.collection_name)
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                
                # Add documents
                texts = []
                metadatas = []
                ids = []
                
                for i, doc in enumerate(documents):
                    text = self._doc_to_text(doc)
                    if not text.strip():
                        continue
                    
                    texts.append(text)
                    
                    # Prepare metadata (ChromaDB doesn't like nested structures)
                    metadata = {
                        'title': doc.get('title', '')[:500],  # Limit length
                        'source': doc.get('source', '')[:200],
                        'region': doc.get('region', ''),
                        'published': doc.get('published', '')[:50],
                        'sentiment': float(doc.get('sentiment', 0.0)),
                    }
                    
                    # Handle lists by joining them
                    if doc.get('sectors'):
                        if isinstance(doc['sectors'], list):
                            metadata['sectors'] = ','.join(doc['sectors'][:3])  # Limit sectors
                        else:
                            metadata['sectors'] = str(doc['sectors'])[:100]
                    
                    if doc.get('events'):
                        if isinstance(doc['events'], list):
                            metadata['events'] = ','.join(doc['events'][:3])
                        else:
                            metadata['events'] = str(doc['events'])[:100]
                    
                    metadatas.append(metadata)
                    ids.append(str(uuid.uuid4()))
                
                if texts:
                    # Add in batches to avoid memory issues
                    batch_size = 100
                    for i in range(0, len(texts), batch_size):
                        batch_texts = texts[i:i+batch_size]
                        batch_metadata = metadatas[i:i+batch_size]
                        batch_ids = ids[i:i+batch_size]
                        
                        self.collection.add(
                            documents=batch_texts,
                            metadatas=batch_metadata,
                            ids=batch_ids
                        )
                
            except Exception as e:
                print(f"ChromaDB rebuild failed: {e}")
                # Fall back to TF-IDF
                self.use_chromadb = False
                if not self.fallback:
                    self.fallback = SimpleTFIDFRetriever()
                self.fallback.build_index(documents)
        else:
            if not self.fallback:
                self.fallback = SimpleTFIDFRetriever()
            self.fallback.build_index(documents)
    
    def query(self, query_text: str, k: int = 10, region: Optional[str] = None, 
              sectors: Optional[List[str]] = None, events: Optional[List[str]] = None) -> List[Dict]:
        """Query the RAG system with optional filters"""
        
        if self.use_chromadb:
            try:
                # Build where clause for filtering
                where_clause = {}
                if region and region != "Both":
                    where_clause["region"] = region
                
                # Query ChromaDB
                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=min(k, 100),  # ChromaDB has limits
                    where=where_clause if where_clause else None
                )
                
                # Convert to standard format
                formatted_results = []
                if results['documents'] and results['documents'][0]:
                    for i, doc_text in enumerate(results['documents'][0]):
                        metadata = results['metadatas'][0][i]
                        
                        # Reconstruct document
                        doc = {
                            'title': metadata.get('title', ''),
                            'source': metadata.get('source', ''),
                            'region': metadata.get('region', ''),
                            'published': metadata.get('published', ''),
                            'sentiment': metadata.get('sentiment', 0.0),
                            'similarity': 1.0 - results['distances'][0][i],  # Convert distance to similarity
                        }
                        
                        # Handle sectors and events
                        if metadata.get('sectors'):
                            doc['sectors'] = metadata['sectors'].split(',')
                        if metadata.get('events'):
                            doc['events'] = metadata['events'].split(',')
                        
                        formatted_results.append(doc)
                
                return formatted_results
                
            except Exception as e:
                print(f"ChromaDB query failed: {e}")
                # Fall back to TF-IDF
                if not self.fallback:
                    return []
                return self.fallback.query(query_text, k)
        else:
            if not self.fallback:
                return []
            results = self.fallback.query(query_text, k)
            
            # Apply manual filtering for fallback
            if region and region != "Both":
                results = [r for r in results if r.get('region') == region]
            
            if sectors:
                filtered = []
                for r in results:
                    doc_sectors = r.get('sectors', [])
                    if isinstance(doc_sectors, str):
                        doc_sectors = [doc_sectors]
                    if any(s in doc_sectors for s in sectors):
                        filtered.append(r)
                results = filtered
            
            return results[:k]

def format_context(context_items: List[Dict]) -> str:
    """Format retrieved context for the LLM"""
    if not context_items:
        return "No relevant market data found."
    
    formatted = []
    for i, item in enumerate(context_items, 1):
        title = item.get('title', 'Untitled')
        source = item.get('source', 'Unknown')
        published = item.get('published', 'No date')
        sentiment = item.get('sentiment', 0.0)
        similarity = item.get('similarity', 0.0)
        
        # Add summary if available
        summary = item.get('summary', '')
        if len(summary) > 200:
            summary = summary[:197] + "..."
        
        region = item.get('region', 'Unknown')
        sectors = item.get('sectors', [])
        if isinstance(sectors, list):
            sectors_str = ', '.join(sectors[:3])
        else:
            sectors_str = str(sectors)
        
        entry = f"""
[{i}] {title}
Source: {source} | Published: {published} | Region: {region}
Sectors: {sectors_str} | Sentiment: {sentiment:+.2f}
{summary}
"""
        formatted.append(entry.strip())
    
    return "\n\n".join(formatted)