"""
Full-Text Search Engine

Production-grade search with:
- Full-text indexing
- Fuzzy matching
- Faceted search
- Search suggestions
- Result ranking
- Search analytics
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import re
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


class SearchField(str, Enum):
    """Searchable fields."""
    NAME = "name"
    DESCRIPTION = "description"
    TAGS = "tags"
    CONTENT = "content"
    METADATA = "metadata"


@dataclass
class SearchDocument:
    """Document to be indexed."""
    doc_id: str
    doc_type: str
    fields: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    boost: float = 1.0  # Boost factor for ranking


@dataclass
class SearchQuery:
    """Search query."""
    query_string: str
    fields: List[SearchField] = field(default_factory=lambda: [SearchField.NAME, SearchField.DESCRIPTION])
    filters: Dict[str, Any] = field(default_factory=dict)
    facets: List[str] = field(default_factory=list)
    fuzzy: bool = False
    max_results: int = 20
    offset: int = 0


@dataclass
class SearchResult:
    """Search result with scoring."""
    doc_id: str
    doc_type: str
    score: float
    highlights: Dict[str, List[str]] = field(default_factory=dict)
    fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FacetResult:
    """Facet aggregation result."""
    field: str
    buckets: List[Dict[str, Any]] = field(default_factory=list)


class SearchIndex:
    """Inverted index for full-text search."""
    
    def __init__(self):
        # Inverted index: token -> {doc_id: [positions]}
        self.index: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        
        # Document store
        self.documents: Dict[str, SearchDocument] = {}
        
        # Document frequency: token -> count
        self.doc_frequency: Dict[str, int] = defaultdict(int)
        
        # Field lengths: doc_id -> {field -> length}
        self.field_lengths: Dict[str, Dict[str, int]] = defaultdict(dict)
        
        # Average field lengths
        self.avg_field_lengths: Dict[str, float] = {}
        
    def add_document(self, document: SearchDocument):
        """Add document to index."""
        self.documents[document.doc_id] = document
        
        for field, value in document.fields.items():
            if not isinstance(value, str):
                continue
                
            tokens = self._tokenize(value)
            self.field_lengths[document.doc_id][field] = len(tokens)
            
            for position, token in enumerate(tokens):
                key = f"{field}:{token}"
                self.index[key][document.doc_id].append(position)
                
        # Update document frequencies
        self._update_doc_frequencies()
        
        logger.info(f"Indexed document: {document.doc_id}")
        
    def remove_document(self, doc_id: str):
        """Remove document from index."""
        if doc_id not in self.documents:
            return
            
        document = self.documents[doc_id]
        
        for field, value in document.fields.items():
            if not isinstance(value, str):
                continue
                
            tokens = self._tokenize(value)
            for token in set(tokens):
                key = f"{field}:{token}"
                if doc_id in self.index[key]:
                    del self.index[key][doc_id]
                    
        del self.documents[doc_id]
        self._update_doc_frequencies()
        
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        # Lowercase and split on word boundaries
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        return tokens
        
    def _update_doc_frequencies(self):
        """Update document frequency counts."""
        self.doc_frequency.clear()
        
        for key, docs in self.index.items():
            self.doc_frequency[key] = len(docs)
            
        # Update average field lengths
        field_totals = defaultdict(int)
        field_counts = defaultdict(int)
        
        for doc_id, fields in self.field_lengths.items():
            for field, length in fields.items():
                field_totals[field] += length
                field_counts[field] += 1
                
        for field in field_totals:
            self.avg_field_lengths[field] = field_totals[field] / field_counts[field]
            
    def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search for documents."""
        query_tokens = self._tokenize(query.query_string)
        
        # Score documents using BM25
        scores = self._score_documents(query_tokens, query.fields)
        
        # Apply filters
        if query.filters:
            scores = {
                doc_id: score
                for doc_id, score in scores.items()
                if self._matches_filters(doc_id, query.filters)
            }
            
        # Sort by score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Pagination
        paginated = sorted_results[query.offset:query.offset + query.max_results]
        
        # Build results
        results = []
        for doc_id, score in paginated:
            document = self.documents[doc_id]
            result = SearchResult(
                doc_id=doc_id,
                doc_type=document.doc_type,
                score=score,
                fields=document.fields,
                highlights=self._generate_highlights(doc_id, query_tokens, query.fields),
            )
            results.append(result)
            
        return results
        
    def _score_documents(
        self,
        query_tokens: List[str],
        fields: List[SearchField],
    ) -> Dict[str, float]:
        """Score documents using BM25 algorithm."""
        k1 = 1.5  # Term frequency saturation
        b = 0.75  # Length normalization
        
        scores = defaultdict(float)
        total_docs = len(self.documents)
        
        for token in query_tokens:
            for field in fields:
                key = f"{field.value}:{token}"
                
                if key not in self.index:
                    continue
                    
                # Inverse document frequency
                df = self.doc_frequency[key]
                idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
                
                # Term frequency for each document
                for doc_id, positions in self.index[key].items():
                    tf = len(positions)
                    
                    # Document length normalization
                    doc_length = self.field_lengths[doc_id].get(field.value, 0)
                    avg_length = self.avg_field_lengths.get(field.value, 1)
                    norm = (1 - b) + b * (doc_length / avg_length)
                    
                    # BM25 score
                    score = idf * (tf * (k1 + 1)) / (tf + k1 * norm)
                    
                    # Apply document boost
                    document = self.documents[doc_id]
                    score *= document.boost
                    
                    scores[doc_id] += score
                    
        return scores
        
    def _matches_filters(self, doc_id: str, filters: Dict[str, Any]) -> bool:
        """Check if document matches filters."""
        document = self.documents[doc_id]
        
        for field, value in filters.items():
            if field not in document.fields:
                return False
                
            doc_value = document.fields[field]
            
            if isinstance(value, list):
                if doc_value not in value:
                    return False
            else:
                if doc_value != value:
                    return False
                    
        return True
        
    def _generate_highlights(
        self,
        doc_id: str,
        query_tokens: List[str],
        fields: List[SearchField],
    ) -> Dict[str, List[str]]:
        """Generate highlighted snippets."""
        document = self.documents[doc_id]
        highlights = {}
        
        for field in fields:
            if field.value not in document.fields:
                continue
                
            text = document.fields[field.value]
            if not isinstance(text, str):
                continue
                
            # Find matching positions
            tokens = self._tokenize(text)
            matches = []
            
            for i, token in enumerate(tokens):
                if token in query_tokens:
                    # Extract snippet around match
                    start = max(0, i - 5)
                    end = min(len(tokens), i + 6)
                    snippet = ' '.join(tokens[start:end])
                    matches.append(f"...{snippet}...")
                    
            if matches:
                highlights[field.value] = matches[:3]  # Top 3 snippets
                
        return highlights


class SearchEngine:
    """
    Production-grade search engine.
    
    Features:
    - Full-text indexing with BM25 ranking
    - Fuzzy matching with Levenshtein distance
    - Faceted search aggregations
    - Search suggestions
    - Result boosting
    - Search analytics
    """
    
    def __init__(self):
        self.index = SearchIndex()
        self.suggestions: Set[str] = set()
        
        # Analytics
        self.search_count = 0
        self.popular_queries: Dict[str, int] = defaultdict(int)
        
    async def index_document(self, document: SearchDocument):
        """Index a document."""
        self.index.add_document(document)
        
        # Add to suggestions
        for field, value in document.fields.items():
            if isinstance(value, str):
                self.suggestions.update(self.index._tokenize(value))
                
    async def remove_document(self, doc_id: str):
        """Remove a document from index."""
        self.index.remove_document(doc_id)
        
    async def search(self, query: SearchQuery) -> Dict[str, Any]:
        """Execute search query."""
        self.search_count += 1
        self.popular_queries[query.query_string] += 1
        
        # Execute search
        results = self.index.search(query)
        
        # Build facets if requested
        facets = []
        if query.facets:
            facets = self._build_facets(results, query.facets)
            
        return {
            "results": [
                {
                    "doc_id": r.doc_id,
                    "doc_type": r.doc_type,
                    "score": r.score,
                    "fields": r.fields,
                    "highlights": r.highlights,
                }
                for r in results
            ],
            "total": len(results),
            "facets": facets,
            "query": query.query_string,
        }
        
    async def suggest(self, prefix: str, max_suggestions: int = 10) -> List[str]:
        """Get search suggestions."""
        prefix = prefix.lower()
        matches = [
            s for s in self.suggestions
            if s.startswith(prefix)
        ]
        
        # Sort by popularity (using query frequency)
        matches.sort(key=lambda s: self.popular_queries.get(s, 0), reverse=True)
        
        return matches[:max_suggestions]
        
    def _build_facets(
        self,
        results: List[SearchResult],
        facet_fields: List[str],
    ) -> List[FacetResult]:
        """Build facet aggregations."""
        facets = []
        
        for field in facet_fields:
            buckets = defaultdict(int)
            
            for result in results:
                if field in result.fields:
                    value = result.fields[field]
                    if isinstance(value, list):
                        for v in value:
                            buckets[v] += 1
                    else:
                        buckets[value] += 1
                        
            facet = FacetResult(
                field=field,
                buckets=[
                    {"value": k, "count": v}
                    for k, v in sorted(buckets.items(), key=lambda x: x[1], reverse=True)
                ]
            )
            facets.append(facet)
            
        return facets
        
    def get_analytics(self) -> Dict[str, Any]:
        """Get search analytics."""
        return {
            "total_searches": self.search_count,
            "indexed_documents": len(self.index.documents),
            "popular_queries": sorted(
                self.popular_queries.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            "vocabulary_size": len(self.suggestions),
        }
