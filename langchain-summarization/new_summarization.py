import os
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from collections import defaultdict

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Configuration
LANGCHAIN_DIR = Path.cwd()
# Use JSON files for full database provenance (PMCID, text_element_id)
INPUT_JSON_DIR = LANGCHAIN_DIR / "test_results_50_docs" / "umls_entities"
# Legacy TXT files (less provenance data)
INPUT_TXT_DIR = LANGCHAIN_DIR / "test_results_50_docs" / "relevant_texts"
OUTPUT_DIR = LANGCHAIN_DIR / "summarization_results"
SUMMARIES_DIR = OUTPUT_DIR / "summaries"
RULES_DIR = OUTPUT_DIR / "rules"
AUDIT_DIR = OUTPUT_DIR / "audit_trails"

# Create output directories
for directory in [OUTPUT_DIR, SUMMARIES_DIR, RULES_DIR, AUDIT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

print(f"Input JSON (with DB provenance): {INPUT_JSON_DIR}")
print(f"Input TXT (legacy):              {INPUT_TXT_DIR}")
print(f"Output directory:                {OUTPUT_DIR}")
print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# Load environment variables (API keys)
load_dotenv()

# Verify API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("⚠️  Warning: OPENAI_API_KEY not found in environment")
    print("Please set it in .env file or export OPENAI_API_KEY=your_key")
else:
    print("✅ OpenAI API key loaded successfully")

def load_json_files_with_provenance(input_dir: Path, limit: Optional[int] = None) -> List[Dict]:
    """
    Load UMLS entity JSON files with FULL DATABASE PROVENANCE.
    
    Each sentence includes:
    - pmcid: Links to documents.pmcid in database
    - text_element_id: Links to text_elements.id in database
    - section: Section context from text_elements.path_string
    - entity_text, start_char, end_char: Entity position
    - umls_score: Entity linking confidence
    
    This provides complete traceability back to the database schema.
    
    Args:
        input_dir: Directory containing JSON files
        limit: Optional limit on number of files to load
    
    Returns:
        List of dicts with full provenance metadata
    """
    json_files = sorted(input_dir.glob("*.json"))
    
    if limit:
        json_files = json_files[:limit]
    
    loaded_files = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract metadata
            cui = data.get('umls_cui', '')
            concept_name = data.get('canonical_name', '')
            entity_label = data.get('entity_label', '')
            
            # Extract sentences with full provenance
            sentences_with_provenance = []
            for sent_data in data.get('sentences', []):
                sentences_with_provenance.append({
                    'pmcid': sent_data.get('pmcid'),  # Links to documents.pmcid
                    'text_element_id': sent_data.get('text_element_id'),  # Links to text_elements.id
                    'sentence': sent_data.get('sentence'),
                    'section': sent_data.get('section'),
                    'entity_text': sent_data.get('entity_text'),
                    'start_char': sent_data.get('start_char'),
                    'end_char': sent_data.get('end_char'),
                    'umls_score': sent_data.get('umls_score')
                })
            
            # Group by PMCID for statistics
            pmcids = set(s['pmcid'] for s in sentences_with_provenance if s['pmcid'])
            
            loaded_files.append({
                'cui': cui,
                'concept_name': concept_name,
                'entity_label': entity_label,
                'filename': json_file.name,
                'filepath': str(json_file),
                'sentences_with_provenance': sentences_with_provenance,
                # Simple sentence list for compatibility
                'sentences': [s['sentence'] for s in sentences_with_provenance],
                'num_sentences': len(sentences_with_provenance),
                'num_documents': len(pmcids),
                'pmcids': list(pmcids),
                'metadata': {
                    'umls_cui': cui,
                    'canonical_name': concept_name,
                    'entity_label': entity_label,
                    'total_occurrences': data.get('total_occurrences', 0),
                    'unique_entity_texts': data.get('unique_entity_texts', [])
                }
            })
        
        except Exception as e:
            print(f"Warning: Could not load {json_file.name}: {e}")
    
    return loaded_files


# Load JSON files with full provenance
print("Loading JSON files with database provenance...")
json_files = load_json_files_with_provenance(INPUT_JSON_DIR, limit=10)

print(f"\nLoaded {len(json_files)} concept files with DB provenance")
print("\nSample file structure:")
if json_files:
    sample = json_files[0]
    print(f"  CUI: {sample['cui']}")
    print(f"  Concept: {sample['concept_name']}")
    print(f"  Sentences: {sample['num_sentences']}")
    print(f"  Documents (PMCIDs): {sample['num_documents']}")
    if sample['sentences_with_provenance']:
        sent = sample['sentences_with_provenance'][0]
        print(f"\n  Sample sentence provenance:")
        print(f"    PMCID: {sent['pmcid']} -> documents.pmcid")
        print(f"    text_element_id: {sent['text_element_id']} -> text_elements.id")
        print(f"    section: {sent['section']}")
        print(f"    umls_score: {sent['umls_score']}")

# =============================================================================
# AUDITABLE MAP/REDUCE SYSTEM PROMPTS
# =============================================================================
# This system maintains full traceability from final summary -> chunks -> source sentences -> database
#
# Provenance Chain:
#   Final Summary [claim] -> Chunk ID -> Sentence IDs -> PMCID + text_element_id (database)
#
# Output Format: Structured JSON blocks enable machine parsing for audit validation
# =============================================================================

MAP_PROMPT = """<Role>You are a Medical Evidence Analyst processing a chunk of histopathology literature.</Role>

<Context>
Concept: {concept_name}
Chunk ID: {chunk_id}
Source Sentences (each tagged with [SentenceID|PMCID|TextElementID]):
{text}
</Context>

<Task>
Analyze this chunk and produce a STRUCTURED AUDITABLE SUMMARY with full provenance tracking.

CRITICAL AUDIT REQUIREMENTS:
1. Every factual claim MUST cite specific sentence IDs from the input
2. Use the exact citation format: [S1|PMC123|te456] where available
3. Never synthesize or infer claims without direct textual support
4. If information conflicts, cite all sources and note the conflict
</Task>

<OutputFormat>
Return your analysis in this EXACT structure:

```json
{{
  "chunk_id": "{chunk_id}",
  "findings": [
    {{
      "category": "diagnostic|histopathological|treatment|prognostic|risk_factor",
      "claim": "<factual statement>",
      "evidence": ["S1|PMC123|te456", "S3|PMC123|te458"],
      "confidence": "high|medium|low",
      "verbatim_support": "<key quote from source>"
    }}
  ],
  "summary_text": "<narrative summary with inline citations [S1|PMC...]>",
  "audit_metadata": {{
    "sentences_analyzed": <count>,
    "sentences_cited": [<list of cited sentence IDs>],
    "pmcids_referenced": [<list of PMCIDs>],
    "uncited_sentences": [<list of uncited sentence IDs>]
  }}
}}
```
</OutputFormat>

Structured Analysis:"""