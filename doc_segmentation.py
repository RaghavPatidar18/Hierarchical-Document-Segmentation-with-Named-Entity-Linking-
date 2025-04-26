import fitz  # PyMuPDF
import re
import json
import numpy as np
from collections import defaultdict
from verify_llm import VerifyWithLLM
import stanza
import os

stanza_dir = os.path.expanduser("~/.stanza")
if not os.path.exists(os.path.join(stanza_dir, "resources", "en")):
    stanza.download('en')
    print("stanza downloaded")

nlp = stanza.Pipeline('en', processors='tokenize,ner', use_gpu=False)

class HierarchicalDocumentSegmenter:
    def __init__(self, pdf_path, llm_api_key=None, llm_provider="groq"):
        self.pdf_path = pdf_path  # pdf path
        self.full_text = ""  # output of file
        self.formatted_blocks = [] # properties of each word in full text
        self.segments = [] # segments of text and its properties

        self.llm_api_key = llm_api_key # api key for groq
        self.llm_provider = llm_provider 
        
    
    def extract_text_with_formatting(self):
        """Extract text from PDF while preserving formatting information"""
        doc = fitz.open(self.pdf_path)
        full_text_parts = []
        char_index = 0
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"]
                            
                            # Add text to full document
                            full_text_parts.append(text)
                            char_index += len(text)
                            
                            # Store block info for rule-based analysis
                            self.formatted_blocks.append({
                                "text": text,
                                "font": span["font"],
                                "size": span["size"],
                                "flags": span["flags"],
                                "start_index": char_index - len(text),
                                "end_index": char_index,
                                "page": page_num
                            })
                        
                        # Add newline after each line
                        full_text_parts.append("\n")
                        char_index += 1
            
            # Add page break
            full_text_parts.append("\n\n")
            char_index += 2
    
        self.full_text = "".join(full_text_parts)
        
        return self.full_text
    
    def identify_candidate_headings_rule_based(self):
        """Use rules to identify potential section headings"""
        candidate_headings = []
        
        # Group blocks by font properties
        font_groups = defaultdict(list)
        
        for block in self.formatted_blocks:
            # Create a key based on font properties
            font_key = f"{block['font']}*{block['size']}*{block['flags']}"
            font_groups[font_key].append(block)
        
        # Find the most common font properties (likely body text)
        font_counts = {k: len(v) for k, v in font_groups.items()}
        if font_counts:
            body_font = max(font_counts, key=font_counts.get)   # body_font gets the font size of body text
            
            # Identify headings based on font differentiation
            for font_key, blocks in font_groups.items():
                font_size = float(font_key.split('*')[1])
                body_size = float(body_font.split('*')[1])
                
                # If font is larger than body text, it's a candidate heading
                if font_size > body_size * 1.1:  # At least 10% larger
                    for block in blocks:
                        text = block['text'].strip()
                        if text and len(text) < 100:  # Headings are usually short
                            candidate_headings.append({
                                'text': text,
                                'font_size': font_size,
                                'start_index': block['start_index'],
                                'end_index': block['end_index'],
                                'page': block['page']
                            })
        
        # Find headings based on numbering patterns
        heading_patterns = [
            r"^Chapter\s+\d+[\.:]\s*(.+)$",
            r"^\d+[\.:]\s*(.+)$",
            r"^\d+\.\d+[\.:]\s*(.+)$",
            r"^\d+\.\d+\.\d+[\.:]\s*(.+)$",
            r"^[A-Z][\:]\s*(.+)$",
            r"^[ivxlcdm]+[\.:]\s*(.+)$",  
            r"^(?:Section|Part|Unit)\s+\d+[\.:]\s*(.+)$"
        ]
        
        # Finding other potential headings based on regex
        for block in self.formatted_blocks:
            text = block['text'].strip()
            for pattern in heading_patterns:
                if re.match(pattern, text, re.IGNORECASE):
                    candidate_headings.append({
                        'text': text,
                        'font_size': block['size'],
                        'start_index': block['start_index'],
                        'end_index': block['end_index'],
                        'page': block['page']
                    })
        # Sort by position in document
        candidate_headings.sort(key=lambda x: x['start_index'])
        
        return candidate_headings
    
    
    def assign_levels_by_rules(self, candidate_headings):

        # Group headings by font size
        size_groups = defaultdict(list)
        for heading in candidate_headings:
            size_groups[heading['font_size']].append(heading)
        
        # Sort sizes in descending order
        sorted_sizes = sorted(size_groups.keys(), reverse=True)
        
        # Assign levels based on font size ranking
        verified_headings = []
        for i, size in enumerate(sorted_sizes, start=1):
            for heading in size_groups[size]:
                heading_copy = heading.copy()
                heading_copy['level'] = min(i,3)  
                
                # Clean title by removing numbering
                text = heading_copy['text']
                clean_title = re.sub(r"^\s*(?:Chapter\s+\d+|(?:\d+\.)+\d*|[A-Z]\.)\s*", "", text) # cleaning the title
                heading_copy['clean_title'] = clean_title.strip()
                
                verified_headings.append(heading_copy)
        
        # Sort by position in document
        verified_headings.sort(key=lambda x: x['start_index'])
        return verified_headings
    
    def create_hierarchical_segments(self, verified_headings):
        """Build segments from verified headings"""
        segments = []
        
        # Process each heading
        for i, heading in enumerate(verified_headings):
            # Determine segment end (start of next segment or end of document)
            end_index = verified_headings[i+1]['start_index'] if i+1 < len(verified_headings) else len(self.full_text)
            
            # Extract segment text
            segment_text = self.clean_text(self.full_text[heading['end_index']:end_index].strip())
            
            segment = {
                'segment_level': heading['level'],
                'segment_title': heading['clean_title'],
                'segment_text': segment_text,
                'start_index': heading['start_index'],
                'end_index': end_index
            }
            
            # Add other metadata
            segment['segment_date'] = self.extract_segment_date(segment_text)
            segment['segment_source'] = self.extract_segment_source(segment_text)
            
            segments.append(segment)
        
        self.segments = segments
        return segments
    
    def extract_segment_date(self, segment_text):
        """Extract date from segment if available"""
        
        date_patterns = [
            r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # MM/DD/YYYY
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',  # Month DD, YYYY
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b'
        ]
        # extracting the dates using the regex pattern
        for pattern in date_patterns:
            match = re.search(pattern, segment_text)
            if match:
                return match.group(0)
        
        return None
    
    def extract_segment_source(self, segment_text):
        """Extract source/contributor information if available"""
        # Look for common source patterns
        source_patterns = [
            r'(?:Author|By|Contributed by|Written by)[:\s]+([A-Z][a-zA-Z\s\.-]+)',
            r'(?:From|Source)[:\s]+([A-Z][a-zA-Z\s\.-]+)'
        ]
        
        for pattern in source_patterns:
            match = re.search(pattern, segment_text[:1000])  
            if match:
                return match.group(1).strip()
        
        doc = nlp(segment_text)
        for sent in doc.sentences:
            for ent in sent.ents:
                if ent.type == "PERSON" and ent.start_char < 200:  
                    # Verify with LLM this is likely an author
                    return VerifyWithLLM._verify_author_with_llm(ent.text, segment_text[:1000])
        
        
        return None
    
    def clean_text(self,text):
    
        replacements = {
            '\n':' ',
            '–':'-',
            '�':''
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text.strip()

    def extract_named_entities(self):
        """Extract named entities from each segment"""
        
        for segment in self.segments:
            text = segment['segment_text']
            
            doc = nlp(text)
            entities = {
                "persons": [],
                "organizations": [],
                "locations": [],
                "dates": []
            }
            
            #matching the type from stanza ner 
            for sent in doc.sentences:
                for ent in sent.ents:
                    if ent.type == "PERSON":
                        entities["persons"].append(ent.text)
                    elif ent.type == "ORG":
                        entities["organizations"].append(ent.text)
                    elif ent.type in ("GPE", "LOC"):
                        entities["locations"].append(ent.text)
                    elif ent.type == "DATE":
                        entities["dates"].append(ent.text)
            
            # Deduplicate entities
            for key in entities:
                entities[key] = list(set(entities[key]))
            
            segment['named_entities'] = entities
        
        return self.segments
    
    def process_document(self):
        """Run the full document processing pipeline"""

        # Extract text with formatting
        self.extract_text_with_formatting()
        
        # Identify candidate headings using rules
        candidate_headings = self.identify_candidate_headings_rule_based()
        
        # We assign levels to the headings by rules
        verified_headings = self.assign_levels_by_rules(candidate_headings)
        
        # Create hierarchical segments
        self.create_hierarchical_segments(verified_headings)
        
        # Extract named entities
        self.extract_named_entities()
        
        return self.segments
    
    def export_to_json(self, output_path):
       
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.segments, f, indent=2, ensure_ascii=False)
        print("Export done.... Please see segments.json file")
        return output_path

