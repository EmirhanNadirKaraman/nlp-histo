import pubmed_parser as pp
import json
import os

def parse_histo_paper(nxml_path):
    """
    Parses an NXML file to extract metadata, methods, and figure captions.
    """
    
    # 1. Extract Basic Metadata (Title, Journal, ID)
    # Returns a dictionary with keys like 'full_title', 'pmid', 'publication_year'
    meta_dict = pp.parse_pubmed_xml(nxml_path)
    
    # 2. Extract Figure Captions
    # Returns a list of dictionaries containing 'fig_caption' and 'fig_id'
    captions = pp.parse_pubmed_caption(nxml_path)
    
    print(meta_dict)

    # Clean up captions (sometimes they contain XML artifacts)
    structured_captions = []
    for cap in captions:
        structured_captions.append({
            "fig_id": cap.get("fig_id"),
            "caption_text": cap.get("fig_caption"),
            # 'graphic_ref' is often the filename of the image in the package
            "image_filename": cap.get("graphic_ref") 
        })

    # 3. Extract and Filter for "Methods" Section
    # parse_pubmed_paragraph returns a list of dicts: {'section': ..., 'text': ...}
    paragraphs = pp.parse_pubmed_paragraph(nxml_path, all_paragraph=False)

    print("paragraphs", paragraphs)
    
    methods_text = []
    
    # Keywords to identify methods sections
    # Histopathology papers often vary: "Materials and Methods", "Methodology", "Staining Procedure"
    method_keywords = ["method", "material", "procedure", "protocol"]
    
    for p in paragraphs:
        section_title = p.get('section', '').lower()
        text_content = p.get('text', '')
        
        # Check if the section title matches our keywords
        if any(keyword in section_title for keyword in method_keywords):
            methods_text.append({
                "section_title": p.get('section'),
                "content": text_content
            })

    # 4. Construct Final Structured Record
    paper_record = {
        "pmid": meta_dict.get("pmid"),
        "title": meta_dict.get("full_title"),
        "journal": meta_dict.get("journal"),
        "year": meta_dict.get("publication_year"),
        "methods_section": methods_text,
        "figure_captions": structured_captions
    }
    
    return paper_record

# --- Usage Example ---

# Replace this with the path to your actual .nxml file
# If you don't have one, download a sample from PMC Open Access subset
file_path = "files/organized_xmls/PMC1448691.nxml" 

if os.path.exists(file_path):
    data = parse_histo_paper(file_path)
    
    # Print readable JSON output
    print(json.dumps(data, indent=4))
    
    # Option: Save to a JSON file (Acting as your NoSQL entry)
    with open(f"{data['pmid']}_extracted.json", "w") as f:
        json.dump(data, f, indent=4)
        print(f"Data saved for PMID: {data['pmid']}")
else:
    print(f"File not found: {file_path}")
    print("Please ensure you have a valid .nxml file in the directory.")