README



Approach and Thought Process

Build a robust pipeline for hierarchical document segmentation from PDFs, optimized for both accuracy and scalability. I first use PyMuPDF (fitz) to extract text along with formatting metadata (font, size, flags), as purely text-based extraction would lose critical layout information necessary for structural reconstruction. Rule-based heuristics are applied to detect candidate headings by identifying text blocks with larger font sizes or specific numbering patterns (e.g., "1.", "1.1.", "Chapter 2:"), because such visual and syntactic cues are highly reliable and computationally efficient compared to full document parsing with deep models.

After candidate headings are detected, font size is used to assign hierarchical levels (e.g., H1, H2, H3), reflecting the relative importance of sections. For Named Entity Recognition (NER), I chose Stanza over heavier transformers like spaCy transformers or BERT-based models because Stanza strikes a good balance between accuracy and speed, especially important when processing larger documents on limited compute resources. It also supports 60+ multilingual features and is designed for rich literacy. 

For verifying if a detected person entity is truly an author or contributor, Integrate a Groq Llama 3-70B model. I chose this large language model because of its strong performance in factual reasoning and instruction following, critical when asking very targeted questions like "Is X likely to be the author?". The decision to use LLM-based verification adds an additional semantic check that rule-based systems alone cannot achieve, minimizing false positives.

Overall, this hybrid design — combining rule-based extraction for structure and LLM-based semantic validation for metadata — was chosen to optimize for both speed (lightweight extraction) and intelligence (deep context understanding), ensuring scalability across diverse document types without sacrificing quality.



Assumption Made 

Only 3 levels of hierarchy is given




How to run code

In the .env file provide your Groq API key 
 


It is not necessary , work can be done without it also but now LLM is not used for verification of NER. However I will provide my API key in the code in the .zip file.

Run : 
	pip install -r requirements.txt

In main.py file provide input and output path 

Replace "./RaghavResume.pdf" with you input pdf file path 
Output will be in segments.json at the current location



Run :
	python main.py


