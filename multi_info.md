Available tools across all servers:
  - file_list_pdf_files: Search for PDF files by keyword in their filename (case-insensitive). Searches both the allowed and restricted folders. Returns a list of objects each with 'id', 'folder', and 'filename'. Pass an empty string to list all PDF files across both folders. Use the returned 'id' and 'folder' values with read_pdf.
  - file_read_pdf: Read a PDF file and return its full content as Markdown text. For 'allowed' files, no token is needed. For 'restricted' files, supply the correct token. Obtain valid file_id and folder values from list_pdf_files or from the catalog://allowed resource.
  - file_extract_headers: Extract all bold headers from a PDF document. Headers are lines whose entire text is wrapped in double stars (**like this**). Returns an ordered list of header strings, stripped of the surrounding stars. Use the returned header strings with extract_section to retrieve the content beneath a specific header. Internally calls read_pdf, so the same file_id/folder/token rules apply.
  - file_extract_section: Extract the content that follows a specific bold header in a PDF. The header must match exactly one of the headers returned by extract_headers (comparison is case-insensitive and whitespace-tolerant). Content runs from the line after the matched header up to (but not including) the next bold header, or end-of-document. Internally calls read_pdf, so the same file_id/folder/token rules apply.
  - file_summarize_filtered_sections: Performs a targeted cross-document search and generates a concise synthesis.

Use this tool when you need to compare how a specific topic (keyword) is 
addressed across multiple documents within a specific structural context 
(e.g., comparing 'Methodology' or 'Future Work' across several papers).

Args:
    keyword: The specific term, technology, or concept to search for within 
            the section text (case-insensitive).
    section_target: The exact name of the section to target (e.g., 'Introduction', 
                    'Abstract', 'Conclusion', 'Results').
    token: Optional access token for restricted documents.  

Returns:
    A formatted string containing the source filename and a 1-2 sentence 
    summary of the relevant section for every document where the keyword 
    was found. Returns a 'not found' message if no matches occur.

Example:
    If you want to know how different papers introduce 'Reinforcement Learning', 
    call: search_and_summarize_sections(keyword="RL", section_target="Introduction")
  - api_validate_and_fetch_metadata: Searches a paper title and fetches official metadata from Crossref.
Handles malicious input and sanitizes external responses.

Args:
    title (str): the title string of the paper.
Returns:
    str: the serialized dictionary of official_title, publisher, year of the paper
  - rag_search_research_papers: Performs a semantic similarity search across the local research repository (LRAA).

Use this tool when the user asks questions about specific papers, technical 
methodologies, or empirical results stored in the local PDF library. This tool 
retrieves raw text chunks based on the conceptual meaning of the query, 
not just keyword matching.

Args:
    query (str): A detailed search string or specific question. For best results, 
                 use technical terms or full sentences (e.g., "latent space 
                 regularization in GANs").
    n (int): The number of top relevant text chunks to retrieve. Increase 'n' 
             for complex topics requiring broader context. Default is 3.

Returns:
    str: A concatenated string of text segments, each labeled with its 
         source filename and page number for auditing and citation purposes.
