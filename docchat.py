import readline

from dotenv import load_dotenv
load_dotenv()  

def llm(messages, temperature=1):
    '''
    

    >>> llm([
    ...     {'role': 'system', 'content': 'You are a helpful assistant.'},
    ...     {'role': 'system', 'content': 'What is the capital of France?'},
    ...     ], temperature=0)
    'The capital of France is Paris!'
    '''
    import groq
    client = groq.Groq()

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama3-8b-8192",
        temperature=temperature,
    )
    return chat_completion.choices[0].message.content


def load_text(filepath_or_url):
    '''
    This function will load text from a given file path or URL. Supports .txt, .html, and .pdf formats.

    >>> load_text("docs/hello.txt")
    'hello world'
    >>> load_text("docs/hello.html")
    'hello world'
    >>> 'Carmel' in load_text("docs/carmel.pdf")
    True
    '''
    import os
    import requests
    from urllib.parse import urlparse
    from bs4 import BeautifulSoup
    from PyPDF2 import PdfReader
    from io import BytesIO

    def is_url(path):
        parsed = urlparse(path)
        return parsed.scheme in ("http", "https")

    def get_extension(path):
        return os.path.splitext(urlparse(path).path)[1].lower()

    def load_txt(content):
        return content

    def load_html(content):
        soup = BeautifulSoup(content, "html.parser")
        return soup.get_text(separator=' ', strip=True)

    def load_pdf(content):
        try:
            reader = PdfReader(content)
            return "\n".join(page.extract_text() or '' for page in reader.pages).strip()
        except Exception:
            raise ValueError("Could not extract text from PDF: file may be invalid or empty.")

    ext = get_extension(filepath_or_url)

    if is_url(filepath_or_url):
        response = requests.get(filepath_or_url)
        response.raise_for_status()
        content = response.content
    else:
        with open(filepath_or_url, 'rb') as f:
            content = f.read()

    if ext == ".txt":
        return load_txt(content.decode("utf-8", errors="ignore"))
    elif ext == ".html":
        return load_html(content)
    elif ext == ".pdf":
        from io import BytesIO
        return load_pdf(BytesIO(content))
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def chunk_text_by_words(text, max_words=100, overlap=50):
    '''
    This function will split the input text document into chunks of length max_words. Each chunk should overlap previous chunks by the overlap amount.

        >>> text = "The quick brown fox jumps over the lazy dog. It was a sunny day and the birds were singing."
        >>> chunks = chunk_text_by_words(text, max_words=5, overlap=2)
        >>> len(chunks)
        7
        >>> chunks[0]
        'The quick brown fox jumps'
        >>> chunks[1]
        'fox jumps over the lazy'
        >>> chunks[4]
        'sunny day and the birds'
        >>> chunks[-1]
        'singing.'
    '''
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + max_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_words - overlap

    return chunks


def score_chunk(chunk: str, query: str, language: str = "french") -> float:
    '''
    This function will associate a "similarity score" between 0 and 1 to the chunk and query variables. Higher scores should signify "more similar" and lower scores should signify "less similar".

    Examples (English):
        >>> round(score_chunk("The sun is bright and hot.", "How hot is the sun?", language="english"), 2)
        0.22
        >>> round(score_chunk("The red car is speeding down the road.", "What color is the car?", language="english"), 2)
        0.2
        >>> score_chunk("Bananas are yellow.", "How do airplanes fly?", language="english")
        0.0
    '''
    chunk_words = set(chunk.lower().split())
    query_words = set(query.lower().split())

    if not chunk_words or not query_words:
        return 0.0

    intersection = chunk_words & query_words
    union = chunk_words | query_words

    return len(intersection) / len(union)
    

def find_relevant_chunks(text, query, num_chunks=5):
    '''
    This function will: 
    1) split the document into chunks 
    2) compute the score for each of these chunks
    3) return the num_chunks chunks that have the largest score

    >>> text = "The sun is bright and hot. Bananas are yellow. The red car speeds by."
    >>> query = "How hot is the sun?"
    >>> find_relevant_chunks(text, query, num_chunks=1)
    ['The sun is bright and hot. Bananas are yellow. The']
    '''
    chunks = chunk_text_by_words(text, max_words=10, overlap=5)

    # Use accumulator pattern to gather scored chunks
    scored_chunks = []
    for chunk in chunks:
        score = score_chunk(chunk, query)
        scored_chunks.append((chunk, score))

    # Sort and slice
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    top_chunks = [chunk for chunk, _ in scored_chunks[:num_chunks]]

    return top_chunks



if __name__ == '__main__':
    messages = []
    messages.append({
        'role': 'system',
        'content': "You are a helpful assistant. You always provide a brief summary on the input files. You always answer questions in 3 clear and concise sentences.",
    })

    import sys
    import langid  #Language detection

    filepath_or_url = sys.argv[1]
    document_text = load_text(filepath_or_url)

    document_language = langid.classify(document_text[:1000])[0]
    print("Detected language:", document_language)

    summary = llm([
        {'role': 'system', 'content': 'You summarize documents in 3 clear and concise sentences.'},
        {'role': 'user', 'content': f'Summarize this document:\n\n{document_text[:4000]}'}
    ])

    while True:
        # Get input from the user 
        text = input('docchat>')

        # Detect the language of the user's query
        user_language = langid.classify(text)[0]

        # If query is in English but document is not, translate the query
        if user_language == 'en' and document_language != 'en':
            translation_prompt = f"Translate this question into {document_language}:\n\n{text}"
            translated_query = llm([
                {'role': 'system', 'content': 'You are a translation engine.'},
                {'role': 'user', 'content': translation_prompt}
            ])
        else:
            translated_query = text

        top_chunks = find_relevant_chunks(document_text, translated_query, num_chunks=10)
        retrieved_info = "\n\n".join(top_chunks)

        modified_text = f"""
You are doing RAG.

Document summary:
{summary}

Relevant chunks:
{retrieved_info}

The user's question is: {text}
""".strip()

        # Pass that input to llm
        messages.append({
            'role': 'user',
            'content': modified_text,
        })
        result = llm(messages)

        # Add the "assistant" role to the messages list so that the 'llm' has access to the whole conversation history
        # and will know what it has previously says, and update its response based on that information 
        messages.append({
            'role': 'assistant',
            'content': result,
        })

        # Print the llm's response to the user
        print('result=', result)
        