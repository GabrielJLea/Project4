

import base64
import textract
import requests
from bs4 import BeautifulSoup
import argparse
import groq


from dotenv import load_dotenv
load_dotenv()





def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def extract_text_from_pdf(pdf_path):

    text = textract.process(pdf_path, encoding='utf-8')
    return text.decode('utf-8')

def llm_image(image_url):
    try:
        vision_completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Summarize whats in this image."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            ],
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )
        # print(vision_completion.choices[0].message.content)
        chat_bot(vision_completion.choices[0].message.content)

    except groq.BadRequestError:
        base64_image = encode_image(image_url)
        llm_image(f"data:image/jpeg;base64,{base64_image}")
    
        return 

def load_text(file):
    """
    Load text

        >>> load_text('docs/test.txt')  # doctest: +ELLIPSIS
        Invalid document
    """
    try: 
        with open(file, 'r') as fin:
            html = fin.read()
            soup = BeautifulSoup(html, features="lxml")

            # Remove all <style> and <script> elements
            for tag in soup(['style', 'script']):
                tag.decompose()

            # Get clean text
            text = soup.get_text(separator=' ', strip=True)
            # print(text)
            chat_bot(text)
    except (FileNotFoundError, UnicodeDecodeError):
        try:
            llm_image(file)
        except:
            try:
                response = requests.get(file)
                html = response.text
                soup = BeautifulSoup(html, features="lxml")
                for tag in soup(['style', 'script']):
                    tag.decompose()

                text = soup.get_text(separator=' ', strip=True)
                chat_bot(text)
            except requests.exceptions.MissingSchema:
                try:
                    chat_bot(extract_text_from_pdf(file))
                except (textract.exceptions.ShellError, textract.exceptions.MissingFileError):
                    try:
                        llm_image(file)
                    except:
                        print("Invalid document")


    return

def llm(messages, temperature=1):
    '''
    This function is my interface for calling the LLM.
    The messages argument should be a list of dictionaries.

    >>> llm([
    ...     {'role': 'system', 'content': 'You are a helpful assistant.'},
    ...     {'role': 'user', 'content': 'What is the capital of France?'},
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


def chunk_text_by_words(text, max_words=5, overlap=2):
    """
    Splits text into overlapping chunks by word count.

    Examples:
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
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + max_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_words - overlap

    return chunks


import spacy

def load_spacy_model(language: str):
    """
    Loads a spaCy model for the specified language.
    """
    LANGUAGE_MODELS = {
        'french': 'fr_core_news_sm',
        'german': 'de_core_news_sm',
        'spanish': 'es_core_news_sm',
        'english': 'en_core_web_sm',
    }

    if language not in LANGUAGE_MODELS:
        raise ValueError(f"Unsupported language: {language}")

    return spacy.load(LANGUAGE_MODELS[language])


def score_chunk(chunk: str, query: str) -> float:
    """
        >>> score_chunk(set(['soccer', 'football']), set(['football', 'soccer']))
        1.0

        >>> score_chunk(set(['soccer']), set(['foorball', 'soccer']))
        0.5

        >>> score_chunk(set(['I', 'Like', 'to', 'play']), set(['Soccer', 'Football', 'Tennis']))
        0.0
    """
    

    if not chunk or not query:
        score = 0.0
    else:
        score = len(chunk & query) / len(chunk | query)
    return score



def chat_bot(document):

    first = True
    text = input('docchat> ')
    try:
        messages = []
        messages.append({
        'role': 'system',
        'content': 'You are a scientist and always answer questions logically, You always answer in 1-2 sentence.'
        })
        messages.append({
            'role': 'user',
            'content': f'Document:{document}',
        })

        while True:
            # get input from the user
            if not first:
                text = input('docchat> ')
            first = False
            # pass that input to llm
            messages.append({
                'role': 'user',
                'content': text,
            })

            # print('BEFORE RESULT')
            # print("Messages:", messages)
            result = llm(messages)
            
            messages.append({
                'role': 'assistant',
                'content': result,
            })

            # print the llm's response to the user
            print('DOCCHAT:', result)
            import pprint
            # pprint.pprint(messages)
    except groq.APIStatusError:
        messages = []
        messages.append({
            'role': 'system',
            'content': 'You answer questions about the document given to you. You always answer in 1-2 sentence.'
        })

        chunks = chunk_text_by_words(document, 100, 5)
        # print('CHUNKS=', chunks)
        first = True
        while True:
            if not first:
                text = input('docchat> ')
            first = False


            new_doc = find_relevant_chunks(chunks, text, 5)

            # print("Chunks Scored=", chunks_scored)

            
            # print('NEW DOC=', new_doc)
            
            messages.append({
                'role': 'user',
                'content': f'I want to ask you questions about this:{new_doc}',
            })

            messages.append({
                'role': 'user',
                'content': text,
            })

            
            # print("Messages:",messages)
            result = llm(messages)
            print('\nDOCCHAT:', result)

            messages.append({
                'role': 'assistant',
                'content': result,
            })
            
            
def find_relevant_chunks(text, query, num_chunks=5):
    """
    This function will: 
    1) split the document into chunks 
    2) compute the score for each of these chunks
    3) return the `num_chunks` chunks that have the largest score

    >>> text = "The sun is bright and hot. Bananas are yellow. The red car speeds by."
    >>> query = "How hot is the sun?"
    >>> find_relevant_chunks(text, query, num_chunks=1)
    ['The sun is bright and hot. Bananas are yellow. The']
    """
    # Split the text into chunks by words
    chunks = chunk_text_by_words(text, max_words=10, overlap=5)

    # Use accumulator pattern to gather scored chunks
    scored_chunks = []
    for chunk in chunks:
        # Convert chunk and query to sets of words for scoring
        chunk_words = set(chunk.lower().split())
        query_words = set(query.lower().split())
        
        # Compute the score for the chunk
        score = score_chunk(chunk_words, query_words)
        scored_chunks.append((chunk, score))

    # Sort the chunks by score and slice to get the top `num_chunks`
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    top_chunks = [chunk for chunk, _ in scored_chunks[:num_chunks]]

    return top_chunks

if __name__ == '__main__':
    client = groq.Groq()

    parser = argparse.ArgumentParser(
        prog='docsum',
        description='summarize the input document',
        )
    parser.add_argument('filename')

    args = parser.parse_args()


    load_text(args.filename)