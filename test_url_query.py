import requests
import json
import sys

def test_create_and_query(url, query):
    """
    Test the create-and-query endpoint with a URL and query.
    
    Args:
        url: The URL to test with
        query: The query to test with
    """
    api_url = "http://localhost:8000/create-and-query"
    
    payload = {
        "file_path": url,
        "is_url": True,
        "query": query
    }
    
    print(f"Testing with URL: {url}")
    print(f"Query: {query}")
    response = requests.post(api_url, json=payload)
    
    if response.status_code == 200:
        print("Success!")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    if len(sys.argv) > 2:
        url = sys.argv[1]
        query = sys.argv[2]
    else:
        url = "https://www.gutenberg.org/cache/epub/1228/pg1228.txt"
        query = "What is natural selection?"
    
    test_create_and_query(url, query) 