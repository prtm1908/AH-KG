import requests
import json
import sys

def test_create_knowledge_graph(url):
    """
    Test the create-knowledge-graph endpoint with a URL.
    
    Args:
        url: The URL to test with
    """
    api_url = "http://localhost:8000/create-knowledge-graph"
    
    payload = {
        "file_path": url,
        "is_url": True
    }
    
    print(f"Testing with URL: {url}")
    response = requests.post(api_url, json=payload)
    
    if response.status_code == 200:
        print("Success!")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = "https://www.gutenberg.org/cache/epub/1228/pg1228.txt"
    
    test_create_knowledge_graph(url) 