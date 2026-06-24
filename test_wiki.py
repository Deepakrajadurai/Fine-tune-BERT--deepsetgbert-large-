import urllib.request
import re
import html

def get_wiki_paragraphs(title, num_paragraphs=20):
    url = f"https://de.wikipedia.org/wiki/{urllib.parse.quote(title)}"
    print(f"Fetching Wikipedia page: {url}")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            html_content = response.read().decode('utf-8')
        
        # Regex to handle any <p> tag with or without attributes
        paragraphs = re.findall(r'<p\b[^>]*>(.*?)</p>', html_content, re.DOTALL)
        cleaned_paragraphs = []
        for p in paragraphs:
            # Strip HTML tags
            p_clean = re.sub(r'<.*?>', '', p)
            p_clean = html.unescape(p_clean)
            p_clean = re.sub(r'\[\d+\]', '', p_clean) # Remove citations
            p_clean = re.sub(r'\s+', ' ', p_clean).strip()
            # Ignore paragraphs with lots of vertical bars (likely tables/lists)
            if p_clean.count('|') > 2:
                continue
            if len(p_clean.split()) >= 30:
                cleaned_paragraphs.append(p_clean)
                if len(cleaned_paragraphs) >= num_paragraphs:
                    break
        return cleaned_paragraphs
    except Exception as e:
        print(f"Error fetching Wikipedia title '{title}':", e)
        return []

paragraphs = get_wiki_paragraphs("Klimawandel", 5)
print(f"Fetched {len(paragraphs)} paragraphs.")
for i, p in enumerate(paragraphs):
    print(f"P{i+1}: {p[:150]}...")
