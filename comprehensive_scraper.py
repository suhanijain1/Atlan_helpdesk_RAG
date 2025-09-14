"""
Comprehensive scraper for all key Atlan documentation areas
"""

import requests
from bs4 import BeautifulSoup
import json
import time
from urllib.parse import urljoin, urlparse

def scrape_comprehensive_documentation():
    """Scrape all essential Atlan documentation"""
    
    # Complete URL list organized by topic
    documentation_urls = {
        'SSO': [
            'https://docs.atlan.com/product/integrations/identity-management/sso',
            'https://docs.atlan.com/product/integrations/identity-management/sso/how-tos/enable-azure-ad-for-sso',
            'https://docs.atlan.com/product/integrations/identity-management/sso/how-tos/enable-google-for-sso',
            'https://docs.atlan.com/product/integrations/identity-management/sso/how-tos/enable-okta-for-sso',
            'https://docs.atlan.com/product/integrations/identity-management/sso/how-tos/enable-onelogin-for-sso',
            'https://docs.atlan.com/product/integrations/identity-management/sso/how-tos/enable-saml-2-0-for-sso'
        ],
        'API/SDK': [
            'https://developer.atlan.com/getting-started/',
            'https://developer.atlan.com/sdks/',
            'https://docs.atlan.com/get-started/how-tos/getting-started-with-the-apis',
            'https://developer.atlan.com/reference/',
            'https://developer.atlan.com/concepts/',
            'https://developer.atlan.com/patterns/',
            'https://developer.atlan.com/toolkits/'
        ],
        'Product': [
            'https://docs.atlan.com/product/',
            'https://docs.atlan.com/product/capabilities/discovery',
            'https://docs.atlan.com/product/capabilities/governance',
            'https://docs.atlan.com/product/capabilities/insights',
            'https://docs.atlan.com/product/connections',
            'https://docs.atlan.com/product/integrations',
            'https://docs.atlan.com/get-started/what-is-atlan'
        ],
        'How-to': [
            'https://docs.atlan.com/get-started/how-tos/quick-start-for-admins',
            'https://docs.atlan.com/apps/connectors/',
            'https://docs.atlan.com/product/administration',
            'https://docs.atlan.com/get-started/',
            'https://docs.atlan.com/apps/connectors/data-warehouses/snowflake/how-tos/set-up-snowflake',
            'https://docs.atlan.com/apps/connectors/data-warehouses/databricks/how-tos/set-up-databricks'
        ],
        'Best Practices': [
            'https://docs.atlan.com/product/capabilities/governance',
            'https://docs.atlan.com/product/administration',
            'https://docs.atlan.com/support/references/customer-support',
            'https://solutions.atlan.com/overview/'
        ]
    }
    
    all_documents = []
    
    print("🕷️ COMPREHENSIVE ATLAN DOCUMENTATION SCRAPER")
    print("="*70)
    
    for topic, urls in documentation_urls.items():
        print(f"\n📂 SCRAPING {topic.upper()} DOCUMENTATION")
        print("-" * 50)
        
        topic_docs = 0
        
        for url in urls:
            print(f"🔍 {url}")
            
            try:
                response = requests.get(url, timeout=15)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract title
                    title = soup.find('title')
                    title = title.text.strip() if title else url.split('/')[-1]
                    
                    # Extract main content using multiple selectors
                    content = ""
                    content_selectors = [
                        'main',
                        '[role="main"]',
                        '.markdown',
                        'article',
                        '.theme-doc-markdown',
                        '.docusaurus-content',
                        '.content'
                    ]
                    
                    for selector in content_selectors:
                        main_content = soup.select_one(selector)
                        if main_content:
                            # Remove unwanted elements
                            for unwanted in main_content.select(
                                'nav, header, footer, .navbar, .sidebar, '
                                '.pagination, .breadcrumbs, .toc, '
                                '.edit-this-page, .feedback'
                            ):
                                unwanted.decompose()
                            
                            content = main_content.get_text(separator='\n', strip=True)
                            break
                    
                    # Fallback to body if no main content found
                    if not content:
                        body = soup.find('body')
                        if body:
                            content = body.get_text(separator='\n', strip=True)
                    
                    # Clean up content
                    lines = content.split('\n')
                    cleaned_lines = []
                    for line in lines:
                        line = line.strip()
                        if len(line) > 3 and not line.startswith('Skip to'):  # Filter noise
                            cleaned_lines.append(line)
                    
                    content = '\n'.join(cleaned_lines)
                    
                    if content and len(content) > 500:  # Only save substantial content
                        all_documents.append({
                            'url': url,
                            'title': title,
                            'content': content[:30000],  # Reasonable limit
                            'topic': topic,
                            'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S')
                        })
                        topic_docs += 1
                        print(f"   ✅ SUCCESS: {len(content)} chars - {title[:60]}...")
                    else:
                        print(f"   ⚠️  Minimal content: {len(content)} chars")
                
                elif response.status_code == 404:
                    print(f"   ❌ NOT FOUND: 404")
                else:
                    print(f"   ❌ ERROR: HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ EXCEPTION: {e}")
            
            time.sleep(1)  # Be respectful
        
        print(f"\n   📊 {topic}: Successfully scraped {topic_docs} documents")
    
    return all_documents

def explore_and_scrape_additional_content(base_documents):
    """Find and scrape additional related content"""
    
    print(f"\n🔍 EXPLORING FOR ADDITIONAL CONTENT")
    print("="*50)
    
    additional_urls = set()
    
    # Look for links in the scraped content
    for doc in base_documents:
        try:
            soup = BeautifulSoup(doc['content'], 'html.parser')
            # This won't find links since content is text, but we can look for URL patterns
            content = doc['content']
            
            # Look for common documentation patterns
            if 'docs.atlan.com' in content:
                # Extract potential URLs (basic pattern matching)
                import re
                url_pattern = r'https://docs\.atlan\.com/[^\s\)\]\}"]+'
                found_urls = re.findall(url_pattern, content)
                for url in found_urls:
                    additional_urls.add(url)
                    
        except Exception as e:
            continue
    
    print(f"📋 Found {len(additional_urls)} additional URLs to explore")
    
    # Scrape a few high-value additional URLs
    priority_additional = [
        'https://docs.atlan.com/concepts/',
        'https://docs.atlan.com/platform/',
        'https://docs.atlan.com/product/capabilities/',
        'https://developer.atlan.com/snippets/',
        'https://docs.atlan.com/support/'
    ]
    
    additional_docs = []
    for url in priority_additional:
        if url not in [doc['url'] for doc in base_documents]:
            print(f"🔍 Additional: {url}")
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    title = soup.find('title')
                    title = title.text.strip() if title else 'Additional Content'
                    
                    main_content = soup.select_one('main, [role="main"], .markdown')
                    if main_content:
                        content = main_content.get_text(separator='\n', strip=True)
                        if len(content) > 500:
                            additional_docs.append({
                                'url': url,
                                'title': title,
                                'content': content[:25000],
                                'topic': 'Additional',
                                'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S')
                            })
                            print(f"   ✅ Added: {len(content)} chars")
                        else:
                            print(f"   ⚠️  Minimal: {len(content)} chars")
                    else:
                        print(f"   ❌ No content found")
                else:
                    print(f"   ❌ HTTP {response.status_code}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
            time.sleep(1)
    
    return additional_docs

def create_final_knowledge_base():
    """Create the final comprehensive knowledge base"""
    
    print("🚀 CREATING COMPREHENSIVE KNOWLEDGE BASE")
    print("="*70)
    
    # Scrape all core documentation
    core_docs = scrape_comprehensive_documentation()
    
    # Explore for additional content
    additional_docs = explore_and_scrape_additional_content(core_docs)
    
    # Combine all documents
    all_docs = core_docs + additional_docs
    
    # Remove duplicates by URL
    seen_urls = set()
    final_docs = []
    for doc in all_docs:
        if doc['url'] not in seen_urls:
            seen_urls.add(doc['url'])
            final_docs.append(doc)
    
    # Save comprehensive knowledge base
    with open('atlan_knowledge_base.json', 'w') as f:
        json.dump(final_docs, f, indent=2)
    
    print(f"\n📊 FINAL KNOWLEDGE BASE STATISTICS")
    print("="*50)
    print(f"Total documents: {len(final_docs)}")
    
    # Topic breakdown
    topic_counts = {}
    for doc in final_docs:
        topic = doc.get('topic', 'Unknown')
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    for topic, count in sorted(topic_counts.items()):
        print(f"{topic:15}: {count:3d} documents")
    
    # Size statistics
    sizes = [len(doc['content']) for doc in final_docs]
    avg_size = sum(sizes) // len(sizes) if sizes else 0
    total_size = sum(sizes)
    
    print(f"\nContent statistics:")
    print(f"Total content: {total_size:,} characters")
    print(f"Average size: {avg_size:,} characters per document")
    print(f"Largest doc: {max(sizes):,} characters")
    
    print(f"\n💾 Saved to: atlan_knowledge_base.json")
    print(f"🎯 Ready for testing with comprehensive coverage!")
    
    return final_docs

if __name__ == "__main__":
    create_final_knowledge_base()