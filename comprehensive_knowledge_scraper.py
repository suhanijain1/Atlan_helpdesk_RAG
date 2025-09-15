#!/usr/bin/env python3
"""
Comprehensive Knowledge Base Scraper for Atlan RAG System

Targets specific topics that need RAG responses:
- HOW_TO questions
- PRODUCT questions  
- BEST_PRACTICES questions
- API_SDK questions
- SSO questions

Scrapes from:
- https://docs.atlan.com/ (Product, How-to, Best Practices, SSO)
- https://developer.atlan.com/ (API/SDK, Authentication)
"""

import requests
import json
import time
import logging
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from datetime import datetime
import os
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveAtlanScraper:
    def __init__(self, delay=1.0):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.delay = delay
        self.scraped_urls = set()
        self.knowledge_base = []
        
    def extract_content(self, url):
        """Extract clean content from a documentation page"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove navigation, headers, footers, ads
            for element in soup.find_all(['nav', 'header', 'footer', 'aside', 'script', 'style']):
                element.decompose()
            
            # Try different content selectors for docs.atlan.com and developer.atlan.com
            content_selectors = [
                'main',
                '.content',
                '.documentation',
                'article',
                '.main-content',
                '.markdown-body',
                '.post-content',
                '#content',
                '.doc-content'
            ]
            
            content = None
            title = soup.title.string if soup.title else 'No title'
            
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    content = content_elem.get_text(strip=True, separator=' ')
                    break
            
            if not content:
                content = soup.get_text(strip=True, separator=' ')
            
            # Clean up content
            content = ' '.join(content.split())
            
            return {
                'title': title.strip(),
                'content': content,
                'url': url,
                'scraped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'content_length': len(content)
            }
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return None

    def classify_topic(self, title, content, url):
        """Classify document into RAG-relevant topics"""
        title_lower = title.lower()
        content_lower = content.lower()
        url_lower = url.lower()
        
        # API/SDK classification
        api_sdk_keywords = [
            'api', 'sdk', 'authentication', 'token', 'oauth', 'rest', 'endpoint',
            'python', 'java', 'kotlin', 'go', 'cli', 'developer', 'integration',
            'webhook', 'event', 'script', 'automation', 'programmatic'
        ]
        
        # HOW_TO classification  
        how_to_keywords = [
            'how to', 'step by step', 'tutorial', 'guide', 'setup', 'configure',
            'install', 'create', 'manage', 'enable', 'connect', 'integrate'
        ]
        
        # PRODUCT classification
        product_keywords = [
            'feature', 'capability', 'overview', 'introduction', 'about',
            'lineage', 'catalog', 'governance', 'discovery', 'metadata',
            'glossary', 'classification', 'quality', 'profiling'
        ]
        
        # BEST_PRACTICES classification
        best_practices_keywords = [
            'best practice', 'recommendation', 'guideline', 'tip', 'optimization',
            'performance', 'security', 'workflow', 'strategy', 'pattern'
        ]
        
        # SSO classification
        sso_keywords = [
            'sso', 'single sign', 'saml', 'okta', 'azure ad', 'google',
            'identity', 'ldap', 'active directory', 'federation'
        ]
        
        # Check for matches
        text_to_check = f"{title_lower} {content_lower} {url_lower}"
        
        if any(keyword in text_to_check for keyword in sso_keywords):
            return 'SSO'
        elif any(keyword in text_to_check for keyword in api_sdk_keywords):
            return 'API_SDK'
        elif any(keyword in text_to_check for keyword in how_to_keywords):
            return 'HOW_TO'
        elif any(keyword in text_to_check for keyword in best_practices_keywords):
            return 'BEST_PRACTICES'
        elif any(keyword in text_to_check for keyword in product_keywords):
            return 'PRODUCT'
        else:
            return 'OTHER'

    def get_priority_urls(self):
        """Define high-priority URLs for each RAG topic"""
        
        # Base URLs to explore
        base_urls = {
            'docs': 'https://docs.atlan.com',
            'developer': 'https://developer.atlan.com'
        }
        
        # High-priority specific URLs
        priority_urls = [
            # API/SDK Priority URLs
            'https://developer.atlan.com/sdks/',
            'https://developer.atlan.com/sdks/java/',
            'https://developer.atlan.com/sdks/python/',
            'https://developer.atlan.com/sdks/kotlin/',
            'https://developer.atlan.com/sdks/go/',
            'https://developer.atlan.com/snippets/access/tokens/',
            'https://developer.atlan.com/api/',
            'https://developer.atlan.com/events/',
            'https://developer.atlan.com/packages/',
            'https://developer.atlan.com/cli/',
            
            # Authentication & Security
            'https://docs.atlan.com/product/integrations/identity-management/sso/',
            'https://docs.atlan.com/product/integrations/identity-management/sso/how-tos/',
            'https://docs.atlan.com/product/integrations/identity-management/api-tokens/',
            
            # Product Features
            'https://docs.atlan.com/product/discovery/',
            'https://docs.atlan.com/product/lineage/',
            'https://docs.atlan.com/product/governance/',
            'https://docs.atlan.com/product/catalog/',
            'https://docs.atlan.com/product/insights/',
            'https://docs.atlan.com/product/profiling/',
            
            # How-to Guides
            'https://docs.atlan.com/getting-started/',
            'https://docs.atlan.com/tutorials/',
            'https://docs.atlan.com/guide/',
            
            # Connectors & Integrations
            'https://docs.atlan.com/apps/connectors/',
            'https://docs.atlan.com/apps/connectors/data-warehouses/',
            'https://docs.atlan.com/apps/connectors/bi-tools/',
            'https://docs.atlan.com/apps/connectors/object-stores/',
            
            # Best Practices
            'https://docs.atlan.com/best-practices/',
            'https://docs.atlan.com/performance/',
            'https://docs.atlan.com/security/',
        ]
        
        return priority_urls

    def discover_urls_from_sitemap(self, base_url):
        """Try to discover URLs from sitemap"""
        sitemap_urls = [
            f"{base_url}/sitemap.xml",
            f"{base_url}/sitemap_index.xml",
            f"{base_url}/robots.txt"
        ]
        
        discovered_urls = []
        
        for sitemap_url in sitemap_urls:
            try:
                response = self.session.get(sitemap_url, timeout=10)
                if response.status_code == 200:
                    if 'sitemap.xml' in sitemap_url:
                        soup = BeautifulSoup(response.content, 'xml')
                        urls = [url.get_text() for url in soup.find_all('loc')]
                        discovered_urls.extend(urls)
                        logger.info(f"Found {len(urls)} URLs in {sitemap_url}")
                    elif 'robots.txt' in sitemap_url:
                        lines = response.text.split('\n')
                        for line in lines:
                            if line.startswith('Sitemap:'):
                                sitemap_url = line.split(':', 1)[1].strip()
                                discovered_urls.append(sitemap_url)
            except Exception as e:
                logger.debug(f"Could not access {sitemap_url}: {e}")
        
        return discovered_urls[:500]  # Limit to prevent overwhelming

    def scrape_comprehensively(self, max_docs_per_topic=50):
        """Scrape comprehensively for all RAG topics"""
        
        # Get priority URLs
        priority_urls = self.get_priority_urls()
        
        # Try to discover more URLs from sitemaps
        discovered_urls = []
        for base_url in ['https://docs.atlan.com', 'https://developer.atlan.com']:
            discovered_urls.extend(self.discover_urls_from_sitemap(base_url))
        
        # Combine and deduplicate URLs
        all_urls = list(set(priority_urls + discovered_urls))
        
        logger.info(f"Starting comprehensive scraping of {len(all_urls)} URLs")
        
        topic_counts = defaultdict(int)
        successful_scrapes = 0
        
        for i, url in enumerate(all_urls):
            if url in self.scraped_urls:
                continue
                
            logger.info(f"Scraping ({i+1}/{len(all_urls)}): {url}")
            
            # Extract content
            doc_data = self.extract_content(url)
            if not doc_data or len(doc_data['content']) < 200:
                continue
            
            # Classify topic
            topic = self.classify_topic(doc_data['title'], doc_data['content'], url)
            
            # Check if we need more docs for this topic
            if topic_counts[topic] >= max_docs_per_topic:
                logger.debug(f"Skipping {url} - already have enough {topic} docs")
                continue
            
            # Add topic to document
            doc_data['topic'] = topic
            
            # Add to knowledge base
            self.knowledge_base.append(doc_data)
            self.scraped_urls.add(url)
            topic_counts[topic] += 1
            successful_scrapes += 1
            
            logger.info(f"✅ Scraped {topic} doc: {doc_data['title'][:60]}...")
            
            # Rate limiting
            time.sleep(self.delay)
            
            # Progress report every 10 docs
            if successful_scrapes % 10 == 0:
                logger.info(f"Progress: {successful_scrapes} docs scraped")
                for topic, count in topic_counts.items():
                    logger.info(f"  {topic}: {count} docs")
        
        logger.info(f"Comprehensive scraping complete!")
        logger.info(f"Total documents scraped: {successful_scrapes}")
        for topic, count in topic_counts.items():
            logger.info(f"  {topic}: {count} documents")
        
        return self.knowledge_base

    def save_knowledge_base(self, filename='atlan_comprehensive_knowledge_base.json'):
        """Save the knowledge base to file"""
        # Load existing knowledge base to avoid duplicates
        existing_kb = []
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    existing_kb = json.load(f)
                logger.info(f"Loaded {len(existing_kb)} existing documents")
            except Exception as e:
                logger.warning(f"Could not load existing knowledge base: {e}")
        
        # Combine and deduplicate by URL
        existing_urls = {doc['url'] for doc in existing_kb}
        new_docs = [doc for doc in self.knowledge_base if doc['url'] not in existing_urls]
        
        combined_kb = existing_kb + new_docs
        
        # Save combined knowledge base
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(combined_kb, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(combined_kb)} total documents to {filename}")
        logger.info(f"Added {len(new_docs)} new documents")
        
        # Print final topic distribution
        topic_counts = defaultdict(int)
        for doc in combined_kb:
            topic_counts[doc.get('topic', 'unknown')] += 1
        
        logger.info("Final topic distribution:")
        for topic, count in sorted(topic_counts.items()):
            logger.info(f"  {topic}: {count} documents")
        
        return combined_kb

def main():
    """Main scraping function"""
    logger.info("🚀 Starting Comprehensive Atlan Knowledge Base Scraping")
    logger.info("Targeting RAG topics: HOW_TO, PRODUCT, BEST_PRACTICES, API_SDK, SSO")
    
    scraper = ComprehensiveAtlanScraper(delay=1.0)  # 1 second delay between requests
    
    try:
        # Scrape comprehensively
        knowledge_base = scraper.scrape_comprehensively(max_docs_per_topic=75)
        
        # Save to file
        final_kb = scraper.save_knowledge_base('atlan_knowledge_base.json')
        
        logger.info("✅ Comprehensive scraping completed successfully!")
        
        return final_kb
        
    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
        if scraper.knowledge_base:
            logger.info("Saving partial results...")
            scraper.save_knowledge_base('atlan_knowledge_base_partial.json')
    except Exception as e:
        logger.error(f"Error during scraping: {e}")
        if scraper.knowledge_base:
            logger.info("Saving partial results...")
            scraper.save_knowledge_base('atlan_knowledge_base_partial.json')

if __name__ == "__main__":
    main()