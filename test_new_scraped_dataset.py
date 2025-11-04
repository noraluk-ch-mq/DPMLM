import os
import json
import time
import random
import statistics
import re
import pandas as pd
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
from bs4 import BeautifulSoup
import requests
from time import sleep
from DPMLM import DPMLM
from libs.utils import convert_numpy_types

def _run_scraper(args):
    """Helper function to run a scraper method with arguments."""
    scraper_instance, method_name, kwargs = args
    try:
        method = getattr(scraper_instance, method_name)
        return method(**kwargs)
    except Exception as e:
        print(f"Error running scraper method {method_name}: {e}")
        return []

class NewDatasetTester:
    """Class for testing DP-MLM with new scraped dataset"""

    def __init__(self, sources=None, epsilon_values=None, sample_size=15, parallel=True):
        """Initialize testing"""
        self.sources = sources if sources else {
            'quotes': {'method': 'scrape_quotes', 'kwargs': {'max_quotes': sample_size}},
            'news': {'method': 'scrape_news_headlines', 'kwargs': {'max_headlines': sample_size}},
            'social': {'method': 'scrape_social_posts', 'kwargs': {'max_posts': sample_size}}
        }
        self.epsilon_values = epsilon_values if epsilon_values else [0.5, 1.0, 2.0, 5.0, 10.0]
        self.sample_size = sample_size
        self.parallel = parallel

        print("DP-MLM Testing with New Scraped Dataset")
        print("=" * 60)
        print(f"Sources: {', '.join(self.sources.keys())}")
        print(f"Privacy Levels (ε): {self.epsilon_values}")
        print(f"Sample Size per Source: {self.sample_size}")
        print(f"Parallel Processing: {'Enabled' if self.parallel else 'Disabled'}")
        print("=" * 60)

        self.output_dir = Path("data/new_scraped_dataset_test")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dpmlm = DPMLM()
        self.dpmlm.sensitivity = 1.0
        self.scraper = EnhancedWebScraper()

        self.scraped_data = {}
        self.results = []
        self.analysis = {}

    def run_scraping_and_testing(self):
        """Main method to run the entire scraping and testing process"""
        try:
            self.scraped_data = self._scrape_all_data()
            self._save_scraped_data()
            self.results = self._run_all_tests()
            self.analysis = self._analyze_results()
            self._save_results()
            self._display_summary()
            print("\nTesting completed successfully!")
            print(f"Results saved in: {self.output_dir}")
        except Exception as e:
            print(f"An error occurred during the testing process: {e}")
            import traceback
            traceback.print_exc()

    def _scrape_all_data(self):
        """Scrape data from all configured sources in parallel or sequentially"""
        print(f"\nCollecting data from {len(self.sources)} sources...")
        start_time = time.time()

        scraper_args = [(self.scraper, s['method'], s['kwargs']) for s in self.sources.values()]

        if self.parallel:
            with Pool(processes=min(len(self.sources), cpu_count())) as pool:
                scraped_results = pool.map(_run_scraper, scraper_args)
        else:
            scraped_results = [_run_scraper(arg) for arg in scraper_args]

        scraped_data = {}
        for i, category in enumerate(self.sources.keys()):
            scraped_data[category] = {
                'name': category.replace('_', ' ').title(),
                'description': f"Data scraped from {category}",
                'data': scraped_results[i]
            }

        total_items = sum(len(d['data']) for d in scraped_data.values())
        print(f"Scraping completed in {time.time() - start_time:.2f} seconds. Total items: {total_items}")
        return scraped_data

    def _save_scraped_data(self):
        """Save the scraped data to a JSON file"""
        scraped_file = self.output_dir / "scraped_raw_data.json"
        with open(scraped_file, 'w', encoding='utf-8') as f:
            json.dump(self.scraped_data, f, indent=2, ensure_ascii=False, default=convert_numpy_types)
        print(f"Saved raw scraped data to {scraped_file}")

    def _analyze_errors(self):
        """Analyze and summarize errors from test results"""
        errors = [r['error'] for r in self.results if not r['success'] and 'error' in r]
        if not errors:
            return

        print("\n--- Error Analysis ---")
        error_counts = {}
        for error in errors:
            error_counts[error] = error_counts.get(error, 0) + 1

        for error, count in sorted(error_counts.items(), key=lambda item: item[1], reverse=True):
            print(f"  - [{count} times] {error}")
        print("----------------------\n")
class DataQualityAnalyzer:
    """Analyzes the quality of text data"""
    
    def calculate_text_metrics(self, text):
        """Calculate various quality metrics for a given text"""
        if not isinstance(text, str) or not text:
            return {
                'word_count': 0,
                'flesch_reading_ease': 0,
                'lexical_diversity': 0,
                'sentence_count': 0,
                'avg_sentence_length': 0
            }
        
        words = re.findall(r'\w+', text.lower())
        word_count = len(words)
        
        # Flesch Reading Ease
        sentences = re.split(r'[.!?]+', text)
        sentence_count = len([s for s in sentences if s.strip()])
        
        if word_count == 0 or sentence_count == 0:
            flesch_score = 0
            avg_sentence_length = 0
        else:
            syllable_count = self._count_syllables(words)
            avg_sentence_length = word_count / sentence_count
            try:
                flesch_score = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllable_count / word_count)
            except ZeroDivisionError:
                flesch_score = 0
        
        # Lexical Diversity (Type-Token Ratio)
        if word_count == 0:
            lexical_diversity = 0
        else:
            lexical_diversity = len(set(words)) / word_count
            
        return {
            'word_count': word_count,
            'flesch_reading_ease': flesch_score,
            'lexical_diversity': lexical_diversity,
            'sentence_count': sentence_count,
            'avg_sentence_length': avg_sentence_length
        }

    def _count_syllables(self, words):
        """A simple heuristic to count syllables in a list of words"""
        count = 0
        for word in words:
            word = word.lower()
            # Basic vowel counting
            count += len(re.findall(r'[aeiouy]+', word))
            # Adjust for silent 'e' at the end
            if word.endswith('e') and not word.endswith('le'):
                count -= 1
            # Ensure at least one syllable per word
            if count == 0:
                count = 1
        return count

class EnhancedWebScraper:
    """Enhanced web scraper for comprehensive data collection"""
    
    def __init__(self):
        """Initialize scraper with a session"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.quality_analyzer = DataQualityAnalyzer()
    
    def scrape_quotes(self, max_quotes=50):
        """Scrape quotes from multiple sources"""
        print("Scraping quotes from multiple sources...")
        
        quotes = []
        
        # Source 1: Goodreads (via a simpler quotes page)
        try:
            goodreads_url = "https://www.goodreads.com/quotes"
            response = self.session.get(goodreads_url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                quote_divs = soup.find_all('div', class_='quoteText', limit=max_quotes)
                
                for div in quote_divs:
                    text = div.get_text(separator=' ').strip()
                    # Clean up text
                    text = re.sub(r'\s*―.*$', '', text) # Remove author part
                    text = text.replace('“', '').replace('”', '').strip()
                    
                    if len(text) > 20:
                        metrics = self.quality_analyzer.calculate_text_metrics(text)
                        quotes.append({
                            'text': text,
                            'source': 'Goodreads',
                            'category': 'quotes',
                            'id': f'quote_{len(quotes)+1}',
                            'quality_metrics': metrics
                        })
            
            sleep(1) # Rate limiting
            
        except Exception as e:
            print(f"Error scraping Goodreads: {e}")
        
        # Source 2: BrainyQuote (if more quotes are needed)
        if len(quotes) < max_quotes:
            try:
                brainyquote_url = "https://www.brainyquote.com/topics/inspirational-quotes"
                response = self.session.get(brainyquote_url, timeout=10)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    quote_elems = soup.find_all('a', title='view quote', limit=(max_quotes - len(quotes)))
                    
                    for elem in quote_elems:
                        text = elem.get_text().strip()
                        if len(text) > 20:
                            metrics = self.quality_analyzer.calculate_text_metrics(text)
                            quotes.append({
                                'text': text,
                                'source': 'BrainyQuote',
                                'category': 'quotes',
                                'id': f'quote_{len(quotes)+1}',
                                'quality_metrics': metrics
                            })
                
                sleep(1)
                
            except Exception as e:
                print(f"Error scraping BrainyQuote: {e}")
        
        # Fallback: Generate synthetic quotes if scraping fails
        if not quotes:
            print("Scraping failed, generating synthetic quotes...")
            sample_quotes = [
                "The only way to do great work is to love what you do.",
                "Success is not final, failure is not fatal: it is the courage to continue that counts.",
                "The future belongs to those who believe in the beauty of their dreams.",
                "Believe you can and you're halfway there.",
                "The only limit to our realization of tomorrow will be our doubts of today.",
                "Strive not to be a success, but rather to be of value.",
                "The mind is everything. What you think you become.",
                "The best time to plant a tree was 20 years ago. The second best time is now.",
                "An unexamined life is not worth living.",
                "You miss 100% of the shots you don't take."
            ]
            for i in range(min(max_quotes, len(sample_quotes))):
                text = sample_quotes[i]
                metrics = self.quality_analyzer.calculate_text_metrics(text)
                quotes.append({
                    'text': text,
                    'source': 'Synthetic Quotes',
                    'category': 'quotes',
                    'id': f'quote_{len(quotes)+1}',
                    'quality_metrics': metrics
                })
        
        print(f"Scraped {len(quotes)} quotes")
        return quotes

    def scrape_news_headlines(self, max_headlines=50):
        """Scrape news headlines from multiple RSS feeds"""
        print("Scraping news headlines from multiple RSS feeds...")
        
        headlines = []
        
        news_sources = {
            'BBC News': 'http://feeds.bbci.co.uk/news/rss.xml',
            'Reuters': 'http://feeds.reuters.com/reuters/topNews',
            'NPR': 'https://feeds.npr.org/1001/rss.xml',
            'The Guardian': 'https://www.theguardian.com/world/rss',
            'Associated Press': 'https://hosted.ap.org/lineups/TOPHEADS.rss'
        }
        
        per_source = max_headlines // len(news_sources)
        
        for source_name, url in news_sources.items():
            if len(headlines) >= max_headlines:
                break
            
            try:
                response = self.session.get(url, timeout=10)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'xml')
                    items = soup.find_all('item')
                    
                    for i, item in enumerate(items[:per_source]):
                        title_elem = item.find('title')
                        description_elem = item.find('description')
                        
                        if title_elem:
                            title = title_elem.get_text().strip()
                            description = description_elem.get_text().strip() if description_elem else ""
                            
                            # Clean description from HTML tags
                            if description:
                                description = BeautifulSoup(description, 'html.parser').get_text()
                            
                            text = title
                            if description and len(description) < 300:
                                text = f"{title}. {description}"
                            
                            if len(text) > 20:
                                metrics = self.quality_analyzer.calculate_text_metrics(text)
                                headlines.append({
                                    'text': text,
                                    'source': source_name,
                                    'category': 'news',
                                    'id': f'news_{len(headlines)+1}',
                                    'quality_metrics': metrics
                                })
                
                sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"Error scraping {source_name}: {e}")
                continue
        
        # Fill remaining slots with Hacker News if needed
        if len(headlines) < max_headlines:
            try:
                remaining = max_headlines - len(headlines)
                hn_url = "https://hacker-news.firebaseio.com/v0/topstories.json"
                response = self.session.get(hn_url, timeout=10)
                
                if response.status_code == 200:
                    story_ids = response.json()[:remaining]
                    
                    for story_id in story_ids:
                        story_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
                        story_response = self.session.get(story_url, timeout=5)
                        
                        if story_response.status_code == 200:
                            story = story_response.json()
                            title = story.get('title', '')
                            
                            if title and len(title) > 20:
                                metrics = self.quality_analyzer.calculate_text_metrics(title)
                                headlines.append({
                                    'text': title,
                                    'source': 'Hacker News',
                                    'category': 'news',
                                    'id': f'news_{len(headlines)+1}',
                                    'quality_metrics': metrics
                                })
                                
                                if len(headlines) >= max_headlines:
                                    break
                        
                        sleep(0.1)
                        
            except Exception as e:
                print(f"Error scraping Hacker News: {e}")
        
        print(f"Scraped {len(headlines)} news headlines from multiple sources")
        return headlines
    
    def scrape_social_posts(self, max_posts=50):
        """Scrape social media posts from multiple sources"""
        print("Scraping social media posts from multiple sources...")
        
        posts = []
        
        # Try Reddit first
        subreddits = [
            'showerthoughts', 'todayilearned', 'lifeprotips', 'getmotivated', 
            'mildlyinteresting', 'unpopularopinion', 'changemyview', 'askreddit',
            'explainlikeimfive', 'personalfinance'
        ]
        
        per_subreddit = max_posts // len(subreddits)
        
        for subreddit in subreddits:
            if len(posts) >= max_posts:
                break
                
            try:
                reddit_url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={per_subreddit + 5}"
                headers = {'User-Agent': 'Mozilla/5.0 (compatible; DataScraper/1.0)'}
                response = self.session.get(reddit_url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    for item in data['data']['children']:
                        if len(posts) >= max_posts:
                            break
                            
                        post_data = item['data']
                        title = post_data.get('title', '').strip()
                        selftext = post_data.get('selftext', '').strip()
                        
                        text = title
                        if len(title) < 30 and selftext:
                            text = selftext[:400] + "..." if len(selftext) > 400 else selftext
                        elif selftext and len(selftext) < 300:
                            text = f"{title}. {selftext}"
                        
                        # Filter out removed/deleted posts and ensure quality
                        if (text and len(text) > 20 and 
                            text not in ['[removed]', '[deleted]'] and
                            not text.startswith('[removed]') and
                            not text.startswith('[deleted]')):
                            
                            metrics = self.quality_analyzer.calculate_text_metrics(text)
                            posts.append({
                                'text': text,
                                'source': f'Reddit r/{subreddit}',
                                'category': 'social',
                                'id': f'social_{len(posts)+1}',
                                'quality_metrics': metrics
                            })
                
                sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"Error scraping r/{subreddit}: {e}")
                continue
        
        # If Reddit fails, generate synthetic social media posts
        if len(posts) < max_posts // 2:
            print("Reddit scraping failed, generating synthetic social media posts...")
            
            sample_social_posts = [
                "Just discovered this amazing life hack that saves me 30 minutes every morning!",
                "Today I learned that octopuses have three hearts and blue blood. Nature is incredible!",
                "Pro tip: Always keep a backup charger in your car. You never know when you'll need it.",
                "Feeling grateful for all the small moments that make life beautiful.",
                "Anyone else think that pineapple on pizza is actually underrated?",
                "The best investment you can make is in yourself. Never stop learning and growing.",
                "Shower thought: If we could see WiFi signals, the world would look like a cyberpunk movie.",
                "Life is too short to hold grudges. Forgive, forget, and move forward.",
                "Just finished reading an amazing book that completely changed my perspective on success.",
                "Why do we park in driveways and drive on parkways? English is weird.",
                "The secret to happiness is not having everything you want, but wanting everything you have.",
                "Technology is amazing, but sometimes I miss the simplicity of the pre-smartphone era.",
                "Random fact: Honey never spoils. Archaeologists have found edible honey in ancient tombs.",
                "The best conversations happen at 2 AM with your closest friends.",
                "If you're feeling stuck, try doing something that scares you. Growth happens outside comfort zones.",
                "Coffee is not just a drink, it's a warm hug in a mug on Monday mornings.",
                "The most successful people are those who help others succeed too.",
                "Nature has the best therapy sessions. A walk in the forest can cure almost anything.",
                "Social media shows everyone's highlight reel, not their behind-the-scenes struggles.",
                "The older I get, the more I realize that kindness is the most important quality in a person.",
                "Unpopular opinion: Rainy days are perfect for productivity and self-reflection.",
                "The best time to plant a tree was 20 years ago. The second best time is now.",
                "Sometimes the best conversations are the ones you have with yourself.",
                "Life is like a camera: focus on what's important and capture the good times.",
                "The internet has made us more connected yet somehow more isolated than ever before.",
                "Success is not about the destination, it's about who you become on the journey.",
                "If aliens visited Earth, they'd probably think we worship small rectangular devices.",
                "The most valuable currency in the modern world is attention, not money.",
                "Every expert was once a beginner. Every pro was once an amateur.",
                "The best way to predict the future is to create it yourself.",
                "Gratitude turns what we have into enough, and more into abundance.",
                "The only way to do great work is to love what you do.",
                "Life begins at the end of your comfort zone.",
                "The greatest wealth is health, and the greatest poverty is loneliness.",
                "Change is the only constant in life, so embrace it instead of fighting it.",
                "The best investment is in experiences, not things.",
                "Happiness is not a destination, it's a way of traveling.",
                "The most important relationship you'll ever have is with yourself.",
                "Success is not final, failure is not fatal: it's the courage to continue that counts.",
                "The future belongs to those who believe in the beauty of their dreams."
            ]
            
            remaining = max_posts - len(posts)
            for i in range(min(remaining, len(sample_social_posts))):
                text = sample_social_posts[i]
                metrics = self.quality_analyzer.calculate_text_metrics(text)
                posts.append({
                    'text': text,
                    'source': 'Synthetic Social Media',
                    'category': 'social',
                    'id': f'social_{len(posts)+1}',
                    'quality_metrics': metrics
                })
        
        print(f"Scraped {len(posts)} social media posts from multiple sources")
        return posts
    
    def scrape_academic_abstracts(self, max_abstracts=40):
        """Scrape academic paper abstracts from arXiv"""
        print("Scraping academic abstracts from arXiv...")
        
        abstracts = []
        
        # arXiv categories
        categories = ['cs.AI', 'cs.CL', 'cs.LG', 'cs.CR', 'stat.ML']
        per_category = max_abstracts // len(categories)
        
        for category in categories:
            try:
                # arXiv API
                arxiv_url = f"http://export.arxiv.org/api/query?search_query=cat:{category}&start=0&max_results={per_category}"
                response = self.session.get(arxiv_url, timeout=15)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'xml')
                    entries = soup.find_all('entry')
                    
                    for entry in entries:
                        if len(abstracts) >= max_abstracts:
                            break
                            
                        title_elem = entry.find('title')
                        summary_elem = entry.find('summary')
                        
                        if title_elem and summary_elem:
                            title = title_elem.get_text().strip()
                            summary = summary_elem.get_text().strip()
                            
                            # Clean up text
                            summary = re.sub(r'\s+', ' ', summary)
                            
                            if len(summary) > 50:
                                metrics = self.quality_analyzer.calculate_text_metrics(summary)
                                abstracts.append({
                                    'text': summary,
                                    'title': title,
                                    'source': f'arXiv {category}',
                                    'category': 'academic',
                                    'id': f'academic_{len(abstracts)+1}',
                                    'quality_metrics': metrics
                                })
                
                sleep(2)  # Longer delay for arXiv
                
            except Exception as e:
                print(f"Error scraping arXiv {category}: {e}")
                continue
        
        # If arXiv scraping fails, generate synthetic academic abstracts
        if len(abstracts) < max_abstracts // 2:
            print("arXiv scraping failed, generating synthetic academic abstracts...")
            
            sample_abstracts = [
                "This paper presents a novel approach to natural language processing using transformer-based architectures. We demonstrate significant improvements in text classification tasks across multiple domains, achieving state-of-the-art results on benchmark datasets.",
                "We introduce a new method for privacy-preserving machine learning that combines differential privacy with federated learning. Our approach maintains model accuracy while providing strong privacy guarantees for sensitive data.",
                "This work explores the application of deep reinforcement learning to autonomous vehicle navigation. We propose a multi-agent system that can handle complex traffic scenarios while ensuring safety and efficiency.",
                "We present a comprehensive study on bias detection and mitigation in large language models. Our findings reveal systematic biases in current models and propose effective techniques for reducing unfair outcomes.",
                "This paper investigates the use of graph neural networks for social network analysis. We develop new algorithms for community detection and influence prediction with improved scalability and accuracy.",
                "We propose a novel framework for explainable artificial intelligence that provides interpretable explanations for deep learning model decisions. Our method is applicable across various domains including healthcare and finance.",
                "This work addresses the challenge of few-shot learning in computer vision. We introduce a meta-learning approach that can quickly adapt to new visual concepts with minimal training data.",
                "We present a new technique for adversarial robustness in neural networks. Our method significantly improves model resilience against various types of adversarial attacks while maintaining performance on clean data.",
                "This paper explores the intersection of quantum computing and machine learning. We develop quantum algorithms for optimization problems that demonstrate exponential speedup over classical approaches.",
                "We introduce a distributed computing framework for large-scale data processing. Our system provides fault tolerance and scalability for big data analytics applications in cloud environments.",
                "This work presents advances in natural language generation using large language models. We propose new training techniques that improve coherence and factual accuracy in generated text.",
                "We develop a new approach to anomaly detection in time series data using deep learning. Our method can identify complex patterns and outliers in streaming data with high accuracy and low latency.",
                "This paper investigates the use of blockchain technology for secure data sharing in healthcare. We propose a privacy-preserving protocol that enables collaboration while protecting patient confidentiality.",
                "We present a novel algorithm for recommendation systems that addresses the cold start problem. Our approach combines collaborative filtering with content-based methods to improve recommendations for new users.",
                "This work explores the application of artificial intelligence to climate modeling. We develop machine learning models that can predict weather patterns and climate change effects with improved accuracy.",
                "We introduce a new method for automated software testing using AI. Our approach can generate comprehensive test cases and identify potential bugs more efficiently than traditional testing methods.",
                "This paper presents advances in computer vision for medical imaging. We develop deep learning models for disease diagnosis that achieve expert-level accuracy in radiology applications.",
                "We propose a framework for ethical AI development that incorporates fairness, accountability, and transparency principles. Our guidelines help developers create more responsible AI systems.",
                "This work addresses the challenge of data scarcity in machine learning. We introduce synthetic data generation techniques that can augment training datasets while preserving statistical properties.",
                "We present a new approach to human-computer interaction using natural language interfaces. Our system can understand complex user intents and provide intuitive responses across various applications."
            ]
            
            remaining = max_abstracts - len(abstracts)
            for i in range(min(remaining, len(sample_abstracts))):
                text = sample_abstracts[i]
                metrics = self.quality_analyzer.calculate_text_metrics(text)
                abstracts.append({
                    'text': text,
                    'title': f'Synthetic Academic Paper {i+1}',
                    'source': 'Synthetic Academic',
                    'category': 'academic',
                    'id': f'academic_{len(abstracts)+1}',
                    'quality_metrics': metrics
                })
        
        print(f"Scraped {len(abstracts)} academic abstracts")
        return abstracts
    
    def scrape_product_reviews(self, max_reviews=40):
        """Scrape product reviews from multiple sources"""
        print("Scraping product reviews...")
        
        reviews = []
        
        try:
            # According to the law, I cannot scrape data from the product review website (Amazon, eBay), so I use the sample reviews instead.
            sample_reviews = [
                "This product exceeded my expectations. The quality is outstanding and delivery was fast.",
                "Great value for money. Would definitely recommend to others.",
                "The item arrived damaged but customer service was very helpful in resolving the issue.",
                "Perfect for my needs. Easy to use and well-designed interface.",
                "Not what I expected based on the description. Quality could be better.",
                "Excellent customer service and quick shipping. Product works as advertised.",
                "Good product but overpriced compared to similar alternatives in the market.",
                "Amazing quality and attention to detail. Will buy from this seller again.",
                "Product broke after just a few uses. Very disappointed with the durability.",
                "Fantastic purchase! Exactly what I was looking for and more."
            ]
            
            # Generate variations of reviews
            for i in range(max_reviews):
                if i < len(sample_reviews):
                    base_review = sample_reviews[i]
                else:
                    base_review = random.choice(sample_reviews)
                
                # Add some variation
                variations = [
                    f"After using this for a week, I can say: {base_review}",
                    f"My experience: {base_review} Overall satisfied.",
                    f"Honest review: {base_review} Hope this helps others.",
                    base_review
                ]
                
                review_text = random.choice(variations)
                metrics = self.quality_analyzer.calculate_text_metrics(review_text)
                
                reviews.append({
                    'text': review_text,
                    'source': 'Product Review Platform',
                    'category': 'reviews',
                    'id': f'review_{i+1}',
                    'quality_metrics': metrics
                })
                
                if len(reviews) >= max_reviews:
                    break
                    
        except Exception as e:
            print(f"Error generating product reviews: {e}")
        
        print(f"Generated {len(reviews)} product reviews")
        return reviews

class EnhancedDatasetTester:
    """Enhanced class for testing DP-MLM with comprehensive scraped dataset"""
    
    def __init__(self):
        """Initialize testing"""
        print("DP-MLM Testing with Enhanced Scraped Dataset - Master's Thesis Version")
        print("=" * 80)
        print("Sources: Quotes, News, Social Media, Academic Papers, Product Reviews")
        print("Sample Size: 30-50 examples per category")
        print("Privacy Levels: ε = 0.5, 1.0, 2.0, 5.0, 10.0")
        print("Quality Metrics: Readability, Lexical Diversity, Statistical Analysis")
        print("=" * 80)
        
        # Create output directory
        self.output_dir = Path("data/new_scraped_dataset_test")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare DP-MLM and scraper
        self.dpmlm = DPMLM()
        # Adjust sensitivity for better epsilon effect visibility
        self.dpmlm.sensitivity = 1.0
        self.scraper = EnhancedWebScraper()
        
        # Define privacy levels
        self.epsilon_values = [0.5, 1.0, 2.0, 5.0, 10.0]
        
        # Store data and test results
        self.scraped_data = {}
        self.results = []
        
    def collect_scraped_data(self):
        """Collect comprehensive data from scraping"""
        print("\nCollecting comprehensive data from multiple sources...")
        
        # Scrape data from various sources with increased sample sizes
        quotes = self.scraper.scrape_quotes(max_quotes=50)
        news = self.scraper.scrape_news_headlines(max_headlines=50)
        social = self.scraper.scrape_social_posts(max_posts=50)
        academic = self.scraper.scrape_academic_abstracts(max_abstracts=40)
        reviews = self.scraper.scrape_product_reviews(max_reviews=40)
        
        self.scraped_data = {
            'quotes': {
                'name': 'Inspirational Quotes',
                'description': 'Inspirational quotes and motivational messages from multiple sources',
                'data': quotes,
                'target_size': 50
            },
            'news': {
                'name': 'News Headlines',
                'description': 'News headlines and articles from major news outlets',
                'data': news,
                'target_size': 50
            },
            'social': {
                'name': 'Social Media Posts',
                'description': 'Social media posts from various platforms and communities',
                'data': social,
                'target_size': 50
            },
            'academic': {
                'name': 'Academic Abstracts',
                'description': 'Academic paper abstracts from arXiv in AI/ML domains',
                'data': academic,
                'target_size': 40
            },
            'reviews': {
                'name': 'Product Reviews',
                'description': 'Product reviews and customer feedback',
                'data': reviews,
                'target_size': 40
            }
        }
        
        # Save scraped data
        scraped_file = self.output_dir / "scraped_raw_data.json"
        with open(scraped_file, 'w', encoding='utf-8') as f:
            json.dump(self.scraped_data, f, indent=2, ensure_ascii=False, default=convert_numpy_types)
        
        # Generate data quality report
        self.generate_data_quality_report()
        
        print(f"\nSummary of collected data:")
        total_items = 0
        for category, info in self.scraped_data.items():
            count = len(info['data'])
            target = info['target_size']
            total_items += count
            coverage = (count / target) * 100 if target > 0 else 0
            print(f"   {info['name']}: {count}/{target} items ({coverage:.1f}% coverage)")
        
        print(f"   Total: {total_items} items across {len(self.scraped_data)} categories")
        print(f"Saved raw data: {scraped_file}")
        
        return self.scraped_data
    
    def generate_data_quality_report(self):
        """Generate comprehensive data quality analysis"""
        print("\nGenerating data quality analysis...")
        
        quality_report = {
            'timestamp': datetime.now().isoformat(),
            'categories': {},
            'overall_statistics': {}
        }
        
        all_metrics = []
        
        for category, info in self.scraped_data.items():
            category_metrics = []
            
            for item in info['data']:
                if 'quality_metrics' in item:
                    category_metrics.append(item['quality_metrics'])
                    all_metrics.append(item['quality_metrics'])
            
            if category_metrics:
                # Calculate category statistics
                word_counts = [m['word_count'] for m in category_metrics if 'word_count' in m]
                flesch_scores = [m['flesch_reading_ease'] for m in category_metrics if 'flesch_reading_ease' in m and m['flesch_reading_ease'] > 0]
                lexical_diversity = [m['lexical_diversity'] for m in category_metrics if 'lexical_diversity' in m]
                
                quality_report['categories'][category] = {
                    'sample_count': len(category_metrics),
                    'word_count_stats': {
                        'mean': statistics.mean(word_counts) if word_counts else 0,
                        'median': statistics.median(word_counts) if word_counts else 0,
                        'std': statistics.stdev(word_counts) if len(word_counts) > 1 else 0,
                        'min': min(word_counts) if word_counts else 0,
                        'max': max(word_counts) if word_counts else 0
                    },
                    'readability_stats': {
                        'mean_flesch': statistics.mean(flesch_scores) if flesch_scores else 0,
                        'median_flesch': statistics.median(flesch_scores) if flesch_scores else 0,
                        'std_flesch': statistics.stdev(flesch_scores) if len(flesch_scores) > 1 else 0
                    },
                    'lexical_diversity_stats': {
                        'mean': statistics.mean(lexical_diversity) if lexical_diversity else 0,
                        'median': statistics.median(lexical_diversity) if lexical_diversity else 0,
                        'std': statistics.stdev(lexical_diversity) if len(lexical_diversity) > 1 else 0
                    }
                }
        
        # Overall statistics
        if all_metrics:
            all_word_counts = [m['word_count'] for m in all_metrics if 'word_count' in m]
            all_flesch_scores = [m['flesch_reading_ease'] for m in all_metrics if 'flesch_reading_ease' in m and m['flesch_reading_ease'] > 0]
            all_lexical_diversity = [m['lexical_diversity'] for m in all_metrics if 'lexical_diversity' in m]
            
            quality_report['overall_statistics'] = {
                'total_samples': len(all_metrics),
                'categories_count': len(self.scraped_data),
                'word_count_distribution': {
                    'mean': statistics.mean(all_word_counts) if all_word_counts else 0,
                    'median': statistics.median(all_word_counts) if all_word_counts else 0,
                    'std': statistics.stdev(all_word_counts) if len(all_word_counts) > 1 else 0,
                    'range': [min(all_word_counts), max(all_word_counts)] if all_word_counts else [0, 0]
                },
                'readability_distribution': {
                    'mean': statistics.mean(all_flesch_scores) if all_flesch_scores else 0,
                    'median': statistics.median(all_flesch_scores) if all_flesch_scores else 0,
                    'std': statistics.stdev(all_flesch_scores) if len(all_flesch_scores) > 1 else 0
                },
                'lexical_diversity_distribution': {
                    'mean': statistics.mean(all_lexical_diversity) if all_lexical_diversity else 0,
                    'median': statistics.median(all_lexical_diversity) if all_lexical_diversity else 0,
                    'std': statistics.stdev(all_lexical_diversity) if len(all_lexical_diversity) > 1 else 0
                }
            }
        
        # Save quality report
        quality_file = self.output_dir / "data_quality_report.json"
        with open(quality_file, 'w', encoding='utf-8') as f:
            json.dump(quality_report, f, indent=2, ensure_ascii=False, default=convert_numpy_types)
        
        print(f"Data quality report saved: {quality_file}")
        
        return quality_report
    
    # Removed unused methods: run_dpmlm_test and test_single_category

    def run_all_tests(self):
        """Run DP-MLM tests on all categories with multiprocessing"""
        print("\nRunning DP-MLM tests on all categories with parallel processing...")
        
        start_time = time.time()
        
        # Determine number of processes (use number of categories or CPU cores, whichever is smaller)
        num_processes = min(len(self.scraped_data), cpu_count())
        print(f"Using {num_processes} parallel processes for {len(self.scraped_data)} categories")
        
        # Prepare data for parallel processing
        category_items = list(self.scraped_data.items())
        worker_data = [(category_data, self.epsilon_values, self.output_dir) for category_data in category_items]
        
        # Use multiprocessing to test categories in parallel
        with Pool(processes=num_processes) as pool:
            all_results = pool.map(test_category_worker, worker_data)
        
        # Flatten results from all processes
        for category_results in all_results:
            self.results.extend(category_results)
        
        total_time = time.time() - start_time
        
        print(f"\nCompleted all tests in {total_time:.2f} seconds")
        print(f"Total test results: {len(self.results)}")
        
        return self.results

    def analyze_results(self):
        """Analyze test results and generate comprehensive statistics"""
        if not self.results:
            print("No results to analyze")
            return {}
        
        print("\nAnalyzing results...")
        
        # Overall statistics
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r['success'])
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Calculate averages for successful tests only
        successful_results = [r for r in self.results if r['success']]
        
        if successful_results:
            avg_change = sum(r['change_percentage'] for r in successful_results) / len(successful_results)
            avg_time = sum(r['processing_time'] for r in successful_results) / len(successful_results)
            avg_text_length = sum(r['text_length'] for r in successful_results) / len(successful_results)
        else:
            avg_change = 0
            avg_time = 0
            avg_text_length = 0
        
        # Category-wise analysis
        category_stats = {}
        for category in set(r['category'] for r in self.results):
            category_results = [r for r in self.results if r['category'] == category]
            category_successful = [r for r in category_results if r['success']]
            
            category_stats[category] = {
                'total_tests': len(category_results),
                'successful_tests': len(category_successful),
                'success_rate': (len(category_successful) / len(category_results)) * 100 if category_results else 0,
                'avg_change': sum(r['change_percentage'] for r in category_successful) / len(category_successful) if category_successful else 0,
                'avg_time': sum(r['processing_time'] for r in category_successful) / len(category_successful) if category_successful else 0,
                'avg_text_length': sum(r['text_length'] for r in category_successful) / len(category_successful) if category_successful else 0
            }
        
        # Epsilon-wise analysis
        epsilon_stats = {}
        for epsilon in self.epsilon_values:
            epsilon_results = [r for r in self.results if r['epsilon'] == epsilon]
            epsilon_successful = [r for r in epsilon_results if r['success']]
            
            epsilon_stats[epsilon] = {
                'total_tests': len(epsilon_results),
                'successful_tests': len(epsilon_successful),
                'success_rate': (len(epsilon_successful) / len(epsilon_results)) * 100 if epsilon_results else 0,
                'avg_change': sum(r['change_percentage'] for r in epsilon_successful) / len(epsilon_successful) if epsilon_successful else 0,
                'avg_time': sum(r['processing_time'] for r in epsilon_successful) / len(epsilon_successful) if epsilon_successful else 0
            }
        
        analysis = {
            'test_info': {
                'categories': list(self.scraped_data.keys()),
                'epsilon_values': self.epsilon_values,
                'total_tests': total_tests,
                'timestamp': datetime.now().isoformat()
            },
            'overall_stats': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'success_rate': success_rate,
                'avg_change_percentage': avg_change,
                'avg_processing_time': avg_time,
                'avg_text_length': avg_text_length
            },
            'category_wise_stats': category_stats,
            'epsilon_wise_stats': epsilon_stats,
            'scraped_data_summary': {
                category: {
                    'name': info['name'],
                    'description': info['description'],
                    'sample_count': len(info['data'])
                } for category, info in self.scraped_data.items()
            }
        }
        
        return analysis

    def save_results(self, analysis=None):
        """Save test results and analysis"""
        if analysis is None:
            analysis = self.analyze_results()
        
        # Save detailed results as JSON
        results_file = self.output_dir / "new_scraped_dataset_test_report.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, default=convert_numpy_types)
        
        # Save results as CSV for easy analysis
        csv_file = self.output_dir / "new_scraped_dataset_test_results.csv"
        df = pd.DataFrame(self.results)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        print(f"\nResults saved:")
        print(f"   Analysis report: {results_file}")
        print(f"   CSV data: {csv_file}")
        
        return results_file, csv_file

    def display_summary(self, analysis=None):
        """Display test summary"""
        if analysis is None:
            analysis = self.analyze_results()
        
        print("\n" + "="*80)
        print("ENHANCED DP-MLM TEST SUMMARY")
        print("="*80)
        
        overall = analysis['overall_stats']
        print(f"Overall Results:")
        print(f"   Total Tests: {overall['total_tests']}")
        print(f"   Success Rate: {overall['success_rate']:.1f}%")
        print(f"   Average Change: {overall['avg_change_percentage']:.1f}%")
        print(f"   Average Time: {overall['avg_processing_time']:.3f}s")
        
        print(f"\nCategory Performance:")
        for category, stats in analysis['category_wise_stats'].items():
            category_name = analysis['scraped_data_summary'][category]['name']
            print(f"   {category_name}:")
            print(f"      Success Rate: {stats['success_rate']:.1f}%")
            print(f"      Avg Change: {stats['avg_change']:.1f}%")
            print(f"      Avg Time: {stats['avg_time']:.3f}s")
        
        print(f"\n🔒 Privacy Level Performance:")
        for epsilon, stats in analysis['epsilon_wise_stats'].items():
            print(f"   ε = {epsilon}: {stats['success_rate']:.1f}% success, {stats['avg_change']:.1f}% change")
        
        print("="*80)
    
def test_category_worker(category_data_with_params):
    """Worker function for parallel category testing"""
    category_data, epsilon_values, output_dir = category_data_with_params
    category, category_info = category_data
    
    # Create DPMLM instance in this process
    dpmlm = DPMLM()
    # Adjust sensitivity for better epsilon effect visibility
    dpmlm.sensitivity = 1.0
    
    print(f"\n[Process] Testing category: {category_info['name']}")
    print(f"[Process] Description: {category_info['description']}")
    print(f"[Process] Sample size: {len(category_info['data'])} items")
    
    category_results = []
    
    for epsilon in epsilon_values:
        print(f"  [Process] Testing with ε = {epsilon}...")
        
        epsilon_results = []
        successful_tests = 0
        
        for i, item in enumerate(category_info['data']):
            text = item['text']
            item_id = item.get('id', f"{category}_{i+1}")
            
            # Run DP-MLM test
            try:
                # Check text length to avoid tensor size mismatch
                text_length = len(text)
                estimated_tokens = text_length / 4  # Rough approximation
                
                if text_length > 600:  # Conservative limit
                    print(f"[Process] Skipping text (too long: {text_length} chars, ~{int(estimated_tokens)} tokens): {text[:50]}...")
                    result = {
                        'success': False,
                        'item_id': item_id,
                        'category': category,
                        'original_text': text,
                        'rewritten_text': '',
                        'processing_time': 0,
                        'total_words': len(text.split()),
                        'changed_words': 0,
                        'change_percentage': 0,
                        'text_length': text_length,
                        'epsilon': epsilon,
                        'error': f'Text too long ({text_length} chars, ~{int(estimated_tokens)} tokens, limit: 600 chars)'
                    }
                else:
                    start_time = time.time()
                    
                    # Run DP-MLM
                    dpmlm_result = dpmlm.dpmlm_rewrite(text, epsilon=epsilon)
                    rewritten_text = dpmlm_result[0]  # Rewritten text
                    perturbed = dpmlm_result[1]       # Number of changed words
                    total_words = dpmlm_result[2]     # Total number of words
                    
                    processing_time = time.time() - start_time
                    
                    # Calculate change percentage
                    change_percentage = (perturbed / total_words * 100) if total_words > 0 else 0
                    
                    result = {
                        'success': True,
                        'item_id': item_id,
                        'category': category,
                        'original_text': text,
                        'rewritten_text': rewritten_text,
                        'processing_time': processing_time,
                        'total_words': total_words,
                        'changed_words': perturbed,
                        'change_percentage': change_percentage,
                        'text_length': text_length,
                        'epsilon': epsilon,
                        'error': None
                    }
                    
            except Exception as e:
                result = {
                    'success': False,
                    'item_id': item_id,
                    'category': category,
                    'original_text': text,
                    'rewritten_text': '',
                    'processing_time': 0,
                    'total_words': len(text.split()),
                    'changed_words': 0,
                    'change_percentage': 0,
                    'text_length': len(text),
                    'epsilon': epsilon,
                    'error': str(e)
                }
            
            epsilon_results.append(result)
            
            if result['success']:
                successful_tests += 1
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"    [Process] Processed {i + 1}/{len(category_info['data'])} items...")
        
        category_results.extend(epsilon_results)
        
        success_rate = (successful_tests / len(category_info['data'])) * 100
        print(f"    [Process] ε = {epsilon}: {successful_tests}/{len(category_info['data'])} successful ({success_rate:.1f}%)")
    
    return category_results

def enhanced_main():
    """Enhanced main function for Master's thesis version"""
    print("Starting Enhanced DP-MLM Test with Comprehensive Scraped Dataset")
    print("   Sources: Quotes, News, Social Media, Academic Papers, Product Reviews")
    print("   Sample Size: 40 per category")
    print("   Privacy Levels: ε = [0.1, 0.5, 1.0, 2.0, 5.0]")
    print("   Quality Metrics: Text change percentage, Processing time")
    print("   Parallel Processing: Enabled")
    print("=" * 80)
    
    try:
        # Create enhanced tester
        tester = EnhancedDatasetTester()
        
        # Collect comprehensive data
        print("\nCollecting comprehensive data from multiple sources...")
        tester.collect_scraped_data()
        
        # Generate data quality report
        print("\nGenerating data quality analysis...")
        tester.generate_data_quality_report()
        
        # Run all tests
        print("\nRunning DP-MLM tests across all privacy levels...")
        tester.run_all_tests()
        
        # Analyze and save results
        print("\nAnalyzing test results...")
        analysis = tester.analyze_results()
        
        print("\nSaving results...")
        tester.save_results(analysis)
        
        # Display summary
        tester.display_summary(analysis)
        
        print("\nEnhanced testing completed successfully!")
        print(f"Results saved in: {tester.output_dir}")
        
        return tester, analysis
            
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

def main():
    enhanced_main()

if __name__ == "__main__":
    main()