import requests
from bs4 import BeautifulSoup
import re
from time import sleep
from textstat import flesch_reading_ease, flesch_kincaid_grade

class DataQualityAnalyzer:
    """Class for analyzing data quality metrics"""
    
    @staticmethod
    def calculate_text_metrics(text):
        """Calculate various text quality metrics"""
        try:
            word_count = len(text.split())
            char_count = len(text)
            sentence_count = len(re.findall(r'[.!?]+', text))
            
            # Readability scores
            try:
                flesch_score = flesch_reading_ease(text)
                fk_grade = flesch_kincaid_grade(text)
            except:
                flesch_score = 0
                fk_grade = 0
            
            # Lexical diversity (unique words / total words)
            words = text.lower().split()
            unique_words = len(set(words))
            lexical_diversity = unique_words / word_count if word_count > 0 else 0
            
            return {
                'word_count': word_count,
                'char_count': char_count,
                'sentence_count': sentence_count,
                'avg_words_per_sentence': word_count / sentence_count if sentence_count > 0 else 0,
                'flesch_reading_ease': flesch_score,
                'flesch_kincaid_grade': fk_grade,
                'lexical_diversity': lexical_diversity
            }
        except Exception as e:
            return {
                'word_count': 0,
                'char_count': 0,
                'sentence_count': 0,
                'avg_words_per_sentence': 0,
                'flesch_reading_ease': 0,
                'flesch_kincaid_grade': 0,
                'lexical_diversity': 0,
                'error': str(e)
            }

class EnhancedWebScraper:
    """Enhanced class for scraping data from multiple sources"""
    
    def __init__(self):
        """Initialize web scraper"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.quality_analyzer = DataQualityAnalyzer()
        
    def scrape_quotes(self, max_quotes=50):
        """Scrape quotes from multiple sources"""
        print("Scraping quotes from multiple sources...")
        
        quotes = []
        
        # Source 1: quotes.toscrape.com
        try:
            page = 1
            while len(quotes) < max_quotes // 2:
                url = f"http://quotes.toscrape.com/page/{page}/"
                response = self.session.get(url, timeout=10)
                
                if response.status_code != 200:
                    break
                
                soup = BeautifulSoup(response.content, 'html.parser')
                quote_divs = soup.find_all('div', class_='quote')
                
                if not quote_divs:
                    break
                
                for quote_div in quote_divs:
                    if len(quotes) >= max_quotes // 2:
                        break
                    
                    text_elem = quote_div.find('span', class_='text')
                    author_elem = quote_div.find('small', class_='author')
                    
                    if text_elem and author_elem:
                        quote_text = text_elem.get_text().strip().strip('"')
                        author = author_elem.get_text().strip()
                        
                        if len(quote_text) > 20:
                            metrics = self.quality_analyzer.calculate_text_metrics(quote_text)
                            quotes.append({
                                'text': quote_text,
                                'author': author,
                                'source': 'quotes.toscrape.com',
                                'category': 'quotes',
                                'quality_metrics': metrics
                            })
                
                page += 1
                sleep(1)
                
        except Exception as e:
            print(f"Error scraping quotes.toscrape.com: {e}")
        
        # Source 2: Quotable API
        try:
            remaining = max_quotes - len(quotes)
            if remaining > 0:
                for page in range(1, (remaining // 20) + 2):
                    api_url = f"https://api.quotable.io/quotes?page={page}&limit=20"
                    response = self.session.get(api_url, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        for quote in data.get('results', []):
                            if len(quotes) >= max_quotes:
                                break
                                
                            content = quote.get('content', '')
                            author = quote.get('author', '')
                            
                            if content and len(content) > 20:
                                metrics = self.quality_analyzer.calculate_text_metrics(content)
                                quotes.append({
                                    'text': content,
                                    'author': author,
                                    'source': 'Quotable API',
                                    'category': 'quotes',
                                    'quality_metrics': metrics
                                })
                    
                    sleep(0.5)
                    
        except Exception as e:
            print(f"Error scraping Quotable API: {e}")
        
        print(f"Scraped {len(quotes)} quotes from multiple sources")
        return quotes
    
    def scrape_news_headlines(self, max_headlines=50):
        """Scrape news headlines from multiple RSS feeds"""
        print("Scraping news headlines from multiple sources...")
        
        headlines = []
        
        # Multiple RSS sources
        rss_sources = [
            ("http://feeds.bbci.co.uk/news/rss.xml", "BBC News"),
            ("https://feeds.npr.org/1001/rss.xml", "NPR News"),
            ("https://rss.cnn.com/rss/edition.rss", "CNN"),
            ("https://feeds.reuters.com/reuters/topNews", "Reuters"),
            ("https://feeds.washingtonpost.com/rss/world", "Washington Post")
        ]
        
        per_source = max_headlines // len(rss_sources)
        
        for rss_url, source_name in rss_sources:
            try:
                response = self.session.get(rss_url, timeout=10)
                
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
                                    'source': 'arXiv API',
                                    'category': 'academic',
                                    'id': f'academic_{len(abstracts)+1}',
                                    'title': title,
                                    'quality_metrics': metrics
                                })
                
                sleep(1)
                
            except Exception as e:
                print(f"Error scraping arXiv category {category}: {e}")
                continue
        
        print(f"Scraped {len(abstracts)} academic abstracts from arXiv")
        return abstracts
    
    def generate_synthetic_reviews(self, num_reviews=40):
        """Generate synthetic e-commerce product reviews"""
        print(f"Generating {num_reviews} synthetic product reviews...")
        
        reviews = []
        templates = [
            "This {product} is {adjective}! The {feature} is a game-changer.",
            "I'm {emotion} with my new {product}. It's {adjective} and {adjective}.",
            "The {product} {adverb} exceeded my expectations. Highly recommended!",
            "A bit {negative_emotion} with the {product}. The {feature} isn't working as expected.",
            "Great value for the price. The {product} is {adjective} and easy to use.",
            "The quality of the {product} is {adjective}. I use it every day.",
            "Customer service was {adjective} when I had an issue with my {product}.",
            "The {product} arrived {adverb} and in perfect condition.",
            "I've been using the {product} for a few weeks now and I'm {emotion}.",
            "If you're looking for a {adjective} {product}, this is the one to get."
        ]
        
        fillers = {
            'product': ['gadget', 'device', 'tool', 'appliance', 'item', 'system'],
            'adjective': ['amazing', 'fantastic', 'excellent', 'disappointing', 'superb', 'inadequate', 'reliable', 'efficient'],
            'feature': ['battery life', 'user interface', 'design', 'performance', 'build quality'],
            'emotion': ['thrilled', 'impressed', 'satisfied', 'unhappy', 'delighted'],
            'adverb': ['truly', 'definitely', 'surprisingly', 'unfortunately'],
            'negative_emotion': ['disappointed', 'frustrated', 'annoyed']
        }
        
        for i in range(num_reviews):
            template = random.choice(templates)
            review_text = template.format(
                product=random.choice(fillers['product']),
                adjective=random.choice(fillers['adjective']),
                feature=random.choice(fillers['feature']),
                emotion=random.choice(fillers['emotion']),
                adverb=random.choice(fillers['adverb']),
                negative_emotion=random.choice(fillers['negative_emotion'])
            )
            
            metrics = self.quality_analyzer.calculate_text_metrics(review_text)
            reviews.append({
                'text': review_text,
                'source': 'Synthetic Reviews',
                'category': 'reviews',
                'id': f'review_{i+1}',
                'quality_metrics': metrics
            })
            
        print(f"Generated {len(reviews)} synthetic reviews")
        return reviews