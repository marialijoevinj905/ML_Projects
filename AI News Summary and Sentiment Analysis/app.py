import streamlit as st

# MUST be the first Streamlit command
st.set_page_config(
    page_title="News Sentiment Analyzer",
    page_icon="📰",
    layout="wide"
)

import requests
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk
import ssl
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import time
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse

# Download required NLTK data
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

@st.cache_data
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    except:
        pass

download_nltk_data()

class SimpleArticleExtractor:
    """Simple article content extractor using BeautifulSoup"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def extract_content(self, url):
        """Extract article content from URL"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
                script.decompose()
            
            # Try common article selectors
            content_selectors = [
                'article',
                '.article-content',
                '.post-content',
                '.entry-content',
                '.content',
                '[class*="article"]',
                '[class*="post"]',
                'main',
                '.main-content'
            ]
            
            article_text = ""
            
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    for element in elements:
                        text = element.get_text(strip=True)
                        if len(text) > len(article_text):
                            article_text = text
                    if len(article_text) > 200:
                        break
            
            # If no specific selectors found, try paragraphs
            if len(article_text) < 200:
                paragraphs = soup.find_all('p')
                article_text = ' '.join([p.get_text(strip=True) for p in paragraphs])
            
            # Clean up the text
            article_text = re.sub(r'\s+', ' ', article_text)
            article_text = article_text.strip()
            
            return article_text if len(article_text) > 100 else "Content extraction failed - article too short"
            
        except Exception as e:
            return f"Could not extract content: {str(e)}"

class NewsAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.article_extractor = SimpleArticleExtractor()
        
    def fetch_news(self, keyword, language='en', sort_by='publishedAt', page_size=20):
        """Fetch news articles from NewsAPI"""
        try:
            # Calculate date range (last 30 days)
            to_date = datetime.now()
            from_date = to_date - timedelta(days=30)
            
            params = {
                'q': keyword,
                'apiKey': self.api_key,
                'language': language,
                'sortBy': sort_by,
                'pageSize': page_size,
                'from': from_date.strftime('%Y-%m-%d'),
                'to': to_date.strftime('%Y-%m-%d')
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data['status'] == 'ok':
                return data['articles']
            else:
                st.error(f"API Error: {data.get('message', 'Unknown error')}")
                return []
                
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {str(e)}")
            return []
        except Exception as e:
            st.error(f"Error fetching news: {str(e)}")
            return []
    
    def extract_article_content(self, url):
        """Extract full article content using custom extractor"""
        return self.article_extractor.extract_content(url)
    
    def summarize_text(self, text, sentences_count=3):
        """Summarize text using sumy LSA summarizer"""
        try:
            if len(text.strip()) < 100:
                return text[:200] + "..." if len(text) > 200 else text
            
            # Clean text for better summarization
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summarizer = LsaSummarizer()
            summary = summarizer(parser.document, sentences_count)
            
            summary_text = ' '.join([str(sentence) for sentence in summary])
            return summary_text if summary_text else text[:300] + "..."
            
        except Exception as e:
            return text[:300] + "..." if len(text) > 300 else text
    
    def analyze_sentiment_textblob(self, text):
        """Analyze sentiment using TextBlob"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                sentiment = "Positive"
            elif polarity < -0.1:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
                
            return {
                'sentiment': sentiment,
                'polarity': polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except Exception as e:
            return {
                'sentiment': 'Unknown',
                'polarity': 0,
                'subjectivity': 0
            }
    
    def analyze_sentiment_vader(self, text):
        """Analyze sentiment using VADER"""
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            
            if scores['compound'] >= 0.05:
                sentiment = "Positive"
            elif scores['compound'] <= -0.05:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
                
            return {
                'sentiment': sentiment,
                'compound': scores['compound'],
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu']
            }
        except Exception as e:
            return {
                'sentiment': 'Unknown',
                'compound': 0,
                'positive': 0,
                'negative': 0,
                'neutral': 0
            }

def main():
    st.title("📰 News Sentiment Analyzer")
    st.markdown("*Real-time news analysis with sentiment detection and summarization*")
    st.markdown("---")
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # API Key input
    api_key = st.sidebar.text_input(
        "NewsAPI Key", 
        type="password",
        help="Get your free API key from https://newsapi.org"
    )
    
    if not api_key:
        st.warning("⚠️ Please enter your NewsAPI key in the sidebar to get started.")
        st.info("""
        **How to get a NewsAPI key:**
        1. Visit https://newsapi.org
        2. Sign up for a free account
        3. Copy your API key
        4. Paste it in the sidebar
        
        **Free tier includes:** 1,000 requests per day
        """)
        return
    
    # Initialize analyzer
    analyzer = NewsAnalyzer(api_key)
    
    # Main input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        keyword = st.text_input(
            "🔍 Enter keyword to search for news:",
            placeholder="e.g., artificial intelligence, climate change, cryptocurrency, sports"
        )
    
    with col2:
        search_button = st.button("🔍 Search News", type="primary")
    
    # Additional filters
    with st.sidebar:
        st.subheader("🔧 Filters")
        page_size = st.slider("Number of articles", 5, 50, 15)
        sort_by = st.selectbox(
            "Sort by:",
            ["publishedAt", "relevancy", "popularity"],
            index=0,
            help="publishedAt: Most recent first, relevancy: Most relevant first, popularity: Most popular first"
        )
        sentiment_method = st.selectbox(
            "Sentiment Analysis Method:",
            ["Both", "TextBlob", "VADER"],
            index=0,
            help="Both: Show results from both methods, TextBlob: Simple polarity analysis, VADER: Advanced social media optimized"
        )
        
        extract_full_content = st.checkbox(
            "Extract full article content",
            value=True,
            help="Enable to extract and analyze full article content (slower but more accurate)"
        )
    
    if search_button and keyword:
        with st.spinner(f"🔍 Searching for news about '{keyword}'..."):
            articles = analyzer.fetch_news(keyword, page_size=page_size, sort_by=sort_by)
        
        if not articles:
            st.error("❌ No articles found. Try different keywords or check your API key.")
            return
        
        st.success(f"✅ Found {len(articles)} articles!")
        
        # Process articles
        processed_articles = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, article in enumerate(articles):
            status_text.text(f"Processing article {i+1}/{len(articles)}: {article.get('title', 'Unknown')[:50]}...")
            progress_bar.progress((i + 1) / len(articles))
            
            # Determine content to analyze
            content_for_analysis = ""
            
            if extract_full_content and article.get('url'):
                # Try to extract full content
                full_content = analyzer.extract_article_content(article['url'])
                if len(full_content) > 100 and "Could not extract" not in full_content:
                    content_for_analysis = full_content
                else:
                    # Fallback to description and title
                    content_for_analysis = f"{article.get('title', '')} {article.get('description', '')}"
            else:
                # Use available content from API
                content_for_analysis = f"{article.get('title', '')} {article.get('description', '')} {article.get('content', '')}"
            
            # Clean content
            content_for_analysis = re.sub(r'\s+', ' ', content_for_analysis).strip()
            
            # Summarize content if it's long enough
            if len(content_for_analysis) > 300:
                summary = analyzer.summarize_text(content_for_analysis)
            else:
                summary = content_for_analysis
            
            # Analyze sentiment
            textblob_sentiment = analyzer.analyze_sentiment_textblob(content_for_analysis)
            vader_sentiment = analyzer.analyze_sentiment_vader(content_for_analysis)
            
            processed_article = {
                'title': article.get('title', 'No title'),
                'url': article.get('url', ''),
                'source': article.get('source', {}).get('name', 'Unknown'),
                'published_at': article.get('publishedAt', ''),
                'description': article.get('description', ''),
                'summary': summary,
                'textblob_sentiment': textblob_sentiment,
                'vader_sentiment': vader_sentiment,
                'image_url': article.get('urlToImage', ''),
                'content_length': len(content_for_analysis)
            }
            
            processed_articles.append(processed_article)
            
            # Small delay to be respectful to servers
            time.sleep(0.2)
        
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        st.markdown("---")
        st.header("📊 Analysis Results")
        
        # Sentiment overview
        col1, col2, col3, col4 = st.columns(4)
        
        # Count sentiments
        textblob_sentiments = [art['textblob_sentiment']['sentiment'] for art in processed_articles]
        vader_sentiments = [art['vader_sentiment']['sentiment'] for art in processed_articles]
        
        with col1:
            st.metric("📄 Total Articles", len(processed_articles))
        
        with col2:
            positive_count = textblob_sentiments.count('Positive')
            st.metric("😊 Positive (TextBlob)", positive_count)
        
        with col3:
            negative_count = textblob_sentiments.count('Negative')
            st.metric("😞 Negative (TextBlob)", negative_count)
            
        with col4:
            neutral_count = textblob_sentiments.count('Neutral')
            st.metric("😐 Neutral (TextBlob)", neutral_count)
        
        # Sentiment distribution charts
        if len(processed_articles) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                # TextBlob sentiment distribution
                textblob_df = pd.DataFrame({'Sentiment': textblob_sentiments})
                sentiment_counts = textblob_df['Sentiment'].value_counts()
                
                fig1 = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title='TextBlob Sentiment Distribution',
                    color_discrete_map={
                        'Positive': '#2E8B57',
                        'Negative': '#DC143C',
                        'Neutral': '#FFD700'
                    }
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # VADER sentiment distribution
                vader_df = pd.DataFrame({'Sentiment': vader_sentiments})
                vader_counts = vader_df['Sentiment'].value_counts()
                
                fig2 = px.pie(
                    values=vader_counts.values,
                    names=vader_counts.index,
                    title='VADER Sentiment Distribution',
                    color_discrete_map={
                        'Positive': '#2E8B57',
                        'Negative': '#DC143C',
                        'Neutral': '#FFD700'
                    }
                )
                st.plotly_chart(fig2, use_container_width=True)
        
        # Display articles
        st.markdown("---")
        st.header("📄 Article Analysis")
        
        for i, article in enumerate(processed_articles):
            # Create sentiment indicators
            tb_sentiment = article['textblob_sentiment']['sentiment']
            vader_sentiment = article['vader_sentiment']['sentiment']
            
            sentiment_emoji = {
                'Positive': '😊',
                'Negative': '😞',
                'Neutral': '😐'
            }
            
            title_display = f"{sentiment_emoji.get(tb_sentiment, '❓')} {article['title']}"
            
            with st.expander(f"{title_display}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**📰 Source:** {article['source']}")
                    st.markdown(f"**📅 Published:** {article['published_at']}")
                    st.markdown(f"**🔗 URL:** [Read full article]({article['url']})")
                    st.markdown(f"**📝 Content Length:** {article['content_length']} characters")
                    
                    if article['description']:
                        st.markdown("**📄 Description:**")
                        st.write(article['description'])
                    
                    st.markdown("**📋 AI Summary:**")
                    st.write(article['summary'])
                
                with col2:
                    if article['image_url']:
                        try:
                            st.image(article['image_url'], width=200, caption="Article Image")
                        except:
                            st.info("Image not available")
                    
                    # Sentiment analysis results
                    st.markdown("**🎭 Sentiment Analysis:**")
                    
                    if sentiment_method in ["Both", "TextBlob"]:
                        tb_sent = article['textblob_sentiment']
                        sentiment_color = {
                            'Positive': '🟢',
                            'Negative': '🔴',
                            'Neutral': '🟡'
                        }.get(tb_sent['sentiment'], '⚪')
                        
                        st.markdown(f"**TextBlob:** {sentiment_color} {tb_sent['sentiment']}")
                        st.markdown(f"- Polarity: {tb_sent['polarity']:.3f}")
                        st.markdown(f"- Subjectivity: {tb_sent['subjectivity']:.3f}")
                        
                        # Add polarity bar
                        polarity_normalized = (tb_sent['polarity'] + 1) / 2  # Convert from [-1,1] to [0,1]
                        st.progress(polarity_normalized, text=f"Polarity: {tb_sent['polarity']:.3f}")
                    
                    if sentiment_method in ["Both", "VADER"]:
                        vader_sent = article['vader_sentiment']
                        sentiment_color = {
                            'Positive': '🟢',
                            'Negative': '🔴',
                            'Neutral': '🟡'
                        }.get(vader_sent['sentiment'], '⚪')
                        
                        st.markdown(f"**VADER:** {sentiment_color} {vader_sent['sentiment']}")
                        st.markdown(f"- Compound: {vader_sent['compound']:.3f}")
                        
                        # VADER component breakdown
                        st.markdown("**Component Scores:**")
                        st.markdown(f"  - Positive: {vader_sent['positive']:.3f}")
                        st.markdown(f"  - Negative: {vader_sent['negative']:.3f}")
                        st.markdown(f"  - Neutral: {vader_sent['neutral']:.3f}")
                        
                        # Add compound score bar
                        compound_normalized = (vader_sent['compound'] + 1) / 2  # Convert from [-1,1] to [0,1]
                        st.progress(compound_normalized, text=f"Compound: {vader_sent['compound']:.3f}")
        
        # Download results option
        st.markdown("---")
        if st.button("📥 Download Results as CSV"):
            # Prepare data for download
            download_data = []
            for article in processed_articles:
                download_data.append({
                    'Title': article['title'],
                    'Source': article['source'],
                    'URL': article['url'],
                    'Published': article['published_at'],
                    'TextBlob_Sentiment': article['textblob_sentiment']['sentiment'],
                    'TextBlob_Polarity': article['textblob_sentiment']['polarity'],
                    'VADER_Sentiment': article['vader_sentiment']['sentiment'],
                    'VADER_Compound': article['vader_sentiment']['compound'],
                    'Summary': article['summary']
                })
            
            df = pd.DataFrame(download_data)
            csv = df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"news_sentiment_analysis_{keyword}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()