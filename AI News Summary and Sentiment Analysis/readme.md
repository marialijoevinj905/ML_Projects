# 📰 News Sentiment Analyzer

A comprehensive Streamlit application that fetches real-time news articles and performs advanced sentiment analysis using multiple NLP techniques.

## 🌟 Features

- **Real-time News Fetching**: Integration with NewsAPI to fetch the latest articles
- **Dual Sentiment Analysis**: Uses both TextBlob and VADER algorithms for comprehensive sentiment scoring
- **Content Extraction**: Automatically extracts full article content from URLs using web scraping
- **Text Summarization**: Generates concise summaries using LSA (Latent Semantic Analysis)
- **Interactive Visualizations**: Pie charts and progress bars showing sentiment distributions
- **Flexible Filtering**: Sort by relevancy, popularity, or publish date
- **Data Export**: Download analysis results as CSV files
- **User-friendly Interface**: Clean, intuitive Streamlit interface with real-time progress tracking

## 🛠️ Technologies Used

- **Frontend**: Streamlit
- **APIs**: NewsAPI for article fetching
- **NLP Libraries**: 
  - TextBlob for sentiment analysis
  - VADER Sentiment for social media optimized analysis
  - NLTK for text processing
  - Sumy for text summarization
- **Web Scraping**: BeautifulSoup for content extraction
- **Data Visualization**: Plotly Express
- **Data Processing**: Pandas
- **HTTP Requests**: Requests library

## 📋 Prerequisites

- Python 3.7+
- NewsAPI key (free at [newsapi.org](https://newsapi.org))

## 🚀 Installation

1. **Install required packages**
   ```bash
   pip install streamlit requests textblob vaderSentiment sumy nltk pandas plotly beautifulsoup4
   ```

2. **Download NLTK data** (will be done automatically on first run)
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

## 🔧 Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Get NewsAPI Key**
   - Visit [newsapi.org](https://newsapi.org)
   - Sign up for a free account
   - Copy your API key

3. **Configure the app**
   - Enter your NewsAPI key in the sidebar
   - Adjust filters (number of articles, sorting method, sentiment analysis method)
   - Enable/disable full content extraction

4. **Analyze news**
   - Enter keywords to search for
   - Click "Search News"
   - View results with sentiment analysis and summaries

## 🎯 Key Components

### NewsAnalyzer Class
- Handles NewsAPI integration
- Manages article fetching with date filtering
- Coordinates sentiment analysis workflows

### SimpleArticleExtractor Class
- Web scraping functionality using BeautifulSoup
- Content extraction from various article formats
- Text cleaning and preprocessing

### Sentiment Analysis Methods
- **TextBlob**: Provides polarity (-1 to 1) and subjectivity (0 to 1) scores
- **VADER**: Optimized for social media text with compound scoring

### Text Summarization
- Uses Latent Semantic Analysis (LSA) via Sumy
- Configurable summary length
- Fallback to truncated text for short articles

## 📊 Output Features

- **Sentiment Metrics**: Count of positive, negative, and neutral articles
- **Visual Charts**: Interactive pie charts showing sentiment distribution
- **Article Details**: 
  - Source and publication date
  - AI-generated summaries
  - Sentiment scores with progress bars
  - Direct links to original articles
- **Export Options**: CSV download with comprehensive analysis data

## ⚙️ Configuration Options

| Setting | Description | Options |
|---------|-------------|---------|
| Number of Articles | Limit results | 5-50 articles |
| Sort Method | Article ordering | publishedAt, relevancy, popularity |
| Sentiment Method | Analysis approach | Both, TextBlob, VADER |
| Content Extraction | Full article analysis | Enable/Disable |

## 🔍 API Usage

The application uses NewsAPI with the following parameters:
- **Date Range**: Last 30 days
- **Language**: English (configurable)
- **Rate Limiting**: Built-in delays for respectful scraping
- **Error Handling**: Comprehensive exception management

## 📈 Sentiment Scoring

### TextBlob
- **Polarity**: -1 (negative) to 1 (positive)
- **Subjectivity**: 0 (objective) to 1 (subjective)
- **Classification**: >0.1 positive, <-0.1 negative, else neutral

### VADER
- **Compound Score**: -1 (negative) to 1 (positive)
- **Component Scores**: Positive, negative, neutral percentages
- **Classification**: ≥0.05 positive, ≤-0.05 negative, else neutral

## 🚨 Limitations

- NewsAPI free tier: 1,000 requests/day
- Some websites may block content extraction
- Summarization quality depends on article structure
- Rate limiting for respectful web scraping

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 🙏 Acknowledgments

- [NewsAPI](https://newsapi.org) for news data
- [Streamlit](https://streamlit.io) for the web framework
- [TextBlob](https://textblob.readthedocs.io/) and [VADER](https://github.com/cjhutto/vaderSentiment) for sentiment analysis
- [Sumy](https://github.com/miso-belica/sumy) for text summarization

## 📞 Support

For questions or issues, please open an issue in the GitHub repository or contact marialijoevin@gmail.com
