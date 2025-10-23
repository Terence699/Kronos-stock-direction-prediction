#!/usr/bin/env python3
"""
Sentiment Analysis of AAPL Tweets using FinBERT - Independent Scoring Module
===========================================================================

This module performs sentiment analysis of AAPL tweets using:
1. FinBERT model for financial sentiment analysis
2. Daily sentiment aggregation and scoring
3. Returns standardized scores for ensemble learning

Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def install_requirements():
    """Install required packages"""
    import subprocess
    import sys
    
    packages = [
        'transformers',
        'torch',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'scikit-learn'
    ]
    
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def load_finbert_model():
    """Load FinBERT model for sentiment analysis"""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from transformers import pipeline
        
        print("Loading FinBERT model...")
        # Using a financial sentiment analysis model
        model_name = "ProsusAI/finbert"
        
        # Create sentiment analysis pipeline
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            return_all_scores=True
        )
        
        print("FinBERT model loaded successfully!")
        return sentiment_pipeline
        
    except Exception as e:
        print(f"Error loading FinBERT: {e}")
        print("Falling back to basic sentiment analysis...")
        return None

def analyze_sentiment_basic(text):
    """Basic sentiment analysis fallback"""
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
                     'love', 'best', 'awesome', 'brilliant', 'outstanding', 'perfect']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'disappointing',
                     'hate', 'disgusting', 'ugly', 'stupid', 'dumb', 'pathetic']
    
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        return {'POSITIVE': 0.6, 'NEGATIVE': 0.2, 'NEUTRAL': 0.2}
    elif neg_count > pos_count:
        return {'POSITIVE': 0.2, 'NEGATIVE': 0.6, 'NEUTRAL': 0.2}
    else:
        return {'POSITIVE': 0.3, 'NEGATIVE': 0.3, 'NEUTRAL': 0.4}

def analyze_sentiment(text, pipeline):
    """Analyze sentiment of a single text"""
    try:
        if pipeline is None:
            return analyze_sentiment_basic(text)
        
        # Clean text
        text = str(text).strip()
        if len(text) == 0:
            return {'POSITIVE': 0.33, 'NEGATIVE': 0.33, 'NEUTRAL': 0.34}
        
        # Truncate if too long
        if len(text) > 512:
            text = text[:512]
        
        results = pipeline(text)
        
        # Convert to our format
        sentiment_scores = {}
        for result in results[0]:
            label = result['label'].upper()
            score = result['score']
            sentiment_scores[label] = score
        
        return sentiment_scores
        
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return analyze_sentiment_basic(text)

from config import TWEETS_FILE, OUTPUT_DIR

def load_and_process_data():
    """Load and process the tweets data"""
    print("Loading tweets data...")
    
    # Load the CSV file
    df = pd.read_csv(str(TWEETS_FILE))
    
    print(f"Loaded {len(df)} tweets")
    print(f"Columns: {list(df.columns)}")
    
    # Convert date columns
    df['created_at_utc'] = pd.to_datetime(df['created_at_utc'])
    df['created_at_et'] = pd.to_datetime(df['created_at_et'])
    
    # Extract date components
    df['date'] = df['created_at_utc'].dt.date
    df['hour'] = df['created_at_utc'].dt.hour
    df['day_of_week'] = df['created_at_utc'].dt.day_name()
    
    return df

def perform_sentiment_analysis(df, pipeline):
    """Perform sentiment analysis on all tweets"""
    print("Performing sentiment analysis...")
    
    sentiments = []
    sentiment_scores = []
    
    for idx, row in df.iterrows():
        if idx % 50 == 0:
            print(f"Processing tweet {idx + 1}/{len(df)}")
        
        text = row['text']
        sentiment_result = analyze_sentiment(text, pipeline)
        
        # Get the dominant sentiment
        dominant_sentiment = max(sentiment_result, key=sentiment_result.get)
        sentiments.append(dominant_sentiment)
        sentiment_scores.append(sentiment_result)
    
    # Add sentiment columns to dataframe
    df['sentiment'] = sentiments
    df['sentiment_scores'] = sentiment_scores
    
    # Extract individual sentiment scores
    df['positive_score'] = [score.get('POSITIVE', 0) for score in sentiment_scores]
    df['negative_score'] = [score.get('NEGATIVE', 0) for score in sentiment_scores]
    df['neutral_score'] = [score.get('NEUTRAL', 0) for score in sentiment_scores]
    
    return df

def create_visualizations(df):
    """Create comprehensive visualizations"""
    print("Creating visualizations...")
    
    # Create output directory
    output_dir = str(OUTPUT_DIR / 'sentiment_result')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Sentiment Distribution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    sentiment_counts = df['sentiment'].value_counts()
    colors = ['#2E8B57', '#DC143C', '#4682B4']  # Green, Red, Blue
    plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    plt.title('Sentiment Distribution', fontsize=14, fontweight='bold')
    
    # 2. Sentiment Scores Distribution
    plt.subplot(2, 2, 2)
    sentiment_data = [df['positive_score'], df['negative_score'], df['neutral_score']]
    labels = ['Positive', 'Negative', 'Neutral']
    plt.boxplot(sentiment_data, labels=labels)
    plt.title('Sentiment Scores Distribution', fontsize=14, fontweight='bold')
    plt.ylabel('Score')
    
    # 3. Sentiment Over Time
    plt.subplot(2, 2, 3)
    daily_sentiment = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
    daily_sentiment.plot(kind='line', ax=plt.gca(), marker='o')
    plt.title('Sentiment Trends Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Number of Tweets')
    plt.legend(title='Sentiment')
    plt.xticks(rotation=45)
    
    # 4. Engagement vs Sentiment
    plt.subplot(2, 2, 4)
    engagement_by_sentiment = df.groupby('sentiment')['engagement_score'].mean()
    bars = plt.bar(engagement_by_sentiment.index, engagement_by_sentiment.values, 
                   color=['#2E8B57', '#DC143C', '#4682B4'])
    plt.title('Average Engagement by Sentiment', fontsize=14, fontweight='bold')
    plt.xlabel('Sentiment')
    plt.ylabel('Average Engagement Score')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sentiment_overview.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Detailed Sentiment Analysis
    plt.figure(figsize=(15, 10))
    
    # Sentiment by Hour
    plt.subplot(2, 3, 1)
    hourly_sentiment = df.groupby(['hour', 'sentiment']).size().unstack(fill_value=0)
    hourly_sentiment.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.title('Sentiment Distribution by Hour', fontsize=12, fontweight='bold')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Tweets')
    plt.legend(title='Sentiment')
    plt.xticks(rotation=0)
    
    # Sentiment by Day of Week
    plt.subplot(2, 3, 2)
    daily_sentiment = df.groupby(['day_of_week', 'sentiment']).size().unstack(fill_value=0)
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_sentiment = daily_sentiment.reindex(day_order)
    daily_sentiment.plot(kind='bar', ax=plt.gca())
    plt.title('Sentiment by Day of Week', fontsize=12, fontweight='bold')
    plt.xlabel('Day of Week')
    plt.ylabel('Number of Tweets')
    plt.legend(title='Sentiment')
    plt.xticks(rotation=45)
    
    # Sentiment Scores Heatmap
    plt.subplot(2, 3, 3)
    sentiment_corr = df[['positive_score', 'negative_score', 'neutral_score', 'engagement_score']].corr()
    sns.heatmap(sentiment_corr, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f')
    plt.title('Sentiment Scores Correlation', fontsize=12, fontweight='bold')
    
    # Top Positive Tweets
    plt.subplot(2, 3, 4)
    top_positive = df.nlargest(5, 'positive_score')[['text', 'positive_score']]
    y_pos = np.arange(len(top_positive))
    plt.barh(y_pos, top_positive['positive_score'], color='#2E8B57')
    # Clean text for display (remove $ symbols that cause matplotlib issues)
    clean_texts = []
    for text in top_positive['text']:
        clean_text = str(text).replace('$', '').replace('&amp;', '&')
        clean_texts.append(clean_text[:50] + '...' if len(clean_text) > 50 else clean_text)
    plt.yticks(y_pos, clean_texts)
    plt.title('Top 5 Most Positive Tweets', fontsize=12, fontweight='bold')
    plt.xlabel('Positive Score')
    
    # Top Negative Tweets
    plt.subplot(2, 3, 5)
    top_negative = df.nlargest(5, 'negative_score')[['text', 'negative_score']]
    y_pos = np.arange(len(top_negative))
    plt.barh(y_pos, top_negative['negative_score'], color='#DC143C')
    # Clean text for display (remove $ symbols that cause matplotlib issues)
    clean_texts = []
    for text in top_negative['text']:
        clean_text = str(text).replace('$', '').replace('&amp;', '&')
        clean_texts.append(clean_text[:50] + '...' if len(clean_text) > 50 else clean_text)
    plt.yticks(y_pos, clean_texts)
    plt.title('Top 5 Most Negative Tweets', fontsize=12, fontweight='bold')
    plt.xlabel('Negative Score')
    
    # Sentiment vs Engagement Scatter
    plt.subplot(2, 3, 6)
    colors = {'POSITIVE': '#2E8B57', 'NEGATIVE': '#DC143C', 'NEUTRAL': '#4682B4'}
    for sentiment in df['sentiment'].unique():
        subset = df[df['sentiment'] == sentiment]
        plt.scatter(subset['positive_score'], subset['engagement_score'], 
                   c=colors[sentiment], label=sentiment, alpha=0.6)
    plt.title('Positive Score vs Engagement', fontsize=12, fontweight='bold')
    plt.xlabel('Positive Score')
    plt.ylabel('Engagement Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/detailed_sentiment_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. Time Series Analysis
    plt.figure(figsize=(15, 8))
    
    # Daily sentiment trends
    daily_sentiment = df.groupby('date')['sentiment'].value_counts().unstack(fill_value=0)
    daily_sentiment_pct = daily_sentiment.div(daily_sentiment.sum(axis=1), axis=0) * 100
    
    plt.subplot(2, 1, 1)
    for sentiment in daily_sentiment_pct.columns:
        plt.plot(daily_sentiment_pct.index, daily_sentiment_pct[sentiment], 
                marker='o', label=sentiment, linewidth=2)
    plt.title('Daily Sentiment Percentage Trends', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Percentage (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Daily engagement trends
    plt.subplot(2, 1, 2)
    daily_engagement = df.groupby('date')['engagement_score'].mean()
    plt.plot(daily_engagement.index, daily_engagement.values, 
             marker='o', color='purple', linewidth=2)
    plt.title('Daily Average Engagement Score', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Average Engagement Score')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/time_series_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_summary_report(df, output_dir):
    """Generate a comprehensive summary report"""
    print("Generating summary report...")
    
    # Calculate summary statistics
    total_tweets = len(df)
    sentiment_counts = df['sentiment'].value_counts()
    avg_engagement = df['engagement_score'].mean()
    
    # Sentiment percentages
    sentiment_pct = (sentiment_counts / total_tweets * 100).round(2)
    
    # Engagement by sentiment
    engagement_by_sentiment = df.groupby('sentiment')['engagement_score'].agg(['mean', 'std', 'count'])
    
    # Create summary report
    report = f"""
SENTIMENT ANALYSIS REPORT - AAPL TWEETS
========================================

Dataset Overview:
- Total Tweets Analyzed: {total_tweets:,}
- Date Range: {df['date'].min()} to {df['date'].max()}
- Average Engagement Score: {avg_engagement:.2f}

Sentiment Distribution:
- Positive: {sentiment_counts.get('POSITIVE', 0):,} tweets ({sentiment_pct.get('POSITIVE', 0):.1f}%)
- Negative: {sentiment_counts.get('NEGATIVE', 0):,} tweets ({sentiment_pct.get('NEGATIVE', 0):.1f}%)
- Neutral: {sentiment_counts.get('NEUTRAL', 0):,} tweets ({sentiment_pct.get('NEUTRAL', 0):.1f}%)

Engagement Analysis by Sentiment:
"""
    
    for sentiment in engagement_by_sentiment.index:
        mean_eng = engagement_by_sentiment.loc[sentiment, 'mean']
        std_eng = engagement_by_sentiment.loc[sentiment, 'std']
        count = engagement_by_sentiment.loc[sentiment, 'count']
        report += f"- {sentiment}: {mean_eng:.2f} ± {std_eng:.2f} (n={count})\n"
    
    # Top performing tweets
    report += f"""
Top 5 Most Engaging Tweets:
"""
    top_tweets = df.nlargest(5, 'engagement_score')[['text', 'sentiment', 'engagement_score']]
    for idx, row in top_tweets.iterrows():
        text_preview = str(row['text']).replace('$', '').replace('&amp;', '&')
        text_preview = text_preview[:100] + '...' if len(text_preview) > 100 else text_preview
        report += f"- {text_preview}\n  Sentiment: {row['sentiment']}, Engagement: {row['engagement_score']:.2f}\n"
    
    # Save report
    with open(f'{output_dir}/sentiment_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("Summary report saved!")
    print(report)

def save_processed_data(df, output_dir):
    """Save the processed data with sentiment analysis"""
    print("Saving processed data...")
    
    # Save full dataset with sentiment analysis
    df.to_csv(f'{output_dir}/tweets_with_sentiment.csv', index=False)
    
    # Save sentiment summary
    sentiment_summary = df.groupby('sentiment').agg({
        'tweet_id': 'count',
        'engagement_score': ['mean', 'std', 'min', 'max'],
        'like_count': 'mean',
        'retweet_count': 'mean',
        'reply_count': 'mean'
    }).round(2)
    
    sentiment_summary.to_csv(f'{output_dir}/sentiment_summary.csv')
    
    print("Processed data saved!")

class SentimentScoreGenerator:
    """Independent sentiment scoring module for ensemble learning"""
    
    def __init__(self, data_path=None):
        self.data_path = str(data_path or TWEETS_FILE)
        self.processed_df = None
        self.daily_scores = None
        
    def load_data(self):
        """Load and process the tweets data"""
        print("Loading tweets data...")
        
        # Load the CSV file
        df = pd.read_csv(self.data_path)
        
        print(f"Loaded {len(df)} tweets")
        print(f"Columns: {list(df.columns)}")
        
        # Convert date columns
        df['created_at_utc'] = pd.to_datetime(df['created_at_utc'])
        df['created_at_et'] = pd.to_datetime(df['created_at_et'])
        
        # Extract date components
        df['date'] = df['created_at_utc'].dt.date
        df['hour'] = df['created_at_utc'].dt.hour
        df['day_of_week'] = df['created_at_utc'].dt.day_name()
        
        return df
    
    def generate_scores(self):
        """Generate standardized sentiment scores for ensemble learning"""
        try:
            # Install requirements
            install_requirements()
            
            # Load FinBERT model
            pipeline = load_finbert_model()
            
            # Load and process data
            df = self.load_data()
            
            # Perform sentiment analysis
            df = self.perform_sentiment_analysis(df, pipeline)
            self.processed_df = df
            
            # Generate daily sentiment scores
            daily_scores = self._generate_daily_scores(df)
            self.daily_scores = daily_scores
            
            print("Sentiment scoring completed successfully!")
            return daily_scores
            
        except Exception as e:
            print(f"Error in sentiment scoring: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def perform_sentiment_analysis(self, df, pipeline):
        """Perform sentiment analysis on all tweets"""
        print("Performing sentiment analysis...")
        
        sentiments = []
        sentiment_scores = []
        
        for idx, row in df.iterrows():
            if idx % 50 == 0:
                print(f"Processing tweet {idx + 1}/{len(df)}")
            
            text = row['text']
            sentiment_result = analyze_sentiment(text, pipeline)
            
            # Get the dominant sentiment
            dominant_sentiment = max(sentiment_result, key=sentiment_result.get)
            sentiments.append(dominant_sentiment)
            sentiment_scores.append(sentiment_result)
        
        # Add sentiment columns to dataframe
        df['sentiment'] = sentiments
        df['sentiment_scores'] = sentiment_scores
        
        # Extract individual sentiment scores
        df['positive_score'] = [score.get('POSITIVE', 0) for score in sentiment_scores]
        df['negative_score'] = [score.get('NEGATIVE', 0) for score in sentiment_scores]
        df['neutral_score'] = [score.get('NEUTRAL', 0) for score in sentiment_scores]
        
        return df
    
    def _generate_daily_scores(self, df):
        """Generate daily sentiment scores for ensemble learning"""
        print("Generating daily sentiment scores...")
        
        # Group by date and calculate daily metrics
        daily_metrics = df.groupby('date').agg({
            'positive_score': ['mean', 'std', 'count'],
            'negative_score': ['mean', 'std', 'count'],
            'neutral_score': ['mean', 'std', 'count'],
            'engagement_score': ['mean', 'std'],
            'sentiment': lambda x: x.value_counts().to_dict()
        }).reset_index()
        
        # Flatten column names
        daily_metrics.columns = ['date', 'pos_mean', 'pos_std', 'pos_count', 
                               'neg_mean', 'neg_std', 'neg_count',
                               'neu_mean', 'neu_std', 'neu_count',
                               'eng_mean', 'eng_std', 'sentiment_dist']
        
        # Calculate sentiment score (-1 to 1)
        daily_metrics['sentiment_score'] = (
            daily_metrics['pos_mean'] - daily_metrics['neg_mean']
        )
        
        # Calculate sentiment strength (0 to 1)
        daily_metrics['sentiment_strength'] = (
            daily_metrics['pos_mean'] + daily_metrics['neg_mean']
        )
        
        # Calculate volume-weighted sentiment
        daily_metrics['volume_weighted_sentiment'] = (
            daily_metrics['sentiment_score'] * daily_metrics['pos_count']
        )
        
        # Generate prediction scores for 1d, 3d, 5d horizons
        for horizon in ['1d', '3d', '5d']:
            # Simple momentum-based prediction
            if horizon == '1d':
                # 1-day: Use current sentiment
                daily_metrics[f'sentiment_pred_{horizon}'] = daily_metrics['sentiment_score']
            elif horizon == '3d':
                # 3-day: Use 3-day moving average
                daily_metrics[f'sentiment_pred_{horizon}'] = daily_metrics['sentiment_score'].rolling(3, min_periods=1).mean()
            else:  # 5d
                # 5-day: Use 5-day moving average
                daily_metrics[f'sentiment_pred_{horizon}'] = daily_metrics['sentiment_score'].rolling(5, min_periods=1).mean()
        
        # Fill NaN values
        daily_metrics = daily_metrics.fillna(0)
        
        print(f"Generated daily scores for {len(daily_metrics)} days")
        return daily_metrics
    
    def get_performance_metrics(self):
        """Get performance metrics for reporting"""
        if self.daily_scores is None:
            return None
            
        metrics = {}
        for horizon in ['1d', '3d', '5d']:
            pred_col = f'sentiment_pred_{horizon}'
            if pred_col in self.daily_scores.columns:
                # Calculate basic statistics
                pred_scores = self.daily_scores[pred_col]
                metrics[f'sentiment_{horizon}'] = {
                    'mean_score': pred_scores.mean(),
                    'std_score': pred_scores.std(),
                    'positive_days': (pred_scores > 0).sum(),
                    'negative_days': (pred_scores < 0).sum(),
                    'total_days': len(pred_scores)
                }
        
        return metrics

def main():
    """Main execution function for standalone testing"""
    print("AAPL Tweets Sentiment Analysis - Independent Module")
    print("="*60)
    
    # Initialize sentiment score generator
    sentiment_generator = SentimentScoreGenerator()
    
    # Generate scores
    scores_df = sentiment_generator.generate_scores()
    
    if scores_df is not None:
        # Save results
        results_file = str((OUTPUT_DIR.parent) / 'sentiment_scores.csv')
        scores_df.to_csv(results_file, index=False)
        print(f"Sentiment scores saved to '{results_file}'")
        
        # Print performance metrics
        metrics = sentiment_generator.get_performance_metrics()
        if metrics:
            print("\nSentiment Performance Metrics:")
            for horizon, metric in metrics.items():
                print(f"{horizon}: Mean={metric['mean_score']:.4f}, "
                      f"Positive Days={metric['positive_days']}, "
                      f"Negative Days={metric['negative_days']}")
        
        return scores_df
    else:
        print("Failed to generate sentiment scores")
        return None

if __name__ == "__main__":
    scores = main()
