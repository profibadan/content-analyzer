import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

df = pd.read_csv("data/nebraska_tweets_0423-0427.csv")

# plot total vol
def plot_total_vol(df):
    df_plot = df
    df_plot['count'] = 1
    df_plot = df_plot.set_index(pd.DatetimeIndex(df_plot['created_at'])).resample('H').sum()[['count']]
    df_plot.plot(title='Total Tweet Volume per Hour')
    plt.xlabel("Time")
    plt.ylabel("Tweet Volume")
    plt.show()

def plot_content_vol(df):
    df_plot = pd.get_dummies(df[['content_label', 'created_at']], prefix=[None], columns=['content_label'])
    df_plot = df_plot.set_index(pd.DatetimeIndex(df_plot['created_at'])).resample('H').sum()
    df_plot.plot(title='Content Tweet Volume per Hour')
    plt.xlabel("Time")
    plt.ylabel("Tweet Volume")
    plt.show()

def plot_sentiment_vol(df):
    df_plot = pd.get_dummies(df[['sentiment_label', 'created_at']], prefix=[None], columns=['sentiment_label'])
    df_plot = df_plot.set_index(pd.DatetimeIndex(df_plot['created_at'])).resample('H').sum()
    df_plot.plot(title='Sentiment Tweet Volume per Hour')
    plt.xlabel("Time")
    plt.ylabel("Tweet Volume")
    plt.show()

def plot_content_sentiment_vol(df, sentiment='na'):
    df_plot = pd.get_dummies(df[['content_label', 'created_at']].loc[df['sentiment_label']==sentiment], prefix=[None], columns=['content_label'])
    df_plot = df_plot.set_index(pd.DatetimeIndex(df_plot['created_at'])).resample('H').sum()
    df_plot.plot(title=f'Content (sent={sentiment}) Tweet Volume per Hour')
    plt.xlabel("Time")
    plt.ylabel("Tweet Volume")
    plt.show()

def plot_content_edge_type_vol(df, edge_type='retweet'):
    if edge_type == 'retweet':
        df_plot = pd.get_dummies(df[['content_label', 'created_at']].loc[~df['rt_id'].isna()], prefix=[None],
                                 columns=['content_label'])
    elif edge_type == 'reply':
        df_plot = pd.get_dummies(df[['content_label', 'created_at']].loc[~df['reply_to_user_id'].isna()], prefix=[None],
                                 columns=['content_label'])
    elif edge_type == 'other':
        df_plot = pd.get_dummies(df[['content_label', 'created_at']].loc[(df['reply_to_user_id'].isna()) &
                                                                         (df['rt_id'].isna())], prefix=[None],
                                 columns=['content_label'])
    elif edge_type == 'mention':
        df_plot = pd.get_dummies(df[['content_label', 'created_at']].loc[(~df['user_mentions'].isna()) &
                                                                         (df['rt_id'].isna()) &
                                                                         (df['reply_to_user_id'].isna())], prefix=[None],
                                 columns=['content_label'])
    df_plot = df_plot.set_index(pd.DatetimeIndex(df_plot['created_at'])).resample('H').sum()
    df_plot.plot(title=f'Content (type={edge_type}) Tweet Volume per Hour')
    plt.xlabel("Time")
    plt.ylabel("Tweet Volume")
    plt.show()


def plot_mentions(df, num_mentions=10, mention_type='retweet'):
    df_plot = df[['created_at']]
    if mention_type == 'retweet':
        mentions_col = df.loc[~df['rt_id'].isna()].user_mentions.to_list()
    elif mention_type == 'reply':
        mentions_col = df.loc[~df['reply_to_user_id'].isna()].user_mentions.to_list()
    elif mention_type == 'other':
        mentions_col = df.loc[(df['reply_to_user_id'].isna()) & (df['rt_id'].isna())].user_mentions.to_list()
    mentions = sum([um.split(",") for um in mentions_col if pd.notna(um)], [])
    top_mentions = [m[0] for m in Counter(mentions).most_common(num_mentions)]

    for tm in top_mentions:
        tm_count = [1 if tm in str(m).split(",") else 0 for m in df.user_mentions.to_list() ]
        df_plot[tm] = tm_count
        if mention_type == 'retweet':
            df_plot[tm].loc[df['rt_id'].isna()] = 0
        elif mention_type == 'reply':
            df_plot[tm].loc[df['reply_to_user_id'].isna()] = 0
        elif mention_type == 'other':
            df_plot[tm].loc[(~df['reply_to_user_id'].isna()) & (~df['rt_id'].isna())] = 0

    df_plot = df_plot.set_index(pd.DatetimeIndex(df_plot['created_at'])).resample('H').sum()
    df_plot.plot(title=f'User Mentions ({mention_type}) Volume per Hour')
    plt.xlabel("Time")
    plt.ylabel("Tweet Volume")
    plt.show()


with open("data/smart_common_words.txt", 'r') as f:
    STOPWORDS = f.read().split("\n")


def plot_words(df, num_words=10, ngram=2,  stopwords=STOPWORDS, include_hashtags=False):
    #TODO: figure out how to avoid getting all words from the same tweet
    def words_to_ngram(list_of_words):
        """ Converts list of 1-gram words to list of n-gram tokens
        """
        list_of_words = [w for w in list_of_words if (not w.startswith("@")) and (w.lower() not in stopwords)]
        if not include_hashtags:
            list_of_words = [w for w in list_of_words if not w.startswith("#")]
        list_of_words = [list_of_words[i:i + ngram] for i in range(len(list_of_words) - ngram + 1)]
        return [" ".join(w) for w in list_of_words]

    df_plot = df[['created_at']]
    text_col = df.text.to_list()
    words = sum([t.split() for t in text_col], [])
    words = words_to_ngram(words)

    top_words = [w[0] for w in Counter(words).most_common(num_words)]

    for tw in top_words:
        tw_count = [1 if tw in words_to_ngram(w.split()) else 0 for w in text_col ]
        df_plot[tw] = tw_count

    df_plot = df_plot.set_index(pd.DatetimeIndex(df_plot['created_at'])).resample('H').sum()
    df_plot.plot(title=f'Tweet (Word) Volume per Hour')
    plt.xlabel("Time")
    plt.ylabel("Tweet Volume")
    plt.show()
