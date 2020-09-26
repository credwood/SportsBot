"""
This module connects to Twitter's API using Tweepy
and returns up to 20 conversations fulfilling init parameters
"""
import os
from collections import defaultdict
import tweepy
from .datasets import _save_data, Tweet, Conversation

def _create_api():
    """
    Wrapper for tweepy's API method
    """
    akey, asecretkey = os.environ["AKEY"],os.environ["ASECRETKEY"]
    atoken, asecret = os.environ["ATOKEN"],os.environ["ASECRET"]
    auth = tweepy.OAuthHandler(akey, asecretkey)
    auth.set_access_token(atoken, asecret)
    api = tweepy.API(auth, wait_on_rate_limit=True,
        wait_on_rate_limit_notify=True)
    try:
        api.verify_credentials()
    except Exception as exception:
        raise exception
    return api
    
def get_conversations(search_terms, filter_terms, jsonlines_file='output.jsonl'):
    """
    Collects up to 20 relevant conversations using Tweepy's wrapper for Twitter's API,
    processes them into dataclasses and stored by jsonlines file.
    """
    api = _create_api()
    conversations = _find_conversation(search_terms, filter_terms, api)
    _save_data(conversations,jsonlines_file)
    return conversations

def _get_thread(tweet,api):
    """
    calls `_find_first_tweet` and `_get_subsequent`, concatenates
    the returned values with the initial tweet and returns a full
    conversation.
    """
    reply_status = tweet.in_reply_to_status_id
    before_initial_tweet = _find_first_tweet(reply_status,api)
    initial_tweet = [Tweet(
                            tweet.id,
                            tweet.user.screen_name,
                            tweet.user.name,
                            tweet.full_text,
                            tweet.lang,
                            tweet.created_at
                          )
                    ]
    after_initial_tweet = _get_subsequent(tweet,api)
    full_conv = before_initial_tweet + initial_tweet + after_initial_tweet
    stat_dict = defaultdict()
    conversation_class = Conversation(full_conv, stat_dict)
    return conversation_class

def _find_first_tweet(reply_status, api, prev_tweets=None):
    """
    This function gets tweets prior to initial tweet

    """
    prev_tweets = [] if prev_tweets is None else prev_tweets
    if reply_status is None:
        return prev_tweets[::-1]
    try:
        tweet = api.get_status(reply_status, tweet_mode='extended',wait_on_rate_limit=True)

        #maybe the language condition isn't necessary?
        #if status.lang == language:
        prev_tweets.append(Tweet(
                                tweet.id,
                                tweet.user.screen_name,
                                tweet.user.name,
                                tweet.full_text,
                                tweet.lang,
                                tweet.created_at
                                )
                            )
        reply_status = tweet.in_reply_to_status_id
        return _find_first_tweet(reply_status,api,prev_tweets)

    except tweepy.TweepError as exception:
        print(exception)
        return prev_tweets[::-1]

def _get_subsequent(tweet, api, subsequent_tweets=None):
    """
    This function gets subsequent tweets. There's no convenient way to get replies
    to tweets. It's necessary to use the API's search function to find tweets whose
    `in_reply_to_status_id` field matches the initial tweet's `id` field.
    """
    subsequent_tweets = [] if subsequent_tweets is None else subsequent_tweets
    tweet_id = tweet.id
    user_name = tweet.user.screen_name

    replies = tweepy.Cursor(api.search, q='to:'+user_name+' -filter:retweets',
        since_id=tweet_id, max_id=None, tweet_mode='extended').items()

    while True:
        try:
            reply = replies.next()
            if reply.in_reply_to_status_id == tweet_id:
                subsequent_tweets.append(Tweet(
                                                reply.id,
                                                reply.user.screen_name,
                                                reply.user.name,
                                                reply.full_text,
                                                reply.lang,
                                                reply.created_at
                                                )
                                            )
                return _get_subsequent(reply, api,subsequent_tweets)

        except tweepy.TweepError as exception:
            print(exception)
        except StopIteration:
            break

    return subsequent_tweets

def _find_conversation(name, terms, api):
    """
    Initial search for tweets. Will find up to 20 tweets
    fulfilling the search criteria. This function calls `_get_thread`
    for each tweet to find and return a full conversation.
    """
    conversations_lst = []
    subtract_terms = ''
    for term in terms:
        subtract_terms += ' -'+term
    found_tweets = tweepy.Cursor(api.search,
                        q=name+subtract_terms+" -filter:retweets",
                        timeout=999999,
                        tweet_mode='extended').items(20)
    i = 1
    while True:
        try:
            tweet = found_tweets.next()
            conversations_lst.append(_get_thread(tweet,api))
            i += 1
        except tweepy.TweepError as exception:
            print(exception)
        except StopIteration:
            break
    return conversations_lst
