"""
This module connects to Twitter's API using Tweepy
and returns up to 20 conversations based on user's parameters
"""
import os
import tweepy
from .datasets import _save_data, Tweet, _prepare_conv_template

def _create_api():
    """
    Wrapper for Tweepy's API method
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

def get_conversations(search_terms,
                        filter_terms,
                        template_topic,
                        jsonlines_file='output.jsonl',
                        max_conversation_length=10):
    """
    Collects up to 50 relevant conversations using Tweepy's wrapper for Twitter's API,
    processes them into dataclasses and stores them with jsonlines file.
    """
    api = _create_api()
    conversations = _find_conversation(
                                search_terms,
                                filter_terms,
                                api,
                                template_topic,
                                max_conversation_length
                    )
    _save_data(conversations,jsonlines_file)
    return conversations

def _find_conversation(name, filter_terms, api, template_topic, max_conversation_length):
    """
    Initial search for tweets. Will find up to 50 tweets
    fulfilling the search criteria. This function calls `_get_thread`
    for each tweet which returns a full conversation.
    """
    conversations_lst = []
    subtract_terms = _filter_terms(filter_terms)
    found_tweets = tweepy.Cursor(api.search,
                        q=name+subtract_terms+" -filter:retweets",
                        timeout=999999,
                        tweet_mode='extended').items(50)
    while True:
        try:
            tweet = found_tweets.next()
            conversation_obj = _get_thread(tweet,api,filter_terms,template_topic)
            if conversation_obj and 1 < len(conversation_obj.thread) <= max_conversation_length:
                conversations_lst.append(conversation_obj)
        except tweepy.TweepError as exception:
            print(exception)
        except StopIteration:
            break
    return conversations_lst

def _get_thread(tweet,api,filter_list,template_topic):
    """
    calls `_find_first_tweet` and `_get_subsequent`, concatenates
    these values with the initial tweet and returns the full thread in order.
    """
    reply_status = tweet.in_reply_to_status_id
    before_initial_tweet = _find_first_tweet(reply_status,api,filter_list)
    initial_tweet = [Tweet(
                            tweet.id,
                            tweet.user.screen_name,
                            tweet.user.name,
                            tweet.full_text,
                            tweet.lang,
                            tweet.created_at,
                            tweet.user.followers_count,
                            tweet.user.friends_count,
                            tweet.user.description
                          )
                    ]
    after_initial_tweet = _get_subsequent(tweet,api,filter_list)
    if (before_initial_tweet is False) or (after_initial_tweet is False):
        return False
    full_conv = before_initial_tweet + initial_tweet + after_initial_tweet
    conversation_class = _prepare_conv_template(full_conv, template_topic)
    return conversation_class

def _find_first_tweet(reply_status, api, filter_list, prev_tweets=None):
    """
    This function gets tweets prior to initial tweet

    """
    prev_tweets = [] if prev_tweets is None else prev_tweets
    if reply_status is None:
        return prev_tweets[::-1]
    try:
        tweet = api.get_status(reply_status, tweet_mode='extended',wait_on_rate_limit=True)
        if _filter_terms(filter_list, tweet=tweet,find_first=True):
            return False
        #maybe the language condition isn't necessary?
        #if status.lang == language:
        prev_tweets.append(Tweet(
                                tweet.id,
                                tweet.user.screen_name,
                                tweet.user.name,
                                tweet.full_text,
                                tweet.lang,
                                tweet.created_at,
                                tweet.user.followers_count,
                                tweet.user.friends_count,
                                tweet.user.description
                                )
                            )
        reply_status = tweet.in_reply_to_status_id
        return _find_first_tweet(reply_status, api, filter_list, prev_tweets)
    except tweepy.TweepError as exception:
        print(exception)
        return False

def _get_subsequent(tweet, api, filter_list, subsequent_tweets=None):
    """
    This function gets subsequent tweets. It's necessary to use the API's
    search function to find tweets whose `in_reply_to_status_id` field
    matches the initial tweet's `id` field.
    """
    subsequent_tweets = [] if subsequent_tweets is None else subsequent_tweets
    tweet_id = tweet.id
    user_name = tweet.user.screen_name
    subtract_terms = _filter_terms(filter_list)
    replies = tweepy.Cursor(api.search, q='to:'+user_name+subtract_terms+' -filter:retweets',
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
                                                reply.created_at,
                                                tweet.user.followers_count,
                                                tweet.user.friends_count,
                                                tweet.user.description
                                                )
                                            )
                return _get_subsequent(reply, api, filter_list, subsequent_tweets)

        except tweepy.TweepError as exception:
            print(exception)
            return False
        except StopIteration:
            break

    return subsequent_tweets

def _filter_terms(filters, tweet=False,find_first=False):
    if find_first:
        for term in filters:
            if term in tweet.full_text:
                return True
        return False
    else:
        subtract_terms = ''
        for term in filters:
            subtract_terms += ' -'+term
        return subtract_terms
