"""
This module connects to Twitter's API using Tweepy
and returns up to 20 conversations fulfilling init parameters
"""
import random
import tweepy
#import time
from sportsconfig import akey, asecretkey, atoken, asecret

def create_api():
    """
    Wrapper for tweepy's API method
    """
    auth = tweepy.OAuthHandler(akey, asecretkey)
    auth.set_access_token(atoken, asecret)

    api = tweepy.API(auth, wait_on_rate_limit=True,
        wait_on_rate_limit_notify=True)
    try:
        api.verify_credentials()
    except Exception as exception:
        raise exception
    return api

class GetConversations:
    """
    Stores search parameters and processed conversations
    """

    def __init__(self,search_terms, filter_terms, language='en'):
        self.filters_terms=filter_terms
        self.search_terms=search_terms
        self.language=language
        self.conversations = self.find_conversation(search_terms,filter_terms)
    def get_thread(self,tweet,api):
        """
        this is an unfortunately hacky function because there's no convenient way
        to get replies to tweets. It's necessary to use the API's search function to find
        tweets whose `in_reply_to_status_id` field matches the initial tweet's `id` field.
        Getting previous tweets is simply recursion on the `in_reply_to_status_id` field.
        """

        reply_status = tweet.in_reply_to_status_id

        def find_first_tweet(reply_status, prev_tweets=None):
            """
            this function gets tweets prior to original input

            """
            if prev_tweets is None:
                prev_tweets = []
            if reply_status is None:
                return prev_tweets[::-1]
            try:
                status = api.get_status(reply_status, tweet_mode='extended',wait_on_rate_limit=True)

                #maybe the language condition isn't necessary?
                #if status.lang == language:
                prev_tweets.append((status.user.screen_name, status.full_text))
                reply_status = status.in_reply_to_status_id
                return find_first_tweet(reply_status,prev_tweets)

            except tweepy.TweepError as exception:
                print(exception)
                return prev_tweets[::-1]

        def get_subsequent(tweet, api, subsequent_tweets=None):
            """
            this function gets subsequent tweets
            """
            if subsequent_tweets is None:
                subsequent_tweets = []
            tweet_id = tweet.id
            user_name = tweet.user.screen_name

            replies = tweepy.Cursor(api.search, q='to:'+user_name+' -filter:retweets',
                since_id=tweet_id, max_id=None, tweet_mode='extended').items()

            while True:
                try:
                    reply = replies.next()
                    if reply.in_reply_to_status_id == tweet_id:
                        subsequent_tweets.append((reply.user.screen_name, reply.full_text))
                        return get_subsequent(reply, api,subsequent_tweets)

                except tweepy.TweepError as exception:
                    print(exception)
                except StopIteration:
                    break

            return subsequent_tweets

        before_initial_tweet = find_first_tweet(reply_status)
        initial_tweet = [(tweet.user.screen_name, tweet.full_text)]
        subsequent_tweets = get_subsequent(tweet,api)
        return before_initial_tweet + initial_tweet + subsequent_tweets



    def find_conversation(self,name, terms):
        """
        Initial search for tweets. Will return up to 20 conversations
        fulfilling the search criteria
        """
        api = create_api()
        conversations_list=[]
        subtract_terms = ''
        for term in terms:
            subtract_terms += ' -'+term
        found_tweets = tweepy.Cursor(api.search,
                            q=name+subtract_terms+" -filter:retweets",
                            timeout=999999,
                            tweet_mode='extended').items(20)
        while True:
            try:
                tweet = found_tweets.next()
                conversations_list.append(self.get_thread(tweet,api))
            except tweepy.TweepError as exception:
                print(exception)
            except StopIteration:
                break
        return conversations_list


def process_tweets(topic):
    """
    puts conversations into a list with a question for the
    model to answer at the end
    """
    processed_tweets = []
    for conversation in conversations.conversations:
        if conversation is None:
            continue
        names = set([])
        for tup in conversation:
            names.add(tup[0])
        name = random.choice(list(names))
        temp = [f"{tup[0]}: {tup[1]}" for tup in conversation]
        temp.append(f"\n--\nQuestion: Does {name} like {topic}? \nAnswer:")
        processed_tweets.append(temp)
    return processed_tweets


#will get conversations about the lakers, but without the keywords/phrases in the list.
if __name__ == '__main__':
    filter_out = ['race','racist','china','chinese','"black lives matter"', 'police', '"blue lives matter"']
    conversations = GetConversations('lakers',filter_out)
    sample = process_tweets("lakers")
    print(sample[0])
