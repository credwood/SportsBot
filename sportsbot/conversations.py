import tweepy
from api import create_api
import time
import random

class get_conversations:

    def __init__(self,search_terms, filter_terms, language='en'):
        self.filters_terms=filter_terms
        self.search_terms=search_terms
        self.language=language
        self.conversations = self.find_conversation(search_terms,filter_terms)


    def get_thread(self,tweet,api, language):
        """
        this is an unfortunately hacky function because there's no convenient way to get replies to tweets.
        It's necessary to use the API's search function to find tweets whose `in_reply_to_status_id` field matches the
        initial tweet's `id` field. Getting previous tweets is simply recursion on the `in_reply_to_status_id` field.
        """

        reply_status = tweet.in_reply_to_status_id

        def find_first_tweet(reply_status, prev_tweets=[]):
            """
            this function gets tweets prior to original input
            this isn't done. needs to match up better with the previous threads

            """
            if reply_status is None: return prev_tweets[::-1]
            try:
                status = api.get_status(reply_status, tweet_mode='extended',wait_on_rate_limit=True)

                #maybe the language condition isn't necessary?
                #if status.lang == language:
                prev_tweets.append((status.user.screen_name, status.full_text))
                reply_status = status.in_reply_to_status_id
                return find_first_tweet(reply_status,prev_tweets)

            except tweepy.TweepError as e:
                print(e)
                return prev_tweets[::-1]

        def get_subsequent(tweet, api, subsequent_tweets=[]):
            """
            this function gets subsequent tweets
            """
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

                except tweepy.TweepError as e:
                    print(e)
                except StopIteration:
                    break

            return subsequent_tweets

        return find_first_tweet(reply_status,language)+ [(tweet.user.screen_name, tweet.full_text)] + get_subsequent(tweet,api,language)



    def find_conversation(self,name, terms):
        api = create_api()
        conversations=[]
        subtract_terms = ''
        for term in terms: subtract_terms += ' -'+term
        c = tweepy.Cursor(api.search,q=name+subtract_terms+" -filter:retweets", timeout=999999, tweet_mode='extended').items(20)
        while True:
            try:
                tweet = c.next()
                conversations.append(self.get_thread(tweet,api))
            except tweepy.TweepError as e:
                print(e)
            except StopIteration:
                break
        return conversations


def process_tweets(topic):
    processed_tweets = []
    for conversation in conversations.conversations:
        if conversation is None: continue
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
    conversations = get_conversations('lakers', ['race','racist','china','chinese','"black lives matter"', 'police', '"blue lives matter"'])
    sample = process_tweets("lakers")
    print(sample[0])
