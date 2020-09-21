import tweepy
from sportsconfig import *

def create_api():
    auth = tweepy.OAuthHandler(akey, asecretkey)
    auth.set_access_token(atoken, asecret)

    api = tweepy.API(auth, wait_on_rate_limit=True,
        wait_on_rate_limit_notify=True)
    try:
        api.verify_credentials()
    except Exception as e:
        raise e

    return api
