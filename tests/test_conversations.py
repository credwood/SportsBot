import unittest
import numpy as np
from sportsbot.conversations import get_conversations

class TestTweepyStream(unittest.TestCase):
    """
    these functions will cerate a .jsonl output file
    """
    def test_get_convs(self):
        filter_out= ['"bad words', "bad", "words"]
        search = '"grazing goats"'
        get_conversations(search, filter_out)
    
    def test_likely_long_convs(self):
        filter_out = []
        search = "I"
        get_conversations(search, filter_out)

