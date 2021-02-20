# these dictionaries are not modified by the code
# Ideally users will provdide their own label mappings
# but these are provided as defualt `labels_dicts` and 
# `class_dict` values. Please see the main README for more information

classes_dict = {
                "Q1": {
                ' No': 0,
                ' Unsure': 1,
                ' Yes': 2,
                'invalid': 3
                },
                "Q2": {
                ' Negative': 0,
                ' Positive': 1,
                ' None': 2,
                ' Defensive': 3,
                ' Neutral': 4,
                ' Unsure':5,
                'invalid': 6
                },
                "yelp": {
                ' Negative': 0,
                ' Positive': 1,
                ' Neutral': 2,
                'invalid': 3
                },
                "Q3": {
                ' Opposition': 0,
                ' Discussion': 1,
                ' Agreement': 2,
                ' Unsure': 3,
                ' None': 4,
                'invalid': 5
                }
}

label_dict = {"all_values": {
                  1: " No",
                  2: " Remote",
                  3: " Unsure",
                  4: " Probably",
                  5: " Yes",
                  6: " Neutral",
                  7: " None",
                  8: " Positive",
                  9: " Defensive",
                  10: " Negative",
                  11: " Opposition",
                  12: " Discussion",
                  13: " Agreement"
              },
              "bucketed_labels":{
                  1: [" No", " Remote"],
                  2: [" No", " Remote"],
                  3: [" Unsure"],
                  4: [" Probably", " Yes"],
                  5: [" Probably"," Yes"] ,
                  6: [" Neutral", " None"],
                  7: [" None", " Neutral"],
                  8: [" Positive"],
                  9: [" Defensive"],
                  10: [" Negative"],
                  11: [" Opposition",],
                  12: [" Discussion"],
                  13: [" Agreement"]    
              },
              "baseline_accuracy": 0.333
}