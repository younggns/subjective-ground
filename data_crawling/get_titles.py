import os
import pandas as pd
from tqdm import tqdm
import time
from datetime import datetime
import pickle
# For pushshift.io
import praw
from praw.models import Submission

"""
Replace the client id and client secret with your own API access
"""
reddit = praw.Reddit(user_agent="Comment Extraction (by /u/INSERTUSERNAME)",
                     client_id="#####", client_secret="#####")

df = pd.read_csv(os.getcwd()+'/../data/reddit-morality/Comms__(Sub_AITA)__(Redditor_active).tsv', sep='\t')
submission_ids = [elem.split('/')[4] for elem in df['permalink'].tolist()]

ids, titles, selftexts = [], [], []
for _sub_id in tqdm(submission_ids):
    try:
        submission = reddit.submission(_sub_id)
        titles.append(str(submission.title))
        selftexts.append(str(submission.selftext))
        ids.append(_sub_id)
    except Exception as e:
        print(e)
        continue

_df = pd.DataFrame(data={'id':ids, 'title':titles, 'selftext':selftexts})
_df.to_csv(os.getcwd()+'/../data/reddit-morality/Titles__(Sub_AITA)__(Redditor_active).tsv', sep='\t', index=False)