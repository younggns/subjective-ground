# Done by pushshift.io api
import os
import argparse
import time
import pandas as pd
import praw
from praw.models import MoreComments

"""
Replace the client id and client secret with your own API access
"""
reddit = praw.Reddit(user_agent="Comment Extraction (by /u/INSERTUSERNAME)",
                     client_id="#####", client_secret="#####")

def valid_inst(elem):
    return elem == elem and elem is not None

def invalid_comment(elem):
    return isinstance(elem, MoreComments) or elem.is_submitter

def get_comments_from_submission_id(_id):
    
    submission = reddit.submission(id=_id)
    posts = []
    for c in submission.comments[:]:
        if invalid_comment(c):
            continue

        if valid_inst(c.author) and valid_inst(c.body) and valid_inst(c.permalink):
            posts.append((str(c.author), str(c.body), str(c.permalink)))

    return posts

def load_ids():
    df_moral_tr = pd.read_csv(os.getcwd()+'/../data/social-chem-101/AITA-train.tsv', sep='\t')
    df_moral_ev = pd.read_csv(os.getcwd()+'/../data/social-chem-101/AITA-dev.tsv', sep='\t')
    df_moral_te = pd.read_csv(os.getcwd()+'/../data/social-chem-101/AITA-test.tsv', sep='\t')

    df_moral = pd.concat([df_moral_tr, df_moral_ev, df_moral_te])
    df_moral_AITA = df_moral[df_moral.area=='amitheasshole']
    
    return [elem.split('reddit/amitheasshole/')[1] for elem in df_moral_AITA['situation-short-id'].tolist()]

def main():
    socchem_ids = load_ids()

    data_dict = {'author':[], 'comment':[], 'permalink':[]}
    print_counter = 0

    for _id in socchem_ids[:10]:
        
        if int(print_counter / 100) > 0:
            print("currently requested:", len(data_dict['author']))
            print_counter = 0

        try:
            comments = get_comments_from_submission_id(_id)
            for c in comments:
                data_dict['author'].append(c[0])
                data_dict['comment'].append(c[1])
                data_dict['permalink'].append(c[2])
            print_counter += 1
        except Exception as err:
            print(err)
            time.sleep(10)

    df_result = pd.DataFrame(data=data_dict)

    reddit_directory = os.getcwd()+'/../data/reddit-morality'
    if not os.path.exists(reddit_directory):
        os.makedirs(reddit_directory)
    df_result.to_csv(reddit_directory+'/Comms__(Sub_Sochem)__(Redditor_all).tsv.tsv', sep='\t', index=False)

if __name__ == '__main__':
    main()