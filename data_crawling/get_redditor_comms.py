import os
import pandas as pd
from tqdm import tqdm
import time
from datetime import datetime

# For pushshift.io
import requests
import json

from collections import Counter

def get_redditor():
    df = pd.read_csv(os.getcwd()+'/../data/reddit-morality/Comms__(Sub_Sochem)__(Redditor_all).tsv.tsv', sep='\t')

    author_cnt = Counter()
    author_cnt.update(df.author.tolist())
    return sorted(author_cnt.items(), key=lambda x:x[1], reverse=True)

def adjust_len(data_dict):
    max_len = 0
    for key in data_dict:
        if len(data_dict[key]) > max_len:
            max_len = len(data_dict[key])
    for key in data_dict:
        if len(data_dict[key]) < max_len:
            data_dict[key].append('')
    return data_dict

def crawl_comments(base_url, data_dict):
    sleep_count = 0
    while True:
        try:
            response = requests.get(base_url)
            r = json.loads(response.content)
            if len(r['data']) == 0:
                break
            else:
                for submission in r['data']:
                    if 'body' not in submission:
                        continue

                    data_dict['body'].append(submission['body'])
                    data_dict['author'].append(submission['author'])
                    data_dict['subreddit'].append(submission['subreddit'])

                    if 'permalink' in submission:
                        data_dict['permalink'].append(submission['permalink'])
                    else:
                        data_dict['permalink'].append('')
                    
                    date_str = datetime.fromtimestamp(submission['created_utc']).strftime('%Y-%m-%d')
                    data_dict['created_utc'].append(date_str)

                    data_dict = adjust_len(data_dict)
                
                base_url = base_url.split('&before=')[0] + '&before=' + str(r['data'][-1]['created_utc'])
                if r['data'][-1]['created_utc'] < 1372697994: # by 2013 Jul 1
                    print("Crawled all after Jul 2013")
                    break
                sleep_count = 0

        except Exception as e:
            # print(e, "Trying to sleep on url,", base_url)
            time.sleep(6)
            sleep_count += 1
            if sleep_count >= 10:
                break

    return data_dict

def main():
    all_redditors_sorted = get_redditor()

    threshold = 30
    active_redditors = [str(elem[0]) for elem in all_redditors_sorted if elem[1] >= threshold]

    comment_dict = {'author':[], 'subreddit':[], 'body':[], 'permalink':[], 'created_utc':[]}

    for _redditor in tqdm(active_redditors):
        base_comm_url = 'https://api.pushshift.io/reddit/search/comment?sort=desc&sort_type=created_utc&size=500&author='+_redditor

        comment_dict = crawl_comments(base_comm_url, comment_dict)

    df_comm = pd.DataFrame(data=comment_dict)
    df_comm.sort_values(by=['author'], inplace=True, ignore_index=True)

    df_comm_AITA = df_comm[df_comm['subreddit']=='AmITheAsshole']

    # df_comm.to_csv(os.getcwd()+'/../data/reddit/Top'+str(threshold)+'redditors_comms.tsv', sep='\t', index_label='index')
    df_comm_AITA.to_csv(os.getcwd()+'/../data/reddit-morality/Comms__(Sub_AITA)__(Redditor_active).tsv', sep='\t', index_label='index')
    


if __name__ == '__main__':
    main()