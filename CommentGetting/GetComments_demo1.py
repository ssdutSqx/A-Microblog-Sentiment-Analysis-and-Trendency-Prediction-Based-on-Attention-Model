from fake_useragent import UserAgent # 虚拟用户 防止封号
import re
import requests
import json
import pandas as pd
import time
import random
import csv



def get_one_page(url):
    list = []
    html = requests.get(url, headers=headers, cookies=cookies)
    jsondata = html.text
    # print(jsondata)
    data = json.loads(jsondata) # 这一步是把字符串解码
    # print(data)

    for comment in data['data']['data']:
        commenttext = comment['text']
        list.append(commenttext)

    return list


def parse_one_page(html_return):
    # pattern = re.compile(r'com.(\d+)"><img alt="(.+?)" src="(.+?)" usercard="(.+?)"></a>.*?</a>：(.+?)</div>', re.S)
    # data = re.findall(pattern, html_return)
    # print(data)
    comment_list = []
    for data in html_return:
        list = data.split("</a>:") # 如果有图片的问题解决不了
        if len(list) > 1:
            print(list[1])
            print()
            comment_list.append(list[1])
    return comment_list


# def write_to_file(data):
#     data_to_write = pd.DataFrame(data)
#     data_to_write.to_csv('test.csv', header=False, index=False, mode='a+')  # 去掉表头行和索引列


def main(i):
    url = 'https://m.weibo.cn/api/comments/show?id=4131150395559419&page=' + str(i)
    print(i,url)
    html_return = get_one_page(url)  # 去掉了不必要的参数后的url
    data = parse_one_page(html_return)

    f = open("CommentsDemo.txt", "w", encoding="utf-8")
    for text in data:
        f.write(text)
        f.write('\n')
    f.close()


headers = {'User-Agent': UserAgent().random}

cookies = {'Cookie': 'balabala'}


for i in range(1,3): # 这里存在问题 如果page大于一定数量就会出错
    main(i)
    time.sleep(random.uniform(2, 6))
