import requests
from lxml import html
import numpy as np
import os


def extract_info(url, subId):
    page = requests.get(url)
    tree = html.fromstring(page.text)
    temp1 = tree.xpath('//div[@class="page_inner"]/*[last()]/text()')
    temp2 = tree.xpath('//div[@class="page_inner"]/*[last()-1]/text()')
    if len(temp1) == 0:
        numPage = 1
    elif temp1[0] == u'\u203a\u203a':
        numPage = int(temp2[0])
    elif '(' in temp1[0]:
        numPage = int(temp1[0].split(u'\xa0')[3])
    else:
        numPage = int(temp1[0])
    # Iterate through all pages to extract data
    result = []
    for i in range(1, numPage + 1):
        payload = {'page': str(i)}
        p = requests.get(url, params=payload)
        t = html.fromstring(p.text)
        s1 = '//li[@class="user odd" or @class="user"]' + \
             '/div/strong/a/span[2]/../@href'
        s2 = '//li[@class="user odd" or @class="user"]' + \
             '/div/strong/a/span[2]/@class'
        l1 = t.xpath(s1)
        l2 = t.xpath(s2)
        l1 = [e.split('/')[2] for e in l1]
        l2 = [e.split(' ')[0][5:] for e in l2]
        result = result + zip([str(subId)] * len(l1), l1, l2)
    return result


def scrape_subject_rating(url, subId):
    page = requests.get(url)
    tree = html.fromstring(page.text)
    s = tree.xpath("//ul[@id='navMenuNeue']/li/a/@class")
    if not any(['focus' in e for e in s]):
        return []
    s = tree.xpath("//ul[@id='navMenuNeue']/li[1]/a/@class")[0]
    result = []
    if 'focus' in s and 'anime' in s:
        for i in ['collections', 'dropped']:
            print "doing " + i
            result = result + extract_info(url + '/' + i, subId)
    return result


def scrape_ratings(frm, to):
    url = 'http://bgm.tv/subject/'
    n = frm
    result = []
    while (n <= to):
        print "starting subject " + str(n)
        temp = scrape_subject_rating(url + str(n), n)
        if temp is False:
            break
        else:
            result = result + temp
        print "finished subject " + str(n)
        n += 1
    path = os.path.join(os.pardir, os.pardir,
                        'data', 'ratings_{}_{}.csv'.format(frm, to))
    np.savetxt(path, np.array(result), delimiter=',', fmt='%s')


def main():
    for i in range(1, 125000, 5000):
        scrape_ratings(i, i + 5000 - 1)


if __name__ == '__main__':
    main()
