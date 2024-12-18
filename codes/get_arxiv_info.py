import requests
from lxml import etree
import arxiv
import json
import time
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.tmt.v20180321 import tmt_client, models
from tqdm import tqdm
from datetime import datetime


SecretId = ""
SecretKey = ""

def translate_tencent(text):
    try:
        cred = credential.Credential(SecretId, SecretKey)
        httpProfile = HttpProfile()
        httpProfile.endpoint = "tmt.tencentcloudapi.com"

        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        client = tmt_client.TmtClient(cred, "ap-shanghai", clientProfile)

        req = models.TextTranslateRequest()
        params = {
            "SourceText": text,
            "Source": "en",
            "Target": "zh",
            "ProjectId": 0
        }
        req.from_json_string(json.dumps(params))

        resp = client.TextTranslate(req)
        return resp.TargetText.replace('\n', ' ')

    except TencentCloudSDKException as err:
        return err
    

def main(url, task='main'):
    strhtml = requests.get(url)

    html = etree.HTML(strhtml.text)
    ids = html.xpath('//p[@class="list-title is-inline-block"]/a/text()')
    ids =  [c.replace('arXiv:','') for c in ids]

    search = arxiv.Search(
        id_list = ids,
        max_results = 50,
        sort_by = arxiv.SortCriterion.SubmittedDate,
        # sort_order = arxiv.SortOrder.Descending
    )

    papers = []
    papers_cn = []
    index = 1
    for result in tqdm(search.results()):
        authors = ""
        authors = authors + str(result.authors[0])
        for i in range(1, len(result.authors)):
            authors = authors + ", " + str(result.authors[i])
        # print(authors, '->', result.title)
        # print(translate_tencent(result.summary.replace('\n', '')))
        if result.comment:
            comment = result.comment.replace('\n', '') + "\n\n"
        else:
            comment = ""
        subdate = result.updated.strftime("%Y-%m-%d")
        if subdate == datetime.now().strftime("%Y-%m-%d"):
            subdate = "**NEW** " + subdate
        time.sleep(1)
        paper = "## **{}. {}**\n\n{}\n\n{}**SubmitDate**: {}    [abs]({}) [paper-pdf]({})\n\n**Authors**: {}\n\n**Abstract**: {}\n\n\n\n".format(
            index,
            result.title, 
            result.primary_category,
            comment,
            subdate, 
            result.entry_id, 
            result.pdf_url, 
            authors,
            result.summary.replace('\n', ' '))
        paper_cn = "## **{}. {}**\n\n{} {}\n\n{}**SubmitDate**: {}    [abs]({}) [paper-pdf]({})\n\n**Authors**: {}\n\n**Abstract**: {}\n\n摘要: {}\n\n\n\n".format(
            index,
            result.title, 
            translate_tencent(result.title), 
            result.primary_category,
            comment,
            subdate,
            result.entry_id, 
            result.pdf_url, 
            authors,
            result.summary.replace('\n', ' '), 
            translate_tencent(result.summary.replace('\n', ' ')))
        index += 1
        papers.append(paper)
        papers_cn.append(paper_cn)
        # print(paper)

    if task != 'LLM':
        fo = open("NewAdversarialAttackPaper/README_CN.md", "w", encoding= 'utf-8')
    else:
        fo = open("NewAdversarialAttackPaper/README_LLM_CN.md", "w", encoding= 'utf-8')

    if task != 'LLM':
        fo.write("# Latest Adversarial Attack Papers\n**update at {}**\n\n翻译来自 https://cloud.tencent.com/document/product/551/15619\n\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        for p in papers_cn:
            fo.write(p)
        fo.close()
    else:
        fo.write("# Latest Large Language Model Attack Papers\n**update at {}**\n\n翻译来自 https://cloud.tencent.com/document/product/551/15619\n\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        for p in papers_cn:
            fo.write(p)
        fo.close()

    if task != 'LLM':
        fo = open("NewAdversarialAttackPaper/README.md", "w", encoding= 'utf-8')
    else:
        fo = open("NewAdversarialAttackPaper/README_LLM.md", "w", encoding= 'utf-8')

    if task != 'LLM':
        fo.write("# Latest Adversarial Attack Papers\n**update at {}**\n\n[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)\n\n[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)\n\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        for p in papers:
            fo.write(p)
        fo.close()
    else:
        fo.write("# Latest Large Language Model Attack Papers\n**update at {}**\n\n[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)\n\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        for p in papers:
            fo.write(p)
        fo.close()

url = 'https://arxiv.org/search/?query=large+language+model+attack&searchtype=all&source=header&order=-submitted_date'
main(url, task='LLM')
url = 'https://arxiv.org/search/?query=adversarial+attack&searchtype=all&source=header&order=-submitted_date'
main(url, task='at')