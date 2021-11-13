import html
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


'''
Use Tencent translate to get chinese abstract
'''
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

url = 'https://arxiv.org/search/?query=adversarial+attack&searchtype=all&source=header'
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
    time.sleep(1)
    paper = "## **{}**\n\n{} {}\n\n{}**SubmitDate**: {}    [paper-pdf]({})\n\n**Authors**: {}\n\n**Abstracts**: {}\n\n摘要: {}\n\n\n\n".format(
        result.title, 
        translate_tencent(result.title), 
        result.primary_category,
        comment,
        result.updated.strftime("%Y-%m-%d"),
        result.pdf_url, 
        authors,
        result.summary.replace('\n', ' '), 
        translate_tencent(result.summary.replace('\n', ' ')), )
    papers.append(paper)
    # print(paper)

fo = open("README.md", "w", encoding= 'utf-8')

fo.write("# NewAdversarialAttackPaper\n")
for p in papers:
    fo.write(p)
fo.close()
