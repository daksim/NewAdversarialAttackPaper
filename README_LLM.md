# Latest Large Language Model Attack Papers
**update at 2024-08-13 18:57:01**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. A RAG-Based Question-Answering Solution for Cyber-Attack Investigation and Attribution**

cs.CR

Accepted at SECAI 2024 (ESORICS 2024)

**SubmitDate**: 2024-08-12    [abs](http://arxiv.org/abs/2408.06272v1) [paper-pdf](http://arxiv.org/pdf/2408.06272v1)

**Authors**: Sampath Rajapaksha, Ruby Rani, Erisa Karafili

**Abstract**: In the constantly evolving field of cybersecurity, it is imperative for analysts to stay abreast of the latest attack trends and pertinent information that aids in the investigation and attribution of cyber-attacks. In this work, we introduce the first question-answering (QA) model and its application that provides information to the cybersecurity experts about cyber-attacks investigations and attribution. Our QA model is based on Retrieval Augmented Generation (RAG) techniques together with a Large Language Model (LLM) and provides answers to the users' queries based on either our knowledge base (KB) that contains curated information about cyber-attacks investigations and attribution or on outside resources provided by the users. We have tested and evaluated our QA model with various types of questions, including KB-based, metadata-based, specific documents from the KB, and external sources-based questions. We compared the answers for KB-based questions with those from OpenAI's GPT-3.5 and the latest GPT-4o LLMs. Our proposed QA model outperforms OpenAI's GPT models by providing the source of the answers and overcoming the hallucination limitations of the GPT models, which is critical for cyber-attack investigation and attribution. Additionally, our analysis showed that when the RAG QA model is given few-shot examples rather than zero-shot instructions, it generates better answers compared to cases where no examples are supplied in addition to the query.



## **2. On Effects of Steering Latent Representation for Large Language Model Unlearning**

cs.CL

15 pages, 5 figures, 8 tables

**SubmitDate**: 2024-08-12    [abs](http://arxiv.org/abs/2408.06223v1) [paper-pdf](http://arxiv.org/pdf/2408.06223v1)

**Authors**: Dang Huu-Tien, Trung-Tin Pham, Hoang Thanh-Tung, Naoya Inoue

**Abstract**: Representation Misdirection for Unlearning (RMU), which steers model representation in the intermediate layer to a target random representation, is an effective method for large language model (LLM) unlearning. Despite its high performance, the underlying cause and explanation remain underexplored. In this paper, we first theoretically demonstrate that steering forget representations in the intermediate layer reduces token confidence, causing LLMs to generate wrong or nonsense responses. Second, we investigate how the coefficient influences the alignment of forget-sample representations with the random direction and hint at the optimal coefficient values for effective unlearning across different network layers. Third, we show that RMU unlearned models are robust against adversarial jailbreak attacks. Last, our empirical analysis shows that RMU is less effective when applied to the middle and later layers in LLMs. To resolve this drawback, we propose Adaptive RMU -- a simple yet effective alternative method that makes unlearning effective with most layers. Extensive experiments demonstrate that Adaptive RMU significantly improves the unlearning performance compared to prior art while incurring no additional computational cost.



## **3. Nob-MIAs: Non-biased Membership Inference Attacks Assessment on Large Language Models with Ex-Post Dataset Construction**

cs.CR

**SubmitDate**: 2024-08-12    [abs](http://arxiv.org/abs/2408.05968v1) [paper-pdf](http://arxiv.org/pdf/2408.05968v1)

**Authors**: Cédric Eichler, Nathan Champeil, Nicolas Anciaux, Alexandra Bensamoun, Heber Hwang Arcolezi, José Maria De Fuentes

**Abstract**: The rise of Large Language Models (LLMs) has triggered legal and ethical concerns, especially regarding the unauthorized use of copyrighted materials in their training datasets. This has led to lawsuits against tech companies accused of using protected content without permission. Membership Inference Attacks (MIAs) aim to detect whether specific documents were used in a given LLM pretraining, but their effectiveness is undermined by biases such as time-shifts and n-gram overlaps.   This paper addresses the evaluation of MIAs on LLMs with partially inferable training sets, under the ex-post hypothesis, which acknowledges inherent distributional biases between members and non-members datasets. We propose and validate algorithms to create ``non-biased'' and ``non-classifiable'' datasets for fairer MIA assessment. Experiments using the Gutenberg dataset on OpenLamma and Pythia show that neutralizing known biases alone is insufficient. Our methods produce non-biased ex-post datasets with AUC-ROC scores comparable to those previously obtained on genuinely random datasets, validating our approach. Globally, MIAs yield results close to random, with only one being effective on both random and our datasets, but its performance decreases when bias is removed.



## **4. Multimodal Large Language Models for Phishing Webpage Detection and Identification**

cs.CR

To appear in eCrime 2024

**SubmitDate**: 2024-08-12    [abs](http://arxiv.org/abs/2408.05941v1) [paper-pdf](http://arxiv.org/pdf/2408.05941v1)

**Authors**: Jehyun Lee, Peiyuan Lim, Bryan Hooi, Dinil Mon Divakaran

**Abstract**: To address the challenging problem of detecting phishing webpages, researchers have developed numerous solutions, in particular those based on machine learning (ML) algorithms. Among these, brand-based phishing detection that uses models from Computer Vision to detect if a given webpage is imitating a well-known brand has received widespread attention. However, such models are costly and difficult to maintain, as they need to be retrained with labeled dataset that has to be regularly and continuously collected. Besides, they also need to maintain a good reference list of well-known websites and related meta-data for effective performance.   In this work, we take steps to study the efficacy of large language models (LLMs), in particular the multimodal LLMs, in detecting phishing webpages. Given that the LLMs are pretrained on a large corpus of data, we aim to make use of their understanding of different aspects of a webpage (logo, theme, favicon, etc.) to identify the brand of a given webpage and compare the identified brand with the domain name in the URL to detect a phishing attack. We propose a two-phase system employing LLMs in both phases: the first phase focuses on brand identification, while the second verifies the domain. We carry out comprehensive evaluations on a newly collected dataset. Our experiments show that the LLM-based system achieves a high detection rate at high precision; importantly, it also provides interpretable evidence for the decisions. Our system also performs significantly better than a state-of-the-art brand-based phishing detection system while demonstrating robustness against two known adversarial attacks.



## **5. LLM-Based Robust Product Classification in Commerce and Compliance**

cs.CL

11 pages

**SubmitDate**: 2024-08-11    [abs](http://arxiv.org/abs/2408.05874v1) [paper-pdf](http://arxiv.org/pdf/2408.05874v1)

**Authors**: Sina Gholamian, Gianfranco Romani, Bartosz Rudnikowicz, Laura Skylaki

**Abstract**: Product classification is a crucial task in international trade, as compliance regulations are verified and taxes and duties are applied based on product categories. Manual classification of products is time-consuming and error-prone, and the sheer volume of products imported and exported renders the manual process infeasible. Consequently, e-commerce platforms and enterprises involved in international trade have turned to automatic product classification using machine learning. However, current approaches do not consider the real-world challenges associated with product classification, such as very abbreviated and incomplete product descriptions. In addition, recent advancements in generative Large Language Models (LLMs) and their reasoning capabilities are mainly untapped in product classification and e-commerce. In this research, we explore the real-life challenges of industrial classification and we propose data perturbations that allow for realistic data simulation. Furthermore, we employ LLM-based product classification to improve the robustness of the prediction in presence of incomplete data. Our research shows that LLMs with in-context learning outperform the supervised approaches in the clean-data scenario. Additionally, we illustrate that LLMs are significantly more robust than the supervised approaches when data attacks are present.



## **6. PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models**

cs.CR

To appear in USENIX Security Symposium 2025. The code is available at  https://github.com/sleeepeer/PoisonedRAG

**SubmitDate**: 2024-08-11    [abs](http://arxiv.org/abs/2402.07867v2) [paper-pdf](http://arxiv.org/pdf/2402.07867v2)

**Authors**: Wei Zou, Runpeng Geng, Binghui Wang, Jinyuan Jia

**Abstract**: Large language models (LLMs) have achieved remarkable success due to their exceptional generative capabilities. Despite their success, they also have inherent limitations such as a lack of up-to-date knowledge and hallucination. Retrieval-Augmented Generation (RAG) is a state-of-the-art technique to mitigate these limitations. The key idea of RAG is to ground the answer generation of an LLM on external knowledge retrieved from a knowledge database. Existing studies mainly focus on improving the accuracy or efficiency of RAG, leaving its security largely unexplored. We aim to bridge the gap in this work. We find that the knowledge database in a RAG system introduces a new and practical attack surface. Based on this attack surface, we propose PoisonedRAG, the first knowledge corruption attack to RAG, where an attacker could inject a few malicious texts into the knowledge database of a RAG system to induce an LLM to generate an attacker-chosen target answer for an attacker-chosen target question. We formulate knowledge corruption attacks as an optimization problem, whose solution is a set of malicious texts. Depending on the background knowledge (e.g., black-box and white-box settings) of an attacker on a RAG system, we propose two solutions to solve the optimization problem, respectively. Our results show PoisonedRAG could achieve a 90% attack success rate when injecting five malicious texts for each target question into a knowledge database with millions of texts. We also evaluate several defenses and our results show they are insufficient to defend against PoisonedRAG, highlighting the need for new defenses.



## **7. Using Retriever Augmented Large Language Models for Attack Graph Generation**

cs.CR

**SubmitDate**: 2024-08-11    [abs](http://arxiv.org/abs/2408.05855v1) [paper-pdf](http://arxiv.org/pdf/2408.05855v1)

**Authors**: Renascence Tarafder Prapty, Ashish Kundu, Arun Iyengar

**Abstract**: As the complexity of modern systems increases, so does the importance of assessing their security posture through effective vulnerability management and threat modeling techniques. One powerful tool in the arsenal of cybersecurity professionals is the attack graph, a representation of all potential attack paths within a system that an adversary might exploit to achieve a certain objective. Traditional methods of generating attack graphs involve expert knowledge, manual curation, and computational algorithms that might not cover the entire threat landscape due to the ever-evolving nature of vulnerabilities and exploits. This paper explores the approach of leveraging large language models (LLMs), such as ChatGPT, to automate the generation of attack graphs by intelligently chaining Common Vulnerabilities and Exposures (CVEs) based on their preconditions and effects. It also shows how to utilize LLMs to create attack graphs from threat reports.



## **8. Bot or Human? Detecting ChatGPT Imposters with A Single Question**

cs.CL

**SubmitDate**: 2024-08-11    [abs](http://arxiv.org/abs/2305.06424v4) [paper-pdf](http://arxiv.org/pdf/2305.06424v4)

**Authors**: Hong Wang, Xuan Luo, Weizhi Wang, Xifeng Yan

**Abstract**: Large language models (LLMs) like GPT-4 have recently demonstrated impressive capabilities in natural language understanding and generation. However, there is a concern that they can be misused for malicious purposes, such as fraud or denial-of-service attacks. Therefore, it is crucial to develop methods for detecting whether the party involved in a conversation is a bot or a human. In this paper, we propose a framework named FLAIR, Finding Large Language Model Authenticity via a Single Inquiry and Response, to detect conversational bots in an online manner. Specifically, we target a single question scenario that can effectively differentiate human users from bots. The questions are divided into two categories: those that are easy for humans but difficult for bots (e.g., counting, substitution, searching, and ASCII art reasoning), and those that are easy for bots but difficult for humans (e.g., memorization and computation). Our approach shows different strengths of these questions in their effectiveness, providing a new way for online service providers to protect themselves against nefarious activities. Our code and question set are available at https://github.com/hongwang600/FLAIR.



## **9. Unbridled Icarus: A Survey of the Potential Perils of Image Inputs in Multimodal Large Language Model Security**

cs.CR

8 pages, 1 figure. Accepted to 2024 IEEE International Conference on  Systems, Man, and Cybernetics

**SubmitDate**: 2024-08-11    [abs](http://arxiv.org/abs/2404.05264v2) [paper-pdf](http://arxiv.org/pdf/2404.05264v2)

**Authors**: Yihe Fan, Yuxin Cao, Ziyu Zhao, Ziyao Liu, Shaofeng Li

**Abstract**: Multimodal Large Language Models (MLLMs) demonstrate remarkable capabilities that increasingly influence various aspects of our daily lives, constantly defining the new boundary of Artificial General Intelligence (AGI). Image modalities, enriched with profound semantic information and a more continuous mathematical nature compared to other modalities, greatly enhance the functionalities of MLLMs when integrated. However, this integration serves as a double-edged sword, providing attackers with expansive vulnerabilities to exploit for highly covert and harmful attacks. The pursuit of reliable AI systems like powerful MLLMs has emerged as a pivotal area of contemporary research. In this paper, we endeavor to demostrate the multifaceted risks associated with the incorporation of image modalities into MLLMs. Initially, we delineate the foundational components and training processes of MLLMs. Subsequently, we construct a threat model, outlining the security vulnerabilities intrinsic to MLLMs. Moreover, we analyze and summarize existing scholarly discourses on MLLMs' attack and defense mechanisms, culminating in suggestions for the future research on MLLM security. Through this comprehensive analysis, we aim to deepen the academic understanding of MLLM security challenges and propel forward the development of trustworthy MLLM systems.



## **10. Utilizing Large Language Models to Optimize the Detection and Explainability of Phishing Websites**

cs.CR

**SubmitDate**: 2024-08-11    [abs](http://arxiv.org/abs/2408.05667v1) [paper-pdf](http://arxiv.org/pdf/2408.05667v1)

**Authors**: Sayak Saha Roy, Shirin Nilizadeh

**Abstract**: In this paper, we introduce PhishLang, an open-source, lightweight Large Language Model (LLM) specifically designed for phishing website detection through contextual analysis of the website. Unlike traditional heuristic or machine learning models that rely on static features and struggle to adapt to new threats and deep learning models that are computationally intensive, our model utilizes the advanced language processing capabilities of LLMs to learn granular features that are characteristic of phishing attacks. Furthermore, PhishLang operates with minimal data preprocessing and offers performance comparable to leading deep learning tools, while being significantly faster and less resource-intensive. Over a 3.5-month testing period, PhishLang successfully identified approximately 26K phishing URLs, many of which were undetected by popular antiphishing blocklists, thus demonstrating its potential to aid current detection measures. We also evaluate PhishLang against several realistic adversarial attacks and develop six patches that make it very robust against such threats. Furthermore, we integrate PhishLang with GPT-3.5 Turbo to create \textit{explainable blocklisting} - warnings that provide users with contextual information about different features that led to a website being marked as phishing. Finally, we have open-sourced the PhishLang framework and developed a Chromium-based browser extension and URL scanner website, which implement explainable warnings for end-users.



## **11. Preserving Privacy in Large Language Models: A Survey on Current Threats and Solutions**

cs.CR

GitHub repository:  https://github.com/michele17284/Awesome-Privacy-Preserving-LLMs

**SubmitDate**: 2024-08-10    [abs](http://arxiv.org/abs/2408.05212v1) [paper-pdf](http://arxiv.org/pdf/2408.05212v1)

**Authors**: Michele Miranda, Elena Sofia Ruzzetti, Andrea Santilli, Fabio Massimo Zanzotto, Sébastien Bratières, Emanuele Rodolà

**Abstract**: Large Language Models (LLMs) represent a significant advancement in artificial intelligence, finding applications across various domains. However, their reliance on massive internet-sourced datasets for training brings notable privacy issues, which are exacerbated in critical domains (e.g., healthcare). Moreover, certain application-specific scenarios may require fine-tuning these models on private data. This survey critically examines the privacy threats associated with LLMs, emphasizing the potential for these models to memorize and inadvertently reveal sensitive information. We explore current threats by reviewing privacy attacks on LLMs and propose comprehensive solutions for integrating privacy mechanisms throughout the entire learning pipeline. These solutions range from anonymizing training datasets to implementing differential privacy during training or inference and machine unlearning after training. Our comprehensive review of existing literature highlights ongoing challenges, available tools, and future directions for preserving privacy in LLMs. This work aims to guide the development of more secure and trustworthy AI systems by providing a thorough understanding of privacy preservation methods and their effectiveness in mitigating risks.



## **12. Towards Resilient and Efficient LLMs: A Comparative Study of Efficiency, Performance, and Adversarial Robustness**

cs.CL

**SubmitDate**: 2024-08-09    [abs](http://arxiv.org/abs/2408.04585v2) [paper-pdf](http://arxiv.org/pdf/2408.04585v2)

**Authors**: Xiaojing Fan, Chunliang Tao

**Abstract**: With the increasing demand for practical applications of Large Language Models (LLMs), many attention-efficient models have been developed to balance performance and computational cost. However, the adversarial robustness of these models remains under-explored. In this work, we design a framework to investigate the trade-off between efficiency, performance, and adversarial robustness of LLMs by comparing three prominent models with varying levels of complexity and efficiency -- Transformer++, Gated Linear Attention (GLA) Transformer, and MatMul-Free LM -- utilizing the GLUE and AdvGLUE datasets. The AdvGLUE dataset extends the GLUE dataset with adversarial samples designed to challenge model robustness. Our results show that while the GLA Transformer and MatMul-Free LM achieve slightly lower accuracy on GLUE tasks, they demonstrate higher efficiency and either superior or comparative robustness on AdvGLUE tasks compared to Transformer++ across different attack levels. These findings highlight the potential of simplified architectures to achieve a compelling balance between efficiency, performance, and adversarial robustness, offering valuable insights for applications where resource constraints and resilience to adversarial attacks are critical.



## **13. AttackER: Towards Enhancing Cyber-Attack Attribution with a Named Entity Recognition Dataset**

cs.CR

Submitted to WISE 2024

**SubmitDate**: 2024-08-09    [abs](http://arxiv.org/abs/2408.05149v1) [paper-pdf](http://arxiv.org/pdf/2408.05149v1)

**Authors**: Pritam Deka, Sampath Rajapaksha, Ruby Rani, Amirah Almutairi, Erisa Karafili

**Abstract**: Cyber-attack attribution is an important process that allows experts to put in place attacker-oriented countermeasures and legal actions. The analysts mainly perform attribution manually, given the complex nature of this task. AI and, more specifically, Natural Language Processing (NLP) techniques can be leveraged to support cybersecurity analysts during the attribution process. However powerful these techniques are, they need to deal with the lack of datasets in the attack attribution domain. In this work, we will fill this gap and will provide, to the best of our knowledge, the first dataset on cyber-attack attribution. We designed our dataset with the primary goal of extracting attack attribution information from cybersecurity texts, utilizing named entity recognition (NER) methodologies from the field of NLP. Unlike other cybersecurity NER datasets, ours offers a rich set of annotations with contextual details, including some that span phrases and sentences. We conducted extensive experiments and applied NLP techniques to demonstrate the dataset's effectiveness for attack attribution. These experiments highlight the potential of Large Language Models (LLMs) capabilities to improve the NER tasks in cybersecurity datasets for cyber-attack attribution.



## **14. ConfusedPilot: Compromising Enterprise Information Integrity and Confidentiality with Copilot for Microsoft 365**

cs.CR

**SubmitDate**: 2024-08-09    [abs](http://arxiv.org/abs/2408.04870v1) [paper-pdf](http://arxiv.org/pdf/2408.04870v1)

**Authors**: Ayush RoyChowdhury, Mulong Luo, Prateek Sahu, Sarbartha Banerjee, Mohit Tiwari

**Abstract**: Retrieval augmented generation (RAG) is a process where a large language model (LLM) retrieves useful information from a database and then generates the responses. It is becoming popular in enterprise settings for daily business operations. For example, Copilot for Microsoft 365 has accumulated millions of businesses. However, the security implications of adopting such RAG-based systems are unclear.   In this paper, we introduce ConfusedPilot, a class of security vulnerabilities of RAG systems that confuse Copilot and cause integrity and confidentiality violations in its responses. First, we investigate a vulnerability that embeds malicious text in the modified prompt in RAG, corrupting the responses generated by the LLM. Second, we demonstrate a vulnerability that leaks secret data, which leverages the caching mechanism during retrieval. Third, we investigate how both vulnerabilities can be exploited to propagate misinformation within the enterprise and ultimately impact its operations, such as sales and manufacturing. We also discuss the root cause of these attacks by investigating the architecture of a RAG-based system. This study highlights the security vulnerabilities in today's RAG-based systems and proposes design guidelines to secure future RAG-based systems.



## **15. ChatGPT Meets Iris Biometrics**

cs.CV

Published at IJCB 2024

**SubmitDate**: 2024-08-09    [abs](http://arxiv.org/abs/2408.04868v1) [paper-pdf](http://arxiv.org/pdf/2408.04868v1)

**Authors**: Parisa Farmanifard, Arun Ross

**Abstract**: This study utilizes the advanced capabilities of the GPT-4 multimodal Large Language Model (LLM) to explore its potential in iris recognition - a field less common and more specialized than face recognition. By focusing on this niche yet crucial area, we investigate how well AI tools like ChatGPT can understand and analyze iris images. Through a series of meticulously designed experiments employing a zero-shot learning approach, the capabilities of ChatGPT-4 was assessed across various challenging conditions including diverse datasets, presentation attacks, occlusions such as glasses, and other real-world variations. The findings convey ChatGPT-4's remarkable adaptability and precision, revealing its proficiency in identifying distinctive iris features, while also detecting subtle effects like makeup on iris recognition. A comparative analysis with Gemini Advanced - Google's AI model - highlighted ChatGPT-4's better performance and user experience in complex iris analysis tasks. This research not only validates the use of LLMs for specialized biometric applications but also emphasizes the importance of nuanced query framing and interaction design in extracting significant insights from biometric data. Our findings suggest a promising path for future research and the development of more adaptable, efficient, robust and interactive biometric security solutions.



## **16. h4rm3l: A Dynamic Benchmark of Composable Jailbreak Attacks for LLM Safety Assessment**

cs.CR

**SubmitDate**: 2024-08-09    [abs](http://arxiv.org/abs/2408.04811v1) [paper-pdf](http://arxiv.org/pdf/2408.04811v1)

**Authors**: Moussa Koulako Bala Doumbouya, Ananjan Nandi, Gabriel Poesia, Davide Ghilardi, Anna Goldie, Federico Bianchi, Dan Jurafsky, Christopher D. Manning

**Abstract**: The safety of Large Language Models (LLMs) remains a critical concern due to a lack of adequate benchmarks for systematically evaluating their ability to resist generating harmful content. Previous efforts towards automated red teaming involve static or templated sets of illicit requests and adversarial prompts which have limited utility given jailbreak attacks' evolving and composable nature. We propose a novel dynamic benchmark of composable jailbreak attacks to move beyond static datasets and taxonomies of attacks and harms. Our approach consists of three components collectively called h4rm3l: (1) a domain-specific language that formally expresses jailbreak attacks as compositions of parameterized prompt transformation primitives, (2) bandit-based few-shot program synthesis algorithms that generate novel attacks optimized to penetrate the safety filters of a target black box LLM, and (3) open-source automated red-teaming software employing the previous two components. We use h4rm3l to generate a dataset of 2656 successful novel jailbreak attacks targeting 6 state-of-the-art (SOTA) open-source and proprietary LLMs. Several of our synthesized attacks are more effective than previously reported ones, with Attack Success Rates exceeding 90% on SOTA closed language models such as claude-3-haiku and GPT4-o. By generating datasets of jailbreak attacks in a unified formal representation, h4rm3l enables reproducible benchmarking and automated red-teaming, contributes to understanding LLM safety limitations, and supports the development of robust defenses in an increasingly LLM-integrated world.   Warning: This paper and related research artifacts contain offensive and potentially disturbing prompts and model-generated content.



## **17. Tamper-Resistant Safeguards for Open-Weight LLMs**

cs.LG

Website: https://www.tamper-resistant-safeguards.com

**SubmitDate**: 2024-08-08    [abs](http://arxiv.org/abs/2408.00761v2) [paper-pdf](http://arxiv.org/pdf/2408.00761v2)

**Authors**: Rishub Tamirisa, Bhrugu Bharathi, Long Phan, Andy Zhou, Alice Gatti, Tarun Suresh, Maxwell Lin, Justin Wang, Rowan Wang, Ron Arel, Andy Zou, Dawn Song, Bo Li, Dan Hendrycks, Mantas Mazeika

**Abstract**: Rapid advances in the capabilities of large language models (LLMs) have raised widespread concerns regarding their potential for malicious use. Open-weight LLMs present unique challenges, as existing safeguards lack robustness to tampering attacks that modify model weights. For example, recent works have demonstrated that refusal and unlearning safeguards can be trivially removed with a few steps of fine-tuning. These vulnerabilities necessitate new approaches for enabling the safe release of open-weight LLMs. We develop a method, called TAR, for building tamper-resistant safeguards into open-weight LLMs such that adversaries cannot remove the safeguards even after thousands of steps of fine-tuning. In extensive evaluations and red teaming analyses, we find that our method greatly improves tamper-resistance while preserving benign capabilities. Our results demonstrate that tamper-resistance is a tractable problem, opening up a promising new avenue to improve the safety and security of open-weight LLMs.



## **18. Ensemble everything everywhere: Multi-scale aggregation for adversarial robustness**

cs.CV

34 pages, 25 figures, appendix

**SubmitDate**: 2024-08-08    [abs](http://arxiv.org/abs/2408.05446v1) [paper-pdf](http://arxiv.org/pdf/2408.05446v1)

**Authors**: Stanislav Fort, Balaji Lakshminarayanan

**Abstract**: Adversarial examples pose a significant challenge to the robustness, reliability and alignment of deep neural networks. We propose a novel, easy-to-use approach to achieving high-quality representations that lead to adversarial robustness through the use of multi-resolution input representations and dynamic self-ensembling of intermediate layer predictions. We demonstrate that intermediate layer predictions exhibit inherent robustness to adversarial attacks crafted to fool the full classifier, and propose a robust aggregation mechanism based on Vickrey auction that we call \textit{CrossMax} to dynamically ensemble them. By combining multi-resolution inputs and robust ensembling, we achieve significant adversarial robustness on CIFAR-10 and CIFAR-100 datasets without any adversarial training or extra data, reaching an adversarial accuracy of $\approx$72% (CIFAR-10) and $\approx$48% (CIFAR-100) on the RobustBench AutoAttack suite ($L_\infty=8/255)$ with a finetuned ImageNet-pretrained ResNet152. This represents a result comparable with the top three models on CIFAR-10 and a +5 % gain compared to the best current dedicated approach on CIFAR-100. Adding simple adversarial training on top, we get $\approx$78% on CIFAR-10 and $\approx$51% on CIFAR-100, improving SOTA by 5 % and 9 % respectively and seeing greater gains on the harder dataset. We validate our approach through extensive experiments and provide insights into the interplay between adversarial robustness, and the hierarchical nature of deep representations. We show that simple gradient-based attacks against our model lead to human-interpretable images of the target classes as well as interpretable image changes. As a byproduct, using our multi-resolution prior, we turn pre-trained classifiers and CLIP models into controllable image generators and develop successful transferable attacks on large vision language models.



## **19. Duwak: Dual Watermarks in Large Language Models**

cs.LG

**SubmitDate**: 2024-08-08    [abs](http://arxiv.org/abs/2403.13000v2) [paper-pdf](http://arxiv.org/pdf/2403.13000v2)

**Authors**: Chaoyi Zhu, Jeroen Galjaard, Pin-Yu Chen, Lydia Y. Chen

**Abstract**: As large language models (LLM) are increasingly used for text generation tasks, it is critical to audit their usages, govern their applications, and mitigate their potential harms. Existing watermark techniques are shown effective in embedding single human-imperceptible and machine-detectable patterns without significantly affecting generated text quality and semantics. However, the efficiency in detecting watermarks, i.e., the minimum number of tokens required to assert detection with significance and robustness against post-editing, is still debatable. In this paper, we propose, Duwak, to fundamentally enhance the efficiency and quality of watermarking by embedding dual secret patterns in both token probability distribution and sampling schemes. To mitigate expression degradation caused by biasing toward certain tokens, we design a contrastive search to watermark the sampling scheme, which minimizes the token repetition and enhances the diversity. We theoretically explain the interdependency of the two watermarks within Duwak. We evaluate Duwak extensively on Llama2 under various post-editing attacks, against four state-of-the-art watermarking techniques and combinations of them. Our results show that Duwak marked text achieves the highest watermarked text quality at the lowest required token count for detection, up to 70% tokens less than existing approaches, especially under post paraphrasing.



## **20. Towards Explainable Network Intrusion Detection using Large Language Models**

cs.CR

**SubmitDate**: 2024-08-08    [abs](http://arxiv.org/abs/2408.04342v1) [paper-pdf](http://arxiv.org/pdf/2408.04342v1)

**Authors**: Paul R. B. Houssel, Priyanka Singh, Siamak Layeghy, Marius Portmann

**Abstract**: Large Language Models (LLMs) have revolutionised natural language processing tasks, particularly as chat agents. However, their applicability to threat detection problems remains unclear. This paper examines the feasibility of employing LLMs as a Network Intrusion Detection System (NIDS), despite their high computational requirements, primarily for the sake of explainability. Furthermore, considerable resources have been invested in developing LLMs, and they may offer utility for NIDS. Current state-of-the-art NIDS rely on artificial benchmarking datasets, resulting in skewed performance when applied to real-world networking environments. Therefore, we compare the GPT-4 and LLama3 models against traditional architectures and transformer-based models to assess their ability to detect malicious NetFlows without depending on artificially skewed datasets, but solely on their vast pre-trained acquired knowledge. Our results reveal that, although LLMs struggle with precise attack detection, they hold significant potential for a path towards explainable NIDS. Our preliminary exploration shows that LLMs are unfit for the detection of Malicious NetFlows. Most promisingly, however, these exhibit significant potential as complementary agents in NIDS, particularly in providing explanations and aiding in threat response when integrated with Retrieval Augmented Generation (RAG) and function calling capabilities.



## **21. Multi-Turn Context Jailbreak Attack on Large Language Models From First Principles**

cs.CL

**SubmitDate**: 2024-08-08    [abs](http://arxiv.org/abs/2408.04686v1) [paper-pdf](http://arxiv.org/pdf/2408.04686v1)

**Authors**: Xiongtao Sun, Deyue Zhang, Dongdong Yang, Quanchen Zou, Hui Li

**Abstract**: Large language models (LLMs) have significantly enhanced the performance of numerous applications, from intelligent conversations to text generation. However, their inherent security vulnerabilities have become an increasingly significant challenge, especially with respect to jailbreak attacks. Attackers can circumvent the security mechanisms of these LLMs, breaching security constraints and causing harmful outputs. Focusing on multi-turn semantic jailbreak attacks, we observe that existing methods lack specific considerations for the role of multiturn dialogues in attack strategies, leading to semantic deviations during continuous interactions. Therefore, in this paper, we establish a theoretical foundation for multi-turn attacks by considering their support in jailbreak attacks, and based on this, propose a context-based contextual fusion black-box jailbreak attack method, named Context Fusion Attack (CFA). This method approach involves filtering and extracting key terms from the target, constructing contextual scenarios around these terms, dynamically integrating the target into the scenarios, replacing malicious key terms within the target, and thereby concealing the direct malicious intent. Through comparisons on various mainstream LLMs and red team datasets, we have demonstrated CFA's superior success rate, divergence, and harmfulness compared to other multi-turn attack strategies, particularly showcasing significant advantages on Llama3 and GPT-4.



## **22. Effective Prompt Extraction from Language Models**

cs.CL

**SubmitDate**: 2024-08-07    [abs](http://arxiv.org/abs/2307.06865v3) [paper-pdf](http://arxiv.org/pdf/2307.06865v3)

**Authors**: Yiming Zhang, Nicholas Carlini, Daphne Ippolito

**Abstract**: The text generated by large language models is commonly controlled by prompting, where a prompt prepended to a user's query guides the model's output. The prompts used by companies to guide their models are often treated as secrets, to be hidden from the user making the query. They have even been treated as commodities to be bought and sold on marketplaces. However, anecdotal reports have shown adversarial users employing prompt extraction attacks to recover these prompts. In this paper, we present a framework for systematically measuring the effectiveness of these attacks. In experiments with 3 different sources of prompts and 11 underlying large language models, we find that simple text-based attacks can in fact reveal prompts with high probability. Our framework determines with high precision whether an extracted prompt is the actual secret prompt, rather than a model hallucination. Prompt extraction from real systems such as Claude 3 and ChatGPT further suggest that system prompts can be revealed by an adversary despite existing defenses in place.



## **23. Decoding Biases: Automated Methods and LLM Judges for Gender Bias Detection in Language Models**

cs.CL

6 pages paper content, 17 pages of appendix

**SubmitDate**: 2024-08-07    [abs](http://arxiv.org/abs/2408.03907v1) [paper-pdf](http://arxiv.org/pdf/2408.03907v1)

**Authors**: Shachi H Kumar, Saurav Sahay, Sahisnu Mazumder, Eda Okur, Ramesh Manuvinakurike, Nicole Beckage, Hsuan Su, Hung-yi Lee, Lama Nachman

**Abstract**: Large Language Models (LLMs) have excelled at language understanding and generating human-level text. However, even with supervised training and human alignment, these LLMs are susceptible to adversarial attacks where malicious users can prompt the model to generate undesirable text. LLMs also inherently encode potential biases that can cause various harmful effects during interactions. Bias evaluation metrics lack standards as well as consensus and existing methods often rely on human-generated templates and annotations which are expensive and labor intensive. In this work, we train models to automatically create adversarial prompts to elicit biased responses from target LLMs. We present LLM- based bias evaluation metrics and also analyze several existing automatic evaluation methods and metrics. We analyze the various nuances of model responses, identify the strengths and weaknesses of model families, and assess where evaluation methods fall short. We compare these metrics to human evaluation and validate that the LLM-as-a-Judge metric aligns with human judgement on bias in response generation.



## **24. Trading Devil Final: Backdoor attack via Stock market and Bayesian Optimization**

cs.LG

END :jumps-Diffusion and stock market: Better quantify uncertainty in  financial simulations

**SubmitDate**: 2024-08-07    [abs](http://arxiv.org/abs/2407.14573v2) [paper-pdf](http://arxiv.org/pdf/2407.14573v2)

**Authors**: Orson Mengara

**Abstract**: Since the advent of generative artificial intelligence, every company and researcher has been rushing to develop their own generative models, whether commercial or not. Given the large number of users of these powerful new tools, there is currently no intrinsically verifiable way to explain from the ground up what happens when LLMs (large language models) learn. For example, those based on automatic speech recognition systems, which have to rely on huge and astronomical amounts of data collected from all over the web to produce fast and efficient results, In this article, we develop a backdoor attack called MarketBackFinal 2.0, based on acoustic data poisoning, MarketBackFinal 2.0 is mainly based on modern stock market models. In order to show the possible vulnerabilities of speech-based transformers that may rely on LLMs.



## **25. EnJa: Ensemble Jailbreak on Large Language Models**

cs.CR

**SubmitDate**: 2024-08-07    [abs](http://arxiv.org/abs/2408.03603v1) [paper-pdf](http://arxiv.org/pdf/2408.03603v1)

**Authors**: Jiahao Zhang, Zilong Wang, Ruofan Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: As Large Language Models (LLMs) are increasingly being deployed in safety-critical applications, their vulnerability to potential jailbreaks -- malicious prompts that can disable the safety mechanism of LLMs -- has attracted growing research attention. While alignment methods have been proposed to protect LLMs from jailbreaks, many have found that aligned LLMs can still be jailbroken by carefully crafted malicious prompts, producing content that violates policy regulations. Existing jailbreak attacks on LLMs can be categorized into prompt-level methods which make up stories/logic to circumvent safety alignment and token-level attack methods which leverage gradient methods to find adversarial tokens. In this work, we introduce the concept of Ensemble Jailbreak and explore methods that can integrate prompt-level and token-level jailbreak into a more powerful hybrid jailbreak attack. Specifically, we propose a novel EnJa attack to hide harmful instructions using prompt-level jailbreak, boost the attack success rate using a gradient-based attack, and connect the two types of jailbreak attacks via a template-based connector. We evaluate the effectiveness of EnJa on several aligned models and show that it achieves a state-of-the-art attack success rate with fewer queries and is much stronger than any individual jailbreak.



## **26. Empirical Analysis of Large Vision-Language Models against Goal Hijacking via Visual Prompt Injection**

cs.CL

8 pages, 6 figures, Accepted to NAACL 2024 SRW

**SubmitDate**: 2024-08-07    [abs](http://arxiv.org/abs/2408.03554v1) [paper-pdf](http://arxiv.org/pdf/2408.03554v1)

**Authors**: Subaru Kimura, Ryota Tanaka, Shumpei Miyawaki, Jun Suzuki, Keisuke Sakaguchi

**Abstract**: We explore visual prompt injection (VPI) that maliciously exploits the ability of large vision-language models (LVLMs) to follow instructions drawn onto the input image. We propose a new VPI method, "goal hijacking via visual prompt injection" (GHVPI), that swaps the execution task of LVLMs from an original task to an alternative task designated by an attacker. The quantitative analysis indicates that GPT-4V is vulnerable to the GHVPI and demonstrates a notable attack success rate of 15.8%, which is an unignorable security risk. Our analysis also shows that successful GHVPI requires high character recognition capability and instruction-following ability in LVLMs.



## **27. A Study on Prompt Injection Attack Against LLM-Integrated Mobile Robotic Systems**

cs.RO

**SubmitDate**: 2024-08-07    [abs](http://arxiv.org/abs/2408.03515v1) [paper-pdf](http://arxiv.org/pdf/2408.03515v1)

**Authors**: Wenxiao Zhang, Xiangrui Kong, Conan Dewitt, Thomas Braunl, Jin B. Hong

**Abstract**: The integration of Large Language Models (LLMs) like GPT-4o into robotic systems represents a significant advancement in embodied artificial intelligence. These models can process multi-modal prompts, enabling them to generate more context-aware responses. However, this integration is not without challenges. One of the primary concerns is the potential security risks associated with using LLMs in robotic navigation tasks. These tasks require precise and reliable responses to ensure safe and effective operation. Multi-modal prompts, while enhancing the robot's understanding, also introduce complexities that can be exploited maliciously. For instance, adversarial inputs designed to mislead the model can lead to incorrect or dangerous navigational decisions. This study investigates the impact of prompt injections on mobile robot performance in LLM-integrated systems and explores secure prompt strategies to mitigate these risks. Our findings demonstrate a substantial overall improvement of approximately 30.8% in both attack detection and system performance with the implementation of robust defence mechanisms, highlighting their critical role in enhancing security and reliability in mission-oriented tasks.



## **28. Best-of-Venom: Attacking RLHF by Injecting Poisoned Preference Data**

cs.CL

**SubmitDate**: 2024-08-06    [abs](http://arxiv.org/abs/2404.05530v2) [paper-pdf](http://arxiv.org/pdf/2404.05530v2)

**Authors**: Tim Baumgärtner, Yang Gao, Dana Alon, Donald Metzler

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) is a popular method for aligning Language Models (LM) with human values and preferences. RLHF requires a large number of preference pairs as training data, which are often used in both the Supervised Fine-Tuning and Reward Model training and therefore publicly available datasets are commonly used. In this work, we study to what extent a malicious actor can manipulate the LMs generations by poisoning the preferences, i.e., injecting poisonous preference pairs into these datasets and the RLHF training process. We propose strategies to build poisonous preference pairs and test their performance by poisoning two widely used preference datasets. Our results show that preference poisoning is highly effective: injecting a small amount of poisonous data (1-5\% of the original dataset), we can effectively manipulate the LM to generate a target entity in a target sentiment (positive or negative). The findings from our experiments also shed light on strategies to defend against the preference poisoning attack.



## **29. Rethinking Jailbreaking through the Lens of Representation Engineering**

cs.CL

21 pages, 20 figures, 6 tables

**SubmitDate**: 2024-08-06    [abs](http://arxiv.org/abs/2401.06824v3) [paper-pdf](http://arxiv.org/pdf/2401.06824v3)

**Authors**: Tianlong Li, Shihan Dou, Wenhao Liu, Muling Wu, Changze Lv, Rui Zheng, Xiaoqing Zheng, Xuanjing Huang

**Abstract**: The recent surge in jailbreaking methods has revealed the vulnerability of Large Language Models (LLMs) to malicious inputs. While earlier research has primarily concentrated on increasing the success rates of jailbreaking attacks, the underlying mechanism for safeguarding LLMs remains underexplored. This study investigates the vulnerability of safety-aligned LLMs by uncovering specific activity patterns within the representation space generated by LLMs. Such ``safety patterns'' can be identified with only a few pairs of contrastive queries in a simple method and function as ``keys'' (used as a metaphor for security defense capability) that can be used to open or lock Pandora's Box of LLMs. Extensive experiments demonstrate that the robustness of LLMs against jailbreaking can be lessened or augmented by attenuating or strengthening the identified safety patterns. These findings deepen our understanding of jailbreaking phenomena and call for the LLM community to address the potential misuse of open-source LLMs.



## **30. Compromising Embodied Agents with Contextual Backdoor Attacks**

cs.AI

**SubmitDate**: 2024-08-06    [abs](http://arxiv.org/abs/2408.02882v1) [paper-pdf](http://arxiv.org/pdf/2408.02882v1)

**Authors**: Aishan Liu, Yuguang Zhou, Xianglong Liu, Tianyuan Zhang, Siyuan Liang, Jiakai Wang, Yanjun Pu, Tianlin Li, Junqi Zhang, Wenbo Zhou, Qing Guo, Dacheng Tao

**Abstract**: Large language models (LLMs) have transformed the development of embodied intelligence. By providing a few contextual demonstrations, developers can utilize the extensive internal knowledge of LLMs to effortlessly translate complex tasks described in abstract language into sequences of code snippets, which will serve as the execution logic for embodied agents. However, this paper uncovers a significant backdoor security threat within this process and introduces a novel method called \method{}. By poisoning just a few contextual demonstrations, attackers can covertly compromise the contextual environment of a black-box LLM, prompting it to generate programs with context-dependent defects. These programs appear logically sound but contain defects that can activate and induce unintended behaviors when the operational agent encounters specific triggers in its interactive environment. To compromise the LLM's contextual environment, we employ adversarial in-context generation to optimize poisoned demonstrations, where an LLM judge evaluates these poisoned prompts, reporting to an additional LLM that iteratively optimizes the demonstration in a two-player adversarial game using chain-of-thought reasoning. To enable context-dependent behaviors in downstream agents, we implement a dual-modality activation strategy that controls both the generation and execution of program defects through textual and visual triggers. We expand the scope of our attack by developing five program defect modes that compromise key aspects of confidentiality, integrity, and availability in embodied agents. To validate the effectiveness of our approach, we conducted extensive experiments across various tasks, including robot planning, robot manipulation, and compositional visual reasoning. Additionally, we demonstrate the potential impact of our approach by successfully attacking real-world autonomous driving systems.



## **31. SEAS: Self-Evolving Adversarial Safety Optimization for Large Language Models**

cs.CL

**SubmitDate**: 2024-08-05    [abs](http://arxiv.org/abs/2408.02632v1) [paper-pdf](http://arxiv.org/pdf/2408.02632v1)

**Authors**: Muxi Diao, Rumei Li, Shiyang Liu, Guogang Liao, Jingang Wang, Xunliang Cai, Weiran Xu

**Abstract**: As large language models (LLMs) continue to advance in capability and influence, ensuring their security and preventing harmful outputs has become crucial. A promising approach to address these concerns involves training models to automatically generate adversarial prompts for red teaming. However, the evolving subtlety of vulnerabilities in LLMs challenges the effectiveness of current adversarial methods, which struggle to specifically target and explore the weaknesses of these models. To tackle these challenges, we introduce the $\mathbf{S}\text{elf-}\mathbf{E}\text{volving }\mathbf{A}\text{dversarial }\mathbf{S}\text{afety }\mathbf{(SEAS)}$ optimization framework, which enhances security by leveraging data generated by the model itself. SEAS operates through three iterative stages: Initialization, Attack, and Adversarial Optimization, refining both the Red Team and Target models to improve robustness and safety. This framework reduces reliance on manual testing and significantly enhances the security capabilities of LLMs. Our contributions include a novel adversarial framework, a comprehensive safety dataset, and after three iterations, the Target model achieves a security level comparable to GPT-4, while the Red Team model shows a marked increase in attack success rate (ASR) against advanced models.



## **32. Practical Attacks against Black-box Code Completion Engines**

cs.CR

**SubmitDate**: 2024-08-05    [abs](http://arxiv.org/abs/2408.02509v1) [paper-pdf](http://arxiv.org/pdf/2408.02509v1)

**Authors**: Slobodan Jenko, Jingxuan He, Niels Mündler, Mark Vero, Martin Vechev

**Abstract**: Modern code completion engines, powered by large language models, have demonstrated impressive capabilities to generate functionally correct code based on surrounding context. As these tools are extensively used by millions of developers, it is crucial to investigate their security implications. In this work, we present INSEC, a novel attack that directs code completion engines towards generating vulnerable code. In line with most commercial completion engines, such as GitHub Copilot, INSEC assumes only black-box query access to the targeted engine, without requiring any knowledge of the engine's internals. Our attack works by inserting a malicious attack string as a short comment in the completion input. To derive the attack string, we design a series of specialized initialization schemes and an optimization procedure for further refinement. We demonstrate the strength of INSEC not only on state-of-the-art open-source models but also on black-box commercial services such as the OpenAI API and GitHub Copilot. On a comprehensive set of security-critical test cases covering 16 CWEs across 5 programming languages, INSEC significantly increases the likelihood of the considered completion engines in generating unsafe code by >50% in absolute, while maintaining the ability in producing functionally correct code. At the same time, our attack has low resource requirements, and can be developed for a cost of well under ten USD on commodity hardware.



## **33. Why Are My Prompts Leaked? Unraveling Prompt Extraction Threats in Customized Large Language Models**

cs.CL

**SubmitDate**: 2024-08-05    [abs](http://arxiv.org/abs/2408.02416v1) [paper-pdf](http://arxiv.org/pdf/2408.02416v1)

**Authors**: Zi Liang, Haibo Hu, Qingqing Ye, Yaxin Xiao, Haoyang Li

**Abstract**: The drastic increase of large language models' (LLMs) parameters has led to a new research direction of fine-tuning-free downstream customization by prompts, i.e., task descriptions. While these prompt-based services (e.g. OpenAI's GPTs) play an important role in many businesses, there has emerged growing concerns about the prompt leakage, which undermines the intellectual properties of these services and causes downstream attacks. In this paper, we analyze the underlying mechanism of prompt leakage, which we refer to as prompt memorization, and develop corresponding defending strategies. By exploring the scaling laws in prompt extraction, we analyze key attributes that influence prompt extraction, including model sizes, prompt lengths, as well as the types of prompts. Then we propose two hypotheses that explain how LLMs expose their prompts. The first is attributed to the perplexity, i.e. the familiarity of LLMs to texts, whereas the second is based on the straightforward token translation path in attention matrices. To defend against such threats, we investigate whether alignments can undermine the extraction of prompts. We find that current LLMs, even those with safety alignments like GPT-4, are highly vulnerable to prompt extraction attacks, even under the most straightforward user attacks. Therefore, we put forward several defense strategies with the inspiration of our findings, which achieve 83.8\% and 71.0\% drop in the prompt extraction rate for Llama2-7B and GPT-3.5, respectively. Source code is avaliable at \url{https://github.com/liangzid/PromptExtractionEval}.



## **34. Open Sesame! Universal Black Box Jailbreaking of Large Language Models**

cs.CL

Accepted at SeT-LLM @ ICLR 2024

**SubmitDate**: 2024-08-05    [abs](http://arxiv.org/abs/2309.01446v4) [paper-pdf](http://arxiv.org/pdf/2309.01446v4)

**Authors**: Raz Lapid, Ron Langberg, Moshe Sipper

**Abstract**: Large language models (LLMs), designed to provide helpful and safe responses, often rely on alignment techniques to align with user intent and social guidelines. Unfortunately, this alignment can be exploited by malicious actors seeking to manipulate an LLM's outputs for unintended purposes. In this paper we introduce a novel approach that employs a genetic algorithm (GA) to manipulate LLMs when model architecture and parameters are inaccessible. The GA attack works by optimizing a universal adversarial prompt that -- when combined with a user's query -- disrupts the attacked model's alignment, resulting in unintended and potentially harmful outputs. Our novel approach systematically reveals a model's limitations and vulnerabilities by uncovering instances where its responses deviate from expected behavior. Through extensive experiments we demonstrate the efficacy of our technique, thus contributing to the ongoing discussion on responsible AI development by providing a diagnostic tool for evaluating and enhancing alignment of LLMs with human intent. To our knowledge this is the first automated universal black box jailbreak attack.



## **35. A Lean Transformer Model for Dynamic Malware Analysis and Detection**

cs.CR

**SubmitDate**: 2024-08-05    [abs](http://arxiv.org/abs/2408.02313v1) [paper-pdf](http://arxiv.org/pdf/2408.02313v1)

**Authors**: Tony Quertier, Benjamin Marais, Grégoire Barrué, Stéphane Morucci, Sévan Azé, Sébastien Salladin

**Abstract**: Malware is a fast-growing threat to the modern computing world and existing lines of defense are not efficient enough to address this issue. This is mainly due to the fact that many prevention solutions rely on signature-based detection methods that can easily be circumvented by hackers. Therefore, there is a recurrent need for behavior-based analysis where a suspicious file is ran in a secured environment and its traces are collected to reports for analysis. Previous works have shown some success leveraging Neural Networks and API calls sequences extracted from these execution reports.   Recently, Large Language Models and Generative AI have demonstrated impressive capabilities mainly in Natural Language Processing tasks and promising applications in the cybersecurity field for both attackers and defenders.   In this paper, we design an Encoder-Only model, based on the Transformers architecture, to detect malicious files, digesting their API call sequences collected by an execution emulation solution. We are also limiting the size of the model architecture and the number of its parameters since it is often considered that Large Language Models may be overkill for specific tasks such as the one we are dealing with hereafter. In addition to achieving decent detection results, this approach has the advantage of reducing our carbon footprint by limiting training and inference times and facilitating technical operations with less hardware requirements.   We also carry out some analysis of our results and highlight the limits and possible improvements when using Transformers to analyze malicious files.



## **36. LLM Lies: Hallucinations are not Bugs, but Features as Adversarial Examples**

cs.CL

**SubmitDate**: 2024-08-04    [abs](http://arxiv.org/abs/2310.01469v3) [paper-pdf](http://arxiv.org/pdf/2310.01469v3)

**Authors**: Jia-Yu Yao, Kun-Peng Ning, Zhen-Hui Liu, Mu-Nan Ning, Yu-Yang Liu, Li Yuan

**Abstract**: Large Language Models (LLMs), including GPT-3.5, LLaMA, and PaLM, seem to be knowledgeable and able to adapt to many tasks. However, we still cannot completely trust their answers, since LLMs suffer from \textbf{hallucination}\textemdash fabricating non-existent facts, deceiving users with or without their awareness. However, the reasons for their existence and pervasiveness remain unclear. In this paper, we demonstrate that nonsensical prompts composed of random tokens can also elicit the LLMs to respond with hallucinations. Moreover, we provide both theoretical and experimental evidence that transformers can be manipulated to produce specific pre-define tokens by perturbing its input sequence. This phenomenon forces us to revisit that \emph{hallucination may be another view of adversarial examples}, and it shares similar characteristics with conventional adversarial examples as a basic property of LLMs. Therefore, we formalize an automatic hallucination triggering method as the \textit{hallucination attack} in an adversarial way. Finally, we explore the basic properties of attacked adversarial prompts and propose a simple yet effective defense strategy. Our code is released on GitHub\footnote{https://github.com/PKU-YuanGroup/Hallucination-Attack}.



## **37. Towards Automatic Hands-on-Keyboard Attack Detection Using LLMs in EDR Solutions**

cs.CR

**SubmitDate**: 2024-08-04    [abs](http://arxiv.org/abs/2408.01993v1) [paper-pdf](http://arxiv.org/pdf/2408.01993v1)

**Authors**: Amit Portnoy, Ehud Azikri, Shay Kels

**Abstract**: Endpoint Detection and Remediation (EDR) platforms are essential for identifying and responding to cyber threats. This study presents a novel approach using Large Language Models (LLMs) to detect Hands-on-Keyboard (HOK) cyberattacks. Our method involves converting endpoint activity data into narrative forms that LLMs can analyze to distinguish between normal operations and potential HOK attacks. We address the challenges of interpreting endpoint data by segmenting narratives into windows and employing a dual training strategy. The results demonstrate that LLM-based models have the potential to outperform traditional machine learning methods, offering a promising direction for enhancing EDR capabilities and apply LLMs in cybersecurity.



## **38. InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents**

cs.CL

36 pages, 6 figures, 13 tables (ACL 2024 Findings)

**SubmitDate**: 2024-08-04    [abs](http://arxiv.org/abs/2403.02691v3) [paper-pdf](http://arxiv.org/pdf/2403.02691v3)

**Authors**: Qiusi Zhan, Zhixiang Liang, Zifan Ying, Daniel Kang

**Abstract**: Recent work has embodied LLMs as agents, allowing them to access tools, perform actions, and interact with external content (e.g., emails or websites). However, external content introduces the risk of indirect prompt injection (IPI) attacks, where malicious instructions are embedded within the content processed by LLMs, aiming to manipulate these agents into executing detrimental actions against users. Given the potentially severe consequences of such attacks, establishing benchmarks to assess and mitigate these risks is imperative.   In this work, we introduce InjecAgent, a benchmark designed to assess the vulnerability of tool-integrated LLM agents to IPI attacks. InjecAgent comprises 1,054 test cases covering 17 different user tools and 62 attacker tools. We categorize attack intentions into two primary types: direct harm to users and exfiltration of private data. We evaluate 30 different LLM agents and show that agents are vulnerable to IPI attacks, with ReAct-prompted GPT-4 vulnerable to attacks 24% of the time. Further investigation into an enhanced setting, where the attacker instructions are reinforced with a hacking prompt, shows additional increases in success rates, nearly doubling the attack success rate on the ReAct-prompted GPT-4. Our findings raise questions about the widespread deployment of LLM Agents. Our benchmark is available at https://github.com/uiuc-kang-lab/InjecAgent.



## **39. AttackEval: How to Evaluate the Effectiveness of Jailbreak Attacking on Large Language Models**

cs.CL

34 pages, 6 figures

**SubmitDate**: 2024-08-03    [abs](http://arxiv.org/abs/2401.09002v5) [paper-pdf](http://arxiv.org/pdf/2401.09002v5)

**Authors**: Dong shu, Mingyu Jin, Chong Zhang, Liangyao Li, Zihao Zhou, Yongfeng Zhang

**Abstract**: Ensuring the security of large language models (LLMs) against attacks has become increasingly urgent, with jailbreak attacks representing one of the most sophisticated threats. To deal with such risks, we introduce an innovative framework that can help evaluate the effectiveness of jailbreak attacks on LLMs. Unlike traditional binary evaluations focusing solely on the robustness of LLMs, our method assesses the effectiveness of the attacking prompts themselves. We present two distinct evaluation frameworks: a coarse-grained evaluation and a fine-grained evaluation. Each framework uses a scoring range from 0 to 1, offering unique perspectives and allowing for the assessment of attack effectiveness in different scenarios. Additionally, we develop a comprehensive ground truth dataset specifically tailored for jailbreak prompts. This dataset serves as a crucial benchmark for our current study and provides a foundational resource for future research. By comparing with traditional evaluation methods, our study shows that the current results align with baseline metrics while offering a more nuanced and fine-grained assessment. It also helps identify potentially harmful attack prompts that might appear harmless in traditional evaluations. Overall, our work establishes a solid foundation for assessing a broader range of attack prompts in the area of prompt injection.



## **40. MCGMark: An Encodable and Robust Online Watermark for LLM-Generated Malicious Code**

cs.CR

**SubmitDate**: 2024-08-02    [abs](http://arxiv.org/abs/2408.01354v1) [paper-pdf](http://arxiv.org/pdf/2408.01354v1)

**Authors**: Kaiwen Ning, Jiachi Chen, Qingyuan Zhong, Tao Zhang, Yanlin Wang, Wei Li, Yu Zhang, Weizhe Zhang, Zibin Zheng

**Abstract**: With the advent of large language models (LLMs), numerous software service providers (SSPs) are dedicated to developing LLMs customized for code generation tasks, such as CodeLlama and Copilot. However, these LLMs can be leveraged by attackers to create malicious software, which may pose potential threats to the software ecosystem. For example, they can automate the creation of advanced phishing malware. To address this issue, we first conduct an empirical study and design a prompt dataset, MCGTest, which involves approximately 400 person-hours of work and consists of 406 malicious code generation tasks. Utilizing this dataset, we propose MCGMark, the first robust, code structure-aware, and encodable watermarking approach to trace LLM-generated code. We embed encodable information by controlling the token selection and ensuring the output quality based on probabilistic outliers. Additionally, we enhance the robustness of the watermark by considering the structural features of malicious code, preventing the embedding of the watermark in easily modified positions, such as comments. We validate the effectiveness and robustness of MCGMark on the DeepSeek-Coder. MCGMark achieves an embedding success rate of 88.9% within a maximum output limit of 400 tokens. Furthermore, it also demonstrates strong robustness and has minimal impact on the quality of the output code. Our approach assists SSPs in tracing and holding responsible parties accountable for malicious code generated by LLMs.



## **41. Chat AI: A Seamless Slurm-Native Solution for HPC-Based Services**

cs.DC

Various improvements to explanations and form and updated graphs to  include data points up to 30.07.2024

**SubmitDate**: 2024-08-02    [abs](http://arxiv.org/abs/2407.00110v2) [paper-pdf](http://arxiv.org/pdf/2407.00110v2)

**Authors**: Ali Doosthosseini, Jonathan Decker, Hendrik Nolte, Julian M. Kunkel

**Abstract**: The widespread adoption of large language models (LLMs) has created a pressing need for an efficient, secure and private serving infrastructure, which allows researchers to run open source or custom fine-tuned LLMs and ensures users that their data remains private and is not stored without their consent. While high-performance computing (HPC) systems equipped with state-of-the-art GPUs are well-suited for training LLMs, their batch scheduling paradigm is not designed to support real-time serving of AI applications. Cloud systems, on the other hand, are well suited for web services but commonly lack access to the computational power of HPC clusters, especially expensive and scarce high-end GPUs, which are required for optimal inference speed. We propose an architecture with an implementation consisting of a web service that runs on a cloud VM with secure access to a scalable backend running a multitude of LLM models on HPC systems. By offering a web service using our HPC infrastructure to host LLMs, we leverage the trusted environment of local universities and research centers to offer a private and secure alternative to commercial LLM services. Our solution natively integrates with the HPC batch scheduler Slurm, enabling seamless deployment on HPC clusters, and is able to run side by side with regular Slurm workloads, while utilizing gaps in the schedule created by Slurm. In order to ensure the security of the HPC system, we use the SSH ForceCommand directive to construct a robust circuit breaker, which prevents successful attacks on the web-facing server from affecting the cluster. We have successfully deployed our system as a production service, and made the source code available at \url{https://github.com/gwdg/chat-ai}



## **42. A Survey of Text Watermarking in the Era of Large Language Models**

cs.CL

35 pages, 11 figures, 2 tables

**SubmitDate**: 2024-08-02    [abs](http://arxiv.org/abs/2312.07913v6) [paper-pdf](http://arxiv.org/pdf/2312.07913v6)

**Authors**: Aiwei Liu, Leyi Pan, Yijian Lu, Jingjing Li, Xuming Hu, Xi Zhang, Lijie Wen, Irwin King, Hui Xiong, Philip S. Yu

**Abstract**: Text watermarking algorithms are crucial for protecting the copyright of textual content. Historically, their capabilities and application scenarios were limited. However, recent advancements in large language models (LLMs) have revolutionized these techniques. LLMs not only enhance text watermarking algorithms with their advanced abilities but also create a need for employing these algorithms to protect their own copyrights or prevent potential misuse. This paper conducts a comprehensive survey of the current state of text watermarking technology, covering four main aspects: (1) an overview and comparison of different text watermarking techniques; (2) evaluation methods for text watermarking algorithms, including their detectability, impact on text or LLM quality, robustness under target or untargeted attacks; (3) potential application scenarios for text watermarking technology; (4) current challenges and future directions for text watermarking. This survey aims to provide researchers with a thorough understanding of text watermarking technology in the era of LLM, thereby promoting its further advancement.



## **43. SLIP: Securing LLMs IP Using Weights Decomposition**

cs.CR

**SubmitDate**: 2024-08-01    [abs](http://arxiv.org/abs/2407.10886v2) [paper-pdf](http://arxiv.org/pdf/2407.10886v2)

**Authors**: Yehonathan Refael, Adam Hakim, Lev Greenberg, Tal Aviv, Satya Lokam, Ben Fishman, Shachar Seidman

**Abstract**: Large language models (LLMs) have recently seen widespread adoption, in both academia and industry. As these models grow, they become valuable intellectual property (IP), reflecting enormous investments by their owners. Moreover, the high cost of cloud-based deployment has driven interest towards deployment to edge devices, yet this risks exposing valuable parameters to theft and unauthorized use. Current methods to protect models' IP on the edge have limitations in terms of practicality, loss in accuracy, or suitability to requirements. In this paper, we introduce a novel hybrid inference algorithm, named SLIP, designed to protect edge-deployed models from theft. SLIP is the first hybrid protocol that is both practical for real-world applications and provably secure, while having zero accuracy degradation and minimal impact on latency. It involves partitioning the model between two computing resources, one secure but expensive, and another cost-effective but vulnerable. This is achieved through matrix decomposition, ensuring that the secure resource retains a maximally sensitive portion of the model's IP while performing a minimal amount of computations, and vice versa for the vulnerable resource. Importantly, the protocol includes security guarantees that prevent attackers from exploiting the partition to infer the secured information. Finally, we present experimental results that show the robustness and effectiveness of our method, positioning it as a compelling solution for protecting LLMs.



## **44. Prover-Verifier Games improve legibility of LLM outputs**

cs.CL

**SubmitDate**: 2024-08-01    [abs](http://arxiv.org/abs/2407.13692v2) [paper-pdf](http://arxiv.org/pdf/2407.13692v2)

**Authors**: Jan Hendrik Kirchner, Yining Chen, Harri Edwards, Jan Leike, Nat McAleese, Yuri Burda

**Abstract**: One way to increase confidence in the outputs of Large Language Models (LLMs) is to support them with reasoning that is clear and easy to check -- a property we call legibility. We study legibility in the context of solving grade-school math problems and show that optimizing chain-of-thought solutions only for answer correctness can make them less legible. To mitigate the loss in legibility, we propose a training algorithm inspired by Prover-Verifier Game from Anil et al. (2021). Our algorithm iteratively trains small verifiers to predict solution correctness, "helpful" provers to produce correct solutions that the verifier accepts, and "sneaky" provers to produce incorrect solutions that fool the verifier. We find that the helpful prover's accuracy and the verifier's robustness to adversarial attacks increase over the course of training. Furthermore, we show that legibility training transfers to time-constrained humans tasked with verifying solution correctness. Over course of LLM training human accuracy increases when checking the helpful prover's solutions, and decreases when checking the sneaky prover's solutions. Hence, training for checkability by small verifiers is a plausible technique for increasing output legibility. Our results suggest legibility training against small verifiers as a practical avenue for increasing legibility of large LLMs to humans, and thus could help with alignment of superhuman models.



## **45. Pathway to Secure and Trustworthy 6G for LLMs: Attacks, Defense, and Opportunities**

cs.CR

7 pages, 4 figures

**SubmitDate**: 2024-08-01    [abs](http://arxiv.org/abs/2408.00722v1) [paper-pdf](http://arxiv.org/pdf/2408.00722v1)

**Authors**: Sunder Ali Khowaja, Parus Khuwaja, Kapal Dev, Hussam Al Hamadi, Engin Zeydan

**Abstract**: Recently, large language models (LLMs) have been gaining a lot of interest due to their adaptability and extensibility in emerging applications, including communication networks. It is anticipated that 6G mobile edge computing networks will be able to support LLMs as a service, as they provide ultra reliable low-latency communications and closed loop massive connectivity. However, LLMs are vulnerable to data and model privacy issues that affect the trustworthiness of LLMs to be deployed for user-based services. In this paper, we explore the security vulnerabilities associated with fine-tuning LLMs in 6G networks, in particular the membership inference attack. We define the characteristics of an attack network that can perform a membership inference attack if the attacker has access to the fine-tuned model for the downstream task. We show that the membership inference attacks are effective for any downstream task, which can lead to a personal data breach when using LLM as a service. The experimental results show that the attack success rate of maximum 92% can be achieved on named entity recognition task. Based on the experimental analysis, we discuss possible defense mechanisms and present possible research directions to make the LLMs more trustworthy in the context of 6G networks.



## **46. Jailbreaking Text-to-Image Models with LLM-Based Agents**

cs.CR

**SubmitDate**: 2024-08-01    [abs](http://arxiv.org/abs/2408.00523v1) [paper-pdf](http://arxiv.org/pdf/2408.00523v1)

**Authors**: Yingkai Dong, Zheng Li, Xiangtao Meng, Ning Yu, Shanqing Guo

**Abstract**: Recent advancements have significantly improved automated task-solving capabilities using autonomous agents powered by large language models (LLMs). However, most LLM-based agents focus on dialogue, programming, or specialized domains, leaving gaps in addressing generative AI safety tasks. These gaps are primarily due to the challenges posed by LLM hallucinations and the lack of clear guidelines. In this paper, we propose Atlas, an advanced LLM-based multi-agent framework that integrates an efficient fuzzing workflow to target generative AI models, specifically focusing on jailbreak attacks against text-to-image (T2I) models with safety filters. Atlas utilizes a vision-language model (VLM) to assess whether a prompt triggers the T2I model's safety filter. It then iteratively collaborates with both LLM and VLM to generate an alternative prompt that bypasses the filter. Atlas also enhances the reasoning abilities of LLMs in attack scenarios by leveraging multi-agent communication, in-context learning (ICL) memory mechanisms, and the chain-of-thought (COT) approach. Our evaluation demonstrates that Atlas successfully jailbreaks several state-of-the-art T2I models in a black-box setting, which are equipped with multi-modal safety filters. In addition, Atlas outperforms existing methods in both query efficiency and the quality of the generated images.



## **47. Autonomous LLM-Enhanced Adversarial Attack for Text-to-Motion**

cs.CV

**SubmitDate**: 2024-08-01    [abs](http://arxiv.org/abs/2408.00352v1) [paper-pdf](http://arxiv.org/pdf/2408.00352v1)

**Authors**: Honglei Miao, Fan Ma, Ruijie Quan, Kun Zhan, Yi Yang

**Abstract**: Human motion generation driven by deep generative models has enabled compelling applications, but the ability of text-to-motion (T2M) models to produce realistic motions from text prompts raises security concerns if exploited maliciously. Despite growing interest in T2M, few methods focus on safeguarding these models against adversarial attacks, with existing work on text-to-image models proving insufficient for the unique motion domain. In the paper, we propose ALERT-Motion, an autonomous framework leveraging large language models (LLMs) to craft targeted adversarial attacks against black-box T2M models. Unlike prior methods modifying prompts through predefined rules, ALERT-Motion uses LLMs' knowledge of human motion to autonomously generate subtle yet powerful adversarial text descriptions. It comprises two key modules: an adaptive dispatching module that constructs an LLM-based agent to iteratively refine and search for adversarial prompts; and a multimodal information contrastive module that extracts semantically relevant motion information to guide the agent's search. Through this LLM-driven approach, ALERT-Motion crafts adversarial prompts querying victim models to produce outputs closely matching targeted motions, while avoiding obvious perturbations. Evaluations across popular T2M models demonstrate ALERT-Motion's superiority over previous methods, achieving higher attack success rates with stealthier adversarial prompts. This pioneering work on T2M adversarial attacks highlights the urgency of developing defensive measures as motion generation technology advances, urging further research into safe and responsible deployment.



## **48. Can Editing LLMs Inject Harm?**

cs.CL

The first two authors contributed equally. 9 pages for main paper, 36  pages including appendix. The code, results, dataset for this paper and more  resources are on the project website: https://llm-editing.github.io

**SubmitDate**: 2024-07-31    [abs](http://arxiv.org/abs/2407.20224v2) [paper-pdf](http://arxiv.org/pdf/2407.20224v2)

**Authors**: Canyu Chen, Baixiang Huang, Zekun Li, Zhaorun Chen, Shiyang Lai, Xiongxiao Xu, Jia-Chen Gu, Jindong Gu, Huaxiu Yao, Chaowei Xiao, Xifeng Yan, William Yang Wang, Philip Torr, Dawn Song, Kai Shu

**Abstract**: Knowledge editing techniques have been increasingly adopted to efficiently correct the false or outdated knowledge in Large Language Models (LLMs), due to the high cost of retraining from scratch. Meanwhile, one critical but under-explored question is: can knowledge editing be used to inject harm into LLMs? In this paper, we propose to reformulate knowledge editing as a new type of safety threat for LLMs, namely Editing Attack, and conduct a systematic investigation with a newly constructed dataset EditAttack. Specifically, we focus on two typical safety risks of Editing Attack including Misinformation Injection and Bias Injection. For the risk of misinformation injection, we first categorize it into commonsense misinformation injection and long-tail misinformation injection. Then, we find that editing attacks can inject both types of misinformation into LLMs, and the effectiveness is particularly high for commonsense misinformation injection. For the risk of bias injection, we discover that not only can biased sentences be injected into LLMs with high effectiveness, but also one single biased sentence injection can cause a bias increase in general outputs of LLMs, which are even highly irrelevant to the injected sentence, indicating a catastrophic impact on the overall fairness of LLMs. Then, we further illustrate the high stealthiness of editing attacks, measured by their impact on the general knowledge and reasoning capacities of LLMs, and show the hardness of defending editing attacks with empirical evidence. Our discoveries demonstrate the emerging misuse risks of knowledge editing techniques on compromising the safety alignment of LLMs.



## **49. Figure it Out: Analyzing-based Jailbreak Attack on Large Language Models**

cs.CR

**SubmitDate**: 2024-07-31    [abs](http://arxiv.org/abs/2407.16205v2) [paper-pdf](http://arxiv.org/pdf/2407.16205v2)

**Authors**: Shi Lin, Rongchang Li, Xun Wang, Changting Lin, Wenpeng Xing, Meng Han

**Abstract**: The rapid development of Large Language Models (LLMs) has brought remarkable generative capabilities across diverse tasks. However, despite the impressive achievements, these models still have numerous security vulnerabilities, particularly when faced with jailbreak attacks. Therefore, by investigating jailbreak attacks, we can uncover hidden weaknesses in LLMs and guide us in developing more robust defense mechanisms to fortify their security. In this paper, we further explore the boundary of jailbreak attacks on LLMs and propose Analyzing-based Jailbreak (ABJ). This effective jailbreak attack method takes advantage of LLMs' growing analyzing and reasoning capability and reveals their underlying vulnerabilities when facing analysis-based tasks. We conduct a detailed evaluation of ABJ across various open-source and closed-source LLMs, which achieves 94.8% Attack Success Rate (ASR) and 1.06 Attack Efficiency (AE) on GPT-4-turbo-0409, demonstrating state-of-the-art attack effectiveness and efficiency. Our research highlights the importance of prioritizing and enhancing the safety of LLMs to mitigate the risks of misuse.The code is publicly available at https://github.com/theshi-1128/ABJ-Attack.



## **50. TAROT: Task-Oriented Authorship Obfuscation Using Policy Optimization Methods**

cs.CL

**SubmitDate**: 2024-07-31    [abs](http://arxiv.org/abs/2407.21630v1) [paper-pdf](http://arxiv.org/pdf/2407.21630v1)

**Authors**: Gabriel Loiseau, Damien Sileo, Damien Riquet, Maxime Meyer, Marc Tommasi

**Abstract**: Authorship obfuscation aims to disguise the identity of an author within a text by altering the writing style, vocabulary, syntax, and other linguistic features associated with the text author. This alteration needs to balance privacy and utility. While strong obfuscation techniques can effectively hide the author's identity, they often degrade the quality and usefulness of the text for its intended purpose. Conversely, maintaining high utility tends to provide insufficient privacy, making it easier for an adversary to de-anonymize the author. Thus, achieving an optimal trade-off between these two conflicting objectives is crucial. In this paper, we propose TAROT: Task-Oriented Authorship Obfuscation Using Policy Optimization, a new unsupervised authorship obfuscation method whose goal is to optimize the privacy-utility trade-off by regenerating the entire text considering its downstream utility. Our approach leverages policy optimization as a fine-tuning paradigm over small language models in order to rewrite texts by preserving author identity and downstream task utility. We show that our approach largely reduce the accuracy of attackers while preserving utility. We make our code and models publicly available.



