# Latest Large Language Model Attack Papers
**update at 2024-11-22 10:18:22**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. GASP: Efficient Black-Box Generation of Adversarial Suffixes for Jailbreaking LLMs**

cs.LG

28 pages, 9 tables, 13 figures; under review at CVPR '25

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2411.14133v1) [paper-pdf](http://arxiv.org/pdf/2411.14133v1)

**Authors**: Advik Raj Basani, Xiao Zhang

**Abstract**: Large Language Models (LLMs) have shown impressive proficiency across a range of natural language processing tasks yet remain vulnerable to adversarial prompts, known as jailbreak attacks, carefully designed to elicit harmful responses from LLMs. Traditional methods rely on manual heuristics, which suffer from limited generalizability. While being automatic, optimization-based attacks often produce unnatural jailbreak prompts that are easy to detect by safety filters or require high computational overhead due to discrete token optimization. Witnessing the limitations of existing jailbreak methods, we introduce Generative Adversarial Suffix Prompter (GASP), a novel framework that combines human-readable prompt generation with Latent Bayesian Optimization (LBO) to improve adversarial suffix creation in a fully black-box setting. GASP leverages LBO to craft adversarial suffixes by efficiently exploring continuous embedding spaces, gradually optimizing the model to improve attack efficacy while balancing prompt coherence through a targeted iterative refinement procedure. Our experiments show that GASP can generate natural jailbreak prompts, significantly improving attack success rates, reducing training times, and accelerating inference speed, thus making it an efficient and scalable solution for red-teaming LLMs.



## **2. RAG-Thief: Scalable Extraction of Private Data from Retrieval-Augmented Generation Applications with Agent-based Attacks**

cs.CR

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2411.14110v1) [paper-pdf](http://arxiv.org/pdf/2411.14110v1)

**Authors**: Changyue Jiang, Xudong Pan, Geng Hong, Chenfu Bao, Min Yang

**Abstract**: While large language models (LLMs) have achieved notable success in generative tasks, they still face limitations, such as lacking up-to-date knowledge and producing hallucinations. Retrieval-Augmented Generation (RAG) enhances LLM performance by integrating external knowledge bases, providing additional context which significantly improves accuracy and knowledge coverage. However, building these external knowledge bases often requires substantial resources and may involve sensitive information. In this paper, we propose an agent-based automated privacy attack called RAG-Thief, which can extract a scalable amount of private data from the private database used in RAG applications. We conduct a systematic study on the privacy risks associated with RAG applications, revealing that the vulnerability of LLMs makes the private knowledge bases suffer significant privacy risks. Unlike previous manual attacks which rely on traditional prompt injection techniques, RAG-Thief starts with an initial adversarial query and learns from model responses, progressively generating new queries to extract as many chunks from the knowledge base as possible. Experimental results show that our RAG-Thief can extract over 70% information from the private knowledge bases within customized RAG applications deployed on local machines and real-world platforms, including OpenAI's GPTs and ByteDance's Coze. Our findings highlight the privacy vulnerabilities in current RAG applications and underscore the pressing need for stronger safeguards.



## **3. Verifying the Robustness of Automatic Credibility Assessment**

cs.CL

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2303.08032v3) [paper-pdf](http://arxiv.org/pdf/2303.08032v3)

**Authors**: Piotr Przybyła, Alexander Shvets, Horacio Saggion

**Abstract**: Text classification methods have been widely investigated as a way to detect content of low credibility: fake news, social media bots, propaganda, etc. Quite accurate models (likely based on deep neural networks) help in moderating public electronic platforms and often cause content creators to face rejection of their submissions or removal of already published texts. Having the incentive to evade further detection, content creators try to come up with a slightly modified version of the text (known as an attack with an adversarial example) that exploit the weaknesses of classifiers and result in a different output. Here we systematically test the robustness of common text classifiers against available attacking techniques and discover that, indeed, meaning-preserving changes in input text can mislead the models. The approaches we test focus on finding vulnerable spans in text and replacing individual characters or words, taking into account the similarity between the original and replacement content. We also introduce BODEGA: a benchmark for testing both victim models and attack methods on four misinformation detection tasks in an evaluation framework designed to simulate real use-cases of content moderation. The attacked tasks include (1) fact checking and detection of (2) hyperpartisan news, (3) propaganda and (4) rumours. Our experimental results show that modern large language models are often more vulnerable to attacks than previous, smaller solutions, e.g. attacks on GEMMA being up to 27\% more successful than those on BERT. Finally, we manually analyse a subset adversarial examples and check what kinds of modifications are used in successful attacks.



## **4. Next-Generation Phishing: How LLM Agents Empower Cyber Attackers**

cs.CR

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2411.13874v1) [paper-pdf](http://arxiv.org/pdf/2411.13874v1)

**Authors**: Khalifa Afane, Wenqi Wei, Ying Mao, Junaid Farooq, Juntao Chen

**Abstract**: The escalating threat of phishing emails has become increasingly sophisticated with the rise of Large Language Models (LLMs). As attackers exploit LLMs to craft more convincing and evasive phishing emails, it is crucial to assess the resilience of current phishing defenses. In this study we conduct a comprehensive evaluation of traditional phishing detectors, such as Gmail Spam Filter, Apache SpamAssassin, and Proofpoint, as well as machine learning models like SVM, Logistic Regression, and Naive Bayes, in identifying both traditional and LLM-rephrased phishing emails. We also explore the emerging role of LLMs as phishing detection tools, a method already adopted by companies like NTT Security Holdings and JPMorgan Chase. Our results reveal notable declines in detection accuracy for rephrased emails across all detectors, highlighting critical weaknesses in current phishing defenses. As the threat landscape evolves, our findings underscore the need for stronger security controls and regulatory oversight on LLM-generated content to prevent its misuse in creating advanced phishing attacks. This study contributes to the development of more effective Cyber Threat Intelligence (CTI) by leveraging LLMs to generate diverse phishing variants that can be used for data augmentation, harnessing the power of LLMs to enhance phishing detection, and paving the way for more robust and adaptable threat detection systems.



## **5. TransLinkGuard: Safeguarding Transformer Models Against Model Stealing in Edge Deployment**

cs.CR

Accepted by ACM MM24 Conference

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2404.11121v2) [paper-pdf](http://arxiv.org/pdf/2404.11121v2)

**Authors**: Qinfeng Li, Zhiqiang Shen, Zhenghan Qin, Yangfan Xie, Xuhong Zhang, Tianyu Du, Jianwei Yin

**Abstract**: Proprietary large language models (LLMs) have been widely applied in various scenarios. Additionally, deploying LLMs on edge devices is trending for efficiency and privacy reasons. However, edge deployment of proprietary LLMs introduces new security challenges: edge-deployed models are exposed as white-box accessible to users, enabling adversaries to conduct effective model stealing (MS) attacks. Unfortunately, existing defense mechanisms fail to provide effective protection. Specifically, we identify four critical protection properties that existing methods fail to simultaneously satisfy: (1) maintaining protection after a model is physically copied; (2) authorizing model access at request level; (3) safeguarding runtime reverse engineering; (4) achieving high security with negligible runtime overhead. To address the above issues, we propose TransLinkGuard, a plug-and-play model protection approach against model stealing on edge devices. The core part of TransLinkGuard is a lightweight authorization module residing in a secure environment, e.g., TEE. The authorization module can freshly authorize each request based on its input. Extensive experiments show that TransLinkGuard achieves the same security protection as the black-box security guarantees with negligible overhead.



## **6. AttentionBreaker: Adaptive Evolutionary Optimization for Unmasking Vulnerabilities in LLMs through Bit-Flip Attacks**

cs.CR

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2411.13757v1) [paper-pdf](http://arxiv.org/pdf/2411.13757v1)

**Authors**: Sanjay Das, Swastik Bhattacharya, Souvik Kundu, Shamik Kundu, Anand Menon, Arnab Raha, Kanad Basu

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing (NLP), excelling in tasks like text generation and summarization. However, their increasing adoption in mission-critical applications raises concerns about hardware-based threats, particularly bit-flip attacks (BFAs). BFAs, enabled by fault injection methods such as Rowhammer, target model parameters in memory, compromising both integrity and performance. Identifying critical parameters for BFAs in the vast parameter space of LLMs poses significant challenges. While prior research suggests transformer-based architectures are inherently more robust to BFAs compared to traditional deep neural networks, we challenge this assumption. For the first time, we demonstrate that as few as three bit-flips can cause catastrophic performance degradation in an LLM with billions of parameters. Current BFA techniques are inadequate for exploiting this vulnerability due to the difficulty of efficiently identifying critical parameters within the immense parameter space. To address this, we propose AttentionBreaker, a novel framework tailored for LLMs that enables efficient traversal of the parameter space to identify critical parameters. Additionally, we introduce GenBFA, an evolutionary optimization strategy designed to refine the search further, isolating the most critical bits for an efficient and effective attack. Empirical results reveal the profound vulnerability of LLMs to AttentionBreaker. For example, merely three bit-flips (4.129 x 10^-9% of total parameters) in the LLaMA3-8B-Instruct 8-bit quantized (W8) model result in a complete performance collapse: accuracy on MMLU tasks drops from 67.3% to 0%, and Wikitext perplexity skyrockets from 12.6 to 4.72 x 10^5. These findings underscore the effectiveness of AttentionBreaker in uncovering and exploiting critical vulnerabilities within LLM architectures.



## **7. SoK: A Systems Perspective on Compound AI Threats and Countermeasures**

cs.CR

13 pages, 4 figures, 2 tables

**SubmitDate**: 2024-11-20    [abs](http://arxiv.org/abs/2411.13459v1) [paper-pdf](http://arxiv.org/pdf/2411.13459v1)

**Authors**: Sarbartha Banerjee, Prateek Sahu, Mulong Luo, Anjo Vahldiek-Oberwagner, Neeraja J. Yadwadkar, Mohit Tiwari

**Abstract**: Large language models (LLMs) used across enterprises often use proprietary models and operate on sensitive inputs and data. The wide range of attack vectors identified in prior research - targeting various software and hardware components used in training and inference - makes it extremely challenging to enforce confidentiality and integrity policies.   As we advance towards constructing compound AI inference pipelines that integrate multiple large language models (LLMs), the attack surfaces expand significantly. Attackers now focus on the AI algorithms as well as the software and hardware components associated with these systems. While current research often examines these elements in isolation, we find that combining cross-layer attack observations can enable powerful end-to-end attacks with minimal assumptions about the threat model. Given, the sheer number of existing attacks at each layer, we need a holistic and systemized understanding of different attack vectors at each layer.   This SoK discusses different software and hardware attacks applicable to compound AI systems and demonstrates how combining multiple attack mechanisms can reduce the threat model assumptions required for an isolated attack. Next, we systematize the ML attacks in lines with the Mitre Att&ck framework to better position each attack based on the threat model. Finally, we outline the existing countermeasures for both software and hardware layers and discuss the necessity of a comprehensive defense strategy to enable the secure and high-performance deployment of compound AI systems.



## **8. WaterPark: A Robustness Assessment of Language Model Watermarking**

cs.CR

22 pages

**SubmitDate**: 2024-11-20    [abs](http://arxiv.org/abs/2411.13425v1) [paper-pdf](http://arxiv.org/pdf/2411.13425v1)

**Authors**: Jiacheng Liang, Zian Wang, Lauren Hong, Shouling Ji, Ting Wang

**Abstract**: To mitigate the misuse of large language models (LLMs), such as disinformation, automated phishing, and academic cheating, there is a pressing need for the capability of identifying LLM-generated texts. Watermarking emerges as one promising solution: it plants statistical signals into LLMs' generative processes and subsequently verifies whether LLMs produce given texts. Various watermarking methods (``watermarkers'') have been proposed; yet, due to the lack of unified evaluation platforms, many critical questions remain under-explored: i) What are the strengths/limitations of various watermarkers, especially their attack robustness? ii) How do various design choices impact their robustness? iii) How to optimally operate watermarkers in adversarial environments?   To fill this gap, we systematize existing LLM watermarkers and watermark removal attacks, mapping out their design spaces. We then develop WaterPark, a unified platform that integrates 10 state-of-the-art watermarkers and 12 representative attacks. More importantly, leveraging WaterPark, we conduct a comprehensive assessment of existing watermarkers, unveiling the impact of various design choices on their attack robustness. For instance, a watermarker's resilience to increasingly intensive attacks hinges on its context dependency. We further explore the best practices to operate watermarkers in adversarial environments. For instance, using a generic detector alongside a watermark-specific detector improves the security of vulnerable watermarkers. We believe our study sheds light on current LLM watermarking techniques while WaterPark serves as a valuable testbed to facilitate future research.



## **9. CryptoFormalEval: Integrating LLMs and Formal Verification for Automated Cryptographic Protocol Vulnerability Detection**

cs.CR

**SubmitDate**: 2024-11-20    [abs](http://arxiv.org/abs/2411.13627v1) [paper-pdf](http://arxiv.org/pdf/2411.13627v1)

**Authors**: Cristian Curaba, Denis D'Ambrosi, Alessandro Minisini, Natalia Pérez-Campanero Antolín

**Abstract**: Cryptographic protocols play a fundamental role in securing modern digital infrastructure, but they are often deployed without prior formal verification. This could lead to the adoption of distributed systems vulnerable to attack vectors. Formal verification methods, on the other hand, require complex and time-consuming techniques that lack automatization. In this paper, we introduce a benchmark to assess the ability of Large Language Models (LLMs) to autonomously identify vulnerabilities in new cryptographic protocols through interaction with Tamarin: a theorem prover for protocol verification. We created a manually validated dataset of novel, flawed, communication protocols and designed a method to automatically verify the vulnerabilities found by the AI agents. Our results about the performances of the current frontier models on the benchmark provides insights about the possibility of cybersecurity applications by integrating LLMs with symbolic reasoning systems.



## **10. TAPT: Test-Time Adversarial Prompt Tuning for Robust Inference in Vision-Language Models**

cs.CV

**SubmitDate**: 2024-11-20    [abs](http://arxiv.org/abs/2411.13136v1) [paper-pdf](http://arxiv.org/pdf/2411.13136v1)

**Authors**: Xin Wang, Kai Chen, Jiaming Zhang, Jingjing Chen, Xingjun Ma

**Abstract**: Large pre-trained Vision-Language Models (VLMs) such as CLIP have demonstrated excellent zero-shot generalizability across various downstream tasks. However, recent studies have shown that the inference performance of CLIP can be greatly degraded by small adversarial perturbations, especially its visual modality, posing significant safety threats. To mitigate this vulnerability, in this paper, we propose a novel defense method called Test-Time Adversarial Prompt Tuning (TAPT) to enhance the inference robustness of CLIP against visual adversarial attacks. TAPT is a test-time defense method that learns defensive bimodal (textual and visual) prompts to robustify the inference process of CLIP. Specifically, it is an unsupervised method that optimizes the defensive prompts for each test sample by minimizing a multi-view entropy and aligning adversarial-clean distributions. We evaluate the effectiveness of TAPT on 11 benchmark datasets, including ImageNet and 10 other zero-shot datasets, demonstrating that it enhances the zero-shot adversarial robustness of the original CLIP by at least 48.9% against AutoAttack (AA), while largely maintaining performance on clean examples. Moreover, TAPT outperforms existing adversarial prompt tuning methods across various backbones, achieving an average robustness improvement of at least 36.6%.



## **11. When Backdoors Speak: Understanding LLM Backdoor Attacks Through Model-Generated Explanations**

cs.CR

**SubmitDate**: 2024-11-19    [abs](http://arxiv.org/abs/2411.12701v1) [paper-pdf](http://arxiv.org/pdf/2411.12701v1)

**Authors**: Huaizhi Ge, Yiming Li, Qifan Wang, Yongfeng Zhang, Ruixiang Tang

**Abstract**: Large Language Models (LLMs) are vulnerable to backdoor attacks, where hidden triggers can maliciously manipulate model behavior. While several backdoor attack methods have been proposed, the mechanisms by which backdoor functions operate in LLMs remain underexplored. In this paper, we move beyond attacking LLMs and investigate backdoor functionality through the novel lens of natural language explanations. Specifically, we leverage LLMs' generative capabilities to produce human-understandable explanations for their decisions, allowing us to compare explanations for clean and poisoned samples. We explore various backdoor attacks and embed the backdoor into LLaMA models for multiple tasks. Our experiments show that backdoored models produce higher-quality explanations for clean data compared to poisoned data, while generating significantly more consistent explanations for poisoned data than for clean data. We further analyze the explanation generation process, revealing that at the token level, the explanation token of poisoned samples only appears in the final few transformer layers of the LLM. At the sentence level, attention dynamics indicate that poisoned inputs shift attention from the input context when generating the explanation. These findings deepen our understanding of backdoor attack mechanisms in LLMs and offer a framework for detecting such vulnerabilities through explainability techniques, contributing to the development of more secure LLMs.



## **12. Exploring adversarial robustness of JPEG AI: methodology, comparison and new methods**

eess.IV

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11795v1) [paper-pdf](http://arxiv.org/pdf/2411.11795v1)

**Authors**: Egor Kovalev, Georgii Bychkov, Khaled Abud, Aleksandr Gushchin, Anna Chistyakova, Sergey Lavrushkin, Dmitriy Vatolin, Anastasia Antsiferova

**Abstract**: Adversarial robustness of neural networks is an increasingly important area of research, combining studies on computer vision models, large language models (LLMs), and others. With the release of JPEG AI - the first standard for end-to-end neural image compression (NIC) methods - the question of its robustness has become critically significant. JPEG AI is among the first international, real-world applications of neural-network-based models to be embedded in consumer devices. However, research on NIC robustness has been limited to open-source codecs and a narrow range of attacks. This paper proposes a new methodology for measuring NIC robustness to adversarial attacks. We present the first large-scale evaluation of JPEG AI's robustness, comparing it with other NIC models. Our evaluation results and code are publicly available online (link is hidden for a blind review).



## **13. DAWN: Designing Distributed Agents in a Worldwide Network**

cs.NI

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2410.22339v2) [paper-pdf](http://arxiv.org/pdf/2410.22339v2)

**Authors**: Zahra Aminiranjbar, Jianan Tang, Qiudan Wang, Shubha Pant, Mahesh Viswanathan

**Abstract**: The rapid evolution of Large Language Models (LLMs) has transformed them from basic conversational tools into sophisticated entities capable of complex reasoning and decision-making. These advancements have led to the development of specialized LLM-based agents designed for diverse tasks such as coding and web browsing. As these agents become more capable, the need for a robust framework that facilitates global communication and collaboration among them towards advanced objectives has become increasingly critical. Distributed Agents in a Worldwide Network (DAWN) addresses this need by offering a versatile framework that integrates LLM-based agents with traditional software systems, enabling the creation of agentic applications suited for a wide range of use cases. DAWN enables distributed agents worldwide to register and be easily discovered through Gateway Agents. Collaborations among these agents are coordinated by a Principal Agent equipped with reasoning strategies. DAWN offers three operational modes: No-LLM Mode for deterministic tasks, Copilot for augmented decision-making, and LLM Agent for autonomous operations. Additionally, DAWN ensures the safety and security of agent collaborations globally through a dedicated safety, security, and compliance layer, protecting the network against attackers and adhering to stringent security and compliance standards. These features make DAWN a robust network for deploying agent-based applications across various industries.



## **14. TrojanRobot: Backdoor Attacks Against Robotic Manipulation in the Physical World**

cs.RO

Initial version with preliminary results. We welcome any feedback or  suggestions

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11683v1) [paper-pdf](http://arxiv.org/pdf/2411.11683v1)

**Authors**: Xianlong Wang, Hewen Pan, Hangtao Zhang, Minghui Li, Shengshan Hu, Ziqi Zhou, Lulu Xue, Peijin Guo, Yichen Wang, Wei Wan, Aishan Liu, Leo Yu Zhang

**Abstract**: Robotic manipulation refers to the autonomous handling and interaction of robots with objects using advanced techniques in robotics and artificial intelligence. The advent of powerful tools such as large language models (LLMs) and large vision-language models (LVLMs) has significantly enhanced the capabilities of these robots in environmental perception and decision-making. However, the introduction of these intelligent agents has led to security threats such as jailbreak attacks and adversarial attacks.   In this research, we take a further step by proposing a backdoor attack specifically targeting robotic manipulation and, for the first time, implementing backdoor attack in the physical world. By embedding a backdoor visual language model into the visual perception module within the robotic system, we successfully mislead the robotic arm's operation in the physical world, given the presence of common items as triggers. Experimental evaluations in the physical world demonstrate the effectiveness of the proposed backdoor attack.



## **15. Enhancing Vision-Language Model Safety through Progressive Concept-Bottleneck-Driven Alignment**

cs.CV

arXiv admin note: substantial text overlap with arXiv:2405.13581

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11543v1) [paper-pdf](http://arxiv.org/pdf/2411.11543v1)

**Authors**: Zhendong Liu, Yuanbi Nie, Yingshui Tan, Xiangyu Yue, Qiushi Cui, Chongjun Wang, Xiaoyong Zhu, Bo Zheng

**Abstract**: Benefiting from the powerful capabilities of Large Language Models (LLMs), pre-trained visual encoder models connected to LLMs form Vision Language Models (VLMs). However, recent research shows that the visual modality in VLMs is highly vulnerable, allowing attackers to bypass safety alignment in LLMs through visually transmitted content, launching harmful attacks. To address this challenge, we propose a progressive concept-based alignment strategy, PSA-VLM, which incorporates safety modules as concept bottlenecks to enhance visual modality safety alignment. By aligning model predictions with specific safety concepts, we improve defenses against risky images, enhancing explainability and controllability while minimally impacting general performance. Our method is obtained through two-stage training. The low computational cost of the first stage brings very effective performance improvement, and the fine-tuning of the language model in the second stage further improves the safety performance. Our method achieves state-of-the-art results on popular VLM safety benchmark.



## **16. Membership Inference Attack against Long-Context Large Language Models**

cs.CL

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11424v1) [paper-pdf](http://arxiv.org/pdf/2411.11424v1)

**Authors**: Zixiong Wang, Gaoyang Liu, Yang Yang, Chen Wang

**Abstract**: Recent advances in Large Language Models (LLMs) have enabled them to overcome their context window limitations, and demonstrate exceptional retrieval and reasoning capacities on longer context. Quesion-answering systems augmented with Long-Context Language Models (LCLMs) can automatically search massive external data and incorporate it into their contexts, enabling faithful predictions and reducing issues such as hallucinations and knowledge staleness. Existing studies targeting LCLMs mainly concentrate on addressing the so-called lost-in-the-middle problem or improving the inference effiencicy, leaving their privacy risks largely unexplored. In this paper, we aim to bridge this gap and argue that integrating all information into the long context makes it a repository of sensitive information, which often contains private data such as medical records or personal identities. We further investigate the membership privacy within LCLMs external context, with the aim of determining whether a given document or sequence is included in the LCLMs context. Our basic idea is that if a document lies in the context, it will exhibit a low generation loss or a high degree of semantic similarity to the contents generated by LCLMs. We for the first time propose six membership inference attack (MIA) strategies tailored for LCLMs and conduct extensive experiments on various popular models. Empirical results demonstrate that our attacks can accurately infer membership status in most cases, e.g., 90.66% attack F1-score on Multi-document QA datasets with LongChat-7b-v1.5-32k, highlighting significant risks of membership leakage within LCLMs input contexts. Furthermore, we examine the underlying reasons why LCLMs are susceptible to revealing such membership information.



## **17. The Dark Side of Trust: Authority Citation-Driven Jailbreak Attacks on Large Language Models**

cs.LG

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11407v1) [paper-pdf](http://arxiv.org/pdf/2411.11407v1)

**Authors**: Xikang Yang, Xuehai Tang, Jizhong Han, Songlin Hu

**Abstract**: The widespread deployment of large language models (LLMs) across various domains has showcased their immense potential while exposing significant safety vulnerabilities. A major concern is ensuring that LLM-generated content aligns with human values. Existing jailbreak techniques reveal how this alignment can be compromised through specific prompts or adversarial suffixes. In this study, we introduce a new threat: LLMs' bias toward authority. While this inherent bias can improve the quality of outputs generated by LLMs, it also introduces a potential vulnerability, increasing the risk of producing harmful content. Notably, the biases in LLMs is the varying levels of trust given to different types of authoritative information in harmful queries. For example, malware development often favors trust GitHub. To better reveal the risks with LLM, we propose DarkCite, an adaptive authority citation matcher and generator designed for a black-box setting. DarkCite matches optimal citation types to specific risk types and generates authoritative citations relevant to harmful instructions, enabling more effective jailbreak attacks on aligned LLMs.Our experiments show that DarkCite achieves a higher attack success rate (e.g., LLama-2 at 76% versus 68%) than previous methods. To counter this risk, we propose an authenticity and harm verification defense strategy, raising the average defense pass rate (DPR) from 11% to 74%. More importantly, the ability to link citations to the content they encompass has become a foundational function in LLMs, amplifying the influence of LLMs' bias toward authority.



## **18. Hacking Back the AI-Hacker: Prompt Injection as a Defense Against LLM-driven Cyberattacks**

cs.CR

v0.2 (evaluated on more agents)

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2410.20911v2) [paper-pdf](http://arxiv.org/pdf/2410.20911v2)

**Authors**: Dario Pasquini, Evgenios M. Kornaropoulos, Giuseppe Ateniese

**Abstract**: Large language models (LLMs) are increasingly being harnessed to automate cyberattacks, making sophisticated exploits more accessible and scalable. In response, we propose a new defense strategy tailored to counter LLM-driven cyberattacks. We introduce Mantis, a defensive framework that exploits LLMs' susceptibility to adversarial inputs to undermine malicious operations. Upon detecting an automated cyberattack, Mantis plants carefully crafted inputs into system responses, leading the attacker's LLM to disrupt their own operations (passive defense) or even compromise the attacker's machine (active defense). By deploying purposefully vulnerable decoy services to attract the attacker and using dynamic prompt injections for the attacker's LLM, Mantis can autonomously hack back the attacker. In our experiments, Mantis consistently achieved over 95% effectiveness against automated LLM-driven attacks. To foster further research and collaboration, Mantis is available as an open-source tool: https://github.com/pasquini-dario/project_mantis



## **19. Adapting to Cyber Threats: A Phishing Evolution Network (PEN) Framework for Phishing Generation and Analyzing Evolution Patterns using Large Language Models**

cs.CR

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11389v1) [paper-pdf](http://arxiv.org/pdf/2411.11389v1)

**Authors**: Fengchao Chen, Tingmin Wu, Van Nguyen, Shuo Wang, Hongsheng Hu, Alsharif Abuadbba, Carsten Rudolph

**Abstract**: Phishing remains a pervasive cyber threat, as attackers craft deceptive emails to lure victims into revealing sensitive information. While Artificial Intelligence (AI), particularly deep learning, has become a key component in defending against phishing attacks, these approaches face critical limitations. The scarcity of publicly available, diverse, and updated data, largely due to privacy concerns, constrains their effectiveness. As phishing tactics evolve rapidly, models trained on limited, outdated data struggle to detect new, sophisticated deception strategies, leaving systems vulnerable to an ever-growing array of attacks. Addressing this gap is essential to strengthening defenses in an increasingly hostile cyber landscape. To address this gap, we propose the Phishing Evolution Network (PEN), a framework leveraging large language models (LLMs) and adversarial training mechanisms to continuously generate high quality and realistic diverse phishing samples, and analyze features of LLM-provided phishing to understand evolving phishing patterns. We evaluate the quality and diversity of phishing samples generated by PEN and find that it produces over 80% realistic phishing samples, effectively expanding phishing datasets across seven dominant types. These PEN-generated samples enhance the performance of current phishing detectors, leading to a 40% improvement in detection accuracy. Additionally, the use of PEN significantly boosts model robustness, reducing detectors' sensitivity to perturbations by up to 60%, thereby decreasing attack success rates under adversarial conditions. When we analyze the phishing patterns that are used in LLM-generated phishing, the cognitive complexity and the tone of time limitation are detected with statistically significant differences compared with existing phishing.



## **20. CROW: Eliminating Backdoors from Large Language Models via Internal Consistency Regularization**

cs.CL

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.12768v1) [paper-pdf](http://arxiv.org/pdf/2411.12768v1)

**Authors**: Nay Myat Min, Long H. Pham, Yige Li, Jun Sun

**Abstract**: Recent studies reveal that Large Language Models (LLMs) are susceptible to backdoor attacks, where adversaries embed hidden triggers that manipulate model responses. Existing backdoor defense methods are primarily designed for vision or classification tasks, and are thus ineffective for text generation tasks, leaving LLMs vulnerable. We introduce Internal Consistency Regularization (CROW), a novel defense using consistency regularization finetuning to address layer-wise inconsistencies caused by backdoor triggers. CROW leverages the intuition that clean models exhibit smooth, consistent transitions in hidden representations across layers, whereas backdoored models show noticeable fluctuation when triggered. By enforcing internal consistency through adversarial perturbations and regularization, CROW neutralizes backdoor effects without requiring clean reference models or prior trigger knowledge, relying only on a small set of clean data. This makes it practical for deployment across various LLM architectures. Experimental results demonstrate that CROW consistently achieves a significant reductions in attack success rates across diverse backdoor strategies and tasks, including negative sentiment, targeted refusal, and code injection, on models such as Llama-2 (7B, 13B), CodeLlama (7B, 13B) and Mistral-7B, while preserving the model's generative capabilities.



## **21. Exploring the Adversarial Vulnerabilities of Vision-Language-Action Models in Robotics**

cs.RO

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.13587v1) [paper-pdf](http://arxiv.org/pdf/2411.13587v1)

**Authors**: Taowen Wang, Dongfang Liu, James Chenhao Liang, Wenhao Yang, Qifan Wang, Cheng Han, Jiebo Luo, Ruixiang Tang

**Abstract**: Recently in robotics, Vision-Language-Action (VLA) models have emerged as a transformative approach, enabling robots to execute complex tasks by integrating visual and linguistic inputs within an end-to-end learning framework. While VLA models offer significant capabilities, they also introduce new attack surfaces, making them vulnerable to adversarial attacks. With these vulnerabilities largely unexplored, this paper systematically quantifies the robustness of VLA-based robotic systems. Recognizing the unique demands of robotic execution, our attack objectives target the inherent spatial and functional characteristics of robotic systems. In particular, we introduce an untargeted position-aware attack objective that leverages spatial foundations to destabilize robotic actions, and a targeted attack objective that manipulates the robotic trajectory. Additionally, we design an adversarial patch generation approach that places a small, colorful patch within the camera's view, effectively executing the attack in both digital and physical environments. Our evaluation reveals a marked degradation in task success rates, with up to a 100\% reduction across a suite of simulated robotic tasks, highlighting critical security gaps in current VLA architectures. By unveiling these vulnerabilities and proposing actionable evaluation metrics, this work advances both the understanding and enhancement of safety for VLA-based robotic systems, underscoring the necessity for developing robust defense strategies prior to physical-world deployments.



## **22. JailbreakLens: Interpreting Jailbreak Mechanism in the Lens of Representation and Circuit**

cs.CR

18 pages, 10 figures

**SubmitDate**: 2024-11-17    [abs](http://arxiv.org/abs/2411.11114v1) [paper-pdf](http://arxiv.org/pdf/2411.11114v1)

**Authors**: Zeqing He, Zhibo Wang, Zhixuan Chu, Huiyu Xu, Rui Zheng, Kui Ren, Chun Chen

**Abstract**: Despite the outstanding performance of Large language models (LLMs) in diverse tasks, they are vulnerable to jailbreak attacks, wherein adversarial prompts are crafted to bypass their security mechanisms and elicit unexpected responses.Although jailbreak attacks are prevalent, the understanding of their underlying mechanisms remains limited. Recent studies have explain typical jailbreaking behavior (e.g., the degree to which the model refuses to respond) of LLMs by analyzing the representation shifts in their latent space caused by jailbreak prompts or identifying key neurons that contribute to the success of these attacks. However, these studies neither explore diverse jailbreak patterns nor provide a fine-grained explanation from the failure of circuit to the changes of representational, leaving significant gaps in uncovering the jailbreak mechanism. In this paper, we propose JailbreakLens, an interpretation framework that analyzes jailbreak mechanisms from both representation (which reveals how jailbreaks alter the model's harmfulness perception) and circuit perspectives (which uncovers the causes of these deceptions by identifying key circuits contributing to the vulnerability), tracking their evolution throughout the entire response generation process. We then conduct an in-depth evaluation of jailbreak behavior on four mainstream LLMs under seven jailbreak strategies. Our evaluation finds that jailbreak prompts amplify components that reinforce affirmative responses while suppressing those that produce refusal. Although this manipulation shifts model representations toward safe clusters to deceive the LLM, leading it to provide detailed responses instead of refusals, it still produce abnormal activation which can be caught in the circuit analysis.



## **23. Safely Learning with Private Data: A Federated Learning Framework for Large Language Model**

cs.CR

**SubmitDate**: 2024-11-17    [abs](http://arxiv.org/abs/2406.14898v3) [paper-pdf](http://arxiv.org/pdf/2406.14898v3)

**Authors**: JiaYing Zheng, HaiNan Zhang, LingXiang Wang, WangJie Qiu, HongWei Zheng, ZhiMing Zheng

**Abstract**: Private data, being larger and quality-higher than public data, can greatly improve large language models (LLM). However, due to privacy concerns, this data is often dispersed in multiple silos, making its secure utilization for LLM training a challenge. Federated learning (FL) is an ideal solution for training models with distributed private data, but traditional frameworks like FedAvg are unsuitable for LLM due to their high computational demands on clients. An alternative, split learning, offloads most training parameters to the server while training embedding and output layers locally, making it more suitable for LLM. Nonetheless, it faces significant challenges in security and efficiency. Firstly, the gradients of embeddings are prone to attacks, leading to potential reverse engineering of private data. Furthermore, the server's limitation of handle only one client's training request at a time hinders parallel training, severely impacting training efficiency. In this paper, we propose a Federated Learning framework for LLM, named FL-GLM, which prevents data leakage caused by both server-side and peer-client attacks while improving training efficiency. Specifically, we first place the input block and output block on local client to prevent embedding gradient attacks from server. Secondly, we employ key-encryption during client-server communication to prevent reverse engineering attacks from peer-clients. Lastly, we employ optimization methods like client-batching or server-hierarchical, adopting different acceleration methods based on the actual computational capabilities of the server. Experimental results on NLU and generation tasks demonstrate that FL-GLM achieves comparable metrics to centralized chatGLM model, validating the effectiveness of our federated learning framework.



## **24. Risks of Practicing Large Language Models in Smart Grid: Threat Modeling and Validation**

cs.CR

**SubmitDate**: 2024-11-17    [abs](http://arxiv.org/abs/2405.06237v2) [paper-pdf](http://arxiv.org/pdf/2405.06237v2)

**Authors**: Jiangnan Li, Yingyuan Yang, Jinyuan Sun

**Abstract**: Large language models (LLMs) represent significant breakthroughs in artificial intelligence and hold considerable potential for applications within smart grids. However, as demonstrated in previous literature, AI technologies are susceptible to various types of attacks. It is crucial to investigate and evaluate the risks associated with LLMs before deploying them in critical infrastructure like smart grids. In this paper, we systematically evaluated the risks of LLMs and identified two major types of attacks relevant to potential smart grid LLM applications, presenting the corresponding threat models. We also validated these attacks using popular LLMs and real smart grid data. Our validation demonstrates that attackers are capable of injecting bad data and retrieving domain knowledge from LLMs employed in different smart grid applications.



## **25. Playing Language Game with LLMs Leads to Jailbreaking**

cs.CL

**SubmitDate**: 2024-11-16    [abs](http://arxiv.org/abs/2411.12762v1) [paper-pdf](http://arxiv.org/pdf/2411.12762v1)

**Authors**: Yu Peng, Zewen Long, Fangming Dong, Congyi Li, Shu Wu, Kai Chen

**Abstract**: The advent of large language models (LLMs) has spurred the development of numerous jailbreak techniques aimed at circumventing their security defenses against malicious attacks. An effective jailbreak approach is to identify a domain where safety generalization fails, a phenomenon known as mismatched generalization. In this paper, we introduce two novel jailbreak methods based on mismatched generalization: natural language games and custom language games, both of which effectively bypass the safety mechanisms of LLMs, with various kinds and different variants, making them hard to defend and leading to high attack rates. Natural language games involve the use of synthetic linguistic constructs and the actions intertwined with these constructs, such as the Ubbi Dubbi language. Building on this phenomenon, we propose the custom language games method: by engaging with LLMs using a variety of custom rules, we successfully execute jailbreak attacks across multiple LLM platforms. Extensive experiments demonstrate the effectiveness of our methods, achieving success rates of 93% on GPT-4o, 89% on GPT-4o-mini and 83% on Claude-3.5-Sonnet. Furthermore, to investigate the generalizability of safety alignments, we fine-tuned Llama-3.1-70B with the custom language games to achieve safety alignment within our datasets and found that when interacting through other language games, the fine-tuned models still failed to identify harmful content. This finding indicates that the safety alignment knowledge embedded in LLMs fails to generalize across different linguistic formats, thus opening new avenues for future research in this area.



## **26. SQL Injection Jailbreak: a structural disaster of large language models**

cs.CR

**SubmitDate**: 2024-11-16    [abs](http://arxiv.org/abs/2411.01565v2) [paper-pdf](http://arxiv.org/pdf/2411.01565v2)

**Authors**: Jiawei Zhao, Kejiang Chen, Weiming Zhang, Nenghai Yu

**Abstract**: In recent years, the rapid development of large language models (LLMs) has brought new vitality to the various domains and generated substantial social and economic benefits. However, the swift advancement of LLMs has introduced new security vulnerabilities. Jailbreak, a form of attack that induces LLMs to output harmful content through carefully crafted prompts, poses a challenge to the safe and trustworthy development of LLMs. Previous jailbreak attack methods primarily exploited the internal capabilities of the model. Among them, one category leverages the model's implicit capabilities for jailbreak attacks, where the attacker is unaware of the exact reasons for the attack's success. The other category utilizes the model's explicit capabilities for jailbreak attacks, where the attacker understands the reasons for the attack's success. For example, these attacks exploit the model's abilities in coding, contextual learning, or understanding ASCII characters. However, these earlier jailbreak attacks have certain limitations, as they only exploit the inherent capabilities of the model. In this paper, we propose a novel jailbreak method, SQL Injection Jailbreak (SIJ), which utilizes the construction of input prompts by LLMs to inject jailbreak information into user prompts, enabling successful jailbreak of the LLMs. Our SIJ method achieves nearly 100\% attack success rates on five well-known open-source LLMs in the context of AdvBench, while incurring lower time costs compared to previous methods. More importantly, SIJ reveals a new vulnerability in LLMs that urgently needs to be addressed. To this end, we propose a defense method called Self-Reminder-Key and demonstrate its effectiveness through experiments. Our code is available at \href{https://github.com/weiyezhimeng/SQL-Injection-Jailbreak}{https://github.com/weiyezhimeng/SQL-Injection-Jailbreak}.



## **27. Insights and Current Gaps in Open-Source LLM Vulnerability Scanners: A Comparative Analysis**

cs.CR

15 pages, 11 figures

**SubmitDate**: 2024-11-16    [abs](http://arxiv.org/abs/2410.16527v3) [paper-pdf](http://arxiv.org/pdf/2410.16527v3)

**Authors**: Jonathan Brokman, Omer Hofman, Oren Rachmil, Inderjeet Singh, Vikas Pahuja, Rathina Sabapathy Aishvariya Priya, Amit Giloni, Roman Vainshtein, Hisashi Kojima

**Abstract**: This report presents a comparative analysis of open-source vulnerability scanners for conversational large language models (LLMs). As LLMs become integral to various applications, they also present potential attack surfaces, exposed to security risks such as information leakage and jailbreak attacks. Our study evaluates prominent scanners - Garak, Giskard, PyRIT, and CyberSecEval - that adapt red-teaming practices to expose these vulnerabilities. We detail the distinctive features and practical use of these scanners, outline unifying principles of their design and perform quantitative evaluations to compare them. These evaluations uncover significant reliability issues in detecting successful attacks, highlighting a fundamental gap for future development. Additionally, we contribute a preliminary labelled dataset, which serves as an initial step to bridge this gap. Based on the above, we provide strategic recommendations to assist organizations choose the most suitable scanner for their red-teaming needs, accounting for customizability, test suite comprehensiveness, and industry-specific use cases.



## **28. Sim-CLIP: Unsupervised Siamese Adversarial Fine-Tuning for Robust and Semantically-Rich Vision-Language Models**

cs.CV

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2407.14971v2) [paper-pdf](http://arxiv.org/pdf/2407.14971v2)

**Authors**: Md Zarif Hossain, Ahmed Imteaj

**Abstract**: Vision-language models (VLMs) have achieved significant strides in recent times specially in multimodal tasks, yet they remain susceptible to adversarial attacks on their vision components. To address this, we propose Sim-CLIP, an unsupervised adversarial fine-tuning method that enhances the robustness of the widely-used CLIP vision encoder against such attacks while maintaining semantic richness and specificity. By employing a Siamese architecture with cosine similarity loss, Sim-CLIP learns semantically meaningful and attack-resilient visual representations without requiring large batch sizes or momentum encoders. Our results demonstrate that VLMs enhanced with Sim-CLIP's fine-tuned CLIP encoder exhibit significantly enhanced robustness against adversarial attacks, while preserving semantic meaning of the perturbed images. Notably, Sim-CLIP does not require additional training or fine-tuning of the VLM itself; replacing the original vision encoder with our fine-tuned Sim-CLIP suffices to provide robustness. This work underscores the significance of reinforcing foundational models like CLIP to safeguard the reliability of downstream VLM applications, paving the way for more secure and effective multimodal systems.



## **29. Comparing Robustness Against Adversarial Attacks in Code Generation: LLM-Generated vs. Human-Written**

cs.SE

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2411.10565v1) [paper-pdf](http://arxiv.org/pdf/2411.10565v1)

**Authors**: Md Abdul Awal, Mrigank Rochan, Chanchal K. Roy

**Abstract**: Thanks to the widespread adoption of Large Language Models (LLMs) in software engineering research, the long-standing dream of automated code generation has become a reality on a large scale. Nowadays, LLMs such as GitHub Copilot and ChatGPT are extensively used in code generation for enterprise and open-source software development and maintenance. Despite their unprecedented successes in code generation, research indicates that codes generated by LLMs exhibit vulnerabilities and security issues. Several studies have been conducted to evaluate code generated by LLMs, considering various aspects such as security, vulnerability, code smells, and robustness. While some studies have compared the performance of LLMs with that of humans in various software engineering tasks, there's a notable gap in research: no studies have directly compared human-written and LLM-generated code for their robustness analysis. To fill this void, this paper introduces an empirical study to evaluate the adversarial robustness of Pre-trained Models of Code (PTMCs) fine-tuned on code written by humans and generated by LLMs against adversarial attacks for software clone detection. These attacks could potentially undermine software security and reliability. We consider two datasets, two state-of-the-art PTMCs, two robustness evaluation criteria, and three metrics to use in our experiments. Regarding effectiveness criteria, PTMCs fine-tuned on human-written code always demonstrate more robustness than those fine-tuned on LLMs-generated code. On the other hand, in terms of adversarial code quality, in 75% experimental combinations, PTMCs fine-tuned on the human-written code exhibit more robustness than the PTMCs fine-tuned on the LLMs-generated code.



## **30. On the Privacy Risk of In-context Learning**

cs.LG

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2411.10512v1) [paper-pdf](http://arxiv.org/pdf/2411.10512v1)

**Authors**: Haonan Duan, Adam Dziedzic, Mohammad Yaghini, Nicolas Papernot, Franziska Boenisch

**Abstract**: Large language models (LLMs) are excellent few-shot learners. They can perform a wide variety of tasks purely based on natural language prompts provided to them. These prompts contain data of a specific downstream task -- often the private dataset of a party, e.g., a company that wants to leverage the LLM for their purposes. We show that deploying prompted models presents a significant privacy risk for the data used within the prompt by instantiating a highly effective membership inference attack. We also observe that the privacy risk of prompted models exceeds fine-tuned models at the same utility levels. After identifying the model's sensitivity to their prompts -- in the form of a significantly higher prediction confidence on the prompted data -- as a cause for the increased risk, we propose ensembling as a mitigation strategy. By aggregating over multiple different versions of a prompted model, membership inference risk can be decreased.



## **31. Optimization-based Prompt Injection Attack to LLM-as-a-Judge**

cs.CR

To appear in the Proceedings of The ACM Conference on Computer and  Communications Security (CCS), 2024

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2403.17710v3) [paper-pdf](http://arxiv.org/pdf/2403.17710v3)

**Authors**: Jiawen Shi, Zenghui Yuan, Yinuo Liu, Yue Huang, Pan Zhou, Lichao Sun, Neil Zhenqiang Gong

**Abstract**: LLM-as-a-Judge uses a large language model (LLM) to select the best response from a set of candidates for a given question. LLM-as-a-Judge has many applications such as LLM-powered search, reinforcement learning with AI feedback (RLAIF), and tool selection. In this work, we propose JudgeDeceiver, an optimization-based prompt injection attack to LLM-as-a-Judge. JudgeDeceiver injects a carefully crafted sequence into an attacker-controlled candidate response such that LLM-as-a-Judge selects the candidate response for an attacker-chosen question no matter what other candidate responses are. Specifically, we formulate finding such sequence as an optimization problem and propose a gradient based method to approximately solve it. Our extensive evaluation shows that JudgeDeceive is highly effective, and is much more effective than existing prompt injection attacks that manually craft the injected sequences and jailbreak attacks when extended to our problem. We also show the effectiveness of JudgeDeceiver in three case studies, i.e., LLM-powered search, RLAIF, and tool selection. Moreover, we consider defenses including known-answer detection, perplexity detection, and perplexity windowed detection. Our results show these defenses are insufficient, highlighting the urgent need for developing new defense strategies. Our implementation is available at this repository: https://github.com/ShiJiawenwen/JudgeDeceiver.



## **32. IDEATOR: Jailbreaking Large Vision-Language Models Using Themselves**

cs.CV

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2411.00827v2) [paper-pdf](http://arxiv.org/pdf/2411.00827v2)

**Authors**: Ruofan Wang, Bo Wang, Xiaosen Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: As large Vision-Language Models (VLMs) grow in prominence, ensuring their safe deployment has become critical. Recent studies have explored VLM robustness against jailbreak attacks--techniques that exploit model vulnerabilities to elicit harmful outputs. However, the limited availability of diverse multi-modal data has led current approaches to rely heavily on adversarial or manually crafted images derived from harmful text datasets, which may lack effectiveness and diversity across different contexts. In this paper, we propose a novel jailbreak method named IDEATOR, which autonomously generates malicious image-text pairs for black-box jailbreak attacks. IDEATOR is based on the insight that VLMs themselves could serve as powerful red team models for generating multimodal jailbreak prompts. Specifically, IDEATOR uses a VLM to create targeted jailbreak texts and pairs them with jailbreak images generated by a state-of-the-art diffusion model. Our extensive experiments demonstrate IDEATOR's high effectiveness and transferability. Notably, it achieves a 94% success rate in jailbreaking MiniGPT-4 with an average of only 5.34 queries, and high success rates of 82%, 88%, and 75% when transferred to LLaVA, InstructBLIP, and Meta's Chameleon, respectively. IDEATOR uncovers specific vulnerabilities in VLMs under black-box conditions, underscoring the need for improved safety mechanisms.



## **33. Security and Privacy Challenges of Large Language Models: A Survey**

cs.CL

**SubmitDate**: 2024-11-14    [abs](http://arxiv.org/abs/2402.00888v2) [paper-pdf](http://arxiv.org/pdf/2402.00888v2)

**Authors**: Badhan Chandra Das, M. Hadi Amini, Yanzhao Wu

**Abstract**: Large Language Models (LLMs) have demonstrated extraordinary capabilities and contributed to multiple fields, such as generating and summarizing text, language translation, and question-answering. Nowadays, LLM is becoming a very popular tool in computerized language processing tasks, with the capability to analyze complicated linguistic patterns and provide relevant and appropriate responses depending on the context. While offering significant advantages, these models are also vulnerable to security and privacy attacks, such as jailbreaking attacks, data poisoning attacks, and Personally Identifiable Information (PII) leakage attacks. This survey provides a thorough review of the security and privacy challenges of LLMs for both training data and users, along with the application-based risks in various domains, such as transportation, education, and healthcare. We assess the extent of LLM vulnerabilities, investigate emerging security and privacy attacks for LLMs, and review the potential defense mechanisms. Additionally, the survey outlines existing research gaps in this domain and highlights future research directions.



## **34. AutoDefense: Multi-Agent LLM Defense against Jailbreak Attacks**

cs.LG

**SubmitDate**: 2024-11-14    [abs](http://arxiv.org/abs/2403.04783v2) [paper-pdf](http://arxiv.org/pdf/2403.04783v2)

**Authors**: Yifan Zeng, Yiran Wu, Xiao Zhang, Huazheng Wang, Qingyun Wu

**Abstract**: Despite extensive pre-training in moral alignment to prevent generating harmful information, large language models (LLMs) remain vulnerable to jailbreak attacks. In this paper, we propose AutoDefense, a multi-agent defense framework that filters harmful responses from LLMs. With the response-filtering mechanism, our framework is robust against different jailbreak attack prompts, and can be used to defend different victim models. AutoDefense assigns different roles to LLM agents and employs them to complete the defense task collaboratively. The division in tasks enhances the overall instruction-following of LLMs and enables the integration of other defense components as tools. With AutoDefense, small open-source LMs can serve as agents and defend larger models against jailbreak attacks. Our experiments show that AutoDefense can effectively defense against different jailbreak attacks, while maintaining the performance at normal user request. For example, we reduce the attack success rate on GPT-3.5 from 55.74% to 7.95% using LLaMA-2-13b with a 3-agent system. Our code and data are publicly available at https://github.com/XHMY/AutoDefense.



## **35. DROJ: A Prompt-Driven Attack against Large Language Models**

cs.CL

**SubmitDate**: 2024-11-14    [abs](http://arxiv.org/abs/2411.09125v1) [paper-pdf](http://arxiv.org/pdf/2411.09125v1)

**Authors**: Leyang Hu, Boran Wang

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional capabilities across various natural language processing tasks. Due to their training on internet-sourced datasets, LLMs can sometimes generate objectionable content, necessitating extensive alignment with human feedback to avoid such outputs. Despite massive alignment efforts, LLMs remain susceptible to adversarial jailbreak attacks, which usually are manipulated prompts designed to circumvent safety mechanisms and elicit harmful responses. Here, we introduce a novel approach, Directed Rrepresentation Optimization Jailbreak (DROJ), which optimizes jailbreak prompts at the embedding level to shift the hidden representations of harmful queries towards directions that are more likely to elicit affirmative responses from the model. Our evaluations on LLaMA-2-7b-chat model show that DROJ achieves a 100\% keyword-based Attack Success Rate (ASR), effectively preventing direct refusals. However, the model occasionally produces repetitive and non-informative responses. To mitigate this, we introduce a helpfulness system prompt that enhances the utility of the model's responses. Our code is available at https://github.com/Leon-Leyang/LLM-Safeguard.



## **36. Trustful LLMs: Customizing and Grounding Text Generation with Knowledge Bases and Dual Decoders**

cs.CL

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2411.07870v2) [paper-pdf](http://arxiv.org/pdf/2411.07870v2)

**Authors**: Xiaofeng Zhu, Jaya Krishna Mandivarapu

**Abstract**: Although people are impressed by the content generation skills of large language models, the use of LLMs, such as ChatGPT, is limited by the domain grounding of the content. The correctness and groundedness of the generated content need to be based on a verified context, such as results from Retrieval-Augmented Generation (RAG). One important issue when adapting LLMs to a customized domain is that the generated responses are often incomplete, or the additions are not verified and may even be hallucinated. Prior studies on hallucination detection have focused on evaluation metrics, which are not easily adaptable to dynamic domains and can be vulnerable to attacks like jail-breaking. In this work, we propose 1) a post-processing algorithm that leverages knowledge triplets in RAG context to correct hallucinations and 2) a dual-decoder model that fuses RAG context to guide the generation process.



## **37. Defending Large Language Models Against Attacks With Residual Stream Activation Analysis**

cs.CR

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2406.03230v4) [paper-pdf](http://arxiv.org/pdf/2406.03230v4)

**Authors**: Amelia Kawasaki, Andrew Davis, Houssam Abbas

**Abstract**: The widespread adoption of Large Language Models (LLMs), exemplified by OpenAI's ChatGPT, brings to the forefront the imperative to defend against adversarial threats on these models. These attacks, which manipulate an LLM's output by introducing malicious inputs, undermine the model's integrity and the trust users place in its outputs. In response to this challenge, our paper presents an innovative defensive strategy, given white box access to an LLM, that harnesses residual activation analysis between transformer layers of the LLM. We apply a novel methodology for analyzing distinctive activation patterns in the residual streams for attack prompt classification. We curate multiple datasets to demonstrate how this method of classification has high accuracy across multiple types of attack scenarios, including our newly-created attack dataset. Furthermore, we enhance the model's resilience by integrating safety fine-tuning techniques for LLMs in order to measure its effect on our capability to detect attacks. The results underscore the effectiveness of our approach in enhancing the detection and mitigation of adversarial inputs, advancing the security framework within which LLMs operate.



## **38. LLMStinger: Jailbreaking LLMs using RL fine-tuned LLMs**

cs.LG

Accepted at AAAI 2025

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2411.08862v1) [paper-pdf](http://arxiv.org/pdf/2411.08862v1)

**Authors**: Piyush Jha, Arnav Arora, Vijay Ganesh

**Abstract**: We introduce LLMStinger, a novel approach that leverages Large Language Models (LLMs) to automatically generate adversarial suffixes for jailbreak attacks. Unlike traditional methods, which require complex prompt engineering or white-box access, LLMStinger uses a reinforcement learning (RL) loop to fine-tune an attacker LLM, generating new suffixes based on existing attacks for harmful questions from the HarmBench benchmark. Our method significantly outperforms existing red-teaming approaches (we compared against 15 of the latest methods), achieving a +57.2% improvement in Attack Success Rate (ASR) on LLaMA2-7B-chat and a +50.3% ASR increase on Claude 2, both models known for their extensive safety measures. Additionally, we achieved a 94.97% ASR on GPT-3.5 and 99.4% on Gemma-2B-it, demonstrating the robustness and adaptability of LLMStinger across open and closed-source models.



## **39. Target-driven Attack for Large Language Models**

cs.CL

12 pages, 7 figures. This work is an extension of the  arXiv:2404.07234 work. We propose new methods. 27th European Conference on  Artificial Intelligence 2024

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2411.07268v2) [paper-pdf](http://arxiv.org/pdf/2411.07268v2)

**Authors**: Chong Zhang, Mingyu Jin, Dong Shu, Taowen Wang, Dongfang Liu, Xiaobo Jin

**Abstract**: Current large language models (LLM) provide a strong foundation for large-scale user-oriented natural language tasks. Many users can easily inject adversarial text or instructions through the user interface, thus causing LLM model security challenges like the language model not giving the correct answer. Although there is currently a large amount of research on black-box attacks, most of these black-box attacks use random and heuristic strategies. It is unclear how these strategies relate to the success rate of attacks and thus effectively improve model robustness. To solve this problem, we propose our target-driven black-box attack method to maximize the KL divergence between the conditional probabilities of the clean text and the attack text to redefine the attack's goal. We transform the distance maximization problem into two convex optimization problems based on the attack goal to solve the attack text and estimate the covariance. Furthermore, the projected gradient descent algorithm solves the vector corresponding to the attack text. Our target-driven black-box attack approach includes two attack strategies: token manipulation and misinformation attack. Experimental results on multiple Large Language Models and datasets demonstrate the effectiveness of our attack method.



## **40. DAGER: Exact Gradient Inversion for Large Language Models**

cs.LG

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2405.15586v2) [paper-pdf](http://arxiv.org/pdf/2405.15586v2)

**Authors**: Ivo Petrov, Dimitar I. Dimitrov, Maximilian Baader, Mark Niklas Müller, Martin Vechev

**Abstract**: Federated learning works by aggregating locally computed gradients from multiple clients, thus enabling collaborative training without sharing private client data. However, prior work has shown that the data can actually be recovered by the server using so-called gradient inversion attacks. While these attacks perform well when applied on images, they are limited in the text domain and only permit approximate reconstruction of small batches and short input sequences. In this work, we propose DAGER, the first algorithm to recover whole batches of input text exactly. DAGER leverages the low-rank structure of self-attention layer gradients and the discrete nature of token embeddings to efficiently check if a given token sequence is part of the client data. We use this check to exactly recover full batches in the honest-but-curious setting without any prior on the data for both encoder- and decoder-based architectures using exhaustive heuristic search and a greedy approach, respectively. We provide an efficient GPU implementation of DAGER and show experimentally that it recovers full batches of size up to 128 on large language models (LLMs), beating prior attacks in speed (20x at same batch size), scalability (10x larger batches), and reconstruction quality (ROUGE-1/2 > 0.99).



## **41. The VLLM Safety Paradox: Dual Ease in Jailbreak Attack and Defense**

cs.CR

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2411.08410v1) [paper-pdf](http://arxiv.org/pdf/2411.08410v1)

**Authors**: Yangyang Guo, Fangkai Jiao, Liqiang Nie, Mohan Kankanhalli

**Abstract**: The vulnerability of Vision Large Language Models (VLLMs) to jailbreak attacks appears as no surprise. However, recent defense mechanisms against these attacks have reached near-saturation performance on benchmarks, often with minimal effort. This simultaneous high performance in both attack and defense presents a perplexing paradox. Resolving it is critical for advancing the development of trustworthy models. To address this research gap, we first investigate why VLLMs are prone to these attacks. We then make a key observation: existing defense mechanisms suffer from an \textbf{over-prudence} problem, resulting in unexpected abstention even in the presence of benign inputs. Additionally, we find that the two representative evaluation methods for jailbreak often exhibit chance agreement. This limitation makes it potentially misleading when evaluating attack strategies or defense mechanisms. Beyond these empirical observations, our another contribution in this work is to repurpose the guardrails of LLMs on the shelf, as an effective alternative detector prior to VLLM response. We believe these findings offer useful insights to rethink the foundational development of VLLM safety with respect to benchmark datasets, evaluation methods, and defense strategies.



## **42. MultiKG: Multi-Source Threat Intelligence Aggregation for High-Quality Knowledge Graph Representation of Attack Techniques**

cs.CR

21 pages, 15 figures, 8 tables

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2411.08359v1) [paper-pdf](http://arxiv.org/pdf/2411.08359v1)

**Authors**: Jian Wang, Tiantian Zhu, Chunlin Xiong, Yan Chen

**Abstract**: The construction of attack technique knowledge graphs aims to transform various types of attack knowledge into structured representations for more effective attack procedure modeling. Existing methods typically rely on textual data, such as Cyber Threat Intelligence (CTI) reports, which are often coarse-grained and unstructured, resulting in incomplete and inaccurate knowledge graphs. To address these issues, we expand attack knowledge sources by incorporating audit logs and static code analysis alongside CTI reports, providing finer-grained data for constructing attack technique knowledge graphs.   We propose MultiKG, a fully automated framework that integrates multiple threat knowledge sources. MultiKG processes data from CTI reports, dynamic logs, and static code separately, then merges them into a unified attack knowledge graph. Through system design and the utilization of the Large Language Model (LLM), MultiKG automates the analysis, construction, and merging of attack graphs across these sources, producing a fine-grained, multi-source attack knowledge graph.   We implemented MultiKG and evaluated it using 1,015 real attack techniques and 9,006 attack intelligence entries from CTI reports. Results show that MultiKG effectively extracts attack knowledge graphs from diverse sources and aggregates them into accurate, comprehensive representations. Through case studies, we demonstrate that our approach directly benefits security tasks such as attack reconstruction and detection.



## **43. Can adversarial attacks by large language models be attributed?**

cs.AI

7 pages, 1 figure

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.08003v1) [paper-pdf](http://arxiv.org/pdf/2411.08003v1)

**Authors**: Manuel Cebrian, Jan Arne Telle

**Abstract**: Attributing outputs from Large Language Models (LLMs) in adversarial settings-such as cyberattacks and disinformation-presents significant challenges that are likely to grow in importance. We investigate this attribution problem using formal language theory, specifically language identification in the limit as introduced by Gold and extended by Angluin. By modeling LLM outputs as formal languages, we analyze whether finite text samples can uniquely pinpoint the originating model. Our results show that due to the non-identifiability of certain language classes, under some mild assumptions about overlapping outputs from fine-tuned models it is theoretically impossible to attribute outputs to specific LLMs with certainty. This holds also when accounting for expressivity limitations of Transformer architectures. Even with direct model access or comprehensive monitoring, significant computational hurdles impede attribution efforts. These findings highlight an urgent need for proactive measures to mitigate risks posed by adversarial LLM use as their influence continues to expand.



## **44. Chain Association-based Attacking and Shielding Natural Language Processing Systems**

cs.CL

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.07843v1) [paper-pdf](http://arxiv.org/pdf/2411.07843v1)

**Authors**: Jiacheng Huang, Long Chen

**Abstract**: Association as a gift enables people do not have to mention something in completely straightforward words and allows others to understand what they intend to refer to. In this paper, we propose a chain association-based adversarial attack against natural language processing systems, utilizing the comprehension gap between humans and machines. We first generate a chain association graph for Chinese characters based on the association paradigm for building search space of potential adversarial examples. Then, we introduce an discrete particle swarm optimization algorithm to search for the optimal adversarial examples. We conduct comprehensive experiments and show that advanced natural language processing models and applications, including large language models, are vulnerable to our attack, while humans appear good at understanding the perturbed text. We also explore two methods, including adversarial training and associative graph-based recovery, to shield systems from chain association-based attack. Since a few examples that use some derogatory terms, this paper contains materials that may be offensive or upsetting to some people.



## **45. Revisiting the Adversarial Robustness of Vision Language Models: a Multimodal Perspective**

cs.CV

17 pages, 13 figures

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2404.19287v3) [paper-pdf](http://arxiv.org/pdf/2404.19287v3)

**Authors**: Wanqi Zhou, Shuanghao Bai, Danilo P. Mandic, Qibin Zhao, Badong Chen

**Abstract**: Pretrained vision-language models (VLMs) like CLIP exhibit exceptional generalization across diverse downstream tasks. While recent studies reveal their vulnerability to adversarial attacks, research to date has primarily focused on enhancing the robustness of image encoders against image-based attacks, with defenses against text-based and multimodal attacks remaining largely unexplored. To this end, this work presents the first comprehensive study on improving the adversarial robustness of VLMs against attacks targeting image, text, and multimodal inputs. This is achieved by proposing multimodal contrastive adversarial training (MMCoA). Such an approach strengthens the robustness of both image and text encoders by aligning the clean text embeddings with adversarial image embeddings, and adversarial text embeddings with clean image embeddings. The robustness of the proposed MMCoA is examined against existing defense methods over image, text, and multimodal attacks on the CLIP model. Extensive experiments on 15 datasets across two tasks reveal the characteristics of different adversarial defense methods under distinct distribution shifts and dataset complexities across the three attack types. This paves the way for a unified framework of adversarial robustness against different modality attacks, opening up new possibilities for securing VLMs against multimodal attacks. The code is available at https://github.com/ElleZWQ/MMCoA.git.



## **46. Zer0-Jack: A Memory-efficient Gradient-based Jailbreaking Method for Black-box Multi-modal Large Language Models**

cs.LG

Accepted to Neurips SafeGenAi Workshop 2024

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.07559v1) [paper-pdf](http://arxiv.org/pdf/2411.07559v1)

**Authors**: Tiejin Chen, Kaishen Wang, Hua Wei

**Abstract**: Jailbreaking methods, which induce Multi-modal Large Language Models (MLLMs) to output harmful responses, raise significant safety concerns. Among these methods, gradient-based approaches, which use gradients to generate malicious prompts, have been widely studied due to their high success rates in white-box settings, where full access to the model is available. However, these methods have notable limitations: they require white-box access, which is not always feasible, and involve high memory usage. To address scenarios where white-box access is unavailable, attackers often resort to transfer attacks. In transfer attacks, malicious inputs generated using white-box models are applied to black-box models, but this typically results in reduced attack performance. To overcome these challenges, we propose Zer0-Jack, a method that bypasses the need for white-box access by leveraging zeroth-order optimization. We propose patch coordinate descent to efficiently generate malicious image inputs to directly attack black-box MLLMs, which significantly reduces memory usage further. Through extensive experiments, Zer0-Jack achieves a high attack success rate across various models, surpassing previous transfer-based methods and performing comparably with existing white-box jailbreak techniques. Notably, Zer0-Jack achieves a 95\% attack success rate on MiniGPT-4 with the Harmful Behaviors Multi-modal Dataset on a black-box setting, demonstrating its effectiveness. Additionally, we show that Zer0-Jack can directly attack commercial MLLMs such as GPT-4o. Codes are provided in the supplement.



## **47. On Active Privacy Auditing in Supervised Fine-tuning for White-Box Language Models**

cs.CL

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.07070v2) [paper-pdf](http://arxiv.org/pdf/2411.07070v2)

**Authors**: Qian Sun, Hanpeng Wu, Xi Sheryl Zhang

**Abstract**: The pretraining and fine-tuning approach has become the leading technique for various NLP applications. However, recent studies reveal that fine-tuning data, due to their sensitive nature, domain-specific characteristics, and identifiability, pose significant privacy concerns. To help develop more privacy-resilient fine-tuning models, we introduce a novel active privacy auditing framework, dubbed Parsing, designed to identify and quantify privacy leakage risks during the supervised fine-tuning (SFT) of language models (LMs). The framework leverages improved white-box membership inference attacks (MIAs) as the core technology, utilizing novel learning objectives and a two-stage pipeline to monitor the privacy of the LMs' fine-tuning process, maximizing the exposure of privacy risks. Additionally, we have improved the effectiveness of MIAs on large LMs including GPT-2, Llama2, and certain variants of them. Our research aims to provide the SFT community of LMs with a reliable, ready-to-use privacy auditing tool, and to offer valuable insights into safeguarding privacy during the fine-tuning process. Experimental results confirm the framework's efficiency across various models and tasks, emphasizing notable privacy concerns in the fine-tuning process. Project code available for https://anonymous.4open.science/r/PARSING-4817/.



## **48. vTune: Verifiable Fine-Tuning for LLMs Through Backdooring**

cs.LG

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.06611v2) [paper-pdf](http://arxiv.org/pdf/2411.06611v2)

**Authors**: Eva Zhang, Arka Pal, Akilesh Potti, Micah Goldblum

**Abstract**: As fine-tuning large language models (LLMs) becomes increasingly prevalent, users often rely on third-party services with limited visibility into their fine-tuning processes. This lack of transparency raises the question: how do consumers verify that fine-tuning services are performed correctly? For instance, a service provider could claim to fine-tune a model for each user, yet simply send all users back the same base model. To address this issue, we propose vTune, a simple method that uses a small number of backdoor data points added to the training data to provide a statistical test for verifying that a provider fine-tuned a custom model on a particular user's dataset. Unlike existing works, vTune is able to scale to verification of fine-tuning on state-of-the-art LLMs, and can be used both with open-source and closed-source models. We test our approach across several model families and sizes as well as across multiple instruction-tuning datasets, and find that the statistical test is satisfied with p-values on the order of $\sim 10^{-40}$, with no negative impact on downstream task performance. Further, we explore several attacks that attempt to subvert vTune and demonstrate the method's robustness to these attacks.



## **49. Rapid Response: Mitigating LLM Jailbreaks with a Few Examples**

cs.CL

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.07494v1) [paper-pdf](http://arxiv.org/pdf/2411.07494v1)

**Authors**: Alwin Peng, Julian Michael, Henry Sleight, Ethan Perez, Mrinank Sharma

**Abstract**: As large language models (LLMs) grow more powerful, ensuring their safety against misuse becomes crucial. While researchers have focused on developing robust defenses, no method has yet achieved complete invulnerability to attacks. We propose an alternative approach: instead of seeking perfect adversarial robustness, we develop rapid response techniques to look to block whole classes of jailbreaks after observing only a handful of attacks. To study this setting, we develop RapidResponseBench, a benchmark that measures a defense's robustness against various jailbreak strategies after adapting to a few observed examples. We evaluate five rapid response methods, all of which use jailbreak proliferation, where we automatically generate additional jailbreaks similar to the examples observed. Our strongest method, which fine-tunes an input classifier to block proliferated jailbreaks, reduces attack success rate by a factor greater than 240 on an in-distribution set of jailbreaks and a factor greater than 15 on an out-of-distribution set, having observed just one example of each jailbreaking strategy. Moreover, further studies suggest that the quality of proliferation model and number of proliferated examples play an key role in the effectiveness of this defense. Overall, our results highlight the potential of responding rapidly to novel jailbreaks to limit LLM misuse.



## **50. DrAttack: Prompt Decomposition and Reconstruction Makes Powerful LLM Jailbreakers**

cs.CR

**SubmitDate**: 2024-11-11    [abs](http://arxiv.org/abs/2402.16914v3) [paper-pdf](http://arxiv.org/pdf/2402.16914v3)

**Authors**: Xirui Li, Ruochen Wang, Minhao Cheng, Tianyi Zhou, Cho-Jui Hsieh

**Abstract**: The safety alignment of Large Language Models (LLMs) is vulnerable to both manual and automated jailbreak attacks, which adversarially trigger LLMs to output harmful content. However, current methods for jailbreaking LLMs, which nest entire harmful prompts, are not effective at concealing malicious intent and can be easily identified and rejected by well-aligned LLMs. This paper discovers that decomposing a malicious prompt into separated sub-prompts can effectively obscure its underlying malicious intent by presenting it in a fragmented, less detectable form, thereby addressing these limitations. We introduce an automatic prompt \textbf{D}ecomposition and \textbf{R}econstruction framework for jailbreak \textbf{Attack} (DrAttack). DrAttack includes three key components: (a) `Decomposition' of the original prompt into sub-prompts, (b) `Reconstruction' of these sub-prompts implicitly by in-context learning with semantically similar but harmless reassembling demo, and (c) a `Synonym Search' of sub-prompts, aiming to find sub-prompts' synonyms that maintain the original intent while jailbreaking LLMs. An extensive empirical study across multiple open-source and closed-source LLMs demonstrates that, with a significantly reduced number of queries, DrAttack obtains a substantial gain of success rate over prior SOTA prompt-only attackers. Notably, the success rate of 78.0\% on GPT-4 with merely 15 queries surpassed previous art by 33.1\%. The project is available at https://github.com/xirui-li/DrAttack.



