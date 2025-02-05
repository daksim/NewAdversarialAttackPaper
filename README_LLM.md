# Latest Large Language Model Attack Papers
**update at 2025-02-05 12:08:58**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Medical Multimodal Model Stealing Attacks via Adversarial Domain Alignment**

cs.CR

Accepted at AAAI 2025

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2502.02438v1) [paper-pdf](http://arxiv.org/pdf/2502.02438v1)

**Authors**: Yaling Shen, Zhixiong Zhuang, Kun Yuan, Maria-Irina Nicolae, Nassir Navab, Nicolas Padoy, Mario Fritz

**Abstract**: Medical multimodal large language models (MLLMs) are becoming an instrumental part of healthcare systems, assisting medical personnel with decision making and results analysis. Models for radiology report generation are able to interpret medical imagery, thus reducing the workload of radiologists. As medical data is scarce and protected by privacy regulations, medical MLLMs represent valuable intellectual property. However, these assets are potentially vulnerable to model stealing, where attackers aim to replicate their functionality via black-box access. So far, model stealing for the medical domain has focused on classification; however, existing attacks are not effective against MLLMs. In this paper, we introduce Adversarial Domain Alignment (ADA-STEAL), the first stealing attack against medical MLLMs. ADA-STEAL relies on natural images, which are public and widely available, as opposed to their medical counterparts. We show that data augmentation with adversarial noise is sufficient to overcome the data distribution gap between natural images and the domain-specific distribution of the victim MLLM. Experiments on the IU X-RAY and MIMIC-CXR radiology datasets demonstrate that Adversarial Domain Alignment enables attackers to steal the medical MLLM without any access to medical data.



## **2. JailbreakEval: An Integrated Toolkit for Evaluating Jailbreak Attempts Against Large Language Models**

cs.CR

This is the Extended Version for the Poster at NDSS Symposium 2025,  Feb 24-28, 2025. Our code is available at  https://github.com/ThuCCSLab/JailbreakEval

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2406.09321v2) [paper-pdf](http://arxiv.org/pdf/2406.09321v2)

**Authors**: Delong Ran, Jinyuan Liu, Yichen Gong, Jingyi Zheng, Xinlei He, Tianshuo Cong, Anyu Wang

**Abstract**: Jailbreak attacks induce Large Language Models (LLMs) to generate harmful responses, posing severe misuse threats. Though research on jailbreak attacks and defenses is emerging, there is no consensus on evaluating jailbreaks, i.e., the methods to assess the harmfulness of an LLM's response are varied. Each approach has its own set of strengths and weaknesses, impacting their alignment with human values, as well as the time and financial cost. This diversity challenges researchers in choosing suitable evaluation methods and comparing different attacks and defenses. In this paper, we conduct a comprehensive analysis of jailbreak evaluation methodologies, drawing from nearly 90 jailbreak research published between May 2023 and April 2024. Our study introduces a systematic taxonomy of jailbreak evaluators, offering indepth insights into their strengths and weaknesses, along with the current status of their adaptation. To aid further research, we propose JailbreakEval, a toolkit for evaluating jailbreak attempts. JailbreakEval includes various evaluators out-of-the-box, enabling users to obtain results with a single command or customized evaluation workflows. In summary, we regard JailbreakEval to be a catalyst that simplifies the evaluation process in jailbreak research and fosters an inclusive standard for jailbreak evaluation within the community.



## **3. Is poisoning a real threat to LLM alignment? Maybe more so than you think**

cs.LG

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2406.12091v3) [paper-pdf](http://arxiv.org/pdf/2406.12091v3)

**Authors**: Pankayaraj Pathmanathan, Souradip Chakraborty, Xiangyu Liu, Yongyuan Liang, Furong Huang

**Abstract**: Recent advancements in Reinforcement Learning with Human Feedback (RLHF) have significantly impacted the alignment of Large Language Models (LLMs). The sensitivity of reinforcement learning algorithms such as Proximal Policy Optimization (PPO) has led to new line work on Direct Policy Optimization (DPO), which treats RLHF in a supervised learning framework. The increased practical use of these RLHF methods warrants an analysis of their vulnerabilities. In this work, we investigate the vulnerabilities of DPO to poisoning attacks under different scenarios and compare the effectiveness of preference poisoning, a first of its kind. We comprehensively analyze DPO's vulnerabilities under different types of attacks, i.e., backdoor and non-backdoor attacks, and different poisoning methods across a wide array of language models, i.e., LLama 7B, Mistral 7B, and Gemma 7B. We find that unlike PPO-based methods, which, when it comes to backdoor attacks, require at least 4\% of the data to be poisoned to elicit harmful behavior, we exploit the true vulnerabilities of DPO more simply so we can poison the model with only as much as 0.5\% of the data. We further investigate the potential reasons behind the vulnerability and how well this vulnerability translates into backdoor vs non-backdoor attacks.



## **4. STAIR: Improving Safety Alignment with Introspective Reasoning**

cs.CL

22 pages, 8 figures

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2502.02384v1) [paper-pdf](http://arxiv.org/pdf/2502.02384v1)

**Authors**: Yichi Zhang, Siyuan Zhang, Yao Huang, Zeyu Xia, Zhengwei Fang, Xiao Yang, Ranjie Duan, Dong Yan, Yinpeng Dong, Jun Zhu

**Abstract**: Ensuring the safety and harmlessness of Large Language Models (LLMs) has become equally critical as their performance in applications. However, existing safety alignment methods typically suffer from safety-performance trade-offs and the susceptibility to jailbreak attacks, primarily due to their reliance on direct refusals for malicious queries. In this paper, we propose STAIR, a novel framework that integrates SafeTy Alignment with Itrospective Reasoning. We enable LLMs to identify safety risks through step-by-step analysis by self-improving chain-of-thought (CoT) reasoning with safety awareness. STAIR first equips the model with a structured reasoning capability and then advances safety alignment via iterative preference optimization on step-level reasoning data generated using our newly proposed Safety-Informed Monte Carlo Tree Search (SI-MCTS). We further train a process reward model on this data to guide test-time searches for improved responses. Extensive experiments show that STAIR effectively mitigates harmful outputs while better preserving helpfulness, compared to instinctive alignment strategies. With test-time scaling, STAIR achieves a safety performance comparable to Claude-3.5 against popular jailbreak attacks. Relevant resources in this work are available at https://github.com/thu-ml/STAIR.



## **5. SHIELD: APT Detection and Intelligent Explanation Using LLM**

cs.CR

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2502.02342v1) [paper-pdf](http://arxiv.org/pdf/2502.02342v1)

**Authors**: Parth Atulbhai Gandhi, Prasanna N. Wudali, Yonatan Amaru, Yuval Elovici, Asaf Shabtai

**Abstract**: Advanced persistent threats (APTs) are sophisticated cyber attacks that can remain undetected for extended periods, making their mitigation particularly challenging. Given their persistence, significant effort is required to detect them and respond effectively. Existing provenance-based attack detection methods often lack interpretability and suffer from high false positive rates, while investigation approaches are either supervised or limited to known attacks. To address these challenges, we introduce SHIELD, a novel approach that combines statistical anomaly detection and graph-based analysis with the contextual analysis capabilities of large language models (LLMs). SHIELD leverages the implicit knowledge of LLMs to uncover hidden attack patterns in provenance data, while reducing false positives and providing clear, interpretable attack descriptions. This reduces analysts' alert fatigue and makes it easier for them to understand the threat landscape. Our extensive evaluation demonstrates SHIELD's effectiveness and computational efficiency in real-world scenarios. SHIELD was shown to outperform state-of-the-art methods, achieving higher precision and recall. SHIELD's integration of anomaly detection, LLM-driven contextual analysis, and advanced graph-based correlation establishes a new benchmark for APT detection.



## **6. BadRobot: Jailbreaking Embodied LLMs in the Physical World**

cs.CY

Accepted to ICLR 2025. Project page:  https://Embodied-LLMs-Safety.github.io

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2407.20242v4) [paper-pdf](http://arxiv.org/pdf/2407.20242v4)

**Authors**: Hangtao Zhang, Chenyu Zhu, Xianlong Wang, Ziqi Zhou, Changgan Yin, Minghui Li, Lulu Xue, Yichen Wang, Shengshan Hu, Aishan Liu, Peijin Guo, Leo Yu Zhang

**Abstract**: Embodied AI represents systems where AI is integrated into physical entities. Large Language Model (LLM), which exhibits powerful language understanding abilities, has been extensively employed in embodied AI by facilitating sophisticated task planning. However, a critical safety issue remains overlooked: could these embodied LLMs perpetrate harmful behaviors? In response, we introduce BadRobot, a novel attack paradigm aiming to make embodied LLMs violate safety and ethical constraints through typical voice-based user-system interactions. Specifically, three vulnerabilities are exploited to achieve this type of attack: (i) manipulation of LLMs within robotic systems, (ii) misalignment between linguistic outputs and physical actions, and (iii) unintentional hazardous behaviors caused by world knowledge's flaws. Furthermore, we construct a benchmark of various malicious physical action queries to evaluate BadRobot's attack performance. Based on this benchmark, extensive experiments against existing prominent embodied LLM frameworks (e.g., Voxposer, Code as Policies, and ProgPrompt) demonstrate the effectiveness of our BadRobot.



## **7. Adversarial Reasoning at Jailbreaking Time**

cs.LG

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01633v1) [paper-pdf](http://arxiv.org/pdf/2502.01633v1)

**Authors**: Mahdi Sabbaghi, Paul Kassianik, George Pappas, Yaron Singer, Amin Karbasi, Hamed Hassani

**Abstract**: As large language models (LLMs) are becoming more capable and widespread, the study of their failure cases is becoming increasingly important. Recent advances in standardizing, measuring, and scaling test-time compute suggest new methodologies for optimizing models to achieve high performance on hard tasks. In this paper, we apply these advances to the task of model jailbreaking: eliciting harmful responses from aligned LLMs. We develop an adversarial reasoning approach to automatic jailbreaking via test-time computation that achieves SOTA attack success rates (ASR) against many aligned LLMs, even the ones that aim to trade inference-time compute for adversarial robustness. Our approach introduces a new paradigm in understanding LLM vulnerabilities, laying the foundation for the development of more robust and trustworthy AI systems.



## **8. Robust-LLaVA: On the Effectiveness of Large-Scale Robust Image Encoders for Multi-modal Large Language Models**

cs.CV

Under Review

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01576v1) [paper-pdf](http://arxiv.org/pdf/2502.01576v1)

**Authors**: Hashmat Shadab Malik, Fahad Shamshad, Muzammal Naseer, Karthik Nandakumar, Fahad Khan, Salman Khan

**Abstract**: Multi-modal Large Language Models (MLLMs) excel in vision-language tasks but remain vulnerable to visual adversarial perturbations that can induce hallucinations, manipulate responses, or bypass safety mechanisms. Existing methods seek to mitigate these risks by applying constrained adversarial fine-tuning to CLIP vision encoders on ImageNet-scale data, ensuring their generalization ability is preserved. However, this limited adversarial training restricts robustness and broader generalization. In this work, we explore an alternative approach of leveraging existing vision classification models that have been adversarially pre-trained on large-scale data. Our analysis reveals two principal contributions: (1) the extensive scale and diversity of adversarial pre-training enables these models to demonstrate superior robustness against diverse adversarial threats, ranging from imperceptible perturbations to advanced jailbreaking attempts, without requiring additional adversarial training, and (2) end-to-end MLLM integration with these robust models facilitates enhanced adaptation of language components to robust visual features, outperforming existing plug-and-play methodologies on complex reasoning tasks. Through systematic evaluation across visual question-answering, image captioning, and jail-break attacks, we demonstrate that MLLMs trained with these robust models achieve superior adversarial robustness while maintaining favorable clean performance. Our framework achieves 2x and 1.5x average robustness gains in captioning and VQA tasks, respectively, and delivers over 10% improvement against jailbreak attacks. Code and pretrained models will be available at https://github.com/HashmatShadab/Robust-LLaVA.



## **9. Scaling Up Membership Inference: When and How Attacks Succeed on Large Language Models**

cs.CL

Findings of NAACL 2025. Our code is available at  https://github.com/parameterlab/mia-scaling

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2411.00154v2) [paper-pdf](http://arxiv.org/pdf/2411.00154v2)

**Authors**: Haritz Puerto, Martin Gubri, Sangdoo Yun, Seong Joon Oh

**Abstract**: Membership inference attacks (MIA) attempt to verify the membership of a given data sample in the training set for a model. MIA has become relevant in recent years, following the rapid development of large language models (LLM). Many are concerned about the usage of copyrighted materials for training them and call for methods for detecting such usage. However, recent research has largely concluded that current MIA methods do not work on LLMs. Even when they seem to work, it is usually because of the ill-designed experimental setup where other shortcut features enable "cheating." In this work, we argue that MIA still works on LLMs, but only when multiple documents are presented for testing. We construct new benchmarks that measure the MIA performances at a continuous scale of data samples, from sentences (n-grams) to a collection of documents (multiple chunks of tokens). To validate the efficacy of current MIA approaches at greater scales, we adapt a recent work on Dataset Inference (DI) for the task of binary membership detection that aggregates paragraph-level MIA features to enable MIA at document and collection of documents level. This baseline achieves the first successful MIA on pre-trained and fine-tuned LLMs.



## **10. Topic-FlipRAG: Topic-Orientated Adversarial Opinion Manipulation Attacks to Retrieval-Augmented Generation Models**

cs.CL

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01386v1) [paper-pdf](http://arxiv.org/pdf/2502.01386v1)

**Authors**: Yuyang Gong, Zhuo Chen, Miaokun Chen, Fengchang Yu, Wei Lu, Xiaofeng Wang, Xiaozhong Liu, Jiawei Liu

**Abstract**: Retrieval-Augmented Generation (RAG) systems based on Large Language Models (LLMs) have become essential for tasks such as question answering and content generation. However, their increasing impact on public opinion and information dissemination has made them a critical focus for security research due to inherent vulnerabilities. Previous studies have predominantly addressed attacks targeting factual or single-query manipulations. In this paper, we address a more practical scenario: topic-oriented adversarial opinion manipulation attacks on RAG models, where LLMs are required to reason and synthesize multiple perspectives, rendering them particularly susceptible to systematic knowledge poisoning. Specifically, we propose Topic-FlipRAG, a two-stage manipulation attack pipeline that strategically crafts adversarial perturbations to influence opinions across related queries. This approach combines traditional adversarial ranking attack techniques and leverages the extensive internal relevant knowledge and reasoning capabilities of LLMs to execute semantic-level perturbations. Experiments show that the proposed attacks effectively shift the opinion of the model's outputs on specific topics, significantly impacting user information perception. Current mitigation methods cannot effectively defend against such attacks, highlighting the necessity for enhanced safeguards for RAG systems, and offering crucial insights for LLM security research.



## **11. Detecting Backdoor Samples in Contrastive Language Image Pretraining**

cs.LG

ICLR2025

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01385v1) [paper-pdf](http://arxiv.org/pdf/2502.01385v1)

**Authors**: Hanxun Huang, Sarah Erfani, Yige Li, Xingjun Ma, James Bailey

**Abstract**: Contrastive language-image pretraining (CLIP) has been found to be vulnerable to poisoning backdoor attacks where the adversary can achieve an almost perfect attack success rate on CLIP models by poisoning only 0.01\% of the training dataset. This raises security concerns on the current practice of pretraining large-scale models on unscrutinized web data using CLIP. In this work, we analyze the representations of backdoor-poisoned samples learned by CLIP models and find that they exhibit unique characteristics in their local subspace, i.e., their local neighborhoods are far more sparse than that of clean samples. Based on this finding, we conduct a systematic study on detecting CLIP backdoor attacks and show that these attacks can be easily and efficiently detected by traditional density ratio-based local outlier detectors, whereas existing backdoor sample detection methods fail. Our experiments also reveal that an unintentional backdoor already exists in the original CC3M dataset and has been trained into a popular open-source model released by OpenCLIP. Based on our detector, one can clean up a million-scale web dataset (e.g., CC3M) efficiently within 15 minutes using 4 Nvidia A100 GPUs. The code is publicly available in our \href{https://github.com/HanxunH/Detect-CLIP-Backdoor-Samples}{GitHub repository}.



## **12. Improving the Robustness of Representation Misdirection for Large Language Model Unlearning**

cs.CL

12 pages, 4 figures, 1 table

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2501.19202v2) [paper-pdf](http://arxiv.org/pdf/2501.19202v2)

**Authors**: Dang Huu-Tien, Hoang Thanh-Tung, Le-Minh Nguyen, Naoya Inoue

**Abstract**: Representation Misdirection (RM) and variants are established large language model (LLM) unlearning methods with state-of-the-art performance. In this paper, we show that RM methods inherently reduce models' robustness, causing them to misbehave even when a single non-adversarial forget-token is in the retain-query. Toward understanding underlying causes, we reframe the unlearning process as backdoor attacks and defenses: forget-tokens act as backdoor triggers that, when activated in retain-queries, cause disruptions in RM models' behaviors, similar to successful backdoor attacks. To mitigate this vulnerability, we propose Random Noise Augmentation -- a model and method agnostic approach with theoretical guarantees for improving the robustness of RM methods. Extensive experiments demonstrate that RNA significantly improves the robustness of RM models while enhancing the unlearning performances.



## **13. PBI-Attack: Prior-Guided Bimodal Interactive Black-Box Jailbreak Attack for Toxicity Maximization**

cs.CR

Prior-Guided Bimodal Interactive Black-Box Jailbreak Attack for  Toxicity Maximization

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2412.05892v3) [paper-pdf](http://arxiv.org/pdf/2412.05892v3)

**Authors**: Ruoxi Cheng, Yizhong Ding, Shuirong Cao, Ranjie Duan, Xiaoshuang Jia, Shaowei Yuan, Zhiqiang Wang, Xiaojun Jia

**Abstract**: Understanding the vulnerabilities of Large Vision Language Models (LVLMs) to jailbreak attacks is essential for their responsible real-world deployment. Most previous work requires access to model gradients, or is based on human knowledge (prompt engineering) to complete jailbreak, and they hardly consider the interaction of images and text, resulting in inability to jailbreak in black box scenarios or poor performance. To overcome these limitations, we propose a Prior-Guided Bimodal Interactive Black-Box Jailbreak Attack for toxicity maximization, referred to as PBI-Attack. Our method begins by extracting malicious features from a harmful corpus using an alternative LVLM and embedding these features into a benign image as prior information. Subsequently, we enhance these features through bidirectional cross-modal interaction optimization, which iteratively optimizes the bimodal perturbations in an alternating manner through greedy search, aiming to maximize the toxicity of the generated response. The toxicity level is quantified using a well-trained evaluation model. Experiments demonstrate that PBI-Attack outperforms previous state-of-the-art jailbreak methods, achieving an average attack success rate of 92.5% across three open-source LVLMs and around 67.3% on three closed-source LVLMs. Disclaimer: This paper contains potentially disturbing and offensive content.



## **14. Eliciting Language Model Behaviors with Investigator Agents**

cs.LG

20 pages, 7 figures

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01236v1) [paper-pdf](http://arxiv.org/pdf/2502.01236v1)

**Authors**: Xiang Lisa Li, Neil Chowdhury, Daniel D. Johnson, Tatsunori Hashimoto, Percy Liang, Sarah Schwettmann, Jacob Steinhardt

**Abstract**: Language models exhibit complex, diverse behaviors when prompted with free-form text, making it difficult to characterize the space of possible outputs. We study the problem of behavior elicitation, where the goal is to search for prompts that induce specific target behaviors (e.g., hallucinations or harmful responses) from a target language model. To navigate the exponentially large space of possible prompts, we train investigator models to map randomly-chosen target behaviors to a diverse distribution of outputs that elicit them, similar to amortized Bayesian inference. We do this through supervised fine-tuning, reinforcement learning via DPO, and a novel Frank-Wolfe training objective to iteratively discover diverse prompting strategies. Our investigator models surface a variety of effective and human-interpretable prompts leading to jailbreaks, hallucinations, and open-ended aberrant behaviors, obtaining a 100% attack success rate on a subset of AdvBench (Harmful Behaviors) and an 85% hallucination rate.



## **15. The dark deep side of DeepSeek: Fine-tuning attacks against the safety alignment of CoT-enabled models**

cs.CR

12 Pages

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01225v1) [paper-pdf](http://arxiv.org/pdf/2502.01225v1)

**Authors**: Zhiyuan Xu, Joseph Gardiner, Sana Belguith

**Abstract**: Large language models are typically trained on vast amounts of data during the pre-training phase, which may include some potentially harmful information. Fine-tuning attacks can exploit this by prompting the model to reveal such behaviours, leading to the generation of harmful content. In this paper, we focus on investigating the performance of the Chain of Thought based reasoning model, DeepSeek, when subjected to fine-tuning attacks. Specifically, we explore how fine-tuning manipulates the model's output, exacerbating the harmfulness of its responses while examining the interaction between the Chain of Thought reasoning and adversarial inputs. Through this study, we aim to shed light on the vulnerability of Chain of Thought enabled models to fine-tuning attacks and the implications for their safety and ethical deployment.



## **16. Jailbreaking with Universal Multi-Prompts**

cs.CL

Accepted by NAACL Findings 2025

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01154v1) [paper-pdf](http://arxiv.org/pdf/2502.01154v1)

**Authors**: Yu-Ling Hsu, Hsuan Su, Shang-Tse Chen

**Abstract**: Large language models (LLMs) have seen rapid development in recent years, revolutionizing various applications and significantly enhancing convenience and productivity. However, alongside their impressive capabilities, ethical concerns and new types of attacks, such as jailbreaking, have emerged. While most prompting techniques focus on optimizing adversarial inputs for individual cases, resulting in higher computational costs when dealing with large datasets. Less research has addressed the more general setting of training a universal attacker that can transfer to unseen tasks. In this paper, we introduce JUMP, a prompt-based method designed to jailbreak LLMs using universal multi-prompts. We also adapt our approach for defense, which we term DUMP. Experimental results demonstrate that our method for optimizing universal multi-prompts outperforms existing techniques.



## **17. When LLMs Go Online: The Emerging Threat of Web-Enabled LLMs**

cs.CR

20 pages, To appear in Usenix Security 2025

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2410.14569v3) [paper-pdf](http://arxiv.org/pdf/2410.14569v3)

**Authors**: Hanna Kim, Minkyoo Song, Seung Ho Na, Seungwon Shin, Kimin Lee

**Abstract**: Recent advancements in Large Language Models (LLMs) have established them as agentic systems capable of planning and interacting with various tools. These LLM agents are often paired with web-based tools, enabling access to diverse sources and real-time information. Although these advancements offer significant benefits across various applications, they also increase the risk of malicious use, particularly in cyberattacks involving personal information. In this work, we investigate the risks associated with misuse of LLM agents in cyberattacks involving personal data. Specifically, we aim to understand: 1) how potent LLM agents can be when directed to conduct cyberattacks, 2) how cyberattacks are enhanced by web-based tools, and 3) how affordable and easy it becomes to launch cyberattacks using LLM agents. We examine three attack scenarios: the collection of Personally Identifiable Information (PII), the generation of impersonation posts, and the creation of spear-phishing emails. Our experiments reveal the effectiveness of LLM agents in these attacks: LLM agents achieved a precision of up to 95.9% in collecting PII, generated impersonation posts where 93.9% of them were deemed authentic, and boosted click rate of phishing links in spear phishing emails by 46.67%. Additionally, our findings underscore the limitations of existing safeguards in contemporary commercial LLMs, emphasizing the urgent need for robust security measures to prevent the misuse of LLM agents.



## **18. Tool Unlearning for Tool-Augmented LLMs**

cs.LG

https://clu-uml.github.io/MU-Bench-Project-Page/

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01083v1) [paper-pdf](http://arxiv.org/pdf/2502.01083v1)

**Authors**: Jiali Cheng, Hadi Amiri

**Abstract**: Tool-augmented large language models (LLMs) are often trained on datasets of query-response pairs, which embed the ability to use tools or APIs directly into the parametric knowledge of LLMs. Tool-augmented LLMs need the ability to forget learned tools due to security vulnerabilities, privacy regulations, or tool deprecations. However, ``tool unlearning'' has not been investigated in unlearning literature. We introduce this novel task, which requires addressing distinct challenges compared to traditional unlearning: knowledge removal rather than forgetting individual samples, the high cost of optimizing LLMs, and the need for principled evaluation metrics. To bridge these gaps, we propose ToolDelete, the first approach for unlearning tools from tool-augmented LLMs. It implements three key properties to address the above challenges for effective tool unlearning and introduces a new membership inference attack (MIA) model for effective evaluation. Extensive experiments on multiple tool learning datasets and tool-augmented LLMs show that ToolDelete effectively unlearns randomly selected tools, while preserving the LLM's knowledge on non-deleted tools and maintaining performance on general tasks.



## **19. SQL Injection Jailbreak: A Structural Disaster of Large Language Models**

cs.CR

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2411.01565v4) [paper-pdf](http://arxiv.org/pdf/2411.01565v4)

**Authors**: Jiawei Zhao, Kejiang Chen, Weiming Zhang, Nenghai Yu

**Abstract**: In recent years, the rapid development of large language models (LLMs) has brought new vitality into various domains, generating substantial social and economic benefits. However, this swift advancement has also introduced new vulnerabilities. Jailbreaking, a form of attack that induces LLMs to produce harmful content through carefully crafted prompts, presents a significant challenge to the safe and trustworthy development of LLMs. Previous jailbreak methods primarily exploited the internal properties or capabilities of LLMs, such as optimization-based jailbreak methods and methods that leveraged the model's context-learning abilities. In this paper, we introduce a novel jailbreak method, SQL Injection Jailbreak (SIJ), which targets the external properties of LLMs, specifically, the way LLMs construct input prompts. By injecting jailbreak information into user prompts, SIJ successfully induces the model to output harmful content. Our SIJ method achieves near 100\% attack success rates on five well-known open-source LLMs on the AdvBench and HEx-PHI, while incurring lower time costs compared to previous methods. For closed-source models, SIJ achieves near 100% attack success rate on GPT-3.5-turbo. Additionally, SIJ exposes a new vulnerability in LLMs that urgently requires mitigation. To address this, we propose a simple defense method called Self-Reminder-Key to counter SIJ and demonstrate its effectiveness through experimental results. Our code is available at https://github.com/weiyezhimeng/SQL-Injection-Jailbreak.



## **20. Refining Adaptive Zeroth-Order Optimization at Ease**

cs.LG

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01014v1) [paper-pdf](http://arxiv.org/pdf/2502.01014v1)

**Authors**: Yao Shu, Qixin Zhang, Kun He, Zhongxiang Dai

**Abstract**: Recently, zeroth-order (ZO) optimization plays an essential role in scenarios where gradient information is inaccessible or unaffordable, such as black-box systems and resource-constrained environments. While existing adaptive methods such as ZO-AdaMM have shown promise, they are fundamentally limited by their underutilization of moment information during optimization, usually resulting in underperforming convergence. To overcome these limitations, this paper introduces Refined Adaptive Zeroth-Order Optimization (R-AdaZO). Specifically, we first show the untapped variance reduction effect of first moment estimate on ZO gradient estimation, which improves the accuracy and stability of ZO updates. We then refine the second moment estimate based on these variance-reduced gradient estimates to better capture the geometry of the optimization landscape, enabling a more effective scaling of ZO updates. We present rigorous theoretical analysis to show (I) the first analysis to the variance reduction of first moment estimate in ZO optimization, (II) the improved second moment estimates with a more accurate approximation of its variance-free ideal, (III) the first variance-aware convergence framework for adaptive ZO methods, which may be of independent interest, and (IV) the faster convergence of R-AdaZO than existing baselines like ZO-AdaMM. Our extensive experiments, including synthetic problems, black-box adversarial attack, and memory-efficient fine-tuning of large language models (LLMs), further verify the superior convergence of R-AdaZO, indicating that R-AdaZO offers an improved solution for real-world ZO optimization challenges.



## **21. Encrypted Large Model Inference: The Equivariant Encryption Paradigm**

cs.CR

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01013v1) [paper-pdf](http://arxiv.org/pdf/2502.01013v1)

**Authors**: James Buban, Hongyang Zhang, Claudio Angione, Harry Yang, Ahmad Farhan, Seyfal Sultanov, Michael Du, Xuran Ma, Zihao Wang, Yue Zhao, Arria Owlia, Fielding Johnston, Patrick Colangelo

**Abstract**: Large scale deep learning model, such as modern language models and diffusion architectures, have revolutionized applications ranging from natural language processing to computer vision. However, their deployment in distributed or decentralized environments raises significant privacy concerns, as sensitive data may be exposed during inference. Traditional techniques like secure multi-party computation, homomorphic encryption, and differential privacy offer partial remedies but often incur substantial computational overhead, latency penalties, or limited compatibility with non-linear network operations. In this work, we introduce Equivariant Encryption (EE), a novel paradigm designed to enable secure, "blind" inference on encrypted data with near zero performance overhead. Unlike fully homomorphic approaches that encrypt the entire computational graph, EE selectively obfuscates critical internal representations within neural network layers while preserving the exact functionality of both linear and a prescribed set of non-linear operations. This targeted encryption ensures that raw inputs, intermediate activations, and outputs remain confidential, even when processed on untrusted infrastructure. We detail the theoretical foundations of EE, compare its performance and integration complexity against conventional privacy preserving techniques, and demonstrate its applicability across a range of architectures, from convolutional networks to large language models. Furthermore, our work provides a comprehensive threat analysis, outlining potential attack vectors and baseline strategies, and benchmarks EE against standard inference pipelines in decentralized settings. The results confirm that EE maintains high fidelity and throughput, effectively bridging the gap between robust data confidentiality and the stringent efficiency requirements of modern, large scale model inference.



## **22. Time-Reversal Provides Unsupervised Feedback to LLMs**

cs.CL

Accepted as a spotlight in NeurIPS 2024

**SubmitDate**: 2025-02-02    [abs](http://arxiv.org/abs/2412.02626v3) [paper-pdf](http://arxiv.org/pdf/2412.02626v3)

**Authors**: Yerram Varun, Rahul Madhavan, Sravanti Addepalli, Arun Suggala, Karthikeyan Shanmugam, Prateek Jain

**Abstract**: Large Language Models (LLMs) are typically trained to predict in the forward direction of time. However, recent works have shown that prompting these models to look back and critique their own generations can produce useful feedback. Motivated by this, we explore the question of whether LLMs can be empowered to think (predict and score) backwards to provide unsupervised feedback that complements forward LLMs. Towards this, we introduce Time Reversed Language Models (TRLMs), which can score and generate queries when conditioned on responses, effectively functioning in the reverse direction of time. Further, to effectively infer in the response to query direction, we pre-train and fine-tune a language model (TRLM-Ba) in the reverse token order from scratch. We show empirically (and theoretically in a stylized setting) that time-reversed models can indeed complement forward model predictions when used to score the query given response for re-ranking multiple forward generations. We obtain up to 5\% improvement on the widely used AlpacaEval Leaderboard over the competent baseline of best-of-N re-ranking using self log-perplexity scores. We further show that TRLM scoring outperforms conventional forward scoring of response given query, resulting in significant gains in applications such as citation generation and passage retrieval. We next leverage the generative ability of TRLM to augment or provide unsupervised feedback to input safety filters of LLMs, demonstrating a drastic reduction in false negative rate with negligible impact on false positive rates against several attacks published on the popular JailbreakBench leaderboard.



## **23. Gandalf the Red: Adaptive Security for LLMs**

cs.LG

Niklas Pfister, V\'aclav Volhejn and Manuel Knott contributed equally

**SubmitDate**: 2025-02-02    [abs](http://arxiv.org/abs/2501.07927v2) [paper-pdf](http://arxiv.org/pdf/2501.07927v2)

**Authors**: Niklas Pfister, Václav Volhejn, Manuel Knott, Santiago Arias, Julia Bazińska, Mykhailo Bichurin, Alan Commike, Janet Darling, Peter Dienes, Matthew Fiedler, David Haber, Matthias Kraft, Marco Lancini, Max Mathys, Damián Pascual-Ortiz, Jakub Podolak, Adrià Romero-López, Kyriacos Shiarlis, Andreas Signer, Zsolt Terek, Athanasios Theocharis, Daniel Timbrell, Samuel Trautwein, Samuel Watts, Yun-Han Wu, Mateo Rojas-Carulla

**Abstract**: Current evaluations of defenses against prompt attacks in large language model (LLM) applications often overlook two critical factors: the dynamic nature of adversarial behavior and the usability penalties imposed on legitimate users by restrictive defenses. We propose D-SEC (Dynamic Security Utility Threat Model), which explicitly separates attackers from legitimate users, models multi-step interactions, and expresses the security-utility in an optimizable form. We further address the shortcomings in existing evaluations by introducing Gandalf, a crowd-sourced, gamified red-teaming platform designed to generate realistic, adaptive attack. Using Gandalf, we collect and release a dataset of 279k prompt attacks. Complemented by benign user data, our analysis reveals the interplay between security and utility, showing that defenses integrated in the LLM (e.g., system prompts) can degrade usability even without blocking requests. We demonstrate that restricted application domains, defense-in-depth, and adaptive defenses are effective strategies for building secure and useful LLM applications.



## **24. From Compliance to Exploitation: Jailbreak Prompt Attacks on Multimodal LLMs**

cs.CR

**SubmitDate**: 2025-02-02    [abs](http://arxiv.org/abs/2502.00735v1) [paper-pdf](http://arxiv.org/pdf/2502.00735v1)

**Authors**: Chun Wai Chiu, Linghan Huang, Bo Li, Huaming Chen

**Abstract**: Large Language Models (LLMs) have seen widespread applications across various domains due to their growing ability to process diverse types of input data, including text, audio, image and video. While LLMs have demonstrated outstanding performance in understanding and generating contexts for different scenarios, they are vulnerable to prompt-based attacks, which are mostly via text input. In this paper, we introduce the first voice-based jailbreak attack against multimodal LLMs, termed as Flanking Attack, which can process different types of input simultaneously towards the multimodal LLMs. Our work is motivated by recent advancements in monolingual voice-driven large language models, which have introduced new attack surfaces beyond traditional text-based vulnerabilities for LLMs. To investigate these risks, we examine the frontier multimodal LLMs, which can be accessed via different types of inputs such as audio input, focusing on how adversarial prompts can bypass its defense mechanisms. We propose a novel strategy, in which the disallowed prompt is flanked by benign, narrative-driven prompts. It is integrated in the Flanking Attack which attempts to humanizes the interaction context and execute the attack through a fictional setting. To better evaluate the attack performance, we present a semi-automated self-assessment framework for policy violation detection. We demonstrate that Flank Attack is capable of manipulating state-of-the-art LLMs into generating misaligned and forbidden outputs, which achieves an average attack success rate ranging from 0.67 to 0.93 across seven forbidden scenarios. These findings highlight both the potency of prompt-based obfuscation in voice-enabled contexts and the limitations of current LLMs' moderation safeguards and the urgent need for advanced defense strategies to address the challenges posed by evolving, context-rich attacks.



## **25. "I am bad": Interpreting Stealthy, Universal and Robust Audio Jailbreaks in Audio-Language Models**

cs.LG

**SubmitDate**: 2025-02-02    [abs](http://arxiv.org/abs/2502.00718v1) [paper-pdf](http://arxiv.org/pdf/2502.00718v1)

**Authors**: Isha Gupta, David Khachaturov, Robert Mullins

**Abstract**: The rise of multimodal large language models has introduced innovative human-machine interaction paradigms but also significant challenges in machine learning safety. Audio-Language Models (ALMs) are especially relevant due to the intuitive nature of spoken communication, yet little is known about their failure modes. This paper explores audio jailbreaks targeting ALMs, focusing on their ability to bypass alignment mechanisms. We construct adversarial perturbations that generalize across prompts, tasks, and even base audio samples, demonstrating the first universal jailbreaks in the audio modality, and show that these remain effective in simulated real-world conditions. Beyond demonstrating attack feasibility, we analyze how ALMs interpret these audio adversarial examples and reveal them to encode imperceptible first-person toxic speech - suggesting that the most effective perturbations for eliciting toxic outputs specifically embed linguistic features within the audio signal. These results have important implications for understanding the interactions between different modalities in multimodal models, and offer actionable insights for enhancing defenses against adversarial audio attacks.



## **26. LLM Safety Alignment is Divergence Estimation in Disguise**

cs.LG

**SubmitDate**: 2025-02-02    [abs](http://arxiv.org/abs/2502.00657v1) [paper-pdf](http://arxiv.org/pdf/2502.00657v1)

**Authors**: Rajdeep Haldar, Ziyi Wang, Qifan Song, Guang Lin, Yue Xing

**Abstract**: We propose a theoretical framework demonstrating that popular Large Language Model (LLM) alignment methods, including Reinforcement Learning from Human Feedback (RLHF) and alternatives, fundamentally function as divergence estimators between aligned (preferred or safe) and unaligned (less-preferred or harmful) distributions. This explains the separation phenomenon between safe and harmful prompts in the model hidden representation after alignment. Inspired by the theoretical results, we identify that some alignment methods are better than others in terms of separation and, introduce a new method, KLDO, and further demonstrate the implication of our theories. We advocate for compliance-refusal datasets over preference datasets to enhance safety alignment, supported by both theoretical reasoning and empirical evidence. Additionally, to quantify safety separation, we leverage a distance metric in the representation space and statistically validate its efficacy as a statistical significant indicator of LLM resilience against jailbreak attacks.



## **27. Towards Robust Multimodal Large Language Models Against Jailbreak Attacks**

cs.CR

**SubmitDate**: 2025-02-02    [abs](http://arxiv.org/abs/2502.00653v1) [paper-pdf](http://arxiv.org/pdf/2502.00653v1)

**Authors**: Ziyi Yin, Yuanpu Cao, Han Liu, Ting Wang, Jinghui Chen, Fenhlong Ma

**Abstract**: While multimodal large language models (MLLMs) have achieved remarkable success in recent advancements, their susceptibility to jailbreak attacks has come to light. In such attacks, adversaries exploit carefully crafted prompts to coerce models into generating harmful or undesirable content. Existing defense mechanisms often rely on external inference steps or safety alignment training, both of which are less effective and impractical when facing sophisticated adversarial perturbations in white-box scenarios. To address these challenges and bolster MLLM robustness, we introduce SafeMLLM by adopting an adversarial training framework that alternates between an attack step for generating adversarial noise and a model updating step. At the attack step, SafeMLLM generates adversarial perturbations through a newly proposed contrastive embedding attack (CoE-Attack), which optimizes token embeddings under a contrastive objective. SafeMLLM then updates model parameters to neutralize the perturbation effects while preserving model utility on benign inputs. We evaluate SafeMLLM across six MLLMs and six jailbreak methods spanning multiple modalities. Experimental results show that SafeMLLM effectively defends against diverse attacks, maintaining robust performance and utilities.



## **28. Self-Instruct Few-Shot Jailbreaking: Decompose the Attack into Pattern and Behavior Learning**

cs.AI

**SubmitDate**: 2025-02-01    [abs](http://arxiv.org/abs/2501.07959v2) [paper-pdf](http://arxiv.org/pdf/2501.07959v2)

**Authors**: Jiaqi Hua, Wanxu Wei

**Abstract**: Recently, several works have been conducted on jailbreaking Large Language Models (LLMs) with few-shot malicious demos. In particular, Zheng et al. focus on improving the efficiency of Few-Shot Jailbreaking (FSJ) by injecting special tokens into the demos and employing demo-level random search, known as Improved Few-Shot Jailbreaking (I-FSJ). Nevertheless, we notice that this method may still require a long context to jailbreak advanced models e.g. 32 shots of demos for Meta-Llama-3-8B-Instruct (Llama-3) \cite{llama3modelcard}. In this paper, we discuss the limitations of I-FSJ and propose Self-Instruct Few-Shot Jailbreaking (Self-Instruct-FSJ) facilitated with the demo-level greedy search. This framework decomposes the FSJ attack into pattern and behavior learning to exploit the model's vulnerabilities in a more generalized and efficient way. We conduct elaborate experiments to evaluate our method on common open-source models and compare it with baseline algorithms. Our code is available at https://github.com/iphosi/Self-Instruct-FSJ.



## **29. Riddle Me This! Stealthy Membership Inference for Retrieval-Augmented Generation**

cs.CR

**SubmitDate**: 2025-02-01    [abs](http://arxiv.org/abs/2502.00306v1) [paper-pdf](http://arxiv.org/pdf/2502.00306v1)

**Authors**: Ali Naseh, Yuefeng Peng, Anshuman Suri, Harsh Chaudhari, Alina Oprea, Amir Houmansadr

**Abstract**: Retrieval-Augmented Generation (RAG) enables Large Language Models (LLMs) to generate grounded responses by leveraging external knowledge databases without altering model parameters. Although the absence of weight tuning prevents leakage via model parameters, it introduces the risk of inference adversaries exploiting retrieved documents in the model's context. Existing methods for membership inference and data extraction often rely on jailbreaking or carefully crafted unnatural queries, which can be easily detected or thwarted with query rewriting techniques common in RAG systems. In this work, we present Interrogation Attack (IA), a membership inference technique targeting documents in the RAG datastore. By crafting natural-text queries that are answerable only with the target document's presence, our approach demonstrates successful inference with just 30 queries while remaining stealthy; straightforward detectors identify adversarial prompts from existing methods up to ~76x more frequently than those generated by our attack. We observe a 2x improvement in TPR@1%FPR over prior inference attacks across diverse RAG configurations, all while costing less than $0.02 per document inference.



## **30. Byzantine-Resilient Zero-Order Optimization for Communication-Efficient Heterogeneous Federated Learning**

cs.LG

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2502.00193v1) [paper-pdf](http://arxiv.org/pdf/2502.00193v1)

**Authors**: Maximilian Egger, Mayank Bakshi, Rawad Bitar

**Abstract**: We introduce CyBeR-0, a Byzantine-resilient federated zero-order optimization method that is robust under Byzantine attacks and provides significant savings in uplink and downlink communication costs. We introduce transformed robust aggregation to give convergence guarantees for general non-convex objectives under client data heterogeneity. Empirical evaluations for standard learning tasks and fine-tuning large language models show that CyBeR-0 exhibits stable performance with only a few scalars per-round communication cost and reduced memory requirements.



## **31. UniGuard: Towards Universal Safety Guardrails for Jailbreak Attacks on Multimodal Large Language Models**

cs.CL

14 pages

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2411.01703v2) [paper-pdf](http://arxiv.org/pdf/2411.01703v2)

**Authors**: Sejoon Oh, Yiqiao Jin, Megha Sharma, Donghyun Kim, Eric Ma, Gaurav Verma, Srijan Kumar

**Abstract**: Multimodal large language models (MLLMs) have revolutionized vision-language understanding but remain vulnerable to multimodal jailbreak attacks, where adversarial inputs are meticulously crafted to elicit harmful or inappropriate responses. We propose UniGuard, a novel multimodal safety guardrail that jointly considers the unimodal and cross-modal harmful signals. UniGuard trains a multimodal guardrail to minimize the likelihood of generating harmful responses in a toxic corpus. The guardrail can be seamlessly applied to any input prompt during inference with minimal computational costs. Extensive experiments demonstrate the generalizability of UniGuard across multiple modalities, attack strategies, and multiple state-of-the-art MLLMs, including LLaVA, Gemini Pro, GPT-4o, MiniGPT-4, and InstructBLIP. Notably, this robust defense mechanism maintains the models' overall vision-language understanding capabilities.



## **32. Enhancing Model Defense Against Jailbreaks with Proactive Safety Reasoning**

cs.CR

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2501.19180v1) [paper-pdf](http://arxiv.org/pdf/2501.19180v1)

**Authors**: Xianglin Yang, Gelei Deng, Jieming Shi, Tianwei Zhang, Jin Song Dong

**Abstract**: Large language models (LLMs) are vital for a wide range of applications yet remain susceptible to jailbreak threats, which could lead to the generation of inappropriate responses. Conventional defenses, such as refusal and adversarial training, often fail to cover corner cases or rare domains, leaving LLMs still vulnerable to more sophisticated attacks. We propose a novel defense strategy, Safety Chain-of-Thought (SCoT), which harnesses the enhanced \textit{reasoning capabilities} of LLMs for proactive assessment of harmful inputs, rather than simply blocking them. SCoT augments any refusal training datasets to critically analyze the intent behind each request before generating answers. By employing proactive reasoning, SCoT enhances the generalization of LLMs across varied harmful queries and scenarios not covered in the safety alignment corpus. Additionally, it generates detailed refusals specifying the rules violated. Comparative evaluations show that SCoT significantly surpasses existing defenses, reducing vulnerability to out-of-distribution issues and adversarial manipulations while maintaining strong general capabilities.



## **33. Targeted Vaccine: Safety Alignment for Large Language Models against Harmful Fine-Tuning via Layer-wise Perturbation**

cs.LG

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2410.09760v3) [paper-pdf](http://arxiv.org/pdf/2410.09760v3)

**Authors**: Guozhi Liu, Weiwei Lin, Tiansheng Huang, Ruichao Mo, Qi Mu, Li Shen

**Abstract**: Harmful fine-tuning attack poses a serious threat to the online fine-tuning service. Vaccine, a recent alignment-stage defense, applies uniform perturbation to all layers of embedding to make the model robust to the simulated embedding drift. However, applying layer-wise uniform perturbation may lead to excess perturbations for some particular safety-irrelevant layers, resulting in defense performance degradation and unnecessary memory consumption. To address this limitation, we propose Targeted Vaccine (T-Vaccine), a memory-efficient safety alignment method that applies perturbation to only selected layers of the model. T-Vaccine follows two core steps: First, it uses gradient norm as a statistical metric to identify the safety-critical layers. Second, instead of applying uniform perturbation across all layers, T-Vaccine only applies perturbation to the safety-critical layers while keeping other layers frozen during training. Results show that T-Vaccine outperforms Vaccine in terms of both defense effectiveness and resource efficiency. Comparison with other defense baselines, e.g., RepNoise and TAR also demonstrate the superiority of T-Vaccine. Notably, T-Vaccine is the first defense that can address harmful fine-tuning issues for a 7B pre-trained models trained on consumer GPUs with limited memory (e.g., RTX 4090). Our code is available at https://github.com/Lslland/T-Vaccine.



## **34. Towards the Worst-case Robustness of Large Language Models**

cs.LG

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2501.19040v1) [paper-pdf](http://arxiv.org/pdf/2501.19040v1)

**Authors**: Huanran Chen, Yinpeng Dong, Zeming Wei, Hang Su, Jun Zhu

**Abstract**: Recent studies have revealed the vulnerability of Large Language Models (LLMs) to adversarial attacks, where the adversary crafts specific input sequences to induce harmful, violent, private, or incorrect outputs. Although various defenses have been proposed, they have not been evaluated by strong adaptive attacks, leaving the worst-case robustness of LLMs still intractable. By developing a stronger white-box attack, our evaluation results indicate that most typical defenses achieve nearly 0\% robustness.To solve this, we propose \textit{DiffTextPure}, a general defense that diffuses the (adversarial) input prompt using any pre-defined smoothing distribution, and purifies the diffused input using a pre-trained language model. Theoretically, we derive tight robustness lower bounds for all smoothing distributions using Fractal Knapsack or 0-1 Knapsack solvers. Under this framework, we certify the robustness of a specific case -- smoothing LLMs using a uniform kernel -- against \textit{any possible attack} with an average $\ell_0$ perturbation of 2.02 or an average suffix length of 6.41.



## **35. Importing Phantoms: Measuring LLM Package Hallucination Vulnerabilities**

cs.LG

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2501.19012v1) [paper-pdf](http://arxiv.org/pdf/2501.19012v1)

**Authors**: Arjun Krishna, Erick Galinkin, Leon Derczynski, Jeffrey Martin

**Abstract**: Large Language Models (LLMs) have become an essential tool in the programmer's toolkit, but their tendency to hallucinate code can be used by malicious actors to introduce vulnerabilities to broad swathes of the software supply chain. In this work, we analyze package hallucination behaviour in LLMs across popular programming languages examining both existing package references and fictional dependencies. By analyzing this package hallucination behaviour we find potential attacks and suggest defensive strategies to defend against these attacks. We discover that package hallucination rate is predicated not only on model choice, but also programming language, model size, and specificity of the coding task request. The Pareto optimality boundary between code generation performance and package hallucination is sparsely populated, suggesting that coding models are not being optimized for secure code. Additionally, we find an inverse correlation between package hallucination rate and the HumanEval coding benchmark, offering a heuristic for evaluating the propensity of a model to hallucinate packages. Our metrics, findings and analyses provide a base for future models, securing AI-assisted software development workflows against package supply chain attacks.



## **36. ASTPrompter: Weakly Supervised Automated Language Model Red-Teaming to Identify Low-Perplexity Toxic Prompts**

cs.CL

10 pages, 7 pages of appendix, 3 tables, 3 figures

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2407.09447v3) [paper-pdf](http://arxiv.org/pdf/2407.09447v3)

**Authors**: Amelia F. Hardy, Houjun Liu, Bernard Lange, Duncan Eddy, Mykel J. Kochenderfer

**Abstract**: Conventional approaches for the automated red-teaming of large language models (LLMs) aim to identify prompts that elicit toxic outputs from a frozen language model (the defender). This often results in the prompting model (the adversary) producing text that is unlikely to arise during autoregression. In response, we propose a reinforcement learning formulation of LLM red-teaming designed to discover prompts that both (1) elicit toxic outputs from a defender and (2) have low perplexity as scored by that defender. These prompts are the most pertinent in a red-teaming setting because the defender generates them with high probability. We solve this formulation with an online and weakly supervised form of Identity Preference Optimization (IPO), attacking models ranging from 137M to 7.8B parameters. Our policy performs competitively, producing prompts that induce defender toxicity at a rate of 2-23 times higher than baseline across model scales. Importantly, these prompts have lower perplexity than both automatically generated and human-written attacks. Furthermore, our method creates black-box attacks with 5.4-14 times increased toxicity. To assess the downstream utility of our method, we use rollouts from our policy as negative examples for downstream toxicity tuning and demonstrate improved safety.



## **37. LLM Cyber Evaluations Don't Capture Real-World Risk**

cs.CR

11 pages

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2502.00072v1) [paper-pdf](http://arxiv.org/pdf/2502.00072v1)

**Authors**: Kamilė Lukošiūtė, Adam Swanda

**Abstract**: Large language models (LLMs) are demonstrating increasing prowess in cybersecurity applications, creating creating inherent risks alongside their potential for strengthening defenses. In this position paper, we argue that current efforts to evaluate risks posed by these capabilities are misaligned with the goal of understanding real-world impact. Evaluating LLM cybersecurity risk requires more than just measuring model capabilities -- it demands a comprehensive risk assessment that incorporates analysis of threat actor adoption behavior and potential for impact. We propose a risk assessment framework for LLM cyber capabilities and apply it to a case study of language models used as cybersecurity assistants. Our evaluation of frontier models reveals high compliance rates but moderate accuracy on realistic cyber assistance tasks. However, our framework suggests that this particular use case presents only moderate risk due to limited operational advantages and impact potential. Based on these findings, we recommend several improvements to align research priorities with real-world impact assessment, including closer academia-industry collaboration, more realistic modeling of attacker behavior, and inclusion of economic metrics in evaluations. This work represents an important step toward more effective assessment and mitigation of LLM-enabled cybersecurity risks.



## **38. Evaluating LLM-based Personal Information Extraction and Countermeasures**

cs.CR

To appear in USENIX Security Symposium 2025

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2408.07291v3) [paper-pdf](http://arxiv.org/pdf/2408.07291v3)

**Authors**: Yupei Liu, Yuqi Jia, Jinyuan Jia, Neil Zhenqiang Gong

**Abstract**: Automatically extracting personal information -- such as name, phone number, and email address -- from publicly available profiles at a large scale is a stepstone to many other security attacks including spear phishing. Traditional methods -- such as regular expression, keyword search, and entity detection -- achieve limited success at such personal information extraction. In this work, we perform a systematic measurement study to benchmark large language model (LLM) based personal information extraction and countermeasures. Towards this goal, we present a framework for LLM-based extraction attacks; collect four datasets including a synthetic dataset generated by GPT-4 and three real-world datasets with manually labeled eight categories of personal information; introduce a novel mitigation strategy based on prompt injection; and systematically benchmark LLM-based attacks and countermeasures using ten LLMs and five datasets. Our key findings include: LLM can be misused by attackers to accurately extract various personal information from personal profiles; LLM outperforms traditional methods; and prompt injection can defend against strong LLM-based attacks, reducing the attack to less effective traditional ones.



## **39. Trading Inference-Time Compute for Adversarial Robustness**

cs.LG

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2501.18841v1) [paper-pdf](http://arxiv.org/pdf/2501.18841v1)

**Authors**: Wojciech Zaremba, Evgenia Nitishinskaya, Boaz Barak, Stephanie Lin, Sam Toyer, Yaodong Yu, Rachel Dias, Eric Wallace, Kai Xiao, Johannes Heidecke, Amelia Glaese

**Abstract**: We conduct experiments on the impact of increasing inference-time compute in reasoning models (specifically OpenAI o1-preview and o1-mini) on their robustness to adversarial attacks. We find that across a variety of attacks, increased inference-time compute leads to improved robustness. In many cases (with important exceptions), the fraction of model samples where the attack succeeds tends to zero as the amount of test-time compute grows. We perform no adversarial training for the tasks we study, and we increase inference-time compute by simply allowing the models to spend more compute on reasoning, independently of the form of attack. Our results suggest that inference-time compute has the potential to improve adversarial robustness for Large Language Models. We also explore new attacks directed at reasoning models, as well as settings where inference-time compute does not improve reliability, and speculate on the reasons for these as well as ways to address them.



## **40. Constitutional Classifiers: Defending against Universal Jailbreaks across Thousands of Hours of Red Teaming**

cs.CL

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2501.18837v1) [paper-pdf](http://arxiv.org/pdf/2501.18837v1)

**Authors**: Mrinank Sharma, Meg Tong, Jesse Mu, Jerry Wei, Jorrit Kruthoff, Scott Goodfriend, Euan Ong, Alwin Peng, Raj Agarwal, Cem Anil, Amanda Askell, Nathan Bailey, Joe Benton, Emma Bluemke, Samuel R. Bowman, Eric Christiansen, Hoagy Cunningham, Andy Dau, Anjali Gopal, Rob Gilson, Logan Graham, Logan Howard, Nimit Kalra, Taesung Lee, Kevin Lin, Peter Lofgren, Francesco Mosconi, Clare O'Hara, Catherine Olsson, Linda Petrini, Samir Rajani, Nikhil Saxena, Alex Silverstein, Tanya Singh, Theodore Sumers, Leonard Tang, Kevin K. Troy, Constantin Weisser, Ruiqi Zhong, Giulio Zhou, Jan Leike, Jared Kaplan, Ethan Perez

**Abstract**: Large language models (LLMs) are vulnerable to universal jailbreaks-prompting strategies that systematically bypass model safeguards and enable users to carry out harmful processes that require many model interactions, like manufacturing illegal substances at scale. To defend against these attacks, we introduce Constitutional Classifiers: safeguards trained on synthetic data, generated by prompting LLMs with natural language rules (i.e., a constitution) specifying permitted and restricted content. In over 3,000 estimated hours of red teaming, no red teamer found a universal jailbreak that could extract information from an early classifier-guarded LLM at a similar level of detail to an unguarded model across most target queries. On automated evaluations, enhanced classifiers demonstrated robust defense against held-out domain-specific jailbreaks. These classifiers also maintain deployment viability, with an absolute 0.38% increase in production-traffic refusals and a 23.7% inference overhead. Our work demonstrates that defending against universal jailbreaks while maintaining practical deployment viability is tractable.



## **41. IsolateGPT: An Execution Isolation Architecture for LLM-Based Agentic Systems**

cs.CR

Accepted by the Network and Distributed System Security (NDSS)  Symposium 2025

**SubmitDate**: 2025-01-30    [abs](http://arxiv.org/abs/2403.04960v2) [paper-pdf](http://arxiv.org/pdf/2403.04960v2)

**Authors**: Yuhao Wu, Franziska Roesner, Tadayoshi Kohno, Ning Zhang, Umar Iqbal

**Abstract**: Large language models (LLMs) extended as systems, such as ChatGPT, have begun supporting third-party applications. These LLM apps leverage the de facto natural language-based automated execution paradigm of LLMs: that is, apps and their interactions are defined in natural language, provided access to user data, and allowed to freely interact with each other and the system. These LLM app ecosystems resemble the settings of earlier computing platforms, where there was insufficient isolation between apps and the system. Because third-party apps may not be trustworthy, and exacerbated by the imprecision of natural language interfaces, the current designs pose security and privacy risks for users. In this paper, we evaluate whether these issues can be addressed through execution isolation and what that isolation might look like in the context of LLM-based systems, where there are arbitrary natural language-based interactions between system components, between LLM and apps, and between apps. To that end, we propose IsolateGPT, a design architecture that demonstrates the feasibility of execution isolation and provides a blueprint for implementing isolation, in LLM-based systems. We evaluate IsolateGPT against a number of attacks and demonstrate that it protects against many security, privacy, and safety issues that exist in non-isolated LLM-based systems, without any loss of functionality. The performance overhead incurred by IsolateGPT to improve security is under 30% for three-quarters of tested queries.



## **42. Exploring Audio Editing Features as User-Centric Privacy Defenses Against Emotion Inference Attacks**

cs.CR

Accepted for presentation(Poster) at PPAI-25: The 6th AAAI Workshop  on Privacy-Preserving Artificial Intelligence

**SubmitDate**: 2025-01-30    [abs](http://arxiv.org/abs/2501.18727v1) [paper-pdf](http://arxiv.org/pdf/2501.18727v1)

**Authors**: Mohd. Farhan Israk Soumik, W. K. M. Mithsara, Abdur R. Shahid, Ahmed Imteaj

**Abstract**: The rapid proliferation of speech-enabled technologies, including virtual assistants, video conferencing platforms, and wearable devices, has raised significant privacy concerns, particularly regarding the inference of sensitive emotional information from audio data. Existing privacy-preserving methods often compromise usability and security, limiting their adoption in practical scenarios. This paper introduces a novel, user-centric approach that leverages familiar audio editing techniques, specifically pitch and tempo manipulation, to protect emotional privacy without sacrificing usability. By analyzing popular audio editing applications on Android and iOS platforms, we identified these features as both widely available and usable. We rigorously evaluated their effectiveness against a threat model, considering adversarial attacks from diverse sources, including Deep Neural Networks (DNNs), Large Language Models (LLMs), and and reversibility testing. Our experiments, conducted on three distinct datasets, demonstrate that pitch and tempo manipulation effectively obfuscates emotional data. Additionally, we explore the design principles for lightweight, on-device implementation to ensure broad applicability across various devices and platforms.



## **43. BounTCHA: A CAPTCHA Utilizing Boundary Identification in AI-extended Videos**

cs.CR

22 pages, 15 figures

**SubmitDate**: 2025-01-30    [abs](http://arxiv.org/abs/2501.18565v1) [paper-pdf](http://arxiv.org/pdf/2501.18565v1)

**Authors**: Lehao Lin, Ke Wang, Maha Abdallah, Wei Cai

**Abstract**: In recent years, the rapid development of artificial intelligence (AI) especially multi-modal Large Language Models (MLLMs), has enabled it to understand text, images, videos, and other multimedia data, allowing AI systems to execute various tasks based on human-provided prompts. However, AI-powered bots have increasingly been able to bypass most existing CAPTCHA systems, posing significant security threats to web applications. This makes the design of new CAPTCHA mechanisms an urgent priority. We observe that humans are highly sensitive to shifts and abrupt changes in videos, while current AI systems still struggle to comprehend and respond to such situations effectively. Based on this observation, we design and implement BounTCHA, a CAPTCHA mechanism that leverages human perception of boundaries in video transitions and disruptions. By utilizing AI's capability to expand original videos with prompts, we introduce unexpected twists and changes to create a pipeline for generating short videos for CAPTCHA purposes. We develop a prototype and conduct experiments to collect data on humans' time biases in boundary identification. This data serves as a basis for distinguishing between human users and bots. Additionally, we perform a detailed security analysis of BounTCHA, demonstrating its resilience against various types of attacks. We hope that BounTCHA will act as a robust defense, safeguarding millions of web applications in the AI-driven era.



## **44. Illusions of Relevance: Using Content Injection Attacks to Deceive Retrievers, Rerankers, and LLM Judges**

cs.IR

**SubmitDate**: 2025-01-30    [abs](http://arxiv.org/abs/2501.18536v1) [paper-pdf](http://arxiv.org/pdf/2501.18536v1)

**Authors**: Manveer Singh Tamber, Jimmy Lin

**Abstract**: Consider a scenario in which a user searches for information, only to encounter texts flooded with misleading or non-relevant content. This scenario exemplifies a simple yet potent vulnerability in neural Information Retrieval (IR) pipelines: content injection attacks. We find that embedding models for retrieval, rerankers, and large language model (LLM) relevance judges are vulnerable to these attacks, in which adversaries insert misleading text into passages to manipulate model judgements. We identify two primary threats: (1) inserting unrelated or harmful content within passages that still appear deceptively "relevant", and (2) inserting entire queries or key query terms into passages to boost their perceived relevance. While the second tactic has been explored in prior research, we present, to our knowledge, the first empirical analysis of the first threat, demonstrating how state-of-the-art models can be easily misled. Our study systematically examines the factors that influence an attack's success, such as the placement of injected content and the balance between relevant and non-relevant material. Additionally, we explore various defense strategies, including adversarial passage classifiers, retriever fine-tuning to discount manipulated content, and prompting LLM judges to adopt a more cautious approach. However, we find that these countermeasures often involve trade-offs, sacrificing effectiveness for attack robustness and sometimes penalizing legitimate documents in the process. Our findings highlight the need for stronger defenses against these evolving adversarial strategies to maintain the trustworthiness of IR systems. We release our code and scripts to facilitate further research.



## **45. Rethinking Bottlenecks in Safety Fine-Tuning of Vision Language Models**

cs.CV

**SubmitDate**: 2025-01-30    [abs](http://arxiv.org/abs/2501.18533v1) [paper-pdf](http://arxiv.org/pdf/2501.18533v1)

**Authors**: Yi Ding, Lijun Li, Bing Cao, Jing Shao

**Abstract**: Large Vision-Language Models (VLMs) have achieved remarkable performance across a wide range of tasks. However, their deployment in safety-critical domains poses significant challenges. Existing safety fine-tuning methods, which focus on textual or multimodal content, fall short in addressing challenging cases or disrupt the balance between helpfulness and harmlessness. Our evaluation highlights a safety reasoning gap: these methods lack safety visual reasoning ability, leading to such bottlenecks. To address this limitation and enhance both visual perception and reasoning in safety-critical contexts, we propose a novel dataset that integrates multi-image inputs with safety Chain-of-Thought (CoT) labels as fine-grained reasoning logic to improve model performance. Specifically, we introduce the Multi-Image Safety (MIS) dataset, an instruction-following dataset tailored for multi-image safety scenarios, consisting of training and test splits. Our experiments demonstrate that fine-tuning InternVL2.5-8B with MIS significantly outperforms both powerful open-source models and API-based models in challenging multi-image tasks requiring safety-related visual reasoning. This approach not only delivers exceptional safety performance but also preserves general capabilities without any trade-offs. Specifically, fine-tuning with MIS increases average accuracy by 0.83% across five general benchmarks and reduces the Attack Success Rate (ASR) on multiple safety benchmarks by a large margin. Data and Models are released under: \href{https://dripnowhy.github.io/MIS/}{\texttt{https://dripnowhy.github.io/MIS/}}



## **46. Differentially Private Steering for Large Language Model Alignment**

cs.CL

ICLR 2025; Code: https://github.com/UKPLab/iclr2025-psa

**SubmitDate**: 2025-01-30    [abs](http://arxiv.org/abs/2501.18532v1) [paper-pdf](http://arxiv.org/pdf/2501.18532v1)

**Authors**: Anmol Goel, Yaxi Hu, Iryna Gurevych, Amartya Sanyal

**Abstract**: Aligning Large Language Models (LLMs) with human values and away from undesirable behaviors (such as hallucination) has become increasingly important. Recently, steering LLMs towards a desired behavior via activation editing has emerged as an effective method to mitigate harmful generations at inference-time. Activation editing modifies LLM representations by preserving information from positive demonstrations (e.g., truthful) and minimising information from negative demonstrations (e.g., hallucinations). When these demonstrations come from a private dataset, the aligned LLM may leak private information contained in those private samples. In this work, we present the first study of aligning LLM behavior with private datasets. Our work proposes the \textit{\underline{P}rivate \underline{S}teering for LLM \underline{A}lignment (PSA)} algorithm to edit LLM activations with differential privacy (DP) guarantees. We conduct extensive experiments on seven different benchmarks with open-source LLMs of different sizes (0.5B to 7B) and model families (LlaMa, Qwen, Mistral and Gemma). Our results show that PSA achieves DP guarantees for LLM alignment with minimal loss in performance, including alignment metrics, open-ended text generation quality, and general-purpose reasoning. We also develop the first Membership Inference Attack (MIA) for evaluating and auditing the empirical privacy for the problem of LLM steering via activation editing. Our attack is tailored for activation editing and relies solely on the generated texts without their associated probabilities. Our experiments support the theoretical guarantees by showing improved guarantees for our \textit{PSA} algorithm compared to several existing non-private techniques.



## **47. xJailbreak: Representation Space Guided Reinforcement Learning for Interpretable LLM Jailbreaking**

cs.CL

**SubmitDate**: 2025-01-30    [abs](http://arxiv.org/abs/2501.16727v2) [paper-pdf](http://arxiv.org/pdf/2501.16727v2)

**Authors**: Sunbowen Lee, Shiwen Ni, Chi Wei, Shuaimin Li, Liyang Fan, Ahmadreza Argha, Hamid Alinejad-Rokny, Ruifeng Xu, Yicheng Gong, Min Yang

**Abstract**: Safety alignment mechanism are essential for preventing large language models (LLMs) from generating harmful information or unethical content. However, cleverly crafted prompts can bypass these safety measures without accessing the model's internal parameters, a phenomenon known as black-box jailbreak. Existing heuristic black-box attack methods, such as genetic algorithms, suffer from limited effectiveness due to their inherent randomness, while recent reinforcement learning (RL) based methods often lack robust and informative reward signals. To address these challenges, we propose a novel black-box jailbreak method leveraging RL, which optimizes prompt generation by analyzing the embedding proximity between benign and malicious prompts. This approach ensures that the rewritten prompts closely align with the intent of the original prompts while enhancing the attack's effectiveness. Furthermore, we introduce a comprehensive jailbreak evaluation framework incorporating keywords, intent matching, and answer validation to provide a more rigorous and holistic assessment of jailbreak success. Experimental results show the superiority of our approach, achieving state-of-the-art (SOTA) performance on several prominent open and closed-source LLMs, including Qwen2.5-7B-Instruct, Llama3.1-8B-Instruct, and GPT-4o-0806. Our method sets a new benchmark in jailbreak attack effectiveness, highlighting potential vulnerabilities in LLMs. The codebase for this work is available at https://github.com/Aegis1863/xJailbreak.



## **48. Exploring Potential Prompt Injection Attacks in Federated Military LLMs and Their Mitigation**

cs.LG

7 pages

**SubmitDate**: 2025-01-30    [abs](http://arxiv.org/abs/2501.18416v1) [paper-pdf](http://arxiv.org/pdf/2501.18416v1)

**Authors**: Youngjoon Lee, Taehyun Park, Yunho Lee, Jinu Gong, Joonhyuk Kang

**Abstract**: Federated Learning (FL) is increasingly being adopted in military collaborations to develop Large Language Models (LLMs) while preserving data sovereignty. However, prompt injection attacks-malicious manipulations of input prompts-pose new threats that may undermine operational security, disrupt decision-making, and erode trust among allies. This perspective paper highlights four potential vulnerabilities in federated military LLMs: secret data leakage, free-rider exploitation, system disruption, and misinformation spread. To address these potential risks, we propose a human-AI collaborative framework that introduces both technical and policy countermeasures. On the technical side, our framework uses red/blue team wargaming and quality assurance to detect and mitigate adversarial behaviors of shared LLM weights. On the policy side, it promotes joint AI-human policy development and verification of security protocols. Our findings will guide future research and emphasize proactive strategies for emerging military contexts.



## **49. Joint Optimization of Prompt Security and System Performance in Edge-Cloud LLM Systems**

cs.CR

**SubmitDate**: 2025-01-30    [abs](http://arxiv.org/abs/2501.18663v1) [paper-pdf](http://arxiv.org/pdf/2501.18663v1)

**Authors**: Haiyang Huang, Tianhui Meng, Weijia Jia

**Abstract**: Large language models (LLMs) have significantly facilitated human life, and prompt engineering has improved the efficiency of these models. However, recent years have witnessed a rise in prompt engineering-empowered attacks, leading to issues such as privacy leaks, increased latency, and system resource wastage. Though safety fine-tuning based methods with Reinforcement Learning from Human Feedback (RLHF) are proposed to align the LLMs, existing security mechanisms fail to cope with fickle prompt attacks, highlighting the necessity of performing security detection on prompts. In this paper, we jointly consider prompt security, service latency, and system resource optimization in Edge-Cloud LLM (EC-LLM) systems under various prompt attacks. To enhance prompt security, a vector-database-enabled lightweight attack detector is proposed. We formalize the problem of joint prompt detection, latency, and resource optimization into a multi-stage dynamic Bayesian game model. The equilibrium strategy is determined by predicting the number of malicious tasks and updating beliefs at each stage through Bayesian updates. The proposed scheme is evaluated on a real implemented EC-LLM system, and the results demonstrate that our approach offers enhanced security, reduces the service latency for benign users, and decreases system resource consumption compared to state-of-the-art algorithms.



## **50. Jailbreaking LLMs' Safeguard with Universal Magic Words for Text Embedding Models**

cs.CL

**SubmitDate**: 2025-01-30    [abs](http://arxiv.org/abs/2501.18280v1) [paper-pdf](http://arxiv.org/pdf/2501.18280v1)

**Authors**: Haoyu Liang, Youran Sun, Yunfeng Cai, Jun Zhu, Bo Zhang

**Abstract**: The security issue of large language models (LLMs) has gained significant attention recently, with various defense mechanisms developed to prevent harmful outputs, among which safeguards based on text embedding models serve as a fundamental defense. Through testing, we discover that the distribution of text embedding model outputs is significantly biased with a large mean. Inspired by this observation, we propose novel efficient methods to search for universal magic words that can attack text embedding models. The universal magic words as suffixes can move the embedding of any text towards the bias direction, therefore manipulate the similarity of any text pair and mislead safeguards. By appending magic words to user prompts and requiring LLMs to end answers with magic words, attackers can jailbreak the safeguard. To eradicate this security risk, we also propose defense mechanisms against such attacks, which can correct the biased distribution of text embeddings in a train-free manner.



