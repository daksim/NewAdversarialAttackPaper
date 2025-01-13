# Latest Large Language Model Attack Papers
**update at 2025-01-13 10:29:36**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. LUMIA: Linear probing for Unimodal and MultiModal Membership Inference Attacks leveraging internal LLM states**

cs.CR

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2411.19876v3) [paper-pdf](http://arxiv.org/pdf/2411.19876v3)

**Authors**: Luis Ibanez-Lissen, Lorena Gonzalez-Manzano, Jose Maria de Fuentes, Nicolas Anciaux, Joaquin Garcia-Alfaro

**Abstract**: Large Language Models (LLMs) are increasingly used in a variety of applications, but concerns around membership inference have grown in parallel. Previous efforts focus on black-to-grey-box models, thus neglecting the potential benefit from internal LLM information. To address this, we propose the use of Linear Probes (LPs) as a method to detect Membership Inference Attacks (MIAs) by examining internal activations of LLMs. Our approach, dubbed LUMIA, applies LPs layer-by-layer to get fine-grained data on the model inner workings. We test this method across several model architectures, sizes and datasets, including unimodal and multimodal tasks. In unimodal MIA, LUMIA achieves an average gain of 15.71 % in Area Under the Curve (AUC) over previous techniques. Remarkably, LUMIA reaches AUC>60% in 65.33% of cases -- an increment of 46.80% against the state of the art. Furthermore, our approach reveals key insights, such as the model layers where MIAs are most detectable. In multimodal models, LPs indicate that visual inputs can significantly contribute to detect MIAs -- AUC>60% is reached in 85.90% of experiments.



## **2. Model Inversion in Split Learning for Personalized LLMs: New Insights from Information Bottleneck Theory**

cs.LG

8 pages

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.05965v1) [paper-pdf](http://arxiv.org/pdf/2501.05965v1)

**Authors**: Yunmeng Shu, Shaofeng Li, Tian Dong, Yan Meng, Haojin Zhu

**Abstract**: Personalized Large Language Models (LLMs) have become increasingly prevalent, showcasing the impressive capabilities of models like GPT-4. This trend has also catalyzed extensive research on deploying LLMs on mobile devices. Feasible approaches for such edge-cloud deployment include using split learning. However, previous research has largely overlooked the privacy leakage associated with intermediate representations transmitted from devices to servers. This work is the first to identify model inversion attacks in the split learning framework for LLMs, emphasizing the necessity of secure defense. For the first time, we introduce mutual information entropy to understand the information propagation of Transformer-based LLMs and assess privacy attack performance for LLM blocks. To address the issue of representations being sparser and containing less information than embeddings, we propose a two-stage attack system in which the first part projects representations into the embedding space, and the second part uses a generative model to recover text from these embeddings. This design breaks down the complexity and achieves attack scores of 38%-75% in various scenarios, with an over 60% improvement over the SOTA. This work comprehensively highlights the potential privacy risks during the deployment of personalized LLMs on the edge side.



## **3. Effective faking of verbal deception detection with target-aligned adversarial attacks**

cs.CL

preprint

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.05962v1) [paper-pdf](http://arxiv.org/pdf/2501.05962v1)

**Authors**: Bennett Kleinberg, Riccardo Loconte, Bruno Verschuere

**Abstract**: Background: Deception detection through analysing language is a promising avenue using both human judgments and automated machine learning judgments. For both forms of credibility assessment, automated adversarial attacks that rewrite deceptive statements to appear truthful pose a serious threat. Methods: We used a dataset of 243 truthful and 262 fabricated autobiographical stories in a deception detection task for humans and machine learning models. A large language model was tasked to rewrite deceptive statements so that they appear truthful. In Study 1, humans who made a deception judgment or used the detailedness heuristic and two machine learning models (a fine-tuned language model and a simple n-gram model) judged original or adversarial modifications of deceptive statements. In Study 2, we manipulated the target alignment of the modifications, i.e. tailoring the attack to whether the statements would be assessed by humans or computer models. Results: When adversarial modifications were aligned with their target, human (d=-0.07 and d=-0.04) and machine judgments (51% accuracy) dropped to the chance level. When the attack was not aligned with the target, both human heuristics judgments (d=0.30 and d=0.36) and machine learning predictions (63-78%) were significantly better than chance. Conclusions: Easily accessible language models can effectively help anyone fake deception detection efforts both by humans and machine learning models. Robustness against adversarial modifications for humans and machines depends on that target alignment. We close with suggestions on advancing deception research with adversarial attack designs.



## **4. Image-based Multimodal Models as Intruders: Transferable Multimodal Attacks on Video-based MLLMs**

cs.CV

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.01042v2) [paper-pdf](http://arxiv.org/pdf/2501.01042v2)

**Authors**: Linhao Huang, Xue Jiang, Zhiqiang Wang, Wentao Mo, Xi Xiao, Bo Han, Yongjie Yin, Feng Zheng

**Abstract**: Video-based multimodal large language models (V-MLLMs) have shown vulnerability to adversarial examples in video-text multimodal tasks. However, the transferability of adversarial videos to unseen models--a common and practical real world scenario--remains unexplored. In this paper, we pioneer an investigation into the transferability of adversarial video samples across V-MLLMs. We find that existing adversarial attack methods face significant limitations when applied in black-box settings for V-MLLMs, which we attribute to the following shortcomings: (1) lacking generalization in perturbing video features, (2) focusing only on sparse key-frames, and (3) failing to integrate multimodal information. To address these limitations and deepen the understanding of V-MLLM vulnerabilities in black-box scenarios, we introduce the Image-to-Video MLLM (I2V-MLLM) attack. In I2V-MLLM, we utilize an image-based multimodal model (IMM) as a surrogate model to craft adversarial video samples. Multimodal interactions and temporal information are integrated to disrupt video representations within the latent space, improving adversarial transferability. In addition, a perturbation propagation technique is introduced to handle different unknown frame sampling strategies. Experimental results demonstrate that our method can generate adversarial examples that exhibit strong transferability across different V-MLLMs on multiple video-text multimodal tasks. Compared to white-box attacks on these models, our black-box attacks (using BLIP-2 as surrogate model) achieve competitive performance, with average attack success rates of 55.48% on MSVD-QA and 58.26% on MSRVTT-QA for VideoQA tasks, respectively. Our code will be released upon acceptance.



## **5. BaThe: Defense against the Jailbreak Attack in Multimodal Large Language Models by Treating Harmful Instruction as Backdoor Trigger**

cs.CR

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2408.09093v2) [paper-pdf](http://arxiv.org/pdf/2408.09093v2)

**Authors**: Yulin Chen, Haoran Li, Yirui Zhang, Zihao Zheng, Yangqiu Song, Bryan Hooi

**Abstract**: Multimodal Large Language Models (MLLMs) have showcased impressive performance in a variety of multimodal tasks. On the other hand, the integration of additional image modality may allow the malicious users to inject harmful content inside the images for jailbreaking. Unlike text-based LLMs, where adversaries need to select discrete tokens to conceal their malicious intent using specific algorithms, the continuous nature of image signals provides a direct opportunity for adversaries to inject harmful intentions. In this work, we propose $\textbf{BaThe}$ ($\textbf{Ba}$ckdoor $\textbf{T}$rigger S$\textbf{h}$i$\textbf{e}$ld), a simple yet effective jailbreak defense mechanism. Our work is motivated by recent research on jailbreak backdoor attack and virtual prompt backdoor attack in generative language models. Jailbreak backdoor attack uses harmful instructions combined with manually crafted strings as triggers to make the backdoored model generate prohibited responses. We assume that harmful instructions can function as triggers, and if we alternatively set rejection responses as the triggered response, the backdoored model then can defend against jailbreak attacks. We achieve this by utilizing virtual rejection prompt, similar to the virtual prompt backdoor attack. We embed the virtual rejection prompt into the soft text embeddings, which we call ``wedge''. Our comprehensive experiments demonstrate that BaThe effectively mitigates various types of jailbreak attacks and is adaptable to defend against unseen attacks, with minimal impact on MLLMs' performance.



## **6. Safeguarding System Prompts for LLMs**

cs.CR

15 pages, 5 figures, 2 tables

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2412.13426v2) [paper-pdf](http://arxiv.org/pdf/2412.13426v2)

**Authors**: Zhifeng Jiang, Zhihua Jin, Guoliang He

**Abstract**: Large language models (LLMs) are increasingly utilized in applications where system prompts, which guide model outputs, play a crucial role. These prompts often contain business logic and sensitive information, making their protection essential. However, adversarial and even regular user queries can exploit LLM vulnerabilities to expose these hidden prompts. To address this issue, we propose PromptKeeper, a robust defense mechanism designed to safeguard system prompts. PromptKeeper tackles two core challenges: reliably detecting prompt leakage and mitigating side-channel vulnerabilities when leakage occurs. By framing detection as a hypothesis-testing problem, PromptKeeper effectively identifies both explicit and subtle leakage. Upon detection, it regenerates responses using a dummy prompt, ensuring that outputs remain indistinguishable from typical interactions when no leakage is present. PromptKeeper ensures robust protection against prompt extraction attacks via either adversarial or regular queries, while preserving conversational capability and runtime efficiency during benign user interactions.



## **7. RAG-WM: An Efficient Black-Box Watermarking Approach for Retrieval-Augmented Generation of Large Language Models**

cs.CR

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2501.05249v1) [paper-pdf](http://arxiv.org/pdf/2501.05249v1)

**Authors**: Peizhuo Lv, Mengjie Sun, Hao Wang, Xiaofeng Wang, Shengzhi Zhang, Yuxuan Chen, Kai Chen, Limin Sun

**Abstract**: In recent years, tremendous success has been witnessed in Retrieval-Augmented Generation (RAG), widely used to enhance Large Language Models (LLMs) in domain-specific, knowledge-intensive, and privacy-sensitive tasks. However, attackers may steal those valuable RAGs and deploy or commercialize them, making it essential to detect Intellectual Property (IP) infringement. Most existing ownership protection solutions, such as watermarks, are designed for relational databases and texts. They cannot be directly applied to RAGs because relational database watermarks require white-box access to detect IP infringement, which is unrealistic for the knowledge base in RAGs. Meanwhile, post-processing by the adversary's deployed LLMs typically destructs text watermark information. To address those problems, we propose a novel black-box "knowledge watermark" approach, named RAG-WM, to detect IP infringement of RAGs. RAG-WM uses a multi-LLM interaction framework, comprising a Watermark Generator, Shadow LLM & RAG, and Watermark Discriminator, to create watermark texts based on watermark entity-relationship tuples and inject them into the target RAG. We evaluate RAG-WM across three domain-specific and two privacy-sensitive tasks on four benchmark LLMs. Experimental results show that RAG-WM effectively detects the stolen RAGs in various deployed LLMs. Furthermore, RAG-WM is robust against paraphrasing, unrelated content removal, knowledge insertion, and knowledge expansion attacks. Lastly, RAG-WM can also evade watermark detection approaches, highlighting its promising application in detecting IP infringement of RAG systems.



## **8. Turning Logic Against Itself : Probing Model Defenses Through Contrastive Questions**

cs.CL

Our code is publicly available at  https://github.com/UKPLab/POATE-attack

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2501.01872v2) [paper-pdf](http://arxiv.org/pdf/2501.01872v2)

**Authors**: Rachneet Sachdeva, Rima Hazra, Iryna Gurevych

**Abstract**: Large language models, despite extensive alignment with human values and ethical principles, remain vulnerable to sophisticated jailbreak attacks that exploit their reasoning abilities. Existing safety measures often detect overt malicious intent but fail to address subtle, reasoning-driven vulnerabilities. In this work, we introduce POATE (Polar Opposite query generation, Adversarial Template construction, and Elaboration), a novel jailbreak technique that harnesses contrastive reasoning to provoke unethical responses. POATE crafts semantically opposing intents and integrates them with adversarial templates, steering models toward harmful outputs with remarkable subtlety. We conduct extensive evaluation across six diverse language model families of varying parameter sizes to demonstrate the robustness of the attack, achieving significantly higher attack success rates (~44%) compared to existing methods. To counter this, we propose Intent-Aware CoT and Reverse Thinking CoT, which decompose queries to detect malicious intent and reason in reverse to evaluate and reject harmful responses. These methods enhance reasoning robustness and strengthen the model's defense against adversarial exploits.



## **9. Trading Devil RL: Backdoor attack via Stock market, Bayesian Optimization and Reinforcement Learning**

cs.LG

End of data poisoning research!: Navier-stokes equations (3D;  update); Reinforcement Learning (RL); HFT (High Frequency Trading); Limit  Order Markets and backdoor attack detection

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2412.17908v2) [paper-pdf](http://arxiv.org/pdf/2412.17908v2)

**Authors**: Orson Mengara

**Abstract**: With the rapid development of generative artificial intelligence, particularly large language models, a number of sub-fields of deep learning have made significant progress and are now very useful in everyday applications. For example, well-known financial institutions simulate a wide range of scenarios for various models created by their research teams using reinforcement learning, both before production and after regular operations. In this work, we propose a backdoor attack that focuses solely on data poisoning. This particular backdoor attack is classified as an attack without prior consideration or trigger, and we name it FinanceLLMsBackRL. Our aim is to examine the potential effects of large language models that use reinforcement learning systems for text production or speech recognition, finance, physics, or the ecosystem of contemporary artificial intelligence models.



## **10. SpaLLM-Guard: Pairing SMS Spam Detection Using Open-source and Commercial LLMs**

cs.CR

17 pages

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2501.04985v1) [paper-pdf](http://arxiv.org/pdf/2501.04985v1)

**Authors**: Muhammad Salman, Muhammad Ikram, Nardine Basta, Mohamed Ali Kaafar

**Abstract**: The increasing threat of SMS spam, driven by evolving adversarial techniques and concept drift, calls for more robust and adaptive detection methods. In this paper, we evaluate the potential of large language models (LLMs), both open-source and commercial, for SMS spam detection, comparing their performance across zero-shot, few-shot, fine-tuning, and chain-of-thought prompting approaches. Using a comprehensive dataset of SMS messages, we assess the spam detection capabilities of prominent LLMs such as GPT-4, DeepSeek, LLAMA-2, and Mixtral. Our findings reveal that while zero-shot learning provides convenience, it is unreliable for effective spam detection. Few-shot learning, particularly with carefully selected examples, improves detection but exhibits variability across models. Fine-tuning emerges as the most effective strategy, with Mixtral achieving 98.6% accuracy and a balanced false positive and false negative rate below 2%, meeting the criteria for robust spam detection. Furthermore, we explore the resilience of these models to adversarial attacks, finding that fine-tuning significantly enhances robustness against both perceptible and imperceptible manipulations. Lastly, we investigate the impact of concept drift and demonstrate that fine-tuned LLMs, especially when combined with few-shot learning, can mitigate its effects, maintaining high performance even on evolving spam datasets. This study highlights the importance of fine-tuning and tailored learning strategies to deploy LLMs effectively for real-world SMS spam detection



## **11. Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency**

cs.CR

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2501.04931v1) [paper-pdf](http://arxiv.org/pdf/2501.04931v1)

**Authors**: Shiji Zhao, Ranjie Duan, Fengxiang Wang, Chi Chen, Caixin Kang, Jialing Tao, YueFeng Chen, Hui Xue, Xingxing Wei

**Abstract**: Multimodal Large Language Models (MLLMs) have achieved impressive performance and have been put into practical use in commercial applications, but they still have potential safety mechanism vulnerabilities. Jailbreak attacks are red teaming methods that aim to bypass safety mechanisms and discover MLLMs' potential risks. Existing MLLMs' jailbreak methods often bypass the model's safety mechanism through complex optimization methods or carefully designed image and text prompts. Despite achieving some progress, they have a low attack success rate on commercial closed-source MLLMs. Unlike previous research, we empirically find that there exists a Shuffle Inconsistency between MLLMs' comprehension ability and safety ability for the shuffled harmful instruction. That is, from the perspective of comprehension ability, MLLMs can understand the shuffled harmful text-image instructions well. However, they can be easily bypassed by the shuffled harmful instructions from the perspective of safety ability, leading to harmful responses. Then we innovatively propose a text-image jailbreak attack named SI-Attack. Specifically, to fully utilize the Shuffle Inconsistency and overcome the shuffle randomness, we apply a query-based black-box optimization method to select the most harmful shuffled inputs based on the feedback of the toxic judge model. A series of experiments show that SI-Attack can improve the attack's performance on three benchmarks. In particular, SI-Attack can obviously improve the attack success rate for commercial MLLMs such as GPT-4o or Claude-3.5-Sonnet.



## **12. Navigating the Designs of Privacy-Preserving Fine-tuning for Large Language Models**

cs.LG

4 pages, 2 figures

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2501.04323v2) [paper-pdf](http://arxiv.org/pdf/2501.04323v2)

**Authors**: Haonan Shi, Tu Ouyang, An Wang

**Abstract**: Instruction tuning has proven effective in enhancing Large Language Models' (LLMs) performance on downstream tasks. However, real-world fine-tuning faces inherent conflicts between model providers' intellectual property protection, clients' data privacy requirements, and tuning costs. While recent approaches like split learning and offsite tuning demonstrate promising architectures for privacy-preserving fine-tuning, there is a gap in systematically addressing the multidimensional trade-offs required for diverse real-world deployments. We propose several indicative evaluation metrics to guide design trade-offs for privacy-preserving fine-tuning and a series of example designs, collectively named GuardedTuning; they result from novel combinations of system architectures with adapted privacy-enhancement methods and emerging computation techniques. Each design represents distinct trade-offs across model utility, privacy guarantees, and costs. Experimental results demonstrate that these designs protect against data reconstruction attacks while maintaining competitive fine-tuning performance.



## **13. Watch Out for Your Guidance on Generation! Exploring Conditional Backdoor Attacks against Large Language Models**

cs.CL

The paper has been accepted to AAAI 2025

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2404.14795v5) [paper-pdf](http://arxiv.org/pdf/2404.14795v5)

**Authors**: Jiaming He, Wenbo Jiang, Guanyu Hou, Wenshu Fan, Rui Zhang, Hongwei Li

**Abstract**: Mainstream backdoor attacks on large language models (LLMs) typically set a fixed trigger in the input instance and specific responses for triggered queries. However, the fixed trigger setting (e.g., unusual words) may be easily detected by human detection, limiting the effectiveness and practicality in real-world scenarios. To enhance the stealthiness of backdoor activation, we present a new poisoning paradigm against LLMs triggered by specifying generation conditions, which are commonly adopted strategies by users during model inference. The poisoned model performs normally for output under normal/other generation conditions, while becomes harmful for output under target generation conditions. To achieve this objective, we introduce BrieFool, an efficient attack framework. It leverages the characteristics of generation conditions by efficient instruction sampling and poisoning data generation, thereby influencing the behavior of LLMs under target conditions. Our attack can be generally divided into two types with different targets: Safety unalignment attack and Ability degradation attack. Our extensive experiments demonstrate that BrieFool is effective across safety domains and ability domains, achieving higher success rates than baseline methods, with 94.3 % on GPT-3.5-turbo



## **14. Not all tokens are created equal: Perplexity Attention Weighted Networks for AI generated text detection**

cs.CL

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2501.03940v1) [paper-pdf](http://arxiv.org/pdf/2501.03940v1)

**Authors**: Pablo Miralles-González, Javier Huertas-Tato, Alejandro Martín, David Camacho

**Abstract**: The rapid advancement in large language models (LLMs) has significantly enhanced their ability to generate coherent and contextually relevant text, raising concerns about the misuse of AI-generated content and making it critical to detect it. However, the task remains challenging, particularly in unseen domains or with unfamiliar LLMs. Leveraging LLM next-token distribution outputs offers a theoretically appealing approach for detection, as they encapsulate insights from the models' extensive pre-training on diverse corpora. Despite its promise, zero-shot methods that attempt to operationalize these outputs have met with limited success. We hypothesize that one of the problems is that they use the mean to aggregate next-token distribution metrics across tokens, when some tokens are naturally easier or harder to predict and should be weighted differently. Based on this idea, we propose the Perplexity Attention Weighted Network (PAWN), which uses the last hidden states of the LLM and positions to weight the sum of a series of features based on metrics from the next-token distribution across the sequence length. Although not zero-shot, our method allows us to cache the last hidden states and next-token distribution metrics on disk, greatly reducing the training resource requirements. PAWN shows competitive and even better performance in-distribution than the strongest baselines (fine-tuned LMs) with a fraction of their trainable parameters. Our model also generalizes better to unseen domains and source models, with smaller variability in the decision boundary across distribution shifts. It is also more robust to adversarial attacks, and if the backbone has multilingual capabilities, it presents decent generalization to languages not seen during supervised training, with LLaMA3-1B reaching a mean macro-averaged F1 score of 81.46% in cross-validation with nine languages.



## **15. PSA-VLM: Enhancing Vision-Language Model Safety through Progressive Concept-Bottleneck-Driven Alignment**

cs.CV

arXiv admin note: substantial text overlap with arXiv:2405.13581

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2411.11543v3) [paper-pdf](http://arxiv.org/pdf/2411.11543v3)

**Authors**: Zhendong Liu, Yuanbi Nie, Yingshui Tan, Jiaheng Liu, Xiangyu Yue, Qiushi Cui, Chongjun Wang, Xiaoyong Zhu, Bo Zheng

**Abstract**: Benefiting from the powerful capabilities of Large Language Models (LLMs), pre-trained visual encoder models connected to LLMs form Vision Language Models (VLMs). However, recent research shows that the visual modality in VLMs is highly vulnerable, allowing attackers to bypass safety alignment in LLMs through visually transmitted content, launching harmful attacks. To address this challenge, we propose a progressive concept-based alignment strategy, PSA-VLM, which incorporates safety modules as concept bottlenecks to enhance visual modality safety alignment. By aligning model predictions with specific safety concepts, we improve defenses against risky images, enhancing explainability and controllability while minimally impacting general performance. Our method is obtained through two-stage training. The low computational cost of the first stage brings very effective performance improvement, and the fine-tuning of the language model in the second stage further improves the safety performance. Our method achieves state-of-the-art results on popular VLM safety benchmark.



## **16. PhishAgent: A Robust Multimodal Agent for Phishing Webpage Detection**

cs.CR

Accepted at AAAI 2025

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2408.10738v2) [paper-pdf](http://arxiv.org/pdf/2408.10738v2)

**Authors**: Tri Cao, Chengyu Huang, Yuexin Li, Huilin Wang, Amy He, Nay Oo, Bryan Hooi

**Abstract**: Phishing attacks are a major threat to online security, exploiting user vulnerabilities to steal sensitive information. Various methods have been developed to counteract phishing, each with varying levels of accuracy, but they also face notable limitations. In this study, we introduce PhishAgent, a multimodal agent that combines a wide range of tools, integrating both online and offline knowledge bases with Multimodal Large Language Models (MLLMs). This combination leads to broader brand coverage, which enhances brand recognition and recall. Furthermore, we propose a multimodal information retrieval framework designed to extract the relevant top k items from offline knowledge bases, using available information from a webpage, including logos and HTML. Our empirical results, based on three real-world datasets, demonstrate that the proposed framework significantly enhances detection accuracy and reduces both false positives and false negatives, while maintaining model efficiency. Additionally, PhishAgent shows strong resilience against various types of adversarial attacks.



## **17. MRJ-Agent: An Effective Jailbreak Agent for Multi-Round Dialogue**

cs.AI

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2411.03814v2) [paper-pdf](http://arxiv.org/pdf/2411.03814v2)

**Authors**: Fengxiang Wang, Ranjie Duan, Peng Xiao, Xiaojun Jia, Shiji Zhao, Cheng Wei, YueFeng Chen, Chongwen Wang, Jialing Tao, Hang Su, Jun Zhu, Hui Xue

**Abstract**: Large Language Models (LLMs) demonstrate outstanding performance in their reservoir of knowledge and understanding capabilities, but they have also been shown to be prone to illegal or unethical reactions when subjected to jailbreak attacks. To ensure their responsible deployment in critical applications, it is crucial to understand the safety capabilities and vulnerabilities of LLMs. Previous works mainly focus on jailbreak in single-round dialogue, overlooking the potential jailbreak risks in multi-round dialogues, which are a vital way humans interact with and extract information from LLMs. Some studies have increasingly concentrated on the risks associated with jailbreak in multi-round dialogues. These efforts typically involve the use of manually crafted templates or prompt engineering techniques. However, due to the inherent complexity of multi-round dialogues, their jailbreak performance is limited. To solve this problem, we propose a novel multi-round dialogue jailbreaking agent, emphasizing the importance of stealthiness in identifying and mitigating potential threats to human values posed by LLMs. We propose a risk decomposition strategy that distributes risks across multiple rounds of queries and utilizes psychological strategies to enhance attack strength. Extensive experiments show that our proposed method surpasses other attack methods and achieves state-of-the-art attack success rate. We will make the corresponding code and dataset available for future research. The code will be released soon.



## **18. Practical Secure Inference Algorithm for Fine-tuned Large Language Model Based on Fully Homomorphic Encryption**

cs.CR

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2501.01672v2) [paper-pdf](http://arxiv.org/pdf/2501.01672v2)

**Authors**: Zhang Ruoyan, Zheng Zhongxiang, Bao Wankang

**Abstract**: Large language models(LLMs) are currently at the forefront of the machine learning field, which show a broad application prospect but at the same time expose some risks of privacy leakage. We combined Fully Homomorphic Encryption(FHE) and provable security theory with Parameter-Efficient Fine-Tuning(PEFT) to propose an efficient and secure inference scheme for LLMs. More specially, we focus on pre-trained LLMs which rely on open-sourced base model and then fine-tuned with the private datasets by LoRA. This is a popular road-map for Vertical Domain Models such as LawGPT and BenTsao. We use two key technologies below. Firstly, we divide the whole model into the public part and the private part. The weights of public part are publicly accessible(e.g. the open-sourced base model) while the private part needs to be protected(e.g. the LoRA matrices). In this way, the overhead brought by computing on private data can be greatly reduced. Secondly, we propose a general method to transform a linear layer into another one which provides security against model extraction attacks and preserves its original functionality, which denoted as Private Linear Layer(PLL). Then we use this method on the LoRA matrices to make sure that the server protects their private weights without restricting the user's input. We also show that the difficulty of performing model extraction attacks for PLL can be reduced to the well-known hard problem Learning with Errors(LWE). Combing this method with FHE, we can protect user's input at the same time. In this paper, we use the open-source model ChatGLM2-6B as the base model which is fine-tuned by LoRA. Experimental results show the inference efficiency of our scheme reaches 1.61s/token which displays that the scheme has good practicality.



## **19. ChatBug: A Common Vulnerability of Aligned LLMs Induced by Chat Templates**

cs.CR

This paper is accepted to AAAI 2025

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2406.12935v2) [paper-pdf](http://arxiv.org/pdf/2406.12935v2)

**Authors**: Fengqing Jiang, Zhangchen Xu, Luyao Niu, Bill Yuchen Lin, Radha Poovendran

**Abstract**: Large language models (LLMs) are expected to follow instructions from users and engage in conversations. Techniques to enhance LLMs' instruction-following capabilities typically fine-tune them using data structured according to a predefined chat template. Although chat templates are shown to be effective in optimizing LLM performance, their impact on safety alignment of LLMs has been less understood, which is crucial for deploying LLMs safely at scale.   In this paper, we investigate how chat templates affect safety alignment of LLMs. We identify a common vulnerability, named ChatBug, that is introduced by chat templates. Our key insight to identify ChatBug is that the chat templates provide a rigid format that need to be followed by LLMs, but not by users. Hence, a malicious user may not necessarily follow the chat template when prompting LLMs. Instead, malicious users could leverage their knowledge of the chat template and accordingly craft their prompts to bypass safety alignments of LLMs. We develop two attacks to exploit the ChatBug vulnerability. We demonstrate that a malicious user can exploit the ChatBug vulnerability of eight state-of-the-art (SOTA) LLMs and effectively elicit unintended responses from these models. Moreover, we show that ChatBug can be exploited by existing jailbreak attacks to enhance their attack success rates. We investigate potential countermeasures to ChatBug. Our results show that while adversarial training effectively mitigates the ChatBug vulnerability, the victim model incurs significant performance degradation. These results highlight the trade-off between safety alignment and helpfulness. Developing new methods for instruction tuning to balance this trade-off is an open and critical direction for future research



## **20. HuRef: HUman-REadable Fingerprint for Large Language Models**

cs.CL

NeurIPS 2024

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2312.04828v5) [paper-pdf](http://arxiv.org/pdf/2312.04828v5)

**Authors**: Boyi Zeng, Lizheng Wang, Yuncong Hu, Yi Xu, Chenghu Zhou, Xinbing Wang, Yu Yu, Zhouhan Lin

**Abstract**: Protecting the copyright of large language models (LLMs) has become crucial due to their resource-intensive training and accompanying carefully designed licenses. However, identifying the original base model of an LLM is challenging due to potential parameter alterations. In this study, we introduce HuRef, a human-readable fingerprint for LLMs that uniquely identifies the base model without interfering with training or exposing model parameters to the public. We first observe that the vector direction of LLM parameters remains stable after the model has converged during pretraining, with negligible perturbations through subsequent training steps, including continued pretraining, supervised fine-tuning, and RLHF, which makes it a sufficient condition to identify the base model. The necessity is validated by continuing to train an LLM with an extra term to drive away the model parameters' direction and the model becomes damaged. However, this direction is vulnerable to simple attacks like dimension permutation or matrix rotation, which significantly change it without affecting performance. To address this, leveraging the Transformer structure, we systematically analyze potential attacks and define three invariant terms that identify an LLM's base model. Due to the potential risk of information leakage, we cannot publish invariant terms directly. Instead, we map them to a Gaussian vector using an encoder, then convert it into a natural image using StyleGAN2, and finally publish the image. In our black-box setting, all fingerprinting steps are internally conducted by the LLMs owners. To ensure the published fingerprints are honestly generated, we introduced Zero-Knowledge Proof (ZKP). Experimental results across various LLMs demonstrate the effectiveness of our method. The code is available at https://github.com/LUMIA-Group/HuRef.



## **21. Adversarial Vulnerabilities in Large Language Models for Time Series Forecasting**

cs.LG

11 pages, 5 figures

**SubmitDate**: 2025-01-06    [abs](http://arxiv.org/abs/2412.08099v2) [paper-pdf](http://arxiv.org/pdf/2412.08099v2)

**Authors**: Fuqiang Liu, Sicong Jiang, Luis Miranda-Moreno, Seongjin Choi, Lijun Sun

**Abstract**: Large Language Models (LLMs) have recently demonstrated significant potential in the field of time series forecasting, offering impressive capabilities in handling complex temporal data. However, their robustness and reliability in real-world applications remain under-explored, particularly concerning their susceptibility to adversarial attacks. In this paper, we introduce a targeted adversarial attack framework for LLM-based time series forecasting. By employing both gradient-free and black-box optimization methods, we generate minimal yet highly effective perturbations that significantly degrade the forecasting accuracy across multiple datasets and LLM architectures. Our experiments, which include models like TimeGPT and LLM-Time with GPT-3.5, GPT-4, LLaMa, and Mistral, show that adversarial attacks lead to much more severe performance degradation than random noise, and demonstrate the broad effectiveness of our attacks across different LLMs. The results underscore the critical vulnerabilities of LLMs in time series forecasting, highlighting the need for robust defense mechanisms to ensure their reliable deployment in practical applications.



## **22. Pathway to Secure and Trustworthy ZSM for LLMs: Attacks, Defense, and Opportunities**

cs.CR

7 pages, 4 figures

**SubmitDate**: 2025-01-06    [abs](http://arxiv.org/abs/2408.00722v2) [paper-pdf](http://arxiv.org/pdf/2408.00722v2)

**Authors**: Sunder Ali Khowaja, Parus Khuwaja, Kapal Dev, Hussam Al Hamadi, Engin Zeydan

**Abstract**: Recently, large language models (LLMs) have been gaining a lot of interest due to their adaptability and extensibility in emerging applications, including communication networks. It is anticipated that ZSM networks will be able to support LLMs as a service, as they provide ultra reliable low-latency communications and closed loop massive connectivity. However, LLMs are vulnerable to data and model privacy issues that affect the trustworthiness of LLMs to be deployed for user-based services. In this paper, we explore the security vulnerabilities associated with fine-tuning LLMs in ZSM networks, in particular the membership inference attack. We define the characteristics of an attack network that can perform a membership inference attack if the attacker has access to the fine-tuned model for the downstream task. We show that the membership inference attacks are effective for any downstream task, which can lead to a personal data breach when using LLM as a service. The experimental results show that the attack success rate of maximum 92% can be achieved on named entity recognition task. Based on the experimental analysis, we discuss possible defense mechanisms and present possible research directions to make the LLMs more trustworthy in the context of ZSM networks.



## **23. FlipedRAG: Black-Box Opinion Manipulation Attacks to Retrieval-Augmented Generation of Large Language Models**

cs.IR

arXiv admin note: text overlap with arXiv:2407.13757

**SubmitDate**: 2025-01-06    [abs](http://arxiv.org/abs/2501.02968v1) [paper-pdf](http://arxiv.org/pdf/2501.02968v1)

**Authors**: Zhuo Chen, Yuyang Gong, Miaokun Chen, Haotan Liu, Qikai Cheng, Fan Zhang, Wei Lu, Xiaozhong Liu, Jiawei Liu

**Abstract**: Retrieval-Augmented Generation (RAG) addresses hallucination and real-time constraints by dynamically retrieving relevant information from a knowledge database to supplement the LLMs' input. When presented with a query, RAG selects the most semantically similar texts from its knowledge bases and uses them as context for the LLMs to generate more accurate responses. RAG also creates a new attack surface, especially since RAG databases are frequently sourced from public domains. While existing studies have predominantly focused on optimizing RAG's performance and efficiency, emerging research has begun addressing the security concerns associated with RAG. However, these works have some limitations, typically focusing on either white-box methodologies or heuristic-based black-box attacks. Furthermore, prior research has mainly targeted simple factoid question answering, which is neither practically challenging nor resistant to correction. In this paper, we unveil a more realistic and threatening scenario: opinion manipulation for controversial topics against RAG. Particularly, we propose a novel RAG black-box attack method, termed FlipedRAG, which is transfer-based. By leveraging instruction engineering, we obtain partial retrieval model outputs from black-box RAG system, facilitating the training of surrogate models to enhance the effectiveness of opinion manipulation attack. Extensive experimental results confirms that our approach significantly enhances the average success rate of opinion manipulation by 16.7%. It achieves an average of a 50% directional change in the opinion polarity of RAG responses across four themes. Additionally, it induces a 20% shift in user cognition. Furthermore, we discuss the efficacy of potential defense mechanisms and conclude that they are insufficient in mitigating this type of attack, highlighting the urgent need to develop novel defensive strategies.



## **24. LlamaPartialSpoof: An LLM-Driven Fake Speech Dataset Simulating Disinformation Generation**

eess.AS

5 pages, ICASSP 2025

**SubmitDate**: 2025-01-06    [abs](http://arxiv.org/abs/2409.14743v2) [paper-pdf](http://arxiv.org/pdf/2409.14743v2)

**Authors**: Hieu-Thi Luong, Haoyang Li, Lin Zhang, Kong Aik Lee, Eng Siong Chng

**Abstract**: Previous fake speech datasets were constructed from a defender's perspective to develop countermeasure (CM) systems without considering diverse motivations of attackers. To better align with real-life scenarios, we created LlamaPartialSpoof, a 130-hour dataset that contains both fully and partially fake speech, using a large language model (LLM) and voice cloning technologies to evaluate the robustness of CMs. By examining valuable information for both attackers and defenders, we identify several key vulnerabilities in current CM systems, which can be exploited to enhance attack success rates, including biases toward certain text-to-speech models or concatenation methods. Our experimental results indicate that the current fake speech detection system struggle to generalize to unseen scenarios, achieving a best performance of 24.49% equal error rate.



## **25. Layer-Level Self-Exposure and Patch: Affirmative Token Mitigation for Jailbreak Attack Defense**

cs.CR

**SubmitDate**: 2025-01-05    [abs](http://arxiv.org/abs/2501.02629v1) [paper-pdf](http://arxiv.org/pdf/2501.02629v1)

**Authors**: Yang Ouyang, Hengrui Gu, Shuhang Lin, Wenyue Hua, Jie Peng, Bhavya Kailkhura, Tianlong Chen, Kaixiong Zhou

**Abstract**: As large language models (LLMs) are increasingly deployed in diverse applications, including chatbot assistants and code generation, aligning their behavior with safety and ethical standards has become paramount. However, jailbreak attacks, which exploit vulnerabilities to elicit unintended or harmful outputs, threaten LLMs' safety significantly. In this paper, we introduce Layer-AdvPatcher, a novel methodology designed to defend against jailbreak attacks by utilizing an unlearning strategy to patch specific layers within LLMs through self-augmented datasets. Our insight is that certain layer(s), tend to produce affirmative tokens when faced with harmful prompts. By identifying these layers and adversarially exposing them to generate more harmful data, one can understand their inherent and diverse vulnerabilities to attacks. With these exposures, we then "unlearn" these issues, reducing the impact of affirmative tokens and hence minimizing jailbreak risks while keeping the model's responses to safe queries intact. We conduct extensive experiments on two models, four benchmark datasets, and multiple state-of-the-art jailbreak benchmarks to demonstrate the efficacy of our approach. Results indicate that our framework reduces the harmfulness and attack success rate of jailbreak attacks without compromising utility for benign queries compared to recent defense methods.



## **26. DiffusionAttacker: Diffusion-Driven Prompt Manipulation for LLM Jailbreak**

cs.CL

**SubmitDate**: 2025-01-05    [abs](http://arxiv.org/abs/2412.17522v2) [paper-pdf](http://arxiv.org/pdf/2412.17522v2)

**Authors**: Hao Wang, Hao Li, Junda Zhu, Xinyuan Wang, Chengwei Pan, MinLie Huang, Lei Sha

**Abstract**: Large Language Models (LLMs) are susceptible to generating harmful content when prompted with carefully crafted inputs, a vulnerability known as LLM jailbreaking. As LLMs become more powerful, studying jailbreak methods is critical to enhancing security and aligning models with human values. Traditionally, jailbreak techniques have relied on suffix addition or prompt templates, but these methods suffer from limited attack diversity. This paper introduces DiffusionAttacker, an end-to-end generative approach for jailbreak rewriting inspired by diffusion models. Our method employs a sequence-to-sequence (seq2seq) text diffusion model as a generator, conditioning on the original prompt and guiding the denoising process with a novel attack loss. Unlike previous approaches that use autoregressive LLMs to generate jailbreak prompts, which limit the modification of already generated tokens and restrict the rewriting space, DiffusionAttacker utilizes a seq2seq diffusion model, allowing more flexible token modifications. This approach preserves the semantic content of the original prompt while producing harmful content. Additionally, we leverage the Gumbel-Softmax technique to make the sampling process from the diffusion model's output distribution differentiable, eliminating the need for iterative token search. Extensive experiments on Advbench and Harmbench demonstrate that DiffusionAttacker outperforms previous methods across various evaluation metrics, including attack success rate (ASR), fluency, and diversity.



## **27. A Practical Examination of AI-Generated Text Detectors for Large Language Models**

cs.CL

8 pages. Submitted to NAACL

**SubmitDate**: 2025-01-04    [abs](http://arxiv.org/abs/2412.05139v2) [paper-pdf](http://arxiv.org/pdf/2412.05139v2)

**Authors**: Brian Tufts, Xuandong Zhao, Lei Li

**Abstract**: The proliferation of large language models has raised growing concerns about their misuse, particularly in cases where AI-generated text is falsely attributed to human authors. Machine-generated content detectors claim to effectively identify such text under various conditions and from any language model. This paper critically evaluates these claims by assessing several popular detectors (RADAR, Wild, T5Sentinel, Fast-DetectGPT, GPTID, LogRank, Binoculars) on a range of domains, datasets, and models that these detectors have not previously encountered. We employ various prompting strategies to simulate adversarial attacks, demonstrating that even moderate efforts can significantly evade detection. We emphasize the importance of the true positive rate at a specific false positive rate (TPR@FPR) metric and demonstrate that these detectors perform poorly in certain settings, with TPR@.01 as low as 0%. Our findings suggest that both trained and zero-shot detectors struggle to maintain high sensitivity while achieving a reasonable true positive rate.



## **28. A Survey of Recent Backdoor Attacks and Defenses in Large Language Models**

cs.CR

Accepted in TMLR

**SubmitDate**: 2025-01-04    [abs](http://arxiv.org/abs/2406.06852v5) [paper-pdf](http://arxiv.org/pdf/2406.06852v5)

**Authors**: Shuai Zhao, Meihuizi Jia, Zhongliang Guo, Leilei Gan, Xiaoyu Xu, Xiaobao Wu, Jie Fu, Yichao Feng, Fengjun Pan, Luu Anh Tuan

**Abstract**: Large Language Models (LLMs), which bridge the gap between human language understanding and complex problem-solving, achieve state-of-the-art performance on several NLP tasks, particularly in few-shot and zero-shot settings. Despite the demonstrable efficacy of LLMs, due to constraints on computational resources, users have to engage with open-source language models or outsource the entire training process to third-party platforms. However, research has demonstrated that language models are susceptible to potential security vulnerabilities, particularly in backdoor attacks. Backdoor attacks are designed to introduce targeted vulnerabilities into language models by poisoning training samples or model weights, allowing attackers to manipulate model responses through malicious triggers. While existing surveys on backdoor attacks provide a comprehensive overview, they lack an in-depth examination of backdoor attacks specifically targeting LLMs. To bridge this gap and grasp the latest trends in the field, this paper presents a novel perspective on backdoor attacks for LLMs by focusing on fine-tuning methods. Specifically, we systematically classify backdoor attacks into three categories: full-parameter fine-tuning, parameter-efficient fine-tuning, and no fine-tuning Based on insights from a substantial review, we also discuss crucial issues for future research on backdoor attacks, such as further exploring attack algorithms that do not require fine-tuning, or developing more covert attack algorithms.



## **29. AVTrustBench: Assessing and Enhancing Reliability and Robustness in Audio-Visual LLMs**

cs.CV

**SubmitDate**: 2025-01-03    [abs](http://arxiv.org/abs/2501.02135v1) [paper-pdf](http://arxiv.org/pdf/2501.02135v1)

**Authors**: Sanjoy Chowdhury, Sayan Nag, Subhrajyoti Dasgupta, Yaoting Wang, Mohamed Elhoseiny, Ruohan Gao, Dinesh Manocha

**Abstract**: With the rapid advancement of Multi-modal Large Language Models (MLLMs), several diagnostic benchmarks have recently been developed to assess these models' multi-modal reasoning proficiency. However, these benchmarks are restricted to assessing primarily the visual aspect and do not examine the holistic audio-visual (AV) understanding. Moreover, currently, there are no benchmarks that investigate the capabilities of AVLLMs to calibrate their responses when presented with perturbed inputs. To this end, we introduce Audio-Visual Trustworthiness assessment Benchmark (AVTrustBench), comprising 600K samples spanning over 9 meticulously crafted tasks, evaluating the capabilities of AVLLMs across three distinct dimensions: Adversarial attack, Compositional reasoning, and Modality-specific dependency. Using our benchmark we extensively evaluate 13 state-of-the-art AVLLMs. The findings reveal that the majority of existing models fall significantly short of achieving human-like comprehension, offering valuable insights for future research directions. To alleviate the limitations in the existing approaches, we further propose a robust, model-agnostic calibrated audio-visual preference optimization based training strategy CAVPref, obtaining a gain up to 30.19% across all 9 tasks. We will publicly release our code and benchmark to facilitate future research in this direction.



## **30. Auto-RT: Automatic Jailbreak Strategy Exploration for Red-Teaming Large Language Models**

cs.CR

**SubmitDate**: 2025-01-03    [abs](http://arxiv.org/abs/2501.01830v1) [paper-pdf](http://arxiv.org/pdf/2501.01830v1)

**Authors**: Yanjiang Liu, Shuhen Zhou, Yaojie Lu, Huijia Zhu, Weiqiang Wang, Hongyu Lin, Ben He, Xianpei Han, Le Sun

**Abstract**: Automated red-teaming has become a crucial approach for uncovering vulnerabilities in large language models (LLMs). However, most existing methods focus on isolated safety flaws, limiting their ability to adapt to dynamic defenses and uncover complex vulnerabilities efficiently. To address this challenge, we propose Auto-RT, a reinforcement learning framework that automatically explores and optimizes complex attack strategies to effectively uncover security vulnerabilities through malicious queries. Specifically, we introduce two key mechanisms to reduce exploration complexity and improve strategy optimization: 1) Early-terminated Exploration, which accelerate exploration by focusing on high-potential attack strategies; and 2) Progressive Reward Tracking algorithm with intermediate downgrade models, which dynamically refine the search trajectory toward successful vulnerability exploitation. Extensive experiments across diverse LLMs demonstrate that, by significantly improving exploration efficiency and automatically optimizing attack strategies, Auto-RT detects a boarder range of vulnerabilities, achieving a faster detection speed and 16.63\% higher success rates compared to existing methods.



## **31. How Toxic Can You Get? Search-based Toxicity Testing for Large Language Models**

cs.SE

**SubmitDate**: 2025-01-03    [abs](http://arxiv.org/abs/2501.01741v1) [paper-pdf](http://arxiv.org/pdf/2501.01741v1)

**Authors**: Simone Corbo, Luca Bancale, Valeria De Gennaro, Livia Lestingi, Vincenzo Scotti, Matteo Camilli

**Abstract**: Language is a deep-rooted means of perpetration of stereotypes and discrimination. Large Language Models (LLMs), now a pervasive technology in our everyday lives, can cause extensive harm when prone to generating toxic responses. The standard way to address this issue is to align the LLM, which, however, dampens the issue without constituting a definitive solution. Therefore, testing LLM even after alignment efforts remains crucial for detecting any residual deviations with respect to ethical standards. We present EvoTox, an automated testing framework for LLMs' inclination to toxicity, providing a way to quantitatively assess how much LLMs can be pushed towards toxic responses even in the presence of alignment. The framework adopts an iterative evolution strategy that exploits the interplay between two LLMs, the System Under Test (SUT) and the Prompt Generator steering SUT responses toward higher toxicity. The toxicity level is assessed by an automated oracle based on an existing toxicity classifier. We conduct a quantitative and qualitative empirical evaluation using four state-of-the-art LLMs as evaluation subjects having increasing complexity (7-13 billion parameters). Our quantitative evaluation assesses the cost-effectiveness of four alternative versions of EvoTox against existing baseline methods, based on random search, curated datasets of toxic prompts, and adversarial attacks. Our qualitative assessment engages human evaluators to rate the fluency of the generated prompts and the perceived toxicity of the responses collected during the testing sessions. Results indicate that the effectiveness, in terms of detected toxicity level, is significantly higher than the selected baseline methods (effect size up to 1.0 against random search and up to 0.99 against adversarial attacks). Furthermore, EvoTox yields a limited cost overhead (from 22% to 35% on average).



## **32. Heuristic-Induced Multimodal Risk Distribution Jailbreak Attack for Multimodal Large Language Models**

cs.CR

**SubmitDate**: 2025-01-03    [abs](http://arxiv.org/abs/2412.05934v2) [paper-pdf](http://arxiv.org/pdf/2412.05934v2)

**Authors**: Ma Teng, Jia Xiaojun, Duan Ranjie, Li Xinfeng, Huang Yihao, Chu Zhixuan, Liu Yang, Ren Wenqi

**Abstract**: With the rapid advancement of multimodal large language models (MLLMs), concerns regarding their security have increasingly captured the attention of both academia and industry. Although MLLMs are vulnerable to jailbreak attacks, designing effective multimodal jailbreak attacks poses unique challenges, especially given the distinct protective measures implemented across various modalities in commercial models. Previous works concentrate risks into a single modality, resulting in limited jailbreak performance. In this paper, we propose a heuristic-induced multimodal risk distribution jailbreak attack method, called HIMRD, which consists of two elements: multimodal risk distribution strategy and heuristic-induced search strategy. The multimodal risk distribution strategy is used to segment harmful instructions across multiple modalities to effectively circumvent MLLMs' security protection. The heuristic-induced search strategy identifies two types of prompts: the understanding-enhancing prompt, which helps the MLLM reconstruct the malicious prompt, and the inducing prompt, which increases the likelihood of affirmative outputs over refusals, enabling a successful jailbreak attack. Extensive experiments demonstrate that this approach effectively uncovers vulnerabilities in MLLMs, achieving an average attack success rate of 90% across seven popular open-source MLLMs and an average attack success rate of around 68% in three popular closed-source MLLMs. Our code will coming soon. Warning: This paper contains offensive and harmful examples, reader discretion is advised.



## **33. Spot Risks Before Speaking! Unraveling Safety Attention Heads in Large Vision-Language Models**

cs.LG

**SubmitDate**: 2025-01-03    [abs](http://arxiv.org/abs/2501.02029v1) [paper-pdf](http://arxiv.org/pdf/2501.02029v1)

**Authors**: Ziwei Zheng, Junyao Zhao, Le Yang, Lijun He, Fan Li

**Abstract**: With the integration of an additional modality, large vision-language models (LVLMs) exhibit greater vulnerability to safety risks (e.g., jailbreaking) compared to their language-only predecessors. Although recent studies have devoted considerable effort to the post-hoc alignment of LVLMs, the inner safety mechanisms remain largely unexplored. In this paper, we discover that internal activations of LVLMs during the first token generation can effectively identify malicious prompts across different attacks. This inherent safety perception is governed by sparse attention heads, which we term ``safety heads." Further analysis reveals that these heads act as specialized shields against malicious prompts; ablating them leads to higher attack success rates, while the model's utility remains unaffected. By locating these safety heads and concatenating their activations, we construct a straightforward but powerful malicious prompt detector that integrates seamlessly into the generation process with minimal extra inference overhead. Despite its simple structure of a logistic regression model, the detector surprisingly exhibits strong zero-shot generalization capabilities. Experiments across various prompt-based attacks confirm the effectiveness of leveraging safety heads to protect LVLMs. Code is available at \url{https://github.com/Ziwei-Zheng/SAHs}.



## **34. BARTPredict: Empowering IoT Security with LLM-Driven Cyber Threat Prediction**

cs.CR

**SubmitDate**: 2025-01-03    [abs](http://arxiv.org/abs/2501.01664v1) [paper-pdf](http://arxiv.org/pdf/2501.01664v1)

**Authors**: Alaeddine Diaf, Abdelaziz Amara Korba, Nour Elislem Karabadji, Yacine Ghamri-Doudane

**Abstract**: The integration of Internet of Things (IoT) technology in various domains has led to operational advancements, but it has also introduced new vulnerabilities to cybersecurity threats, as evidenced by recent widespread cyberattacks on IoT devices. Intrusion detection systems are often reactive, triggered by specific patterns or anomalies observed within the network. To address this challenge, this work proposes a proactive approach to anticipate and preemptively mitigate malicious activities, aiming to prevent potential damage before it occurs. This paper proposes an innovative intrusion prediction framework empowered by Pre-trained Large Language Models (LLMs). The framework incorporates two LLMs: a fine-tuned Bidirectional and AutoRegressive Transformers (BART) model for predicting network traffic and a fine-tuned Bidirectional Encoder Representations from Transformers (BERT) model for evaluating the predicted traffic. By harnessing the bidirectional capabilities of BART the framework then identifies malicious packets among these predictions. Evaluated using the CICIoT2023 IoT attack dataset, our framework showcases a notable enhancement in predictive performance, attaining an impressive 98% overall accuracy, providing a powerful response to the cybersecurity challenges that confront IoT networks.



## **35. CySecBench: Generative AI-based CyberSecurity-focused Prompt Dataset for Benchmarking Large Language Models**

cs.CR

**SubmitDate**: 2025-01-02    [abs](http://arxiv.org/abs/2501.01335v1) [paper-pdf](http://arxiv.org/pdf/2501.01335v1)

**Authors**: Johan Wahréus, Ahmed Mohamed Hussain, Panos Papadimitratos

**Abstract**: Numerous studies have investigated methods for jailbreaking Large Language Models (LLMs) to generate harmful content. Typically, these methods are evaluated using datasets of malicious prompts designed to bypass security policies established by LLM providers. However, the generally broad scope and open-ended nature of existing datasets can complicate the assessment of jailbreaking effectiveness, particularly in specific domains, notably cybersecurity. To address this issue, we present and publicly release CySecBench, a comprehensive dataset containing 12662 prompts specifically designed to evaluate jailbreaking techniques in the cybersecurity domain. The dataset is organized into 10 distinct attack-type categories, featuring close-ended prompts to enable a more consistent and accurate assessment of jailbreaking attempts. Furthermore, we detail our methodology for dataset generation and filtration, which can be adapted to create similar datasets in other domains. To demonstrate the utility of CySecBench, we propose and evaluate a jailbreaking approach based on prompt obfuscation. Our experimental results show that this method successfully elicits harmful content from commercial black-box LLMs, achieving Success Rates (SRs) of 65% with ChatGPT and 88% with Gemini; in contrast, Claude demonstrated greater resilience with a jailbreaking SR of 17%. Compared to existing benchmark approaches, our method shows superior performance, highlighting the value of domain-specific evaluation datasets for assessing LLM security measures. Moreover, when evaluated using prompts from a widely used dataset (i.e., AdvBench), it achieved an SR of 78.5%, higher than the state-of-the-art methods.



## **36. Safeguarding Large Language Models in Real-time with Tunable Safety-Performance Trade-offs**

cs.CL

**SubmitDate**: 2025-01-02    [abs](http://arxiv.org/abs/2501.02018v1) [paper-pdf](http://arxiv.org/pdf/2501.02018v1)

**Authors**: Joao Fonseca, Andrew Bell, Julia Stoyanovich

**Abstract**: Large Language Models (LLMs) have been shown to be susceptible to jailbreak attacks, or adversarial attacks used to illicit high risk behavior from a model. Jailbreaks have been exploited by cybercriminals and blackhat actors to cause significant harm, highlighting the critical need to safeguard widely-deployed models. Safeguarding approaches, which include fine-tuning models or having LLMs "self-reflect", may lengthen the inference time of a model, incur a computational penalty, reduce the semantic fluency of an output, and restrict ``normal'' model behavior. Importantly, these Safety-Performance Trade-offs (SPTs) remain an understudied area. In this work, we introduce a novel safeguard, called SafeNudge, that combines Controlled Text Generation with "nudging", or using text interventions to change the behavior of a model. SafeNudge triggers during text-generation while a jailbreak attack is being executed, and can reduce successful jailbreak attempts by 30% by guiding the LLM towards a safe responses. It adds minimal latency to inference and has a negligible impact on the semantic fluency of outputs. Further, we allow for tunable SPTs. SafeNudge is open-source and available through https://pypi.org/, and is compatible with models loaded with the Hugging Face "transformers" library.



## **37. Security Attacks on LLM-based Code Completion Tools**

cs.CL

Paper accepted at AAAI 2025

**SubmitDate**: 2025-01-02    [abs](http://arxiv.org/abs/2408.11006v4) [paper-pdf](http://arxiv.org/pdf/2408.11006v4)

**Authors**: Wen Cheng, Ke Sun, Xinyu Zhang, Wei Wang

**Abstract**: The rapid development of large language models (LLMs) has significantly advanced code completion capabilities, giving rise to a new generation of LLM-based Code Completion Tools (LCCTs). Unlike general-purpose LLMs, these tools possess unique workflows, integrating multiple information sources as input and prioritizing code suggestions over natural language interaction, which introduces distinct security challenges. Additionally, LCCTs often rely on proprietary code datasets for training, raising concerns about the potential exposure of sensitive data. This paper exploits these distinct characteristics of LCCTs to develop targeted attack methodologies on two critical security risks: jailbreaking and training data extraction attacks. Our experimental results expose significant vulnerabilities within LCCTs, including a 99.4% success rate in jailbreaking attacks on GitHub Copilot and a 46.3% success rate on Amazon Q. Furthermore, We successfully extracted sensitive user data from GitHub Copilot, including 54 real email addresses and 314 physical addresses associated with GitHub usernames. Our study also demonstrates that these code-based attack methods are effective against general-purpose LLMs, such as the GPT series, highlighting a broader security misalignment in the handling of code by modern LLMs. These findings underscore critical security challenges associated with LCCTs and suggest essential directions for strengthening their security frameworks. The example code and attack samples from our research are provided at https://github.com/Sensente/Security-Attacks-on-LCCTs.



## **38. TrustRAG: Enhancing Robustness and Trustworthiness in RAG**

cs.CL

**SubmitDate**: 2025-01-01    [abs](http://arxiv.org/abs/2501.00879v1) [paper-pdf](http://arxiv.org/pdf/2501.00879v1)

**Authors**: Huichi Zhou, Kin-Hei Lee, Zhonghao Zhan, Yue Chen, Zhenhao Li

**Abstract**: Retrieval-Augmented Generation (RAG) systems enhance large language models (LLMs) by integrating external knowledge sources, enabling more accurate and contextually relevant responses tailored to user queries. However, these systems remain vulnerable to corpus poisoning attacks that can significantly degrade LLM performance through the injection of malicious content. To address these challenges, we propose TrustRAG, a robust framework that systematically filters compromised and irrelevant content before it reaches the language model. Our approach implements a two-stage defense mechanism: first, it employs K-means clustering to identify potential attack patterns in retrieved documents based on their semantic embeddings, effectively isolating suspicious content. Second, it leverages cosine similarity and ROUGE metrics to detect malicious documents while resolving discrepancies between the model's internal knowledge and external information through a self-assessment process. TrustRAG functions as a plug-and-play, training-free module that integrates seamlessly with any language model, whether open or closed-source, maintaining high contextual relevance while strengthening defenses against attacks. Through extensive experimental validation, we demonstrate that TrustRAG delivers substantial improvements in retrieval accuracy, efficiency, and attack resistance compared to existing approaches across multiple model architectures and datasets. We have made TrustRAG available as open-source software at \url{https://github.com/HuichiZhou/TrustRAG}.



## **39. Dynamics of Adversarial Attacks on Large Language Model-Based Search Engines**

cs.CL

**SubmitDate**: 2025-01-01    [abs](http://arxiv.org/abs/2501.00745v1) [paper-pdf](http://arxiv.org/pdf/2501.00745v1)

**Authors**: Xiyang Hu

**Abstract**: The increasing integration of Large Language Model (LLM) based search engines has transformed the landscape of information retrieval. However, these systems are vulnerable to adversarial attacks, especially ranking manipulation attacks, where attackers craft webpage content to manipulate the LLM's ranking and promote specific content, gaining an unfair advantage over competitors. In this paper, we study the dynamics of ranking manipulation attacks. We frame this problem as an Infinitely Repeated Prisoners' Dilemma, where multiple players strategically decide whether to cooperate or attack. We analyze the conditions under which cooperation can be sustained, identifying key factors such as attack costs, discount rates, attack success rates, and trigger strategies that influence player behavior. We identify tipping points in the system dynamics, demonstrating that cooperation is more likely to be sustained when players are forward-looking. However, from a defense perspective, we find that simply reducing attack success probabilities can, paradoxically, incentivize attacks under certain conditions. Furthermore, defensive measures to cap the upper bound of attack success rates may prove futile in some scenarios. These insights highlight the complexity of securing LLM-based systems. Our work provides a theoretical foundation and practical insights for understanding and mitigating their vulnerabilities, while emphasizing the importance of adaptive security strategies and thoughtful ecosystem design.



## **40. From Sands to Mansions: Simulating Full Attack Chain with LLM-Organized Knowledge**

cs.CR

**SubmitDate**: 2024-12-31    [abs](http://arxiv.org/abs/2407.16928v2) [paper-pdf](http://arxiv.org/pdf/2407.16928v2)

**Authors**: Lingzhi Wang, Zhenyuan Li, Zonghan Guo, Yi Jiang, Kyle Jung, Kedar Thiagarajan, Jiahui Wang, Zhengkai Wang, Emily Wei, Xiangmin Shen, Yan Chen

**Abstract**: Adversarial dynamics are intrinsic to the nature of offense and defense in cyberspace, with both attackers and defenders continuously evolving their technologies. Given the wide array of security products available, users often face challenges in selecting the most effective solutions. Furthermore, traditional benchmarks based on single-point attacks are increasingly inadequate, failing to accurately reflect the full range of attacker capabilities and falling short in properly evaluating the effectiveness of defense products. Automated multi-stage attack simulations offer a promising approach to enhance system evaluation efficiency and aid in analyzing the effectiveness of detection systems. However, simulating a full attack chain is complex and requires significant time and expertise from security professionals, facing several challenges, including limited coverage of attack techniques, a high level of required expertise, and a lack of execution detail. In this paper, we model automatic attack simulation as a planning problem. By using the Planning Domain Definition Language (PDDL) to formally describe the attack simulation problem, and combining domain knowledge of both the problem and the domain space, we enable the planning of attack paths through standardized, domain-independent planning algorithms. We explore the potential of Large Language Models (LLMs) to summarize and analyze knowledge from existing attack documentation and reports, facilitating automated attack planning. We introduce Aurora, a system that autonomously simulates full attack chains based on external attack tools and threat intelligence reports.



## **41. Toward Intelligent and Secure Cloud: Large Language Model Empowered Proactive Defense**

cs.CR

7 pages; In submission

**SubmitDate**: 2024-12-30    [abs](http://arxiv.org/abs/2412.21051v1) [paper-pdf](http://arxiv.org/pdf/2412.21051v1)

**Authors**: Yuyang Zhou, Guang Cheng, Kang Du, Zihan Chen

**Abstract**: The rapid evolution of cloud computing technologies and the increasing number of cloud applications have provided a large number of benefits in daily lives. However, the diversity and complexity of different components pose a significant challenge to cloud security, especially when dealing with sophisticated and advanced cyberattacks. Recent advancements in generative foundation models (GFMs), particularly in the large language models (LLMs), offer promising solutions for security intelligence. By exploiting the powerful abilities in language understanding, data analysis, task inference, action planning, and code generation, we present LLM-PD, a novel proactive defense architecture that defeats various threats in a proactive manner. LLM-PD can efficiently make a decision through comprehensive data analysis and sequential reasoning, as well as dynamically creating and deploying actionable defense mechanisms on the target cloud. Furthermore, it can flexibly self-evolve based on experience learned from previous interactions and adapt to new attack scenarios without additional training. The experimental results demonstrate its remarkable ability in terms of defense effectiveness and efficiency, particularly highlighting an outstanding success rate when compared with other existing methods.



## **42. Unsupervised dense retrieval with conterfactual contrastive learning**

cs.IR

arXiv admin note: text overlap with arXiv:2107.07773 by other authors

**SubmitDate**: 2024-12-30    [abs](http://arxiv.org/abs/2412.20756v1) [paper-pdf](http://arxiv.org/pdf/2412.20756v1)

**Authors**: Haitian Chen, Qingyao Ai, Xiao Wang, Yiqun Liu, Fen Lin, Qin Liu

**Abstract**: Efficiently retrieving a concise set of candidates from a large document corpus remains a pivotal challenge in Information Retrieval (IR). Neural retrieval models, particularly dense retrieval models built with transformers and pretrained language models, have been popular due to their superior performance. However, criticisms have also been raised on their lack of explainability and vulnerability to adversarial attacks. In response to these challenges, we propose to improve the robustness of dense retrieval models by enhancing their sensitivity of fine-graned relevance signals. A model achieving sensitivity in this context should exhibit high variances when documents' key passages determining their relevance to queries have been modified, while maintaining low variances for other changes in irrelevant passages. This sensitivity allows a dense retrieval model to produce robust results with respect to attacks that try to promote documents without actually increasing their relevance. It also makes it possible to analyze which part of a document is actually relevant to a query, and thus improve the explainability of the retrieval model. Motivated by causality and counterfactual analysis, we propose a series of counterfactual regularization methods based on game theory and unsupervised learning with counterfactual passages. Experiments show that, our method can extract key passages without reliance on the passage-level relevance annotations. Moreover, the regularized dense retrieval models exhibit heightened robustness against adversarial attacks, surpassing the state-of-the-art anti-attack methods.



## **43. SafeSynthDP: Leveraging Large Language Models for Privacy-Preserving Synthetic Data Generation Using Differential Privacy**

cs.LG

15 pages, 1 figure, 5 tables

**SubmitDate**: 2024-12-30    [abs](http://arxiv.org/abs/2412.20641v1) [paper-pdf](http://arxiv.org/pdf/2412.20641v1)

**Authors**: Md Mahadi Hasan Nahid, Sadid Bin Hasan

**Abstract**: Machine learning (ML) models frequently rely on training data that may include sensitive or personal information, raising substantial privacy concerns. Legislative frameworks such as the General Data Protection Regulation (GDPR) and the California Consumer Privacy Act (CCPA) have necessitated the development of strategies that preserve privacy while maintaining the utility of data. In this paper, we investigate the capability of Large Language Models (LLMs) to generate synthetic datasets integrated with Differential Privacy (DP) mechanisms, thereby enabling data-driven research and model training without direct exposure of sensitive information. Our approach incorporates DP-based noise injection methods, including Laplace and Gaussian distributions, into the data generation process. We then evaluate the utility of these DP-enhanced synthetic datasets by comparing the performance of ML models trained on them against models trained on the original data. To substantiate privacy guarantees, we assess the resilience of the generated synthetic data to membership inference attacks and related threats. The experimental results demonstrate that integrating DP within LLM-driven synthetic data generation offers a viable balance between privacy protection and data utility. This study provides a foundational methodology and insight into the privacy-preserving capabilities of LLMs, paving the way for compliant and effective ML research and applications.



## **44. HALLUCINOGEN: A Benchmark for Evaluating Object Hallucination in Large Visual-Language Models**

cs.CV

**SubmitDate**: 2024-12-29    [abs](http://arxiv.org/abs/2412.20622v1) [paper-pdf](http://arxiv.org/pdf/2412.20622v1)

**Authors**: Ashish Seth, Dinesh Manocha, Chirag Agarwal

**Abstract**: Large Vision-Language Models (LVLMs) have demonstrated remarkable performance in performing complex multimodal tasks. However, they are still plagued by object hallucination: the misidentification or misclassification of objects present in images. To this end, we propose HALLUCINOGEN, a novel visual question answering (VQA) object hallucination attack benchmark that utilizes diverse contextual reasoning prompts to evaluate object hallucination in state-of-the-art LVLMs. We design a series of contextual reasoning hallucination prompts to evaluate LVLMs' ability to accurately identify objects in a target image while asking them to perform diverse visual-language tasks such as identifying, locating or performing visual reasoning around specific objects. Further, we extend our benchmark to high-stakes medical applications and introduce MED-HALLUCINOGEN, hallucination attacks tailored to the biomedical domain, and evaluate the hallucination performance of LVLMs on medical images, a critical area where precision is crucial. Finally, we conduct extensive evaluations of eight LVLMs and two hallucination mitigation strategies across multiple datasets to show that current generic and medical LVLMs remain susceptible to hallucination attacks.



## **45. Pirates of the RAG: Adaptively Attacking LLMs to Leak Knowledge Bases**

cs.AI

**SubmitDate**: 2024-12-29    [abs](http://arxiv.org/abs/2412.18295v2) [paper-pdf](http://arxiv.org/pdf/2412.18295v2)

**Authors**: Christian Di Maio, Cristian Cosci, Marco Maggini, Valentina Poggioni, Stefano Melacci

**Abstract**: The growing ubiquity of Retrieval-Augmented Generation (RAG) systems in several real-world services triggers severe concerns about their security. A RAG system improves the generative capabilities of a Large Language Models (LLM) by a retrieval mechanism which operates on a private knowledge base, whose unintended exposure could lead to severe consequences, including breaches of private and sensitive information. This paper presents a black-box attack to force a RAG system to leak its private knowledge base which, differently from existing approaches, is adaptive and automatic. A relevance-based mechanism and an attacker-side open-source LLM favor the generation of effective queries to leak most of the (hidden) knowledge base. Extensive experimentation proves the quality of the proposed algorithm in different RAG pipelines and domains, comparing to very recent related approaches, which turn out to be either not fully black-box, not adaptive, or not based on open-source models. The findings from our study remark the urgent need for more robust privacy safeguards in the design and deployment of RAG systems.



## **46. Can Watermarked LLMs be Identified by Users via Crafted Prompts?**

cs.CR

30 pages, 5 figures, 11 tables

**SubmitDate**: 2024-12-28    [abs](http://arxiv.org/abs/2410.03168v2) [paper-pdf](http://arxiv.org/pdf/2410.03168v2)

**Authors**: Aiwei Liu, Sheng Guan, Yiming Liu, Leyi Pan, Yifei Zhang, Liancheng Fang, Lijie Wen, Philip S. Yu, Xuming Hu

**Abstract**: Text watermarking for Large Language Models (LLMs) has made significant progress in detecting LLM outputs and preventing misuse. Current watermarking techniques offer high detectability, minimal impact on text quality, and robustness to text editing. However, current researches lack investigation into the imperceptibility of watermarking techniques in LLM services. This is crucial as LLM providers may not want to disclose the presence of watermarks in real-world scenarios, as it could reduce user willingness to use the service and make watermarks more vulnerable to attacks. This work is the first to investigate the imperceptibility of watermarked LLMs. We design an identification algorithm called Water-Probe that detects watermarks through well-designed prompts to the LLM. Our key motivation is that current watermarked LLMs expose consistent biases under the same watermark key, resulting in similar differences across prompts under different watermark keys. Experiments show that almost all mainstream watermarking algorithms are easily identified with our well-designed prompts, while Water-Probe demonstrates a minimal false positive rate for non-watermarked LLMs. Finally, we propose that the key to enhancing the imperceptibility of watermarked LLMs is to increase the randomness of watermark key selection. Based on this, we introduce the Water-Bag strategy, which significantly improves watermark imperceptibility by merging multiple watermark keys.



## **47. Defending Against Network Attacks for Secure AI Agent Migration in Vehicular Metaverses**

cs.NI

**SubmitDate**: 2024-12-28    [abs](http://arxiv.org/abs/2412.20154v1) [paper-pdf](http://arxiv.org/pdf/2412.20154v1)

**Authors**: Xinru Wen, Jinbo Wen, Ming Xiao, Jiawen Kang, Tao Zhang, Xiaohuan Li, Chuanxi Chen, Dusit Niyato

**Abstract**: Vehicular metaverses, blending traditional vehicular networks with metaverse technology, are expected to revolutionize fields such as autonomous driving. As virtual intelligent assistants in vehicular metaverses, Artificial Intelligence (AI) agents powered by large language models can create immersive 3D virtual spaces for passengers to enjoy on-broad vehicular applications and services. To provide users with seamless and engaging virtual interactions, resource-limited vehicles offload AI agents to RoadSide Units (RSUs) with adequate communication and computational capabilities. Due to the mobility of vehicles and the limited coverage of RSUs, AI agents need to migrate from one RSU to another RSU. However, potential network attacks pose significant challenges to ensuring reliable and efficient AI agent migration. In this paper, we first explore specific network attacks including traffic-based attacks (i.e., DDoS attacks) and infrastructure-based attacks (i.e., malicious RSU attacks). Then, we model the AI agent migration process as a Partially Observable Markov Decision Process (POMDP) and apply multi-agent proximal policy optimization algorithms to mitigate DDoS attacks. In addition, we propose a trust assessment mechanism to counter malicious RSU attacks. Numerical results validate that the proposed solutions effectively defend against these network attacks and reduce the total latency of AI agent migration by approximately 43.3%.



## **48. On the Validity of Traditional Vulnerability Scoring Systems for Adversarial Attacks against LLMs**

cs.CR

101 pages, 3 figures

**SubmitDate**: 2024-12-28    [abs](http://arxiv.org/abs/2412.20087v1) [paper-pdf](http://arxiv.org/pdf/2412.20087v1)

**Authors**: Atmane Ayoub Mansour Bahar, Ahmad Samer Wazan

**Abstract**: This research investigates the effectiveness of established vulnerability metrics, such as the Common Vulnerability Scoring System (CVSS), in evaluating attacks against Large Language Models (LLMs), with a focus on Adversarial Attacks (AAs). The study explores the influence of both general and specific metric factors in determining vulnerability scores, providing new perspectives on potential enhancements to these metrics.   This study adopts a quantitative approach, calculating and comparing the coefficient of variation of vulnerability scores across 56 adversarial attacks on LLMs. The attacks, sourced from various research papers, and obtained through online databases, were evaluated using multiple vulnerability metrics. Scores were determined by averaging the values assessed by three distinct LLMs. The results indicate that existing scoring-systems yield vulnerability scores with minimal variation across different attacks, suggesting that many of the metric factors are inadequate for assessing adversarial attacks on LLMs. This is particularly true for context-specific factors or those with predefined value sets, such as those in CVSS. These findings support the hypothesis that current vulnerability metrics, especially those with rigid values, are limited in evaluating AAs on LLMs, highlighting the need for the development of more flexible, generalized metrics tailored to such attacks.   This research offers a fresh analysis of the effectiveness and applicability of established vulnerability metrics, particularly in the context of Adversarial Attacks on Large Language Models, both of which have gained significant attention in recent years. Through extensive testing and calculations, the study underscores the limitations of these metrics and opens up new avenues for improving and refining vulnerability assessment frameworks specifically tailored for LLMs.



## **49. LLM-Virus: Evolutionary Jailbreak Attack on Large Language Models**

cs.CR

**SubmitDate**: 2024-12-28    [abs](http://arxiv.org/abs/2501.00055v1) [paper-pdf](http://arxiv.org/pdf/2501.00055v1)

**Authors**: Miao Yu, Junfeng Fang, Yingjie Zhou, Xing Fan, Kun Wang, Shirui Pan, Qingsong Wen

**Abstract**: While safety-aligned large language models (LLMs) are increasingly used as the cornerstone for powerful systems such as multi-agent frameworks to solve complex real-world problems, they still suffer from potential adversarial queries, such as jailbreak attacks, which attempt to induce harmful content. Researching attack methods allows us to better understand the limitations of LLM and make trade-offs between helpfulness and safety. However, existing jailbreak attacks are primarily based on opaque optimization techniques (e.g. token-level gradient descent) and heuristic search methods like LLM refinement, which fall short in terms of transparency, transferability, and computational cost. In light of these limitations, we draw inspiration from the evolution and infection processes of biological viruses and propose LLM-Virus, a jailbreak attack method based on evolutionary algorithm, termed evolutionary jailbreak. LLM-Virus treats jailbreak attacks as both an evolutionary and transfer learning problem, utilizing LLMs as heuristic evolutionary operators to ensure high attack efficiency, transferability, and low time cost. Our experimental results on multiple safety benchmarks show that LLM-Virus achieves competitive or even superior performance compared to existing attack methods.



## **50. B-AVIBench: Towards Evaluating the Robustness of Large Vision-Language Model on Black-box Adversarial Visual-Instructions**

cs.CV

Accepted by IEEE Transactions on Information Forensics & Security

**SubmitDate**: 2024-12-28    [abs](http://arxiv.org/abs/2403.09346v2) [paper-pdf](http://arxiv.org/pdf/2403.09346v2)

**Authors**: Hao Zhang, Wenqi Shao, Hong Liu, Yongqiang Ma, Ping Luo, Yu Qiao, Nanning Zheng, Kaipeng Zhang

**Abstract**: Large Vision-Language Models (LVLMs) have shown significant progress in responding well to visual-instructions from users. However, these instructions, encompassing images and text, are susceptible to both intentional and inadvertent attacks. Despite the critical importance of LVLMs' robustness against such threats, current research in this area remains limited. To bridge this gap, we introduce B-AVIBench, a framework designed to analyze the robustness of LVLMs when facing various Black-box Adversarial Visual-Instructions (B-AVIs), including four types of image-based B-AVIs, ten types of text-based B-AVIs, and nine types of content bias B-AVIs (such as gender, violence, cultural, and racial biases, among others). We generate 316K B-AVIs encompassing five categories of multimodal capabilities (ten tasks) and content bias. We then conduct a comprehensive evaluation involving 14 open-source LVLMs to assess their performance. B-AVIBench also serves as a convenient tool for practitioners to evaluate the robustness of LVLMs against B-AVIs. Our findings and extensive experimental results shed light on the vulnerabilities of LVLMs, and highlight that inherent biases exist even in advanced closed-source LVLMs like GeminiProVision and GPT-4V. This underscores the importance of enhancing the robustness, security, and fairness of LVLMs. The source code and benchmark are available at https://github.com/zhanghao5201/B-AVIBench.



