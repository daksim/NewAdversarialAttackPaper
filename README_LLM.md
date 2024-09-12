# Latest Large Language Model Attack Papers
**update at 2024-09-12 10:28:55**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Securing Vision-Language Models with a Robust Encoder Against Jailbreak and Adversarial Attacks**

cs.CV

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2409.07353v1) [paper-pdf](http://arxiv.org/pdf/2409.07353v1)

**Authors**: Md Zarif Hossain, Ahmed Imteaj

**Abstract**: Large Vision-Language Models (LVLMs), trained on multimodal big datasets, have significantly advanced AI by excelling in vision-language tasks. However, these models remain vulnerable to adversarial attacks, particularly jailbreak attacks, which bypass safety protocols and cause the model to generate misleading or harmful responses. This vulnerability stems from both the inherent susceptibilities of LLMs and the expanded attack surface introduced by the visual modality. We propose Sim-CLIP+, a novel defense mechanism that adversarially fine-tunes the CLIP vision encoder by leveraging a Siamese architecture. This approach maximizes cosine similarity between perturbed and clean samples, facilitating resilience against adversarial manipulations. Sim-CLIP+ offers a plug-and-play solution, allowing seamless integration into existing LVLM architectures as a robust vision encoder. Unlike previous defenses, our method requires no structural modifications to the LVLM and incurs minimal computational overhead. Sim-CLIP+ demonstrates effectiveness against both gradient-based adversarial attacks and various jailbreak techniques. We evaluate Sim-CLIP+ against three distinct jailbreak attack strategies and perform clean evaluations using standard downstream datasets, including COCO for image captioning and OKVQA for visual question answering. Extensive experiments demonstrate that Sim-CLIP+ maintains high clean accuracy while substantially improving robustness against both gradient-based adversarial attacks and jailbreak techniques. Our code and robust vision encoders are available at https://github.com/speedlab-git/Robust-Encoder-against-Jailbreak-attack.git.



## **2. AEGIS: Online Adaptive AI Content Safety Moderation with Ensemble of LLM Experts**

cs.LG

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2404.05993v2) [paper-pdf](http://arxiv.org/pdf/2404.05993v2)

**Authors**: Shaona Ghosh, Prasoon Varshney, Erick Galinkin, Christopher Parisien

**Abstract**: As Large Language Models (LLMs) and generative AI become more widespread, the content safety risks associated with their use also increase. We find a notable deficiency in high-quality content safety datasets and benchmarks that comprehensively cover a wide range of critical safety areas. To address this, we define a broad content safety risk taxonomy, comprising 13 critical risk and 9 sparse risk categories. Additionally, we curate AEGISSAFETYDATASET, a new dataset of approximately 26, 000 human-LLM interaction instances, complete with human annotations adhering to the taxonomy. We plan to release this dataset to the community to further research and to help benchmark LLM models for safety. To demonstrate the effectiveness of the dataset, we instruction-tune multiple LLM-based safety models. We show that our models (named AEGISSAFETYEXPERTS), not only surpass or perform competitively with the state-of-the-art LLM-based safety models and general purpose LLMs, but also exhibit robustness across multiple jail-break attack categories. We also show how using AEGISSAFETYDATASET during the LLM alignment phase does not negatively impact the performance of the aligned models on MT Bench scores. Furthermore, we propose AEGIS, a novel application of a no-regret online adaptation framework with strong theoretical guarantees, to perform content moderation with an ensemble of LLM content safety experts in deployment



## **3. The Philosopher's Stone: Trojaning Plugins of Large Language Models**

cs.CR

Accepted by NDSS Symposium 2025. Please cite this paper as "Tian  Dong, Minhui Xue, Guoxing Chen, Rayne Holland, Yan Meng, Shaofeng Li, Zhen  Liu, Haojin Zhu. The Philosopher's Stone: Trojaning Plugins of Large Language  Models. In the 32nd Annual Network and Distributed System Security Symposium  (NDSS 2025)."

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2312.00374v3) [paper-pdf](http://arxiv.org/pdf/2312.00374v3)

**Authors**: Tian Dong, Minhui Xue, Guoxing Chen, Rayne Holland, Yan Meng, Shaofeng Li, Zhen Liu, Haojin Zhu

**Abstract**: Open-source Large Language Models (LLMs) have recently gained popularity because of their comparable performance to proprietary LLMs. To efficiently fulfill domain-specialized tasks, open-source LLMs can be refined, without expensive accelerators, using low-rank adapters. However, it is still unknown whether low-rank adapters can be exploited to control LLMs. To address this gap, we demonstrate that an infected adapter can induce, on specific triggers,an LLM to output content defined by an adversary and to even maliciously use tools. To train a Trojan adapter, we propose two novel attacks, POLISHED and FUSION, that improve over prior approaches. POLISHED uses a superior LLM to align na\"ively poisoned data based on our insight that it can better inject poisoning knowledge during training. In contrast, FUSION leverages a novel over-poisoning procedure to transform a benign adapter into a malicious one by magnifying the attention between trigger and target in model weights. In our experiments, we first conduct two case studies to demonstrate that a compromised LLM agent can use malware to control the system (e.g., a LLM-driven robot) or to launch a spear-phishing attack. Then, in terms of targeted misinformation, we show that our attacks provide higher attack effectiveness than the existing baseline and, for the purpose of attracting downloads, preserve or improve the adapter's utility. Finally, we designed and evaluated three potential defenses. However, none proved entirely effective in safeguarding against our attacks, highlighting the need for more robust defenses supporting a secure LLM supply chain.



## **4. Well, that escalated quickly: The Single-Turn Crescendo Attack (STCA)**

cs.CR

**SubmitDate**: 2024-09-10    [abs](http://arxiv.org/abs/2409.03131v2) [paper-pdf](http://arxiv.org/pdf/2409.03131v2)

**Authors**: Alan Aqrawi, Arian Abbasi

**Abstract**: This paper introduces a new method for adversarial attacks on large language models (LLMs) called the Single-Turn Crescendo Attack (STCA). Building on the multi-turn crescendo attack method introduced by Russinovich, Salem, and Eldan (2024), which gradually escalates the context to provoke harmful responses, the STCA achieves similar outcomes in a single interaction. By condensing the escalation into a single, well-crafted prompt, the STCA bypasses typical moderation filters that LLMs use to prevent inappropriate outputs. This technique reveals vulnerabilities in current LLMs and emphasizes the importance of stronger safeguards in responsible AI (RAI). The STCA offers a novel method that has not been previously explored.



## **5. Espresso: Robust Concept Filtering in Text-to-Image Models**

cs.CV

**SubmitDate**: 2024-09-09    [abs](http://arxiv.org/abs/2404.19227v5) [paper-pdf](http://arxiv.org/pdf/2404.19227v5)

**Authors**: Anudeep Das, Vasisht Duddu, Rui Zhang, N. Asokan

**Abstract**: Diffusion based text-to-image models are trained on large datasets scraped from the Internet, potentially containing unacceptable concepts (e.g., copyright infringing or unsafe). We need concept removal techniques (CRTs) which are effective in preventing the generation of images with unacceptable concepts, utility-preserving on acceptable concepts, and robust against evasion with adversarial prompts. None of the prior CRTs satisfy all these requirements simultaneously. We introduce Espresso, the first robust concept filter based on Contrastive Language-Image Pre-Training (CLIP). We configure CLIP to identify unacceptable concepts in generated images using the distance of their embeddings to the text embeddings of both unacceptable and acceptable concepts. This lets us fine-tune for robustness by separating the text embeddings of unacceptable and acceptable concepts while preserving their pairing with image embeddings for utility. We present a pipeline to evaluate various CRTs, attacks against them, and show that Espresso, is more effective and robust than prior CRTs, while retaining utility.



## **6. Exploiting the Vulnerability of Large Language Models via Defense-Aware Architectural Backdoor**

cs.CR

**SubmitDate**: 2024-09-09    [abs](http://arxiv.org/abs/2409.01952v2) [paper-pdf](http://arxiv.org/pdf/2409.01952v2)

**Authors**: Abdullah Arafat Miah, Yu Bi

**Abstract**: Deep neural networks (DNNs) have long been recognized as vulnerable to backdoor attacks. By providing poisoned training data in the fine-tuning process, the attacker can implant a backdoor into the victim model. This enables input samples meeting specific textual trigger patterns to be classified as target labels of the attacker's choice. While such black-box attacks have been well explored in both computer vision and natural language processing (NLP), backdoor attacks relying on white-box attack philosophy have hardly been thoroughly investigated. In this paper, we take the first step to introduce a new type of backdoor attack that conceals itself within the underlying model architecture. Specifically, we propose to design separate backdoor modules consisting of two functions: trigger detection and noise injection. The add-on modules of model architecture layers can detect the presence of input trigger tokens and modify layer weights using Gaussian noise to disturb the feature distribution of the baseline model. We conduct extensive experiments to evaluate our attack methods using two model architecture settings on five different large language datasets. We demonstrate that the training-free architectural backdoor on a large language model poses a genuine threat. Unlike the-state-of-art work, it can survive the rigorous fine-tuning and retraining process, as well as evade output probability-based defense methods (i.e. BDDR). All the code and data is available https://github.com/SiSL-URI/Arch_Backdoor_LLM.



## **7. Jailbreaking Text-to-Image Models with LLM-Based Agents**

cs.CR

**SubmitDate**: 2024-09-09    [abs](http://arxiv.org/abs/2408.00523v2) [paper-pdf](http://arxiv.org/pdf/2408.00523v2)

**Authors**: Yingkai Dong, Zheng Li, Xiangtao Meng, Ning Yu, Shanqing Guo

**Abstract**: Recent advancements have significantly improved automated task-solving capabilities using autonomous agents powered by large language models (LLMs). However, most LLM-based agents focus on dialogue, programming, or specialized domains, leaving their potential for addressing generative AI safety tasks largely unexplored. In this paper, we propose Atlas, an advanced LLM-based multi-agent framework targeting generative AI models, specifically focusing on jailbreak attacks against text-to-image (T2I) models with built-in safety filters. Atlas consists of two agents, namely the mutation agent and the selection agent, each comprising four key modules: a vision-language model (VLM) or LLM brain, planning, memory, and tool usage. The mutation agent uses its VLM brain to determine whether a prompt triggers the T2I model's safety filter. It then collaborates iteratively with the LLM brain of the selection agent to generate new candidate jailbreak prompts with the highest potential to bypass the filter. In addition to multi-agent communication, we leverage in-context learning (ICL) memory mechanisms and the chain-of-thought (COT) approach to learn from past successes and failures, thereby enhancing Atlas's performance. Our evaluation demonstrates that Atlas successfully jailbreaks several state-of-the-art T2I models equipped with multi-modal safety filters in a black-box setting. Additionally, Atlas outperforms existing methods in both query efficiency and the quality of generated images. This work convincingly demonstrates the successful application of LLM-based agents in studying the safety vulnerabilities of popular text-to-image generation models. We urge the community to consider advanced techniques like ours in response to the rapidly evolving text-to-image generation field.



## **8. Fine-Tuning, Quantization, and LLMs: Navigating Unintended Outcomes**

cs.CR

**SubmitDate**: 2024-09-09    [abs](http://arxiv.org/abs/2404.04392v3) [paper-pdf](http://arxiv.org/pdf/2404.04392v3)

**Authors**: Divyanshu Kumar, Anurakt Kumar, Sahil Agarwal, Prashanth Harshangi

**Abstract**: Large Language Models (LLMs) have gained widespread adoption across various domains, including chatbots and auto-task completion agents. However, these models are susceptible to safety vulnerabilities such as jailbreaking, prompt injection, and privacy leakage attacks. These vulnerabilities can lead to the generation of malicious content, unauthorized actions, or the disclosure of confidential information. While foundational LLMs undergo alignment training and incorporate safety measures, they are often subject to fine-tuning, or doing quantization resource-constrained environments. This study investigates the impact of these modifications on LLM safety, a critical consideration for building reliable and secure AI systems. We evaluate foundational models including Mistral, Llama series, Qwen, and MosaicML, along with their fine-tuned variants. Our comprehensive analysis reveals that fine-tuning generally increases the success rates of jailbreak attacks, while quantization has variable effects on attack success rates. Importantly, we find that properly implemented guardrails significantly enhance resistance to jailbreak attempts. These findings contribute to our understanding of LLM vulnerabilities and provide insights for developing more robust safety strategies in the deployment of language models.



## **9. A Study on Prompt Injection Attack Against LLM-Integrated Mobile Robotic Systems**

cs.RO

**SubmitDate**: 2024-09-09    [abs](http://arxiv.org/abs/2408.03515v2) [paper-pdf](http://arxiv.org/pdf/2408.03515v2)

**Authors**: Wenxiao Zhang, Xiangrui Kong, Conan Dewitt, Thomas Braunl, Jin B. Hong

**Abstract**: The integration of Large Language Models (LLMs) like GPT-4o into robotic systems represents a significant advancement in embodied artificial intelligence. These models can process multi-modal prompts, enabling them to generate more context-aware responses. However, this integration is not without challenges. One of the primary concerns is the potential security risks associated with using LLMs in robotic navigation tasks. These tasks require precise and reliable responses to ensure safe and effective operation. Multi-modal prompts, while enhancing the robot's understanding, also introduce complexities that can be exploited maliciously. For instance, adversarial inputs designed to mislead the model can lead to incorrect or dangerous navigational decisions. This study investigates the impact of prompt injections on mobile robot performance in LLM-integrated systems and explores secure prompt strategies to mitigate these risks. Our findings demonstrate a substantial overall improvement of approximately 30.8% in both attack detection and system performance with the implementation of robust defence mechanisms, highlighting their critical role in enhancing security and reliability in mission-oriented tasks.



## **10. PIP: Detecting Adversarial Examples in Large Vision-Language Models via Attention Patterns of Irrelevant Probe Questions**

cs.CV

Accepted by ACM Multimedia 2024 BNI track (Oral)

**SubmitDate**: 2024-09-08    [abs](http://arxiv.org/abs/2409.05076v1) [paper-pdf](http://arxiv.org/pdf/2409.05076v1)

**Authors**: Yudong Zhang, Ruobing Xie, Jiansheng Chen, Xingwu Sun, Yu Wang

**Abstract**: Large Vision-Language Models (LVLMs) have demonstrated their powerful multimodal capabilities. However, they also face serious safety problems, as adversaries can induce robustness issues in LVLMs through the use of well-designed adversarial examples. Therefore, LVLMs are in urgent need of detection tools for adversarial examples to prevent incorrect responses. In this work, we first discover that LVLMs exhibit regular attention patterns for clean images when presented with probe questions. We propose an unconventional method named PIP, which utilizes the attention patterns of one randomly selected irrelevant probe question (e.g., "Is there a clock?") to distinguish adversarial examples from clean examples. Regardless of the image to be tested and its corresponding question, PIP only needs to perform one additional inference of the image to be tested and the probe question, and then achieves successful detection of adversarial examples. Even under black-box attacks and open dataset scenarios, our PIP, coupled with a simple SVM, still achieves more than 98% recall and a precision of over 90%. Our PIP is the first attempt to detect adversarial attacks on LVLMs via simple irrelevant probe questions, shedding light on deeper understanding and introspection within LVLMs. The code is available at https://github.com/btzyd/pip.



## **11. Using Large Language Models for Template Detection from Security Event Logs**

cs.CR

**SubmitDate**: 2024-09-08    [abs](http://arxiv.org/abs/2409.05045v1) [paper-pdf](http://arxiv.org/pdf/2409.05045v1)

**Authors**: Risto Vaarandi, Hayretdin Bahsi

**Abstract**: In modern IT systems and computer networks, real-time and offline event log analysis is a crucial part of cyber security monitoring. In particular, event log analysis techniques are essential for the timely detection of cyber attacks and for assisting security experts with the analysis of past security incidents. The detection of line patterns or templates from unstructured textual event logs has been identified as an important task of event log analysis since detected templates represent event types in the event log and prepare the logs for downstream online or offline security monitoring tasks. During the last two decades, a number of template mining algorithms have been proposed. However, many proposed algorithms rely on traditional data mining techniques, and the usage of Large Language Models (LLMs) has received less attention so far. Also, most approaches that harness LLMs are supervised, and unsupervised LLM-based template mining remains an understudied area. The current paper addresses this research gap and investigates the application of LLMs for unsupervised detection of templates from unstructured security event logs.



## **12. Vision-fused Attack: Advancing Aggressive and Stealthy Adversarial Text against Neural Machine Translation**

cs.CL

IJCAI 2024

**SubmitDate**: 2024-09-08    [abs](http://arxiv.org/abs/2409.05021v1) [paper-pdf](http://arxiv.org/pdf/2409.05021v1)

**Authors**: Yanni Xue, Haojie Hao, Jiakai Wang, Qiang Sheng, Renshuai Tao, Yu Liang, Pu Feng, Xianglong Liu

**Abstract**: While neural machine translation (NMT) models achieve success in our daily lives, they show vulnerability to adversarial attacks. Despite being harmful, these attacks also offer benefits for interpreting and enhancing NMT models, thus drawing increased research attention. However, existing studies on adversarial attacks are insufficient in both attacking ability and human imperceptibility due to their sole focus on the scope of language. This paper proposes a novel vision-fused attack (VFA) framework to acquire powerful adversarial text, i.e., more aggressive and stealthy. Regarding the attacking ability, we design the vision-merged solution space enhancement strategy to enlarge the limited semantic solution space, which enables us to search for adversarial candidates with higher attacking ability. For human imperceptibility, we propose the perception-retained adversarial text selection strategy to align the human text-reading mechanism. Thus, the finally selected adversarial text could be more deceptive. Extensive experiments on various models, including large language models (LLMs) like LLaMA and GPT-3.5, strongly support that VFA outperforms the comparisons by large margins (up to 81%/14% improvements on ASR/SSIM).



## **13. TF-Attack: Transferable and Fast Adversarial Attacks on Large Language Models**

cs.CL

14 pages, 6 figures

**SubmitDate**: 2024-09-08    [abs](http://arxiv.org/abs/2408.13985v3) [paper-pdf](http://arxiv.org/pdf/2408.13985v3)

**Authors**: Zelin Li, Kehai Chen, Lemao Liu, Xuefeng Bai, Mingming Yang, Yang Xiang, Min Zhang

**Abstract**: With the great advancements in large language models (LLMs), adversarial attacks against LLMs have recently attracted increasing attention. We found that pre-existing adversarial attack methodologies exhibit limited transferability and are notably inefficient, particularly when applied to LLMs. In this paper, we analyze the core mechanisms of previous predominant adversarial attack methods, revealing that 1) the distributions of importance score differ markedly among victim models, restricting the transferability; 2) the sequential attack processes induces substantial time overheads. Based on the above two insights, we introduce a new scheme, named TF-Attack, for Transferable and Fast adversarial attacks on LLMs. TF-Attack employs an external LLM as a third-party overseer rather than the victim model to identify critical units within sentences. Moreover, TF-Attack introduces the concept of Importance Level, which allows for parallel substitutions of attacks. We conduct extensive experiments on 6 widely adopted benchmarks, evaluating the proposed method through both automatic and human metrics. Results show that our method consistently surpasses previous methods in transferability and delivers significant speed improvements, up to 20 times faster than earlier attack strategies.



## **14. DistillSeq: A Framework for Safety Alignment Testing in Large Language Models using Knowledge Distillation**

cs.SE

**SubmitDate**: 2024-09-08    [abs](http://arxiv.org/abs/2407.10106v4) [paper-pdf](http://arxiv.org/pdf/2407.10106v4)

**Authors**: Mingke Yang, Yuqi Chen, Yi Liu, Ling Shi

**Abstract**: Large Language Models (LLMs) have showcased their remarkable capabilities in diverse domains, encompassing natural language understanding, translation, and even code generation. The potential for LLMs to generate harmful content is a significant concern. This risk necessitates rigorous testing and comprehensive evaluation of LLMs to ensure safe and responsible use. However, extensive testing of LLMs requires substantial computational resources, making it an expensive endeavor. Therefore, exploring cost-saving strategies during the testing phase is crucial to balance the need for thorough evaluation with the constraints of resource availability. To address this, our approach begins by transferring the moderation knowledge from an LLM to a small model. Subsequently, we deploy two distinct strategies for generating malicious queries: one based on a syntax tree approach, and the other leveraging an LLM-based method. Finally, our approach incorporates a sequential filter-test process designed to identify test cases that are prone to eliciting toxic responses. Our research evaluated the efficacy of DistillSeq across four LLMs: GPT-3.5, GPT-4.0, Vicuna-13B, and Llama-13B. In the absence of DistillSeq, the observed attack success rates on these LLMs stood at 31.5% for GPT-3.5, 21.4% for GPT-4.0, 28.3% for Vicuna-13B, and 30.9% for Llama-13B. However, upon the application of DistillSeq, these success rates notably increased to 58.5%, 50.7%, 52.5%, and 54.4%, respectively. This translated to an average escalation in attack success rate by a factor of 93.0% when compared to scenarios without the use of DistillSeq. Such findings highlight the significant enhancement DistillSeq offers in terms of reducing the time and resource investment required for effectively testing LLMs.



## **15. Exploring Straightforward Conversational Red-Teaming**

cs.CL

**SubmitDate**: 2024-09-07    [abs](http://arxiv.org/abs/2409.04822v1) [paper-pdf](http://arxiv.org/pdf/2409.04822v1)

**Authors**: George Kour, Naama Zwerdling, Marcel Zalmanovici, Ateret Anaby-Tavor, Ora Nova Fandina, Eitan Farchi

**Abstract**: Large language models (LLMs) are increasingly used in business dialogue systems but they pose security and ethical risks. Multi-turn conversations, where context influences the model's behavior, can be exploited to produce undesired responses. In this paper, we examine the effectiveness of utilizing off-the-shelf LLMs in straightforward red-teaming approaches, where an attacker LLM aims to elicit undesired output from a target LLM, comparing both single-turn and conversational red-teaming tactics. Our experiments offer insights into various usage strategies that significantly affect their performance as red teamers. They suggest that off-the-shelf models can act as effective red teamers and even adjust their attack strategy based on past attempts, although their effectiveness decreases with greater alignment.



## **16. Goal-guided Generative Prompt Injection Attack on Large Language Models**

cs.CR

11 pages, 6 figures

**SubmitDate**: 2024-09-07    [abs](http://arxiv.org/abs/2404.07234v2) [paper-pdf](http://arxiv.org/pdf/2404.07234v2)

**Authors**: Chong Zhang, Mingyu Jin, Qinkai Yu, Chengzhi Liu, Haochen Xue, Xiaobo Jin

**Abstract**: Current large language models (LLMs) provide a strong foundation for large-scale user-oriented natural language tasks. A large number of users can easily inject adversarial text or instructions through the user interface, thus causing LLMs model security challenges. Although there is currently a large amount of research on prompt injection attacks, most of these black-box attacks use heuristic strategies. It is unclear how these heuristic strategies relate to the success rate of attacks and thus effectively improve model robustness. To solve this problem, we redefine the goal of the attack: to maximize the KL divergence between the conditional probabilities of the clean text and the adversarial text. Furthermore, we prove that maximizing the KL divergence is equivalent to maximizing the Mahalanobis distance between the embedded representation $x$ and $x'$ of the clean text and the adversarial text when the conditional probability is a Gaussian distribution and gives a quantitative relationship on $x$ and $x'$. Then we designed a simple and effective goal-guided generative prompt injection strategy (G2PIA) to find an injection text that satisfies specific constraints to achieve the optimal attack effect approximately. It is particularly noteworthy that our attack method is a query-free black-box attack method with low computational cost. Experimental results on seven LLM models and four datasets show the effectiveness of our attack method.



## **17. Recent Advances in Attack and Defense Approaches of Large Language Models**

cs.CR

**SubmitDate**: 2024-09-06    [abs](http://arxiv.org/abs/2409.03274v2) [paper-pdf](http://arxiv.org/pdf/2409.03274v2)

**Authors**: Jing Cui, Yishi Xu, Zhewei Huang, Shuchang Zhou, Jianbin Jiao, Junge Zhang

**Abstract**: Large Language Models (LLMs) have revolutionized artificial intelligence and machine learning through their advanced text processing and generating capabilities. However, their widespread deployment has raised significant safety and reliability concerns. Established vulnerabilities in deep neural networks, coupled with emerging threat models, may compromise security evaluations and create a false sense of security. Given the extensive research in the field of LLM security, we believe that summarizing the current state of affairs will help the research community better understand the present landscape and inform future developments. This paper reviews current research on LLM vulnerabilities and threats, and evaluates the effectiveness of contemporary defense mechanisms. We analyze recent studies on attack vectors and model weaknesses, providing insights into attack mechanisms and the evolving threat landscape. We also examine current defense strategies, highlighting their strengths and limitations. By contrasting advancements in attack and defense methodologies, we identify research gaps and propose future directions to enhance LLM security. Our goal is to advance the understanding of LLM safety challenges and guide the development of more robust security measures.



## **18. LLM-PBE: Assessing Data Privacy in Large Language Models**

cs.CR

**SubmitDate**: 2024-09-06    [abs](http://arxiv.org/abs/2408.12787v2) [paper-pdf](http://arxiv.org/pdf/2408.12787v2)

**Authors**: Qinbin Li, Junyuan Hong, Chulin Xie, Jeffrey Tan, Rachel Xin, Junyi Hou, Xavier Yin, Zhun Wang, Dan Hendrycks, Zhangyang Wang, Bo Li, Bingsheng He, Dawn Song

**Abstract**: Large Language Models (LLMs) have become integral to numerous domains, significantly advancing applications in data management, mining, and analysis. Their profound capabilities in processing and interpreting complex language data, however, bring to light pressing concerns regarding data privacy, especially the risk of unintentional training data leakage. Despite the critical nature of this issue, there has been no existing literature to offer a comprehensive assessment of data privacy risks in LLMs. Addressing this gap, our paper introduces LLM-PBE, a toolkit crafted specifically for the systematic evaluation of data privacy risks in LLMs. LLM-PBE is designed to analyze privacy across the entire lifecycle of LLMs, incorporating diverse attack and defense strategies, and handling various data types and metrics. Through detailed experimentation with multiple LLMs, LLM-PBE facilitates an in-depth exploration of data privacy concerns, shedding light on influential factors such as model size, data characteristics, and evolving temporal dimensions. This study not only enriches the understanding of privacy issues in LLMs but also serves as a vital resource for future research in the field. Aimed at enhancing the breadth of knowledge in this area, the findings, resources, and our full technical report are made available at https://llm-pbe.github.io/, providing an open platform for academic and practical advancements in LLM privacy assessment.



## **19. SelfDefend: LLMs Can Defend Themselves against Jailbreaking in a Practical Manner**

cs.CR

This paper completes its earlier vision paper, available at  arXiv:2402.15727. Updated to the latest analysis and results

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2406.05498v2) [paper-pdf](http://arxiv.org/pdf/2406.05498v2)

**Authors**: Xunguang Wang, Daoyuan Wu, Zhenlan Ji, Zongjie Li, Pingchuan Ma, Shuai Wang, Yingjiu Li, Yang Liu, Ning Liu, Juergen Rahmel

**Abstract**: Jailbreaking is an emerging adversarial attack that bypasses the safety alignment deployed in off-the-shelf large language models (LLMs) and has evolved into multiple categories: human-based, optimization-based, generation-based, and the recent indirect and multilingual jailbreaks. However, delivering a practical jailbreak defense is challenging because it needs to not only handle all the above jailbreak attacks but also incur negligible delays to user prompts, as well as be compatible with both open-source and closed-source LLMs. Inspired by how the traditional security concept of shadow stacks defends against memory overflow attacks, this paper introduces a generic LLM jailbreak defense framework called SelfDefend, which establishes a shadow LLM as a defense instance to concurrently protect the target LLM instance in the normal stack and collaborate with it for checkpoint-based access control. The effectiveness of SelfDefend builds upon our observation that existing LLMs (both target and defense LLMs) have the capability to identify harmful prompts or intentions in user queries, which we empirically validate using the commonly used GPT-3.5/4 models across all major jailbreak attacks. To further improve the defense's robustness and minimize costs, we employ a data distillation approach to tune dedicated open-source defense models. These models outperform six state-of-the-art defenses and match the performance of GPT-4-based SelfDefend, with significantly lower extra delays. We also empirically show that the tuned models are robust to adaptive jailbreaks and prompt injections.



## **20. Towards Neural Network based Cognitive Models of Dynamic Decision-Making by Humans**

cs.LG

Our code is available at https://github.com/shshnkreddy/NCM-HDM

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2407.17622v2) [paper-pdf](http://arxiv.org/pdf/2407.17622v2)

**Authors**: Changyu Chen, Shashank Reddy Chirra, Maria José Ferreira, Cleotilde Gonzalez, Arunesh Sinha, Pradeep Varakantham

**Abstract**: Modeling human cognitive processes in dynamic decision-making tasks has been an endeavor in AI for a long time because such models can help make AI systems more intuitive, personalized, mitigate any human biases, and enhance training in simulation. Some initial work has attempted to utilize neural networks (and large language models) but often assumes one common model for all humans and aims to emulate human behavior in aggregate. However, the behavior of each human is distinct, heterogeneous, and relies on specific past experiences in certain tasks. For instance, consider two individuals responding to a phishing email: one who has previously encountered and identified similar threats may recognize it quickly, while another without such experience might fall for the scam. In this work, we build on Instance Based Learning (IBL) that posits that human decisions are based on similar situations encountered in the past. However, IBL relies on simple fixed form functions to capture the mapping from past situations to current decisions. To that end, we propose two new attention-based neural network models to have open form non-linear functions to model distinct and heterogeneous human decision-making in dynamic settings. We experiment with two distinct datasets gathered from human subject experiment data, one focusing on detection of phishing email by humans and another where humans act as attackers in a cybersecurity setting and decide on an attack option. We conducted extensive experiments with our two neural network models, IBL, and GPT3.5, and demonstrate that the neural network models outperform IBL significantly in representing human decision-making, while providing similar interpretability of human decisions as IBL. Overall, our work yields promising results for further use of neural networks in cognitive modeling of human decision making.



## **21. Unleashing the potential of prompt engineering in Large Language Models: a comprehensive review**

cs.CL

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2310.14735v5) [paper-pdf](http://arxiv.org/pdf/2310.14735v5)

**Authors**: Banghao Chen, Zhaofeng Zhang, Nicolas Langrené, Shengxin Zhu

**Abstract**: This comprehensive review delves into the pivotal role of prompt engineering in unleashing the capabilities of Large Language Models (LLMs). The development of Artificial Intelligence (AI), from its inception in the 1950s to the emergence of advanced neural networks and deep learning architectures, has made a breakthrough in LLMs, with models such as GPT-4o and Claude-3, and in Vision-Language Models (VLMs), with models such as CLIP and ALIGN. Prompt engineering is the process of structuring inputs, which has emerged as a crucial technique to maximize the utility and accuracy of these models. This paper explores both foundational and advanced methodologies of prompt engineering, including techniques such as self-consistency, chain-of-thought, and generated knowledge, which significantly enhance model performance. Additionally, it examines the prompt method of VLMs through innovative approaches such as Context Optimization (CoOp), Conditional Context Optimization (CoCoOp), and Multimodal Prompt Learning (MaPLe). Critical to this discussion is the aspect of AI security, particularly adversarial attacks that exploit vulnerabilities in prompt engineering. Strategies to mitigate these risks and enhance model robustness are thoroughly reviewed. The evaluation of prompt methods is also addressed, through both subjective and objective metrics, ensuring a robust analysis of their efficacy. This review also reflects the essential role of prompt engineering in advancing AI capabilities, providing a structured framework for future research and application.



## **22. LLM Detectors Still Fall Short of Real World: Case of LLM-Generated Short News-Like Posts**

cs.CL

20 pages, 7 tables, 13 figures, under consideration for EMNLP

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2409.03291v1) [paper-pdf](http://arxiv.org/pdf/2409.03291v1)

**Authors**: Henrique Da Silva Gameiro, Andrei Kucharavy, Ljiljana Dolamic

**Abstract**: With the emergence of widely available powerful LLMs, disinformation generated by large Language Models (LLMs) has become a major concern. Historically, LLM detectors have been touted as a solution, but their effectiveness in the real world is still to be proven. In this paper, we focus on an important setting in information operations -- short news-like posts generated by moderately sophisticated attackers.   We demonstrate that existing LLM detectors, whether zero-shot or purpose-trained, are not ready for real-world use in that setting. All tested zero-shot detectors perform inconsistently with prior benchmarks and are highly vulnerable to sampling temperature increase, a trivial attack absent from recent benchmarks. A purpose-trained detector generalizing across LLMs and unseen attacks can be developed, but it fails to generalize to new human-written texts.   We argue that the former indicates domain-specific benchmarking is needed, while the latter suggests a trade-off between the adversarial evasion resilience and overfitting to the reference human text, with both needing evaluation in benchmarks and currently absent. We believe this suggests a re-consideration of current LLM detector benchmarking approaches and provides a dynamically extensible benchmark to allow it (https://github.com/Reliable-Information-Lab-HEVS/dynamic_llm_detector_benchmark).



## **23. Revisiting Character-level Adversarial Attacks for Language Models**

cs.LG

Accepted in ICML 2024

**SubmitDate**: 2024-09-04    [abs](http://arxiv.org/abs/2405.04346v2) [paper-pdf](http://arxiv.org/pdf/2405.04346v2)

**Authors**: Elias Abad Rocamora, Yongtao Wu, Fanghui Liu, Grigorios G. Chrysos, Volkan Cevher

**Abstract**: Adversarial attacks in Natural Language Processing apply perturbations in the character or token levels. Token-level attacks, gaining prominence for their use of gradient-based methods, are susceptible to altering sentence semantics, leading to invalid adversarial examples. While character-level attacks easily maintain semantics, they have received less attention as they cannot easily adopt popular gradient-based methods, and are thought to be easy to defend. Challenging these beliefs, we introduce Charmer, an efficient query-based adversarial attack capable of achieving high attack success rate (ASR) while generating highly similar adversarial examples. Our method successfully targets both small (BERT) and large (Llama 2) models. Specifically, on BERT with SST-2, Charmer improves the ASR in 4.84% points and the USE similarity in 8% points with respect to the previous art. Our implementation is available in https://github.com/LIONS-EPFL/Charmer.



## **24. Alignment-Aware Model Extraction Attacks on Large Language Models**

cs.CR

Source code: https://github.com/liangzid/alignmentExtraction

**SubmitDate**: 2024-09-04    [abs](http://arxiv.org/abs/2409.02718v1) [paper-pdf](http://arxiv.org/pdf/2409.02718v1)

**Authors**: Zi Liang, Qingqing Ye, Yanyun Wang, Sen Zhang, Yaxin Xiao, Ronghua Li, Jianliang Xu, Haibo Hu

**Abstract**: Model extraction attacks (MEAs) on large language models (LLMs) have received increasing research attention lately. Existing attack methods on LLMs inherit the extraction strategies from those designed for deep neural networks (DNNs) yet neglect the inconsistency of training tasks between MEA and LLMs' alignments. As such, they result in poor attack performances. To tackle this issue, we present Locality Reinforced Distillation (LoRD), a novel model extraction attack algorithm specifically for LLMs. In particular, we design a policy-gradient-style training task, which utilizes victim models' responses as a signal to guide the crafting of preference for the local model. Theoretical analysis has shown that i) LoRD's convergence procedure in MEAs is consistent with the alignments of LLMs, and ii) LoRD can reduce query complexity while mitigating watermark protection through exploration-based stealing. Extensive experiments on domain-specific extractions demonstrate the superiority of our method by examining the extraction of various state-of-the-art commercial LLMs.



## **25. Unveiling the Vulnerability of Private Fine-Tuning in Split-Based Frameworks for Large Language Models: A Bidirectionally Enhanced Attack**

cs.CR

ACM Conference on Computer and Communications Security 2024 (CCS 24)

**SubmitDate**: 2024-09-04    [abs](http://arxiv.org/abs/2409.00960v2) [paper-pdf](http://arxiv.org/pdf/2409.00960v2)

**Authors**: Guanzhong Chen, Zhenghan Qin, Mingxin Yang, Yajie Zhou, Tao Fan, Tianyu Du, Zenglin Xu

**Abstract**: Recent advancements in pre-trained large language models (LLMs) have significantly influenced various domains. Adapting these models for specific tasks often involves fine-tuning (FT) with private, domain-specific data. However, privacy concerns keep this data undisclosed, and the computational demands for deploying LLMs pose challenges for resource-limited data holders. This has sparked interest in split learning (SL), a Model-as-a-Service (MaaS) paradigm that divides LLMs into smaller segments for distributed training and deployment, transmitting only intermediate activations instead of raw data. SL has garnered substantial interest in both industry and academia as it aims to balance user data privacy, model ownership, and resource challenges in the private fine-tuning of LLMs. Despite its privacy claims, this paper reveals significant vulnerabilities arising from the combination of SL and LLM-FT: the Not-too-far property of fine-tuning and the auto-regressive nature of LLMs. Exploiting these vulnerabilities, we propose Bidirectional Semi-white-box Reconstruction (BiSR), the first data reconstruction attack (DRA) designed to target both the forward and backward propagation processes of SL. BiSR utilizes pre-trained weights as prior knowledge, combining a learning-based attack with a bidirectional optimization-based approach for highly effective data reconstruction. Additionally, it incorporates a Noise-adaptive Mixture of Experts (NaMoE) model to enhance reconstruction performance under perturbation. We conducted systematic experiments on various mainstream LLMs and different setups, empirically demonstrating BiSR's state-of-the-art performance. Furthermore, we thoroughly examined three representative defense mechanisms, showcasing our method's capability to reconstruct private data even in the presence of these defenses.



## **26. $\textit{MMJ-Bench}$: A Comprehensive Study on Jailbreak Attacks and Defenses for Vision Language Models**

cs.CR

**SubmitDate**: 2024-09-04    [abs](http://arxiv.org/abs/2408.08464v2) [paper-pdf](http://arxiv.org/pdf/2408.08464v2)

**Authors**: Fenghua Weng, Yue Xu, Chengyan Fu, Wenjie Wang

**Abstract**: As deep learning advances, Large Language Models (LLMs) and their multimodal counterparts, Vision-Language Models (VLMs), have shown exceptional performance in many real-world tasks. However, VLMs face significant security challenges, such as jailbreak attacks, where attackers attempt to bypass the model's safety alignment to elicit harmful responses. The threat of jailbreak attacks on VLMs arises from both the inherent vulnerabilities of LLMs and the multiple information channels that VLMs process. While various attacks and defenses have been proposed, there is a notable gap in unified and comprehensive evaluations, as each method is evaluated on different dataset and metrics, making it impossible to compare the effectiveness of each method. To address this gap, we introduce \textit{MMJ-Bench}, a unified pipeline for evaluating jailbreak attacks and defense techniques for VLMs. Through extensive experiments, we assess the effectiveness of various attack methods against SoTA VLMs and evaluate the impact of defense mechanisms on both defense effectiveness and model utility for normal tasks. Our comprehensive evaluation contribute to the field by offering a unified and systematic evaluation framework and the first public-available benchmark for VLM jailbreak research. We also demonstrate several insightful findings that highlights directions for future studies.



## **27. LLM Defenses Are Not Robust to Multi-Turn Human Jailbreaks Yet**

cs.LG

**SubmitDate**: 2024-09-04    [abs](http://arxiv.org/abs/2408.15221v2) [paper-pdf](http://arxiv.org/pdf/2408.15221v2)

**Authors**: Nathaniel Li, Ziwen Han, Ian Steneker, Willow Primack, Riley Goodside, Hugh Zhang, Zifan Wang, Cristina Menghini, Summer Yue

**Abstract**: Recent large language model (LLM) defenses have greatly improved models' ability to refuse harmful queries, even when adversarially attacked. However, LLM defenses are primarily evaluated against automated adversarial attacks in a single turn of conversation, an insufficient threat model for real-world malicious use. We demonstrate that multi-turn human jailbreaks uncover significant vulnerabilities, exceeding 70% attack success rate (ASR) on HarmBench against defenses that report single-digit ASRs with automated single-turn attacks. Human jailbreaks also reveal vulnerabilities in machine unlearning defenses, successfully recovering dual-use biosecurity knowledge from unlearned models. We compile these results into Multi-Turn Human Jailbreaks (MHJ), a dataset of 2,912 prompts across 537 multi-turn jailbreaks. We publicly release MHJ alongside a compendium of jailbreak tactics developed across dozens of commercial red teaming engagements, supporting research towards stronger LLM defenses.



## **28. RACONTEUR: A Knowledgeable, Insightful, and Portable LLM-Powered Shell Command Explainer**

cs.CR

Accepted by NDSS Symposium 2025. Please cite this paper as "Jiangyi  Deng, Xinfeng Li, Yanjiao Chen, Yijie Bai, Haiqin Weng, Yan Liu, Tao Wei,  Wenyuan Xu. RACONTEUR: A Knowledgeable, Insightful, and Portable LLM-Powered  Shell Command Explainer. In the 32nd Annual Network and Distributed System  Security Symposium (NDSS 2025)."

**SubmitDate**: 2024-09-03    [abs](http://arxiv.org/abs/2409.02074v1) [paper-pdf](http://arxiv.org/pdf/2409.02074v1)

**Authors**: Jiangyi Deng, Xinfeng Li, Yanjiao Chen, Yijie Bai, Haiqin Weng, Yan Liu, Tao Wei, Wenyuan Xu

**Abstract**: Malicious shell commands are linchpins to many cyber-attacks, but may not be easy to understand by security analysts due to complicated and often disguised code structures. Advances in large language models (LLMs) have unlocked the possibility of generating understandable explanations for shell commands. However, existing general-purpose LLMs suffer from a lack of expert knowledge and a tendency to hallucinate in the task of shell command explanation. In this paper, we present Raconteur, a knowledgeable, expressive and portable shell command explainer powered by LLM. Raconteur is infused with professional knowledge to provide comprehensive explanations on shell commands, including not only what the command does (i.e., behavior) but also why the command does it (i.e., purpose). To shed light on the high-level intent of the command, we also translate the natural-language-based explanation into standard technique & tactic defined by MITRE ATT&CK, the worldwide knowledge base of cybersecurity. To enable Raconteur to explain unseen private commands, we further develop a documentation retriever to obtain relevant information from complementary documentations to assist the explanation process. We have created a large-scale dataset for training and conducted extensive experiments to evaluate the capability of Raconteur in shell command explanation. The experiments verify that Raconteur is able to provide high-quality explanations and in-depth insight of the intent of the command.



## **29. FuzzCoder: Byte-level Fuzzing Test via Large Language Model**

cs.CL

11 pages

**SubmitDate**: 2024-09-03    [abs](http://arxiv.org/abs/2409.01944v1) [paper-pdf](http://arxiv.org/pdf/2409.01944v1)

**Authors**: Liqun Yang, Jian Yang, Chaoren Wei, Guanglin Niu, Ge Zhang, Yunli Wang, Linzheng ChaI, Wanxu Xia, Hongcheng Guo, Shun Zhang, Jiaheng Liu, Yuwei Yin, Junran Peng, Jiaxin Ma, Liang Sun, Zhoujun Li

**Abstract**: Fuzzing is an important dynamic program analysis technique designed for finding vulnerabilities in complex software. Fuzzing involves presenting a target program with crafted malicious input to cause crashes, buffer overflows, memory errors, and exceptions. Crafting malicious inputs in an efficient manner is a difficult open problem and the best approaches often apply uniform random mutations to pre-existing valid inputs. In this work, we propose to adopt fine-tuned large language models (FuzzCoder) to learn patterns in the input files from successful attacks to guide future fuzzing explorations. Specifically, we develop a framework to leverage the code LLMs to guide the mutation process of inputs in fuzzing. The mutation process is formulated as the sequence-to-sequence modeling, where LLM receives a sequence of bytes and then outputs the mutated byte sequence. FuzzCoder is fine-tuned on the created instruction dataset (Fuzz-Instruct), where the successful fuzzing history is collected from the heuristic fuzzing tool. FuzzCoder can predict mutation locations and strategies locations in input files to trigger abnormal behaviors of the program. Experimental results show that FuzzCoder based on AFL (American Fuzzy Lop) gain significant improvements in terms of effective proportion of mutation (EPM) and number of crashes (NC) for various input formats including ELF, JPG, MP3, and XML.



## **30. Safeguarding AI Agents: Developing and Analyzing Safety Architectures**

cs.CR

**SubmitDate**: 2024-09-03    [abs](http://arxiv.org/abs/2409.03793v1) [paper-pdf](http://arxiv.org/pdf/2409.03793v1)

**Authors**: Ishaan Domkundwar, Mukunda N S

**Abstract**: AI agents, specifically powered by large language models, have demonstrated exceptional capabilities in various applications where precision and efficacy are necessary. However, these agents come with inherent risks, including the potential for unsafe or biased actions, vulnerability to adversarial attacks, lack of transparency, and tendency to generate hallucinations. As AI agents become more prevalent in critical sectors of the industry, the implementation of effective safety protocols becomes increasingly important. This paper addresses the critical need for safety measures in AI systems, especially ones that collaborate with human teams. We propose and evaluate three frameworks to enhance safety protocols in AI agent systems: an LLM-powered input-output filter, a safety agent integrated within the system, and a hierarchical delegation-based system with embedded safety checks. Our methodology involves implementing these frameworks and testing them against a set of unsafe agentic use cases, providing a comprehensive evaluation of their effectiveness in mitigating risks associated with AI agent deployment. We conclude that these frameworks can significantly strengthen the safety and security of AI agent systems, minimizing potential harmful actions or outputs. Our work contributes to the ongoing effort to create safe and reliable AI applications, particularly in automated operations, and provides a foundation for developing robust guardrails to ensure the responsible use of AI agents in real-world applications.



## **31. SafeEmbodAI: a Safety Framework for Mobile Robots in Embodied AI Systems**

cs.RO

**SubmitDate**: 2024-09-03    [abs](http://arxiv.org/abs/2409.01630v1) [paper-pdf](http://arxiv.org/pdf/2409.01630v1)

**Authors**: Wenxiao Zhang, Xiangrui Kong, Thomas Braunl, Jin B. Hong

**Abstract**: Embodied AI systems, including AI-powered robots that autonomously interact with the physical world, stand to be significantly advanced by Large Language Models (LLMs), which enable robots to better understand complex language commands and perform advanced tasks with enhanced comprehension and adaptability, highlighting their potential to improve embodied AI capabilities. However, this advancement also introduces safety challenges, particularly in robotic navigation tasks. Improper safety management can lead to failures in complex environments and make the system vulnerable to malicious command injections, resulting in unsafe behaviours such as detours or collisions. To address these issues, we propose \textit{SafeEmbodAI}, a safety framework for integrating mobile robots into embodied AI systems. \textit{SafeEmbodAI} incorporates secure prompting, state management, and safety validation mechanisms to secure and assist LLMs in reasoning through multi-modal data and validating responses. We designed a metric to evaluate mission-oriented exploration, and evaluations in simulated environments demonstrate that our framework effectively mitigates threats from malicious commands and improves performance in various environment settings, ensuring the safety of embodied AI systems. Notably, In complex environments with mixed obstacles, our method demonstrates a significant performance increase of 267\% compared to the baseline in attack scenarios, highlighting its robustness in challenging conditions.



## **32. Antidote: Post-fine-tuning Safety Alignment for Large Language Models against Harmful Fine-tuning**

cs.AI

**SubmitDate**: 2024-09-03    [abs](http://arxiv.org/abs/2408.09600v2) [paper-pdf](http://arxiv.org/pdf/2408.09600v2)

**Authors**: Tiansheng Huang, Gautam Bhattacharya, Pratik Joshi, Josh Kimball, Ling Liu

**Abstract**: Safety aligned Large Language Models (LLMs) are vulnerable to harmful fine-tuning attacks \cite{qi2023fine}-- a few harmful data mixed in the fine-tuning dataset can break the LLMs's safety alignment. Existing mitigation strategies include alignment stage solutions \cite{huang2024vaccine, rosati2024representation} and fine-tuning stage solutions \cite{huang2024lazy,mukhoti2023fine}. However, our evaluation shows that both categories of defenses fail \textit{when some specific training hyper-parameters are chosen} -- a large learning rate or a large number of training epochs in the fine-tuning stage can easily invalidate the defense, which however, is necessary to guarantee finetune performance. To this end, we propose Antidote, a post-fine-tuning stage solution, which remains \textbf{\textit{agnostic to the training hyper-parameters in the fine-tuning stage}}. Antidote relies on the philosophy that by removing the harmful parameters, the harmful model can be recovered from the harmful behaviors, regardless of how those harmful parameters are formed in the fine-tuning stage. With this philosophy, we introduce a one-shot pruning stage after harmful fine-tuning to remove the harmful weights that are responsible for the generation of harmful content. Despite its embarrassing simplicity, empirical results show that Antidote can reduce harmful score while maintaining accuracy on downstream tasks.Our project page is at \url{https://huangtiansheng.github.io/Antidote_gh_page/}



## **33. Membership Inference Attacks Against In-Context Learning**

cs.CR

To Appear in the ACM Conference on Computer and Communications  Security, October 14-18, 2024

**SubmitDate**: 2024-09-02    [abs](http://arxiv.org/abs/2409.01380v1) [paper-pdf](http://arxiv.org/pdf/2409.01380v1)

**Authors**: Rui Wen, Zheng Li, Michael Backes, Yang Zhang

**Abstract**: Adapting Large Language Models (LLMs) to specific tasks introduces concerns about computational efficiency, prompting an exploration of efficient methods such as In-Context Learning (ICL). However, the vulnerability of ICL to privacy attacks under realistic assumptions remains largely unexplored. In this work, we present the first membership inference attack tailored for ICL, relying solely on generated texts without their associated probabilities. We propose four attack strategies tailored to various constrained scenarios and conduct extensive experiments on four popular large language models. Empirical results show that our attacks can accurately determine membership status in most cases, e.g., 95\% accuracy advantage against LLaMA, indicating that the associated risks are much higher than those shown by existing probability-based attacks. Additionally, we propose a hybrid attack that synthesizes the strengths of the aforementioned strategies, achieving an accuracy advantage of over 95\% in most cases. Furthermore, we investigate three potential defenses targeting data, instruction, and output. Results demonstrate combining defenses from orthogonal dimensions significantly reduces privacy leakage and offers enhanced privacy assurances.



## **34. Privacy-Aware Document Visual Question Answering**

cs.CV

35 pages, 12 figures, accepted for publication at the 18th  International Conference on Document Analysis and Recognition, ICDAR 2024

**SubmitDate**: 2024-09-02    [abs](http://arxiv.org/abs/2312.10108v2) [paper-pdf](http://arxiv.org/pdf/2312.10108v2)

**Authors**: Rubèn Tito, Khanh Nguyen, Marlon Tobaben, Raouf Kerkouche, Mohamed Ali Souibgui, Kangsoo Jung, Joonas Jälkö, Vincent Poulain D'Andecy, Aurelie Joseph, Lei Kang, Ernest Valveny, Antti Honkela, Mario Fritz, Dimosthenis Karatzas

**Abstract**: Document Visual Question Answering (DocVQA) has quickly grown into a central task of document understanding. But despite the fact that documents contain sensitive or copyrighted information, none of the current DocVQA methods offers strong privacy guarantees. In this work, we explore privacy in the domain of DocVQA for the first time, highlighting privacy issues in state of the art multi-modal LLM models used for DocVQA, and explore possible solutions. Specifically, we focus on invoice processing as a realistic document understanding scenario, and propose a large scale DocVQA dataset comprising invoice documents and associated questions and answers. We employ a federated learning scheme, that reflects the real-life distribution of documents in different businesses, and we explore the use case where the data of the invoice provider is the sensitive information to be protected. We demonstrate that non-private models tend to memorise, a behaviour that can lead to exposing private information. We then evaluate baseline training schemes employing federated learning and differential privacy in this multi-modal scenario, where the sensitive information might be exposed through either or both of the two input modalities: vision (document image) or language (OCR tokens). Finally, we design attacks exploiting the memorisation effect of the model, and demonstrate their effectiveness in probing a representative DocVQA models.



## **35. MedFuzz: Exploring the Robustness of Large Language Models in Medical Question Answering**

cs.CL

9 pages, 3 figures, 2 algorithms, appendix

**SubmitDate**: 2024-09-01    [abs](http://arxiv.org/abs/2406.06573v2) [paper-pdf](http://arxiv.org/pdf/2406.06573v2)

**Authors**: Robert Osazuwa Ness, Katie Matton, Hayden Helm, Sheng Zhang, Junaid Bajwa, Carey E. Priebe, Eric Horvitz

**Abstract**: Large language models (LLM) have achieved impressive performance on medical question-answering benchmarks. However, high benchmark accuracy does not imply that the performance generalizes to real-world clinical settings. Medical question-answering benchmarks rely on assumptions consistent with quantifying LLM performance but that may not hold in the open world of the clinic. Yet LLMs learn broad knowledge that can help the LLM generalize to practical conditions regardless of unrealistic assumptions in celebrated benchmarks. We seek to quantify how well LLM medical question-answering benchmark performance generalizes when benchmark assumptions are violated. Specifically, we present an adversarial method that we call MedFuzz (for medical fuzzing). MedFuzz attempts to modify benchmark questions in ways aimed at confounding the LLM. We demonstrate the approach by targeting strong assumptions about patient characteristics presented in the MedQA benchmark. Successful "attacks" modify a benchmark item in ways that would be unlikely to fool a medical expert but nonetheless "trick" the LLM into changing from a correct to an incorrect answer. Further, we present a permutation test technique that can ensure a successful attack is statistically significant. We show how to use performance on a "MedFuzzed" benchmark, as well as individual successful attacks. The methods show promise at providing insights into the ability of an LLM to operate robustly in more realistic settings.



## **36. The Dark Side of Human Feedback: Poisoning Large Language Models via User Inputs**

cs.CL

**SubmitDate**: 2024-09-01    [abs](http://arxiv.org/abs/2409.00787v1) [paper-pdf](http://arxiv.org/pdf/2409.00787v1)

**Authors**: Bocheng Chen, Hanqing Guo, Guangjing Wang, Yuanda Wang, Qiben Yan

**Abstract**: Large Language Models (LLMs) have demonstrated great capabilities in natural language understanding and generation, largely attributed to the intricate alignment process using human feedback. While alignment has become an essential training component that leverages data collected from user queries, it inadvertently opens up an avenue for a new type of user-guided poisoning attacks. In this paper, we present a novel exploration into the latent vulnerabilities of the training pipeline in recent LLMs, revealing a subtle yet effective poisoning attack via user-supplied prompts to penetrate alignment training protections. Our attack, even without explicit knowledge about the target LLMs in the black-box setting, subtly alters the reward feedback mechanism to degrade model performance associated with a particular keyword, all while remaining inconspicuous. We propose two mechanisms for crafting malicious prompts: (1) the selection-based mechanism aims at eliciting toxic responses that paradoxically score high rewards, and (2) the generation-based mechanism utilizes optimizable prefixes to control the model output. By injecting 1\% of these specially crafted prompts into the data, through malicious users, we demonstrate a toxicity score up to two times higher when a specific trigger word is used. We uncover a critical vulnerability, emphasizing that irrespective of the reward model, rewards applied, or base language model employed, if training harnesses user-generated prompts, a covert compromise of the LLMs is not only feasible but potentially inevitable.



## **37. Automatic Pseudo-Harmful Prompt Generation for Evaluating False Refusals in Large Language Models**

cs.CL

**SubmitDate**: 2024-09-01    [abs](http://arxiv.org/abs/2409.00598v1) [paper-pdf](http://arxiv.org/pdf/2409.00598v1)

**Authors**: Bang An, Sicheng Zhu, Ruiyi Zhang, Michael-Andrei Panaitescu-Liess, Yuancheng Xu, Furong Huang

**Abstract**: Safety-aligned large language models (LLMs) sometimes falsely refuse pseudo-harmful prompts, like "how to kill a mosquito," which are actually harmless. Frequent false refusals not only frustrate users but also provoke a public backlash against the very values alignment seeks to protect. In this paper, we propose the first method to auto-generate diverse, content-controlled, and model-dependent pseudo-harmful prompts. Using this method, we construct an evaluation dataset called PHTest, which is ten times larger than existing datasets, covers more false refusal patterns, and separately labels controversial prompts. We evaluate 20 LLMs on PHTest, uncovering new insights due to its scale and labeling. Our findings reveal a trade-off between minimizing false refusals and improving safety against jailbreak attacks. Moreover, we show that many jailbreak defenses significantly increase the false refusal rates, thereby undermining usability. Our method and dataset can help developers evaluate and fine-tune safer and more usable LLMs. Our code and dataset are available at https://github.com/umd-huang-lab/FalseRefusal



## **38. Forget to Flourish: Leveraging Machine-Unlearning on Pretrained Language Models for Privacy Leakage**

cs.LG

**SubmitDate**: 2024-08-30    [abs](http://arxiv.org/abs/2408.17354v1) [paper-pdf](http://arxiv.org/pdf/2408.17354v1)

**Authors**: Md Rafi Ur Rashid, Jing Liu, Toshiaki Koike-Akino, Shagufta Mehnaz, Ye Wang

**Abstract**: Fine-tuning large language models on private data for downstream applications poses significant privacy risks in potentially exposing sensitive information. Several popular community platforms now offer convenient distribution of a large variety of pre-trained models, allowing anyone to publish without rigorous verification. This scenario creates a privacy threat, as pre-trained models can be intentionally crafted to compromise the privacy of fine-tuning datasets. In this study, we introduce a novel poisoning technique that uses model-unlearning as an attack tool. This approach manipulates a pre-trained language model to increase the leakage of private data during the fine-tuning process. Our method enhances both membership inference and data extraction attacks while preserving model utility. Experimental results across different models, datasets, and fine-tuning setups demonstrate that our attacks significantly surpass baseline performance. This work serves as a cautionary note for users who download pre-trained models from unverified sources, highlighting the potential risks involved.



## **39. Jailbreak Attacks and Defenses Against Large Language Models: A Survey**

cs.CR

**SubmitDate**: 2024-08-30    [abs](http://arxiv.org/abs/2407.04295v2) [paper-pdf](http://arxiv.org/pdf/2407.04295v2)

**Authors**: Sibo Yi, Yule Liu, Zhen Sun, Tianshuo Cong, Xinlei He, Jiaxing Song, Ke Xu, Qi Li

**Abstract**: Large Language Models (LLMs) have performed exceptionally in various text-generative tasks, including question answering, translation, code completion, etc. However, the over-assistance of LLMs has raised the challenge of "jailbreaking", which induces the model to generate malicious responses against the usage policy and society by designing adversarial prompts. With the emergence of jailbreak attack methods exploiting different vulnerabilities in LLMs, the corresponding safety alignment measures are also evolving. In this paper, we propose a comprehensive and detailed taxonomy of jailbreak attack and defense methods. For instance, the attack methods are divided into black-box and white-box attacks based on the transparency of the target model. Meanwhile, we classify defense methods into prompt-level and model-level defenses. Additionally, we further subdivide these attack and defense methods into distinct sub-classes and present a coherent diagram illustrating their relationships. We also conduct an investigation into the current evaluation methods and compare them from different perspectives. Our findings aim to inspire future research and practical implementations in safeguarding LLMs against adversarial attacks. Above all, although jailbreak remains a significant concern within the community, we believe that our work enhances the understanding of this domain and provides a foundation for developing more secure LLMs.



## **40. WET: Overcoming Paraphrasing Vulnerabilities in Embeddings-as-a-Service with Linear Transformation Watermarks**

cs.CR

Work in Progress

**SubmitDate**: 2024-08-29    [abs](http://arxiv.org/abs/2409.04459v1) [paper-pdf](http://arxiv.org/pdf/2409.04459v1)

**Authors**: Anudeex Shetty, Qiongkai Xu, Jey Han Lau

**Abstract**: Embeddings-as-a-Service (EaaS) is a service offered by large language model (LLM) developers to supply embeddings generated by LLMs. Previous research suggests that EaaS is prone to imitation attacks -- attacks that clone the underlying EaaS model by training another model on the queried embeddings. As a result, EaaS watermarks are introduced to protect the intellectual property of EaaS providers. In this paper, we first show that existing EaaS watermarks can be removed by paraphrasing when attackers clone the model. Subsequently, we propose a novel watermarking technique that involves linearly transforming the embeddings, and show that it is empirically and theoretically robust against paraphrasing.



## **41. PromptSmooth: Certifying Robustness of Medical Vision-Language Models via Prompt Learning**

cs.CV

Accepted to MICCAI 2024

**SubmitDate**: 2024-08-29    [abs](http://arxiv.org/abs/2408.16769v1) [paper-pdf](http://arxiv.org/pdf/2408.16769v1)

**Authors**: Noor Hussein, Fahad Shamshad, Muzammal Naseer, Karthik Nandakumar

**Abstract**: Medical vision-language models (Med-VLMs) trained on large datasets of medical image-text pairs and later fine-tuned for specific tasks have emerged as a mainstream paradigm in medical image analysis. However, recent studies have highlighted the susceptibility of these Med-VLMs to adversarial attacks, raising concerns about their safety and robustness. Randomized smoothing is a well-known technique for turning any classifier into a model that is certifiably robust to adversarial perturbations. However, this approach requires retraining the Med-VLM-based classifier so that it classifies well under Gaussian noise, which is often infeasible in practice. In this paper, we propose a novel framework called PromptSmooth to achieve efficient certified robustness of Med-VLMs by leveraging the concept of prompt learning. Given any pre-trained Med-VLM, PromptSmooth adapts it to handle Gaussian noise by learning textual prompts in a zero-shot or few-shot manner, achieving a delicate balance between accuracy and robustness, while minimizing the computational overhead. Moreover, PromptSmooth requires only a single model to handle multiple noise levels, which substantially reduces the computational cost compared to traditional methods that rely on training a separate model for each noise level. Comprehensive experiments based on three Med-VLMs and across six downstream datasets of various imaging modalities demonstrate the efficacy of PromptSmooth. Our code and models are available at https://github.com/nhussein/promptsmooth.



## **42. Emerging Vulnerabilities in Frontier Models: Multi-Turn Jailbreak Attacks**

cs.CR

**SubmitDate**: 2024-08-29    [abs](http://arxiv.org/abs/2409.00137v1) [paper-pdf](http://arxiv.org/pdf/2409.00137v1)

**Authors**: Tom Gibbs, Ethan Kosak-Hine, George Ingebretsen, Jason Zhang, Julius Broomfield, Sara Pieri, Reihaneh Iranmanesh, Reihaneh Rabbany, Kellin Pelrine

**Abstract**: Large language models (LLMs) are improving at an exceptional rate. However, these models are still susceptible to jailbreak attacks, which are becoming increasingly dangerous as models become increasingly powerful. In this work, we introduce a dataset of jailbreaks where each example can be input in both a single or a multi-turn format. We show that while equivalent in content, they are not equivalent in jailbreak success: defending against one structure does not guarantee defense against the other. Similarly, LLM-based filter guardrails also perform differently depending on not just the input content but the input structure. Thus, vulnerabilities of frontier models should be studied in both single and multi-turn settings; this dataset provides a tool to do so.



## **43. The Dark Side of Function Calling: Pathways to Jailbreaking Large Language Models**

cs.CR

**SubmitDate**: 2024-08-29    [abs](http://arxiv.org/abs/2407.17915v3) [paper-pdf](http://arxiv.org/pdf/2407.17915v3)

**Authors**: Zihui Wu, Haichang Gao, Jianping He, Ping Wang

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities, but their power comes with significant security considerations. While extensive research has been conducted on the safety of LLMs in chat mode, the security implications of their function calling feature have been largely overlooked. This paper uncovers a critical vulnerability in the function calling process of LLMs, introducing a novel "jailbreak function" attack method that exploits alignment discrepancies, user coercion, and the absence of rigorous safety filters. Our empirical study, conducted on six state-of-the-art LLMs including GPT-4o, Claude-3.5-Sonnet, and Gemini-1.5-pro, reveals an alarming average success rate of over 90\% for this attack. We provide a comprehensive analysis of why function calls are susceptible to such attacks and propose defensive strategies, including the use of defensive prompts. Our findings highlight the urgent need for enhanced security measures in the function calling capabilities of LLMs, contributing to the field of AI safety by identifying a previously unexplored risk, designing an effective attack method, and suggesting practical defensive measures. Our code is available at https://github.com/wooozihui/jailbreakfunction.



## **44. FRACTURED-SORRY-Bench: Framework for Revealing Attacks in Conversational Turns Undermining Refusal Efficacy and Defenses over SORRY-Bench**

cs.CL

4 pages, 2 tables

**SubmitDate**: 2024-08-28    [abs](http://arxiv.org/abs/2408.16163v1) [paper-pdf](http://arxiv.org/pdf/2408.16163v1)

**Authors**: Aman Priyanshu, Supriti Vijay

**Abstract**: This paper introduces FRACTURED-SORRY-Bench, a framework for evaluating the safety of Large Language Models (LLMs) against multi-turn conversational attacks. Building upon the SORRY-Bench dataset, we propose a simple yet effective method for generating adversarial prompts by breaking down harmful queries into seemingly innocuous sub-questions. Our approach achieves a maximum increase of +46.22\% in Attack Success Rates (ASRs) across GPT-4, GPT-4o, GPT-4o-mini, and GPT-3.5-Turbo models compared to baseline methods. We demonstrate that this technique poses a challenge to current LLM safety measures and highlights the need for more robust defenses against subtle, multi-turn attacks.



## **45. Evading AI-Generated Content Detectors using Homoglyphs**

cs.CL

**SubmitDate**: 2024-08-28    [abs](http://arxiv.org/abs/2406.11239v2) [paper-pdf](http://arxiv.org/pdf/2406.11239v2)

**Authors**: Aldan Creo, Shushanta Pudasaini

**Abstract**: The advent of large language models (LLMs) has enabled the generation of text that increasingly exhibits human-like characteristics. As the detection of such content is of significant importance, numerous studies have been conducted with the aim of developing reliable AI-generated text detectors. These detectors have demonstrated promising results on test data, but recent research has revealed that they can be circumvented by employing different techniques. In this paper, we present homoglyph-based attacks ($a \rightarrow {\alpha}$) as a means of circumventing existing detectors. A comprehensive evaluation was conducted to assess the effectiveness of these attacks on seven detectors, including ArguGPT, Binoculars, DetectGPT, Fast-DetectGPT, Ghostbuster, OpenAI's detector, and watermarking techniques, on five different datasets. Our findings demonstrate that homoglyph-based attacks can effectively circumvent state-of-the-art detectors, leading them to classify all texts as either AI-generated or human-written (decreasing the average Matthews Correlation Coefficient from 0.64 to -0.01). We then examine the effectiveness of these attacks by analyzing how homoglyphs impact different families of detectors. Finally, we discuss the implications of these findings and potential defenses against such attacks.



## **46. Large Language Model Sentinel: LLM Agent for Adversarial Purification**

cs.CL

**SubmitDate**: 2024-08-28    [abs](http://arxiv.org/abs/2405.20770v3) [paper-pdf](http://arxiv.org/pdf/2405.20770v3)

**Authors**: Guang Lin, Qibin Zhao

**Abstract**: Over the past two years, the use of large language models (LLMs) has advanced rapidly. While these LLMs offer considerable convenience, they also raise security concerns, as LLMs are vulnerable to adversarial attacks by some well-designed textual perturbations. In this paper, we introduce a novel defense technique named Large LAnguage MOdel Sentinel (LLAMOS), which is designed to enhance the adversarial robustness of LLMs by purifying the adversarial textual examples before feeding them into the target LLM. Our method comprises two main components: a) Agent instruction, which can simulate a new agent for adversarial defense, altering minimal characters to maintain the original meaning of the sentence while defending against attacks; b) Defense guidance, which provides strategies for modifying clean or adversarial examples to ensure effective defense and accurate outputs from the target LLMs. Remarkably, the defense agent demonstrates robust defensive capabilities even without learning from adversarial examples. Additionally, we conduct an intriguing adversarial experiment where we develop two agents, one for defense and one for attack, and engage them in mutual confrontation. During the adversarial interactions, neither agent completely beat the other. Extensive experiments on both open-source and closed-source LLMs demonstrate that our method effectively defends against adversarial attacks, thereby enhancing adversarial robustness.



## **47. Investigating Coverage Criteria in Large Language Models: An In-Depth Study Through Jailbreak Attacks**

cs.SE

**SubmitDate**: 2024-08-27    [abs](http://arxiv.org/abs/2408.15207v1) [paper-pdf](http://arxiv.org/pdf/2408.15207v1)

**Authors**: Shide Zhou, Tianlin Li, Kailong Wang, Yihao Huang, Ling Shi, Yang Liu, Haoyu Wang

**Abstract**: The swift advancement of large language models (LLMs) has profoundly shaped the landscape of artificial intelligence; however, their deployment in sensitive domains raises grave concerns, particularly due to their susceptibility to malicious exploitation. This situation underscores the insufficiencies in pre-deployment testing, highlighting the urgent need for more rigorous and comprehensive evaluation methods. This study presents a comprehensive empirical analysis assessing the efficacy of conventional coverage criteria in identifying these vulnerabilities, with a particular emphasis on the pressing issue of jailbreak attacks. Our investigation begins with a clustering analysis of the hidden states in LLMs, demonstrating that intrinsic characteristics of these states can distinctly differentiate between various types of queries. Subsequently, we assess the performance of these criteria across three critical dimensions: criterion level, layer level, and token level. Our findings uncover significant disparities in neuron activation patterns between the processing of normal and jailbreak queries, thereby corroborating the clustering results. Leveraging these findings, we propose an innovative approach for the real-time detection of jailbreak attacks by utilizing neural activation features. Our classifier demonstrates remarkable accuracy, averaging 96.33% in identifying jailbreak queries, including those that could lead to adversarial attacks. The importance of our research lies in its comprehensive approach to addressing the intricate challenges of LLM security. By enabling instantaneous detection from the model's first token output, our method holds promise for future systems integrating LLMs, offering robust real-time detection capabilities. This study advances our understanding of LLM security testing, and lays a critical foundation for the development of more resilient AI systems.



## **48. Detecting AI Flaws: Target-Driven Attacks on Internal Faults in Language Models**

cs.CL

**SubmitDate**: 2024-08-27    [abs](http://arxiv.org/abs/2408.14853v1) [paper-pdf](http://arxiv.org/pdf/2408.14853v1)

**Authors**: Yuhao Du, Zhuo Li, Pengyu Cheng, Xiang Wan, Anningzhe Gao

**Abstract**: Large Language Models (LLMs) have become a focal point in the rapidly evolving field of artificial intelligence. However, a critical concern is the presence of toxic content within the pre-training corpus of these models, which can lead to the generation of inappropriate outputs. Investigating methods for detecting internal faults in LLMs can help us understand their limitations and improve their security. Existing methods primarily focus on jailbreaking attacks, which involve manually or automatically constructing adversarial content to prompt the target LLM to generate unexpected responses. These methods rely heavily on prompt engineering, which is time-consuming and usually requires specially designed questions. To address these challenges, this paper proposes a target-driven attack paradigm that focuses on directly eliciting the target response instead of optimizing the prompts. We introduce the use of another LLM as the detector for toxic content, referred to as ToxDet. Given a target toxic response, ToxDet can generate a possible question and a preliminary answer to provoke the target model into producing desired toxic responses with meanings equivalent to the provided one. ToxDet is trained by interacting with the target LLM and receiving reward signals from it, utilizing reinforcement learning for the optimization process. While the primary focus of the target models is on open-source LLMs, the fine-tuned ToxDet can also be transferred to attack black-box models such as GPT-4o, achieving notable results. Experimental results on AdvBench and HH-Harmless datasets demonstrate the effectiveness of our methods in detecting the tendencies of target LLMs to generate harmful responses. This algorithm not only exposes vulnerabilities but also provides a valuable resource for researchers to strengthen their models against such attacks.



## **49. Image-to-Text Logic Jailbreak: Your Imagination can Help You Do Anything**

cs.CR

**SubmitDate**: 2024-08-26    [abs](http://arxiv.org/abs/2407.02534v2) [paper-pdf](http://arxiv.org/pdf/2407.02534v2)

**Authors**: Xiaotian Zou, Ke Li, Yongkang Chen

**Abstract**: Large Visual Language Model\textbfs (VLMs) such as GPT-4V have achieved remarkable success in generating comprehensive and nuanced responses. Researchers have proposed various benchmarks for evaluating the capabilities of VLMs. With the integration of visual and text inputs in VLMs, new security issues emerge, as malicious attackers can exploit multiple modalities to achieve their objectives. This has led to increasing attention on the vulnerabilities of VLMs to jailbreak. Most existing research focuses on generating adversarial images or nonsensical image to jailbreak these models. However, no researchers evaluate whether logic understanding capabilities of VLMs in flowchart can influence jailbreak. Therefore, to fill this gap, this paper first introduces a novel dataset Flow-JD specifically designed to evaluate the logic-based flowchart jailbreak capabilities of VLMs. We conduct an extensive evaluation on GPT-4o, GPT-4V, other 5 SOTA open source VLMs and the jailbreak rate is up to 92.8%. Our research reveals significant vulnerabilities in current VLMs concerning image-to-text jailbreak and these findings underscore the the urgency for the development of robust and effective future defenses.



## **50. Investigating the Effectiveness of Bayesian Spam Filters in Detecting LLM-modified Spam Mails**

cs.CR

EAI International Conference on Digital Forensics & Cyber Crime 2024

**SubmitDate**: 2024-08-26    [abs](http://arxiv.org/abs/2408.14293v1) [paper-pdf](http://arxiv.org/pdf/2408.14293v1)

**Authors**: Malte Josten, Torben Weis

**Abstract**: Spam and phishing remain critical threats in cybersecurity, responsible for nearly 90% of security incidents. As these attacks grow in sophistication, the need for robust defensive mechanisms intensifies. Bayesian spam filters, like the widely adopted open-source SpamAssassin, are essential tools in this fight. However, the emergence of large language models (LLMs) such as ChatGPT presents new challenges. These models are not only powerful and accessible, but also inexpensive to use, raising concerns about their misuse in crafting sophisticated spam emails that evade traditional spam filters. This work aims to evaluate the robustness and effectiveness of SpamAssassin against LLM-modified email content. We developed a pipeline to test this vulnerability. Our pipeline modifies spam emails using GPT-3.5 Turbo and assesses SpamAssassin's ability to classify these modified emails correctly. The results show that SpamAssassin misclassified up to 73.7% of LLM-modified spam emails as legitimate. In contrast, a simpler dictionary-replacement attack showed a maximum success rate of only 0.4%. These findings highlight the significant threat posed by LLM-modified spam, especially given the cost-efficiency of such attacks (0.17 cents per email). This paper provides crucial insights into the vulnerabilities of current spam filters and the need for continuous improvement in cybersecurity measures.



