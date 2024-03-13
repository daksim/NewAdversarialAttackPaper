# Latest Large Language Model Attack Papers
**update at 2024-03-13 09:59:24**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal Large Language Models**

cs.CV

**SubmitDate**: 2024-03-12    [abs](http://arxiv.org/abs/2311.17600v2) [paper-pdf](http://arxiv.org/pdf/2311.17600v2)

**Authors**: Xin Liu, Yichen Zhu, Jindong Gu, Yunshi Lan, Chao Yang, Yu Qiao

**Abstract**: The security concerns surrounding Large Language Models (LLMs) have been extensively explored, yet the safety of Multimodal Large Language Models (MLLMs) remains understudied. In this paper, we observe that Multimodal Large Language Models (MLLMs) can be easily compromised by query-relevant images, as if the text query itself were malicious. To address this, we introduce MM-SafetyBench, a comprehensive framework designed for conducting safety-critical evaluations of MLLMs against such image-based manipulations. We have compiled a dataset comprising 13 scenarios, resulting in a total of 5,040 text-image pairs. Our analysis across 12 state-of-the-art models reveals that MLLMs are susceptible to breaches instigated by our approach, even when the equipped LLMs have been safety-aligned. In response, we propose a straightforward yet effective prompting strategy to enhance the resilience of MLLMs against these types of attacks. Our work underscores the need for a concerted effort to strengthen and enhance the safety measures of open-source MLLMs against potential malicious exploits. The resource is available at \href{this https URL}{https://github.com/isXinLiu/MM-SafetyBench}.



## **2. Poisoning Programs by Un-Repairing Code: Security Concerns of AI-generated Code**

cs.CR

Accepted at The 1st IEEE International Workshop on Reliable and  Secure AI for Software Engineering (ReSAISE), co-located with ISSRE 2023

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.06675v1) [paper-pdf](http://arxiv.org/pdf/2403.06675v1)

**Authors**: Cristina Improta

**Abstract**: AI-based code generators have gained a fundamental role in assisting developers in writing software starting from natural language (NL). However, since these large language models are trained on massive volumes of data collected from unreliable online sources (e.g., GitHub, Hugging Face), AI models become an easy target for data poisoning attacks, in which an attacker corrupts the training data by injecting a small amount of poison into it, i.e., astutely crafted malicious samples. In this position paper, we address the security of AI code generators by identifying a novel data poisoning attack that results in the generation of vulnerable code. Next, we devise an extensive evaluation of how these attacks impact state-of-the-art models for code generation. Lastly, we discuss potential solutions to overcome this threat.



## **3. FedPIT: Towards Privacy-preserving and Few-shot Federated Instruction Tuning**

cs.CR

Work in process

**SubmitDate**: 2024-03-10    [abs](http://arxiv.org/abs/2403.06131v1) [paper-pdf](http://arxiv.org/pdf/2403.06131v1)

**Authors**: Zhuo Zhang, Jingyuan Zhang, Jintao Huang, Lizhen Qu, Hongzhi Zhang, Zenglin Xu

**Abstract**: Instruction tuning has proven essential for enhancing the performance of large language models (LLMs) in generating human-aligned responses. However, collecting diverse, high-quality instruction data for tuning poses challenges, particularly in privacy-sensitive domains. Federated instruction tuning (FedIT) has emerged as a solution, leveraging federated learning from multiple data owners while preserving privacy. Yet, it faces challenges due to limited instruction data and vulnerabilities to training data extraction attacks. To address these issues, we propose a novel federated algorithm, FedPIT, which utilizes LLMs' in-context learning capability to self-generate task-specific synthetic data for training autonomously. Our method employs parameter-isolated training to maintain global parameters trained on synthetic data and local parameters trained on augmented local data, effectively thwarting data extraction attacks. Extensive experiments on real-world medical data demonstrate the effectiveness of FedPIT in improving federated few-shot performance while preserving privacy and robustness against data heterogeneity.



## **4. Language-Driven Anchors for Zero-Shot Adversarial Robustness**

cs.CV

Accepted by CVPR 2024

**SubmitDate**: 2024-03-10    [abs](http://arxiv.org/abs/2301.13096v3) [paper-pdf](http://arxiv.org/pdf/2301.13096v3)

**Authors**: Xiao Li, Wei Zhang, Yining Liu, Zhanhao Hu, Bo Zhang, Xiaolin Hu

**Abstract**: Deep Neural Networks (DNNs) are known to be susceptible to adversarial attacks. Previous researches mainly focus on improving adversarial robustness in the fully supervised setting, leaving the challenging domain of zero-shot adversarial robustness an open question. In this work, we investigate this domain by leveraging the recent advances in large vision-language models, such as CLIP, to introduce zero-shot adversarial robustness to DNNs. We propose LAAT, a Language-driven, Anchor-based Adversarial Training strategy. LAAT utilizes the features of a text encoder for each category as fixed anchors (normalized feature embeddings) for each category, which are then employed for adversarial training. By leveraging the semantic consistency of the text encoders, LAAT aims to enhance the adversarial robustness of the image model on novel categories. However, naively using text encoders leads to poor results. Through analysis, we identified the issue to be the high cosine similarity between text encoders. We then design an expansion algorithm and an alignment cross-entropy loss to alleviate the problem. Our experimental results demonstrated that LAAT significantly improves zero-shot adversarial robustness over state-of-the-art methods. LAAT has the potential to enhance adversarial robustness by large-scale multimodal models, especially when labeled data is unavailable during training.



## **5. From Chatbots to PhishBots? -- Preventing Phishing scams created using ChatGPT, Google Bard and Claude**

cs.CR

**SubmitDate**: 2024-03-10    [abs](http://arxiv.org/abs/2310.19181v2) [paper-pdf](http://arxiv.org/pdf/2310.19181v2)

**Authors**: Sayak Saha Roy, Poojitha Thota, Krishna Vamsi Naragam, Shirin Nilizadeh

**Abstract**: The advanced capabilities of Large Language Models (LLMs) have made them invaluable across various applications, from conversational agents and content creation to data analysis, research, and innovation. However, their effectiveness and accessibility also render them susceptible to abuse for generating malicious content, including phishing attacks. This study explores the potential of using four popular commercially available LLMs, i.e., ChatGPT (GPT 3.5 Turbo), GPT 4, Claude, and Bard, to generate functional phishing attacks using a series of malicious prompts. We discover that these LLMs can generate both phishing websites and emails that can convincingly imitate well-known brands and also deploy a range of evasive tactics that are used to elude detection mechanisms employed by anti-phishing systems. These attacks can be generated using unmodified or "vanilla" versions of these LLMs without requiring any prior adversarial exploits such as jailbreaking. We evaluate the performance of the LLMs towards generating these attacks and find that they can also be utilized to create malicious prompts that, in turn, can be fed back to the model to generate phishing scams - thus massively reducing the prompt-engineering effort required by attackers to scale these threats. As a countermeasure, we build a BERT-based automated detection tool that can be used for the early detection of malicious prompts to prevent LLMs from generating phishing content. Our model is transferable across all four commercial LLMs, attaining an average accuracy of 96% for phishing website prompts and 94% for phishing email prompts. We also disclose the vulnerabilities to the concerned LLMs, with Google acknowledging it as a severe issue. Our detection model is available for use at Hugging Face, as well as a ChatGPT Actions plugin.



## **6. Can LLMs Follow Simple Rules?**

cs.AI

Project website: https://eecs.berkeley.edu/~normanmu/llm_rules;  revised content

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2311.04235v3) [paper-pdf](http://arxiv.org/pdf/2311.04235v3)

**Authors**: Norman Mu, Sarah Chen, Zifan Wang, Sizhe Chen, David Karamardian, Lulwa Aljeraisy, Basel Alomair, Dan Hendrycks, David Wagner

**Abstract**: As Large Language Models (LLMs) are deployed with increasing real-world responsibilities, it is important to be able to specify and constrain the behavior of these systems in a reliable manner. Model developers may wish to set explicit rules for the model, such as "do not generate abusive content", but these may be circumvented by jailbreaking techniques. Existing evaluations of adversarial attacks and defenses on LLMs generally require either expensive manual review or unreliable heuristic checks. To address this issue, we propose Rule-following Language Evaluation Scenarios (RuLES), a programmatic framework for measuring rule-following ability in LLMs. RuLES consists of 14 simple text scenarios in which the model is instructed to obey various rules while interacting with the user. Each scenario has a programmatic evaluation function to determine whether the model has broken any rules in a conversation. Our evaluations of proprietary and open models show that almost all current models struggle to follow scenario rules, even on straightforward test cases. We also demonstrate that simple optimization attacks suffice to significantly increase failure rates on test cases. We conclude by exploring two potential avenues for improvement: test-time steering and supervised fine-tuning.



## **7. Warfare:Breaking the Watermark Protection of AI-Generated Content**

cs.CV

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2310.07726v3) [paper-pdf](http://arxiv.org/pdf/2310.07726v3)

**Authors**: Guanlin Li, Yifei Chen, Jie Zhang, Jiwei Li, Shangwei Guo, Tianwei Zhang

**Abstract**: AI-Generated Content (AIGC) is gaining great popularity, with many emerging commercial services and applications. These services leverage advanced generative models, such as latent diffusion models and large language models, to generate creative content (e.g., realistic images and fluent sentences) for users. The usage of such generated content needs to be highly regulated, as the service providers need to ensure the users do not violate the usage policies (e.g., abuse for commercialization, generating and distributing unsafe content). A promising solution to achieve this goal is watermarking, which adds unique and imperceptible watermarks on the content for service verification and attribution. Numerous watermarking approaches have been proposed recently. However, in this paper, we show that an adversary can easily break these watermarking mechanisms. Specifically, we consider two possible attacks. (1) Watermark removal: the adversary can easily erase the embedded watermark from the generated content and then use it freely bypassing the regulation of the service provider. (2) Watermark forging: the adversary can create illegal content with forged watermarks from another user, causing the service provider to make wrong attributions. We propose Warfare, a unified methodology to achieve both attacks in a holistic way. The key idea is to leverage a pre-trained diffusion model for content processing and a generative adversarial network for watermark removal or forging. We evaluate Warfare on different datasets and embedding setups. The results prove that it can achieve high success rates while maintaining the quality of the generated content. Compared to existing diffusion model-based attacks, Warfare is 5,050~11,000x faster.



## **8. On Protecting the Data Privacy of Large Language Models (LLMs): A Survey**

cs.CR

18 pages, 4 figures

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2403.05156v1) [paper-pdf](http://arxiv.org/pdf/2403.05156v1)

**Authors**: Biwei Yan, Kun Li, Minghui Xu, Yueyan Dong, Yue Zhang, Zhaochun Ren, Xiuzheng Cheng

**Abstract**: Large language models (LLMs) are complex artificial intelligence systems capable of understanding, generating and translating human language. They learn language patterns by analyzing large amounts of text data, allowing them to perform writing, conversation, summarizing and other language tasks. When LLMs process and generate large amounts of data, there is a risk of leaking sensitive information, which may threaten data privacy. This paper concentrates on elucidating the data privacy concerns associated with LLMs to foster a comprehensive understanding. Specifically, a thorough investigation is undertaken to delineate the spectrum of data privacy threats, encompassing both passive privacy leakage and active privacy attacks within LLMs. Subsequently, we conduct an assessment of the privacy protection mechanisms employed by LLMs at various stages, followed by a detailed examination of their efficacy and constraints. Finally, the discourse extends to delineate the challenges encountered and outline prospective directions for advancement in the realm of LLM privacy protection.



## **9. Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models**

cs.CL

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2312.14197v3) [paper-pdf](http://arxiv.org/pdf/2312.14197v3)

**Authors**: Jingwei Yi, Yueqi Xie, Bin Zhu, Emre Kiciman, Guangzhong Sun, Xing Xie, Fangzhao Wu

**Abstract**: The integration of large language models (LLMs) with external content has enabled more up-to-date and wide-ranging applications of LLMs, such as Microsoft Copilot. However, this integration has also exposed LLMs to the risk of indirect prompt injection attacks, where an attacker can embed malicious instructions within external content, compromising LLM output and causing responses to deviate from user expectations. To investigate this important but underexplored issue, we introduce the first benchmark for indirect prompt injection attacks, named BIPIA, to evaluate the risk of such attacks. Based on the evaluation, our work makes a key analysis of the underlying reason for the success of the attack, namely the inability of LLMs to distinguish between instructions and external content and the absence of LLMs' awareness to not execute instructions within external content. Building upon this analysis, we develop two black-box methods based on prompt learning and a white-box defense method based on fine-tuning with adversarial training accordingly. Experimental results demonstrate that black-box defenses are highly effective in mitigating these attacks, while the white-box defense reduces the attack success rate to near-zero levels. Overall, our work systematically investigates indirect prompt injection attacks by introducing a benchmark, analyzing the underlying reason for the success of the attack, and developing an initial set of defenses.



## **10. SecGPT: An Execution Isolation Architecture for LLM-Based Systems**

cs.CR

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2403.04960v1) [paper-pdf](http://arxiv.org/pdf/2403.04960v1)

**Authors**: Yuhao Wu, Franziska Roesner, Tadayoshi Kohno, Ning Zhang, Umar Iqbal

**Abstract**: Large language models (LLMs) extended as systems, such as ChatGPT, have begun supporting third-party applications. These LLM apps leverage the de facto natural language-based automated execution paradigm of LLMs: that is, apps and their interactions are defined in natural language, provided access to user data, and allowed to freely interact with each other and the system. These LLM app ecosystems resemble the settings of earlier computing platforms, where there was insufficient isolation between apps and the system. Because third-party apps may not be trustworthy, and exacerbated by the imprecision of the natural language interfaces, the current designs pose security and privacy risks for users. In this paper, we propose SecGPT, an architecture for LLM-based systems that aims to mitigate the security and privacy issues that arise with the execution of third-party apps. SecGPT's key idea is to isolate the execution of apps and more precisely mediate their interactions outside of their isolated environments. We evaluate SecGPT against a number of case study attacks and demonstrate that it protects against many security, privacy, and safety issues that exist in non-isolated LLM-based systems. The performance overhead incurred by SecGPT to improve security is under 0.3x for three-quarters of the tested queries. To foster follow-up research, we release SecGPT's source code at https://github.com/llm-platform-security/SecGPT.



## **11. Automatic and Universal Prompt Injection Attacks against Large Language Models**

cs.AI

Pre-print, code is available at  https://github.com/SheltonLiu-N/Universal-Prompt-Injection

**SubmitDate**: 2024-03-07    [abs](http://arxiv.org/abs/2403.04957v1) [paper-pdf](http://arxiv.org/pdf/2403.04957v1)

**Authors**: Xiaogeng Liu, Zhiyuan Yu, Yizhe Zhang, Ning Zhang, Chaowei Xiao

**Abstract**: Large Language Models (LLMs) excel in processing and generating human language, powered by their ability to interpret and follow instructions. However, their capabilities can be exploited through prompt injection attacks. These attacks manipulate LLM-integrated applications into producing responses aligned with the attacker's injected content, deviating from the user's actual requests. The substantial risks posed by these attacks underscore the need for a thorough understanding of the threats. Yet, research in this area faces challenges due to the lack of a unified goal for such attacks and their reliance on manually crafted prompts, complicating comprehensive assessments of prompt injection robustness. We introduce a unified framework for understanding the objectives of prompt injection attacks and present an automated gradient-based method for generating highly effective and universal prompt injection data, even in the face of defensive measures. With only five training samples (0.3% relative to the test data), our attack can achieve superior performance compared with baselines. Our findings emphasize the importance of gradient-based testing, which can avoid overestimation of robustness, especially for defense mechanisms.



## **12. Membership Inference Attacks and Privacy in Topic Modeling**

cs.CR

9 pages + appendices and references. 9 figures. Submitted to USENIX  '24

**SubmitDate**: 2024-03-07    [abs](http://arxiv.org/abs/2403.04451v1) [paper-pdf](http://arxiv.org/pdf/2403.04451v1)

**Authors**: Nico Manzonelli, Wanrong Zhang, Salil Vadhan

**Abstract**: Recent research shows that large language models are susceptible to privacy attacks that infer aspects of the training data. However, it is unclear if simpler generative models, like topic models, share similar vulnerabilities. In this work, we propose an attack against topic models that can confidently identify members of the training data in Latent Dirichlet Allocation. Our results suggest that the privacy risks associated with generative modeling are not restricted to large neural models. Additionally, to mitigate these vulnerabilities, we explore differentially private (DP) topic modeling. We propose a framework for private topic modeling that incorporates DP vocabulary selection as a pre-processing step, and show that it improves privacy while having limited effects on practical utility.



## **13. PPTC-R benchmark: Towards Evaluating the Robustness of Large Language Models for PowerPoint Task Completion**

cs.CL

LLM evaluation, Multi-turn, Multi-language, Multi-modal benchmark

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.03788v1) [paper-pdf](http://arxiv.org/pdf/2403.03788v1)

**Authors**: Zekai Zhang, Yiduo Guo, Yaobo Liang, Dongyan Zhao, Nan Duan

**Abstract**: The growing dependence on Large Language Models (LLMs) for finishing user instructions necessitates a comprehensive understanding of their robustness to complex task completion in real-world situations. To address this critical need, we propose the PowerPoint Task Completion Robustness benchmark (PPTC-R) to measure LLMs' robustness to the user PPT task instruction and software version. Specifically, we construct adversarial user instructions by attacking user instructions at sentence, semantic, and multi-language levels. To assess the robustness of Language Models to software versions, we vary the number of provided APIs to simulate both the newest version and earlier version settings. Subsequently, we test 3 closed-source and 4 open-source LLMs using a benchmark that incorporates these robustness settings, aiming to evaluate how deviations impact LLMs' API calls for task completion. We find that GPT-4 exhibits the highest performance and strong robustness in our benchmark, particularly in the version update and the multilingual settings. However, we find that all LLMs lose their robustness when confronted with multiple challenges (e.g., multi-turn) simultaneously, leading to significant performance drops. We further analyze the robustness behavior and error reasons of LLMs in our benchmark, which provide valuable insights for researchers to understand the LLM's robustness in task completion and develop more robust LLMs and agents. We release the code and data at \url{https://github.com/ZekaiGalaxy/PPTCR}.



## **14. ImgTrojan: Jailbreaking Vision-Language Models with ONE Image**

cs.CV

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.02910v2) [paper-pdf](http://arxiv.org/pdf/2403.02910v2)

**Authors**: Xijia Tao, Shuai Zhong, Lei Li, Qi Liu, Lingpeng Kong

**Abstract**: There has been an increasing interest in the alignment of large language models (LLMs) with human values. However, the safety issues of their integration with a vision module, or vision language models (VLMs), remain relatively underexplored. In this paper, we propose a novel jailbreaking attack against VLMs, aiming to bypass their safety barrier when a user inputs harmful instructions. A scenario where our poisoned (image, text) data pairs are included in the training data is assumed. By replacing the original textual captions with malicious jailbreak prompts, our method can perform jailbreak attacks with the poisoned images. Moreover, we analyze the effect of poison ratios and positions of trainable parameters on our attack's success rate. For evaluation, we design two metrics to quantify the success rate and the stealthiness of our attack. Together with a list of curated harmful instructions, a benchmark for measuring attack efficacy is provided. We demonstrate the efficacy of our attack by comparing it with baseline methods.



## **15. Human vs. Machine: Language Models and Wargames**

cs.CY

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.03407v1) [paper-pdf](http://arxiv.org/pdf/2403.03407v1)

**Authors**: Max Lamparth, Anthony Corso, Jacob Ganz, Oriana Skylar Mastro, Jacquelyn Schneider, Harold Trinkunas

**Abstract**: Wargames have a long history in the development of military strategy and the response of nations to threats or attacks. The advent of artificial intelligence (AI) promises better decision-making and increased military effectiveness. However, there is still debate about how AI systems, especially large language models (LLMs), behave as compared to humans. To this end, we use a wargame experiment with 107 national security expert human players designed to look at crisis escalation in a fictional US-China scenario and compare human players to LLM-simulated responses. We find considerable agreement in the LLM and human responses but also significant quantitative and qualitative differences between simulated and human players in the wargame, motivating caution to policymakers before handing over autonomy or following AI-based strategy recommendations.



## **16. PETA: Parameter-Efficient Trojan Attacks**

cs.CL

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2310.00648v4) [paper-pdf](http://arxiv.org/pdf/2310.00648v4)

**Authors**: Lauren Hong, Ting Wang

**Abstract**: Parameter-efficient fine-tuning (PEFT) enables efficient adaptation of pre-trained language models (PLMs) to specific tasks. By tuning only a minimal set of (extra) parameters, PEFT achieves performance that is comparable to standard fine-tuning. However, despite its prevalent use, the security implications of PEFT remain largely unexplored. In this paper, we take the initial steps and present PETA, a novel trojan attack that compromises the weights of PLMs by accounting for downstream adaptation through bilevel optimization: the upper-level objective embeds the backdoor into a model while the lower-level objective simulates PEFT to both retain the PLM's task-specific performance and ensure that the backdoor persists after fine-tuning. With extensive evaluation across a variety of downstream tasks and trigger designs, we demonstrate PETA's effectiveness in terms of both attack success rate and clean accuracy, even when the attacker does not have full knowledge of the victim user's training process.



## **17. Gradient Cuff: Detecting Jailbreak Attacks on Large Language Models by Exploring Refusal Loss Landscapes**

cs.CR

Project page:  https://huggingface.co/spaces/TrustSafeAI/GradientCuff-Jailbreak-Defense

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.00867v2) [paper-pdf](http://arxiv.org/pdf/2403.00867v2)

**Authors**: Xiaomeng Hu, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Large Language Models (LLMs) are becoming a prominent generative AI tool, where the user enters a query and the LLM generates an answer. To reduce harm and misuse, efforts have been made to align these LLMs to human values using advanced training techniques such as Reinforcement Learning from Human Feedback (RLHF). However, recent studies have highlighted the vulnerability of LLMs to adversarial jailbreak attempts aiming at subverting the embedded safety guardrails. To address this challenge, this paper defines and investigates the Refusal Loss of LLMs and then proposes a method called Gradient Cuff to detect jailbreak attempts. Gradient Cuff exploits the unique properties observed in the refusal loss landscape, including functional values and its smoothness, to design an effective two-step detection strategy. Experimental results on two aligned LLMs (LLaMA-2-7B-Chat and Vicuna-7B-V1.5) and six types of jailbreak attacks (GCG, AutoDAN, PAIR, TAP, Base64, and LRL) show that Gradient Cuff can significantly improve the LLM's rejection capability for malicious jailbreak queries, while maintaining the model's performance for benign user queries by adjusting the detection threshold.



## **18. Measuring Impacts of Poisoning on Model Parameters and Neuron Activations: A Case Study of Poisoning CodeBERT**

cs.SE

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2402.12936v2) [paper-pdf](http://arxiv.org/pdf/2402.12936v2)

**Authors**: Aftab Hussain, Md Rafiqul Islam Rabin, Navid Ayoobi, Mohammad Amin Alipour

**Abstract**: Large language models (LLMs) have revolutionized software development practices, yet concerns about their safety have arisen, particularly regarding hidden backdoors, aka trojans. Backdoor attacks involve the insertion of triggers into training data, allowing attackers to manipulate the behavior of the model maliciously. In this paper, we focus on analyzing the model parameters to detect potential backdoor signals in code models. Specifically, we examine attention weights and biases, activation values, and context embeddings of the clean and poisoned CodeBERT models. Our results suggest noticeable patterns in activation values and context embeddings of poisoned samples for the poisoned CodeBERT model; however, attention weights and biases do not show any significant differences. This work contributes to ongoing efforts in white-box detection of backdoor signals in LLMs of code through the analysis of parameters and activations.



## **19. InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents**

cs.CL

26 pages, 5 figures, 7 tables

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.02691v1) [paper-pdf](http://arxiv.org/pdf/2403.02691v1)

**Authors**: Qiusi Zhan, Zhixiang Liang, Zifan Ying, Daniel Kang

**Abstract**: Recent work has embodied LLMs as agents, allowing them to access tools, perform actions, and interact with external content (e.g., emails or websites). However, external content introduces the risk of indirect prompt injection (IPI) attacks, where malicious instructions are embedded within the content processed by LLMs, aiming to manipulate these agents into executing detrimental actions against users. Given the potentially severe consequences of such attacks, establishing benchmarks to assess and mitigate these risks is imperative.   In this work, we introduce InjecAgent, a benchmark designed to assess the vulnerability of tool-integrated LLM agents to IPI attacks. InjecAgent comprises 1,054 test cases covering 17 different user tools and 62 attacker tools. We categorize attack intentions into two primary types: direct harm to users and exfiltration of private data. We evaluate 30 different LLM agents and show that agents are vulnerable to IPI attacks, with ReAct-prompted GPT-4 vulnerable to attacks 24% of the time. Further investigation into an enhanced setting, where the attacker instructions are reinforced with a hacking prompt, shows additional increases in success rates, nearly doubling the attack success rate on the ReAct-prompted GPT-4. Our findings raise questions about the widespread deployment of LLM Agents. Our benchmark is available at https://github.com/uiuc-kang-lab/InjecAgent.



## **20. KnowPhish: Large Language Models Meet Multimodal Knowledge Graphs for Enhancing Reference-Based Phishing Detection**

cs.CR

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2403.02253v1) [paper-pdf](http://arxiv.org/pdf/2403.02253v1)

**Authors**: Yuexin Li, Chengyu Huang, Shumin Deng, Mei Lin Lock, Tri Cao, Nay Oo, Bryan Hooi, Hoon Wei Lim

**Abstract**: Phishing attacks have inflicted substantial losses on individuals and businesses alike, necessitating the development of robust and efficient automated phishing detection approaches. Reference-based phishing detectors (RBPDs), which compare the logos on a target webpage to a known set of logos, have emerged as the state-of-the-art approach. However, a major limitation of existing RBPDs is that they rely on a manually constructed brand knowledge base, making it infeasible to scale to a large number of brands, which results in false negative errors due to the insufficient brand coverage of the knowledge base. To address this issue, we propose an automated knowledge collection pipeline, using which we collect and release a large-scale multimodal brand knowledge base, KnowPhish, containing 20k brands with rich information about each brand. KnowPhish can be used to boost the performance of existing RBPDs in a plug-and-play manner. A second limitation of existing RBPDs is that they solely rely on the image modality, ignoring useful textual information present in the webpage HTML. To utilize this textual information, we propose a Large Language Model (LLM)-based approach to extract brand information of webpages from text. Our resulting multimodal phishing detection approach, KnowPhish Detector (KPD), can detect phishing webpages with or without logos. We evaluate KnowPhish and KPD on a manually validated dataset, and on a field study under Singapore's local context, showing substantial improvements in effectiveness and efficiency compared to state-of-the-art baselines.



## **21. Rethinking Model Ensemble in Transfer-based Adversarial Attacks**

cs.CV

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2303.09105v2) [paper-pdf](http://arxiv.org/pdf/2303.09105v2)

**Authors**: Huanran Chen, Yichi Zhang, Yinpeng Dong, Xiao Yang, Hang Su, Jun Zhu

**Abstract**: It is widely recognized that deep learning models lack robustness to adversarial examples. An intriguing property of adversarial examples is that they can transfer across different models, which enables black-box attacks without any knowledge of the victim model. An effective strategy to improve the transferability is attacking an ensemble of models. However, previous works simply average the outputs of different models, lacking an in-depth analysis on how and why model ensemble methods can strongly improve the transferability. In this paper, we rethink the ensemble in adversarial attacks and define the common weakness of model ensemble with two properties: 1) the flatness of loss landscape; and 2) the closeness to the local optimum of each model. We empirically and theoretically show that both properties are strongly correlated with the transferability and propose a Common Weakness Attack (CWA) to generate more transferable adversarial examples by promoting these two properties. Experimental results on both image classification and object detection tasks validate the effectiveness of our approach to improving the adversarial transferability, especially when attacking adversarially trained models. We also successfully apply our method to attack a black-box large vision-language model -- Google's Bard, showing the practical effectiveness. Code is available at \url{https://github.com/huanranchen/AdversarialAttacks}.



## **22. One Prompt Word is Enough to Boost Adversarial Robustness for Pre-trained Vision-Language Models**

cs.CV

CVPR2024

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2403.01849v1) [paper-pdf](http://arxiv.org/pdf/2403.01849v1)

**Authors**: Lin Li, Haoyan Guan, Jianing Qiu, Michael Spratling

**Abstract**: Large pre-trained Vision-Language Models (VLMs) like CLIP, despite having remarkable generalization ability, are highly vulnerable to adversarial examples. This work studies the adversarial robustness of VLMs from the novel perspective of the text prompt instead of the extensively studied model weights (frozen in this work). We first show that the effectiveness of both adversarial attack and defense are sensitive to the used text prompt. Inspired by this, we propose a method to improve resilience to adversarial attacks by learning a robust text prompt for VLMs. The proposed method, named Adversarial Prompt Tuning (APT), is effective while being both computationally and data efficient. Extensive experiments are conducted across 15 datasets and 4 data sparsity schemes (from 1-shot to full training data settings) to show APT's superiority over hand-engineered prompts and other state-of-the-art adaption methods. APT demonstrated excellent abilities in terms of the in-distribution performance and the generalization under input distribution shift and across datasets. Surprisingly, by simply adding one learned word to the prompts, APT can significantly boost the accuracy and robustness (epsilon=4/255) over the hand-engineered prompts by +13% and +8.5% on average respectively. The improvement further increases, in our most effective setting, to +26.4% for accuracy and +16.7% for robustness. Code is available at https://github.com/TreeLLi/APT.



## **23. SALAD-Bench: A Hierarchical and Comprehensive Safety Benchmark for Large Language Models**

cs.CL

fix institution typo

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2402.05044v3) [paper-pdf](http://arxiv.org/pdf/2402.05044v3)

**Authors**: Lijun Li, Bowen Dong, Ruohui Wang, Xuhao Hu, Wangmeng Zuo, Dahua Lin, Yu Qiao, Jing Shao

**Abstract**: In the rapidly evolving landscape of Large Language Models (LLMs), ensuring robust safety measures is paramount. To meet this crucial need, we propose \emph{SALAD-Bench}, a safety benchmark specifically designed for evaluating LLMs, attack, and defense methods. Distinguished by its breadth, SALAD-Bench transcends conventional benchmarks through its large scale, rich diversity, intricate taxonomy spanning three levels, and versatile functionalities.SALAD-Bench is crafted with a meticulous array of questions, from standard queries to complex ones enriched with attack, defense modifications and multiple-choice. To effectively manage the inherent complexity, we introduce an innovative evaluators: the LLM-based MD-Judge for QA pairs with a particular focus on attack-enhanced queries, ensuring a seamless, and reliable evaluation. Above components extend SALAD-Bench from standard LLM safety evaluation to both LLM attack and defense methods evaluation, ensuring the joint-purpose utility. Our extensive experiments shed light on the resilience of LLMs against emerging threats and the efficacy of contemporary defense tactics. Data and evaluator are released under https://github.com/OpenSafetyLab/SALAD-BENCH.



## **24. LLMs Can Defend Themselves Against Jailbreaking in a Practical Manner: A Vision Paper**

cs.CR

Fixed the bibliography reference issue in our LLM jailbreak defense  vision paper submitted on 24 Feb 2024

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2402.15727v2) [paper-pdf](http://arxiv.org/pdf/2402.15727v2)

**Authors**: Daoyuan Wu, Shuai Wang, Yang Liu, Ning Liu

**Abstract**: Jailbreaking is an emerging adversarial attack that bypasses the safety alignment deployed in off-the-shelf large language models (LLMs). A considerable amount of research exists proposing more effective jailbreak attacks, including the recent Greedy Coordinate Gradient (GCG) attack, jailbreak template-based attacks such as using "Do-Anything-Now" (DAN), and multilingual jailbreak. In contrast, the defensive side has been relatively less explored. This paper proposes a lightweight yet practical defense called SELFDEFEND, which can defend against all existing jailbreak attacks with minimal delay for jailbreak prompts and negligible delay for normal user prompts. Our key insight is that regardless of the kind of jailbreak strategies employed, they eventually need to include a harmful prompt (e.g., "how to make a bomb") in the prompt sent to LLMs, and we found that existing LLMs can effectively recognize such harmful prompts that violate their safety policies. Based on this insight, we design a shadow stack that concurrently checks whether a harmful prompt exists in the user prompt and triggers a checkpoint in the normal stack once a token of "No" or a harmful prompt is output. The latter could also generate an explainable LLM response to adversarial prompts. We demonstrate our idea of SELFDEFEND works in various jailbreak scenarios through manual analysis in GPT-3.5/4. We also list three future directions to further enhance SELFDEFEND.



## **25. Multilingual Jailbreak Challenges in Large Language Models**

cs.CL

ICLR 2024

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2310.06474v3) [paper-pdf](http://arxiv.org/pdf/2310.06474v3)

**Authors**: Yue Deng, Wenxuan Zhang, Sinno Jialin Pan, Lidong Bing

**Abstract**: While large language models (LLMs) exhibit remarkable capabilities across a wide range of tasks, they pose potential safety concerns, such as the ``jailbreak'' problem, wherein malicious instructions can manipulate LLMs to exhibit undesirable behavior. Although several preventive measures have been developed to mitigate the potential risks associated with LLMs, they have primarily focused on English. In this study, we reveal the presence of multilingual jailbreak challenges within LLMs and consider two potential risky scenarios: unintentional and intentional. The unintentional scenario involves users querying LLMs using non-English prompts and inadvertently bypassing the safety mechanisms, while the intentional scenario concerns malicious users combining malicious instructions with multilingual prompts to deliberately attack LLMs. The experimental results reveal that in the unintentional scenario, the rate of unsafe content increases as the availability of languages decreases. Specifically, low-resource languages exhibit about three times the likelihood of encountering harmful content compared to high-resource languages, with both ChatGPT and GPT-4. In the intentional scenario, multilingual prompts can exacerbate the negative impact of malicious instructions, with astonishingly high rates of unsafe output: 80.92\% for ChatGPT and 40.71\% for GPT-4. To handle such a challenge in the multilingual context, we propose a novel \textsc{Self-Defense} framework that automatically generates multilingual training data for safety fine-tuning. Experimental results show that ChatGPT fine-tuned with such data can achieve a substantial reduction in unsafe content generation. Data is available at \url{https://github.com/DAMO-NLP-SG/multilingual-safety-for-LLMs}.



## **26. Breaking Down the Defenses: A Comparative Survey of Attacks on Large Language Models**

cs.CR

**SubmitDate**: 2024-03-03    [abs](http://arxiv.org/abs/2403.04786v1) [paper-pdf](http://arxiv.org/pdf/2403.04786v1)

**Authors**: Arijit Ghosh Chowdhury, Md Mofijul Islam, Vaibhav Kumar, Faysal Hossain Shezan, Vaibhav Kumar, Vinija Jain, Aman Chadha

**Abstract**: Large Language Models (LLMs) have become a cornerstone in the field of Natural Language Processing (NLP), offering transformative capabilities in understanding and generating human-like text. However, with their rising prominence, the security and vulnerability aspects of these models have garnered significant attention. This paper presents a comprehensive survey of the various forms of attacks targeting LLMs, discussing the nature and mechanisms of these attacks, their potential impacts, and current defense strategies. We delve into topics such as adversarial attacks that aim to manipulate model outputs, data poisoning that affects model training, and privacy concerns related to training data exploitation. The paper also explores the effectiveness of different attack methodologies, the resilience of LLMs against these attacks, and the implications for model integrity and user trust. By examining the latest research, we provide insights into the current landscape of LLM vulnerabilities and defense mechanisms. Our objective is to offer a nuanced understanding of LLM attacks, foster awareness within the AI community, and inspire robust solutions to mitigate these risks in future developments.



## **27. Ignore This Title and HackAPrompt: Exposing Systemic Vulnerabilities of LLMs through a Global Scale Prompt Hacking Competition**

cs.CR

34 pages, 8 figures Codebase:  https://github.com/PromptLabs/hackaprompt Dataset:  https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset/blob/main/README.md  Playground: https://huggingface.co/spaces/hackaprompt/playground

**SubmitDate**: 2024-03-03    [abs](http://arxiv.org/abs/2311.16119v3) [paper-pdf](http://arxiv.org/pdf/2311.16119v3)

**Authors**: Sander Schulhoff, Jeremy Pinto, Anaum Khan, Louis-François Bouchard, Chenglei Si, Svetlina Anati, Valen Tagliabue, Anson Liu Kost, Christopher Carnahan, Jordan Boyd-Graber

**Abstract**: Large Language Models (LLMs) are deployed in interactive contexts with direct user engagement, such as chatbots and writing assistants. These deployments are vulnerable to prompt injection and jailbreaking (collectively, prompt hacking), in which models are manipulated to ignore their original instructions and follow potentially malicious ones. Although widely acknowledged as a significant security threat, there is a dearth of large-scale resources and quantitative studies on prompt hacking. To address this lacuna, we launch a global prompt hacking competition, which allows for free-form human input attacks. We elicit 600K+ adversarial prompts against three state-of-the-art LLMs. We describe the dataset, which empirically verifies that current LLMs can indeed be manipulated via prompt hacking. We also present a comprehensive taxonomical ontology of the types of adversarial prompts.



## **28. Analysis of Privacy Leakage in Federated Large Language Models**

cs.CR

**SubmitDate**: 2024-03-02    [abs](http://arxiv.org/abs/2403.04784v1) [paper-pdf](http://arxiv.org/pdf/2403.04784v1)

**Authors**: Minh N. Vu, Truc Nguyen, Tre' R. Jeter, My T. Thai

**Abstract**: With the rapid adoption of Federated Learning (FL) as the training and tuning protocol for applications utilizing Large Language Models (LLMs), recent research highlights the need for significant modifications to FL to accommodate the large-scale of LLMs. While substantial adjustments to the protocol have been introduced as a response, comprehensive privacy analysis for the adapted FL protocol is currently lacking.   To address this gap, our work delves into an extensive examination of the privacy analysis of FL when used for training LLMs, both from theoretical and practical perspectives. In particular, we design two active membership inference attacks with guaranteed theoretical success rates to assess the privacy leakages of various adapted FL configurations. Our theoretical findings are translated into practical attacks, revealing substantial privacy vulnerabilities in popular LLMs, including BERT, RoBERTa, DistilBERT, and OpenAI's GPTs, across multiple real-world language datasets. Additionally, we conduct thorough experiments to evaluate the privacy leakage of these models when data is protected by state-of-the-art differential privacy (DP) mechanisms.



## **29. AutoDefense: Multi-Agent LLM Defense against Jailbreak Attacks**

cs.LG

**SubmitDate**: 2024-03-02    [abs](http://arxiv.org/abs/2403.04783v1) [paper-pdf](http://arxiv.org/pdf/2403.04783v1)

**Authors**: Yifan Zeng, Yiran Wu, Xiao Zhang, Huazheng Wang, Qingyun Wu

**Abstract**: Despite extensive pre-training and fine-tuning in moral alignment to prevent generating harmful information at user request, large language models (LLMs) remain vulnerable to jailbreak attacks. In this paper, we propose AutoDefense, a response-filtering based multi-agent defense framework that filters harmful responses from LLMs. This framework assigns different roles to LLM agents and employs them to complete the defense task collaboratively. The division in tasks enhances the overall instruction-following of LLMs and enables the integration of other defense components as tools. AutoDefense can adapt to various sizes and kinds of open-source LLMs that serve as agents. Through conducting extensive experiments on a large scale of harmful and safe prompts, we validate the effectiveness of the proposed AutoDefense in improving the robustness against jailbreak attacks, while maintaining the performance at normal user request. Our code and data are publicly available at https://github.com/XHMY/AutoDefense.



## **30. Accelerating Greedy Coordinate Gradient via Probe Sampling**

cs.CL

**SubmitDate**: 2024-03-02    [abs](http://arxiv.org/abs/2403.01251v1) [paper-pdf](http://arxiv.org/pdf/2403.01251v1)

**Authors**: Yiran Zhao, Wenyue Zheng, Tianle Cai, Xuan Long Do, Kenji Kawaguchi, Anirudh Goyal, Michael Shieh

**Abstract**: Safety of Large Language Models (LLMs) has become a central issue given their rapid progress and wide applications. Greedy Coordinate Gradient (GCG) is shown to be effective in constructing prompts containing adversarial suffixes to break the presumingly safe LLMs, but the optimization of GCG is time-consuming and limits its practicality. To reduce the time cost of GCG and enable more comprehensive studies of LLM safety, in this work, we study a new algorithm called $\texttt{Probe sampling}$ to accelerate the GCG algorithm. At the core of the algorithm is a mechanism that dynamically determines how similar a smaller draft model's predictions are to the target model's predictions for prompt candidates. When the target model is similar to the draft model, we rely heavily on the draft model to filter out a large number of potential prompt candidates to reduce the computation time. Probe sampling achieves up to $5.6$ times speedup using Llama2-7b and leads to equal or improved attack success rate (ASR) on the AdvBench.



## **31. Inexact Unlearning Needs More Careful Evaluations to Avoid a False Sense of Privacy**

cs.LG

**SubmitDate**: 2024-03-02    [abs](http://arxiv.org/abs/2403.01218v1) [paper-pdf](http://arxiv.org/pdf/2403.01218v1)

**Authors**: Jamie Hayes, Ilia Shumailov, Eleni Triantafillou, Amr Khalifa, Nicolas Papernot

**Abstract**: The high cost of model training makes it increasingly desirable to develop techniques for unlearning. These techniques seek to remove the influence of a training example without having to retrain the model from scratch. Intuitively, once a model has unlearned, an adversary that interacts with the model should no longer be able to tell whether the unlearned example was included in the model's training set or not. In the privacy literature, this is known as membership inference. In this work, we discuss adaptations of Membership Inference Attacks (MIAs) to the setting of unlearning (leading to their ``U-MIA'' counterparts). We propose a categorization of existing U-MIAs into ``population U-MIAs'', where the same attacker is instantiated for all examples, and ``per-example U-MIAs'', where a dedicated attacker is instantiated for each example. We show that the latter category, wherein the attacker tailors its membership prediction to each example under attack, is significantly stronger. Indeed, our results show that the commonly used U-MIAs in the unlearning literature overestimate the privacy protection afforded by existing unlearning techniques on both vision and language models. Our investigation reveals a large variance in the vulnerability of different examples to per-example U-MIAs. In fact, several unlearning algorithms lead to a reduced vulnerability for some, but not all, examples that we wish to unlearn, at the expense of increasing it for other examples. Notably, we find that the privacy protection for the remaining training examples may worsen as a consequence of unlearning. We also discuss the fundamental difficulty of equally protecting all examples using existing unlearning schemes, due to the different rates at which examples are unlearned. We demonstrate that naive attempts at tailoring unlearning stopping criteria to different examples fail to alleviate these issues.



## **32. Prompt Injection attack against LLM-integrated Applications**

cs.CR

**SubmitDate**: 2024-03-02    [abs](http://arxiv.org/abs/2306.05499v2) [paper-pdf](http://arxiv.org/pdf/2306.05499v2)

**Authors**: Yi Liu, Gelei Deng, Yuekang Li, Kailong Wang, Zihao Wang, Xiaofeng Wang, Tianwei Zhang, Yepang Liu, Haoyu Wang, Yan Zheng, Yang Liu

**Abstract**: Large Language Models (LLMs), renowned for their superior proficiency in language comprehension and generation, stimulate a vibrant ecosystem of applications around them. However, their extensive assimilation into various services introduces significant security risks. This study deconstructs the complexities and implications of prompt injection attacks on actual LLM-integrated applications. Initially, we conduct an exploratory analysis on ten commercial applications, highlighting the constraints of current attack strategies in practice. Prompted by these limitations, we subsequently formulate HouYi, a novel black-box prompt injection attack technique, which draws inspiration from traditional web injection attacks. HouYi is compartmentalized into three crucial elements: a seamlessly-incorporated pre-constructed prompt, an injection prompt inducing context partition, and a malicious payload designed to fulfill the attack objectives. Leveraging HouYi, we unveil previously unknown and severe attack outcomes, such as unrestricted arbitrary LLM usage and uncomplicated application prompt theft. We deploy HouYi on 36 actual LLM-integrated applications and discern 31 applications susceptible to prompt injection. 10 vendors have validated our discoveries, including Notion, which has the potential to impact millions of users. Our investigation illuminates both the possible risks of prompt injection attacks and the possible tactics for mitigation.



## **33. Knowledge Sanitization of Large Language Models**

cs.CL

**SubmitDate**: 2024-03-02    [abs](http://arxiv.org/abs/2309.11852v2) [paper-pdf](http://arxiv.org/pdf/2309.11852v2)

**Authors**: Yoichi Ishibashi, Hidetoshi Shimodaira

**Abstract**: We explore a knowledge sanitization approach to mitigate the privacy concerns associated with large language models (LLMs). LLMs trained on a large corpus of Web data can memorize and potentially reveal sensitive or confidential information, raising critical security concerns. Our technique efficiently fine-tunes these models using the Low-Rank Adaptation (LoRA) method, prompting them to generate harmless responses such as ``I don't know'' when queried about specific information. Experimental results in a closed-book question-answering task show that our straightforward method not only minimizes particular knowledge leakage but also preserves the overall performance of LLMs. These two advantages strengthen the defense against extraction attacks and reduces the emission of harmful content such as hallucinations.



## **34. AutoAttacker: A Large Language Model Guided System to Implement Automatic Cyber-attacks**

cs.CR

**SubmitDate**: 2024-03-02    [abs](http://arxiv.org/abs/2403.01038v1) [paper-pdf](http://arxiv.org/pdf/2403.01038v1)

**Authors**: Jiacen Xu, Jack W. Stokes, Geoff McDonald, Xuesong Bai, David Marshall, Siyue Wang, Adith Swaminathan, Zhou Li

**Abstract**: Large language models (LLMs) have demonstrated impressive results on natural language tasks, and security researchers are beginning to employ them in both offensive and defensive systems. In cyber-security, there have been multiple research efforts that utilize LLMs focusing on the pre-breach stage of attacks like phishing and malware generation. However, so far there lacks a comprehensive study regarding whether LLM-based systems can be leveraged to simulate the post-breach stage of attacks that are typically human-operated, or "hands-on-keyboard" attacks, under various attack techniques and environments.   As LLMs inevitably advance, they may be able to automate both the pre- and post-breach attack stages. This shift may transform organizational attacks from rare, expert-led events to frequent, automated operations requiring no expertise and executed at automation speed and scale. This risks fundamentally changing global computer security and correspondingly causing substantial economic impacts, and a goal of this work is to better understand these risks now so we can better prepare for these inevitable ever-more-capable LLMs on the horizon. On the immediate impact side, this research serves three purposes. First, an automated LLM-based, post-breach exploitation framework can help analysts quickly test and continually improve their organization's network security posture against previously unseen attacks. Second, an LLM-based penetration test system can extend the effectiveness of red teams with a limited number of human analysts. Finally, this research can help defensive systems and teams learn to detect novel attack behaviors preemptively before their use in the wild....



## **35. DrAttack: Prompt Decomposition and Reconstruction Makes Powerful LLM Jailbreakers**

cs.CR

**SubmitDate**: 2024-03-01    [abs](http://arxiv.org/abs/2402.16914v2) [paper-pdf](http://arxiv.org/pdf/2402.16914v2)

**Authors**: Xirui Li, Ruochen Wang, Minhao Cheng, Tianyi Zhou, Cho-Jui Hsieh

**Abstract**: The safety alignment of Large Language Models (LLMs) is vulnerable to both manual and automated jailbreak attacks, which adversarially trigger LLMs to output harmful content. However, current methods for jailbreaking LLMs, which nest entire harmful prompts, are not effective at concealing malicious intent and can be easily identified and rejected by well-aligned LLMs. This paper discovers that decomposing a malicious prompt into separated sub-prompts can effectively obscure its underlying malicious intent by presenting it in a fragmented, less detectable form, thereby addressing these limitations. We introduce an automatic prompt \textbf{D}ecomposition and \textbf{R}econstruction framework for jailbreak \textbf{Attack} (DrAttack). DrAttack includes three key components: (a) `Decomposition' of the original prompt into sub-prompts, (b) `Reconstruction' of these sub-prompts implicitly by in-context learning with semantically similar but harmless reassembling demo, and (c) a `Synonym Search' of sub-prompts, aiming to find sub-prompts' synonyms that maintain the original intent while jailbreaking LLMs. An extensive empirical study across multiple open-source and closed-source LLMs demonstrates that, with a significantly reduced number of queries, DrAttack obtains a substantial gain of success rate over prior SOTA prompt-only attackers. Notably, the success rate of 78.0\% on GPT-4 with merely 15 queries surpassed previous art by 33.1\%. The project is available at https://github.com/xirui-li/DrAttack.



## **36. Teach LLMs to Phish: Stealing Private Information from Language Models**

cs.CR

ICLR 2024

**SubmitDate**: 2024-03-01    [abs](http://arxiv.org/abs/2403.00871v1) [paper-pdf](http://arxiv.org/pdf/2403.00871v1)

**Authors**: Ashwinee Panda, Christopher A. Choquette-Choo, Zhengming Zhang, Yaoqing Yang, Prateek Mittal

**Abstract**: When large language models are trained on private data, it can be a significant privacy risk for them to memorize and regurgitate sensitive information. In this work, we propose a new practical data extraction attack that we call "neural phishing". This attack enables an adversary to target and extract sensitive or personally identifiable information (PII), e.g., credit card numbers, from a model trained on user data with upwards of 10% attack success rates, at times, as high as 50%. Our attack assumes only that an adversary can insert as few as 10s of benign-appearing sentences into the training dataset using only vague priors on the structure of the user data.



## **37. Here's a Free Lunch: Sanitizing Backdoored Models with Model Merge**

cs.CL

work in progress

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2402.19334v1) [paper-pdf](http://arxiv.org/pdf/2402.19334v1)

**Authors**: Ansh Arora, Xuanli He, Maximilian Mozes, Srinibas Swain, Mark Dras, Qiongkai Xu

**Abstract**: The democratization of pre-trained language models through open-source initiatives has rapidly advanced innovation and expanded access to cutting-edge technologies. However, this openness also brings significant security risks, including backdoor attacks, where hidden malicious behaviors are triggered by specific inputs, compromising natural language processing (NLP) system integrity and reliability. This paper suggests that merging a backdoored model with other homogeneous models can remediate backdoor vulnerabilities even if such models are not entirely secure. In our experiments, we explore various models (BERT-Base, RoBERTa-Large, Llama2-7B, and Mistral-7B) and datasets (SST-2, OLID, AG News, and QNLI). Compared to multiple advanced defensive approaches, our method offers an effective and efficient inference-stage defense against backdoor attacks without additional resources or specific knowledge. Our approach consistently outperforms the other advanced baselines, leading to an average of 75% reduction in the attack success rate. Since model merging has been an established approach for improving model performance, the extra advantage it provides regarding defense can be seen as a cost-free bonus.



## **38. PRSA: Prompt Reverse Stealing Attacks against Large Language Models**

cs.CR

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2402.19200v1) [paper-pdf](http://arxiv.org/pdf/2402.19200v1)

**Authors**: Yong Yang, Xuhong Zhang, Yi Jiang, Xi Chen, Haoyu Wang, Shouling Ji, Zonghui Wang

**Abstract**: Prompt, recognized as crucial intellectual property, enables large language models (LLMs) to perform specific tasks without the need of fine-tuning, underscoring their escalating importance. With the rise of prompt-based services, such as prompt marketplaces and LLM applications, providers often display prompts' capabilities through input-output examples to attract users. However, this paradigm raises a pivotal security concern: does the exposure of input-output pairs pose the risk of potential prompt leakage, infringing on the intellectual property rights of the developers? To our knowledge, this problem still has not been comprehensively explored yet. To remedy this gap, in this paper, we perform the first in depth exploration and propose a novel attack framework for reverse-stealing prompts against commercial LLMs, namely PRSA. The main idea of PRSA is that by analyzing the critical features of the input-output pairs, we mimic and gradually infer (steal) the target prompts. In detail, PRSA mainly consists of two key phases: prompt mutation and prompt pruning. In the mutation phase, we propose a prompt attention algorithm based on differential feedback to capture these critical features for effectively inferring the target prompts. In the prompt pruning phase, we identify and mask the words dependent on specific inputs, enabling the prompts to accommodate diverse inputs for generalization. Through extensive evaluation, we verify that PRSA poses a severe threat in real world scenarios. We have reported these findings to prompt service providers and actively collaborate with them to take protective measures for prompt copyright.



## **39. A Semantic Invariant Robust Watermark for Large Language Models**

cs.CR

ICLR2024, 21 pages, 10 figures, 6 tables

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2310.06356v2) [paper-pdf](http://arxiv.org/pdf/2310.06356v2)

**Authors**: Aiwei Liu, Leyi Pan, Xuming Hu, Shiao Meng, Lijie Wen

**Abstract**: Watermark algorithms for large language models (LLMs) have achieved extremely high accuracy in detecting text generated by LLMs. Such algorithms typically involve adding extra watermark logits to the LLM's logits at each generation step. However, prior algorithms face a trade-off between attack robustness and security robustness. This is because the watermark logits for a token are determined by a certain number of preceding tokens; a small number leads to low security robustness, while a large number results in insufficient attack robustness. In this work, we propose a semantic invariant watermarking method for LLMs that provides both attack robustness and security robustness. The watermark logits in our work are determined by the semantics of all preceding tokens. Specifically, we utilize another embedding LLM to generate semantic embeddings for all preceding tokens, and then these semantic embeddings are transformed into the watermark logits through our trained watermark model. Subsequent analyses and experiments demonstrated the attack robustness of our method in semantically invariant settings: synonym substitution and text paraphrasing settings. Finally, we also show that our watermark possesses adequate security robustness. Our code and data are available at https://github.com/THU-BPM/Robust_Watermark.



## **40. Typographic Attacks in Large Multimodal Models Can be Alleviated by More Informative Prompts**

cs.CV

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2402.19150v1) [paper-pdf](http://arxiv.org/pdf/2402.19150v1)

**Authors**: Hao Cheng, Erjia Xiao, Renjing Xu

**Abstract**: Large Multimodal Models (LMMs) rely on pre-trained Vision Language Models (VLMs) and Large Language Models (LLMs) to perform amazing emergent abilities on various multimodal tasks in the joint space of vision and language. However, the Typographic Attack, which shows disruption to VLMs, has also been certified as a security vulnerability to LMMs. In this work, we first comprehensively investigate the distractibility of LMMs by typography. In particular, we introduce the Typographic Dataset designed to evaluate distractibility across various multi-modal subtasks, such as object recognition, visual attributes detection, enumeration, arithmetic computation, and commonsense reasoning. To further study the effect of typographic patterns on performance, we also scrutinize the effect of tuning various typographic factors, encompassing font size, color, opacity, and spatial positioning of typos. We discover that LMMs can partially distinguish visual contents and typos when confronting typographic attacks, which suggests that embeddings from vision encoders contain enough information to distinguish visual contents and typos in images. Inspired by such phenomena, we demonstrate that CLIP's performance of zero-shot classification on typo-ridden images can be significantly improved by providing more informative texts to match images. Furthermore, we also prove that LMMs can utilize more informative prompts to leverage information in embeddings to differentiate between visual content and typos. Finally, we propose a prompt information enhancement method that can effectively mitigate the effects of typography.



## **41. Cognitive Overload: Jailbreaking Large Language Models with Overloaded Logical Thinking**

cs.CL

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2311.09827v2) [paper-pdf](http://arxiv.org/pdf/2311.09827v2)

**Authors**: Nan Xu, Fei Wang, Ben Zhou, Bang Zheng Li, Chaowei Xiao, Muhao Chen

**Abstract**: While large language models (LLMs) have demonstrated increasing power, they have also given rise to a wide range of harmful behaviors. As representatives, jailbreak attacks can provoke harmful or unethical responses from LLMs, even after safety alignment. In this paper, we investigate a novel category of jailbreak attacks specifically designed to target the cognitive structure and processes of LLMs. Specifically, we analyze the safety vulnerability of LLMs in the face of (1) multilingual cognitive overload, (2) veiled expression, and (3) effect-to-cause reasoning. Different from previous jailbreak attacks, our proposed cognitive overload is a black-box attack with no need for knowledge of model architecture or access to model weights. Experiments conducted on AdvBench and MasterKey reveal that various LLMs, including both popular open-source model Llama 2 and the proprietary model ChatGPT, can be compromised through cognitive overload. Motivated by cognitive psychology work on managing cognitive load, we further investigate defending cognitive overload attack from two perspectives. Empirical studies show that our cognitive overload from three perspectives can jailbreak all studied LLMs successfully, while existing defense strategies can hardly mitigate the caused malicious uses effectively.



## **42. Vaccine: Perturbation-aware Alignment for Large Language Model**

cs.LG

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2402.01109v3) [paper-pdf](http://arxiv.org/pdf/2402.01109v3)

**Authors**: Tiansheng Huang, Sihao Hu, Ling Liu

**Abstract**: The new paradigm of finetuning-as-a-service introduces a new attack surface for Large Language Models (LLMs): a few harmful data uploaded by users can easily trick the finetuning to produce an alignment-broken model. We conduct an empirical analysis and uncover a \textit{harmful embedding drift} phenomenon, showing a probable cause of the alignment-broken effect. Inspired by our findings, we propose Vaccine, a perturbation-aware alignment technique to mitigate the security risk of users finetuning. The core idea of Vaccine is to produce invariant hidden embeddings by progressively adding crafted perturbation to them in the alignment phase. This enables the embeddings to withstand harmful perturbation from un-sanitized user data in the finetuning phase. Our results on open source mainstream LLMs (e.g., Llama2, Opt, Vicuna) demonstrate that Vaccine can boost the robustness of alignment against harmful prompts induced embedding drift while reserving reasoning ability towards benign prompts. Our code is available at \url{https://github.com/git-disl/Vaccine}.



## **43. Defending Large Language Models against Jailbreak Attacks via Semantic Smoothing**

cs.CL

37 pages

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.16192v2) [paper-pdf](http://arxiv.org/pdf/2402.16192v2)

**Authors**: Jiabao Ji, Bairu Hou, Alexander Robey, George J. Pappas, Hamed Hassani, Yang Zhang, Eric Wong, Shiyu Chang

**Abstract**: Aligned large language models (LLMs) are vulnerable to jailbreaking attacks, which bypass the safeguards of targeted LLMs and fool them into generating objectionable content. While initial defenses show promise against token-based threat models, there do not exist defenses that provide robustness against semantic attacks and avoid unfavorable trade-offs between robustness and nominal performance. To meet this need, we propose SEMANTICSMOOTH, a smoothing-based defense that aggregates the predictions of multiple semantically transformed copies of a given input prompt. Experimental results demonstrate that SEMANTICSMOOTH achieves state-of-the-art robustness against GCG, PAIR, and AutoDAN attacks while maintaining strong nominal performance on instruction following benchmarks such as InstructionFollowing and AlpacaEval. The codes will be publicly available at https://github.com/UCSB-NLP-Chang/SemanticSmooth.



## **44. Defending LLMs against Jailbreaking Attacks via Backtranslation**

cs.CL

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.16459v2) [paper-pdf](http://arxiv.org/pdf/2402.16459v2)

**Authors**: Yihan Wang, Zhouxing Shi, Andrew Bai, Cho-Jui Hsieh

**Abstract**: Although many large language models (LLMs) have been trained to refuse harmful requests, they are still vulnerable to jailbreaking attacks, which rewrite the original prompt to conceal its harmful intent. In this paper, we propose a new method for defending LLMs against jailbreaking attacks by ``backtranslation''. Specifically, given an initial response generated by the target LLM from an input prompt, our backtranslation prompts a language model to infer an input prompt that can lead to the response. The inferred prompt is called the backtranslated prompt which tends to reveal the actual intent of the original prompt, since it is generated based on the LLM's response and is not directly manipulated by the attacker. We then run the target LLM again on the backtranslated prompt, and we refuse the original prompt if the model refuses the backtranslated prompt. We explain that the proposed defense provides several benefits on its effectiveness and efficiency. We empirically demonstrate that our defense significantly outperforms the baselines, in the cases that are hard for the baselines, and our defense also has little impact on the generation quality for benign input prompts.



## **45. A New Era in LLM Security: Exploring Security Concerns in Real-World LLM-based Systems**

cs.CR

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.18649v1) [paper-pdf](http://arxiv.org/pdf/2402.18649v1)

**Authors**: Fangzhou Wu, Ning Zhang, Somesh Jha, Patrick McDaniel, Chaowei Xiao

**Abstract**: Large Language Model (LLM) systems are inherently compositional, with individual LLM serving as the core foundation with additional layers of objects such as plugins, sandbox, and so on. Along with the great potential, there are also increasing concerns over the security of such probabilistic intelligent systems. However, existing studies on LLM security often focus on individual LLM, but without examining the ecosystem through the lens of LLM systems with other objects (e.g., Frontend, Webtool, Sandbox, and so on). In this paper, we systematically analyze the security of LLM systems, instead of focusing on the individual LLMs. To do so, we build on top of the information flow and formulate the security of LLM systems as constraints on the alignment of the information flow within LLM and between LLM and other objects. Based on this construction and the unique probabilistic nature of LLM, the attack surface of the LLM system can be decomposed into three key components: (1) multi-layer security analysis, (2) analysis of the existence of constraints, and (3) analysis of the robustness of these constraints. To ground this new attack surface, we propose a multi-layer and multi-step approach and apply it to the state-of-art LLM system, OpenAI GPT4. Our investigation exposes several security issues, not just within the LLM model itself but also in its integration with other components. We found that although the OpenAI GPT4 has designed numerous safety constraints to improve its safety features, these safety constraints are still vulnerable to attackers. To further demonstrate the real-world threats of our discovered vulnerabilities, we construct an end-to-end attack where an adversary can illicitly acquire the user's chat history, all without the need to manipulate the user's input or gain direct access to OpenAI GPT4. Our demo is in the link: https://fzwark.github.io/LLM-System-Attack-Demo/



## **46. Making Them Ask and Answer: Jailbreaking Large Language Models in Few Queries via Disguise and Reconstruction**

cs.CR

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.18104v1) [paper-pdf](http://arxiv.org/pdf/2402.18104v1)

**Authors**: Tong Liu, Yingjie Zhang, Zhe Zhao, Yinpeng Dong, Guozhu Meng, Kai Chen

**Abstract**: In recent years, large language models (LLMs) have demonstrated notable success across various tasks, but the trustworthiness of LLMs is still an open problem. One specific threat is the potential to generate toxic or harmful responses. Attackers can craft adversarial prompts that induce harmful responses from LLMs. In this work, we pioneer a theoretical foundation in LLMs security by identifying bias vulnerabilities within the safety fine-tuning and design a black-box jailbreak method named DRA (Disguise and Reconstruction Attack), which conceals harmful instructions through disguise and prompts the model to reconstruct the original harmful instruction within its completion. We evaluate DRA across various open-source and close-source models, showcasing state-of-the-art jailbreak success rates and attack efficiency. Notably, DRA boasts a 90\% attack success rate on LLM chatbots GPT-4.



## **47. EmMark: Robust Watermarks for IP Protection of Embedded Quantized Large Language Models**

cs.CR

Accept to DAC 2024

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2402.17938v1) [paper-pdf](http://arxiv.org/pdf/2402.17938v1)

**Authors**: Ruisi Zhang, Farinaz Koushanfar

**Abstract**: This paper introduces EmMark,a novel watermarking framework for protecting the intellectual property (IP) of embedded large language models deployed on resource-constrained edge devices. To address the IP theft risks posed by malicious end-users, EmMark enables proprietors to authenticate ownership by querying the watermarked model weights and matching the inserted signatures. EmMark's novelty lies in its strategic watermark weight parameters selection, nsuring robustness and maintaining model quality. Extensive proof-of-concept evaluations of models from OPT and LLaMA-2 families demonstrate EmMark's fidelity, achieving 100% success in watermark extraction with model performance preservation. EmMark also showcased its resilience against watermark removal and forging attacks.



## **48. LLM-Resistant Math Word Problem Generation via Adversarial Attacks**

cs.CL

Code is available at  https://github.com/ruoyuxie/adversarial_mwps_generation

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2402.17916v1) [paper-pdf](http://arxiv.org/pdf/2402.17916v1)

**Authors**: Roy Xie, Chengxuan Huang, Junlin Wang, Bhuwan Dhingra

**Abstract**: Large language models (LLMs) have significantly transformed the educational landscape. As current plagiarism detection tools struggle to keep pace with LLMs' rapid advancements, the educational community faces the challenge of assessing students' true problem-solving abilities in the presence of LLMs. In this work, we explore a new paradigm for ensuring fair evaluation -- generating adversarial examples which preserve the structure and difficulty of the original questions aimed for assessment, but are unsolvable by LLMs. Focusing on the domain of math word problems, we leverage abstract syntax trees to structurally generate adversarial examples that cause LLMs to produce incorrect answers by simply editing the numeric values in the problems. We conduct experiments on various open- and closed-source LLMs, quantitatively and qualitatively demonstrating that our method significantly degrades their math problem-solving ability. We identify shared vulnerabilities among LLMs and propose a cost-effective approach to attack high-cost models. Additionally, we conduct automatic analysis on math problems and investigate the cause of failure to guide future research on LLM's mathematical capability.



## **49. Mitigating Fine-tuning Jailbreak Attack with Backdoor Enhanced Alignment**

cs.CR

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2402.14968v2) [paper-pdf](http://arxiv.org/pdf/2402.14968v2)

**Authors**: Jiongxiao Wang, Jiazhao Li, Yiquan Li, Xiangyu Qi, Junjie Hu, Yixuan Li, Patrick McDaniel, Muhao Chen, Bo Li, Chaowei Xiao

**Abstract**: Despite the general capabilities of Large Language Models (LLMs) like GPT-4 and Llama-2, these models still request fine-tuning or adaptation with customized data when it comes to meeting the specific business demands and intricacies of tailored use cases. However, this process inevitably introduces new safety threats, particularly against the Fine-tuning based Jailbreak Attack (FJAttack), where incorporating just a few harmful examples into the fine-tuning dataset can significantly compromise the model safety. Though potential defenses have been proposed by incorporating safety examples into the fine-tuning dataset to reduce the safety issues, such approaches require incorporating a substantial amount of safety examples, making it inefficient. To effectively defend against the FJAttack with limited safety examples, we propose a Backdoor Enhanced Safety Alignment method inspired by an analogy with the concept of backdoor attacks. In particular, we construct prefixed safety examples by integrating a secret prompt, acting as a "backdoor trigger", that is prefixed to safety examples. Our comprehensive experiments demonstrate that through the Backdoor Enhanced Safety Alignment with adding as few as 11 prefixed safety examples, the maliciously fine-tuned LLMs will achieve similar safety performance as the original aligned models. Furthermore, we also explore the effectiveness of our method in a more practical setting where the fine-tuning data consists of both FJAttack examples and the fine-tuning task data. Our method shows great efficacy in defending against FJAttack without harming the performance of fine-tuning tasks.



## **50. Semantic Mirror Jailbreak: Genetic Algorithm Based Jailbreak Prompts Against Open-source LLMs**

cs.CL

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2402.14872v2) [paper-pdf](http://arxiv.org/pdf/2402.14872v2)

**Authors**: Xiaoxia Li, Siyuan Liang, Jiyi Zhang, Han Fang, Aishan Liu, Ee-Chien Chang

**Abstract**: Large Language Models (LLMs), used in creative writing, code generation, and translation, generate text based on input sequences but are vulnerable to jailbreak attacks, where crafted prompts induce harmful outputs. Most jailbreak prompt methods use a combination of jailbreak templates followed by questions to ask to create jailbreak prompts. However, existing jailbreak prompt designs generally suffer from excessive semantic differences, resulting in an inability to resist defenses that use simple semantic metrics as thresholds. Jailbreak prompts are semantically more varied than the original questions used for queries. In this paper, we introduce a Semantic Mirror Jailbreak (SMJ) approach that bypasses LLMs by generating jailbreak prompts that are semantically similar to the original question. We model the search for jailbreak prompts that satisfy both semantic similarity and jailbreak validity as a multi-objective optimization problem and employ a standardized set of genetic algorithms for generating eligible prompts. Compared to the baseline AutoDAN-GA, SMJ achieves attack success rates (ASR) that are at most 35.4% higher without ONION defense and 85.2% higher with ONION defense. SMJ's better performance in all three semantic meaningfulness metrics of Jailbreak Prompt, Similarity, and Outlier, also means that SMJ is resistant to defenses that use those metrics as thresholds.



