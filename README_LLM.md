# Latest Large Language Model Attack Papers
**update at 2024-10-16 11:21:23**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks**

cs.MA

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11782v1) [paper-pdf](http://arxiv.org/pdf/2410.11782v1)

**Authors**: Guibin Zhang, Yanwei Yue, Xiangguo Sun, Guancheng Wan, Miao Yu, Junfeng Fang, Kun Wang, Dawei Cheng

**Abstract**: Recent advancements in large language model (LLM)-based agents have demonstrated that collective intelligence can significantly surpass the capabilities of individual agents, primarily due to well-crafted inter-agent communication topologies. Despite the diverse and high-performing designs available, practitioners often face confusion when selecting the most effective pipeline for their specific task: \textit{Which topology is the best choice for my task, avoiding unnecessary communication token overhead while ensuring high-quality solution?} In response to this dilemma, we introduce G-Designer, an adaptive, efficient, and robust solution for multi-agent deployment, which dynamically designs task-aware, customized communication topologies. Specifically, G-Designer models the multi-agent system as a multi-agent network, leveraging a variational graph auto-encoder to encode both the nodes (agents) and a task-specific virtual node, and decodes a task-adaptive and high-performing communication topology. Extensive experiments on six benchmarks showcase that G-Designer is: \textbf{(1) high-performing}, achieving superior results on MMLU with accuracy at $84.50\%$ and on HumanEval with pass@1 at $89.90\%$; \textbf{(2) task-adaptive}, architecting communication protocols tailored to task difficulty, reducing token consumption by up to $95.33\%$ on HumanEval; and \textbf{(3) adversarially robust}, defending against agent adversarial attacks with merely $0.3\%$ accuracy drop.



## **2. LLM-Based Robust Product Classification in Commerce and Compliance**

cs.CL

Camera-ready version for Customizable NLP Workshop at EMNLP 2024. 11  pages

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2408.05874v2) [paper-pdf](http://arxiv.org/pdf/2408.05874v2)

**Authors**: Sina Gholamian, Gianfranco Romani, Bartosz Rudnikowicz, Stavroula Skylaki

**Abstract**: Product classification is a crucial task in international trade, as compliance regulations are verified and taxes and duties are applied based on product categories. Manual classification of products is time-consuming and error-prone, and the sheer volume of products imported and exported renders the manual process infeasible. Consequently, e-commerce platforms and enterprises involved in international trade have turned to automatic product classification using machine learning. However, current approaches do not consider the real-world challenges associated with product classification, such as very abbreviated and incomplete product descriptions. In addition, recent advancements in generative Large Language Models (LLMs) and their reasoning capabilities are mainly untapped in product classification and e-commerce. In this research, we explore the real-life challenges of industrial classification and we propose data perturbations that allow for realistic data simulation. Furthermore, we employ LLM-based product classification to improve the robustness of the prediction in presence of incomplete data. Our research shows that LLMs with in-context learning outperform the supervised approaches in the clean-data scenario. Additionally, we illustrate that LLMs are significantly more robust than the supervised approaches when data attacks are present.



## **3. Phantom: General Trigger Attacks on Retrieval Augmented Language Generation**

cs.CR

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2405.20485v2) [paper-pdf](http://arxiv.org/pdf/2405.20485v2)

**Authors**: Harsh Chaudhari, Giorgio Severi, John Abascal, Matthew Jagielski, Christopher A. Choquette-Choo, Milad Nasr, Cristina Nita-Rotaru, Alina Oprea

**Abstract**: Retrieval Augmented Generation (RAG) expands the capabilities of modern large language models (LLMs), by anchoring, adapting, and personalizing their responses to the most relevant knowledge sources. It is particularly useful in chatbot applications, allowing developers to customize LLM output without expensive retraining. Despite their significant utility in various applications, RAG systems present new security risks. In this work, we propose new attack vectors that allow an adversary to inject a single malicious document into a RAG system's knowledge base, and mount a backdoor poisoning attack. We design Phantom, a general two-stage optimization framework against RAG systems, that crafts a malicious poisoned document leading to an integrity violation in the model's output. First, the document is constructed to be retrieved only when a specific trigger sequence of tokens appears in the victim's queries. Second, the document is further optimized with crafted adversarial text that induces various adversarial objectives on the LLM output, including refusal to answer, reputation damage, privacy violations, and harmful behaviors. We demonstrate our attacks on multiple LLM architectures, including Gemma, Vicuna, and Llama, and show that they transfer to GPT-3.5 Turbo and GPT-4. Finally, we successfully conducted a Phantom attack on NVIDIA's black-box production RAG system, "Chat with RTX".



## **4. Efficient and Effective Universal Adversarial Attack against Vision-Language Pre-training Models**

cs.CV

11 pages

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11639v1) [paper-pdf](http://arxiv.org/pdf/2410.11639v1)

**Authors**: Fan Yang, Yihao Huang, Kailong Wang, Ling Shi, Geguang Pu, Yang Liu, Haoyu Wang

**Abstract**: Vision-language pre-training (VLP) models, trained on large-scale image-text pairs, have become widely used across a variety of downstream vision-and-language (V+L) tasks. This widespread adoption raises concerns about their vulnerability to adversarial attacks. Non-universal adversarial attacks, while effective, are often impractical for real-time online applications due to their high computational demands per data instance. Recently, universal adversarial perturbations (UAPs) have been introduced as a solution, but existing generator-based UAP methods are significantly time-consuming. To overcome the limitation, we propose a direct optimization-based UAP approach, termed DO-UAP, which significantly reduces resource consumption while maintaining high attack performance. Specifically, we explore the necessity of multimodal loss design and introduce a useful data augmentation strategy. Extensive experiments conducted on three benchmark VLP datasets, six popular VLP models, and three classical downstream tasks demonstrate the efficiency and effectiveness of DO-UAP. Specifically, our approach drastically decreases the time consumption by 23-fold while achieving a better attack performance.



## **5. Gotcha! This Model Uses My Code! Evaluating Membership Leakage Risks in Code Models**

cs.SE

Accepted by IEEE Transactions on Software Engineering, Camera-Ready  Version

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2310.01166v2) [paper-pdf](http://arxiv.org/pdf/2310.01166v2)

**Authors**: Zhou Yang, Zhipeng Zhao, Chenyu Wang, Jieke Shi, Dongsum Kim, Donggyun Han, David Lo

**Abstract**: Given large-scale source code datasets available in open-source projects and advanced large language models, recent code models have been proposed to address a series of critical software engineering tasks, such as program repair and code completion. The training data of the code models come from various sources, not only the publicly available source code, e.g., open-source projects on GitHub but also the private data such as the confidential source code from companies, which may contain sensitive information (for example, SSH keys and personal information). As a result, the use of these code models may raise new privacy concerns.   In this paper, we focus on a critical yet not well-explored question on using code models: what is the risk of membership information leakage in code models? Membership information leakage refers to the risk that an attacker can infer whether a given data point is included in (i.e., a member of) the training data. To answer this question, we propose Gotcha, a novel membership inference attack method specifically for code models. We investigate the membership leakage risk of code models. Our results reveal a worrying fact that the risk of membership leakage is high: although the previous attack methods are close to random guessing, Gotcha can predict the data membership with a high true positive rate of 0.95 and a low false positive rate of 0.10. We also show that the attacker's knowledge of the victim model (e.g., the model architecture and the pre-training data) impacts the success rate of attacks. Further analysis demonstrates that changing the decoding strategy can mitigate the risk of membership leakage. This study calls for more attention to understanding the privacy of code models and developing more effective countermeasures against such attacks.



## **6. Multi-round jailbreak attack on large language models**

cs.CL

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11533v1) [paper-pdf](http://arxiv.org/pdf/2410.11533v1)

**Authors**: Yihua Zhou, Xiaochuan Shi

**Abstract**: Ensuring the safety and alignment of large language models (LLMs) with human values is crucial for generating responses that are beneficial to humanity. While LLMs have the capability to identify and avoid harmful queries, they remain vulnerable to "jailbreak" attacks, where carefully crafted prompts can induce the generation of toxic content. Traditional single-round jailbreak attacks, such as GCG and AutoDAN, do not alter the sensitive words in the dangerous prompts. Although they can temporarily bypass the model's safeguards through prompt engineering, their success rate drops significantly as the LLM is further fine-tuned, and they cannot effectively circumvent static rule-based filters that remove the hazardous vocabulary.   In this study, to better understand jailbreak attacks, we introduce a multi-round jailbreak approach. This method can rewrite the dangerous prompts, decomposing them into a series of less harmful sub-questions to bypass the LLM's safety checks. We first use the LLM to perform a decomposition task, breaking down a set of natural language questions into a sequence of progressive sub-questions, which are then used to fine-tune the Llama3-8B model, enabling it to decompose hazardous prompts. The fine-tuned model is then used to break down the problematic prompt, and the resulting sub-questions are sequentially asked to the victim model. If the victim model rejects a sub-question, a new decomposition is generated, and the process is repeated until the final objective is achieved. Our experimental results show a 94\% success rate on the llama2-7B and demonstrate the effectiveness of this approach in circumventing static rule-based filters.



## **7. Jigsaw Puzzles: Splitting Harmful Questions to Jailbreak Large Language Models**

cs.CL

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11459v1) [paper-pdf](http://arxiv.org/pdf/2410.11459v1)

**Authors**: Hao Yang, Lizhen Qu, Ehsan Shareghi, Gholamreza Haffari

**Abstract**: Large language models (LLMs) have exhibited outstanding performance in engaging with humans and addressing complex questions by leveraging their vast implicit knowledge and robust reasoning capabilities. However, such models are vulnerable to jailbreak attacks, leading to the generation of harmful responses. Despite recent research on single-turn jailbreak strategies to facilitate the development of defence mechanisms, the challenge of revealing vulnerabilities under multi-turn setting remains relatively under-explored. In this work, we propose Jigsaw Puzzles (JSP), a straightforward yet effective multi-turn jailbreak strategy against the advanced LLMs. JSP splits questions into harmless fractions as the input of each turn, and requests LLMs to reconstruct and respond to questions under multi-turn interaction. Our experimental results demonstrate that the proposed JSP jailbreak bypasses original safeguards against explicitly harmful content, achieving an average attack success rate of 93.76% on 189 harmful queries across 5 advanced LLMs (Gemini-1.5-Pro, Llama-3.1-70B, GPT-4, GPT-4o, GPT-4o-mini). Moreover, JSP achieves a state-of-the-art attack success rate of 92% on GPT-4 on the harmful query benchmark, and exhibits strong resistant to defence strategies. Warning: this paper contains offensive examples.



## **8. Deciphering the Chaos: Enhancing Jailbreak Attacks via Adversarial Prompt Translation**

cs.LG

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11317v1) [paper-pdf](http://arxiv.org/pdf/2410.11317v1)

**Authors**: Qizhang Li, Xiaochen Yang, Wangmeng Zuo, Yiwen Guo

**Abstract**: Automatic adversarial prompt generation provides remarkable success in jailbreaking safely-aligned large language models (LLMs). Existing gradient-based attacks, while demonstrating outstanding performance in jailbreaking white-box LLMs, often generate garbled adversarial prompts with chaotic appearance. These adversarial prompts are difficult to transfer to other LLMs, hindering their performance in attacking unknown victim models. In this paper, for the first time, we delve into the semantic meaning embedded in garbled adversarial prompts and propose a novel method that "translates" them into coherent and human-readable natural language adversarial prompts. In this way, we can effectively uncover the semantic information that triggers vulnerabilities of the model and unambiguously transfer it to the victim model, without overlooking the adversarial information hidden in the garbled text, to enhance jailbreak attacks. It also offers a new approach to discovering effective designs for jailbreak prompts, advancing the understanding of jailbreak attacks. Experimental results demonstrate that our method significantly improves the success rate of jailbreak attacks against various safety-aligned LLMs and outperforms state-of-the-arts by large margins. With at most 10 queries, our method achieves an average attack success rate of 81.8% in attacking 7 commercial closed-source LLMs, including GPT and Claude-3 series, on HarmBench. Our method also achieves over 90% attack success rates against Llama-2-Chat models on AdvBench, despite their outstanding resistance to jailbreak attacks. Code at: https://github.com/qizhangli/Adversarial-Prompt-Translator.



## **9. Eyes Closed, Safety On: Protecting Multimodal LLMs via Image-to-Text Transformation**

cs.CV

ECCV2024 (Project Page: https://gyhdog99.github.io/projects/ecso/)

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2403.09572v4) [paper-pdf](http://arxiv.org/pdf/2403.09572v4)

**Authors**: Yunhao Gou, Kai Chen, Zhili Liu, Lanqing Hong, Hang Xu, Zhenguo Li, Dit-Yan Yeung, James T. Kwok, Yu Zhang

**Abstract**: Multimodal large language models (MLLMs) have shown impressive reasoning abilities. However, they are also more vulnerable to jailbreak attacks than their LLM predecessors. Although still capable of detecting the unsafe responses, we observe that safety mechanisms of the pre-aligned LLMs in MLLMs can be easily bypassed with the introduction of image features. To construct robust MLLMs, we propose ECSO (Eyes Closed, Safety On), a novel training-free protecting approach that exploits the inherent safety awareness of MLLMs, and generates safer responses via adaptively transforming unsafe images into texts to activate the intrinsic safety mechanism of pre-aligned LLMs in MLLMs. Experiments on five state-of-the-art (SoTA) MLLMs demonstrate that ECSO enhances model safety significantly (e.g.,, 37.6% improvement on the MM-SafetyBench (SD+OCR) and 71.3% on VLSafe with LLaVA-1.5-7B), while consistently maintaining utility results on common MLLM benchmarks. Furthermore, we show that ECSO can be used as a data engine to generate supervised-finetuning (SFT) data for MLLM alignment without extra human intervention.



## **10. Cognitive Overload Attack:Prompt Injection for Long Context**

cs.CL

40 pages, 31 Figures

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11272v1) [paper-pdf](http://arxiv.org/pdf/2410.11272v1)

**Authors**: Bibek Upadhayay, Vahid Behzadan, Amin Karbasi

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in performing tasks across various domains without needing explicit retraining. This capability, known as In-Context Learning (ICL), while impressive, exposes LLMs to a variety of adversarial prompts and jailbreaks that manipulate safety-trained LLMs into generating undesired or harmful output. In this paper, we propose a novel interpretation of ICL in LLMs through the lens of cognitive neuroscience, by drawing parallels between learning in human cognition with ICL. We applied the principles of Cognitive Load Theory in LLMs and empirically validate that similar to human cognition, LLMs also suffer from cognitive overload a state where the demand on cognitive processing exceeds the available capacity of the model, leading to potential errors. Furthermore, we demonstrated how an attacker can exploit ICL to jailbreak LLMs through deliberately designed prompts that induce cognitive overload on LLMs, thereby compromising the safety mechanisms of LLMs. We empirically validate this threat model by crafting various cognitive overload prompts and show that advanced models such as GPT-4, Claude-3.5 Sonnet, Claude-3 OPUS, Llama-3-70B-Instruct, Gemini-1.0-Pro, and Gemini-1.5-Pro can be successfully jailbroken, with attack success rates of up to 99.99%. Our findings highlight critical vulnerabilities in LLMs and underscore the urgency of developing robust safeguards. We propose integrating insights from cognitive load theory into the design and evaluation of LLMs to better anticipate and mitigate the risks of adversarial attacks. By expanding our experiments to encompass a broader range of models and by highlighting vulnerabilities in LLMs' ICL, we aim to ensure the development of safer and more reliable AI systems.



## **11. Archilles' Heel in Semi-open LLMs: Hiding Bottom against Recovery Attacks**

cs.LG

10 pages for main content of the paper

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11182v1) [paper-pdf](http://arxiv.org/pdf/2410.11182v1)

**Authors**: Hanbo Huang, Yihan Li, Bowen Jiang, Lin Liu, Ruoyu Sun, Zhuotao Liu, Shiyu Liang

**Abstract**: Closed-source large language models deliver strong performance but have limited downstream customizability. Semi-open models, combining both closed-source and public layers, were introduced to improve customizability. However, parameters in the closed-source layers are found vulnerable to recovery attacks. In this paper, we explore the design of semi-open models with fewer closed-source layers, aiming to increase customizability while ensuring resilience to recovery attacks. We analyze the contribution of closed-source layer to the overall resilience and theoretically prove that in a deep transformer-based model, there exists a transition layer such that even small recovery errors in layers before this layer can lead to recovery failure. Building on this, we propose \textbf{SCARA}, a novel approach that keeps only a few bottom layers as closed-source. SCARA employs a fine-tuning-free metric to estimate the maximum number of layers that can be publicly accessible for customization. We apply it to five models (1.3B to 70B parameters) to construct semi-open models, validating their customizability on six downstream tasks and assessing their resilience against various recovery attacks on sixteen benchmarks. We compare SCARA to baselines and observe that it generally improves downstream customization performance and offers similar resilience with over \textbf{10} times fewer closed-source parameters. We empirically investigate the existence of transition layers, analyze the effectiveness of our scheme and finally discuss its limitations.



## **12. Denial-of-Service Poisoning Attacks against Large Language Models**

cs.CR

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10760v1) [paper-pdf](http://arxiv.org/pdf/2410.10760v1)

**Authors**: Kuofeng Gao, Tianyu Pang, Chao Du, Yong Yang, Shu-Tao Xia, Min Lin

**Abstract**: Recent studies have shown that LLMs are vulnerable to denial-of-service (DoS) attacks, where adversarial inputs like spelling errors or non-semantic prompts trigger endless outputs without generating an [EOS] token. These attacks can potentially cause high latency and make LLM services inaccessible to other users or tasks. However, when there are speech-to-text interfaces (e.g., voice commands to a robot), executing such DoS attacks becomes challenging, as it is difficult to introduce spelling errors or non-semantic prompts through speech. A simple DoS attack in these scenarios would be to instruct the model to "Keep repeating Hello", but we observe that relying solely on natural instructions limits output length, which is bounded by the maximum length of the LLM's supervised finetuning (SFT) data. To overcome this limitation, we propose poisoning-based DoS (P-DoS) attacks for LLMs, demonstrating that injecting a single poisoned sample designed for DoS purposes can break the output length limit. For example, a poisoned sample can successfully attack GPT-4o and GPT-4o mini (via OpenAI's finetuning API) using less than $1, causing repeated outputs up to the maximum inference length (16K tokens, compared to 0.5K before poisoning). Additionally, we perform comprehensive ablation studies on open-source LLMs and extend our method to LLM agents, where attackers can control both the finetuning dataset and algorithm. Our findings underscore the urgent need for defenses against P-DoS attacks to secure LLMs. Our code is available at https://github.com/sail-sg/P-DoS.



## **13. Derail Yourself: Multi-turn LLM Jailbreak Attack through Self-discovered Clues**

cs.CL

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10700v1) [paper-pdf](http://arxiv.org/pdf/2410.10700v1)

**Authors**: Qibing Ren, Hao Li, Dongrui Liu, Zhanxu Xie, Xiaoya Lu, Yu Qiao, Lei Sha, Junchi Yan, Lizhuang Ma, Jing Shao

**Abstract**: This study exposes the safety vulnerabilities of Large Language Models (LLMs) in multi-turn interactions, where malicious users can obscure harmful intents across several queries. We introduce ActorAttack, a novel multi-turn attack method inspired by actor-network theory, which models a network of semantically linked actors as attack clues to generate diverse and effective attack paths toward harmful targets. ActorAttack addresses two main challenges in multi-turn attacks: (1) concealing harmful intents by creating an innocuous conversation topic about the actor, and (2) uncovering diverse attack paths towards the same harmful target by leveraging LLMs' knowledge to specify the correlated actors as various attack clues. In this way, ActorAttack outperforms existing single-turn and multi-turn attack methods across advanced aligned LLMs, even for GPT-o1. We will publish a dataset called SafeMTData, which includes multi-turn adversarial prompts and safety alignment data, generated by ActorAttack. We demonstrate that models safety-tuned using our safety dataset are more robust to multi-turn attacks. Code is available at https://github.com/renqibing/ActorAttack.



## **14. F2A: An Innovative Approach for Prompt Injection by Utilizing Feign Security Detection Agents**

cs.CR

1. Fixed typo in abstract 2. Provisionally completed the article  update to facilitate future version revisions

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.08776v2) [paper-pdf](http://arxiv.org/pdf/2410.08776v2)

**Authors**: Yupeng Ren

**Abstract**: With the rapid development of Large Language Models (LLMs), numerous mature applications of LLMs have emerged in the field of content safety detection. However, we have found that LLMs exhibit blind trust in safety detection agents. The general LLMs can be compromised by hackers with this vulnerability. Hence, this paper proposed an attack named Feign Agent Attack (F2A).Through such malicious forgery methods, adding fake safety detection results into the prompt, the defense mechanism of LLMs can be bypassed, thereby obtaining harmful content and hijacking the normal conversation. Continually, a series of experiments were conducted. In these experiments, the hijacking capability of F2A on LLMs was analyzed and demonstrated, exploring the fundamental reasons why LLMs blindly trust safety detection results. The experiments involved various scenarios where fake safety detection results were injected into prompts, and the responses were closely monitored to understand the extent of the vulnerability. Also, this paper provided a reasonable solution to this attack, emphasizing that it is important for LLMs to critically evaluate the results of augmented agents to prevent the generating harmful content. By doing so, the reliability and security can be significantly improved, protecting the LLMs from F2A.



## **15. On Calibration of LLM-based Guard Models for Reliable Content Moderation**

cs.CR

19 pages, 9 figures

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10414v1) [paper-pdf](http://arxiv.org/pdf/2410.10414v1)

**Authors**: Hongfu Liu, Hengguan Huang, Hao Wang, Xiangming Gu, Ye Wang

**Abstract**: Large language models (LLMs) pose significant risks due to the potential for generating harmful content or users attempting to evade guardrails. Existing studies have developed LLM-based guard models designed to moderate the input and output of threat LLMs, ensuring adherence to safety policies by blocking content that violates these protocols upon deployment. However, limited attention has been given to the reliability and calibration of such guard models. In this work, we empirically conduct comprehensive investigations of confidence calibration for 9 existing LLM-based guard models on 12 benchmarks in both user input and model output classification. Our findings reveal that current LLM-based guard models tend to 1) produce overconfident predictions, 2) exhibit significant miscalibration when subjected to jailbreak attacks, and 3) demonstrate limited robustness to the outputs generated by different types of response models. Additionally, we assess the effectiveness of post-hoc calibration methods to mitigate miscalibration. We demonstrate the efficacy of temperature scaling and, for the first time, highlight the benefits of contextual calibration for confidence calibration of guard models, particularly in the absence of validation sets. Our analysis and experiments underscore the limitations of current LLM-based guard models and provide valuable insights for the future development of well-calibrated guard models toward more reliable content moderation. We also advocate for incorporating reliability evaluation of confidence calibration when releasing future LLM-based guard models.



## **16. Jailbreak Instruction-Tuned LLMs via end-of-sentence MLP Re-weighting**

cs.CL

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10150v1) [paper-pdf](http://arxiv.org/pdf/2410.10150v1)

**Authors**: Yifan Luo, Zhennan Zhou, Meitan Wang, Bin Dong

**Abstract**: In this paper, we investigate the safety mechanisms of instruction fine-tuned large language models (LLMs). We discover that re-weighting MLP neurons can significantly compromise a model's safety, especially for MLPs in end-of-sentence inferences. We hypothesize that LLMs evaluate the harmfulness of prompts during end-of-sentence inferences, and MLP layers plays a critical role in this process. Based on this hypothesis, we develop 2 novel white-box jailbreak methods: a prompt-specific method and a prompt-general method. The prompt-specific method targets individual prompts and optimizes the attack on the fly, while the prompt-general method is pre-trained offline and can generalize to unseen harmful prompts. Our methods demonstrate robust performance across 7 popular open-source LLMs, size ranging from 2B to 72B. Furthermore, our study provides insights into vulnerabilities of instruction-tuned LLM's safety and deepens the understanding of the internal mechanisms of LLMs.



## **17. White-box Multimodal Jailbreaks Against Large Vision-Language Models**

cs.CV

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2405.17894v2) [paper-pdf](http://arxiv.org/pdf/2405.17894v2)

**Authors**: Ruofan Wang, Xingjun Ma, Hanxu Zhou, Chuanjun Ji, Guangnan Ye, Yu-Gang Jiang

**Abstract**: Recent advancements in Large Vision-Language Models (VLMs) have underscored their superiority in various multimodal tasks. However, the adversarial robustness of VLMs has not been fully explored. Existing methods mainly assess robustness through unimodal adversarial attacks that perturb images, while assuming inherent resilience against text-based attacks. Different from existing attacks, in this work we propose a more comprehensive strategy that jointly attacks both text and image modalities to exploit a broader spectrum of vulnerability within VLMs. Specifically, we propose a dual optimization objective aimed at guiding the model to generate affirmative responses with high toxicity. Our attack method begins by optimizing an adversarial image prefix from random noise to generate diverse harmful responses in the absence of text input, thus imbuing the image with toxic semantics. Subsequently, an adversarial text suffix is integrated and co-optimized with the adversarial image prefix to maximize the probability of eliciting affirmative responses to various harmful instructions. The discovered adversarial image prefix and text suffix are collectively denoted as a Universal Master Key (UMK). When integrated into various malicious queries, UMK can circumvent the alignment defenses of VLMs and lead to the generation of objectionable content, known as jailbreaks. The experimental results demonstrate that our universal attack strategy can effectively jailbreak MiniGPT-4 with a 96% success rate, highlighting the vulnerability of VLMs and the urgent need for new alignment strategies.



## **18. BlackDAN: A Black-Box Multi-Objective Approach for Effective and Contextual Jailbreaking of Large Language Models**

cs.CR

**SubmitDate**: 2024-10-13    [abs](http://arxiv.org/abs/2410.09804v1) [paper-pdf](http://arxiv.org/pdf/2410.09804v1)

**Authors**: Xinyuan Wang, Victor Shea-Jay Huang, Renmiao Chen, Hao Wang, Chengwei Pan, Lei Sha, Minlie Huang

**Abstract**: While large language models (LLMs) exhibit remarkable capabilities across various tasks, they encounter potential security risks such as jailbreak attacks, which exploit vulnerabilities to bypass security measures and generate harmful outputs. Existing jailbreak strategies mainly focus on maximizing attack success rate (ASR), frequently neglecting other critical factors, including the relevance of the jailbreak response to the query and the level of stealthiness. This narrow focus on single objectives can result in ineffective attacks that either lack contextual relevance or are easily recognizable. In this work, we introduce BlackDAN, an innovative black-box attack framework with multi-objective optimization, aiming to generate high-quality prompts that effectively facilitate jailbreaking while maintaining contextual relevance and minimizing detectability. BlackDAN leverages Multiobjective Evolutionary Algorithms (MOEAs), specifically the NSGA-II algorithm, to optimize jailbreaks across multiple objectives including ASR, stealthiness, and semantic relevance. By integrating mechanisms like mutation, crossover, and Pareto-dominance, BlackDAN provides a transparent and interpretable process for generating jailbreaks. Furthermore, the framework allows customization based on user preferences, enabling the selection of prompts that balance harmfulness, relevance, and other factors. Experimental results demonstrate that BlackDAN outperforms traditional single-objective methods, yielding higher success rates and improved robustness across various LLMs and multimodal LLMs, while ensuring jailbreak responses are both relevant and less detectable.



## **19. 'Quis custodiet ipsos custodes?' Who will watch the watchmen? On Detecting AI-generated peer-reviews**

cs.CL

EMNLP Main, 17 pages, 5 figures, 9 tables

**SubmitDate**: 2024-10-13    [abs](http://arxiv.org/abs/2410.09770v1) [paper-pdf](http://arxiv.org/pdf/2410.09770v1)

**Authors**: Sandeep Kumar, Mohit Sahu, Vardhan Gacche, Tirthankar Ghosal, Asif Ekbal

**Abstract**: The integrity of the peer-review process is vital for maintaining scientific rigor and trust within the academic community. With the steady increase in the usage of large language models (LLMs) like ChatGPT in academic writing, there is a growing concern that AI-generated texts could compromise scientific publishing, including peer-reviews. Previous works have focused on generic AI-generated text detection or have presented an approach for estimating the fraction of peer-reviews that can be AI-generated. Our focus here is to solve a real-world problem by assisting the editor or chair in determining whether a review is written by ChatGPT or not. To address this, we introduce the Term Frequency (TF) model, which posits that AI often repeats tokens, and the Review Regeneration (RR) model, which is based on the idea that ChatGPT generates similar outputs upon re-prompting. We stress test these detectors against token attack and paraphrasing. Finally, we propose an effective defensive strategy to reduce the effect of paraphrasing on our models. Our findings suggest both our proposed methods perform better than the other AI text detectors. Our RR model is more robust, although our TF model performs better than the RR model without any attacks. We make our code, dataset, and model public.



## **20. Targeted Vaccine: Safety Alignment for Large Language Models against Harmful Fine-Tuning via Layer-wise Perturbation**

cs.LG

**SubmitDate**: 2024-10-13    [abs](http://arxiv.org/abs/2410.09760v1) [paper-pdf](http://arxiv.org/pdf/2410.09760v1)

**Authors**: Guozhi Liu, Weiwei Lin, Tiansheng Huang, Ruichao Mo, Qi Mu, Li Shen

**Abstract**: Harmful fine-tuning attack poses a serious threat to the online fine-tuning service. Vaccine, a recent alignment-stage defense, applies uniform perturbation to all layers of embedding to make the model robust to the simulated embedding drift. However, applying layer-wise uniform perturbation may lead to excess perturbations for some particular safety-irrelevant layers, resulting in defense performance degradation and unnecessary memory consumption. To address this limitation, we propose Targeted Vaccine (T-Vaccine), a memory-efficient safety alignment method that applies perturbation to only selected layers of the model. T-Vaccine follows two core steps: First, it uses gradient norm as a statistical metric to identify the safety-critical layers. Second, instead of applying uniform perturbation across all layers, T-Vaccine only applies perturbation to the safety-critical layers while keeping other layers frozen during training. Results show that T-Vaccine outperforms Vaccine in terms of both defense effectiveness and resource efficiency. Comparison with other defense baselines, e.g., RepNoise and TAR also demonstrate the superiority of T-Vaccine. Notably, T-Vaccine is the first defense that can address harmful fine-tuning issues for a 7B pre-trained models trained on consumer GPUs with limited memory (e.g., RTX 4090). Our code is available at https://github.com/Lslland/T-Vaccine.



## **21. Weak-to-Strong Backdoor Attack for Large Language Models**

cs.CR

**SubmitDate**: 2024-10-13    [abs](http://arxiv.org/abs/2409.17946v3) [paper-pdf](http://arxiv.org/pdf/2409.17946v3)

**Authors**: Shuai Zhao, Leilei Gan, Zhongliang Guo, Xiaobao Wu, Luwei Xiao, Xiaoyu Xu, Cong-Duy Nguyen, Luu Anh Tuan

**Abstract**: Despite being widely applied due to their exceptional capabilities, Large Language Models (LLMs) have been proven to be vulnerable to backdoor attacks. These attacks introduce targeted vulnerabilities into LLMs by poisoning training samples and full-parameter fine-tuning. However, this kind of backdoor attack is limited since they require significant computational resources, especially as the size of LLMs increases. Besides, parameter-efficient fine-tuning (PEFT) offers an alternative but the restricted parameter updating may impede the alignment of triggers with target labels. In this study, we first verify that backdoor attacks with PEFT may encounter challenges in achieving feasible performance. To address these issues and improve the effectiveness of backdoor attacks with PEFT, we propose a novel backdoor attack algorithm from weak to strong based on feature alignment-enhanced knowledge distillation (W2SAttack). Specifically, we poison small-scale language models through full-parameter fine-tuning to serve as the teacher model. The teacher model then covertly transfers the backdoor to the large-scale student model through feature alignment-enhanced knowledge distillation, which employs PEFT. Theoretical analysis reveals that W2SAttack has the potential to augment the effectiveness of backdoor attacks. We demonstrate the superior performance of W2SAttack on classification tasks across four language models, four backdoor attack algorithms, and two different architectures of teacher models. Experimental results indicate success rates close to 100% for backdoor attacks targeting PEFT.



## **22. VLFeedback: A Large-Scale AI Feedback Dataset for Large Vision-Language Models Alignment**

cs.CV

EMNLP 2024 Main Conference camera-ready version. This article  supersedes arXiv:2312.10665

**SubmitDate**: 2024-10-12    [abs](http://arxiv.org/abs/2410.09421v1) [paper-pdf](http://arxiv.org/pdf/2410.09421v1)

**Authors**: Lei Li, Zhihui Xie, Mukai Li, Shunian Chen, Peiyi Wang, Liang Chen, Yazheng Yang, Benyou Wang, Lingpeng Kong, Qi Liu

**Abstract**: As large vision-language models (LVLMs) evolve rapidly, the demand for high-quality and diverse data to align these models becomes increasingly crucial. However, the creation of such data with human supervision proves costly and time-intensive. In this paper, we investigate the efficacy of AI feedback to scale supervision for aligning LVLMs. We introduce VLFeedback, the first large-scale vision-language feedback dataset, comprising over 82K multi-modal instructions and comprehensive rationales generated by off-the-shelf models without human annotations. To evaluate the effectiveness of AI feedback for vision-language alignment, we train Silkie, an LVLM fine-tuned via direct preference optimization on VLFeedback. Silkie showcases exceptional performance regarding helpfulness, visual faithfulness, and safety metrics. It outperforms its base model by 6.9\% and 9.5\% in perception and cognition tasks, reduces hallucination issues on MMHal-Bench, and exhibits enhanced resilience against red-teaming attacks. Furthermore, our analysis underscores the advantage of AI feedback, particularly in fostering preference diversity to deliver more comprehensive improvements. Our dataset, training code and models are available at https://vlf-silkie.github.io.



## **23. Don't Say No: Jailbreaking LLM by Suppressing Refusal**

cs.CL

Update results on Llama3, Llama3.1, Gemma2, Mistral, Qwen2 models and  upon JailbreakBnech, MaliciousInstruct datasets

**SubmitDate**: 2024-10-12    [abs](http://arxiv.org/abs/2404.16369v2) [paper-pdf](http://arxiv.org/pdf/2404.16369v2)

**Authors**: Yukai Zhou, Zhijie Huang, Feiyang Lu, Zhan Qin, Wenjie Wang

**Abstract**: Ensuring the safety alignment of Large Language Models (LLMs) is crucial to generating responses consistent with human values. Despite their ability to recognize and avoid harmful queries, LLMs are vulnerable to jailbreaking attacks, where carefully crafted prompts seduce them to produce toxic content. One category of jailbreak attacks is reformulating the task as an optimization by eliciting the LLM to generate affirmative responses. However, such optimization objective has its own limitations, such as the restriction on the predefined objectionable behaviors, leading to suboptimal attack performance. In this study, we first uncover the reason why vanilla target loss is not optimal, then we explore and enhance the loss objective and introduce the DSN (Don't Say No) attack, which achieves successful attack by suppressing refusal. Another challenge in studying jailbreak attacks is the evaluation, as it is difficult to directly and accurately assess the harmfulness of the responses. The existing evaluation such as refusal keyword matching reveals numerous false positive and false negative instances. To overcome this challenge, we propose an Ensemble Evaluation pipeline that novelly incorporates Natural Language Inference (NLI) contradiction assessment and two external LLM evaluators. Extensive experiments demonstrate the potential of the DSN and effectiveness of Ensemble Evaluation compared to baseline methods.



## **24. Can a large language model be a gaslighter?**

cs.CR

10/26 (Main Body/Total), 8 figures

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2410.09181v1) [paper-pdf](http://arxiv.org/pdf/2410.09181v1)

**Authors**: Wei Li, Luyao Zhu, Yang Song, Ruixi Lin, Rui Mao, Yang You

**Abstract**: Large language models (LLMs) have gained human trust due to their capabilities and helpfulness. However, this in turn may allow LLMs to affect users' mindsets by manipulating language. It is termed as gaslighting, a psychological effect. In this work, we aim to investigate the vulnerability of LLMs under prompt-based and fine-tuning-based gaslighting attacks. Therefore, we propose a two-stage framework DeepCoG designed to: 1) elicit gaslighting plans from LLMs with the proposed DeepGaslighting prompting template, and 2) acquire gaslighting conversations from LLMs through our Chain-of-Gaslighting method. The gaslighting conversation dataset along with a corresponding safe dataset is applied to fine-tuning-based attacks on open-source LLMs and anti-gaslighting safety alignment on these LLMs. Experiments demonstrate that both prompt-based and fine-tuning-based attacks transform three open-source LLMs into gaslighters. In contrast, we advanced three safety alignment strategies to strengthen (by 12.05%) the safety guardrail of LLMs. Our safety alignment strategies have minimal impacts on the utility of LLMs. Empirical studies indicate that an LLM may be a potential gaslighter, even if it passed the harmfulness test on general dangerous queries.



## **25. Defending Against Social Engineering Attacks in the Age of LLMs**

cs.CL

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2406.12263v2) [paper-pdf](http://arxiv.org/pdf/2406.12263v2)

**Authors**: Lin Ai, Tharindu Kumarage, Amrita Bhattacharjee, Zizhou Liu, Zheng Hui, Michael Davinroy, James Cook, Laura Cassani, Kirill Trapeznikov, Matthias Kirchner, Arslan Basharat, Anthony Hoogs, Joshua Garland, Huan Liu, Julia Hirschberg

**Abstract**: The proliferation of Large Language Models (LLMs) poses challenges in detecting and mitigating digital deception, as these models can emulate human conversational patterns and facilitate chat-based social engineering (CSE) attacks. This study investigates the dual capabilities of LLMs as both facilitators and defenders against CSE threats. We develop a novel dataset, SEConvo, simulating CSE scenarios in academic and recruitment contexts, and designed to examine how LLMs can be exploited in these situations. Our findings reveal that, while off-the-shelf LLMs generate high-quality CSE content, their detection capabilities are suboptimal, leading to increased operational costs for defense. In response, we propose ConvoSentinel, a modular defense pipeline that improves detection at both the message and the conversation levels, offering enhanced adaptability and cost-effectiveness. The retrieval-augmented module in ConvoSentinel identifies malicious intent by comparing messages to a database of similar conversations, enhancing CSE detection at all stages. Our study highlights the need for advanced strategies to leverage LLMs in cybersecurity.



## **26. AttnGCG: Enhancing Jailbreaking Attacks on LLMs with Attention Manipulation**

cs.CL

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2410.09040v1) [paper-pdf](http://arxiv.org/pdf/2410.09040v1)

**Authors**: Zijun Wang, Haoqin Tu, Jieru Mei, Bingchen Zhao, Yisen Wang, Cihang Xie

**Abstract**: This paper studies the vulnerabilities of transformer-based Large Language Models (LLMs) to jailbreaking attacks, focusing specifically on the optimization-based Greedy Coordinate Gradient (GCG) strategy. We first observe a positive correlation between the effectiveness of attacks and the internal behaviors of the models. For instance, attacks tend to be less effective when models pay more attention to system prompts designed to ensure LLM safety alignment. Building on this discovery, we introduce an enhanced method that manipulates models' attention scores to facilitate LLM jailbreaking, which we term AttnGCG. Empirically, AttnGCG shows consistent improvements in attack efficacy across diverse LLMs, achieving an average increase of ~7% in the Llama-2 series and ~10% in the Gemma series. Our strategy also demonstrates robust attack transferability against both unseen harmful goals and black-box LLMs like GPT-3.5 and GPT-4. Moreover, we note our attention-score visualization is more interpretable, allowing us to gain better insights into how our targeted attention manipulation facilitates more effective jailbreaking. We release the code at https://github.com/UCSC-VLAA/AttnGCG-attack.



## **27. Controlling Whisper: Universal Acoustic Adversarial Attacks to Control Speech Foundation Models**

cs.SD

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2407.04482v2) [paper-pdf](http://arxiv.org/pdf/2407.04482v2)

**Authors**: Vyas Raina, Mark Gales

**Abstract**: Speech enabled foundation models, either in the form of flexible speech recognition based systems or audio-prompted large language models (LLMs), are becoming increasingly popular. One of the interesting aspects of these models is their ability to perform tasks other than automatic speech recognition (ASR) using an appropriate prompt. For example, the OpenAI Whisper model can perform both speech transcription and speech translation. With the development of audio-prompted LLMs there is the potential for even greater control options. In this work we demonstrate that with this greater flexibility the systems can be susceptible to model-control adversarial attacks. Without any access to the model prompt it is possible to modify the behaviour of the system by appropriately changing the audio input. To illustrate this risk, we demonstrate that it is possible to prepend a short universal adversarial acoustic segment to any input speech signal to override the prompt setting of an ASR foundation model. Specifically, we successfully use a universal adversarial acoustic segment to control Whisper to always perform speech translation, despite being set to perform speech transcription. Overall, this work demonstrates a new form of adversarial attack on multi-tasking speech enabled foundation models that needs to be considered prior to the deployment of this form of model.



## **28. On the Adversarial Transferability of Generalized "Skip Connections"**

cs.LG

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2410.08950v1) [paper-pdf](http://arxiv.org/pdf/2410.08950v1)

**Authors**: Yisen Wang, Yichuan Mo, Dongxian Wu, Mingjie Li, Xingjun Ma, Zhouchen Lin

**Abstract**: Skip connection is an essential ingredient for modern deep models to be deeper and more powerful. Despite their huge success in normal scenarios (state-of-the-art classification performance on natural examples), we investigate and identify an interesting property of skip connections under adversarial scenarios, namely, the use of skip connections allows easier generation of highly transferable adversarial examples. Specifically, in ResNet-like models (with skip connections), we find that using more gradients from the skip connections rather than the residual modules according to a decay factor during backpropagation allows one to craft adversarial examples with high transferability. The above method is termed as Skip Gradient Method (SGM). Although starting from ResNet-like models in vision domains, we further extend SGM to more advanced architectures, including Vision Transformers (ViTs) and models with length-varying paths and other domains, i.e. natural language processing. We conduct comprehensive transfer attacks against various models including ResNets, Transformers, Inceptions, Neural Architecture Search, and Large Language Models (LLMs). We show that employing SGM can greatly improve the transferability of crafted attacks in almost all cases. Furthermore, considering the big complexity for practical use, we further demonstrate that SGM can even improve the transferability on ensembles of models or targeted attacks and the stealthiness against current defenses. At last, we provide theoretical explanations and empirical insights on how SGM works. Our findings not only motivate new adversarial research into the architectural characteristics of models but also open up further challenges for secure model architecture design. Our code is available at https://github.com/mo666666/SGM.



## **29. Do Unlearning Methods Remove Information from Language Model Weights?**

cs.LG

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2410.08827v1) [paper-pdf](http://arxiv.org/pdf/2410.08827v1)

**Authors**: Aghyad Deeb, Fabien Roger

**Abstract**: Large Language Models' knowledge of how to perform cyber-security attacks, create bioweapons, and manipulate humans poses risks of misuse. Previous work has proposed methods to unlearn this knowledge. Historically, it has been unclear whether unlearning techniques are removing information from the model weights or just making it harder to access. To disentangle these two objectives, we propose an adversarial evaluation method to test for the removal of information from model weights: we give an attacker access to some facts that were supposed to be removed, and using those, the attacker tries to recover other facts from the same distribution that cannot be guessed from the accessible facts. We show that using fine-tuning on the accessible facts can recover 88% of the pre-unlearning accuracy when applied to current unlearning methods, revealing the limitations of these methods in removing information from the model weights.



## **30. PoisonBench: Assessing Large Language Model Vulnerability to Data Poisoning**

cs.CR

Tingchen Fu and Fazl Barez are core research contributors

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2410.08811v1) [paper-pdf](http://arxiv.org/pdf/2410.08811v1)

**Authors**: Tingchen Fu, Mrinank Sharma, Philip Torr, Shay B. Cohen, David Krueger, Fazl Barez

**Abstract**: Preference learning is a central component for aligning current LLMs, but this process can be vulnerable to data poisoning attacks. To address this concern, we introduce PoisonBench, a benchmark for evaluating large language models' susceptibility to data poisoning during preference learning. Data poisoning attacks can manipulate large language model responses to include hidden malicious content or biases, potentially causing the model to generate harmful or unintended outputs while appearing to function normally. We deploy two distinct attack types across eight realistic scenarios, assessing 21 widely-used models. Our findings reveal concerning trends: (1) Scaling up parameter size does not inherently enhance resilience against poisoning attacks; (2) There exists a log-linear relationship between the effects of the attack and the data poison ratio; (3) The effect of data poisoning can generalize to extrapolated triggers that are not included in the poisoned data. These results expose weaknesses in current preference learning techniques, highlighting the urgent need for more robust defenses against malicious models and data manipulation.



## **31. RePD: Defending Jailbreak Attack through a Retrieval-based Prompt Decomposition Process**

cs.CR

arXiv admin note: text overlap with arXiv:2403.04783 by other authors

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2410.08660v1) [paper-pdf](http://arxiv.org/pdf/2410.08660v1)

**Authors**: Peiran Wang, Xiaogeng Liu, Chaowei Xiao

**Abstract**: In this study, we introduce RePD, an innovative attack Retrieval-based Prompt Decomposition framework designed to mitigate the risk of jailbreak attacks on large language models (LLMs). Despite rigorous pretraining and finetuning focused on ethical alignment, LLMs are still susceptible to jailbreak exploits. RePD operates on a one-shot learning model, wherein it accesses a database of pre-collected jailbreak prompt templates to identify and decompose harmful inquiries embedded within user prompts. This process involves integrating the decomposition of the jailbreak prompt into the user's original query into a one-shot learning example to effectively teach the LLM to discern and separate malicious components. Consequently, the LLM is equipped to first neutralize any potentially harmful elements before addressing the user's prompt in a manner that aligns with its ethical guidelines. RePD is versatile and compatible with a variety of open-source LLMs acting as agents. Through comprehensive experimentation with both harmful and benign prompts, we have demonstrated the efficacy of our proposed RePD in enhancing the resilience of LLMs against jailbreak attacks, without compromising their performance in responding to typical user requests.



## **32. ART: Automatic Red-teaming for Text-to-Image Models to Protect Benign Users**

cs.CR

Accepted by NeurIPS 2024

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2405.19360v3) [paper-pdf](http://arxiv.org/pdf/2405.19360v3)

**Authors**: Guanlin Li, Kangjie Chen, Shudong Zhang, Jie Zhang, Tianwei Zhang

**Abstract**: Large-scale pre-trained generative models are taking the world by storm, due to their abilities in generating creative content. Meanwhile, safeguards for these generative models are developed, to protect users' rights and safety, most of which are designed for large language models. Existing methods primarily focus on jailbreak and adversarial attacks, which mainly evaluate the model's safety under malicious prompts. Recent work found that manually crafted safe prompts can unintentionally trigger unsafe generations. To further systematically evaluate the safety risks of text-to-image models, we propose a novel Automatic Red-Teaming framework, ART. Our method leverages both vision language model and large language model to establish a connection between unsafe generations and their prompts, thereby more efficiently identifying the model's vulnerabilities. With our comprehensive experiments, we reveal the toxicity of the popular open-source text-to-image models. The experiments also validate the effectiveness, adaptability, and great diversity of ART. Additionally, we introduce three large-scale red-teaming datasets for studying the safety risks associated with text-to-image models. Datasets and models can be found in https://github.com/GuanlinLee/ART.



## **33. Cross-modality Information Check for Detecting Jailbreaking in Multimodal Large Language Models**

cs.CL

12 pages, 9 figures

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2407.21659v3) [paper-pdf](http://arxiv.org/pdf/2407.21659v3)

**Authors**: Yue Xu, Xiuyuan Qi, Zhan Qin, Wenjie Wang

**Abstract**: Multimodal Large Language Models (MLLMs) extend the capacity of LLMs to understand multimodal information comprehensively, achieving remarkable performance in many vision-centric tasks. Despite that, recent studies have shown that these models are susceptible to jailbreak attacks, which refer to an exploitative technique where malicious users can break the safety alignment of the target model and generate misleading and harmful answers. This potential threat is caused by both the inherent vulnerabilities of LLM and the larger attack scope introduced by vision input. To enhance the security of MLLMs against jailbreak attacks, researchers have developed various defense techniques. However, these methods either require modifications to the model's internal structure or demand significant computational resources during the inference phase. Multimodal information is a double-edged sword. While it increases the risk of attacks, it also provides additional data that can enhance safeguards. Inspired by this, we propose Cross-modality Information DEtectoR (CIDER), a plug-and-play jailbreaking detector designed to identify maliciously perturbed image inputs, utilizing the cross-modal similarity between harmful queries and adversarial images. CIDER is independent of the target MLLMs and requires less computation cost. Extensive experimental results demonstrate the effectiveness and efficiency of CIDER, as well as its transferability to both white-box and black-box MLLMs.



## **34. The Last Iterate Advantage: Empirical Auditing and Principled Heuristic Analysis of Differentially Private SGD**

cs.CR

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2410.06186v2) [paper-pdf](http://arxiv.org/pdf/2410.06186v2)

**Authors**: Thomas Steinke, Milad Nasr, Arun Ganesh, Borja Balle, Christopher A. Choquette-Choo, Matthew Jagielski, Jamie Hayes, Abhradeep Guha Thakurta, Adam Smith, Andreas Terzis

**Abstract**: We propose a simple heuristic privacy analysis of noisy clipped stochastic gradient descent (DP-SGD) in the setting where only the last iterate is released and the intermediate iterates remain hidden. Namely, our heuristic assumes a linear structure for the model.   We show experimentally that our heuristic is predictive of the outcome of privacy auditing applied to various training procedures. Thus it can be used prior to training as a rough estimate of the final privacy leakage. We also probe the limitations of our heuristic by providing some artificial counterexamples where it underestimates the privacy leakage.   The standard composition-based privacy analysis of DP-SGD effectively assumes that the adversary has access to all intermediate iterates, which is often unrealistic. However, this analysis remains the state of the art in practice. While our heuristic does not replace a rigorous privacy analysis, it illustrates the large gap between the best theoretical upper bounds and the privacy auditing lower bounds and sets a target for further work to improve the theoretical privacy analyses. We also empirically support our heuristic and show existing privacy auditing attacks are bounded by our heuristic analysis in both vision and language tasks.



## **35. APOLLO: A GPT-based tool to detect phishing emails and generate explanations that warn users**

cs.HC

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2410.07997v1) [paper-pdf](http://arxiv.org/pdf/2410.07997v1)

**Authors**: Giuseppe Desolda, Francesco Greco, Luca Viganò

**Abstract**: Phishing is one of the most prolific cybercriminal activities, with attacks becoming increasingly sophisticated. It is, therefore, imperative to explore novel technologies to improve user protection across both technical and human dimensions. Large Language Models (LLMs) offer significant promise for text processing in various domains, but their use for defense against phishing attacks still remains scarcely explored. In this paper, we present APOLLO, a tool based on OpenAI's GPT-4o to detect phishing emails and generate explanation messages to users about why a specific email is dangerous, thus improving their decision-making capabilities. We have evaluated the performance of APOLLO in classifying phishing emails; the results show that the LLM models have exemplary capabilities in classifying phishing emails (97 percent accuracy in the case of GPT-4o) and that this performance can be further improved by integrating data from third-party services, resulting in a near-perfect classification rate (99 percent accuracy). To assess the perception of the explanations generated by this tool, we also conducted a study with 20 participants, comparing four different explanations presented as phishing warnings. We compared the LLM-generated explanations to four baselines: a manually crafted warning, and warnings from Chrome, Firefox, and Edge browsers. The results show that not only the LLM-generated explanations were perceived as high quality, but also that they can be more understandable, interesting, and trustworthy than the baselines. These findings suggest that using LLMs as a defense against phishing is a very promising approach, with APOLLO representing a proof of concept in this research direction.



## **36. Towards Assurance of LLM Adversarial Robustness using Ontology-Driven Argumentation**

cs.AI

To be published in xAI 2024, late-breaking track

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2410.07962v1) [paper-pdf](http://arxiv.org/pdf/2410.07962v1)

**Authors**: Tomas Bueno Momcilovic, Beat Buesser, Giulio Zizzo, Mark Purcell, Dian Balta

**Abstract**: Despite the impressive adaptability of large language models (LLMs), challenges remain in ensuring their security, transparency, and interpretability. Given their susceptibility to adversarial attacks, LLMs need to be defended with an evolving combination of adversarial training and guardrails. However, managing the implicit and heterogeneous knowledge for continuously assuring robustness is difficult. We introduce a novel approach for assurance of the adversarial robustness of LLMs based on formal argumentation. Using ontologies for formalization, we structure state-of-the-art attacks and defenses, facilitating the creation of a human-readable assurance case, and a machine-readable representation. We demonstrate its application with examples in English language and code translation tasks, and provide implications for theory and practice, by targeting engineers, data scientists, users, and auditors.



## **37. Protecting Your LLMs with Information Bottleneck**

cs.CL

Accepted by Neural Information Processing Systems (NeurIPS 2024)

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2404.13968v3) [paper-pdf](http://arxiv.org/pdf/2404.13968v3)

**Authors**: Zichuan Liu, Zefan Wang, Linjie Xu, Jinyu Wang, Lei Song, Tianchun Wang, Chunlin Chen, Wei Cheng, Jiang Bian

**Abstract**: The advent of large language models (LLMs) has revolutionized the field of natural language processing, yet they might be attacked to produce harmful content. Despite efforts to ethically align LLMs, these are often fragile and can be circumvented by jailbreaking attacks through optimized or manual adversarial prompts. To address this, we introduce the Information Bottleneck Protector (IBProtector), a defense mechanism grounded in the information bottleneck principle, and we modify the objective to avoid trivial solutions. The IBProtector selectively compresses and perturbs prompts, facilitated by a lightweight and trainable extractor, preserving only essential information for the target LLMs to respond with the expected answer. Moreover, we further consider a situation where the gradient is not visible to be compatible with any LLM. Our empirical evaluations show that IBProtector outperforms current defense methods in mitigating jailbreak attempts, without overly affecting response quality or inference speed. Its effectiveness and adaptability across various attack methods and target LLMs underscore the potential of IBProtector as a novel, transferable defense that bolsters the security of LLMs without requiring modifications to the underlying models.



## **38. Universally Optimal Watermarking Schemes for LLMs: from Theory to Practice**

cs.CR

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2410.02890v2) [paper-pdf](http://arxiv.org/pdf/2410.02890v2)

**Authors**: Haiyun He, Yepeng Liu, Ziqiao Wang, Yongyi Mao, Yuheng Bu

**Abstract**: Large Language Models (LLMs) boosts human efficiency but also poses misuse risks, with watermarking serving as a reliable method to differentiate AI-generated content from human-created text. In this work, we propose a novel theoretical framework for watermarking LLMs. Particularly, we jointly optimize both the watermarking scheme and detector to maximize detection performance, while controlling the worst-case Type-I error and distortion in the watermarked text. Within our framework, we characterize the universally minimum Type-II error, showing a fundamental trade-off between detection performance and distortion. More importantly, we identify the optimal type of detectors and watermarking schemes. Building upon our theoretical analysis, we introduce a practical, model-agnostic and computationally efficient token-level watermarking algorithm that invokes a surrogate model and the Gumbel-max trick. Empirical results on Llama-13B and Mistral-8$\times$7B demonstrate the effectiveness of our method. Furthermore, we also explore how robustness can be integrated into our theoretical framework, which provides a foundation for designing future watermarking systems with improved resilience to adversarial attacks.



## **39. Mind Your Questions! Towards Backdoor Attacks on Text-to-Visualization Models**

cs.CR

11 pages, 4 figures

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2410.06782v2) [paper-pdf](http://arxiv.org/pdf/2410.06782v2)

**Authors**: Shuaimin Li, Yuanfeng Song, Xuanang Chen, Anni Peng, Zhuoyue Wan, Chen Jason Zhang, Raymond Chi-Wing Wong

**Abstract**: Text-to-visualization (text-to-vis) models have become valuable tools in the era of big data, enabling users to generate data visualizations and make informed decisions through natural language queries (NLQs). Despite their widespread application, the security vulnerabilities of these models have been largely overlooked. To address this gap, we propose VisPoison, a novel framework designed to identify these vulnerabilities of current text-to-vis models systematically. VisPoison introduces two types of triggers that activate three distinct backdoor attacks, potentially leading to data exposure, misleading visualizations, or denial-of-service (DoS) incidents. The framework features both proactive and passive attack mechanisms: proactive attacks leverage rare-word triggers to access confidential data, while passive attacks, triggered unintentionally by users, exploit a first-word trigger method, causing errors or DoS events in visualizations. Through extensive experiments on both trainable and in-context learning (ICL)-based text-to-vis models, \textit{VisPoison} achieves attack success rates of over 90\%, highlighting the security problem of current text-to-vis models. Additionally, we explore two types of defense mechanisms against these attacks, but the results show that existing countermeasures are insufficient, underscoring the pressing need for more robust security solutions in text-to-vis systems.



## **40. Detecting Training Data of Large Language Models via Expectation Maximization**

cs.CL

14 pages

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2410.07582v1) [paper-pdf](http://arxiv.org/pdf/2410.07582v1)

**Authors**: Gyuwan Kim, Yang Li, Evangelia Spiliopoulou, Jie Ma, Miguel Ballesteros, William Yang Wang

**Abstract**: The widespread deployment of large language models (LLMs) has led to impressive advancements, yet information about their training data, a critical factor in their performance, remains undisclosed. Membership inference attacks (MIAs) aim to determine whether a specific instance was part of a target model's training data. MIAs can offer insights into LLM outputs and help detect and address concerns such as data contamination and compliance with privacy and copyright standards. However, applying MIAs to LLMs presents unique challenges due to the massive scale of pre-training data and the ambiguous nature of membership. Additionally, creating appropriate benchmarks to evaluate MIA methods is not straightforward, as training and test data distributions are often unknown. In this paper, we introduce EM-MIA, a novel MIA method for LLMs that iteratively refines membership scores and prefix scores via an expectation-maximization algorithm, leveraging the duality that the estimates of these scores can be improved by each other. Membership scores and prefix scores assess how each instance is likely to be a member and discriminative as a prefix, respectively. Our method achieves state-of-the-art results on the WikiMIA dataset. To further evaluate EM-MIA, we present OLMoMIA, a benchmark built from OLMo resources, which allows us to control the difficulty of MIA tasks with varying degrees of overlap between training and test data distributions. We believe that EM-MIA serves as a robust MIA method for LLMs and that OLMoMIA provides a valuable resource for comprehensively evaluating MIA approaches, thereby driving future research in this critical area.



## **41. Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning**

cs.CL

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.07163v1) [paper-pdf](http://arxiv.org/pdf/2410.07163v1)

**Authors**: Chongyu Fan, Jiancheng Liu, Licong Lin, Jinghan Jia, Ruiqi Zhang, Song Mei, Sijia Liu

**Abstract**: In this work, we address the problem of large language model (LLM) unlearning, aiming to remove unwanted data influences and associated model capabilities (e.g., copyrighted data or harmful content generation) while preserving essential model utilities, without the need for retraining from scratch. Despite the growing need for LLM unlearning, a principled optimization framework remains lacking. To this end, we revisit the state-of-the-art approach, negative preference optimization (NPO), and identify the issue of reference model bias, which could undermine NPO's effectiveness, particularly when unlearning forget data of varying difficulty. Given that, we propose a simple yet effective unlearning optimization framework, called SimNPO, showing that 'simplicity' in removing the reliance on a reference model (through the lens of simple preference optimization) benefits unlearning. We also provide deeper insights into SimNPO's advantages, supported by analysis using mixtures of Markov chains. Furthermore, we present extensive experiments validating SimNPO's superiority over existing unlearning baselines in benchmarks like TOFU and MUSE, and robustness against relearning attacks. Codes are available at https://github.com/OPTML-Group/Unlearn-Simple.



## **42. Universal Vulnerabilities in Large Language Models: Backdoor Attacks for In-context Learning**

cs.CL

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2401.05949v6) [paper-pdf](http://arxiv.org/pdf/2401.05949v6)

**Authors**: Shuai Zhao, Meihuizi Jia, Luu Anh Tuan, Fengjun Pan, Jinming Wen

**Abstract**: In-context learning, a paradigm bridging the gap between pre-training and fine-tuning, has demonstrated high efficacy in several NLP tasks, especially in few-shot settings. Despite being widely applied, in-context learning is vulnerable to malicious attacks. In this work, we raise security concerns regarding this paradigm. Our studies demonstrate that an attacker can manipulate the behavior of large language models by poisoning the demonstration context, without the need for fine-tuning the model. Specifically, we design a new backdoor attack method, named ICLAttack, to target large language models based on in-context learning. Our method encompasses two types of attacks: poisoning demonstration examples and poisoning demonstration prompts, which can make models behave in alignment with predefined intentions. ICLAttack does not require additional fine-tuning to implant a backdoor, thus preserving the model's generality. Furthermore, the poisoned examples are correctly labeled, enhancing the natural stealth of our attack method. Extensive experimental results across several language models, ranging in size from 1.3B to 180B parameters, demonstrate the effectiveness of our attack method, exemplified by a high average attack success rate of 95.0% across the three datasets on OPT models.



## **43. Prompt Infection: LLM-to-LLM Prompt Injection within Multi-Agent Systems**

cs.MA

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.07283v1) [paper-pdf](http://arxiv.org/pdf/2410.07283v1)

**Authors**: Donghyun Lee, Mo Tiwari

**Abstract**: As Large Language Models (LLMs) grow increasingly powerful, multi-agent systems are becoming more prevalent in modern AI applications. Most safety research, however, has focused on vulnerabilities in single-agent LLMs. These include prompt injection attacks, where malicious prompts embedded in external content trick the LLM into executing unintended or harmful actions, compromising the victim's application. In this paper, we reveal a more dangerous vector: LLM-to-LLM prompt injection within multi-agent systems. We introduce Prompt Infection, a novel attack where malicious prompts self-replicate across interconnected agents, behaving much like a computer virus. This attack poses severe threats, including data theft, scams, misinformation, and system-wide disruption, all while propagating silently through the system. Our extensive experiments demonstrate that multi-agent systems are highly susceptible, even when agents do not publicly share all communications. To address this, we propose LLM Tagging, a defense mechanism that, when combined with existing safeguards, significantly mitigates infection spread. This work underscores the urgent need for advanced security measures as multi-agent LLM systems become more widely adopted.



## **44. Break the Visual Perception: Adversarial Attacks Targeting Encoded Visual Tokens of Large Vision-Language Models**

cs.CV

Accepted to ACMMM 2024

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.06699v1) [paper-pdf](http://arxiv.org/pdf/2410.06699v1)

**Authors**: Yubo Wang, Chaohu Liu, Yanqiu Qu, Haoyu Cao, Deqiang Jiang, Linli Xu

**Abstract**: Large vision-language models (LVLMs) integrate visual information into large language models, showcasing remarkable multi-modal conversational capabilities. However, the visual modules introduces new challenges in terms of robustness for LVLMs, as attackers can craft adversarial images that are visually clean but may mislead the model to generate incorrect answers. In general, LVLMs rely on vision encoders to transform images into visual tokens, which are crucial for the language models to perceive image contents effectively. Therefore, we are curious about one question: Can LVLMs still generate correct responses when the encoded visual tokens are attacked and disrupting the visual information? To this end, we propose a non-targeted attack method referred to as VT-Attack (Visual Tokens Attack), which constructs adversarial examples from multiple perspectives, with the goal of comprehensively disrupting feature representations and inherent relationships as well as the semantic properties of visual tokens output by image encoders. Using only access to the image encoder in the proposed attack, the generated adversarial examples exhibit transferability across diverse LVLMs utilizing the same image encoder and generality across different tasks. Extensive experiments validate the superior attack performance of the VT-Attack over baseline methods, demonstrating its effectiveness in attacking LVLMs with image encoders, which in turn can provide guidance on the robustness of LVLMs, particularly in terms of the stability of the visual feature space.



## **45. FELLAS: Enhancing Federated Sequential Recommendation with LLM as External Services**

cs.IR

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.04927v2) [paper-pdf](http://arxiv.org/pdf/2410.04927v2)

**Authors**: Wei Yuan, Chaoqun Yang, Guanhua Ye, Tong Chen, Quoc Viet Hung Nguyen, Hongzhi Yin

**Abstract**: Federated sequential recommendation (FedSeqRec) has gained growing attention due to its ability to protect user privacy. Unfortunately, the performance of FedSeqRec is still unsatisfactory because the models used in FedSeqRec have to be lightweight to accommodate communication bandwidth and clients' on-device computational resource constraints. Recently, large language models (LLMs) have exhibited strong transferable and generalized language understanding abilities and therefore, in the NLP area, many downstream tasks now utilize LLMs as a service to achieve superior performance without constructing complex models. Inspired by this successful practice, we propose a generic FedSeqRec framework, FELLAS, which aims to enhance FedSeqRec by utilizing LLMs as an external service. Specifically, FELLAS employs an LLM server to provide both item-level and sequence-level representation assistance. The item-level representation service is queried by the central server to enrich the original ID-based item embedding with textual information, while the sequence-level representation service is accessed by each client. However, invoking the sequence-level representation service requires clients to send sequences to the external LLM server. To safeguard privacy, we implement dx-privacy satisfied sequence perturbation, which protects clients' sensitive data with guarantees. Additionally, a contrastive learning-based method is designed to transfer knowledge from the noisy sequence representation to clients' sequential recommendation models. Furthermore, to empirically validate the privacy protection capability of FELLAS, we propose two interacted item inference attacks. Extensive experiments conducted on three datasets with two widely used sequential recommendation models demonstrate the effectiveness and privacy-preserving capability of FELLAS.



## **46. FreqMark: Frequency-Based Watermark for Sentence-Level Detection of LLM-Generated Text**

cs.CL

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.10876v1) [paper-pdf](http://arxiv.org/pdf/2410.10876v1)

**Authors**: Zhenyu Xu, Kun Zhang, Victor S. Sheng

**Abstract**: The increasing use of Large Language Models (LLMs) for generating highly coherent and contextually relevant text introduces new risks, including misuse for unethical purposes such as disinformation or academic dishonesty. To address these challenges, we propose FreqMark, a novel watermarking technique that embeds detectable frequency-based watermarks in LLM-generated text during the token sampling process. The method leverages periodic signals to guide token selection, creating a watermark that can be detected with Short-Time Fourier Transform (STFT) analysis. This approach enables accurate identification of LLM-generated content, even in mixed-text scenarios with both human-authored and LLM-generated segments. Our experiments demonstrate the robustness and precision of FreqMark, showing strong detection capabilities against various attack scenarios such as paraphrasing and token substitution. Results show that FreqMark achieves an AUC improvement of up to 0.98, significantly outperforming existing detection methods.



## **47. Signal Watermark on Large Language Models**

cs.CR

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.06545v1) [paper-pdf](http://arxiv.org/pdf/2410.06545v1)

**Authors**: Zhenyu Xu, Victor S. Sheng

**Abstract**: As Large Language Models (LLMs) become increasingly sophisticated, they raise significant security concerns, including the creation of fake news and academic misuse. Most detectors for identifying model-generated text are limited by their reliance on variance in perplexity and burstiness, and they require substantial computational resources. In this paper, we proposed a watermarking method embedding a specific watermark into the text during its generation by LLMs, based on a pre-defined signal pattern. This technique not only ensures the watermark's invisibility to humans but also maintains the quality and grammatical integrity of model-generated text. We utilize LLMs and Fast Fourier Transform (FFT) for token probability computation and detection of the signal watermark. The unique application of signal processing principles within the realm of text generation by LLMs allows for subtle yet effective embedding of watermarks, which do not compromise the quality or coherence of the generated text. Our method has been empirically validated across multiple LLMs, consistently maintaining high detection accuracy, even with variations in temperature settings during text generation. In the experiment of distinguishing between human-written and watermarked text, our method achieved an AUROC score of 0.97, significantly outperforming existing methods like GPTZero, which scored 0.64. The watermark's resilience to various attacking scenarios further confirms its robustness, addressing significant challenges in model-generated text authentication.



## **48. WAPITI: A Watermark for Finetuned Open-Source LLMs**

cs.CR

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.06467v1) [paper-pdf](http://arxiv.org/pdf/2410.06467v1)

**Authors**: Lingjie Chen, Ruizhong Qiu, Siyu Yuan, Zhining Liu, Tianxin Wei, Hyunsik Yoo, Zhichen Zeng, Deqing Yang, Hanghang Tong

**Abstract**: Watermarking of large language models (LLMs) generation embeds an imperceptible statistical pattern within texts, making it algorithmically detectable. Watermarking is a promising method for addressing potential harm and biases from LLMs, as it enables traceability, accountability, and detection of manipulated content, helping to mitigate unintended consequences. However, for open-source models, watermarking faces two major challenges: (i) incompatibility with fine-tuned models, and (ii) vulnerability to fine-tuning attacks. In this work, we propose WAPITI, a new method that transfers watermarking from base models to fine-tuned models through parameter integration. To the best of our knowledge, we propose the first watermark for fine-tuned open-source LLMs that preserves their fine-tuned capabilities. Furthermore, our approach offers an effective defense against fine-tuning attacks. We test our method on various model architectures and watermarking strategies. Results demonstrate that our method can successfully inject watermarks and is highly compatible with fine-tuned models. Additionally, we offer an in-depth analysis of how parameter editing influences the watermark strength and overall capabilities of the resulting models.



## **49. Hallucinating AI Hijacking Attack: Large Language Models and Malicious Code Recommenders**

cs.CR

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.06462v1) [paper-pdf](http://arxiv.org/pdf/2410.06462v1)

**Authors**: David Noever, Forrest McKee

**Abstract**: The research builds and evaluates the adversarial potential to introduce copied code or hallucinated AI recommendations for malicious code in popular code repositories. While foundational large language models (LLMs) from OpenAI, Google, and Anthropic guard against both harmful behaviors and toxic strings, previous work on math solutions that embed harmful prompts demonstrate that the guardrails may differ between expert contexts. These loopholes would appear in mixture of expert's models when the context of the question changes and may offer fewer malicious training examples to filter toxic comments or recommended offensive actions. The present work demonstrates that foundational models may refuse to propose destructive actions correctly when prompted overtly but may unfortunately drop their guard when presented with a sudden change of context, like solving a computer programming challenge. We show empirical examples with trojan-hosting repositories like GitHub, NPM, NuGet, and popular content delivery networks (CDN) like jsDelivr which amplify the attack surface. In the LLM's directives to be helpful, example recommendations propose application programming interface (API) endpoints which a determined domain-squatter could acquire and setup attack mobile infrastructure that triggers from the naively copied code. We compare this attack to previous work on context-shifting and contrast the attack surface as a novel version of "living off the land" attacks in the malware literature. In the latter case, foundational language models can hijack otherwise innocent user prompts to recommend actions that violate their owners' safety policies when posed directly without the accompanying coding support request.



## **50. Recent advancements in LLM Red-Teaming: Techniques, Defenses, and Ethical Considerations**

cs.CL

16 pages, 2 figures

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.09097v1) [paper-pdf](http://arxiv.org/pdf/2410.09097v1)

**Authors**: Tarun Raheja, Nilay Pochhi

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language processing tasks, but their vulnerability to jailbreak attacks poses significant security risks. This survey paper presents a comprehensive analysis of recent advancements in attack strategies and defense mechanisms within the field of Large Language Model (LLM) red-teaming. We analyze various attack methods, including gradient-based optimization, reinforcement learning, and prompt engineering approaches. We discuss the implications of these attacks on LLM safety and the need for improved defense mechanisms. This work aims to provide a thorough understanding of the current landscape of red-teaming attacks and defenses on LLMs, enabling the development of more secure and reliable language models.



