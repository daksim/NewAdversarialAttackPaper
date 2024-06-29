# Latest Adversarial Attack Papers
**update at 2024-06-29 16:03:18**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Zero-Query Adversarial Attack on Black-box Automatic Speech Recognition Systems**

cs.CR

To appear in the Proceedings of The ACM Conference on Computer and  Communications Security (CCS), 2024

**SubmitDate**: 2024-06-27    [abs](http://arxiv.org/abs/2406.19311v1) [paper-pdf](http://arxiv.org/pdf/2406.19311v1)

**Authors**: Zheng Fang, Tao Wang, Lingchen Zhao, Shenyi Zhang, Bowen Li, Yunjie Ge, Qi Li, Chao Shen, Qian Wang

**Abstract**: In recent years, extensive research has been conducted on the vulnerability of ASR systems, revealing that black-box adversarial example attacks pose significant threats to real-world ASR systems. However, most existing black-box attacks rely on queries to the target ASRs, which is impractical when queries are not permitted. In this paper, we propose ZQ-Attack, a transfer-based adversarial attack on ASR systems in the zero-query black-box setting. Through a comprehensive review and categorization of modern ASR technologies, we first meticulously select surrogate ASRs of diverse types to generate adversarial examples. Following this, ZQ-Attack initializes the adversarial perturbation with a scaled target command audio, rendering it relatively imperceptible while maintaining effectiveness. Subsequently, to achieve high transferability of adversarial perturbations, we propose a sequential ensemble optimization algorithm, which iteratively optimizes the adversarial perturbation on each surrogate model, leveraging collaborative information from other models. We conduct extensive experiments to evaluate ZQ-Attack. In the over-the-line setting, ZQ-Attack achieves a 100% success rate of attack (SRoA) with an average signal-to-noise ratio (SNR) of 21.91dB on 4 online speech recognition services, and attains an average SRoA of 100% and SNR of 19.67dB on 16 open-source ASRs. For commercial intelligent voice control devices, ZQ-Attack also achieves a 100% SRoA with an average SNR of 15.77dB in the over-the-air setting.



## **2. GPTFUZZER: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts**

cs.AI

**SubmitDate**: 2024-06-27    [abs](http://arxiv.org/abs/2309.10253v4) [paper-pdf](http://arxiv.org/pdf/2309.10253v4)

**Authors**: Jiahao Yu, Xingwei Lin, Zheng Yu, Xinyu Xing

**Abstract**: Large language models (LLMs) have recently experienced tremendous popularity and are widely used from casual conversations to AI-driven programming. However, despite their considerable success, LLMs are not entirely reliable and can give detailed guidance on how to conduct harmful or illegal activities. While safety measures can reduce the risk of such outputs, adversarial jailbreak attacks can still exploit LLMs to produce harmful content. These jailbreak templates are typically manually crafted, making large-scale testing challenging.   In this paper, we introduce GPTFuzz, a novel black-box jailbreak fuzzing framework inspired by the AFL fuzzing framework. Instead of manual engineering, GPTFuzz automates the generation of jailbreak templates for red-teaming LLMs. At its core, GPTFuzz starts with human-written templates as initial seeds, then mutates them to produce new templates. We detail three key components of GPTFuzz: a seed selection strategy for balancing efficiency and variability, mutate operators for creating semantically equivalent or similar sentences, and a judgment model to assess the success of a jailbreak attack.   We evaluate GPTFuzz against various commercial and open-source LLMs, including ChatGPT, LLaMa-2, and Vicuna, under diverse attack scenarios. Our results indicate that GPTFuzz consistently produces jailbreak templates with a high success rate, surpassing human-crafted templates. Remarkably, GPTFuzz achieves over 90% attack success rates against ChatGPT and Llama-2 models, even with suboptimal initial seed templates. We anticipate that GPTFuzz will be instrumental for researchers and practitioners in examining LLM robustness and will encourage further exploration into enhancing LLM safety.



## **3. Spiking Convolutional Neural Networks for Text Classification**

cs.NE

**SubmitDate**: 2024-06-27    [abs](http://arxiv.org/abs/2406.19230v1) [paper-pdf](http://arxiv.org/pdf/2406.19230v1)

**Authors**: Changze Lv, Jianhan Xu, Xiaoqing Zheng

**Abstract**: Spiking neural networks (SNNs) offer a promising pathway to implement deep neural networks (DNNs) in a more energy-efficient manner since their neurons are sparsely activated and inferences are event-driven. However, there have been very few works that have demonstrated the efficacy of SNNs in language tasks partially because it is non-trivial to represent words in the forms of spikes and to deal with variable-length texts by SNNs. This work presents a "conversion + fine-tuning" two-step method for training SNNs for text classification and proposes a simple but effective way to encode pre-trained word embeddings as spike trains. We show empirically that after fine-tuning with surrogate gradients, the converted SNNs achieve comparable results to their DNN counterparts with much less energy consumption across multiple datasets for both English and Chinese. We also show that such SNNs are more robust to adversarial attacks than DNNs.



## **4. Understanding the Security Benefits and Overheads of Emerging Industry Solutions to DRAM Read Disturbance**

cs.CR

To appear in DRAMSec 2024

**SubmitDate**: 2024-06-27    [abs](http://arxiv.org/abs/2406.19094v1) [paper-pdf](http://arxiv.org/pdf/2406.19094v1)

**Authors**: Oğuzhan Canpolat, A. Giray Yağlıkçı, Geraldo F. Oliveira, Ataberk Olgun, Oğuz Ergin, Onur Mutlu

**Abstract**: We present the first rigorous security, performance, energy, and cost analyses of the state-of-the-art on-DRAM-die read disturbance mitigation method, Per Row Activation Counting (PRAC), described in JEDEC DDR5 specification's April 2024 update. Unlike prior state-of-the-art that advises the memory controller to periodically issue refresh management (RFM) commands, which provides the DRAM chip with time to perform refreshes, PRAC introduces a new back-off signal. PRAC's back-off signal propagates from the DRAM chip to the memory controller and forces the memory controller to 1) stop serving requests and 2) issue RFM commands. As a result, RFM commands are issued when needed as opposed to periodically, reducing RFM's overheads. We analyze PRAC in four steps. First, we define an adversarial access pattern that represents the worst-case for PRAC's security. Second, we investigate PRAC's configurations and security implications. Our analyses show that PRAC can be configured for secure operation as long as no bitflip occurs before accessing a memory location 10 times. Third, we evaluate the performance impact of PRAC and compare it against prior works using Ramulator 2.0. Our analysis shows that while PRAC incurs less than 13.4% performance overhead for today's DRAM chips, its performance overheads can reach up to 63.2% for future DRAM chips that are more vulnerable to read disturbance bitflips. Fourth, we define an availability adversarial access pattern that exacerbates PRAC's performance overhead to perform a memory performance attack, demonstrating that such an adversarial pattern can hog up to 79% of DRAM throughput and degrade system throughput by up to 65%. We discuss PRAC's implications on future systems and foreshadow future research directions. To aid future research, we open-source our implementations and scripts at https://github.com/CMU-SAFARI/ramulator2.



## **5. Intriguing Properties of Adversarial ML Attacks in the Problem Space [Extended Version]**

cs.CR

This arXiv version (v3) corresponds to an extended version

**SubmitDate**: 2024-06-27    [abs](http://arxiv.org/abs/1911.02142v3) [paper-pdf](http://arxiv.org/pdf/1911.02142v3)

**Authors**: Jacopo Cortellazzi, Feargus Pendlebury, Daniel Arp, Erwin Quiring, Fabio Pierazzi, Lorenzo Cavallaro

**Abstract**: Recent research efforts on adversarial machine learning (ML) have investigated problem-space attacks, focusing on the generation of real evasive objects in domains where, unlike images, there is no clear inverse mapping to the feature space (e.g., software). However, the design, comparison, and real-world implications of problem-space attacks remain underexplored. This article makes three major contributions. Firstly, we propose a general formalization for adversarial ML evasion attacks in the problem-space, which includes the definition of a comprehensive set of constraints on available transformations, preserved semantics, absent artifacts, and plausibility. We shed light on the relationship between feature space and problem space, and we introduce the concept of side-effect features as the by-product of the inverse feature-mapping problem. This enables us to define and prove necessary and sufficient conditions for the existence of problem-space attacks. Secondly, building on our general formalization, we propose a novel problem-space attack on Android malware that overcomes past limitations in terms of semantics and artifacts. We have tested our approach on a dataset with 150K Android apps from 2016 and 2018 which show the practical feasibility of evading a state-of-the-art malware classifier along with its hardened version. Thirdly, we explore the effectiveness of adversarial training as a possible approach to enforce robustness against adversarial samples, evaluating its effectiveness on the considered machine learning models under different scenarios. Our results demonstrate that "adversarial-malware as a service" is a realistic threat, as we automatically generate thousands of realistic and inconspicuous adversarial applications at scale, where on average it takes only a few minutes to generate an adversarial instance.



## **6. Dysca: A Dynamic and Scalable Benchmark for Evaluating Perception Ability of LVLMs**

cs.CV

**SubmitDate**: 2024-06-27    [abs](http://arxiv.org/abs/2406.18849v1) [paper-pdf](http://arxiv.org/pdf/2406.18849v1)

**Authors**: Jie Zhang, Zhongqi Wang, Mengqi Lei, Zheng Yuan, Bei Yan, Shiguang Shan, Xilin Chen

**Abstract**: Currently many benchmarks have been proposed to evaluate the perception ability of the Large Vision-Language Models (LVLMs). However, most benchmarks conduct questions by selecting images from existing datasets, resulting in the potential data leakage. Besides, these benchmarks merely focus on evaluating LVLMs on the realistic style images and clean scenarios, leaving the multi-stylized images and noisy scenarios unexplored. In response to these challenges, we propose a dynamic and scalable benchmark named Dysca for evaluating LVLMs by leveraging synthesis images. Specifically, we leverage Stable Diffusion and design a rule-based method to dynamically generate novel images, questions and the corresponding answers. We consider 51 kinds of image styles and evaluate the perception capability in 20 subtasks. Moreover, we conduct evaluations under 4 scenarios (i.e., Clean, Corruption, Print Attacking and Adversarial Attacking) and 3 question types (i.e., Multi-choices, True-or-false and Free-form). Thanks to the generative paradigm, Dysca serves as a scalable benchmark for easily adding new subtasks and scenarios. A total of 8 advanced open-source LVLMs with 10 checkpoints are evaluated on Dysca, revealing the drawbacks of current LVLMs. The benchmark is released in \url{https://github.com/Benchmark-Dysca/Dysca}.



## **7. A Zero Auxiliary Knowledge Membership Inference Attack on Aggregate Location Data**

cs.CR

To be published in PETS 2024

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.18671v1) [paper-pdf](http://arxiv.org/pdf/2406.18671v1)

**Authors**: Vincent Guan, Florent Guépin, Ana-Maria Cretu, Yves-Alexandre de Montjoye

**Abstract**: Location data is frequently collected from populations and shared in aggregate form to guide policy and decision making. However, the prevalence of aggregated data also raises the privacy concern of membership inference attacks (MIAs). MIAs infer whether an individual's data contributed to the aggregate release. Although effective MIAs have been developed for aggregate location data, these require access to an extensive auxiliary dataset of individual traces over the same locations, which are collected from a similar population. This assumption is often impractical given common privacy practices surrounding location data. To measure the risk of an MIA performed by a realistic adversary, we develop the first Zero Auxiliary Knowledge (ZK) MIA on aggregate location data, which eliminates the need for an auxiliary dataset of real individual traces. Instead, we develop a novel synthetic approach, such that suitable synthetic traces are generated from the released aggregate. We also develop methods to correct for bias and noise, to show that our synthetic-based attack is still applicable when privacy mechanisms are applied prior to release. Using two large-scale location datasets, we demonstrate that our ZK MIA matches the state-of-the-art Knock-Knock (KK) MIA across a wide range of settings, including popular implementations of differential privacy (DP) and suppression of small counts. Furthermore, we show that ZK MIA remains highly effective even when the adversary only knows a small fraction (10%) of their target's location history. This demonstrates that effective MIAs can be performed by realistic adversaries, highlighting the need for strong DP protection.



## **8. WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially) Safer Language Models**

cs.CL

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.18510v1) [paper-pdf](http://arxiv.org/pdf/2406.18510v1)

**Authors**: Liwei Jiang, Kavel Rao, Seungju Han, Allyson Ettinger, Faeze Brahman, Sachin Kumar, Niloofar Mireshghallah, Ximing Lu, Maarten Sap, Yejin Choi, Nouha Dziri

**Abstract**: We introduce WildTeaming, an automatic LLM safety red-teaming framework that mines in-the-wild user-chatbot interactions to discover 5.7K unique clusters of novel jailbreak tactics, and then composes multiple tactics for systematic exploration of novel jailbreaks. Compared to prior work that performed red-teaming via recruited human workers, gradient-based optimization, or iterative revision with LLMs, our work investigates jailbreaks from chatbot users who were not specifically instructed to break the system. WildTeaming reveals previously unidentified vulnerabilities of frontier LLMs, resulting in up to 4.6x more diverse and successful adversarial attacks compared to state-of-the-art jailbreak methods.   While many datasets exist for jailbreak evaluation, very few open-source datasets exist for jailbreak training, as safety training data has been closed even when model weights are open. With WildTeaming we create WildJailbreak, a large-scale open-source synthetic safety dataset with 262K vanilla (direct request) and adversarial (complex jailbreak) prompt-response pairs. To mitigate exaggerated safety behaviors, WildJailbreak provides two contrastive types of queries: 1) harmful queries (vanilla & adversarial) and 2) benign queries that resemble harmful queries in form but contain no harm. As WildJailbreak considerably upgrades the quality and scale of existing safety resources, it uniquely enables us to examine the scaling effects of data and the interplay of data properties and model capabilities during safety training. Through extensive experiments, we identify the training properties that enable an ideal balance of safety behaviors: appropriate safeguarding without over-refusal, effective handling of vanilla and adversarial queries, and minimal, if any, decrease in general capabilities. All components of WildJailbeak contribute to achieving balanced safety behaviors of models.



## **9. WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs**

cs.CL

First two authors contributed equally. Third and fourth authors  contributed equally

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.18495v1) [paper-pdf](http://arxiv.org/pdf/2406.18495v1)

**Authors**: Seungju Han, Kavel Rao, Allyson Ettinger, Liwei Jiang, Bill Yuchen Lin, Nathan Lambert, Yejin Choi, Nouha Dziri

**Abstract**: We introduce WildGuard -- an open, light-weight moderation tool for LLM safety that achieves three goals: (1) identifying malicious intent in user prompts, (2) detecting safety risks of model responses, and (3) determining model refusal rate. Together, WildGuard serves the increasing needs for automatic safety moderation and evaluation of LLM interactions, providing a one-stop tool with enhanced accuracy and broad coverage across 13 risk categories. While existing open moderation tools such as Llama-Guard2 score reasonably well in classifying straightforward model interactions, they lag far behind a prompted GPT-4, especially in identifying adversarial jailbreaks and in evaluating models' refusals, a key measure for evaluating safety behaviors in model responses.   To address these challenges, we construct WildGuardMix, a large-scale and carefully balanced multi-task safety moderation dataset with 92K labeled examples that cover vanilla (direct) prompts and adversarial jailbreaks, paired with various refusal and compliance responses. WildGuardMix is a combination of WildGuardTrain, the training data of WildGuard, and WildGuardTest, a high-quality human-annotated moderation test set with 5K labeled items covering broad risk scenarios. Through extensive evaluations on WildGuardTest and ten existing public benchmarks, we show that WildGuard establishes state-of-the-art performance in open-source safety moderation across all the three tasks compared to ten strong existing open-source moderation models (e.g., up to 26.4% improvement on refusal detection). Importantly, WildGuard matches and sometimes exceeds GPT-4 performance (e.g., up to 3.9% improvement on prompt harmfulness identification). WildGuard serves as a highly effective safety moderator in an LLM interface, reducing the success rate of jailbreak attacks from 79.8% to 2.4%.



## **10. Enhancing Federated Learning with Adaptive Differential Privacy and Priority-Based Aggregation**

cs.LG

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.18491v1) [paper-pdf](http://arxiv.org/pdf/2406.18491v1)

**Authors**: Mahtab Talaei, Iman Izadi

**Abstract**: Federated learning (FL), a novel branch of distributed machine learning (ML), develops global models through a private procedure without direct access to local datasets. However, it is still possible to access the model updates (gradient updates of deep neural networks) transferred between clients and servers, potentially revealing sensitive local information to adversaries using model inversion attacks. Differential privacy (DP) offers a promising approach to addressing this issue by adding noise to the parameters. On the other hand, heterogeneities in data structure, storage, communication, and computational capabilities of devices can cause convergence problems and delays in developing the global model. A personalized weighted averaging of local parameters based on the resources of each device can yield a better aggregated model in each round. In this paper, to efficiently preserve privacy, we propose a personalized DP framework that injects noise based on clients' relative impact factors and aggregates parameters while considering heterogeneities and adjusting properties. To fulfill the DP requirements, we first analyze the convergence boundary of the FL algorithm when impact factors are personalized and fixed throughout the learning process. We then further study the convergence property considering time-varying (adaptive) impact factors.



## **11. MultiAgent Collaboration Attack: Investigating Adversarial Attacks in Large Language Model Collaborations via Debate**

cs.CL

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.14711v2) [paper-pdf](http://arxiv.org/pdf/2406.14711v2)

**Authors**: Alfonso Amayuelas, Xianjun Yang, Antonis Antoniades, Wenyue Hua, Liangming Pan, William Wang

**Abstract**: Large Language Models (LLMs) have shown exceptional results on current benchmarks when working individually. The advancement in their capabilities, along with a reduction in parameter size and inference times, has facilitated the use of these models as agents, enabling interactions among multiple models to execute complex tasks. Such collaborations offer several advantages, including the use of specialized models (e.g. coding), improved confidence through multiple computations, and enhanced divergent thinking, leading to more diverse outputs. Thus, the collaborative use of language models is expected to grow significantly in the coming years. In this work, we evaluate the behavior of a network of models collaborating through debate under the influence of an adversary. We introduce pertinent metrics to assess the adversary's effectiveness, focusing on system accuracy and model agreement. Our findings highlight the importance of a model's persuasive ability in influencing others. Additionally, we explore inference-time methods to generate more compelling arguments and evaluate the potential of prompt-based mitigation as a defensive strategy.



## **12. Detecting Brittle Decisions for Free: Leveraging Margin Consistency in Deep Robust Classifiers**

cs.LG

11 pages, 7 figures, 2 tables, 1 algorithm

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.18451v1) [paper-pdf](http://arxiv.org/pdf/2406.18451v1)

**Authors**: Jonas Ngnawé, Sabyasachi Sahoo, Yann Pequignot, Frédéric Precioso, Christian Gagné

**Abstract**: Despite extensive research on adversarial training strategies to improve robustness, the decisions of even the most robust deep learning models can still be quite sensitive to imperceptible perturbations, creating serious risks when deploying them for high-stakes real-world applications. While detecting such cases may be critical, evaluating a model's vulnerability at a per-instance level using adversarial attacks is computationally too intensive and unsuitable for real-time deployment scenarios. The input space margin is the exact score to detect non-robust samples and is intractable for deep neural networks. This paper introduces the concept of margin consistency -- a property that links the input space margins and the logit margins in robust models -- for efficient detection of vulnerable samples. First, we establish that margin consistency is a necessary and sufficient condition to use a model's logit margin as a score for identifying non-robust samples. Next, through comprehensive empirical analysis of various robustly trained models on CIFAR10 and CIFAR100 datasets, we show that they indicate strong margin consistency with a strong correlation between their input space margins and the logit margins. Then, we show that we can effectively use the logit margin to confidently detect brittle decisions with such models and accurately estimate robust accuracy on an arbitrarily large test set by estimating the input margins only on a small subset. Finally, we address cases where the model is not sufficiently margin-consistent by learning a pseudo-margin from the feature representation. Our findings highlight the potential of leveraging deep representations to efficiently assess adversarial vulnerability in deployment scenarios.



## **13. Are AI-Generated Text Detectors Robust to Adversarial Perturbations?**

cs.CL

Accepted to ACL 2024 main conference

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.01179v2) [paper-pdf](http://arxiv.org/pdf/2406.01179v2)

**Authors**: Guanhua Huang, Yuchen Zhang, Zhe Li, Yongjian You, Mingze Wang, Zhouwang Yang

**Abstract**: The widespread use of large language models (LLMs) has sparked concerns about the potential misuse of AI-generated text, as these models can produce content that closely resembles human-generated text. Current detectors for AI-generated text (AIGT) lack robustness against adversarial perturbations, with even minor changes in characters or words causing a reversal in distinguishing between human-created and AI-generated text. This paper investigates the robustness of existing AIGT detection methods and introduces a novel detector, the Siamese Calibrated Reconstruction Network (SCRN). The SCRN employs a reconstruction network to add and remove noise from text, extracting a semantic representation that is robust to local perturbations. We also propose a siamese calibration technique to train the model to make equally confidence predictions under different noise, which improves the model's robustness against adversarial perturbations. Experiments on four publicly available datasets show that the SCRN outperforms all baseline methods, achieving 6.5\%-18.25\% absolute accuracy improvement over the best baseline method under adversarial attacks. Moreover, it exhibits superior generalizability in cross-domain, cross-genre, and mixed-source scenarios. The code is available at \url{https://github.com/CarlanLark/Robust-AIGC-Detector}.



## **14. SUB-PLAY: Adversarial Policies against Partially Observed Multi-Agent Reinforcement Learning Systems**

cs.LG

To appear in the ACM Conference on Computer and Communications  Security (CCS'24), October 14-18, 2024, Salt Lake City, UT, USA

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2402.03741v3) [paper-pdf](http://arxiv.org/pdf/2402.03741v3)

**Authors**: Oubo Ma, Yuwen Pu, Linkang Du, Yang Dai, Ruo Wang, Xiaolei Liu, Yingcai Wu, Shouling Ji

**Abstract**: Recent advancements in multi-agent reinforcement learning (MARL) have opened up vast application prospects, such as swarm control of drones, collaborative manipulation by robotic arms, and multi-target encirclement. However, potential security threats during the MARL deployment need more attention and thorough investigation. Recent research reveals that attackers can rapidly exploit the victim's vulnerabilities, generating adversarial policies that result in the failure of specific tasks. For instance, reducing the winning rate of a superhuman-level Go AI to around 20%. Existing studies predominantly focus on two-player competitive environments, assuming attackers possess complete global state observation.   In this study, we unveil, for the first time, the capability of attackers to generate adversarial policies even when restricted to partial observations of the victims in multi-agent competitive environments. Specifically, we propose a novel black-box attack (SUB-PLAY) that incorporates the concept of constructing multiple subgames to mitigate the impact of partial observability and suggests sharing transitions among subpolicies to improve attackers' exploitative ability. Extensive evaluations demonstrate the effectiveness of SUB-PLAY under three typical partial observability limitations. Visualization results indicate that adversarial policies induce significantly different activations of the victims' policy networks. Furthermore, we evaluate three potential defenses aimed at exploring ways to mitigate security threats posed by adversarial policies, providing constructive recommendations for deploying MARL in competitive environments.



## **15. Artificial Immune System of Secure Face Recognition Against Adversarial Attacks**

cs.CV

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.18144v1) [paper-pdf](http://arxiv.org/pdf/2406.18144v1)

**Authors**: Min Ren, Yunlong Wang, Yuhao Zhu, Yongzhen Huang, Zhenan Sun, Qi Li, Tieniu Tan

**Abstract**: Insect production for food and feed presents a promising supplement to ensure food safety and address the adverse impacts of agriculture on climate and environment in the future. However, optimisation is required for insect production to realise its full potential. This can be by targeted improvement of traits of interest through selective breeding, an approach which has so far been underexplored and underutilised in insect farming. Here we present a comprehensive review of the selective breeding framework in the context of insect production. We systematically evaluate adjustments of selective breeding techniques to the realm of insects and highlight the essential components integral to the breeding process. The discussion covers every step of a conventional breeding scheme, such as formulation of breeding objectives, phenotyping, estimation of genetic parameters and breeding values, selection of appropriate breeding strategies, and mitigation of issues associated with genetic diversity depletion and inbreeding. This review combines knowledge from diverse disciplines, bridging the gap between animal breeding, quantitative genetics, evolutionary biology, and entomology, offering an integrated view of the insect breeding research area and uniting knowledge which has previously remained scattered across diverse fields of expertise.



## **16. Breaking the Barrier: Enhanced Utility and Robustness in Smoothed DRL Agents**

cs.LG

Published in ICML 2024

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.18062v1) [paper-pdf](http://arxiv.org/pdf/2406.18062v1)

**Authors**: Chung-En Sun, Sicun Gao, Tsui-Wei Weng

**Abstract**: Robustness remains a paramount concern in deep reinforcement learning (DRL), with randomized smoothing emerging as a key technique for enhancing this attribute. However, a notable gap exists in the performance of current smoothed DRL agents, often characterized by significantly low clean rewards and weak robustness. In response to this challenge, our study introduces innovative algorithms aimed at training effective smoothed robust DRL agents. We propose S-DQN and S-PPO, novel approaches that demonstrate remarkable improvements in clean rewards, empirical robustness, and robustness guarantee across standard RL benchmarks. Notably, our S-DQN and S-PPO agents not only significantly outperform existing smoothed agents by an average factor of $2.16\times$ under the strongest attack, but also surpass previous robustly-trained agents by an average factor of $2.13\times$. This represents a significant leap forward in the field. Furthermore, we introduce Smoothed Attack, which is $1.89\times$ more effective in decreasing the rewards of smoothed agents than existing adversarial attacks.



## **17. InstructTA: Instruction-Tuned Targeted Attack for Large Vision-Language Models**

cs.CV

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2312.01886v3) [paper-pdf](http://arxiv.org/pdf/2312.01886v3)

**Authors**: Xunguang Wang, Zhenlan Ji, Pingchuan Ma, Zongjie Li, Shuai Wang

**Abstract**: Large vision-language models (LVLMs) have demonstrated their incredible capability in image understanding and response generation. However, this rich visual interaction also makes LVLMs vulnerable to adversarial examples. In this paper, we formulate a novel and practical targeted attack scenario that the adversary can only know the vision encoder of the victim LVLM, without the knowledge of its prompts (which are often proprietary for service providers and not publicly available) and its underlying large language model (LLM). This practical setting poses challenges to the cross-prompt and cross-model transferability of targeted adversarial attack, which aims to confuse the LVLM to output a response that is semantically similar to the attacker's chosen target text. To this end, we propose an instruction-tuned targeted attack (dubbed \textsc{InstructTA}) to deliver the targeted adversarial attack on LVLMs with high transferability. Initially, we utilize a public text-to-image generative model to "reverse" the target response into a target image, and employ GPT-4 to infer a reasonable instruction $\boldsymbol{p}^\prime$ from the target response. We then form a local surrogate model (sharing the same vision encoder with the victim LVLM) to extract instruction-aware features of an adversarial image example and the target image, and minimize the distance between these two features to optimize the adversarial example. To further improve the transferability with instruction tuning, we augment the instruction $\boldsymbol{p}^\prime$ with instructions paraphrased from GPT-4. Extensive experiments demonstrate the superiority of our proposed method in targeted attack performance and transferability. The code is available at https://github.com/xunguangwang/InstructTA.



## **18. Deciphering the Definition of Adversarial Robustness for post-hoc OOD Detectors**

cs.CR

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2406.15104v2) [paper-pdf](http://arxiv.org/pdf/2406.15104v2)

**Authors**: Peter Lorenz, Mario Fernandez, Jens Müller, Ullrich Köthe

**Abstract**: Detecting out-of-distribution (OOD) inputs is critical for safely deploying deep learning models in real-world scenarios. In recent years, many OOD detectors have been developed, and even the benchmarking has been standardized, i.e. OpenOOD. The number of post-hoc detectors is growing fast and showing an option to protect a pre-trained classifier against natural distribution shifts, claiming to be ready for real-world scenarios. However, its efficacy in handling adversarial examples has been neglected in the majority of studies. This paper investigates the adversarial robustness of the 16 post-hoc detectors on several evasion attacks and discuss a roadmap towards adversarial defense in OOD detectors.



## **19. Diffusion-based Adversarial Purification for Intrusion Detection**

cs.CR

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2406.17606v1) [paper-pdf](http://arxiv.org/pdf/2406.17606v1)

**Authors**: Mohamed Amine Merzouk, Erwan Beurier, Reda Yaich, Nora Boulahia-Cuppens, Frédéric Cuppens

**Abstract**: The escalating sophistication of cyberattacks has encouraged the integration of machine learning techniques in intrusion detection systems, but the rise of adversarial examples presents a significant challenge. These crafted perturbations mislead ML models, enabling attackers to evade detection or trigger false alerts. As a reaction, adversarial purification has emerged as a compelling solution, particularly with diffusion models showing promising results. However, their purification potential remains unexplored in the context of intrusion detection. This paper demonstrates the effectiveness of diffusion models in purifying adversarial examples in network intrusion detection. Through a comprehensive analysis of the diffusion parameters, we identify optimal configurations maximizing adversarial robustness with minimal impact on normal performance. Importantly, this study reveals insights into the relationship between diffusion noise and diffusion steps, representing a novel contribution to the field. Our experiments are carried out on two datasets and against 5 adversarial attacks. The implementation code is publicly available.



## **20. Treatment of Statistical Estimation Problems in Randomized Smoothing for Adversarial Robustness**

stat.ML

comments are welcome

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2406.17830v1) [paper-pdf](http://arxiv.org/pdf/2406.17830v1)

**Authors**: Vaclav Voracek

**Abstract**: Randomized smoothing is a popular certified defense against adversarial attacks. In its essence, we need to solve a problem of statistical estimation which is usually very time-consuming since we need to perform numerous (usually $10^5$) forward passes of the classifier for every point to be certified. In this paper, we review the statistical estimation problems for randomized smoothing to find out if the computational burden is necessary. In particular, we consider the (standard) task of adversarial robustness where we need to decide if a point is robust at a certain radius or not using as few samples as possible while maintaining statistical guarantees. We present estimation procedures employing confidence sequences enjoying the same statistical guarantees as the standard methods, with the optimal sample complexities for the estimation task and empirically demonstrate their good performance. Additionally, we provide a randomized version of Clopper-Pearson confidence intervals resulting in strictly stronger certificates.



## **21. Detection of Synthetic Face Images: Accuracy, Robustness, Generalization**

cs.CV

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2406.17547v1) [paper-pdf](http://arxiv.org/pdf/2406.17547v1)

**Authors**: Nela Petrzelkova, Jan Cech

**Abstract**: An experimental study on detecting synthetic face images is presented. We collected a dataset, called FF5, of five fake face image generators, including recent diffusion models. We find that a simple model trained on a specific image generator can achieve near-perfect accuracy in separating synthetic and real images. The model handles common image distortions (reduced resolution, compression) by using data augmentation. Moreover, partial manipulations, where synthetic images are blended into real ones by inpainting, are identified and the area of the manipulation is localized by a simple model of YOLO architecture. However, the model turned out to be vulnerable to adversarial attacks and does not generalize to unseen generators. Failure to generalize to detect images produced by a newer generator also occurs for recent state-of-the-art methods, which we tested on Realistic Vision, a fine-tuned version of StabilityAI's Stable Diffusion image generator.



## **22. Practical Membership Inference Attacks against Fine-tuned Large Language Models via Self-prompt Calibration**

cs.CL

Repo: https://github.com/wjfu99/MIA-LLMs

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2311.06062v3) [paper-pdf](http://arxiv.org/pdf/2311.06062v3)

**Authors**: Wenjie Fu, Huandong Wang, Chen Gao, Guanghua Liu, Yong Li, Tao Jiang

**Abstract**: Membership Inference Attacks (MIA) aim to infer whether a target data record has been utilized for model training or not. Prior attempts have quantified the privacy risks of language models (LMs) via MIAs, but there is still no consensus on whether existing MIA algorithms can cause remarkable privacy leakage on practical Large Language Models (LLMs). Existing MIAs designed for LMs can be classified into two categories: reference-free and reference-based attacks. They are both based on the hypothesis that training records consistently strike a higher probability of being sampled. Nevertheless, this hypothesis heavily relies on the overfitting of target models, which will be mitigated by multiple regularization methods and the generalization of LLMs. The reference-based attack seems to achieve promising effectiveness in LLMs, which measures a more reliable membership signal by comparing the probability discrepancy between the target model and the reference model. However, the performance of reference-based attack is highly dependent on a reference dataset that closely resembles the training dataset, which is usually inaccessible in the practical scenario. Overall, existing MIAs are unable to effectively unveil privacy leakage over practical fine-tuned LLMs that are overfitting-free and private. We propose a Membership Inference Attack based on Self-calibrated Probabilistic Variation (SPV-MIA). Specifically, since memorization in LLMs is inevitable during the training process and occurs before overfitting, we introduce a more reliable membership signal, probabilistic variation, which is based on memorization rather than overfitting. Furthermore, we introduce a self-prompt approach, which constructs the dataset to fine-tune the reference model by prompting the target LLM itself. In this manner, the adversary can collect a dataset with a similar distribution from public APIs.



## **23. TSynD: Targeted Synthetic Data Generation for Enhanced Medical Image Classification**

cs.CV

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2406.17473v1) [paper-pdf](http://arxiv.org/pdf/2406.17473v1)

**Authors**: Joshua Niemeijer, Jan Ehrhardt, Hristina Uzunova, Heinz Handels

**Abstract**: The usage of medical image data for the training of large-scale machine learning approaches is particularly challenging due to its scarce availability and the costly generation of data annotations, typically requiring the engagement of medical professionals. The rapid development of generative models allows towards tackling this problem by leveraging large amounts of realistic synthetically generated data for the training process. However, randomly choosing synthetic samples, might not be an optimal strategy.   In this work, we investigate the targeted generation of synthetic training data, in order to improve the accuracy and robustness of image classification. Therefore, our approach aims to guide the generative model to synthesize data with high epistemic uncertainty, since large measures of epistemic uncertainty indicate underrepresented data points in the training set. During the image generation we feed images reconstructed by an auto encoder into the classifier and compute the mutual information over the class-probability distribution as a measure for uncertainty.We alter the feature space of the autoencoder through an optimization process with the objective of maximizing the classifier uncertainty on the decoded image. By training on such data we improve the performance and robustness against test time data augmentations and adversarial attacks on several classifications tasks.



## **24. Low-Cost Privacy-Aware Decentralized Learning**

cs.LG

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2403.11795v2) [paper-pdf](http://arxiv.org/pdf/2403.11795v2)

**Authors**: Sayan Biswas, Davide Frey, Romaric Gaudel, Anne-Marie Kermarrec, Dimitri Lerévérend, Rafael Pires, Rishi Sharma, François Taïani

**Abstract**: This paper introduces ZIP-DL, a novel privacy-aware decentralized learning (DL) algorithm that exploits correlated noise to provide strong privacy protection against a local adversary while yielding efficient convergence guarantees for a low communication cost. The progressive neutralization of the added noise during the distributed aggregation process results in ZIP-DL fostering a high model accuracy under privacy guarantees. ZIP-DL further uses a single communication round between each gradient descent, thus minimizing communication overhead. We provide theoretical guarantees for both convergence speed and privacy guarantees, thereby making ZIP-DL applicable to practical scenarios. Our extensive experimental study shows that ZIP-DL significantly outperforms the state-of-the-art in terms of vulnerability/accuracy trade-off. In particular, ZIP-DL (i) reduces the efficacy of linkability attacks by up to 52 percentage points compared to baseline DL, (ii) improves accuracy by up to 37 percent w.r.t. the state-of-the-art privacy-preserving mechanism operating under the same threat model as ours, when configured to provide the same protection against membership inference attacks, and (iii) reduces communication by up to 10.5x against the same competitor for the same level of protection.



## **25. CuDA2: An approach for Incorporating Traitor Agents into Cooperative Multi-Agent Systems**

cs.LG

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2406.17425v1) [paper-pdf](http://arxiv.org/pdf/2406.17425v1)

**Authors**: Zhen Chen, Yong Liao, Youpeng Zhao, Zipeng Dai, Jian Zhao

**Abstract**: Cooperative Multi-Agent Reinforcement Learning (CMARL) strategies are well known to be vulnerable to adversarial perturbations. Previous works on adversarial attacks have primarily focused on white-box attacks that directly perturb the states or actions of victim agents, often in scenarios with a limited number of attacks. However, gaining complete access to victim agents in real-world environments is exceedingly difficult. To create more realistic adversarial attacks, we introduce a novel method that involves injecting traitor agents into the CMARL system. We model this problem as a Traitor Markov Decision Process (TMDP), where traitors cannot directly attack the victim agents but can influence their formation or positioning through collisions. In TMDP, traitors are trained using the same MARL algorithm as the victim agents, with their reward function set as the negative of the victim agents' reward. Despite this, the training efficiency for traitors remains low because it is challenging for them to directly associate their actions with the victim agents' rewards. To address this issue, we propose the Curiosity-Driven Adversarial Attack (CuDA2) framework. CuDA2 enhances the efficiency and aggressiveness of attacks on the specified victim agents' policies while maintaining the optimal policy invariance of the traitors. Specifically, we employ a pre-trained Random Network Distillation (RND) module, where the extra reward generated by the RND module encourages traitors to explore states unencountered by the victim agents. Extensive experiments on various scenarios from SMAC demonstrate that our CuDA2 framework offers comparable or superior adversarial attack capabilities compared to other baselines.



## **26. Nakamoto Consensus under Bounded Processing Capacity**

cs.CR

ACM Conference on Computer and Communications Security (CCS) 2024

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2303.09113v4) [paper-pdf](http://arxiv.org/pdf/2303.09113v4)

**Authors**: Lucianna Kiffer, Joachim Neu, Srivatsan Sridhar, Aviv Zohar, David Tse

**Abstract**: For Nakamoto's longest-chain consensus protocol, whose proof-of-work (PoW) and proof-of-stake (PoS) variants power major blockchains such as Bitcoin and Cardano, we revisit the classic problem of the security-performance tradeoff: Given a network of nodes with finite communication- and computation-resources, against what fraction of adversary power is Nakamoto consensus (NC) secure for a given block production rate? State-of-the-art analyses of NC fail to answer this question, because their bounded-delay model does not capture the rate limits to nodes' processing of blocks, which cause congestion when blocks are released in quick succession. We develop a new analysis technique to prove a refined security-performance tradeoff for PoW NC in a bounded-capacity model. In this model, we show that, in contrast to the classic bounded-delay model, Nakamoto's private attack is no longer the worst attack, and a new attack we call the teasing strategy, that exploits congestion, is strictly worse. In PoS, equivocating blocks can exacerbate congestion, making traditional PoS NC insecure except at very low block production rates. To counter such equivocation spamming, we present a variant of PoS NC we call Blanking NC (BlaNC), which achieves the same resilience as PoW NC.



## **27. I Don't Know You, But I Can Catch You: Real-Time Defense against Diverse Adversarial Patches for Object Detectors**

cs.CR

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2406.10285v2) [paper-pdf](http://arxiv.org/pdf/2406.10285v2)

**Authors**: Zijin Lin, Yue Zhao, Kai Chen, Jinwen He

**Abstract**: Deep neural networks (DNNs) have revolutionized the field of computer vision like object detection with their unparalleled performance. However, existing research has shown that DNNs are vulnerable to adversarial attacks. In the physical world, an adversary could exploit adversarial patches to implement a Hiding Attack (HA) which patches the target object to make it disappear from the detector, and an Appearing Attack (AA) which fools the detector into misclassifying the patch as a specific object. Recently, many defense methods for detectors have been proposed to mitigate the potential threats of adversarial patches. However, such methods still have limitations in generalization, robustness and efficiency. Most defenses are only effective against the HA, leaving the detector vulnerable to the AA.   In this paper, we propose \textit{NutNet}, an innovative model for detecting adversarial patches, with high generalization, robustness and efficiency. With experiments for six detectors including YOLOv2-v4, SSD, Faster RCNN and DETR on both digital and physical domains, the results show that our proposed method can effectively defend against both the HA and AA, with only 0.4\% sacrifice of the clean performance. We compare NutNet with four baseline defense methods for detectors, and our method exhibits an average defense performance that is over 2.4 times and 4.7 times higher than existing approaches for HA and AA, respectively. In addition, NutNet only increases the inference time by 8\%, which can meet the real-time requirements of the detection systems. Demos of NutNet are available at: \url{https://sites.google.com/view/nutnet}.



## **28. ECLIPSE: Expunging Clean-label Indiscriminate Poisons via Sparse Diffusion Purification**

cs.CR

Accepted by ESORICS 2024

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2406.15093v2) [paper-pdf](http://arxiv.org/pdf/2406.15093v2)

**Authors**: Xianlong Wang, Shengshan Hu, Yechao Zhang, Ziqi Zhou, Leo Yu Zhang, Peng Xu, Wei Wan, Hai Jin

**Abstract**: Clean-label indiscriminate poisoning attacks add invisible perturbations to correctly labeled training images, thus dramatically reducing the generalization capability of the victim models. Recently, some defense mechanisms have been proposed such as adversarial training, image transformation techniques, and image purification. However, these schemes are either susceptible to adaptive attacks, built on unrealistic assumptions, or only effective against specific poison types, limiting their universal applicability. In this research, we propose a more universally effective, practical, and robust defense scheme called ECLIPSE. We first investigate the impact of Gaussian noise on the poisons and theoretically prove that any kind of poison will be largely assimilated when imposing sufficient random noise. In light of this, we assume the victim has access to an extremely limited number of clean images (a more practical scene) and subsequently enlarge this sparse set for training a denoising probabilistic model (a universal denoising tool). We then begin by introducing Gaussian noise to absorb the poisons and then apply the model for denoising, resulting in a roughly purified dataset. Finally, to address the trade-off of the inconsistency in the assimilation sensitivity of different poisons by Gaussian noise, we propose a lightweight corruption compensation module to effectively eliminate residual poisons, providing a more universal defense approach. Extensive experiments demonstrate that our defense approach outperforms 10 state-of-the-art defenses. We also propose an adaptive attack against ECLIPSE and verify the robustness of our defense scheme. Our code is available at https://github.com/CGCL-codes/ECLIPSE.



## **29. Automated Adversarial Discovery for Safety Classifiers**

cs.CL

Published at Fourth Workshop on TrustworthyNLP (TrustNLP) at NAACL  2024

**SubmitDate**: 2024-06-24    [abs](http://arxiv.org/abs/2406.17104v1) [paper-pdf](http://arxiv.org/pdf/2406.17104v1)

**Authors**: Yash Kumar Lal, Preethi Lahoti, Aradhana Sinha, Yao Qin, Ananth Balashankar

**Abstract**: Safety classifiers are critical in mitigating toxicity on online forums such as social media and in chatbots. Still, they continue to be vulnerable to emergent, and often innumerable, adversarial attacks. Traditional automated adversarial data generation methods, however, tend to produce attacks that are not diverse, but variations of previously observed harm types. We formalize the task of automated adversarial discovery for safety classifiers - to find new attacks along previously unseen harm dimensions that expose new weaknesses in the classifier. We measure progress on this task along two key axes (1) adversarial success: does the attack fool the classifier? and (2) dimensional diversity: does the attack represent a previously unseen harm type? Our evaluation of existing attack generation methods on the CivilComments toxicity task reveals their limitations: Word perturbation attacks fail to fool classifiers, while prompt-based LLM attacks have more adversarial success, but lack dimensional diversity. Even our best-performing prompt-based method finds new successful attacks on unseen harm dimensions of attacks only 5\% of the time. Automatically finding new harmful dimensions of attack is crucial and there is substantial headroom for future research on our new task.



## **30. Robust Distribution Learning with Local and Global Adversarial Corruptions**

cs.LG

Accepted for presentation at the Conference on Learning Theory (COLT)  2024

**SubmitDate**: 2024-06-24    [abs](http://arxiv.org/abs/2406.06509v2) [paper-pdf](http://arxiv.org/pdf/2406.06509v2)

**Authors**: Sloan Nietert, Ziv Goldfeld, Soroosh Shafiee

**Abstract**: We consider learning in an adversarial environment, where an $\varepsilon$-fraction of samples from a distribution $P$ are arbitrarily modified (global corruptions) and the remaining perturbations have average magnitude bounded by $\rho$ (local corruptions). Given access to $n$ such corrupted samples, we seek a computationally efficient estimator $\hat{P}_n$ that minimizes the Wasserstein distance $\mathsf{W}_1(\hat{P}_n,P)$. In fact, we attack the fine-grained task of minimizing $\mathsf{W}_1(\Pi_\# \hat{P}_n, \Pi_\# P)$ for all orthogonal projections $\Pi \in \mathbb{R}^{d \times d}$, with performance scaling with $\mathrm{rank}(\Pi) = k$. This allows us to account simultaneously for mean estimation ($k=1$), distribution estimation ($k=d$), as well as the settings interpolating between these two extremes. We characterize the optimal population-limit risk for this task and then develop an efficient finite-sample algorithm with error bounded by $\sqrt{\varepsilon k} + \rho + \tilde{O}(d\sqrt{k}n^{-1/(k \lor 2)})$ when $P$ has bounded covariance. This guarantee holds uniformly in $k$ and is minimax optimal up to the sub-optimality of the plug-in estimator when $\rho = \varepsilon = 0$. Our efficient procedure relies on a novel trace norm approximation of an ideal yet intractable 2-Wasserstein projection estimator. We apply this algorithm to robust stochastic optimization, and, in the process, uncover a new method for overcoming the curse of dimensionality in Wasserstein distributionally robust optimization.



## **31. Pandora's White-Box: Precise Training Data Detection and Extraction in Large Language Models**

cs.CR

**SubmitDate**: 2024-06-24    [abs](http://arxiv.org/abs/2402.17012v3) [paper-pdf](http://arxiv.org/pdf/2402.17012v3)

**Authors**: Jeffrey G. Wang, Jason Wang, Marvin Li, Seth Neel

**Abstract**: In this paper we develop state-of-the-art privacy attacks against Large Language Models (LLMs), where an adversary with some access to the model tries to learn something about the underlying training data. Our headline results are new membership inference attacks (MIAs) against pretrained LLMs that perform hundreds of times better than baseline attacks, and a pipeline showing that over 50% (!) of the fine-tuning dataset can be extracted from a fine-tuned LLM in natural settings. We consider varying degrees of access to the underlying model, pretraining and fine-tuning data, and both MIAs and training data extraction. For pretraining data, we propose two new MIAs: a supervised neural network classifier that predicts training data membership on the basis of (dimensionality-reduced) model gradients, as well as a variant of this attack that only requires logit access to the model by leveraging recent model-stealing work on LLMs. To our knowledge this is the first MIA that explicitly incorporates model-stealing information. Both attacks outperform existing black-box baselines, and our supervised attack closes the gap between MIA attack success against LLMs and the strongest known attacks for other machine learning models. In fine-tuning, we find that a simple attack based on the ratio of the loss between the base and fine-tuned models is able to achieve near-perfect MIA performance; we then leverage our MIA to extract a large fraction of the fine-tuning dataset from fine-tuned Pythia and Llama models. Our code is available at github.com/safr-ai-lab/pandora-llm.



## **32. Security of Partially Corrupted Repeater Chains**

quant-ph

**SubmitDate**: 2024-06-24    [abs](http://arxiv.org/abs/2406.16651v1) [paper-pdf](http://arxiv.org/pdf/2406.16651v1)

**Authors**: Adrian Harkness, Walter O. Krawec, Bing Wang

**Abstract**: Quantum Key Distribution allows two parties to establish a secret key that is secure against computationally unbounded adversaries. To extend the distance between parties, quantum networks, and in particular repeater chains, are vital. Typically, security in such scenarios assumes the absolute worst case: namely, an adversary has complete control over all repeaters and fiber links in a network and is able to replace them with perfect devices, thus allowing her to hide her attack within the expected natural noise. In a large-scale network, however, such a powerful attack may be infeasible. In this paper, we analyze the case where the adversary can only corrupt a contiguous subset of a repeater chain connecting Alice and Bob, while some portion of the network near Alice and Bob may be considered safe from attack (though still noisy). We derive a rigorous finite key proof of security assuming this attack model and show that improved performance and noise tolerances are possible.



## **33. UNICAD: A Unified Approach for Attack Detection, Noise Reduction and Novel Class Identification**

cs.CV

**SubmitDate**: 2024-06-24    [abs](http://arxiv.org/abs/2406.16501v1) [paper-pdf](http://arxiv.org/pdf/2406.16501v1)

**Authors**: Alvaro Lopez Pellicer, Kittipos Giatgong, Yi Li, Neeraj Suri, Plamen Angelov

**Abstract**: As the use of Deep Neural Networks (DNNs) becomes pervasive, their vulnerability to adversarial attacks and limitations in handling unseen classes poses significant challenges. The state-of-the-art offers discrete solutions aimed to tackle individual issues covering specific adversarial attack scenarios, classification or evolving learning. However, real-world systems need to be able to detect and recover from a wide range of adversarial attacks without sacrificing classification accuracy and to flexibly act in {\bf unseen} scenarios. In this paper, UNICAD, is proposed as a novel framework that integrates a variety of techniques to provide an adaptive solution.   For the targeted image classification, UNICAD achieves accurate image classification, detects unseen classes, and recovers from adversarial attacks using Prototype and Similarity-based DNNs with denoising autoencoders. Our experiments performed on the CIFAR-10 dataset highlight UNICAD's effectiveness in adversarial mitigation and unseen class classification, outperforming traditional models.



## **34. The Economic Limits of Permissionless Consensus**

cs.DC

**SubmitDate**: 2024-06-24    [abs](http://arxiv.org/abs/2405.09173v2) [paper-pdf](http://arxiv.org/pdf/2405.09173v2)

**Authors**: Eric Budish, Andrew Lewis-Pye, Tim Roughgarden

**Abstract**: The purpose of a consensus protocol is to keep a distributed network of nodes "in sync," even in the presence of an unpredictable communication network and adversarial behavior by some of the participating nodes. In the permissionless setting, these nodes may be operated by unknown players, with each player free to use multiple identifiers and to start or stop running the protocol at any time. Establishing that a permissionless consensus protocol is "secure" thus requires both a distributed computing argument (that the protocol guarantees consistency and liveness unless the fraction of adversarial participation is sufficiently large) and an economic argument (that carrying out an attack would be prohibitively expensive for an attacker). There is a mature toolbox for assembling arguments of the former type; the goal of this paper is to lay the foundations for arguments of the latter type.   An ideal permissionless consensus protocol would, in addition to satisfying standard consistency and liveness guarantees, render consistency violations prohibitively expensive for the attacker without collateral damage to honest participants. We make this idea precise with our notion of the EAAC (expensive to attack in the absence of collapse) property, and prove the following results:   1. In the synchronous and dynamically available setting, with an adversary that controls at least one-half of the overall resources, no protocol can be EAAC.   2. In the partially synchronous and quasi-permissionless setting, with an adversary that controls at least one-third of the overall resources, no protocol can be EAAC.   3. In the synchronous and quasi-permissionless setting, there is a proof-of-stake protocol that, provided the adversary controls less than two-thirds of the overall stake, satisfies the EAAC property.   All three results are optimal with respect to the size of the adversary.



## **35. Investigating the Influence of Prompt-Specific Shortcuts in AI Generated Text Detection**

cs.CL

19 pages, 3 figures, 13 tables, under review

**SubmitDate**: 2024-06-24    [abs](http://arxiv.org/abs/2406.16275v1) [paper-pdf](http://arxiv.org/pdf/2406.16275v1)

**Authors**: Choonghyun Park, Hyuhng Joon Kim, Junyeob Kim, Youna Kim, Taeuk Kim, Hyunsoo Cho, Hwiyeol Jo, Sang-goo Lee, Kang Min Yoo

**Abstract**: AI Generated Text (AIGT) detectors are developed with texts from humans and LLMs of common tasks. Despite the diversity of plausible prompt choices, these datasets are generally constructed with a limited number of prompts. The lack of prompt variation can introduce prompt-specific shortcut features that exist in data collected with the chosen prompt, but do not generalize to others. In this paper, we analyze the impact of such shortcuts in AIGT detection. We propose Feedback-based Adversarial Instruction List Optimization (FAILOpt), an attack that searches for instructions deceptive to AIGT detectors exploiting prompt-specific shortcuts. FAILOpt effectively drops the detection performance of the target detector, comparable to other attacks based on adversarial in-context examples. We also utilize our method to enhance the robustness of the detector by mitigating the shortcuts. Based on the findings, we further train the classifier with the dataset augmented by FAILOpt prompt. The augmented classifier exhibits improvements across generation models, tasks, and attacks. Our code will be available at https://github.com/zxcvvxcz/FAILOpt.



## **36. Pareto Adversarial Robustness: Balancing Spatial Robustness and Sensitivity-based Robustness**

cs.LG

Published in SCIENCE CHINA Information Sciences (SCIS) in 2023.  Please also refer to the published version in the Journal reference  https://www.sciengine.com/SCIS/doi/10.1007/s11432-022-3861-8

**SubmitDate**: 2024-06-23    [abs](http://arxiv.org/abs/2111.01996v3) [paper-pdf](http://arxiv.org/pdf/2111.01996v3)

**Authors**: Ke Sun, Mingjie Li, Zhouchen Lin

**Abstract**: Adversarial robustness, which primarily comprises sensitivity-based robustness and spatial robustness, plays an integral part in achieving robust generalization. In this paper, we endeavor to design strategies to achieve universal adversarial robustness. To achieve this, we first investigate the relatively less-explored realm of spatial robustness. Then, we integrate the existing spatial robustness methods by incorporating both local and global spatial vulnerability into a unified spatial attack and adversarial training approach. Furthermore, we present a comprehensive relationship between natural accuracy, sensitivity-based robustness, and spatial robustness, supported by strong evidence from the perspective of robust representation. Crucially, to reconcile the interplay between the mutual impacts of various robustness components into one unified framework, we incorporate the \textit{Pareto criterion} into the adversarial robustness analysis, yielding a novel strategy called Pareto Adversarial Training for achieving universal robustness. The resulting Pareto front, which delineates the set of optimal solutions, provides an optimal balance between natural accuracy and various adversarial robustness. This sheds light on solutions for achieving universal robustness in the future. To the best of our knowledge, we are the first to consider universal adversarial robustness via multi-objective optimization.



## **37. Federated Adversarial Learning for Robust Autonomous Landing Runway Detection**

cs.CV

ICANN2024

**SubmitDate**: 2024-06-22    [abs](http://arxiv.org/abs/2406.15925v1) [paper-pdf](http://arxiv.org/pdf/2406.15925v1)

**Authors**: Yi Li, Plamen Angelov, Zhengxin Yu, Alvaro Lopez Pellicer, Neeraj Suri

**Abstract**: As the development of deep learning techniques in autonomous landing systems continues to grow, one of the major challenges is trust and security in the face of possible adversarial attacks. In this paper, we propose a federated adversarial learning-based framework to detect landing runways using paired data comprising of clean local data and its adversarial version. Firstly, the local model is pre-trained on a large-scale lane detection dataset. Then, instead of exploiting large instance-adaptive models, we resort to a parameter-efficient fine-tuning method known as scale and shift deep features (SSF), upon the pre-trained model. Secondly, in each SSF layer, distributions of clean local data and its adversarial version are disentangled for accurate statistics estimation. To the best of our knowledge, this marks the first instance of federated learning work that address the adversarial sample problem in landing runway detection. Our experimental evaluations over both synthesis and real images of Landing Approach Runway Detection (LARD) dataset consistently demonstrate good performance of the proposed federated adversarial learning and robust to adversarial attacks.



## **38. A Recipe for Improved Certifiable Robustness**

cs.LG

**SubmitDate**: 2024-06-22    [abs](http://arxiv.org/abs/2310.02513v2) [paper-pdf](http://arxiv.org/pdf/2310.02513v2)

**Authors**: Kai Hu, Klas Leino, Zifan Wang, Matt Fredrikson

**Abstract**: Recent studies have highlighted the potential of Lipschitz-based methods for training certifiably robust neural networks against adversarial attacks. A key challenge, supported both theoretically and empirically, is that robustness demands greater network capacity and more data than standard training. However, effectively adding capacity under stringent Lipschitz constraints has proven more difficult than it may seem, evident by the fact that state-of-the-art approach tend more towards \emph{underfitting} than overfitting. Moreover, we posit that a lack of careful exploration of the design space for Lipshitz-based approaches has left potential performance gains on the table. In this work, we provide a more comprehensive evaluation to better uncover the potential of Lipschitz-based certification methods. Using a combination of novel techniques, design optimizations, and synthesis of prior work, we are able to significantly improve the state-of-the-art VRA for deterministic certification on a variety of benchmark datasets, and over a range of perturbation sizes. Of particular note, we discover that the addition of large ``Cholesky-orthogonalized residual dense'' layers to the end of existing state-of-the-art Lipschitz-controlled ResNet architectures is especially effective for increasing network capacity and performance. Combined with filtered generative data augmentation, our final results further the state of the art deterministic VRA by up to 8.5 percentage points\footnote{Code is available at \url{https://github.com/hukkai/liresnet}}.



## **39. The Effect of Similarity Measures on Accurate Stability Estimates for Local Surrogate Models in Text-based Explainable AI**

cs.LG

11 pages, 8 Tables

**SubmitDate**: 2024-06-22    [abs](http://arxiv.org/abs/2406.15839v1) [paper-pdf](http://arxiv.org/pdf/2406.15839v1)

**Authors**: Christopher Burger, Charles Walter, Thai Le

**Abstract**: Recent work has investigated the vulnerability of local surrogate methods to adversarial perturbations on a machine learning (ML) model's inputs, where the explanation is manipulated while the meaning and structure of the original input remains similar under the complex model. While weaknesses across many methods have been shown to exist, the reasons behind why still remain little explored. Central to the concept of adversarial attacks on explainable AI (XAI) is the similarity measure used to calculate how one explanation differs from another A poor choice of similarity measure can result in erroneous conclusions on the efficacy of an XAI method. Too sensitive a measure results in exaggerated vulnerability, while too coarse understates its weakness. We investigate a variety of similarity measures designed for text-based ranked lists including Kendall's Tau, Spearman's Footrule and Rank-biased Overlap to determine how substantial changes in the type of measure or threshold of success affect the conclusions generated from common adversarial attack processes. Certain measures are found to be overly sensitive, resulting in erroneous estimates of stability.



## **40. DataFreeShield: Defending Adversarial Attacks without Training Data**

cs.LG

ICML 2024

**SubmitDate**: 2024-06-21    [abs](http://arxiv.org/abs/2406.15635v1) [paper-pdf](http://arxiv.org/pdf/2406.15635v1)

**Authors**: Hyeyoon Lee, Kanghyun Choi, Dain Kwon, Sunjong Park, Mayoore Selvarasa Jaiswal, Noseong Park, Jonghyun Choi, Jinho Lee

**Abstract**: Recent advances in adversarial robustness rely on an abundant set of training data, where using external or additional datasets has become a common setting. However, in real life, the training data is often kept private for security and privacy issues, while only the pretrained weight is available to the public. In such scenarios, existing methods that assume accessibility to the original data become inapplicable. Thus we investigate the pivotal problem of data-free adversarial robustness, where we try to achieve adversarial robustness without accessing any real data. Through a preliminary study, we highlight the severity of the problem by showing that robustness without the original dataset is difficult to achieve, even with similar domain datasets. To address this issue, we propose DataFreeShield, which tackles the problem from two perspectives: surrogate dataset generation and adversarial training using the generated data. Through extensive validation, we show that DataFreeShield outperforms baselines, demonstrating that the proposed method sets the first entirely data-free solution for the adversarial robustness problem.



## **41. Efficient Adversarial Training in LLMs with Continuous Attacks**

cs.LG

19 pages, 4 figures

**SubmitDate**: 2024-06-21    [abs](http://arxiv.org/abs/2405.15589v2) [paper-pdf](http://arxiv.org/pdf/2405.15589v2)

**Authors**: Sophie Xhonneux, Alessandro Sordoni, Stephan Günnemann, Gauthier Gidel, Leo Schwinn

**Abstract**: Large language models (LLMs) are vulnerable to adversarial attacks that can bypass their safety guardrails. In many domains, adversarial training has proven to be one of the most promising methods to reliably improve robustness against such attacks. Yet, in the context of LLMs, current methods for adversarial training are hindered by the high computational costs required to perform discrete adversarial attacks at each training iteration. We address this problem by instead calculating adversarial attacks in the continuous embedding space of the LLM, which is orders of magnitudes more efficient. We propose a fast adversarial training algorithm (C-AdvUL) composed of two losses: the first makes the model robust on continuous embedding attacks computed on an adversarial behaviour dataset; the second ensures the usefulness of the final model by fine-tuning on utility data. Moreover, we introduce C-AdvIPO, an adversarial variant of IPO that does not require utility data for adversarially robust alignment. Our empirical evaluation on four models from different families (Gemma, Phi3, Mistral, Zephyr) and at different scales (2B, 3.8B, 7B) shows that both algorithms substantially enhance LLM robustness against discrete attacks (GCG, AutoDAN, PAIR), while maintaining utility. Our results demonstrate that robustness to continuous perturbations can extrapolate to discrete threat models. Thereby, we present a path toward scalable adversarial training algorithms for robustly aligning LLMs.



## **42. Stackelberg Games with $k$-Submodular Function under Distributional Risk-Receptiveness and Robustness**

math.OC

**SubmitDate**: 2024-06-21    [abs](http://arxiv.org/abs/2406.13023v2) [paper-pdf](http://arxiv.org/pdf/2406.13023v2)

**Authors**: Seonghun Park, Manish Bansal

**Abstract**: We study submodular optimization in adversarial context, applicable to machine learning problems such as feature selection using data susceptible to uncertainties and attacks. We focus on Stackelberg games between an attacker (or interdictor) and a defender where the attacker aims to minimize the defender's objective of maximizing a $k$-submodular function. We allow uncertainties arising from the success of attacks and inherent data noise, and address challenges due to incomplete knowledge of the probability distribution of random parameters. Specifically, we introduce Distributionally Risk-Averse $k$-Submodular Interdiction Problem (DRA $k$-SIP) and Distributionally Risk-Receptive $k$-Submodular Interdiction Problem (DRR $k$-SIP) along with finitely convergent exact algorithms for solving them. The DRA $k$-SIP solution allows risk-averse interdictor to develop robust strategies for real-world uncertainties. Conversely, DRR $k$-SIP solution suggests aggressive tactics for attackers, willing to embrace (distributional) risk to inflict maximum damage, identifying critical vulnerable components, which can be used for the defender's defensive strategies. The optimal values derived from both DRA $k$-SIP and DRR $k$-SIP offer a confidence interval-like range for the expected value of the defender's objective function, capturing distributional ambiguity. We conduct computational experiments using instances of feature selection and sensor placement problems, and Wisconsin breast cancer data and synthetic data, respectively.



## **43. AdvQuNN: A Methodology for Analyzing the Adversarial Robustness of Quanvolutional Neural Networks**

quant-ph

7 pages, 6 figures

**SubmitDate**: 2024-06-21    [abs](http://arxiv.org/abs/2403.05596v2) [paper-pdf](http://arxiv.org/pdf/2403.05596v2)

**Authors**: Walid El Maouaki, Alberto Marchisio, Taoufik Said, Mohamed Bennai, Muhammad Shafique

**Abstract**: Recent advancements in quantum computing have led to the development of hybrid quantum neural networks (HQNNs) that employ a mixed set of quantum layers and classical layers, such as Quanvolutional Neural Networks (QuNNs). While several works have shown security threats of classical neural networks, such as adversarial attacks, their impact on QuNNs is still relatively unexplored. This work tackles this problem by designing AdvQuNN, a specialized methodology to investigate the robustness of HQNNs like QuNNs against adversarial attacks. It employs different types of Ansatzes as parametrized quantum circuits and different types of adversarial attacks. This study aims to rigorously assess the influence of quantum circuit architecture on the resilience of QuNN models, which opens up new pathways for enhancing the robustness of QuNNs and advancing the field of quantum cybersecurity. Our results show that, compared to classical convolutional networks, QuNNs achieve up to 60\% higher robustness for the MNIST and 40\% for FMNIST datasets.



## **44. Trading Devil: Robust backdoor attack via Stochastic investment models and Bayesian approach**

cs.CR

(Last update!, a constructive comment from arxiv led to this latest  update ) Stochastic investment models and a Bayesian approach to better  modeling of uncertainty : adversarial machine learning or Stochastic market.  arXiv admin note: substantial text overlap with arXiv:2402.05967 (see this  link to the paper by : Orson Mengara)

**SubmitDate**: 2024-06-21    [abs](http://arxiv.org/abs/2406.10719v3) [paper-pdf](http://arxiv.org/pdf/2406.10719v3)

**Authors**: Orson Mengara

**Abstract**: With the growing use of voice-activated systems and speech recognition technologies, the danger of backdoor attacks on audio data has grown significantly. This research looks at a specific type of attack, known as a Stochastic investment-based backdoor attack (MarketBack), in which adversaries strategically manipulate the stylistic properties of audio to fool speech recognition systems. The security and integrity of machine learning models are seriously threatened by backdoor attacks, in order to maintain the reliability of audio applications and systems, the identification of such attacks becomes crucial in the context of audio data. Experimental results demonstrated that MarketBack is feasible to achieve an average attack success rate close to 100% in seven victim models when poisoning less than 1% of the training data.



## **45. Fingerprint Membership and Identity Inference Against Generative Adversarial Networks**

cs.CV

Paper submitted at "Pattern Recognition Letters", 9 pages, 6 images

**SubmitDate**: 2024-06-21    [abs](http://arxiv.org/abs/2406.15253v1) [paper-pdf](http://arxiv.org/pdf/2406.15253v1)

**Authors**: Saverio Cavasin, Daniele Mari, Simone Milani, Mauro Conti

**Abstract**: Generative models are gaining significant attention as potential catalysts for a novel industrial revolution. Since automated sample generation can be useful to solve privacy and data scarcity issues that usually affect learned biometric models, such technologies became widely spread in this field. In this paper, we assess the vulnerabilities of generative machine learning models concerning identity protection by designing and testing an identity inference attack on fingerprint datasets created by means of a generative adversarial network. Experimental results show that the proposed solution proves to be effective under different configurations and easily extendable to other biometric measurements.



## **46. Injecting Bias in Text-To-Image Models via Composite-Trigger Backdoors**

cs.LG

**SubmitDate**: 2024-06-21    [abs](http://arxiv.org/abs/2406.15213v1) [paper-pdf](http://arxiv.org/pdf/2406.15213v1)

**Authors**: Ali Naseh, Jaechul Roh, Eugene Bagdasaryan, Amir Houmansadr

**Abstract**: Recent advances in large text-conditional image generative models such as Stable Diffusion, Midjourney, and DALL-E 3 have revolutionized the field of image generation, allowing users to produce high-quality, realistic images from textual prompts. While these developments have enhanced artistic creation and visual communication, they also present an underexplored attack opportunity: the possibility of inducing biases by an adversary into the generated images for malicious intentions, e.g., to influence society and spread propaganda. In this paper, we demonstrate the possibility of such a bias injection threat by an adversary who backdoors such models with a small number of malicious data samples; the implemented backdoor is activated when special triggers exist in the input prompt of the backdoored models. On the other hand, the model's utility is preserved in the absence of the triggers, making the attack highly undetectable. We present a novel framework that enables efficient generation of poisoning samples with composite (multi-word) triggers for such an attack. Our extensive experiments using over 1 million generated images and against hundreds of fine-tuned models demonstrate the feasibility of the presented backdoor attack. We illustrate how these biases can bypass conventional detection mechanisms, highlighting the challenges in proving the existence of biases within operational constraints. Our cost analysis confirms the low financial barrier to executing such attacks, underscoring the need for robust defensive strategies against such vulnerabilities in text-to-image generation models.



## **47. From LLMs to MLLMs: Exploring the Landscape of Multimodal Jailbreaking**

cs.CL

**SubmitDate**: 2024-06-21    [abs](http://arxiv.org/abs/2406.14859v1) [paper-pdf](http://arxiv.org/pdf/2406.14859v1)

**Authors**: Siyuan Wang, Zhuohan Long, Zhihao Fan, Zhongyu Wei

**Abstract**: The rapid development of Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) has exposed vulnerabilities to various adversarial attacks. This paper provides a comprehensive overview of jailbreaking research targeting both LLMs and MLLMs, highlighting recent advancements in evaluation benchmarks, attack techniques and defense strategies. Compared to the more advanced state of unimodal jailbreaking, multimodal domain remains underexplored. We summarize the limitations and potential research directions of multimodal jailbreaking, aiming to inspire future research and further enhance the robustness and security of MLLMs.



## **48. Steering Without Side Effects: Improving Post-Deployment Control of Language Models**

cs.CL

**SubmitDate**: 2024-06-21    [abs](http://arxiv.org/abs/2406.15518v1) [paper-pdf](http://arxiv.org/pdf/2406.15518v1)

**Authors**: Asa Cooper Stickland, Alexander Lyzhov, Jacob Pfau, Salsabila Mahdi, Samuel R. Bowman

**Abstract**: Language models (LMs) have been shown to behave unexpectedly post-deployment. For example, new jailbreaks continually arise, allowing model misuse, despite extensive red-teaming and adversarial training from developers. Given most model queries are unproblematic and frequent retraining results in unstable user experience, methods for mitigation of worst-case behavior should be targeted. One such method is classifying inputs as potentially problematic, then selectively applying steering vectors on these problematic inputs, i.e. adding particular vectors to model hidden states. However, steering vectors can also negatively affect model performance, which will be an issue on cases where the classifier was incorrect. We present KL-then-steer (KTS), a technique that decreases the side effects of steering while retaining its benefits, by first training a model to minimize Kullback-Leibler (KL) divergence between a steered and unsteered model on benign inputs, then steering the model that has undergone this training. Our best method prevents 44% of jailbreak attacks compared to the original Llama-2-chat-7B model while maintaining helpfulness (as measured by MT-Bench) on benign requests almost on par with the original LM. To demonstrate the generality and transferability of our method beyond jailbreaks, we show that our KTS model can be steered to reduce bias towards user-suggested answers on TruthfulQA. Code is available: https://github.com/AsaCooperStickland/kl-then-steer.



## **49. FedSecurity: Benchmarking Attacks and Defenses in Federated Learning and Federated LLMs**

cs.CR

**SubmitDate**: 2024-06-21    [abs](http://arxiv.org/abs/2306.04959v5) [paper-pdf](http://arxiv.org/pdf/2306.04959v5)

**Authors**: Shanshan Han, Baturalp Buyukates, Zijian Hu, Han Jin, Weizhao Jin, Lichao Sun, Xiaoyang Wang, Wenxuan Wu, Chulin Xie, Yuhang Yao, Kai Zhang, Qifan Zhang, Yuhui Zhang, Carlee Joe-Wong, Salman Avestimehr, Chaoyang He

**Abstract**: This paper introduces FedSecurity, an end-to-end benchmark that serves as a supplementary component of the FedML library for simulating adversarial attacks and corresponding defense mechanisms in Federated Learning (FL). FedSecurity eliminates the need for implementing the fundamental FL procedures, e.g., FL training and data loading, from scratch, thus enables users to focus on developing their own attack and defense strategies. It contains two key components, including FedAttacker that conducts a variety of attacks during FL training, and FedDefender that implements defensive mechanisms to counteract these attacks. FedSecurity has the following features: i) It offers extensive customization options to accommodate a broad range of machine learning models (e.g., Logistic Regression, ResNet, and GAN) and FL optimizers (e.g., FedAVG, FedOPT, and FedNOVA); ii) it enables exploring the effectiveness of attacks and defenses across different datasets and models; and iii) it supports flexible configuration and customization through a configuration file and some APIs. We further demonstrate FedSecurity's utility and adaptability through federated training of Large Language Models (LLMs) to showcase its potential on a wide range of complex applications.



## **50. AdaNCA: Neural Cellular Automata As Adaptors For More Robust Vision Transformer**

cs.CV

26 pages, 11 figures

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2406.08298v3) [paper-pdf](http://arxiv.org/pdf/2406.08298v3)

**Authors**: Yitao Xu, Tong Zhang, Sabine Süsstrunk

**Abstract**: Vision Transformers (ViTs) have demonstrated remarkable performance in image classification tasks, particularly when equipped with local information via region attention or convolutions. While such architectures improve the feature aggregation from different granularities, they often fail to contribute to the robustness of the networks. Neural Cellular Automata (NCA) enables the modeling of global cell representations through local interactions, with its training strategies and architecture design conferring strong generalization ability and robustness against noisy inputs. In this paper, we propose Adaptor Neural Cellular Automata (AdaNCA) for Vision Transformer that uses NCA as plug-in-play adaptors between ViT layers, enhancing ViT's performance and robustness against adversarial samples as well as out-of-distribution inputs. To overcome the large computational overhead of standard NCAs, we propose Dynamic Interaction for more efficient interaction learning. Furthermore, we develop an algorithm for identifying the most effective insertion points for AdaNCA based on our analysis of AdaNCA placement and robustness improvement. With less than a 3% increase in parameters, AdaNCA contributes to more than 10% absolute improvement in accuracy under adversarial attacks on the ImageNet1K benchmark. Moreover, we demonstrate with extensive evaluations across 8 robustness benchmarks and 4 ViT architectures that AdaNCA, as a plug-in-play module, consistently improves the robustness of ViTs.



