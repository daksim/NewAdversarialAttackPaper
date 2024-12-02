# Latest Large Language Model Attack Papers
**update at 2024-12-02 09:43:53**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Neutralizing Backdoors through Information Conflicts for Large Language Models**

cs.CL

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2411.18280v1) [paper-pdf](http://arxiv.org/pdf/2411.18280v1)

**Authors**: Chen Chen, Yuchen Sun, Xueluan Gong, Jiaxin Gao, Kwok-Yan Lam

**Abstract**: Large language models (LLMs) have seen significant advancements, achieving superior performance in various Natural Language Processing (NLP) tasks, from understanding to reasoning. However, they remain vulnerable to backdoor attacks, where models behave normally for standard queries but generate harmful responses or unintended output when specific triggers are activated. Existing backdoor defenses often suffer from drawbacks that they either focus on detection without removal, rely on rigid assumptions about trigger properties, or prove to be ineffective against advanced attacks like multi-trigger backdoors. In this paper, we present a novel method to eliminate backdoor behaviors from LLMs through the construction of information conflicts using both internal and external mechanisms. Internally, we leverage a lightweight dataset to train a conflict model, which is then merged with the backdoored model to neutralize malicious behaviors by embedding contradictory information within the model's parametric memory. Externally, we incorporate convincing contradictory evidence into the prompt to challenge the model's internal backdoor knowledge. Experimental results on classification and conversational tasks across 4 widely used LLMs demonstrate that our method outperforms 8 state-of-the-art backdoor defense baselines. We can reduce the attack success rate of advanced backdoor attacks by up to 98% while maintaining over 90% clean data accuracy. Furthermore, our method has proven to be robust against adaptive backdoor attacks. The code will be open-sourced upon publication.



## **2. Visual Adversarial Attack on Vision-Language Models for Autonomous Driving**

cs.CV

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2411.18275v1) [paper-pdf](http://arxiv.org/pdf/2411.18275v1)

**Authors**: Tianyuan Zhang, Lu Wang, Xinwei Zhang, Yitong Zhang, Boyi Jia, Siyuan Liang, Shengshan Hu, Qiang Fu, Aishan Liu, Xianglong Liu

**Abstract**: Vision-language models (VLMs) have significantly advanced autonomous driving (AD) by enhancing reasoning capabilities. However, these models remain highly vulnerable to adversarial attacks. While existing research has primarily focused on general VLM attacks, the development of attacks tailored to the safety-critical AD context has been largely overlooked. In this paper, we take the first step toward designing adversarial attacks specifically targeting VLMs in AD, exposing the substantial risks these attacks pose within this critical domain. We identify two unique challenges for effective adversarial attacks on AD VLMs: the variability of textual instructions and the time-series nature of visual scenarios. To this end, we propose ADvLM, the first visual adversarial attack framework specifically designed for VLMs in AD. Our framework introduces Semantic-Invariant Induction, which uses a large language model to create a diverse prompt library of textual instructions with consistent semantic content, guided by semantic entropy. Building on this, we introduce Scenario-Associated Enhancement, an approach where attention mechanisms select key frames and perspectives within driving scenarios to optimize adversarial perturbations that generalize across the entire scenario. Extensive experiments on several AD VLMs over multiple benchmarks show that ADvLM achieves state-of-the-art attack effectiveness. Moreover, real-world attack studies further validate its applicability and potential in practice.



## **3. G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks**

cs.MA

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2410.11782v2) [paper-pdf](http://arxiv.org/pdf/2410.11782v2)

**Authors**: Guibin Zhang, Yanwei Yue, Xiangguo Sun, Guancheng Wan, Miao Yu, Junfeng Fang, Kun Wang, Dawei Cheng

**Abstract**: Recent advancements in large language model (LLM)-based agents have demonstrated that collective intelligence can significantly surpass the capabilities of individual agents, primarily due to well-crafted inter-agent communication topologies. Despite the diverse and high-performing designs available, practitioners often face confusion when selecting the most effective pipeline for their specific task: \textit{Which topology is the best choice for my task, avoiding unnecessary communication token overhead while ensuring high-quality solution?} In response to this dilemma, we introduce G-Designer, an adaptive, efficient, and robust solution for multi-agent deployment, which dynamically designs task-aware, customized communication topologies. Specifically, G-Designer models the multi-agent system as a multi-agent network, leveraging a variational graph auto-encoder to encode both the nodes (agents) and a task-specific virtual node, and decodes a task-adaptive and high-performing communication topology. Extensive experiments on six benchmarks showcase that G-Designer is: \textbf{(1) high-performing}, achieving superior results on MMLU with accuracy at $84.50\%$ and on HumanEval with pass@1 at $89.90\%$; \textbf{(2) task-adaptive}, architecting communication protocols tailored to task difficulty, reducing token consumption by up to $95.33\%$ on HumanEval; and \textbf{(3) adversarially robust}, defending against agent adversarial attacks with merely $0.3\%$ accuracy drop.



## **4. Transferable Ensemble Black-box Jailbreak Attacks on Large Language Models**

cs.CR

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2410.23558v2) [paper-pdf](http://arxiv.org/pdf/2410.23558v2)

**Authors**: Yiqi Yang, Hongye Fu

**Abstract**: In this report, we propose a novel black-box jailbreak attacking framework that incorporates various LLM-as-Attacker methods to deliver transferable and powerful jailbreak attacks. Our method is designed based on three key observations from existing jailbreaking studies and practices. First, we consider an ensemble approach should be more effective in exposing the vulnerabilities of an aligned LLM compared to individual attacks. Second, different malicious instructions inherently vary in their jailbreaking difficulty, necessitating differentiated treatment to ensure more efficient attacks. Finally, the semantic coherence of a malicious instruction is crucial for triggering the defenses of an aligned LLM; therefore, it must be carefully disrupted to manipulate its embedding representation, thereby increasing the jailbreak success rate. We validated our approach by participating in the Competition for LLM and Agent Safety 2024, where our team achieved top performance in the Jailbreaking Attack Track.



## **5. R-MTLLMF: Resilient Multi-Task Large Language Model Fusion at the Wireless Edge**

eess.SP

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2411.18220v1) [paper-pdf](http://arxiv.org/pdf/2411.18220v1)

**Authors**: Aladin Djuhera, Vlad C. Andrei, Mohsen Pourghasemian, Haris Gacanin, Holger Boche, Walid Saad

**Abstract**: Multi-task large language models (MTLLMs) are important for many applications at the wireless edge, where users demand specialized models to handle multiple tasks efficiently. However, training MTLLMs is complex and exhaustive, particularly when tasks are subject to change. Recently, the concept of model fusion via task vectors has emerged as an efficient approach for combining fine-tuning parameters to produce an MTLLM. In this paper, the problem of enabling edge users to collaboratively craft such MTTLMs via tasks vectors is studied, under the assumption of worst-case adversarial attacks. To this end, first the influence of adversarial noise to multi-task model fusion is investigated and a relationship between the so-called weight disentanglement error and the mean squared error (MSE) is derived. Using hypothesis testing, it is directly shown that the MSE increases interference between task vectors, thereby rendering model fusion ineffective. Then, a novel resilient MTLLM fusion (R-MTLLMF) is proposed, which leverages insights about the LLM architecture and fine-tuning process to safeguard task vector aggregation under adversarial noise by realigning the MTLLM. The proposed R-MTLLMF is then compared for both worst-case and ideal transmission scenarios to study the impact of the wireless channel. Extensive model fusion experiments with vision LLMs demonstrate R-MTLLMF's effectiveness, achieving close-to-baseline performance across eight different tasks in ideal noise scenarios and significantly outperforming unprotected model fusion in worst-case scenarios. The results further advocate for additional physical layer protection for a holistic approach to resilience, from both a wireless and LLM perspective.



## **6. Evaluating and Improving the Robustness of Security Attack Detectors Generated by LLMs**

cs.SE

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2411.18216v1) [paper-pdf](http://arxiv.org/pdf/2411.18216v1)

**Authors**: Samuele Pasini, Jinhan Kim, Tommaso Aiello, Rocio Cabrera Lozoya, Antonino Sabetta, Paolo Tonella

**Abstract**: Large Language Models (LLMs) are increasingly used in software development to generate functions, such as attack detectors, that implement security requirements. However, LLMs struggle to generate accurate code, resulting, e.g., in attack detectors that miss well-known attacks when used in practice. This is most likely due to the LLM lacking knowledge about some existing attacks and to the generated code being not evaluated in real usage scenarios. We propose a novel approach integrating Retrieval Augmented Generation (RAG) and Self-Ranking into the LLM pipeline. RAG enhances the robustness of the output by incorporating external knowledge sources, while the Self-Ranking technique, inspired to the concept of Self-Consistency, generates multiple reasoning paths and creates ranks to select the most robust detector. Our extensive empirical study targets code generated by LLMs to detect two prevalent injection attacks in web security: Cross-Site Scripting (XSS) and SQL injection (SQLi). Results show a significant improvement in detection performance compared to baselines, with an increase of up to 71%pt and 37%pt in the F2-Score for XSS and SQLi detection, respectively.



## **7. InputSnatch: Stealing Input in LLM Services via Timing Side-Channel Attacks**

cs.CR

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2411.18191v1) [paper-pdf](http://arxiv.org/pdf/2411.18191v1)

**Authors**: Xinyao Zheng, Husheng Han, Shangyi Shi, Qiyan Fang, Zidong Du, Qi Guo, Xing Hu

**Abstract**: Large language models (LLMs) possess extensive knowledge and question-answering capabilities, having been widely deployed in privacy-sensitive domains like finance and medical consultation. During LLM inferences, cache-sharing methods are commonly employed to enhance efficiency by reusing cached states or responses for the same or similar inference requests. However, we identify that these cache mechanisms pose a risk of private input leakage, as the caching can result in observable variations in response times, making them a strong candidate for a timing-based attack hint. In this study, we propose a novel timing-based side-channel attack to execute input theft in LLMs inference. The cache-based attack faces the challenge of constructing candidate inputs in a large search space to hit and steal cached user queries. To address these challenges, we propose two primary components. The input constructor employs machine learning techniques and LLM-based approaches for vocabulary correlation learning while implementing optimized search mechanisms for generalized input construction. The time analyzer implements statistical time fitting with outlier elimination to identify cache hit patterns, continuously providing feedback to refine the constructor's search strategy. We conduct experiments across two cache mechanisms and the results demonstrate that our approach consistently attains high attack success rates in various applications. Our work highlights the security vulnerabilities associated with performance optimizations, underscoring the necessity of prioritizing privacy and security alongside enhancements in LLM inference.



## **8. Playing Language Game with LLMs Leads to Jailbreaking**

cs.CL

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2411.12762v2) [paper-pdf](http://arxiv.org/pdf/2411.12762v2)

**Authors**: Yu Peng, Zewen Long, Fangming Dong, Congyi Li, Shu Wu, Kai Chen

**Abstract**: The advent of large language models (LLMs) has spurred the development of numerous jailbreak techniques aimed at circumventing their security defenses against malicious attacks. An effective jailbreak approach is to identify a domain where safety generalization fails, a phenomenon known as mismatched generalization. In this paper, we introduce two novel jailbreak methods based on mismatched generalization: natural language games and custom language games, both of which effectively bypass the safety mechanisms of LLMs, with various kinds and different variants, making them hard to defend and leading to high attack rates. Natural language games involve the use of synthetic linguistic constructs and the actions intertwined with these constructs, such as the Ubbi Dubbi language. Building on this phenomenon, we propose the custom language games method: by engaging with LLMs using a variety of custom rules, we successfully execute jailbreak attacks across multiple LLM platforms. Extensive experiments demonstrate the effectiveness of our methods, achieving success rates of 93% on GPT-4o, 89% on GPT-4o-mini and 83% on Claude-3.5-Sonnet. Furthermore, to investigate the generalizability of safety alignments, we fine-tuned Llama-3.1-70B with the custom language games to achieve safety alignment within our datasets and found that when interacting through other language games, the fine-tuned models still failed to identify harmful content. This finding indicates that the safety alignment knowledge embedded in LLMs fails to generalize across different linguistic formats, thus opening new avenues for future research in this area.



## **9. BlackDAN: A Black-Box Multi-Objective Approach for Effective and Contextual Jailbreaking of Large Language Models**

cs.CR

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2410.09804v3) [paper-pdf](http://arxiv.org/pdf/2410.09804v3)

**Authors**: Xinyuan Wang, Victor Shea-Jay Huang, Renmiao Chen, Hao Wang, Chengwei Pan, Lei Sha, Minlie Huang

**Abstract**: While large language models (LLMs) exhibit remarkable capabilities across various tasks, they encounter potential security risks such as jailbreak attacks, which exploit vulnerabilities to bypass security measures and generate harmful outputs. Existing jailbreak strategies mainly focus on maximizing attack success rate (ASR), frequently neglecting other critical factors, including the relevance of the jailbreak response to the query and the level of stealthiness. This narrow focus on single objectives can result in ineffective attacks that either lack contextual relevance or are easily recognizable. In this work, we introduce BlackDAN, an innovative black-box attack framework with multi-objective optimization, aiming to generate high-quality prompts that effectively facilitate jailbreaking while maintaining contextual relevance and minimizing detectability. BlackDAN leverages Multiobjective Evolutionary Algorithms (MOEAs), specifically the NSGA-II algorithm, to optimize jailbreaks across multiple objectives including ASR, stealthiness, and semantic relevance. By integrating mechanisms like mutation, crossover, and Pareto-dominance, BlackDAN provides a transparent and interpretable process for generating jailbreaks. Furthermore, the framework allows customization based on user preferences, enabling the selection of prompts that balance harmfulness, relevance, and other factors. Experimental results demonstrate that BlackDAN outperforms traditional single-objective methods, yielding higher success rates and improved robustness across various LLMs and multimodal LLMs, while ensuring jailbreak responses are both relevant and less detectable.



## **10. Desert Camels and Oil Sheikhs: Arab-Centric Red Teaming of Frontier LLMs**

cs.CL

**SubmitDate**: 2024-11-26    [abs](http://arxiv.org/abs/2410.24049v3) [paper-pdf](http://arxiv.org/pdf/2410.24049v3)

**Authors**: Muhammed Saeed, Elgizouli Mohamed, Mukhtar Mohamed, Shaina Raza, Muhammad Abdul-Mageed, Shady Shehata

**Abstract**: Large language models (LLMs) are widely used but raise ethical concerns due to embedded social biases. This study examines LLM biases against Arabs versus Westerners across eight domains, including women's rights, terrorism, and anti-Semitism and assesses model resistance to perpetuating these biases. To this end, we create two datasets: one to evaluate LLM bias toward Arabs versus Westerners and another to test model safety against prompts that exaggerate negative traits ("jailbreaks"). We evaluate six LLMs -- GPT-4, GPT-4o, LlaMA 3.1 (8B & 405B), Mistral 7B, and Claude 3.5 Sonnet. We find 79% of cases displaying negative biases toward Arabs, with LlaMA 3.1-405B being the most biased. Our jailbreak tests reveal GPT-4o as the most vulnerable, despite being an optimized version, followed by LlaMA 3.1-8B and Mistral 7B. All LLMs except Claude exhibit attack success rates above 87% in three categories. We also find Claude 3.5 Sonnet the safest, but it still displays biases in seven of eight categories. Despite being an optimized version of GPT4, We find GPT-4o to be more prone to biases and jailbreaks, suggesting optimization flaws. Our findings underscore the pressing need for more robust bias mitigation strategies and strengthened security measures in LLMs.



## **11. RTL-Breaker: Assessing the Security of LLMs against Backdoor Attacks on HDL Code Generation**

cs.CR

Accepted at 2025 Design, Automation & Test in Europe (DATE)  Conference

**SubmitDate**: 2024-11-26    [abs](http://arxiv.org/abs/2411.17569v1) [paper-pdf](http://arxiv.org/pdf/2411.17569v1)

**Authors**: Lakshmi Likhitha Mankali, Jitendra Bhandari, Manaar Alam, Ramesh Karri, Michail Maniatakos, Ozgur Sinanoglu, Johann Knechtel

**Abstract**: Large language models (LLMs) have demonstrated remarkable potential with code generation/completion tasks for hardware design. In fact, LLM-based hardware description language (HDL) code generation has enabled the industry to realize complex designs more quickly, reducing the time and effort required in the development cycle. However, the increased reliance on such automation introduces critical security risks. Notably, given that LLMs have to be trained on vast datasets of codes that are typically sourced from publicly available repositories (often without thorough validation), LLMs are susceptible to so-called data poisoning or backdoor attacks. Here, attackers inject malicious code for the training data, which can be carried over into the HDL code generated by LLMs. This threat vector can compromise the security and integrity of entire hardware systems. In this work, we propose RTL-Breaker, a novel backdoor attack framework on LLM-based HDL code generation. RTL-Breaker provides an in-depth analysis for essential aspects of this novel problem: 1) various trigger mechanisms versus their effectiveness for inserting malicious modifications, and 2) side-effects by backdoor attacks on code generation in general, i.e., impact on code quality. RTL-Breaker emphasizes the urgent need for more robust measures to safeguard against such attacks. Toward that end, we open-source our framework and all data.



## **12. PEFTGuard: Detecting Backdoor Attacks Against Parameter-Efficient Fine-Tuning**

cs.CR

20 pages, 8 figures

**SubmitDate**: 2024-11-26    [abs](http://arxiv.org/abs/2411.17453v1) [paper-pdf](http://arxiv.org/pdf/2411.17453v1)

**Authors**: Zhen Sun, Tianshuo Cong, Yule Liu, Chenhao Lin, Xinlei He, Rongmao Chen, Xingshuo Han, Xinyi Huang

**Abstract**: Fine-tuning is an essential process to improve the performance of Large Language Models (LLMs) in specific domains, with Parameter-Efficient Fine-Tuning (PEFT) gaining popularity due to its capacity to reduce computational demands through the integration of low-rank adapters. These lightweight adapters, such as LoRA, can be shared and utilized on open-source platforms. However, adversaries could exploit this mechanism to inject backdoors into these adapters, resulting in malicious behaviors like incorrect or harmful outputs, which pose serious security risks to the community. Unfortunately, few of the current efforts concentrate on analyzing the backdoor patterns or detecting the backdoors in the adapters.   To fill this gap, we first construct (and will release) PADBench, a comprehensive benchmark that contains 13,300 benign and backdoored adapters fine-tuned with various datasets, attack strategies, PEFT methods, and LLMs. Moreover, we propose PEFTGuard, the first backdoor detection framework against PEFT-based adapters. Extensive evaluation upon PADBench shows that PEFTGuard outperforms existing detection methods, achieving nearly perfect detection accuracy (100%) in most cases. Notably, PEFTGuard exhibits zero-shot transferability on three aspects, including different attacks, PEFT methods, and adapter ranks. In addition, we consider various adaptive attacks to demonstrate the high robustness of PEFTGuard. We further explore several possible backdoor mitigation defenses, finding fine-mixing to be the most effective method. We envision our benchmark and method can shed light on future LLM backdoor detection research.



## **13. Practical Membership Inference Attacks against Fine-tuned Large Language Models via Self-prompt Calibration**

cs.CL

Repo: https://github.com/tsinghua-fib-lab/NeurIPS2024_SPV-MIA

**SubmitDate**: 2024-11-26    [abs](http://arxiv.org/abs/2311.06062v4) [paper-pdf](http://arxiv.org/pdf/2311.06062v4)

**Authors**: Wenjie Fu, Huandong Wang, Chen Gao, Guanghua Liu, Yong Li, Tao Jiang

**Abstract**: Membership Inference Attacks (MIA) aim to infer whether a target data record has been utilized for model training or not. Existing MIAs designed for large language models (LLMs) can be bifurcated into two types: reference-free and reference-based attacks. Although reference-based attacks appear promising performance by calibrating the probability measured on the target model with reference models, this illusion of privacy risk heavily depends on a reference dataset that closely resembles the training set. Both two types of attacks are predicated on the hypothesis that training records consistently maintain a higher probability of being sampled. However, this hypothesis heavily relies on the overfitting of target models, which will be mitigated by multiple regularization methods and the generalization of LLMs. Thus, these reasons lead to high false-positive rates of MIAs in practical scenarios. We propose a Membership Inference Attack based on Self-calibrated Probabilistic Variation (SPV-MIA). Specifically, we introduce a self-prompt approach, which constructs the dataset to fine-tune the reference model by prompting the target LLM itself. In this manner, the adversary can collect a dataset with a similar distribution from public APIs. Furthermore, we introduce probabilistic variation, a more reliable membership signal based on LLM memorization rather than overfitting, from which we rediscover the neighbour attack with theoretical grounding. Comprehensive evaluation conducted on three datasets and four exemplary LLMs shows that SPV-MIA raises the AUC of MIAs from 0.7 to a significantly high level of 0.9. Our code and dataset are available at: https://github.com/tsinghua-fib-lab/NeurIPS2024_SPV-MIA



## **14. FATH: Authentication-based Test-time Defense against Indirect Prompt Injection Attacks**

cs.CR

**SubmitDate**: 2024-11-25    [abs](http://arxiv.org/abs/2410.21492v2) [paper-pdf](http://arxiv.org/pdf/2410.21492v2)

**Authors**: Jiongxiao Wang, Fangzhou Wu, Wendi Li, Jinsheng Pan, Edward Suh, Z. Morley Mao, Muhao Chen, Chaowei Xiao

**Abstract**: Large language models (LLMs) have been widely deployed as the backbone with additional tools and text information for real-world applications. However, integrating external information into LLM-integrated applications raises significant security concerns. Among these, prompt injection attacks are particularly threatening, where malicious instructions injected in the external text information can exploit LLMs to generate answers as the attackers desire. While both training-time and test-time defense methods have been developed to mitigate such attacks, the unaffordable training costs associated with training-time methods and the limited effectiveness of existing test-time methods make them impractical. This paper introduces a novel test-time defense strategy, named Formatting AuThentication with Hash-based tags (FATH). Unlike existing approaches that prevent LLMs from answering additional instructions in external text, our method implements an authentication system, requiring LLMs to answer all received instructions with a security policy and selectively filter out responses to user instructions as the final output. To achieve this, we utilize hash-based authentication tags to label each response, facilitating accurate identification of responses according to the user's instructions and improving the robustness against adaptive attacks. Comprehensive experiments demonstrate that our defense method can effectively defend against indirect prompt injection attacks, achieving state-of-the-art performance under Llama3 and GPT3.5 models across various attack methods. Our code is released at: https://github.com/Jayfeather1024/FATH



## **15. Just-in-Time Detection of Silent Security Patches**

cs.CR

**SubmitDate**: 2024-11-25    [abs](http://arxiv.org/abs/2312.01241v3) [paper-pdf](http://arxiv.org/pdf/2312.01241v3)

**Authors**: Xunzhu Tang, Zhenghan Chen, Kisub Kim, Haoye Tian, Saad Ezzini, Jacques Klein

**Abstract**: Open-source code is pervasive. In this setting, embedded vulnerabilities are spreading to downstream software at an alarming rate. While such vulnerabilities are generally identified and addressed rapidly, inconsistent maintenance policies may lead security patches to go unnoticed. Indeed, security patches can be {\em silent}, i.e., they do not always come with comprehensive advisories such as CVEs. This lack of transparency leaves users oblivious to available security updates, providing ample opportunity for attackers to exploit unpatched vulnerabilities. Consequently, identifying silent security patches just in time when they are released is essential for preventing n-day attacks, and for ensuring robust and secure maintenance practices. With LLMDA we propose to (1) leverage large language models (LLMs) to augment patch information with generated code change explanations, (2) design a representation learning approach that explores code-text alignment methodologies for feature combination, (3) implement a label-wise training with labelled instructions for guiding the embedding based on security relevance, and (4) rely on a probabilistic batch contrastive learning mechanism for building a high-precision identifier of security patches. We evaluate LLMDA on the PatchDB and SPI-DB literature datasets and show that our approach substantially improves over the state-of-the-art, notably GraphSPD by 20% in terms of F-Measure on the SPI-DB benchmark.



## **16. Scaling Laws for Black box Adversarial Attacks**

cs.LG

**SubmitDate**: 2024-11-25    [abs](http://arxiv.org/abs/2411.16782v1) [paper-pdf](http://arxiv.org/pdf/2411.16782v1)

**Authors**: Chuan Liu, Huanran Chen, Yichi Zhang, Yinpeng Dong, Jun Zhu

**Abstract**: A longstanding problem of deep learning models is their vulnerability to adversarial examples, which are often generated by applying imperceptible perturbations to natural examples. Adversarial examples exhibit cross-model transferability, enabling to attack black-box models with limited information about their architectures and parameters. Model ensembling is an effective strategy to improve the transferability by attacking multiple surrogate models simultaneously. However, as prior studies usually adopt few models in the ensemble, there remains an open question of whether scaling the number of models can further improve black-box attacks. Inspired by the findings in large foundation models, we investigate the scaling laws of black-box adversarial attacks in this work. By analyzing the relationship between the number of surrogate models and transferability of adversarial examples, we conclude with clear scaling laws, emphasizing the potential of using more surrogate models to enhance adversarial transferability. Extensive experiments verify the claims on standard image classifiers, multimodal large language models, and even proprietary models like GPT-4o, demonstrating consistent scaling effects and impressive attack success rates with more surrogate models. Further studies by visualization indicate that scaled attacks bring better interpretability in semantics, indicating that the common features of models are captured.



## **17. LLMPirate: LLMs for Black-box Hardware IP Piracy**

cs.CR

Accepted by NDSS Symposium 2025

**SubmitDate**: 2024-11-25    [abs](http://arxiv.org/abs/2411.16111v1) [paper-pdf](http://arxiv.org/pdf/2411.16111v1)

**Authors**: Vasudev Gohil, Matthew DeLorenzo, Veera Vishwa Achuta Sai Venkat Nallam, Joey See, Jeyavijayan Rajendran

**Abstract**: The rapid advancement of large language models (LLMs) has enabled the ability to effectively analyze and generate code nearly instantaneously, resulting in their widespread adoption in software development. Following this advancement, researchers and companies have begun integrating LLMs across the hardware design and verification process. However, these highly potent LLMs can also induce new attack scenarios upon security vulnerabilities across the hardware development process. One such attack vector that has not been explored is intellectual property (IP) piracy. Given that this attack can manifest as rewriting hardware designs to evade piracy detection, it is essential to thoroughly evaluate LLM capabilities in performing this task and assess the mitigation abilities of current IP piracy detection tools.   Therefore, in this work, we propose LLMPirate, the first LLM-based technique able to generate pirated variations of circuit designs that successfully evade detection across multiple state-of-the-art piracy detection tools. We devise three solutions to overcome challenges related to integration of LLMs for hardware circuit designs, scalability to large circuits, and effectiveness, resulting in an end-to-end automated, efficient, and practical formulation. We perform an extensive experimental evaluation of LLMPirate using eight LLMs of varying sizes and capabilities and assess their performance in pirating various circuit designs against four state-of-the-art, widely-used piracy detection tools. Our experiments demonstrate that LLMPirate is able to consistently evade detection on 100% of tested circuits across every detection tool. Additionally, we showcase the ramifications of LLMPirate using case studies on IBEX and MOR1KX processors and a GPS module, that we successfully pirate. We envision that our work motivates and fosters the development of better IP piracy detection tools.



## **18. In-Context Experience Replay Facilitates Safety Red-Teaming of Text-to-Image Diffusion Models**

cs.LG

**SubmitDate**: 2024-11-25    [abs](http://arxiv.org/abs/2411.16769v1) [paper-pdf](http://arxiv.org/pdf/2411.16769v1)

**Authors**: Zhi-Yi Chin, Kuan-Chen Mu, Mario Fritz, Pin-Yu Chen, Wei-Chen Chiu

**Abstract**: Text-to-image (T2I) models have shown remarkable progress, but their potential to generate harmful content remains a critical concern in the ML community. While various safety mechanisms have been developed, the field lacks systematic tools for evaluating their effectiveness against real-world misuse scenarios. In this work, we propose ICER, a novel red-teaming framework that leverages Large Language Models (LLMs) and a bandit optimization-based algorithm to generate interpretable and semantic meaningful problematic prompts by learning from past successful red-teaming attempts. Our ICER efficiently probes safety mechanisms across different T2I models without requiring internal access or additional training, making it broadly applicable to deployed systems. Through extensive experiments, we demonstrate that ICER significantly outperforms existing prompt attack methods in identifying model vulnerabilities while maintaining high semantic similarity with intended content. By uncovering that successful jailbreaking instances can systematically facilitate the discovery of new vulnerabilities, our work provides crucial insights for developing more robust safety mechanisms in T2I systems.



## **19. Vaccine: Perturbation-aware Alignment for Large Language Models against Harmful Fine-tuning Attack**

cs.LG

Rejected by ICML2024. Accepted by NeurIPS2024

**SubmitDate**: 2024-11-24    [abs](http://arxiv.org/abs/2402.01109v6) [paper-pdf](http://arxiv.org/pdf/2402.01109v6)

**Authors**: Tiansheng Huang, Sihao Hu, Ling Liu

**Abstract**: The new paradigm of finetuning-as-a-service introduces a new attack surface for Large Language Models (LLMs): a few harmful data uploaded by users can easily trick the finetuning to produce an alignment-broken model. We conduct an empirical analysis and uncover a \textit{harmful embedding drift} phenomenon, showing a probable cause of the alignment-broken effect. Inspired by our findings, we propose Vaccine, a perturbation-aware alignment technique to mitigate the security risk of users finetuning. The core idea of Vaccine is to produce invariant hidden embeddings by progressively adding crafted perturbation to them in the alignment phase. This enables the embeddings to withstand harmful perturbation from un-sanitized user data in the finetuning phase. Our results on open source mainstream LLMs (e.g., Llama2, Opt, Vicuna) demonstrate that Vaccine can boost the robustness of alignment against harmful prompts induced embedding drift while reserving reasoning ability towards benign prompts. Our code is available at \url{https://github.com/git-disl/Vaccine}.



## **20. AmpleGCG: Learning a Universal and Transferable Generative Model of Adversarial Suffixes for Jailbreaking Both Open and Closed LLMs**

cs.CL

Published as a conference paper at COLM 2024  (https://colmweb.org/index.html)

**SubmitDate**: 2024-11-24    [abs](http://arxiv.org/abs/2404.07921v3) [paper-pdf](http://arxiv.org/pdf/2404.07921v3)

**Authors**: Zeyi Liao, Huan Sun

**Abstract**: As large language models (LLMs) become increasingly prevalent and integrated into autonomous systems, ensuring their safety is imperative. Despite significant strides toward safety alignment, recent work GCG~\citep{zou2023universal} proposes a discrete token optimization algorithm and selects the single suffix with the lowest loss to successfully jailbreak aligned LLMs. In this work, we first discuss the drawbacks of solely picking the suffix with the lowest loss during GCG optimization for jailbreaking and uncover the missed successful suffixes during the intermediate steps. Moreover, we utilize those successful suffixes as training data to learn a generative model, named AmpleGCG, which captures the distribution of adversarial suffixes given a harmful query and enables the rapid generation of hundreds of suffixes for any harmful queries in seconds. AmpleGCG achieves near 100\% attack success rate (ASR) on two aligned LLMs (Llama-2-7B-chat and Vicuna-7B), surpassing two strongest attack baselines. More interestingly, AmpleGCG also transfers seamlessly to attack different models, including closed-source LLMs, achieving a 99\% ASR on the latest GPT-3.5. To summarize, our work amplifies the impact of GCG by training a generative model of adversarial suffixes that is universal to any harmful queries and transferable from attacking open-source LLMs to closed-source LLMs. In addition, it can generate 200 adversarial suffixes for one harmful query in only 4 seconds, rendering it more challenging to defend.



## **21. JailBreakV: A Benchmark for Assessing the Robustness of MultiModal Large Language Models against Jailbreak Attacks**

cs.CR

**SubmitDate**: 2024-11-24    [abs](http://arxiv.org/abs/2404.03027v4) [paper-pdf](http://arxiv.org/pdf/2404.03027v4)

**Authors**: Weidi Luo, Siyuan Ma, Xiaogeng Liu, Xiaoyu Guo, Chaowei Xiao

**Abstract**: With the rapid advancements in Multimodal Large Language Models (MLLMs), securing these models against malicious inputs while aligning them with human values has emerged as a critical challenge. In this paper, we investigate an important and unexplored question of whether techniques that successfully jailbreak Large Language Models (LLMs) can be equally effective in jailbreaking MLLMs. To explore this issue, we introduce JailBreakV-28K, a pioneering benchmark designed to assess the transferability of LLM jailbreak techniques to MLLMs, thereby evaluating the robustness of MLLMs against diverse jailbreak attacks. Utilizing a dataset of 2, 000 malicious queries that is also proposed in this paper, we generate 20, 000 text-based jailbreak prompts using advanced jailbreak attacks on LLMs, alongside 8, 000 image-based jailbreak inputs from recent MLLMs jailbreak attacks, our comprehensive dataset includes 28, 000 test cases across a spectrum of adversarial scenarios. Our evaluation of 10 open-source MLLMs reveals a notably high Attack Success Rate (ASR) for attacks transferred from LLMs, highlighting a critical vulnerability in MLLMs that stems from their text-processing capabilities. Our findings underscore the urgent need for future research to address alignment vulnerabilities in MLLMs from both textual and visual inputs.



## **22. InjecGuard: Benchmarking and Mitigating Over-defense in Prompt Injection Guardrail Models**

cs.CL

**SubmitDate**: 2024-11-24    [abs](http://arxiv.org/abs/2410.22770v2) [paper-pdf](http://arxiv.org/pdf/2410.22770v2)

**Authors**: Hao Li, Xiaogeng Liu

**Abstract**: Prompt injection attacks pose a critical threat to large language models (LLMs), enabling goal hijacking and data leakage. Prompt guard models, though effective in defense, suffer from over-defense -- falsely flagging benign inputs as malicious due to trigger word bias. To address this issue, we introduce NotInject, an evaluation dataset that systematically measures over-defense across various prompt guard models. NotInject contains 339 benign samples enriched with trigger words common in prompt injection attacks, enabling fine-grained evaluation. Our results show that state-of-the-art models suffer from over-defense issues, with accuracy dropping close to random guessing levels (60%). To mitigate this, we propose InjecGuard, a novel prompt guard model that incorporates a new training strategy, Mitigating Over-defense for Free (MOF), which significantly reduces the bias on trigger words. InjecGuard demonstrates state-of-the-art performance on diverse benchmarks including NotInject, surpassing the existing best model by 30.8%, offering a robust and open-source solution for detecting prompt injection attacks. The code and datasets are released at https://github.com/SaFoLab-WISC/InjecGuard.



## **23. Semantic Shield: Defending Vision-Language Models Against Backdooring and Poisoning via Fine-grained Knowledge Alignment**

cs.CV

CVPR 2024

**SubmitDate**: 2024-11-23    [abs](http://arxiv.org/abs/2411.15673v1) [paper-pdf](http://arxiv.org/pdf/2411.15673v1)

**Authors**: Alvi Md Ishmam, Christopher Thomas

**Abstract**: In recent years there has been enormous interest in vision-language models trained using self-supervised objectives. However, the use of large-scale datasets scraped from the web for training also makes these models vulnerable to potential security threats, such as backdooring and poisoning attacks. In this paper, we propose a method for mitigating such attacks on contrastively trained vision-language models. Our approach leverages external knowledge extracted from a language model to prevent models from learning correlations between image regions which lack strong alignment with external knowledge. We do this by imposing constraints to enforce that attention paid by the model to visual regions is proportional to the alignment of those regions with external knowledge. We conduct extensive experiments using a variety of recent backdooring and poisoning attacks on multiple datasets and architectures. Our results clearly demonstrate that our proposed approach is highly effective at defending against such attacks across multiple settings, while maintaining model utility and without requiring any changes at inference time



## **24. "Moralized" Multi-Step Jailbreak Prompts: Black-Box Testing of Guardrails in Large Language Models for Verbal Attacks**

cs.CR

This paper has been submitted to Nature Machine Intelligence and  OpenReview preprints. It has 9 pages of text, 3 figures, and 3 tables

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2411.16730v2) [paper-pdf](http://arxiv.org/pdf/2411.16730v2)

**Authors**: Libo Wang

**Abstract**: As the application of large language models continues to expand in various fields, it poses higher challenges to the effectiveness of identifying harmful content generation and guardrail mechanisms. This research aims to evaluate the guardrail effectiveness of GPT-4o, Grok-2 Beta, Llama 3.1 (405B), Gemini 1.5, and Claude 3.5 Sonnet through black-box testing of seemingly ethical multi-step jailbreak prompts. It conducts ethical attacks by designing an identical multi-step prompts that simulates the scenario of "corporate middle managers competing for promotions." The data results show that the guardrails of the above-mentioned LLMs were bypassed and the content of verbal attacks was generated. Claude 3.5 Sonnet's resistance to multi-step jailbreak prompts is more obvious. To ensure objectivity, the experimental process, black box test code, and enhanced guardrail code are uploaded to the GitHub repository: https://github.com/brucewang123456789/GeniusTrail.git.



## **25. Geminio: Language-Guided Gradient Inversion Attacks in Federated Learning**

cs.LG

**SubmitDate**: 2024-11-22    [abs](http://arxiv.org/abs/2411.14937v1) [paper-pdf](http://arxiv.org/pdf/2411.14937v1)

**Authors**: Junjie Shan, Ziqi Zhao, Jialin Lu, Rui Zhang, Siu Ming Yiu, Ka-Ho Chow

**Abstract**: Foundation models that bridge vision and language have made significant progress, inspiring numerous life-enriching applications. However, their potential for misuse to introduce new threats remains largely unexplored. This paper reveals that vision-language models (VLMs) can be exploited to overcome longstanding limitations in gradient inversion attacks (GIAs) within federated learning (FL), where an FL server reconstructs private data samples from gradients shared by victim clients. Current GIAs face challenges in reconstructing high-resolution images, especially when the victim has a large local data batch. While focusing reconstruction on valuable samples rather than the entire batch is promising, existing methods lack the flexibility to allow attackers to specify their target data. In this paper, we introduce Geminio, the first approach to transform GIAs into semantically meaningful, targeted attacks. Geminio enables a brand new privacy attack experience: attackers can describe, in natural language, the types of data they consider valuable, and Geminio will prioritize reconstruction to focus on those high-value samples. This is achieved by leveraging a pretrained VLM to guide the optimization of a malicious global model that, when shared with and optimized by a victim, retains only gradients of samples that match the attacker-specified query. Extensive experiments demonstrate Geminio's effectiveness in pinpointing and reconstructing targeted samples, with high success rates across complex datasets under FL and large batch sizes and showing resilience against existing defenses.



## **26. Who Can Withstand Chat-Audio Attacks? An Evaluation Benchmark for Large Language Models**

cs.SD

**SubmitDate**: 2024-11-22    [abs](http://arxiv.org/abs/2411.14842v1) [paper-pdf](http://arxiv.org/pdf/2411.14842v1)

**Authors**: Wanqi Yang, Yanda Li, Meng Fang, Yunchao Wei, Tianyi Zhou, Ling Chen

**Abstract**: Adversarial audio attacks pose a significant threat to the growing use of large language models (LLMs) in voice-based human-machine interactions. While existing research has primarily focused on model-specific adversarial methods, real-world applications demand a more generalizable and universal approach to audio adversarial attacks. In this paper, we introduce the Chat-Audio Attacks (CAA) benchmark including four distinct types of audio attacks, which aims to explore the the vulnerabilities of LLMs to these audio attacks in conversational scenarios. To evaluate the robustness of LLMs, we propose three evaluation strategies: Standard Evaluation, utilizing traditional metrics to quantify model performance under attacks; GPT-4o-Based Evaluation, which simulates real-world conversational complexities; and Human Evaluation, offering insights into user perception and trust. We evaluate six state-of-the-art LLMs with voice interaction capabilities, including Gemini-1.5-Pro, GPT-4o, and others, using three distinct evaluation methods on the CAA benchmark. Our comprehensive analysis reveals the impact of four types of audio attacks on the performance of these models, demonstrating that GPT-4o exhibits the highest level of resilience.



## **27. Universal and Context-Independent Triggers for Precise Control of LLM Outputs**

cs.CL

**SubmitDate**: 2024-11-22    [abs](http://arxiv.org/abs/2411.14738v1) [paper-pdf](http://arxiv.org/pdf/2411.14738v1)

**Authors**: Jiashuo Liang, Guancheng Li, Yang Yu

**Abstract**: Large language models (LLMs) have been widely adopted in applications such as automated content generation and even critical decision-making systems. However, the risk of prompt injection allows for potential manipulation of LLM outputs. While numerous attack methods have been documented, achieving full control over these outputs remains challenging, often requiring experienced attackers to make multiple attempts and depending heavily on the prompt context. Recent advancements in gradient-based white-box attack techniques have shown promise in tasks like jailbreaks and system prompt leaks. Our research generalizes gradient-based attacks to find a trigger that is (1) Universal: effective irrespective of the target output; (2) Context-Independent: robust across diverse prompt contexts; and (3) Precise Output: capable of manipulating LLM inputs to yield any specified output with high accuracy. We propose a novel method to efficiently discover such triggers and assess the effectiveness of the proposed attack. Furthermore, we discuss the substantial threats posed by such attacks to LLM-based applications, highlighting the potential for adversaries to taking over the decisions and actions made by AI agents.



## **28. Exploring the Adversarial Vulnerabilities of Vision-Language-Action Models in Robotics**

cs.RO

**SubmitDate**: 2024-11-22    [abs](http://arxiv.org/abs/2411.13587v2) [paper-pdf](http://arxiv.org/pdf/2411.13587v2)

**Authors**: Taowen Wang, Dongfang Liu, James Chenhao Liang, Wenhao Yang, Qifan Wang, Cheng Han, Jiebo Luo, Ruixiang Tang

**Abstract**: Recently in robotics, Vision-Language-Action (VLA) models have emerged as a transformative approach, enabling robots to execute complex tasks by integrating visual and linguistic inputs within an end-to-end learning framework. While VLA models offer significant capabilities, they also introduce new attack surfaces, making them vulnerable to adversarial attacks. With these vulnerabilities largely unexplored, this paper systematically quantifies the robustness of VLA-based robotic systems. Recognizing the unique demands of robotic execution, our attack objectives target the inherent spatial and functional characteristics of robotic systems. In particular, we introduce an untargeted position-aware attack objective that leverages spatial foundations to destabilize robotic actions, and a targeted attack objective that manipulates the robotic trajectory. Additionally, we design an adversarial patch generation approach that places a small, colorful patch within the camera's view, effectively executing the attack in both digital and physical environments. Our evaluation reveals a marked degradation in task success rates, with up to a 100\% reduction across a suite of simulated robotic tasks, highlighting critical security gaps in current VLA architectures. By unveiling these vulnerabilities and proposing actionable evaluation metrics, this work advances both the understanding and enhancement of safety for VLA-based robotic systems, underscoring the necessity for developing robust defense strategies prior to physical-world deployments.



## **29. Adversarial Prompt Distillation for Vision-Language Models**

cs.CV

**SubmitDate**: 2024-11-22    [abs](http://arxiv.org/abs/2411.15244v1) [paper-pdf](http://arxiv.org/pdf/2411.15244v1)

**Authors**: Lin Luo, Xin Wang, Bojia Zi, Shihao Zhao, Xingjun Ma

**Abstract**: Large pre-trained Vision-Language Models (VLMs) such as Contrastive Language-Image Pre-Training (CLIP) have been shown to be susceptible to adversarial attacks, raising concerns about their deployment in safety-critical scenarios like autonomous driving and medical diagnosis. One promising approach for improving the robustness of pre-trained VLMs is Adversarial Prompt Tuning (APT), which combines adversarial training with prompt tuning. However, existing APT methods are mostly single-modal methods that design prompt(s) for only the visual or textual modality, limiting their effectiveness in either robustness or clean accuracy. In this work, we propose a novel method called Adversarial Prompt Distillation (APD) that combines APT with knowledge distillation to boost the adversarial robustness of CLIP. Specifically, APD is a bimodal method that adds prompts for both the visual and textual modalities while leveraging a cleanly pre-trained teacher CLIP model to distill and boost the performance of the student CLIP model on downstream tasks. Extensive experiments on multiple benchmark datasets demonstrate the superiority of our APD over the current state-of-the-art APT methods in terms of both natural and adversarial performances. The effectiveness of our APD method validates the possibility of using a non-robust teacher to improve the generalization and robustness of VLMs.



## **30. Memory Backdoor Attacks on Neural Networks**

cs.CR

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2411.14516v1) [paper-pdf](http://arxiv.org/pdf/2411.14516v1)

**Authors**: Eden Luzon, Guy Amit, Roy Weiss, Yisroel Mirsky

**Abstract**: Neural networks, such as image classifiers, are frequently trained on proprietary and confidential datasets. It is generally assumed that once deployed, the training data remains secure, as adversaries are limited to query response interactions with the model, where at best, fragments of arbitrary data can be inferred without any guarantees on their authenticity. In this paper, we propose the memory backdoor attack, where a model is covertly trained to memorize specific training samples and later selectively output them when triggered with an index pattern. What makes this attack unique is that it (1) works even when the tasks conflict (making a classifier output images), (2) enables the systematic extraction of training samples from deployed models and (3) offers guarantees on the extracted authenticity of the data. We demonstrate the attack on image classifiers, segmentation models, and a large language model (LLM). We demonstrate the attack on image classifiers, segmentation models, and a large language model (LLM). With this attack, it is possible to hide thousands of images and texts in modern vision architectures and LLMs respectively, all while maintaining model performance. The memory back door attack poses a significant threat not only to conventional model deployments but also to federated learning paradigms and other modern frameworks. Therefore, we suggest an efficient and effective countermeasure that can be immediately applied and advocate for further work on the topic.



## **31. GASP: Efficient Black-Box Generation of Adversarial Suffixes for Jailbreaking LLMs**

cs.LG

28 pages, 9 tables, 13 figures; under review at CVPR '25

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2411.14133v1) [paper-pdf](http://arxiv.org/pdf/2411.14133v1)

**Authors**: Advik Raj Basani, Xiao Zhang

**Abstract**: Large Language Models (LLMs) have shown impressive proficiency across a range of natural language processing tasks yet remain vulnerable to adversarial prompts, known as jailbreak attacks, carefully designed to elicit harmful responses from LLMs. Traditional methods rely on manual heuristics, which suffer from limited generalizability. While being automatic, optimization-based attacks often produce unnatural jailbreak prompts that are easy to detect by safety filters or require high computational overhead due to discrete token optimization. Witnessing the limitations of existing jailbreak methods, we introduce Generative Adversarial Suffix Prompter (GASP), a novel framework that combines human-readable prompt generation with Latent Bayesian Optimization (LBO) to improve adversarial suffix creation in a fully black-box setting. GASP leverages LBO to craft adversarial suffixes by efficiently exploring continuous embedding spaces, gradually optimizing the model to improve attack efficacy while balancing prompt coherence through a targeted iterative refinement procedure. Our experiments show that GASP can generate natural jailbreak prompts, significantly improving attack success rates, reducing training times, and accelerating inference speed, thus making it an efficient and scalable solution for red-teaming LLMs.



## **32. RAG-Thief: Scalable Extraction of Private Data from Retrieval-Augmented Generation Applications with Agent-based Attacks**

cs.CR

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2411.14110v1) [paper-pdf](http://arxiv.org/pdf/2411.14110v1)

**Authors**: Changyue Jiang, Xudong Pan, Geng Hong, Chenfu Bao, Min Yang

**Abstract**: While large language models (LLMs) have achieved notable success in generative tasks, they still face limitations, such as lacking up-to-date knowledge and producing hallucinations. Retrieval-Augmented Generation (RAG) enhances LLM performance by integrating external knowledge bases, providing additional context which significantly improves accuracy and knowledge coverage. However, building these external knowledge bases often requires substantial resources and may involve sensitive information. In this paper, we propose an agent-based automated privacy attack called RAG-Thief, which can extract a scalable amount of private data from the private database used in RAG applications. We conduct a systematic study on the privacy risks associated with RAG applications, revealing that the vulnerability of LLMs makes the private knowledge bases suffer significant privacy risks. Unlike previous manual attacks which rely on traditional prompt injection techniques, RAG-Thief starts with an initial adversarial query and learns from model responses, progressively generating new queries to extract as many chunks from the knowledge base as possible. Experimental results show that our RAG-Thief can extract over 70% information from the private knowledge bases within customized RAG applications deployed on local machines and real-world platforms, including OpenAI's GPTs and ByteDance's Coze. Our findings highlight the privacy vulnerabilities in current RAG applications and underscore the pressing need for stronger safeguards.



## **33. Verifying the Robustness of Automatic Credibility Assessment**

cs.CL

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2303.08032v3) [paper-pdf](http://arxiv.org/pdf/2303.08032v3)

**Authors**: Piotr Przybyła, Alexander Shvets, Horacio Saggion

**Abstract**: Text classification methods have been widely investigated as a way to detect content of low credibility: fake news, social media bots, propaganda, etc. Quite accurate models (likely based on deep neural networks) help in moderating public electronic platforms and often cause content creators to face rejection of their submissions or removal of already published texts. Having the incentive to evade further detection, content creators try to come up with a slightly modified version of the text (known as an attack with an adversarial example) that exploit the weaknesses of classifiers and result in a different output. Here we systematically test the robustness of common text classifiers against available attacking techniques and discover that, indeed, meaning-preserving changes in input text can mislead the models. The approaches we test focus on finding vulnerable spans in text and replacing individual characters or words, taking into account the similarity between the original and replacement content. We also introduce BODEGA: a benchmark for testing both victim models and attack methods on four misinformation detection tasks in an evaluation framework designed to simulate real use-cases of content moderation. The attacked tasks include (1) fact checking and detection of (2) hyperpartisan news, (3) propaganda and (4) rumours. Our experimental results show that modern large language models are often more vulnerable to attacks than previous, smaller solutions, e.g. attacks on GEMMA being up to 27\% more successful than those on BERT. Finally, we manually analyse a subset adversarial examples and check what kinds of modifications are used in successful attacks.



## **34. Global Challenge for Safe and Secure LLMs Track 1**

cs.CR

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2411.14502v1) [paper-pdf](http://arxiv.org/pdf/2411.14502v1)

**Authors**: Xiaojun Jia, Yihao Huang, Yang Liu, Peng Yan Tan, Weng Kuan Yau, Mun-Thye Mak, Xin Ming Sim, Wee Siong Ng, See Kiong Ng, Hanqing Liu, Lifeng Zhou, Huanqian Yan, Xiaobing Sun, Wei Liu, Long Wang, Yiming Qian, Yong Liu, Junxiao Yang, Zhexin Zhang, Leqi Lei, Renmiao Chen, Yida Lu, Shiyao Cui, Zizhou Wang, Shaohua Li, Yan Wang, Rick Siow Mong Goh, Liangli Zhen, Yingjie Zhang, Zhe Zhao

**Abstract**: This paper introduces the Global Challenge for Safe and Secure Large Language Models (LLMs), a pioneering initiative organized by AI Singapore (AISG) and the CyberSG R&D Programme Office (CRPO) to foster the development of advanced defense mechanisms against automated jailbreaking attacks. With the increasing integration of LLMs in critical sectors such as healthcare, finance, and public administration, ensuring these models are resilient to adversarial attacks is vital for preventing misuse and upholding ethical standards. This competition focused on two distinct tracks designed to evaluate and enhance the robustness of LLM security frameworks. Track 1 tasked participants with developing automated methods to probe LLM vulnerabilities by eliciting undesirable responses, effectively testing the limits of existing safety protocols within LLMs. Participants were challenged to devise techniques that could bypass content safeguards across a diverse array of scenarios, from offensive language to misinformation and illegal activities. Through this process, Track 1 aimed to deepen the understanding of LLM vulnerabilities and provide insights for creating more resilient models.



## **35. Next-Generation Phishing: How LLM Agents Empower Cyber Attackers**

cs.CR

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2411.13874v1) [paper-pdf](http://arxiv.org/pdf/2411.13874v1)

**Authors**: Khalifa Afane, Wenqi Wei, Ying Mao, Junaid Farooq, Juntao Chen

**Abstract**: The escalating threat of phishing emails has become increasingly sophisticated with the rise of Large Language Models (LLMs). As attackers exploit LLMs to craft more convincing and evasive phishing emails, it is crucial to assess the resilience of current phishing defenses. In this study we conduct a comprehensive evaluation of traditional phishing detectors, such as Gmail Spam Filter, Apache SpamAssassin, and Proofpoint, as well as machine learning models like SVM, Logistic Regression, and Naive Bayes, in identifying both traditional and LLM-rephrased phishing emails. We also explore the emerging role of LLMs as phishing detection tools, a method already adopted by companies like NTT Security Holdings and JPMorgan Chase. Our results reveal notable declines in detection accuracy for rephrased emails across all detectors, highlighting critical weaknesses in current phishing defenses. As the threat landscape evolves, our findings underscore the need for stronger security controls and regulatory oversight on LLM-generated content to prevent its misuse in creating advanced phishing attacks. This study contributes to the development of more effective Cyber Threat Intelligence (CTI) by leveraging LLMs to generate diverse phishing variants that can be used for data augmentation, harnessing the power of LLMs to enhance phishing detection, and paving the way for more robust and adaptable threat detection systems.



## **36. Rethinking the Intermediate Features in Adversarial Attacks: Misleading Robotic Models via Adversarial Distillation**

cs.LG

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2411.15222v1) [paper-pdf](http://arxiv.org/pdf/2411.15222v1)

**Authors**: Ke Zhao, Huayang Huang, Miao Li, Yu Wu

**Abstract**: Language-conditioned robotic learning has significantly enhanced robot adaptability by enabling a single model to execute diverse tasks in response to verbal commands. Despite these advancements, security vulnerabilities within this domain remain largely unexplored. This paper addresses this gap by proposing a novel adversarial prompt attack tailored to language-conditioned robotic models. Our approach involves crafting a universal adversarial prefix that induces the model to perform unintended actions when added to any original prompt. We demonstrate that existing adversarial techniques exhibit limited effectiveness when directly transferred to the robotic domain due to the inherent robustness of discretized robotic action spaces. To overcome this challenge, we propose to optimize adversarial prefixes based on continuous action representations, circumventing the discretization process. Additionally, we identify the beneficial impact of intermediate features on adversarial attacks and leverage the negative gradient of intermediate self-attention features to further enhance attack efficacy. Extensive experiments on VIMA models across 13 robot manipulation tasks validate the superiority of our method over existing approaches and demonstrate its transferability across different model variants.



## **37. TransLinkGuard: Safeguarding Transformer Models Against Model Stealing in Edge Deployment**

cs.CR

Accepted by ACM MM24 Conference

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2404.11121v2) [paper-pdf](http://arxiv.org/pdf/2404.11121v2)

**Authors**: Qinfeng Li, Zhiqiang Shen, Zhenghan Qin, Yangfan Xie, Xuhong Zhang, Tianyu Du, Jianwei Yin

**Abstract**: Proprietary large language models (LLMs) have been widely applied in various scenarios. Additionally, deploying LLMs on edge devices is trending for efficiency and privacy reasons. However, edge deployment of proprietary LLMs introduces new security challenges: edge-deployed models are exposed as white-box accessible to users, enabling adversaries to conduct effective model stealing (MS) attacks. Unfortunately, existing defense mechanisms fail to provide effective protection. Specifically, we identify four critical protection properties that existing methods fail to simultaneously satisfy: (1) maintaining protection after a model is physically copied; (2) authorizing model access at request level; (3) safeguarding runtime reverse engineering; (4) achieving high security with negligible runtime overhead. To address the above issues, we propose TransLinkGuard, a plug-and-play model protection approach against model stealing on edge devices. The core part of TransLinkGuard is a lightweight authorization module residing in a secure environment, e.g., TEE. The authorization module can freshly authorize each request based on its input. Extensive experiments show that TransLinkGuard achieves the same security protection as the black-box security guarantees with negligible overhead.



## **38. AttentionBreaker: Adaptive Evolutionary Optimization for Unmasking Vulnerabilities in LLMs through Bit-Flip Attacks**

cs.CR

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2411.13757v1) [paper-pdf](http://arxiv.org/pdf/2411.13757v1)

**Authors**: Sanjay Das, Swastik Bhattacharya, Souvik Kundu, Shamik Kundu, Anand Menon, Arnab Raha, Kanad Basu

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing (NLP), excelling in tasks like text generation and summarization. However, their increasing adoption in mission-critical applications raises concerns about hardware-based threats, particularly bit-flip attacks (BFAs). BFAs, enabled by fault injection methods such as Rowhammer, target model parameters in memory, compromising both integrity and performance. Identifying critical parameters for BFAs in the vast parameter space of LLMs poses significant challenges. While prior research suggests transformer-based architectures are inherently more robust to BFAs compared to traditional deep neural networks, we challenge this assumption. For the first time, we demonstrate that as few as three bit-flips can cause catastrophic performance degradation in an LLM with billions of parameters. Current BFA techniques are inadequate for exploiting this vulnerability due to the difficulty of efficiently identifying critical parameters within the immense parameter space. To address this, we propose AttentionBreaker, a novel framework tailored for LLMs that enables efficient traversal of the parameter space to identify critical parameters. Additionally, we introduce GenBFA, an evolutionary optimization strategy designed to refine the search further, isolating the most critical bits for an efficient and effective attack. Empirical results reveal the profound vulnerability of LLMs to AttentionBreaker. For example, merely three bit-flips (4.129 x 10^-9% of total parameters) in the LLaMA3-8B-Instruct 8-bit quantized (W8) model result in a complete performance collapse: accuracy on MMLU tasks drops from 67.3% to 0%, and Wikitext perplexity skyrockets from 12.6 to 4.72 x 10^5. These findings underscore the effectiveness of AttentionBreaker in uncovering and exploiting critical vulnerabilities within LLM architectures.



## **39. SoK: A Systems Perspective on Compound AI Threats and Countermeasures**

cs.CR

13 pages, 4 figures, 2 tables

**SubmitDate**: 2024-11-20    [abs](http://arxiv.org/abs/2411.13459v1) [paper-pdf](http://arxiv.org/pdf/2411.13459v1)

**Authors**: Sarbartha Banerjee, Prateek Sahu, Mulong Luo, Anjo Vahldiek-Oberwagner, Neeraja J. Yadwadkar, Mohit Tiwari

**Abstract**: Large language models (LLMs) used across enterprises often use proprietary models and operate on sensitive inputs and data. The wide range of attack vectors identified in prior research - targeting various software and hardware components used in training and inference - makes it extremely challenging to enforce confidentiality and integrity policies.   As we advance towards constructing compound AI inference pipelines that integrate multiple large language models (LLMs), the attack surfaces expand significantly. Attackers now focus on the AI algorithms as well as the software and hardware components associated with these systems. While current research often examines these elements in isolation, we find that combining cross-layer attack observations can enable powerful end-to-end attacks with minimal assumptions about the threat model. Given, the sheer number of existing attacks at each layer, we need a holistic and systemized understanding of different attack vectors at each layer.   This SoK discusses different software and hardware attacks applicable to compound AI systems and demonstrates how combining multiple attack mechanisms can reduce the threat model assumptions required for an isolated attack. Next, we systematize the ML attacks in lines with the Mitre Att&ck framework to better position each attack based on the threat model. Finally, we outline the existing countermeasures for both software and hardware layers and discuss the necessity of a comprehensive defense strategy to enable the secure and high-performance deployment of compound AI systems.



## **40. WaterPark: A Robustness Assessment of Language Model Watermarking**

cs.CR

22 pages

**SubmitDate**: 2024-11-20    [abs](http://arxiv.org/abs/2411.13425v1) [paper-pdf](http://arxiv.org/pdf/2411.13425v1)

**Authors**: Jiacheng Liang, Zian Wang, Lauren Hong, Shouling Ji, Ting Wang

**Abstract**: To mitigate the misuse of large language models (LLMs), such as disinformation, automated phishing, and academic cheating, there is a pressing need for the capability of identifying LLM-generated texts. Watermarking emerges as one promising solution: it plants statistical signals into LLMs' generative processes and subsequently verifies whether LLMs produce given texts. Various watermarking methods (``watermarkers'') have been proposed; yet, due to the lack of unified evaluation platforms, many critical questions remain under-explored: i) What are the strengths/limitations of various watermarkers, especially their attack robustness? ii) How do various design choices impact their robustness? iii) How to optimally operate watermarkers in adversarial environments?   To fill this gap, we systematize existing LLM watermarkers and watermark removal attacks, mapping out their design spaces. We then develop WaterPark, a unified platform that integrates 10 state-of-the-art watermarkers and 12 representative attacks. More importantly, leveraging WaterPark, we conduct a comprehensive assessment of existing watermarkers, unveiling the impact of various design choices on their attack robustness. For instance, a watermarker's resilience to increasingly intensive attacks hinges on its context dependency. We further explore the best practices to operate watermarkers in adversarial environments. For instance, using a generic detector alongside a watermark-specific detector improves the security of vulnerable watermarkers. We believe our study sheds light on current LLM watermarking techniques while WaterPark serves as a valuable testbed to facilitate future research.



## **41. CryptoFormalEval: Integrating LLMs and Formal Verification for Automated Cryptographic Protocol Vulnerability Detection**

cs.CR

**SubmitDate**: 2024-11-20    [abs](http://arxiv.org/abs/2411.13627v1) [paper-pdf](http://arxiv.org/pdf/2411.13627v1)

**Authors**: Cristian Curaba, Denis D'Ambrosi, Alessandro Minisini, Natalia Pérez-Campanero Antolín

**Abstract**: Cryptographic protocols play a fundamental role in securing modern digital infrastructure, but they are often deployed without prior formal verification. This could lead to the adoption of distributed systems vulnerable to attack vectors. Formal verification methods, on the other hand, require complex and time-consuming techniques that lack automatization. In this paper, we introduce a benchmark to assess the ability of Large Language Models (LLMs) to autonomously identify vulnerabilities in new cryptographic protocols through interaction with Tamarin: a theorem prover for protocol verification. We created a manually validated dataset of novel, flawed, communication protocols and designed a method to automatically verify the vulnerabilities found by the AI agents. Our results about the performances of the current frontier models on the benchmark provides insights about the possibility of cybersecurity applications by integrating LLMs with symbolic reasoning systems.



## **42. TAPT: Test-Time Adversarial Prompt Tuning for Robust Inference in Vision-Language Models**

cs.CV

**SubmitDate**: 2024-11-20    [abs](http://arxiv.org/abs/2411.13136v1) [paper-pdf](http://arxiv.org/pdf/2411.13136v1)

**Authors**: Xin Wang, Kai Chen, Jiaming Zhang, Jingjing Chen, Xingjun Ma

**Abstract**: Large pre-trained Vision-Language Models (VLMs) such as CLIP have demonstrated excellent zero-shot generalizability across various downstream tasks. However, recent studies have shown that the inference performance of CLIP can be greatly degraded by small adversarial perturbations, especially its visual modality, posing significant safety threats. To mitigate this vulnerability, in this paper, we propose a novel defense method called Test-Time Adversarial Prompt Tuning (TAPT) to enhance the inference robustness of CLIP against visual adversarial attacks. TAPT is a test-time defense method that learns defensive bimodal (textual and visual) prompts to robustify the inference process of CLIP. Specifically, it is an unsupervised method that optimizes the defensive prompts for each test sample by minimizing a multi-view entropy and aligning adversarial-clean distributions. We evaluate the effectiveness of TAPT on 11 benchmark datasets, including ImageNet and 10 other zero-shot datasets, demonstrating that it enhances the zero-shot adversarial robustness of the original CLIP by at least 48.9% against AutoAttack (AA), while largely maintaining performance on clean examples. Moreover, TAPT outperforms existing adversarial prompt tuning methods across various backbones, achieving an average robustness improvement of at least 36.6%.



## **43. When Backdoors Speak: Understanding LLM Backdoor Attacks Through Model-Generated Explanations**

cs.CR

**SubmitDate**: 2024-11-19    [abs](http://arxiv.org/abs/2411.12701v1) [paper-pdf](http://arxiv.org/pdf/2411.12701v1)

**Authors**: Huaizhi Ge, Yiming Li, Qifan Wang, Yongfeng Zhang, Ruixiang Tang

**Abstract**: Large Language Models (LLMs) are vulnerable to backdoor attacks, where hidden triggers can maliciously manipulate model behavior. While several backdoor attack methods have been proposed, the mechanisms by which backdoor functions operate in LLMs remain underexplored. In this paper, we move beyond attacking LLMs and investigate backdoor functionality through the novel lens of natural language explanations. Specifically, we leverage LLMs' generative capabilities to produce human-understandable explanations for their decisions, allowing us to compare explanations for clean and poisoned samples. We explore various backdoor attacks and embed the backdoor into LLaMA models for multiple tasks. Our experiments show that backdoored models produce higher-quality explanations for clean data compared to poisoned data, while generating significantly more consistent explanations for poisoned data than for clean data. We further analyze the explanation generation process, revealing that at the token level, the explanation token of poisoned samples only appears in the final few transformer layers of the LLM. At the sentence level, attention dynamics indicate that poisoned inputs shift attention from the input context when generating the explanation. These findings deepen our understanding of backdoor attack mechanisms in LLMs and offer a framework for detecting such vulnerabilities through explainability techniques, contributing to the development of more secure LLMs.



## **44. Exploring adversarial robustness of JPEG AI: methodology, comparison and new methods**

eess.IV

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11795v1) [paper-pdf](http://arxiv.org/pdf/2411.11795v1)

**Authors**: Egor Kovalev, Georgii Bychkov, Khaled Abud, Aleksandr Gushchin, Anna Chistyakova, Sergey Lavrushkin, Dmitriy Vatolin, Anastasia Antsiferova

**Abstract**: Adversarial robustness of neural networks is an increasingly important area of research, combining studies on computer vision models, large language models (LLMs), and others. With the release of JPEG AI - the first standard for end-to-end neural image compression (NIC) methods - the question of its robustness has become critically significant. JPEG AI is among the first international, real-world applications of neural-network-based models to be embedded in consumer devices. However, research on NIC robustness has been limited to open-source codecs and a narrow range of attacks. This paper proposes a new methodology for measuring NIC robustness to adversarial attacks. We present the first large-scale evaluation of JPEG AI's robustness, comparing it with other NIC models. Our evaluation results and code are publicly available online (link is hidden for a blind review).



## **45. DAWN: Designing Distributed Agents in a Worldwide Network**

cs.NI

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2410.22339v2) [paper-pdf](http://arxiv.org/pdf/2410.22339v2)

**Authors**: Zahra Aminiranjbar, Jianan Tang, Qiudan Wang, Shubha Pant, Mahesh Viswanathan

**Abstract**: The rapid evolution of Large Language Models (LLMs) has transformed them from basic conversational tools into sophisticated entities capable of complex reasoning and decision-making. These advancements have led to the development of specialized LLM-based agents designed for diverse tasks such as coding and web browsing. As these agents become more capable, the need for a robust framework that facilitates global communication and collaboration among them towards advanced objectives has become increasingly critical. Distributed Agents in a Worldwide Network (DAWN) addresses this need by offering a versatile framework that integrates LLM-based agents with traditional software systems, enabling the creation of agentic applications suited for a wide range of use cases. DAWN enables distributed agents worldwide to register and be easily discovered through Gateway Agents. Collaborations among these agents are coordinated by a Principal Agent equipped with reasoning strategies. DAWN offers three operational modes: No-LLM Mode for deterministic tasks, Copilot for augmented decision-making, and LLM Agent for autonomous operations. Additionally, DAWN ensures the safety and security of agent collaborations globally through a dedicated safety, security, and compliance layer, protecting the network against attackers and adhering to stringent security and compliance standards. These features make DAWN a robust network for deploying agent-based applications across various industries.



## **46. TrojanRobot: Backdoor Attacks Against Robotic Manipulation in the Physical World**

cs.RO

Initial version with preliminary results. We welcome any feedback or  suggestions

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11683v1) [paper-pdf](http://arxiv.org/pdf/2411.11683v1)

**Authors**: Xianlong Wang, Hewen Pan, Hangtao Zhang, Minghui Li, Shengshan Hu, Ziqi Zhou, Lulu Xue, Peijin Guo, Yichen Wang, Wei Wan, Aishan Liu, Leo Yu Zhang

**Abstract**: Robotic manipulation refers to the autonomous handling and interaction of robots with objects using advanced techniques in robotics and artificial intelligence. The advent of powerful tools such as large language models (LLMs) and large vision-language models (LVLMs) has significantly enhanced the capabilities of these robots in environmental perception and decision-making. However, the introduction of these intelligent agents has led to security threats such as jailbreak attacks and adversarial attacks.   In this research, we take a further step by proposing a backdoor attack specifically targeting robotic manipulation and, for the first time, implementing backdoor attack in the physical world. By embedding a backdoor visual language model into the visual perception module within the robotic system, we successfully mislead the robotic arm's operation in the physical world, given the presence of common items as triggers. Experimental evaluations in the physical world demonstrate the effectiveness of the proposed backdoor attack.



## **47. Enhancing Vision-Language Model Safety through Progressive Concept-Bottleneck-Driven Alignment**

cs.CV

arXiv admin note: substantial text overlap with arXiv:2405.13581

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11543v1) [paper-pdf](http://arxiv.org/pdf/2411.11543v1)

**Authors**: Zhendong Liu, Yuanbi Nie, Yingshui Tan, Xiangyu Yue, Qiushi Cui, Chongjun Wang, Xiaoyong Zhu, Bo Zheng

**Abstract**: Benefiting from the powerful capabilities of Large Language Models (LLMs), pre-trained visual encoder models connected to LLMs form Vision Language Models (VLMs). However, recent research shows that the visual modality in VLMs is highly vulnerable, allowing attackers to bypass safety alignment in LLMs through visually transmitted content, launching harmful attacks. To address this challenge, we propose a progressive concept-based alignment strategy, PSA-VLM, which incorporates safety modules as concept bottlenecks to enhance visual modality safety alignment. By aligning model predictions with specific safety concepts, we improve defenses against risky images, enhancing explainability and controllability while minimally impacting general performance. Our method is obtained through two-stage training. The low computational cost of the first stage brings very effective performance improvement, and the fine-tuning of the language model in the second stage further improves the safety performance. Our method achieves state-of-the-art results on popular VLM safety benchmark.



## **48. Membership Inference Attack against Long-Context Large Language Models**

cs.CL

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11424v1) [paper-pdf](http://arxiv.org/pdf/2411.11424v1)

**Authors**: Zixiong Wang, Gaoyang Liu, Yang Yang, Chen Wang

**Abstract**: Recent advances in Large Language Models (LLMs) have enabled them to overcome their context window limitations, and demonstrate exceptional retrieval and reasoning capacities on longer context. Quesion-answering systems augmented with Long-Context Language Models (LCLMs) can automatically search massive external data and incorporate it into their contexts, enabling faithful predictions and reducing issues such as hallucinations and knowledge staleness. Existing studies targeting LCLMs mainly concentrate on addressing the so-called lost-in-the-middle problem or improving the inference effiencicy, leaving their privacy risks largely unexplored. In this paper, we aim to bridge this gap and argue that integrating all information into the long context makes it a repository of sensitive information, which often contains private data such as medical records or personal identities. We further investigate the membership privacy within LCLMs external context, with the aim of determining whether a given document or sequence is included in the LCLMs context. Our basic idea is that if a document lies in the context, it will exhibit a low generation loss or a high degree of semantic similarity to the contents generated by LCLMs. We for the first time propose six membership inference attack (MIA) strategies tailored for LCLMs and conduct extensive experiments on various popular models. Empirical results demonstrate that our attacks can accurately infer membership status in most cases, e.g., 90.66% attack F1-score on Multi-document QA datasets with LongChat-7b-v1.5-32k, highlighting significant risks of membership leakage within LCLMs input contexts. Furthermore, we examine the underlying reasons why LCLMs are susceptible to revealing such membership information.



## **49. The Dark Side of Trust: Authority Citation-Driven Jailbreak Attacks on Large Language Models**

cs.LG

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11407v1) [paper-pdf](http://arxiv.org/pdf/2411.11407v1)

**Authors**: Xikang Yang, Xuehai Tang, Jizhong Han, Songlin Hu

**Abstract**: The widespread deployment of large language models (LLMs) across various domains has showcased their immense potential while exposing significant safety vulnerabilities. A major concern is ensuring that LLM-generated content aligns with human values. Existing jailbreak techniques reveal how this alignment can be compromised through specific prompts or adversarial suffixes. In this study, we introduce a new threat: LLMs' bias toward authority. While this inherent bias can improve the quality of outputs generated by LLMs, it also introduces a potential vulnerability, increasing the risk of producing harmful content. Notably, the biases in LLMs is the varying levels of trust given to different types of authoritative information in harmful queries. For example, malware development often favors trust GitHub. To better reveal the risks with LLM, we propose DarkCite, an adaptive authority citation matcher and generator designed for a black-box setting. DarkCite matches optimal citation types to specific risk types and generates authoritative citations relevant to harmful instructions, enabling more effective jailbreak attacks on aligned LLMs.Our experiments show that DarkCite achieves a higher attack success rate (e.g., LLama-2 at 76% versus 68%) than previous methods. To counter this risk, we propose an authenticity and harm verification defense strategy, raising the average defense pass rate (DPR) from 11% to 74%. More importantly, the ability to link citations to the content they encompass has become a foundational function in LLMs, amplifying the influence of LLMs' bias toward authority.



## **50. Hacking Back the AI-Hacker: Prompt Injection as a Defense Against LLM-driven Cyberattacks**

cs.CR

v0.2 (evaluated on more agents)

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2410.20911v2) [paper-pdf](http://arxiv.org/pdf/2410.20911v2)

**Authors**: Dario Pasquini, Evgenios M. Kornaropoulos, Giuseppe Ateniese

**Abstract**: Large language models (LLMs) are increasingly being harnessed to automate cyberattacks, making sophisticated exploits more accessible and scalable. In response, we propose a new defense strategy tailored to counter LLM-driven cyberattacks. We introduce Mantis, a defensive framework that exploits LLMs' susceptibility to adversarial inputs to undermine malicious operations. Upon detecting an automated cyberattack, Mantis plants carefully crafted inputs into system responses, leading the attacker's LLM to disrupt their own operations (passive defense) or even compromise the attacker's machine (active defense). By deploying purposefully vulnerable decoy services to attract the attacker and using dynamic prompt injections for the attacker's LLM, Mantis can autonomously hack back the attacker. In our experiments, Mantis consistently achieved over 95% effectiveness against automated LLM-driven attacks. To foster further research and collaboration, Mantis is available as an open-source tool: https://github.com/pasquini-dario/project_mantis



