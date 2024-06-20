# Latest Large Language Model Attack Papers
**update at 2024-06-20 09:38:29**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Stealth edits for provably fixing or attacking large language models**

cs.AI

24 pages, 9 figures. Open source implementation:  https://github.com/qinghua-zhou/stealth-edits

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.12670v1) [paper-pdf](http://arxiv.org/pdf/2406.12670v1)

**Authors**: Oliver J. Sutton, Qinghua Zhou, Wei Wang, Desmond J. Higham, Alexander N. Gorban, Alexander Bastounis, Ivan Y. Tyukin

**Abstract**: We reveal new methods and the theoretical foundations of techniques for editing large language models. We also show how the new theory can be used to assess the editability of models and to expose their susceptibility to previously unknown malicious attacks. Our theoretical approach shows that a single metric (a specific measure of the intrinsic dimensionality of the model's features) is fundamental to predicting the success of popular editing approaches, and reveals new bridges between disparate families of editing methods. We collectively refer to these approaches as stealth editing methods, because they aim to directly and inexpensively update a model's weights to correct the model's responses to known hallucinating prompts without otherwise affecting the model's behaviour, without requiring retraining. By carefully applying the insight gleaned from our theoretical investigation, we are able to introduce a new network block -- named a jet-pack block -- which is optimised for highly selective model editing, uses only standard network operations, and can be inserted into existing networks. The intrinsic dimensionality metric also determines the vulnerability of a language model to a stealth attack: a small change to a model's weights which changes its response to a single attacker-chosen prompt. Stealth attacks do not require access to or knowledge of the model's training data, therefore representing a potent yet previously unrecognised threat to redistributed foundation models. They are computationally simple enough to be implemented in malware in many cases. Extensive experimental results illustrate and support the method and its theoretical underpinnings. Demos and source code for editing language models are available at https://github.com/qinghua-zhou/stealth-edits.



## **2. MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal Large Language Models**

cs.CV

The datasets were incomplete as they did not include all the  necessary copyrights

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2311.17600v4) [paper-pdf](http://arxiv.org/pdf/2311.17600v4)

**Authors**: Xin Liu, Yichen Zhu, Jindong Gu, Yunshi Lan, Chao Yang, Yu Qiao

**Abstract**: The security concerns surrounding Large Language Models (LLMs) have been extensively explored, yet the safety of Multimodal Large Language Models (MLLMs) remains understudied. In this paper, we observe that Multimodal Large Language Models (MLLMs) can be easily compromised by query-relevant images, as if the text query itself were malicious. To address this, we introduce MM-SafetyBench, a comprehensive framework designed for conducting safety-critical evaluations of MLLMs against such image-based manipulations. We have compiled a dataset comprising 13 scenarios, resulting in a total of 5,040 text-image pairs. Our analysis across 12 state-of-the-art models reveals that MLLMs are susceptible to breaches instigated by our approach, even when the equipped LLMs have been safety-aligned. In response, we propose a straightforward yet effective prompting strategy to enhance the resilience of MLLMs against these types of attacks. Our work underscores the need for a concerted effort to strengthen and enhance the safety measures of open-source MLLMs against potential malicious exploits. The resource is available at https://github.com/isXinLiu/MM-SafetyBench



## **3. Authorship Obfuscation in Multilingual Machine-Generated Text Detection**

cs.CL

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2401.07867v2) [paper-pdf](http://arxiv.org/pdf/2401.07867v2)

**Authors**: Dominik Macko, Robert Moro, Adaku Uchendu, Ivan Srba, Jason Samuel Lucas, Michiharu Yamashita, Nafis Irtiza Tripto, Dongwon Lee, Jakub Simko, Maria Bielikova

**Abstract**: High-quality text generation capability of recent Large Language Models (LLMs) causes concerns about their misuse (e.g., in massive generation/spread of disinformation). Machine-generated text (MGT) detection is important to cope with such threats. However, it is susceptible to authorship obfuscation (AO) methods, such as paraphrasing, which can cause MGTs to evade detection. So far, this was evaluated only in monolingual settings. Thus, the susceptibility of recently proposed multilingual detectors is still unknown. We fill this gap by comprehensively benchmarking the performance of 10 well-known AO methods, attacking 37 MGT detection methods against MGTs in 11 languages (i.e., 10 $\times$ 37 $\times$ 11 = 4,070 combinations). We also evaluate the effect of data augmentation on adversarial robustness using obfuscated texts. The results indicate that all tested AO methods can cause evasion of automated detection in all tested languages, where homoglyph attacks are especially successful. However, some of the AO methods severely damaged the text, making it no longer readable or easily recognizable by humans (e.g., changed language, weird characters).



## **4. Can We Trust Large Language Models Generated Code? A Framework for In-Context Learning, Security Patterns, and Code Evaluations Across Diverse LLMs**

cs.CR

27 pages, Standard Journal Paper submitted to Q1 Elsevier

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.12513v1) [paper-pdf](http://arxiv.org/pdf/2406.12513v1)

**Authors**: Ahmad Mohsin, Helge Janicke, Adrian Wood, Iqbal H. Sarker, Leandros Maglaras, Naeem Janjua

**Abstract**: Large Language Models (LLMs) such as ChatGPT and GitHub Copilot have revolutionized automated code generation in software engineering. However, as these models are increasingly utilized for software development, concerns have arisen regarding the security and quality of the generated code. These concerns stem from LLMs being primarily trained on publicly available code repositories and internet-based textual data, which may contain insecure code. This presents a significant risk of perpetuating vulnerabilities in the generated code, creating potential attack vectors for exploitation by malicious actors. Our research aims to tackle these issues by introducing a framework for secure behavioral learning of LLMs through In-Content Learning (ICL) patterns during the code generation process, followed by rigorous security evaluations. To achieve this, we have selected four diverse LLMs for experimentation. We have evaluated these coding LLMs across three programming languages and identified security vulnerabilities and code smells. The code is generated through ICL with curated problem sets and undergoes rigorous security testing to evaluate the overall quality and trustworthiness of the generated code. Our research indicates that ICL-driven one-shot and few-shot learning patterns can enhance code security, reducing vulnerabilities in various programming scenarios. Developers and researchers should know that LLMs have a limited understanding of security principles. This may lead to security breaches when the generated code is deployed in production systems. Our research highlights LLMs are a potential source of new vulnerabilities to the software supply chain. It is important to consider this when using LLMs for code generation. This research article offers insights into improving LLM security and encourages proactive use of LLMs for code generation to ensure software system safety.



## **5. Identifying and Mitigating Privacy Risks Stemming from Language Models: A Survey**

cs.CL

15 pages

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2310.01424v2) [paper-pdf](http://arxiv.org/pdf/2310.01424v2)

**Authors**: Victoria Smith, Ali Shahin Shamsabadi, Carolyn Ashurst, Adrian Weller

**Abstract**: Large Language Models (LLMs) have shown greatly enhanced performance in recent years, attributed to increased size and extensive training data. This advancement has led to widespread interest and adoption across industries and the public. However, training data memorization in Machine Learning models scales with model size, particularly concerning for LLMs. Memorized text sequences have the potential to be directly leaked from LLMs, posing a serious threat to data privacy. Various techniques have been developed to attack LLMs and extract their training data. As these models continue to grow, this issue becomes increasingly critical. To help researchers and policymakers understand the state of knowledge around privacy attacks and mitigations, including where more work is needed, we present the first SoK on data privacy for LLMs. We (i) identify a taxonomy of salient dimensions where attacks differ on LLMs, (ii) systematize existing attacks, using our taxonomy of dimensions to highlight key trends, (iii) survey existing mitigation strategies, highlighting their strengths and limitations, and (iv) identify key gaps, demonstrating open problems and areas for concern.



## **6. Unique Security and Privacy Threats of Large Language Model: A Comprehensive Survey**

cs.CR

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.07973v2) [paper-pdf](http://arxiv.org/pdf/2406.07973v2)

**Authors**: Shang Wang, Tianqing Zhu, Bo Liu, Ming Ding, Xu Guo, Dayong Ye, Wanlei Zhou, Philip S. Yu

**Abstract**: With the rapid development of artificial intelligence, large language models (LLMs) have made remarkable advancements in natural language processing. These models are trained on vast datasets to exhibit powerful language understanding and generation capabilities across various applications, including machine translation, chatbots, and agents. However, LLMs have revealed a variety of privacy and security issues throughout their life cycle, drawing significant academic and industrial attention. Moreover, the risks faced by LLMs differ significantly from those encountered by traditional language models. Given that current surveys lack a clear taxonomy of unique threat models across diverse scenarios, we emphasize the unique privacy and security threats associated with five specific scenarios: pre-training, fine-tuning, retrieval-augmented generation systems, deployment, and LLM-based agents. Addressing the characteristics of each risk, this survey outlines potential threats and countermeasures. Research on attack and defense situations can offer feasible research directions, enabling more areas to benefit from LLMs.



## **7. Defending Against Social Engineering Attacks in the Age of LLMs**

cs.CL

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.12263v1) [paper-pdf](http://arxiv.org/pdf/2406.12263v1)

**Authors**: Lin Ai, Tharindu Kumarage, Amrita Bhattacharjee, Zizhou Liu, Zheng Hui, Michael Davinroy, James Cook, Laura Cassani, Kirill Trapeznikov, Matthias Kirchner, Arslan Basharat, Anthony Hoogs, Joshua Garland, Huan Liu, Julia Hirschberg

**Abstract**: The proliferation of Large Language Models (LLMs) poses challenges in detecting and mitigating digital deception, as these models can emulate human conversational patterns and facilitate chat-based social engineering (CSE) attacks. This study investigates the dual capabilities of LLMs as both facilitators and defenders against CSE threats. We develop a novel dataset, SEConvo, simulating CSE scenarios in academic and recruitment contexts, and designed to examine how LLMs can be exploited in these situations. Our findings reveal that, while off-the-shelf LLMs generate high-quality CSE content, their detection capabilities are suboptimal, leading to increased operational costs for defense. In response, we propose ConvoSentinel, a modular defense pipeline that improves detection at both the message and the conversation levels, offering enhanced adaptability and cost-effectiveness. The retrieval-augmented module in ConvoSentinel identifies malicious intent by comparing messages to a database of similar conversations, enhancing CSE detection at all stages. Our study highlights the need for advanced strategies to leverage LLMs in cybersecurity.



## **8. Adversarial Attacks on Large Language Models in Medicine**

cs.AI

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.12259v1) [paper-pdf](http://arxiv.org/pdf/2406.12259v1)

**Authors**: Yifan Yang, Qiao Jin, Furong Huang, Zhiyong Lu

**Abstract**: The integration of Large Language Models (LLMs) into healthcare applications offers promising advancements in medical diagnostics, treatment recommendations, and patient care. However, the susceptibility of LLMs to adversarial attacks poses a significant threat, potentially leading to harmful outcomes in delicate medical contexts. This study investigates the vulnerability of LLMs to two types of adversarial attacks in three medical tasks. Utilizing real-world patient data, we demonstrate that both open-source and proprietary LLMs are susceptible to manipulation across multiple tasks. This research further reveals that domain-specific tasks demand more adversarial data in model fine-tuning than general domain tasks for effective attack execution, especially for more capable models. We discover that while integrating adversarial data does not markedly degrade overall model performance on medical benchmarks, it does lead to noticeable shifts in fine-tuned model weights, suggesting a potential pathway for detecting and countering model attacks. This research highlights the urgent need for robust security measures and the development of defensive mechanisms to safeguard LLMs in medical applications, to ensure their safe and effective deployment in healthcare settings.



## **9. CleanGen: Mitigating Backdoor Attacks for Generation Tasks in Large Language Models**

cs.AI

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.12257v1) [paper-pdf](http://arxiv.org/pdf/2406.12257v1)

**Authors**: Yuetai Li, Zhangchen Xu, Fengqing Jiang, Luyao Niu, Dinuka Sahabandu, Bhaskar Ramasubramanian, Radha Poovendran

**Abstract**: The remarkable performance of large language models (LLMs) in generation tasks has enabled practitioners to leverage publicly available models to power custom applications, such as chatbots and virtual assistants. However, the data used to train or fine-tune these LLMs is often undisclosed, allowing an attacker to compromise the data and inject backdoors into the models. In this paper, we develop a novel inference time defense, named CleanGen, to mitigate backdoor attacks for generation tasks in LLMs. CleanGenis a lightweight and effective decoding strategy that is compatible with the state-of-the-art (SOTA) LLMs. Our insight behind CleanGen is that compared to other LLMs, backdoored LLMs assign significantly higher probabilities to tokens representing the attacker-desired contents. These discrepancies in token probabilities enable CleanGen to identify suspicious tokens favored by the attacker and replace them with tokens generated by another LLM that is not compromised by the same attacker, thereby avoiding generation of attacker-desired content. We evaluate CleanGen against five SOTA backdoor attacks. Our results show that CleanGen achieves lower attack success rates (ASR) compared to five SOTA baseline defenses for all five backdoor attacks. Moreover, LLMs deploying CleanGen maintain helpfulness in their responses when serving benign user queries with minimal added computational overhead.



## **10. Privacy-Preserved Neural Graph Databases**

cs.DB

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2312.15591v5) [paper-pdf](http://arxiv.org/pdf/2312.15591v5)

**Authors**: Qi Hu, Haoran Li, Jiaxin Bai, Zihao Wang, Yangqiu Song

**Abstract**: In the era of large language models (LLMs), efficient and accurate data retrieval has become increasingly crucial for the use of domain-specific or private data in the retrieval augmented generation (RAG). Neural graph databases (NGDBs) have emerged as a powerful paradigm that combines the strengths of graph databases (GDBs) and neural networks to enable efficient storage, retrieval, and analysis of graph-structured data which can be adaptively trained with LLMs. The usage of neural embedding storage and Complex neural logical Query Answering (CQA) provides NGDBs with generalization ability. When the graph is incomplete, by extracting latent patterns and representations, neural graph databases can fill gaps in the graph structure, revealing hidden relationships and enabling accurate query answering. Nevertheless, this capability comes with inherent trade-offs, as it introduces additional privacy risks to the domain-specific or private databases. Malicious attackers can infer more sensitive information in the database using well-designed queries such as from the answer sets of where Turing Award winners born before 1950 and after 1940 lived, the living places of Turing Award winner Hinton are probably exposed, although the living places may have been deleted in the training stage due to the privacy concerns. In this work, we propose a privacy-preserved neural graph database (P-NGDB) framework to alleviate the risks of privacy leakage in NGDBs. We introduce adversarial training techniques in the training stage to enforce the NGDBs to generate indistinguishable answers when queried with private information, enhancing the difficulty of inferring sensitive information through combinations of multiple innocuous queries.



## **11. JailGuard: A Universal Detection Framework for LLM Prompt-based Attacks**

cs.CR

28 pages, 9 figures

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2312.10766v3) [paper-pdf](http://arxiv.org/pdf/2312.10766v3)

**Authors**: Xiaoyu Zhang, Cen Zhang, Tianlin Li, Yihao Huang, Xiaojun Jia, Ming Hu, Jie Zhang, Yang Liu, Shiqing Ma, Chao Shen

**Abstract**: Large Language Models (LLMs) and Multi-Modal LLMs (MLLMs) have played a critical role in numerous applications. However, current LLMs are vulnerable to prompt-based attacks, with jailbreaking attacks enabling LLMs to generate harmful content, while hijacking attacks manipulate the model to perform unintended tasks, underscoring the necessity for detection methods. Unfortunately, existing detecting approaches are usually tailored to specific attacks, resulting in poor generalization in detecting various attacks across different modalities. To address it, we propose JailGuard, a universal detection framework for jailbreaking and hijacking attacks across LLMs and MLLMs. JailGuard operates on the principle that attacks are inherently less robust than benign ones, regardless of method or modality. Specifically, JailGuard mutates untrusted inputs to generate variants and leverages the discrepancy of the variants' responses on the model to distinguish attack samples from benign samples. We implement 18 mutators for text and image inputs and design a mutator combination policy to further improve detection generalization. To evaluate the effectiveness of JailGuard, we build the first comprehensive multi-modal attack dataset, containing 11,000 data items across 15 known attack types. The evaluation suggests that JailGuard achieves the best detection accuracy of 86.14%/82.90% on text and image inputs, outperforming state-of-the-art methods by 11.81%-25.73% and 12.20%-21.40%.



## **12. Safety Fine-Tuning at (Almost) No Cost: A Baseline for Vision Large Language Models**

cs.LG

ICML 2024

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2402.02207v2) [paper-pdf](http://arxiv.org/pdf/2402.02207v2)

**Authors**: Yongshuo Zong, Ondrej Bohdal, Tingyang Yu, Yongxin Yang, Timothy Hospedales

**Abstract**: Current vision large language models (VLLMs) exhibit remarkable capabilities yet are prone to generate harmful content and are vulnerable to even the simplest jailbreaking attacks. Our initial analysis finds that this is due to the presence of harmful data during vision-language instruction fine-tuning, and that VLLM fine-tuning can cause forgetting of safety alignment previously learned by the underpinning LLM. To address this issue, we first curate a vision-language safe instruction-following dataset VLGuard covering various harmful categories. Our experiments demonstrate that integrating this dataset into standard vision-language fine-tuning or utilizing it for post-hoc fine-tuning effectively safety aligns VLLMs. This alignment is achieved with minimal impact on, or even enhancement of, the models' helpfulness. The versatility of our safety fine-tuning dataset makes it a valuable resource for safety-testing existing VLLMs, training new models or safeguarding pre-trained VLLMs. Empirical results demonstrate that fine-tuned VLLMs effectively reject unsafe instructions and substantially reduce the success rates of several black-box adversarial attacks, which approach zero in many cases. The code and dataset are available at https://github.com/ys-zong/VLGuard.



## **13. Is poisoning a real threat to LLM alignment? Maybe more so than you think**

cs.LG

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.12091v1) [paper-pdf](http://arxiv.org/pdf/2406.12091v1)

**Authors**: Pankayaraj Pathmanathan, Souradip Chakraborty, Xiangyu Liu, Yongyuan Liang, Furong Huang

**Abstract**: Recent advancements in Reinforcement Learning with Human Feedback (RLHF) have significantly impacted the alignment of Large Language Models (LLMs). The sensitivity of reinforcement learning algorithms such as Proximal Policy Optimization (PPO) has led to new line work on Direct Policy Optimization (DPO), which treats RLHF in a supervised learning framework. The increased practical use of these RLHF methods warrants an analysis of their vulnerabilities. In this work, we investigate the vulnerabilities of DPO to poisoning attacks under different scenarios and compare the effectiveness of preference poisoning, a first of its kind. We comprehensively analyze DPO's vulnerabilities under different types of attacks, i.e., backdoor and non-backdoor attacks, and different poisoning methods across a wide array of language models, i.e., LLama 7B, Mistral 7B, and Gemma 7B. We find that unlike PPO-based methods, which, when it comes to backdoor attacks, require at least 4\% of the data to be poisoned to elicit harmful behavior, we exploit the true vulnerabilities of DPO more simply so we can poison the model with only as much as 0.5\% of the data. We further investigate the potential reasons behind the vulnerability and how well this vulnerability translates into backdoor vs non-backdoor attacks.



## **14. MLLM-Protector: Ensuring MLLM's Safety without Hurting Performance**

cs.CR

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2401.02906v3) [paper-pdf](http://arxiv.org/pdf/2401.02906v3)

**Authors**: Renjie Pi, Tianyang Han, Jianshu Zhang, Yueqi Xie, Rui Pan, Qing Lian, Hanze Dong, Jipeng Zhang, Tong Zhang

**Abstract**: The deployment of multimodal large language models (MLLMs) has brought forth a unique vulnerability: susceptibility to malicious attacks through visual inputs. This paper investigates the novel challenge of defending MLLMs against such attacks. Compared to large language models (LLMs), MLLMs include an additional image modality. We discover that images act as a ``foreign language" that is not considered during safety alignment, making MLLMs more prone to producing harmful responses. Unfortunately, unlike the discrete tokens considered in text-based LLMs, the continuous nature of image signals presents significant alignment challenges, which poses difficulty to thoroughly cover all possible scenarios. This vulnerability is exacerbated by the fact that most state-of-the-art MLLMs are fine-tuned on limited image-text pairs that are much fewer than the extensive text-based pretraining corpus, which makes the MLLMs more prone to catastrophic forgetting of their original abilities during safety fine-tuning. To tackle these challenges, we introduce MLLM-Protector, a plug-and-play strategy that solves two subtasks: 1) identifying harmful responses via a lightweight harm detector, and 2) transforming harmful responses into harmless ones via a detoxifier. This approach effectively mitigates the risks posed by malicious visual inputs without compromising the original performance of MLLMs. Our results demonstrate that MLLM-Protector offers a robust solution to a previously unaddressed aspect of MLLM security.



## **15. Knowledge-to-Jailbreak: One Knowledge Point Worth One Attack**

cs.CL

18 pages, 14 figures, 11 tables

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11682v1) [paper-pdf](http://arxiv.org/pdf/2406.11682v1)

**Authors**: Shangqing Tu, Zhuoran Pan, Wenxuan Wang, Zhexin Zhang, Yuliang Sun, Jifan Yu, Hongning Wang, Lei Hou, Juanzi Li

**Abstract**: Large language models (LLMs) have been increasingly applied to various domains, which triggers increasing concerns about LLMs' safety on specialized domains, e.g. medicine. However, testing the domain-specific safety of LLMs is challenging due to the lack of domain knowledge-driven attacks in existing benchmarks. To bridge this gap, we propose a new task, knowledge-to-jailbreak, which aims to generate jailbreaks from domain knowledge to evaluate the safety of LLMs when applied to those domains. We collect a large-scale dataset with 12,974 knowledge-jailbreak pairs and fine-tune a large language model as jailbreak-generator, to produce domain knowledge-specific jailbreaks. Experiments on 13 domains and 8 target LLMs demonstrate the effectiveness of jailbreak-generator in generating jailbreaks that are both relevant to the given knowledge and harmful to the target LLMs. We also apply our method to an out-of-domain knowledge base, showing that jailbreak-generator can generate jailbreaks that are comparable in harmfulness to those crafted by human experts. Data and code: https://github.com/THU-KEG/Knowledge-to-Jailbreak/.



## **16. Bileve: Securing Text Provenance in Large Language Models Against Spoofing with Bi-level Signature**

cs.CR

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.01946v2) [paper-pdf](http://arxiv.org/pdf/2406.01946v2)

**Authors**: Tong Zhou, Xuandong Zhao, Xiaolin Xu, Shaolei Ren

**Abstract**: Text watermarks for large language models (LLMs) have been commonly used to identify the origins of machine-generated content, which is promising for assessing liability when combating deepfake or harmful content. While existing watermarking techniques typically prioritize robustness against removal attacks, unfortunately, they are vulnerable to spoofing attacks: malicious actors can subtly alter the meanings of LLM-generated responses or even forge harmful content, potentially misattributing blame to the LLM developer. To overcome this, we introduce a bi-level signature scheme, Bileve, which embeds fine-grained signature bits for integrity checks (mitigating spoofing attacks) as well as a coarse-grained signal to trace text sources when the signature is invalid (enhancing detectability) via a novel rank-based sampling strategy. Compared to conventional watermark detectors that only output binary results, Bileve can differentiate 5 scenarios during detection, reliably tracing text provenance and regulating LLMs. The experiments conducted on OPT-1.3B and LLaMA-7B demonstrate the effectiveness of Bileve in defeating spoofing attacks with enhanced detectability.



## **17. ART: Automatic Red-teaming for Text-to-Image Models to Protect Benign Users**

cs.CR

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2405.19360v2) [paper-pdf](http://arxiv.org/pdf/2405.19360v2)

**Authors**: Guanlin Li, Kangjie Chen, Shudong Zhang, Jie Zhang, Tianwei Zhang

**Abstract**: Large-scale pre-trained generative models are taking the world by storm, due to their abilities in generating creative content. Meanwhile, safeguards for these generative models are developed, to protect users' rights and safety, most of which are designed for large language models. Existing methods primarily focus on jailbreak and adversarial attacks, which mainly evaluate the model's safety under malicious prompts. Recent work found that manually crafted safe prompts can unintentionally trigger unsafe generations. To further systematically evaluate the safety risks of text-to-image models, we propose a novel Automatic Red-Teaming framework, ART. Our method leverages both vision language model and large language model to establish a connection between unsafe generations and their prompts, thereby more efficiently identifying the model's vulnerabilities. With our comprehensive experiments, we reveal the toxicity of the popular open-source text-to-image models. The experiments also validate the effectiveness, adaptability, and great diversity of ART. Additionally, we introduce three large-scale red-teaming datasets for studying the safety risks associated with text-to-image models. Datasets and models can be found in https://github.com/GuanlinLee/ART.



## **18. $\texttt{MoE-RBench}$: Towards Building Reliable Language Models with Sparse Mixture-of-Experts**

cs.LG

9 pages, 8 figures, camera ready on ICML2024

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11353v1) [paper-pdf](http://arxiv.org/pdf/2406.11353v1)

**Authors**: Guanjie Chen, Xinyu Zhao, Tianlong Chen, Yu Cheng

**Abstract**: Mixture-of-Experts (MoE) has gained increasing popularity as a promising framework for scaling up large language models (LLMs). However, the reliability assessment of MoE lags behind its surging applications. Moreover, when transferred to new domains such as in fine-tuning MoE models sometimes underperform their dense counterparts. Motivated by the research gap and counter-intuitive phenomenon, we propose $\texttt{MoE-RBench}$, the first comprehensive assessment of SMoE reliability from three aspects: $\textit{(i)}$ safety and hallucination, $\textit{(ii)}$ resilience to adversarial attacks, and $\textit{(iii)}$ out-of-distribution robustness. Extensive models and datasets are tested to compare the MoE to dense networks from these reliability dimensions. Our empirical observations suggest that with appropriate hyperparameters, training recipes, and inference techniques, we can build the MoE model more reliably than the dense LLM. In particular, we find that the robustness of SMoE is sensitive to the basic training settings. We hope that this study can provide deeper insights into how to adapt the pre-trained MoE model to other tasks with higher-generation security, quality, and stability. Codes are available at https://github.com/UNITES-Lab/MoE-RBench



## **19. Adversarial Style Augmentation via Large Language Model for Robust Fake News Detection**

cs.CL

8 pages

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11260v1) [paper-pdf](http://arxiv.org/pdf/2406.11260v1)

**Authors**: Sungwon Park, Sungwon Han, Meeyoung Cha

**Abstract**: The spread of fake news negatively impacts individuals and is regarded as a significant social challenge that needs to be addressed. A number of algorithmic and insightful features have been identified for detecting fake news. However, with the recent LLMs and their advanced generation capabilities, many of the detectable features (e.g., style-conversion attacks) can be altered, making it more challenging to distinguish from real news. This study proposes adversarial style augmentation, AdStyle, to train a fake news detector that remains robust against various style-conversion attacks. Our model's key mechanism is the careful use of LLMs to automatically generate a diverse yet coherent range of style-conversion attack prompts. This improves the generation of prompts that are particularly difficult for the detector to handle. Experiments show that our augmentation strategy improves robustness and detection performance when tested on fake news benchmark datasets.



## **20. Evading AI-Generated Content Detectors using Homoglyphs**

cs.CL

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11239v1) [paper-pdf](http://arxiv.org/pdf/2406.11239v1)

**Authors**: Aldan Creo, Shushanta Pudasaini

**Abstract**: The generation of text that is increasingly human-like has been enabled by the advent of large language models (LLMs). As the detection of AI-generated content holds significant importance in the fight against issues such as misinformation and academic cheating, numerous studies have been conducted to develop reliable LLM detectors. While promising results have been demonstrated by such detectors on test data, recent research has revealed that they can be circumvented by employing different techniques. In this article, homoglyph-based ($a \rightarrow {\alpha}$) attacks that can be used to circumvent existing LLM detectors are presented. The efficacy of the attacks is illustrated by analizing how homoglyphs shift the tokenization of the text, and thus its token loglikelihoods. A comprehensive evaluation is conducted to assess the effectiveness of homoglyphs on state-of-the-art LLM detectors, including Binoculars, DetectGPT, OpenAI's detector, and watermarking techniques, on five different datasets. A significant reduction in the efficiency of all the studied configurations of detectors and datasets, down to an accuracy of 0.5 (random guessing), is demonstrated by the proposed approach. The results show that homoglyph-based attacks can effectively evade existing LLM detectors, and the implications of these findings are discussed along with possible defenses against such attacks.



## **21. GoldCoin: Grounding Large Language Models in Privacy Laws via Contextual Integrity Theory**

cs.CL

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11149v1) [paper-pdf](http://arxiv.org/pdf/2406.11149v1)

**Authors**: Wei Fan, Haoran Li, Zheye Deng, Weiqi Wang, Yangqiu Song

**Abstract**: Privacy issues arise prominently during the inappropriate transmission of information between entities. Existing research primarily studies privacy by exploring various privacy attacks, defenses, and evaluations within narrowly predefined patterns, while neglecting that privacy is not an isolated, context-free concept limited to traditionally sensitive data (e.g., social security numbers), but intertwined with intricate social contexts that complicate the identification and analysis of potential privacy violations. The advent of Large Language Models (LLMs) offers unprecedented opportunities for incorporating the nuanced scenarios outlined in privacy laws to tackle these complex privacy issues. However, the scarcity of open-source relevant case studies restricts the efficiency of LLMs in aligning with specific legal statutes. To address this challenge, we introduce a novel framework, GoldCoin, designed to efficiently ground LLMs in privacy laws for judicial assessing privacy violations. Our framework leverages the theory of contextual integrity as a bridge, creating numerous synthetic scenarios grounded in relevant privacy statutes (e.g., HIPAA), to assist LLMs in comprehending the complex contexts for identifying privacy risks in the real world. Extensive experimental results demonstrate that GoldCoin markedly enhances LLMs' capabilities in recognizing privacy risks across real court cases, surpassing the baselines on different judicial tasks.



## **22. Highlighting the Safety Concerns of Deploying LLMs/VLMs in Robotics**

cs.RO

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2402.10340v4) [paper-pdf](http://arxiv.org/pdf/2402.10340v4)

**Authors**: Xiyang Wu, Souradip Chakraborty, Ruiqi Xian, Jing Liang, Tianrui Guan, Fuxiao Liu, Brian M. Sadler, Dinesh Manocha, Amrit Singh Bedi

**Abstract**: In this paper, we highlight the critical issues of robustness and safety associated with integrating large language models (LLMs) and vision-language models (VLMs) into robotics applications. Recent works focus on using LLMs and VLMs to improve the performance of robotics tasks, such as manipulation and navigation. Despite these improvements, analyzing the safety of such systems remains underexplored yet extremely critical. LLMs and VLMs are highly susceptible to adversarial inputs, prompting a significant inquiry into the safety of robotic systems. This concern is important because robotics operate in the physical world where erroneous actions can result in severe consequences. This paper explores this issue thoroughly, presenting a mathematical formulation of potential attacks on LLM/VLM-based robotic systems and offering experimental evidence of the safety challenges. Our empirical findings highlight a significant vulnerability: simple modifications to the input can drastically reduce system effectiveness. Specifically, our results demonstrate an average performance deterioration of 19.4% under minor input prompt modifications and a more alarming 29.1% under slight perceptual changes. These findings underscore the urgent need for robust countermeasures to ensure the safe and reliable deployment of advanced LLM/VLM-based robotic systems.



## **23. garak: A Framework for Security Probing Large Language Models**

cs.CL

https://garak.ai

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2406.11036v1) [paper-pdf](http://arxiv.org/pdf/2406.11036v1)

**Authors**: Leon Derczynski, Erick Galinkin, Jeffrey Martin, Subho Majumdar, Nanna Inie

**Abstract**: As Large Language Models (LLMs) are deployed and integrated into thousands of applications, the need for scalable evaluation of how models respond to adversarial attacks grows rapidly. However, LLM security is a moving target: models produce unpredictable output, are constantly updated, and the potential adversary is highly diverse: anyone with access to the internet and a decent command of natural language. Further, what constitutes a security weak in one context may not be an issue in a different context; one-fits-all guardrails remain theoretical. In this paper, we argue that it is time to rethink what constitutes ``LLM security'', and pursue a holistic approach to LLM security evaluation, where exploration and discovery of issues are central. To this end, this paper introduces garak (Generative AI Red-teaming and Assessment Kit), a framework which can be used to discover and identify vulnerabilities in a target LLM or dialog system. garak probes an LLM in a structured fashion to discover potential vulnerabilities. The outputs of the framework describe a target model's weaknesses, contribute to an informed discussion of what composes vulnerabilities in unique contexts, and can inform alignment and policy discussions for LLM deployment.



## **24. Threat Modelling and Risk Analysis for Large Language Model (LLM)-Powered Applications**

cs.CR

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2406.11007v1) [paper-pdf](http://arxiv.org/pdf/2406.11007v1)

**Authors**: Stephen Burabari Tete

**Abstract**: The advent of Large Language Models (LLMs) has revolutionized various applications by providing advanced natural language processing capabilities. However, this innovation introduces new cybersecurity challenges. This paper explores the threat modeling and risk analysis specifically tailored for LLM-powered applications. Focusing on potential attacks like data poisoning, prompt injection, SQL injection, jailbreaking, and compositional injection, we assess their impact on security and propose mitigation strategies. We introduce a framework combining STRIDE and DREAD methodologies for proactive threat identification and risk assessment. Furthermore, we examine the feasibility of an end-to-end threat model through a case study of a custom-built LLM-powered application. This model follows Shostack's Four Question Framework, adjusted for the unique threats LLMs present. Our goal is to propose measures that enhance the security of these powerful AI tools, thwarting attacks, and ensuring the reliability and integrity of LLM-integrated systems.



## **25. JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models**

cs.CR

JailbreakBench v1.0: more attack artifacts, more test-time defenses,  a more accurate jailbreak judge (Llama-3-70B with a custom prompt), a larger  dataset of human preferences for selecting a jailbreak judge (300 examples),  an over-refusal evaluation dataset (100 benign/borderline behaviors), a  semantic refusal judge based on Llama-3-8B

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2404.01318v3) [paper-pdf](http://arxiv.org/pdf/2404.01318v3)

**Authors**: Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramer, Hamed Hassani, Eric Wong

**Abstract**: Jailbreak attacks cause large language models (LLMs) to generate harmful, unethical, or otherwise objectionable content. Evaluating these attacks presents a number of challenges, which the current collection of benchmarks and evaluation techniques do not adequately address. First, there is no clear standard of practice regarding jailbreaking evaluation. Second, existing works compute costs and success rates in incomparable ways. And third, numerous works are not reproducible, as they withhold adversarial prompts, involve closed-source code, or rely on evolving proprietary APIs. To address these challenges, we introduce JailbreakBench, an open-sourced benchmark with the following components: (1) an evolving repository of state-of-the-art adversarial prompts, which we refer to as jailbreak artifacts; (2) a jailbreaking dataset comprising 100 behaviors -- both original and sourced from prior work -- which align with OpenAI's usage policies; (3) a standardized evaluation framework at https://github.com/JailbreakBench/jailbreakbench that includes a clearly defined threat model, system prompts, chat templates, and scoring functions; and (4) a leaderboard at https://jailbreakbench.github.io/ that tracks the performance of attacks and defenses for various LLMs. We have carefully considered the potential ethical implications of releasing this benchmark, and believe that it will be a net positive for the community.



## **26. ATM: Adversarial Tuning Multi-agent System Makes a Robust Retrieval-Augmented Generator**

cs.CL

18 pages, 7 figures

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2405.18111v2) [paper-pdf](http://arxiv.org/pdf/2405.18111v2)

**Authors**: Junda Zhu, Lingyong Yan, Haibo Shi, Dawei Yin, Lei Sha

**Abstract**: Large language models (LLMs) are proven to benefit a lot from retrieval-augmented generation (RAG) in alleviating hallucinations confronted with knowledge-intensive questions. RAG adopts information retrieval techniques to inject external knowledge from semantic-relevant documents as input contexts. However, due to today's Internet being flooded with numerous noisy and fabricating content, it is inevitable that RAG systems are vulnerable to these noises and prone to respond incorrectly. To this end, we propose to optimize the retrieval-augmented Generator with a Adversarial Tuning Multi-agent system (ATM). The ATM steers the Generator to have a robust perspective of useful documents for question answering with the help of an auxiliary Attacker agent. The Generator and the Attacker are tuned adversarially for several iterations. After rounds of multi-agent iterative tuning, the Generator can eventually better discriminate useful documents amongst fabrications. The experimental results verify the effectiveness of ATM and we also observe that the Generator can achieve better performance compared to state-of-the-art baselines.



## **27. RWKU: Benchmarking Real-World Knowledge Unlearning for Large Language Models**

cs.CL

48 pages, 7 figures, 12 tables

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2406.10890v1) [paper-pdf](http://arxiv.org/pdf/2406.10890v1)

**Authors**: Zhuoran Jin, Pengfei Cao, Chenhao Wang, Zhitao He, Hongbang Yuan, Jiachun Li, Yubo Chen, Kang Liu, Jun Zhao

**Abstract**: Large language models (LLMs) inevitably memorize sensitive, copyrighted, and harmful knowledge from the training corpus; therefore, it is crucial to erase this knowledge from the models. Machine unlearning is a promising solution for efficiently removing specific knowledge by post hoc modifying models. In this paper, we propose a Real-World Knowledge Unlearning benchmark (RWKU) for LLM unlearning. RWKU is designed based on the following three key factors: (1) For the task setting, we consider a more practical and challenging unlearning setting, where neither the forget corpus nor the retain corpus is accessible. (2) For the knowledge source, we choose 200 real-world famous people as the unlearning targets and show that such popular knowledge is widely present in various LLMs. (3) For the evaluation framework, we design the forget set and the retain set to evaluate the model's capabilities across various real-world applications. Regarding the forget set, we provide four four membership inference attack (MIA) methods and nine kinds of adversarial attack probes to rigorously test unlearning efficacy. Regarding the retain set, we assess locality and utility in terms of neighbor perturbation, general ability, reasoning ability, truthfulness, factuality, and fluency. We conduct extensive experiments across two unlearning scenarios, two models and six baseline methods and obtain some meaningful findings. We release our benchmark and code publicly at http://rwku-bench.github.io for future work.



## **28. KGPA: Robustness Evaluation for Large Language Models via Cross-Domain Knowledge Graphs**

cs.CL

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2406.10802v1) [paper-pdf](http://arxiv.org/pdf/2406.10802v1)

**Authors**: Aihua Pei, Zehua Yang, Shunan Zhu, Ruoxi Cheng, Ju Jia, Lina Wang

**Abstract**: Existing frameworks for assessing robustness of large language models (LLMs) overly depend on specific benchmarks, increasing costs and failing to evaluate performance of LLMs in professional domains due to dataset limitations. This paper proposes a framework that systematically evaluates the robustness of LLMs under adversarial attack scenarios by leveraging knowledge graphs (KGs). Our framework generates original prompts from the triplets of knowledge graphs and creates adversarial prompts by poisoning, assessing the robustness of LLMs through the results of these adversarial attacks. We systematically evaluate the effectiveness of this framework and its modules. Experiments show that adversarial robustness of the ChatGPT family ranks as GPT-4-turbo > GPT-4o > GPT-3.5-turbo, and the robustness of large language models is influenced by the professional domains in which they operate.



## **29. Towards Understanding Jailbreak Attacks in LLMs: A Representation Space Analysis**

cs.CL

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2406.10794v1) [paper-pdf](http://arxiv.org/pdf/2406.10794v1)

**Authors**: Yuping Lin, Pengfei He, Han Xu, Yue Xing, Makoto Yamada, Hui Liu, Jiliang Tang

**Abstract**: Large language models (LLMs) are susceptible to a type of attack known as jailbreaking, which misleads LLMs to output harmful contents. Although there are diverse jailbreak attack strategies, there is no unified understanding on why some methods succeed and others fail. This paper explores the behavior of harmful and harmless prompts in the LLM's representation space to investigate the intrinsic properties of successful jailbreak attacks. We hypothesize that successful attacks share some similar properties: They are effective in moving the representation of the harmful prompt towards the direction to the harmless prompts. We leverage hidden representations into the objective of existing jailbreak attacks to move the attacks along the acceptance direction, and conduct experiments to validate the above hypothesis using the proposed objective. We hope this study provides new insights into understanding how LLMs understand harmfulness information.



## **30. Adversarial Math Word Problem Generation**

cs.CL

Code/data: https://github.com/ruoyuxie/adversarial_mwps_generation

**SubmitDate**: 2024-06-15    [abs](http://arxiv.org/abs/2402.17916v3) [paper-pdf](http://arxiv.org/pdf/2402.17916v3)

**Authors**: Roy Xie, Chengxuan Huang, Junlin Wang, Bhuwan Dhingra

**Abstract**: Large language models (LLMs) have significantly transformed the educational landscape. As current plagiarism detection tools struggle to keep pace with LLMs' rapid advancements, the educational community faces the challenge of assessing students' true problem-solving abilities in the presence of LLMs. In this work, we explore a new paradigm for ensuring fair evaluation -- generating adversarial examples which preserve the structure and difficulty of the original questions aimed for assessment, but are unsolvable by LLMs. Focusing on the domain of math word problems, we leverage abstract syntax trees to structurally generate adversarial examples that cause LLMs to produce incorrect answers by simply editing the numeric values in the problems. We conduct experiments on various open- and closed-source LLMs, quantitatively and qualitatively demonstrating that our method significantly degrades their math problem-solving ability. We identify shared vulnerabilities among LLMs and propose a cost-effective approach to attack high-cost models. Additionally, we conduct automatic analysis to investigate the cause of failure, providing further insights into the limitations of LLMs.



## **31. Emerging Safety Attack and Defense in Federated Instruction Tuning of Large Language Models**

cs.CL

18 pages

**SubmitDate**: 2024-06-15    [abs](http://arxiv.org/abs/2406.10630v1) [paper-pdf](http://arxiv.org/pdf/2406.10630v1)

**Authors**: Rui Ye, Jingyi Chai, Xiangrui Liu, Yaodong Yang, Yanfeng Wang, Siheng Chen

**Abstract**: Federated learning (FL) enables multiple parties to collaboratively fine-tune an large language model (LLM) without the need of direct data sharing. Ideally, by training on decentralized data that is aligned with human preferences and safety principles, federated instruction tuning can result in an LLM that could behave in a helpful and safe manner. In this paper, we for the first time reveal the vulnerability of safety alignment in FedIT by proposing a simple, stealthy, yet effective safety attack method. Specifically, the malicious clients could automatically generate attack data without involving manual efforts and attack the FedIT system by training their local LLMs on such attack data. Unfortunately, this proposed safety attack not only can compromise the safety alignment of LLM trained via FedIT, but also can not be effectively defended against by many existing FL defense methods. Targeting this, we further propose a post-hoc defense method, which could rely on a fully automated pipeline: generation of defense data and further fine-tuning of the LLM. Extensive experiments show that our safety attack method can significantly compromise the LLM's safety alignment (e.g., reduce safety rate by 70\%), which can not be effectively defended by existing defense methods (at most 4\% absolute improvement), while our safety defense method can significantly enhance the attacked LLM's safety alignment (at most 69\% absolute improvement).



## **32. KnowPhish: Large Language Models Meet Multimodal Knowledge Graphs for Enhancing Reference-Based Phishing Detection**

cs.CR

Accepted by USENIX Security 2024

**SubmitDate**: 2024-06-15    [abs](http://arxiv.org/abs/2403.02253v2) [paper-pdf](http://arxiv.org/pdf/2403.02253v2)

**Authors**: Yuexin Li, Chengyu Huang, Shumin Deng, Mei Lin Lock, Tri Cao, Nay Oo, Hoon Wei Lim, Bryan Hooi

**Abstract**: Phishing attacks have inflicted substantial losses on individuals and businesses alike, necessitating the development of robust and efficient automated phishing detection approaches. Reference-based phishing detectors (RBPDs), which compare the logos on a target webpage to a known set of logos, have emerged as the state-of-the-art approach. However, a major limitation of existing RBPDs is that they rely on a manually constructed brand knowledge base, making it infeasible to scale to a large number of brands, which results in false negative errors due to the insufficient brand coverage of the knowledge base. To address this issue, we propose an automated knowledge collection pipeline, using which we collect a large-scale multimodal brand knowledge base, KnowPhish, containing 20k brands with rich information about each brand. KnowPhish can be used to boost the performance of existing RBPDs in a plug-and-play manner. A second limitation of existing RBPDs is that they solely rely on the image modality, ignoring useful textual information present in the webpage HTML. To utilize this textual information, we propose a Large Language Model (LLM)-based approach to extract brand information of webpages from text. Our resulting multimodal phishing detection approach, KnowPhish Detector (KPD), can detect phishing webpages with or without logos. We evaluate KnowPhish and KPD on a manually validated dataset, and a field study under Singapore's local context, showing substantial improvements in effectiveness and efficiency compared to state-of-the-art baselines.



## **33. Semantic Membership Inference Attack against Large Language Models**

cs.LG

**SubmitDate**: 2024-06-14    [abs](http://arxiv.org/abs/2406.10218v1) [paper-pdf](http://arxiv.org/pdf/2406.10218v1)

**Authors**: Hamid Mozaffari, Virendra J. Marathe

**Abstract**: Membership Inference Attacks (MIAs) determine whether a specific data point was included in the training set of a target model. In this paper, we introduce the Semantic Membership Inference Attack (SMIA), a novel approach that enhances MIA performance by leveraging the semantic content of inputs and their perturbations. SMIA trains a neural network to analyze the target model's behavior on perturbed inputs, effectively capturing variations in output probability distributions between members and non-members. We conduct comprehensive evaluations on the Pythia and GPT-Neo model families using the Wikipedia dataset. Our results show that SMIA significantly outperforms existing MIAs; for instance, SMIA achieves an AUC-ROC of 67.39% on Pythia-12B, compared to 58.90% by the second-best attack.



## **34. Defending Large Language Models Against Jailbreak Attacks via Layer-specific Editing**

cs.AI

**SubmitDate**: 2024-06-14    [abs](http://arxiv.org/abs/2405.18166v2) [paper-pdf](http://arxiv.org/pdf/2405.18166v2)

**Authors**: Wei Zhao, Zhe Li, Yige Li, Ye Zhang, Jun Sun

**Abstract**: Large language models (LLMs) are increasingly being adopted in a wide range of real-world applications. Despite their impressive performance, recent studies have shown that LLMs are vulnerable to deliberately crafted adversarial prompts even when aligned via Reinforcement Learning from Human Feedback or supervised fine-tuning. While existing defense methods focus on either detecting harmful prompts or reducing the likelihood of harmful responses through various means, defending LLMs against jailbreak attacks based on the inner mechanisms of LLMs remains largely unexplored. In this work, we investigate how LLMs response to harmful prompts and propose a novel defense method termed \textbf{L}ayer-specific \textbf{Ed}iting (LED) to enhance the resilience of LLMs against jailbreak attacks. Through LED, we reveal that several critical \textit{safety layers} exist among the early layers of LLMs. We then show that realigning these safety layers (and some selected additional layers) with the decoded safe response from selected target layers can significantly improve the alignment of LLMs against jailbreak attacks. Extensive experiments across various LLMs (e.g., Llama2, Mistral) show the effectiveness of LED, which effectively defends against jailbreak attacks while maintaining performance on benign prompts. Our code is available at \url{https://github.com/ledllm/ledllm}.



## **35. REVS: Unlearning Sensitive Information in Language Models via Rank Editing in the Vocabulary Space**

cs.CL

18 pages, 3 figures

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.09325v1) [paper-pdf](http://arxiv.org/pdf/2406.09325v1)

**Authors**: Tomer Ashuach, Martin Tutek, Yonatan Belinkov

**Abstract**: Large language models (LLMs) risk inadvertently memorizing and divulging sensitive or personally identifiable information (PII) seen in training data, causing privacy concerns. Current approaches to address this issue involve costly dataset scrubbing, or model filtering through unlearning and model editing, which can be bypassed through extraction attacks. We propose REVS, a novel model editing method for unlearning sensitive information from LLMs. REVS identifies and modifies a small subset of neurons relevant for each piece of sensitive information. By projecting these neurons to the vocabulary space (unembedding), we pinpoint the components driving its generation. We then compute a model edit based on the pseudo-inverse of the unembedding matrix, and apply it to de-promote generation of the targeted sensitive data. To adequately evaluate our method on truly sensitive information, we curate two datasets: an email dataset inherently memorized by GPT-J, and a synthetic social security number dataset that we tune the model to memorize. Compared to other state-of-the-art model editing methods, REVS demonstrates superior performance in both eliminating sensitive information and robustness to extraction attacks, while retaining integrity of the underlying model. The code and a demo notebook are available at https://technion-cs-nlp.github.io/REVS.



## **36. Bag of Tricks: Benchmarking of Jailbreak Attacks on LLMs**

cs.CR

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.09324v1) [paper-pdf](http://arxiv.org/pdf/2406.09324v1)

**Authors**: Zhao Xu, Fan Liu, Hao Liu

**Abstract**: Although Large Language Models (LLMs) have demonstrated significant capabilities in executing complex tasks in a zero-shot manner, they are susceptible to jailbreak attacks and can be manipulated to produce harmful outputs. Recently, a growing body of research has categorized jailbreak attacks into token-level and prompt-level attacks. However, previous work primarily overlooks the diverse key factors of jailbreak attacks, with most studies concentrating on LLM vulnerabilities and lacking exploration of defense-enhanced LLMs. To address these issues, we evaluate the impact of various attack settings on LLM performance and provide a baseline benchmark for jailbreak attacks, encouraging the adoption of a standardized evaluation framework. Specifically, we evaluate the eight key factors of implementing jailbreak attacks on LLMs from both target-level and attack-level perspectives. We further conduct seven representative jailbreak attacks on six defense methods across two widely used datasets, encompassing approximately 320 experiments with about 50,000 GPU hours on A800-80G. Our experimental results highlight the need for standardized benchmarking to evaluate these attacks on defense-enhanced LLMs. Our code is available at https://github.com/usail-hkust/Bag_of_Tricks_for_LLM_Jailbreaking.



## **37. JailbreakEval: An Integrated Toolkit for Evaluating Jailbreak Attempts Against Large Language Models**

cs.CR

Our code is available at https://github.com/ThuCCSLab/JailbreakEval

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.09321v1) [paper-pdf](http://arxiv.org/pdf/2406.09321v1)

**Authors**: Delong Ran, Jinyuan Liu, Yichen Gong, Jingyi Zheng, Xinlei He, Tianshuo Cong, Anyu Wang

**Abstract**: Jailbreak attacks aim to induce Large Language Models (LLMs) to generate harmful responses for forbidden instructions, presenting severe misuse threats to LLMs. Up to now, research into jailbreak attacks and defenses is emerging, however, there is (surprisingly) no consensus on how to evaluate whether a jailbreak attempt is successful. In other words, the methods to assess the harmfulness of an LLM's response are varied, such as manual annotation or prompting GPT-4 in specific ways. Each approach has its own set of strengths and weaknesses, impacting their alignment with human values, as well as the time and financial cost. This diversity in evaluation presents challenges for researchers in choosing suitable evaluation methods and conducting fair comparisons across different jailbreak attacks and defenses. In this paper, we conduct a comprehensive analysis of jailbreak evaluation methodologies, drawing from nearly ninety jailbreak research released between May 2023 and April 2024. Our study introduces a systematic taxonomy of jailbreak evaluators, offering in-depth insights into their strengths and weaknesses, along with the current status of their adaptation. Moreover, to facilitate subsequent research, we propose JailbreakEval, a user-friendly toolkit focusing on the evaluation of jailbreak attempts. It includes various well-known evaluators out-of-the-box, so that users can obtain evaluation results with only a single command. JailbreakEval also allows users to customize their own evaluation workflow in a unified framework with the ease of development and comparison. In summary, we regard JailbreakEval to be a catalyst that simplifies the evaluation process in jailbreak research and fosters an inclusive standard for jailbreak evaluation within the community.



## **38. A Survey of Backdoor Attacks and Defenses on Large Language Models: Implications for Security Measures**

cs.CR

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.06852v2) [paper-pdf](http://arxiv.org/pdf/2406.06852v2)

**Authors**: Shuai Zhao, Meihuizi Jia, Zhongliang Guo, Leilei Gan, Jie Fu, Yichao Feng, Fengjun Pan, Luu Anh Tuan

**Abstract**: The large language models (LLMs), which bridge the gap between human language understanding and complex problem-solving, achieve state-of-the-art performance on several NLP tasks, particularly in few-shot and zero-shot settings. Despite the demonstrable efficacy of LMMs, due to constraints on computational resources, users have to engage with open-source language models or outsource the entire training process to third-party platforms. However, research has demonstrated that language models are susceptible to potential security vulnerabilities, particularly in backdoor attacks. Backdoor attacks are designed to introduce targeted vulnerabilities into language models by poisoning training samples or model weights, allowing attackers to manipulate model responses through malicious triggers. While existing surveys on backdoor attacks provide a comprehensive overview, they lack an in-depth examination of backdoor attacks specifically targeting LLMs. To bridge this gap and grasp the latest trends in the field, this paper presents a novel perspective on backdoor attacks for LLMs by focusing on fine-tuning methods. Specifically, we systematically classify backdoor attacks into three categories: full-parameter fine-tuning, parameter-efficient fine-tuning, and attacks without fine-tuning. Based on insights from a substantial review, we also discuss crucial issues for future research on backdoor attacks, such as further exploring attack algorithms that do not require fine-tuning, or developing more covert attack algorithms.



## **39. StructuralSleight: Automated Jailbreak Attacks on Large Language Models Utilizing Uncommon Text-Encoded Structure**

cs.CL

12 pages, 4 figures

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.08754v1) [paper-pdf](http://arxiv.org/pdf/2406.08754v1)

**Authors**: Bangxin Li, Hengrui Xing, Chao Huang, Jin Qian, Huangqing Xiao, Linfeng Feng, Cong Tian

**Abstract**: Large Language Models (LLMs) are widely used in natural language processing but face the risk of jailbreak attacks that maliciously induce them to generate harmful content. Existing jailbreak attacks, including character-level and context-level attacks, mainly focus on the prompt of the plain text without specifically exploring the significant influence of its structure. In this paper, we focus on studying how prompt structure contributes to the jailbreak attack. We introduce a novel structure-level attack method based on tail structures that are rarely used during LLM training, which we refer to as Uncommon Text-Encoded Structure (UTES). We extensively study 12 UTESs templates and 6 obfuscation methods to build an effective automated jailbreak tool named StructuralSleight that contains three escalating attack strategies: Structural Attack, Structural and Character/Context Obfuscation Attack, and Fully Obfuscated Structural Attack. Extensive experiments on existing LLMs show that StructuralSleight significantly outperforms baseline methods. In particular, the attack success rate reaches 94.62\% on GPT-4o, which has not been addressed by state-of-the-art techniques.



## **40. Ranking Manipulation for Conversational Search Engines**

cs.CL

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.03589v2) [paper-pdf](http://arxiv.org/pdf/2406.03589v2)

**Authors**: Samuel Pfrommer, Yatong Bai, Tanmay Gautam, Somayeh Sojoudi

**Abstract**: Major search engine providers are rapidly incorporating Large Language Model (LLM)-generated content in response to user queries. These conversational search engines operate by loading retrieved website text into the LLM context for summarization and interpretation. Recent research demonstrates that LLMs are highly vulnerable to jailbreaking and prompt injection attacks, which disrupt the safety and quality goals of LLMs using adversarial strings. This work investigates the impact of prompt injections on the ranking order of sources referenced by conversational search engines. To this end, we introduce a focused dataset of real-world consumer product websites and formalize conversational search ranking as an adversarial problem. Experimentally, we analyze conversational search rankings in the absence of adversarial injections and show that different LLMs vary significantly in prioritizing product name, document content, and context position. We then present a tree-of-attacks-based jailbreaking technique which reliably promotes low-ranked products. Importantly, these attacks transfer effectively to state-of-the-art conversational search engines such as perplexity.ai. Given the strong financial incentive for website owners to boost their search ranking, we argue that our problem formulation is of critical importance for future robustness work.



## **41. RL-JACK: Reinforcement Learning-powered Black-box Jailbreaking Attack against LLMs**

cs.CR

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.08725v1) [paper-pdf](http://arxiv.org/pdf/2406.08725v1)

**Authors**: Xuan Chen, Yuzhou Nie, Lu Yan, Yunshu Mao, Wenbo Guo, Xiangyu Zhang

**Abstract**: Modern large language model (LLM) developers typically conduct a safety alignment to prevent an LLM from generating unethical or harmful content. Recent studies have discovered that the safety alignment of LLMs can be bypassed by jailbreaking prompts. These prompts are designed to create specific conversation scenarios with a harmful question embedded. Querying an LLM with such prompts can mislead the model into responding to the harmful question. The stochastic and random nature of existing genetic methods largely limits the effectiveness and efficiency of state-of-the-art (SOTA) jailbreaking attacks. In this paper, we propose RL-JACK, a novel black-box jailbreaking attack powered by deep reinforcement learning (DRL). We formulate the generation of jailbreaking prompts as a search problem and design a novel RL approach to solve it. Our method includes a series of customized designs to enhance the RL agent's learning efficiency in the jailbreaking context. Notably, we devise an LLM-facilitated action space that enables diverse action variations while constraining the overall search space. We propose a novel reward function that provides meaningful dense rewards for the agent toward achieving successful jailbreaking. Through extensive evaluations, we demonstrate that RL-JACK is overall much more effective than existing jailbreaking attacks against six SOTA LLMs, including large open-source models and commercial models. We also show the RL-JACK's resiliency against three SOTA defenses and its transferability across different models. Finally, we validate the insensitivity of RL-JACK to the variations in key hyper-parameters.



## **42. Adversarial Evasion Attack Efficiency against Large Language Models**

cs.CL

9 pages, 1 table, 2 figures, DCAI 2024 conference

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2406.08050v1) [paper-pdf](http://arxiv.org/pdf/2406.08050v1)

**Authors**: João Vitorino, Eva Maia, Isabel Praça

**Abstract**: Large Language Models (LLMs) are valuable for text classification, but their vulnerabilities must not be disregarded. They lack robustness against adversarial examples, so it is pertinent to understand the impacts of different types of perturbations, and assess if those attacks could be replicated by common users with a small amount of perturbations and a small number of queries to a deployed LLM. This work presents an analysis of the effectiveness, efficiency, and practicality of three different types of adversarial attacks against five different LLMs in a sentiment classification task. The obtained results demonstrated the very distinct impacts of the word-level and character-level attacks. The word attacks were more effective, but the character and more constrained attacks were more practical and required a reduced number of perturbations and queries. These differences need to be considered during the development of adversarial defense strategies to train more robust LLMs for intelligent text classification applications.



## **43. Defending Large Language Models Against Jailbreaking Attacks Through Goal Prioritization**

cs.CL

ACL 2024 Main Conference

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2311.09096v2) [paper-pdf](http://arxiv.org/pdf/2311.09096v2)

**Authors**: Zhexin Zhang, Junxiao Yang, Pei Ke, Fei Mi, Hongning Wang, Minlie Huang

**Abstract**: While significant attention has been dedicated to exploiting weaknesses in LLMs through jailbreaking attacks, there remains a paucity of effort in defending against these attacks. We point out a pivotal factor contributing to the success of jailbreaks: the intrinsic conflict between the goals of being helpful and ensuring safety. Accordingly, we propose to integrate goal prioritization at both training and inference stages to counteract. Implementing goal prioritization during inference substantially diminishes the Attack Success Rate (ASR) of jailbreaking from 66.4% to 3.6% for ChatGPT. And integrating goal prioritization into model training reduces the ASR from 71.0% to 6.6% for Llama2-13B. Remarkably, even in scenarios where no jailbreaking samples are included during training, our approach slashes the ASR by half. Additionally, our findings reveal that while stronger LLMs face greater safety risks, they also possess a greater capacity to be steered towards defending against such attacks, both because of their stronger ability in instruction following. Our work thus contributes to the comprehension of jailbreaking attacks and defenses, and sheds light on the relationship between LLMs' capability and safety. Our code is available at \url{https://github.com/thu-coai/JailbreakDefense_GoalPriority}.



## **44. Dataset and Lessons Learned from the 2024 SaTML LLM Capture-the-Flag Competition**

cs.CR

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2406.07954v1) [paper-pdf](http://arxiv.org/pdf/2406.07954v1)

**Authors**: Edoardo Debenedetti, Javier Rando, Daniel Paleka, Silaghi Fineas Florin, Dragos Albastroiu, Niv Cohen, Yuval Lemberg, Reshmi Ghosh, Rui Wen, Ahmed Salem, Giovanni Cherubin, Santiago Zanella-Beguelin, Robin Schmid, Victor Klemm, Takahiro Miki, Chenhao Li, Stefan Kraft, Mario Fritz, Florian Tramèr, Sahar Abdelnabi, Lea Schönherr

**Abstract**: Large language model systems face important security risks from maliciously crafted messages that aim to overwrite the system's original instructions or leak private data. To study this problem, we organized a capture-the-flag competition at IEEE SaTML 2024, where the flag is a secret string in the LLM system prompt. The competition was organized in two phases. In the first phase, teams developed defenses to prevent the model from leaking the secret. During the second phase, teams were challenged to extract the secrets hidden for defenses proposed by the other teams. This report summarizes the main insights from the competition. Notably, we found that all defenses were bypassed at least once, highlighting the difficulty of designing a successful defense and the necessity for additional research to protect LLM systems. To foster future research in this direction, we compiled a dataset with over 137k multi-turn attack chats and open-sourced the platform.



## **45. Visual-RolePlay: Universal Jailbreak Attack on MultiModal Large Language Models via Role-playing Image Character**

cs.CR

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2405.20773v2) [paper-pdf](http://arxiv.org/pdf/2405.20773v2)

**Authors**: Siyuan Ma, Weidi Luo, Yu Wang, Xiaogeng Liu

**Abstract**: With the advent and widespread deployment of Multimodal Large Language Models (MLLMs), ensuring their safety has become increasingly critical. To achieve this objective, it requires us to proactively discover the vulnerability of MLLMs by exploring the attack methods. Thus, structure-based jailbreak attacks, where harmful semantic content is embedded within images, have been proposed to mislead the models. However, previous structure-based jailbreak methods mainly focus on transforming the format of malicious queries, such as converting harmful content into images through typography, which lacks sufficient jailbreak effectiveness and generalizability. To address these limitations, we first introduce the concept of "Role-play" into MLLM jailbreak attacks and propose a novel and effective method called Visual Role-play (VRP). Specifically, VRP leverages Large Language Models to generate detailed descriptions of high-risk characters and create corresponding images based on the descriptions. When paired with benign role-play instruction texts, these high-risk character images effectively mislead MLLMs into generating malicious responses by enacting characters with negative attributes. We further extend our VRP method into a universal setup to demonstrate its generalizability. Extensive experiments on popular benchmarks show that VRP outperforms the strongest baseline, Query relevant and FigStep, by an average Attack Success Rate (ASR) margin of 14.3% across all models.



## **46. We Have a Package for You! A Comprehensive Analysis of Package Hallucinations by Code Generating LLMs**

cs.SE

18 pages, 8 figures, 7 tables

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2406.10279v1) [paper-pdf](http://arxiv.org/pdf/2406.10279v1)

**Authors**: Joseph Spracklen, Raveen Wijewickrama, A H M Nazmus Sakib, Anindya Maiti, Murtuza Jadliwala

**Abstract**: The reliance of popular programming languages such as Python and JavaScript on centralized package repositories and open-source software, combined with the emergence of code-generating Large Language Models (LLMs), has created a new type of threat to the software supply chain: package hallucinations. These hallucinations, which arise from fact-conflicting errors when generating code using LLMs, represent a novel form of package confusion attack that poses a critical threat to the integrity of the software supply chain. This paper conducts a rigorous and comprehensive evaluation of package hallucinations across different programming languages, settings, and parameters, exploring how different configurations of LLMs affect the likelihood of generating erroneous package recommendations and identifying the root causes of this phenomena. Using 16 different popular code generation models, across two programming languages and two unique prompt datasets, we collect 576,000 code samples which we analyze for package hallucinations. Our findings reveal that 19.7% of generated packages across all the tested LLMs are hallucinated, including a staggering 205,474 unique examples of hallucinated package names, further underscoring the severity and pervasiveness of this threat. We also implemented and evaluated mitigation strategies based on Retrieval Augmented Generation (RAG), self-detected feedback, and supervised fine-tuning. These techniques demonstrably reduced package hallucinations, with hallucination rates for one model dropping below 3%. While the mitigation efforts were effective in reducing hallucination rates, our study reveals that package hallucinations are a systemic and persistent phenomenon that pose a significant challenge for code generating LLMs.



## **47. Defending Against Alignment-Breaking Attacks via Robustly Aligned LLM**

cs.CL

19 Pages, 5 Figures, 8 Tables. Accepted by ACL 2024

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2309.14348v3) [paper-pdf](http://arxiv.org/pdf/2309.14348v3)

**Authors**: Bochuan Cao, Yuanpu Cao, Lu Lin, Jinghui Chen

**Abstract**: Recently, Large Language Models (LLMs) have made significant advancements and are now widely used across various domains. Unfortunately, there has been a rising concern that LLMs can be misused to generate harmful or malicious content. Though a line of research has focused on aligning LLMs with human values and preventing them from producing inappropriate content, such alignments are usually vulnerable and can be bypassed by alignment-breaking attacks via adversarially optimized or handcrafted jailbreaking prompts. In this work, we introduce a Robustly Aligned LLM (RA-LLM) to defend against potential alignment-breaking attacks. RA-LLM can be directly constructed upon an existing aligned LLM with a robust alignment checking function, without requiring any expensive retraining or fine-tuning process of the original LLM. Furthermore, we also provide a theoretical analysis for RA-LLM to verify its effectiveness in defending against alignment-breaking attacks. Through real-world experiments on open-source large language models, we demonstrate that RA-LLM can successfully defend against both state-of-the-art adversarial prompts and popular handcrafted jailbreaking prompts by reducing their attack success rates from nearly 100% to around 10% or less.



## **48. Knowledge Return Oriented Prompting (KROP)**

cs.CR

**SubmitDate**: 2024-06-11    [abs](http://arxiv.org/abs/2406.11880v1) [paper-pdf](http://arxiv.org/pdf/2406.11880v1)

**Authors**: Jason Martin, Kenneth Yeung

**Abstract**: Many Large Language Models (LLMs) and LLM-powered apps deployed today use some form of prompt filter or alignment to protect their integrity. However, these measures aren't foolproof. This paper introduces KROP, a prompt injection technique capable of obfuscating prompt injection attacks, rendering them virtually undetectable to most of these security measures.



## **49. SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks**

cs.LG

**SubmitDate**: 2024-06-11    [abs](http://arxiv.org/abs/2310.03684v4) [paper-pdf](http://arxiv.org/pdf/2310.03684v4)

**Authors**: Alexander Robey, Eric Wong, Hamed Hassani, George J. Pappas

**Abstract**: Despite efforts to align large language models (LLMs) with human intentions, widely-used LLMs such as GPT, Llama, and Claude are susceptible to jailbreaking attacks, wherein an adversary fools a targeted LLM into generating objectionable content. To address this vulnerability, we propose SmoothLLM, the first algorithm designed to mitigate jailbreaking attacks. Based on our finding that adversarially-generated prompts are brittle to character-level changes, our defense randomly perturbs multiple copies of a given input prompt, and then aggregates the corresponding predictions to detect adversarial inputs. Across a range of popular LLMs, SmoothLLM sets the state-of-the-art for robustness against the GCG, PAIR, RandomSearch, and AmpleGCG jailbreaks. SmoothLLM is also resistant against adaptive GCG attacks, exhibits a small, though non-negligible trade-off between robustness and nominal performance, and is compatible with any LLM. Our code is publicly available at \url{https://github.com/arobey1/smooth-llm}.



## **50. Merging Improves Self-Critique Against Jailbreak Attacks**

cs.CL

**SubmitDate**: 2024-06-11    [abs](http://arxiv.org/abs/2406.07188v1) [paper-pdf](http://arxiv.org/pdf/2406.07188v1)

**Authors**: Victor Gallego

**Abstract**: The robustness of large language models (LLMs) against adversarial manipulations, such as jailbreak attacks, remains a significant challenge. In this work, we propose an approach that enhances the self-critique capability of the LLM and further fine-tunes it over sanitized synthetic data. This is done with the addition of an external critic model that can be merged with the original, thus bolstering self-critique capabilities and improving the robustness of the LLMs response to adversarial prompts. Our results demonstrate that the combination of merging and self-critique can reduce the attack success rate of adversaries significantly, thus offering a promising defense mechanism against jailbreak attacks. Code, data and models released at https://github.com/vicgalle/merging-self-critique-jailbreaks .



