# Latest Large Language Model Attack Papers
**update at 2024-02-04 13:13:04**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Vision-LLMs Can Fool Themselves with Self-Generated Typographic Attacks**

cs.CV

**SubmitDate**: 2024-02-01    [abs](http://arxiv.org/abs/2402.00626v1) [paper-pdf](http://arxiv.org/pdf/2402.00626v1)

**Authors**: Maan Qraitem, Nazia Tasnim, Kate Saenko, Bryan A. Plummer

**Abstract**: Recently, significant progress has been made on Large Vision-Language Models (LVLMs); a new class of VL models that make use of large pre-trained language models. Yet, their vulnerability to Typographic attacks, which involve superimposing misleading text onto an image remain unstudied. Furthermore, prior work typographic attacks rely on sampling a random misleading class from a predefined set of classes. However, the random chosen class might not be the most effective attack. To address these issues, we first introduce a novel benchmark uniquely designed to test LVLMs vulnerability to typographic attacks. Furthermore, we introduce a new and more effective typographic attack: Self-Generated typographic attacks. Indeed, our method, given an image, make use of the strong language capabilities of models like GPT-4V by simply prompting them to recommend a typographic attack. Using our novel benchmark, we uncover that typographic attacks represent a significant threat against LVLM(s). Furthermore, we uncover that typographic attacks recommended by GPT-4V using our new method are not only more effective against GPT-4V itself compared to prior work attacks, but also against a host of less capable yet popular open source models like LLaVA, InstructBLIP, and MiniGPT4.



## **2. Hidding the Ghostwriters: An Adversarial Evaluation of AI-Generated Student Essay Detection**

cs.CL

Accepted by EMNLP 2023 Main conference, Oral Presentation

**SubmitDate**: 2024-02-01    [abs](http://arxiv.org/abs/2402.00412v1) [paper-pdf](http://arxiv.org/pdf/2402.00412v1)

**Authors**: Xinlin Peng, Ying Zhou, Ben He, Le Sun, Yingfei Sun

**Abstract**: Large language models (LLMs) have exhibited remarkable capabilities in text generation tasks. However, the utilization of these models carries inherent risks, including but not limited to plagiarism, the dissemination of fake news, and issues in educational exercises. Although several detectors have been proposed to address these concerns, their effectiveness against adversarial perturbations, specifically in the context of student essay writing, remains largely unexplored. This paper aims to bridge this gap by constructing AIG-ASAP, an AI-generated student essay dataset, employing a range of text perturbation methods that are expected to generate high-quality essays while evading detection. Through empirical experiments, we assess the performance of current AIGC detectors on the AIG-ASAP dataset. The results reveal that the existing detectors can be easily circumvented using straightforward automatic adversarial attacks. Specifically, we explore word substitution and sentence substitution perturbation methods that effectively evade detection while maintaining the quality of the generated essays. This highlights the urgent need for more accurate and robust methods to detect AI-generated student essays in the education domain.



## **3. Safety of Multimodal Large Language Models on Images and Text**

cs.CV

**SubmitDate**: 2024-02-01    [abs](http://arxiv.org/abs/2402.00357v1) [paper-pdf](http://arxiv.org/pdf/2402.00357v1)

**Authors**: Xin Liu, Yichen Zhu, Yunshi Lan, Chao Yang, Yu Qiao

**Abstract**: Attracted by the impressive power of Multimodal Large Language Models (MLLMs), the public is increasingly utilizing them to improve the efficiency of daily work. Nonetheless, the vulnerabilities of MLLMs to unsafe instructions bring huge safety risks when these models are deployed in real-world scenarios. In this paper, we systematically survey current efforts on the evaluation, attack, and defense of MLLMs' safety on images and text. We begin with introducing the overview of MLLMs on images and text and understanding of safety, which helps researchers know the detailed scope of our survey. Then, we review the evaluation datasets and metrics for measuring the safety of MLLMs. Next, we comprehensively present attack and defense techniques related to MLLMs' safety. Finally, we analyze several unsolved issues and discuss promising research directions.



## **4. De-identification is not always enough**

cs.CL

**SubmitDate**: 2024-01-31    [abs](http://arxiv.org/abs/2402.00179v1) [paper-pdf](http://arxiv.org/pdf/2402.00179v1)

**Authors**: Atiquer Rahman Sarkar, Yao-Shun Chuang, Noman Mohammed, Xiaoqian Jiang

**Abstract**: For sharing privacy-sensitive data, de-identification is commonly regarded as adequate for safeguarding privacy. Synthetic data is also being considered as a privacy-preserving alternative. Recent successes with numerical and tabular data generative models and the breakthroughs in large generative language models raise the question of whether synthetically generated clinical notes could be a viable alternative to real notes for research purposes. In this work, we demonstrated that (i) de-identification of real clinical notes does not protect records against a membership inference attack, (ii) proposed a novel approach to generate synthetic clinical notes using the current state-of-the-art large language models, (iii) evaluated the performance of the synthetically generated notes in a clinical domain task, and (iv) proposed a way to mount a membership inference attack where the target model is trained with synthetic data. We observed that when synthetically generated notes closely match the performance of real data, they also exhibit similar privacy concerns to the real data. Whether other approaches to synthetically generated clinical notes could offer better trade-offs and become a better alternative to sensitive real notes warrants further investigation.



## **5. LoRec: Large Language Model for Robust Sequential Recommendation against Poisoning Attacks**

cs.IR

**SubmitDate**: 2024-01-31    [abs](http://arxiv.org/abs/2401.17723v1) [paper-pdf](http://arxiv.org/pdf/2401.17723v1)

**Authors**: Kaike Zhang, Qi Cao, Yunfan Wu, Fei Sun, Huawei Shen, Xueqi Cheng

**Abstract**: Sequential recommender systems stand out for their ability to capture users' dynamic interests and the patterns of item-to-item transitions. However, the inherent openness of sequential recommender systems renders them vulnerable to poisoning attacks, where fraudulent users are injected into the training data to manipulate learned patterns. Traditional defense strategies predominantly depend on predefined assumptions or rules extracted from specific known attacks, limiting their generalizability to unknown attack types. To solve the above problems, considering the rich open-world knowledge encapsulated in Large Language Models (LLMs), our research initially focuses on the capabilities of LLMs in the detection of unknown fraudulent activities within recommender systems, a strategy we denote as LLM4Dec. Empirical evaluations demonstrate the substantial capability of LLMs in identifying unknown fraudsters, leveraging their expansive, open-world knowledge.   Building upon this, we propose the integration of LLMs into defense strategies to extend their effectiveness beyond the confines of known attacks. We propose LoRec, an advanced framework that employs LLM-Enhanced Calibration to strengthen the robustness of sequential recommender systems against poisoning attacks. LoRec integrates an LLM-enhanced CalibraTor (LCT) that refines the training process of sequential recommender systems with knowledge derived from LLMs, applying a user-wise reweighting to diminish the impact of fraudsters injected by attacks. By incorporating LLMs' open-world knowledge, the LCT effectively converts the limited, specific priors or rules into a more general pattern of fraudsters, offering improved defenses against poisoning attacks. Our comprehensive experiments validate that LoRec, as a general framework, significantly strengthens the robustness of sequential recommender systems.



## **6. Unified Physical-Digital Face Attack Detection**

cs.CV

12 pages, 8 figures

**SubmitDate**: 2024-01-31    [abs](http://arxiv.org/abs/2401.17699v1) [paper-pdf](http://arxiv.org/pdf/2401.17699v1)

**Authors**: Hao Fang, Ajian Liu, Haocheng Yuan, Junze Zheng, Dingheng Zeng, Yanhong Liu, Jiankang Deng, Sergio Escalera, Xiaoming Liu, Jun Wan, Zhen Lei

**Abstract**: Face Recognition (FR) systems can suffer from physical (i.e., print photo) and digital (i.e., DeepFake) attacks. However, previous related work rarely considers both situations at the same time. This implies the deployment of multiple models and thus more computational burden. The main reasons for this lack of an integrated model are caused by two factors: (1) The lack of a dataset including both physical and digital attacks with ID consistency which means the same ID covers the real face and all attack types; (2) Given the large intra-class variance between these two attacks, it is difficult to learn a compact feature space to detect both attacks simultaneously. To address these issues, we collect a Unified physical-digital Attack dataset, called UniAttackData. The dataset consists of $1,800$ participations of 2 and 12 physical and digital attacks, respectively, resulting in a total of 29,706 videos. Then, we propose a Unified Attack Detection framework based on Vision-Language Models (VLMs), namely UniAttackDetection, which includes three main modules: the Teacher-Student Prompts (TSP) module, focused on acquiring unified and specific knowledge respectively; the Unified Knowledge Mining (UKM) module, designed to capture a comprehensive feature space; and the Sample-Level Prompt Interaction (SLPI) module, aimed at grasping sample-level semantics. These three modules seamlessly form a robust unified attack detection framework. Extensive experiments on UniAttackData and three other datasets demonstrate the superiority of our approach for unified face attack detection.



## **7. Weak-to-Strong Jailbreaking on Large Language Models**

cs.CL

**SubmitDate**: 2024-01-30    [abs](http://arxiv.org/abs/2401.17256v1) [paper-pdf](http://arxiv.org/pdf/2401.17256v1)

**Authors**: Xuandong Zhao, Xianjun Yang, Tianyu Pang, Chao Du, Lei Li, Yu-Xiang Wang, William Yang Wang

**Abstract**: Although significant efforts have been dedicated to aligning large language models (LLMs), red-teaming reports suggest that these carefully aligned LLMs could still be jailbroken through adversarial prompts, tuning, or decoding. Upon examining the jailbreaking vulnerability of aligned LLMs, we observe that the decoding distributions of jailbroken and aligned models differ only in the initial generations. This observation motivates us to propose the weak-to-strong jailbreaking attack, where adversaries can utilize smaller unsafe/aligned LLMs (e.g., 7B) to guide jailbreaking against significantly larger aligned LLMs (e.g., 70B). To jailbreak, one only needs to additionally decode two smaller LLMs once, which involves minimal computation and latency compared to decoding the larger LLMs. The efficacy of this attack is demonstrated through experiments conducted on five models from three different organizations. Our study reveals a previously unnoticed yet efficient way of jailbreaking, exposing an urgent safety issue that needs to be considered when aligning LLMs. As an initial attempt, we propose a defense strategy to protect against such attacks, but creating more advanced defenses remains challenging. The code for replicating the method is available at https://github.com/XuandongZhao/weak-to-strong



## **8. Noise Contrastive Estimation-based Matching Framework for Low-Resource Security Attack Pattern Recognition**

cs.LG

accepted at EACL 2024, in ARR October 2023

**SubmitDate**: 2024-01-30    [abs](http://arxiv.org/abs/2401.10337v3) [paper-pdf](http://arxiv.org/pdf/2401.10337v3)

**Authors**: Tu Nguyen, Nedim Šrndić, Alexander Neth

**Abstract**: Tactics, Techniques and Procedures (TTPs) represent sophisticated attack patterns in the cybersecurity domain, described encyclopedically in textual knowledge bases. Identifying TTPs in cybersecurity writing, often called TTP mapping, is an important and challenging task. Conventional learning approaches often target the problem in the classical multi-class or multilabel classification setting. This setting hinders the learning ability of the model due to a large number of classes (i.e., TTPs), the inevitable skewness of the label distribution and the complex hierarchical structure of the label space. We formulate the problem in a different learning paradigm, where the assignment of a text to a TTP label is decided by the direct semantic similarity between the two, thus reducing the complexity of competing solely over the large labeling space. To that end, we propose a neural matching architecture with an effective sampling-based learn-to-compare mechanism, facilitating the learning process of the matching model despite constrained resources.



## **9. Provably Robust Multi-bit Watermarking for AI-generated Text via Error Correction Code**

cs.CR

**SubmitDate**: 2024-01-30    [abs](http://arxiv.org/abs/2401.16820v1) [paper-pdf](http://arxiv.org/pdf/2401.16820v1)

**Authors**: Wenjie Qu, Dong Yin, Zixin He, Wei Zou, Tianyang Tao, Jinyuan Jia, Jiaheng Zhang

**Abstract**: Large Language Models (LLMs) have been widely deployed for their remarkable capability to generate texts resembling human language. However, they could be misused by criminals to create deceptive content, such as fake news and phishing emails, which raises ethical concerns. Watermarking is a key technique to mitigate the misuse of LLMs, which embeds a watermark (e.g., a bit string) into a text generated by a LLM. Consequently, this enables the detection of texts generated by a LLM as well as the tracing of generated texts to a specific user. The major limitation of existing watermark techniques is that they cannot accurately or efficiently extract the watermark from a text, especially when the watermark is a long bit string. This key limitation impedes their deployment for real-world applications, e.g., tracing generated texts to a specific user.   This work introduces a novel watermarking method for LLM-generated text grounded in \textbf{error-correction codes} to address this challenge. We provide strong theoretical analysis, demonstrating that under bounded adversarial word/token edits (insertion, deletion, and substitution), our method can correctly extract watermarks, offering a provable robustness guarantee. This breakthrough is also evidenced by our extensive experimental results. The experiments show that our method substantially outperforms existing baselines in both accuracy and robustness on benchmark datasets. For instance, when embedding a bit string of length 12 into a 200-token generated text, our approach attains an impressive match rate of $98.4\%$, surpassing the performance of Yoo et al. (state-of-the-art baseline) at $85.6\%$. When subjected to a copy-paste attack involving the injection of 50 tokens to generated texts with 200 words, our method maintains a substantial match rate of $90.8\%$, while the match rate of Yoo et al. diminishes to below $65\%$.



## **10. A Cross-Language Investigation into Jailbreak Attacks in Large Language Models**

cs.CR

**SubmitDate**: 2024-01-30    [abs](http://arxiv.org/abs/2401.16765v1) [paper-pdf](http://arxiv.org/pdf/2401.16765v1)

**Authors**: Jie Li, Yi Liu, Chongyang Liu, Ling Shi, Xiaoning Ren, Yaowen Zheng, Yang Liu, Yinxing Xue

**Abstract**: Large Language Models (LLMs) have become increasingly popular for their advanced text generation capabilities across various domains. However, like any software, they face security challenges, including the risk of 'jailbreak' attacks that manipulate LLMs to produce prohibited content. A particularly underexplored area is the Multilingual Jailbreak attack, where malicious questions are translated into various languages to evade safety filters. Currently, there is a lack of comprehensive empirical studies addressing this specific threat.   To address this research gap, we conducted an extensive empirical study on Multilingual Jailbreak attacks. We developed a novel semantic-preserving algorithm to create a multilingual jailbreak dataset and conducted an exhaustive evaluation on both widely-used open-source and commercial LLMs, including GPT-4 and LLaMa. Additionally, we performed interpretability analysis to uncover patterns in Multilingual Jailbreak attacks and implemented a fine-tuning mitigation method. Our findings reveal that our mitigation strategy significantly enhances model defense, reducing the attack success rate by 96.2%. This study provides valuable insights into understanding and mitigating Multilingual Jailbreak attacks.



## **11. Low-Resource Languages Jailbreak GPT-4**

cs.CL

NeurIPS Workshop on Socially Responsible Language Modelling Research  (SoLaR) 2023. Best Paper Award

**SubmitDate**: 2024-01-27    [abs](http://arxiv.org/abs/2310.02446v2) [paper-pdf](http://arxiv.org/pdf/2310.02446v2)

**Authors**: Zheng-Xin Yong, Cristina Menghini, Stephen H. Bach

**Abstract**: AI safety training and red-teaming of large language models (LLMs) are measures to mitigate the generation of unsafe content. Our work exposes the inherent cross-lingual vulnerability of these safety mechanisms, resulting from the linguistic inequality of safety training data, by successfully circumventing GPT-4's safeguard through translating unsafe English inputs into low-resource languages. On the AdvBenchmark, GPT-4 engages with the unsafe translated inputs and provides actionable items that can get the users towards their harmful goals 79% of the time, which is on par with or even surpassing state-of-the-art jailbreaking attacks. Other high-/mid-resource languages have significantly lower attack success rate, which suggests that the cross-lingual vulnerability mainly applies to low-resource languages. Previously, limited training on low-resource languages primarily affects speakers of those languages, causing technological disparities. However, our work highlights a crucial shift: this deficiency now poses a risk to all LLMs users. Publicly available translation APIs enable anyone to exploit LLMs' safety vulnerabilities. Therefore, our work calls for a more holistic red-teaming efforts to develop robust multilingual safeguards with wide language coverage.



## **12. L-AutoDA: Leveraging Large Language Models for Automated Decision-based Adversarial Attacks**

cs.CR

Under Review of IJCNN 2024

**SubmitDate**: 2024-01-27    [abs](http://arxiv.org/abs/2401.15335v1) [paper-pdf](http://arxiv.org/pdf/2401.15335v1)

**Authors**: Ping Guo, Fei Liu, Xi Lin, Qingchuan Zhao, Qingfu Zhang

**Abstract**: In the rapidly evolving field of machine learning, adversarial attacks present a significant challenge to model robustness and security. Decision-based attacks, which only require feedback on the decision of a model rather than detailed probabilities or scores, are particularly insidious and difficult to defend against. This work introduces L-AutoDA (Large Language Model-based Automated Decision-based Adversarial Attacks), a novel approach leveraging the generative capabilities of Large Language Models (LLMs) to automate the design of these attacks. By iteratively interacting with LLMs in an evolutionary framework, L-AutoDA automatically designs competitive attack algorithms efficiently without much human effort. We demonstrate the efficacy of L-AutoDA on CIFAR-10 dataset, showing significant improvements over baseline methods in both success rate and computational efficiency. Our findings underscore the potential of language models as tools for adversarial attack generation and highlight new avenues for the development of robust AI systems.



## **13. Better Representations via Adversarial Training in Pre-Training: A Theoretical Perspective**

cs.LG

To appear in AISTATS2024

**SubmitDate**: 2024-01-26    [abs](http://arxiv.org/abs/2401.15248v1) [paper-pdf](http://arxiv.org/pdf/2401.15248v1)

**Authors**: Yue Xing, Xiaofeng Lin, Qifan Song, Yi Xu, Belinda Zeng, Guang Cheng

**Abstract**: Pre-training is known to generate universal representations for downstream tasks in large-scale deep learning such as large language models. Existing literature, e.g., \cite{kim2020adversarial}, empirically observe that the downstream tasks can inherit the adversarial robustness of the pre-trained model. We provide theoretical justifications for this robustness inheritance phenomenon. Our theoretical results reveal that feature purification plays an important role in connecting the adversarial robustness of the pre-trained model and the downstream tasks in two-layer neural networks. Specifically, we show that (i) with adversarial training, each hidden node tends to pick only one (or a few) feature; (ii) without adversarial training, the hidden nodes can be vulnerable to attacks. This observation is valid for both supervised pre-training and contrastive learning. With purified nodes, it turns out that clean training is enough to achieve adversarial robustness in downstream tasks.



## **14. A Survey on Large Language Model (LLM) Security and Privacy: The Good, the Bad, and the Ugly**

cs.CR

**SubmitDate**: 2024-01-26    [abs](http://arxiv.org/abs/2312.02003v2) [paper-pdf](http://arxiv.org/pdf/2312.02003v2)

**Authors**: Yifan Yao, Jinhao Duan, Kaidi Xu, Yuanfang Cai, Zhibo Sun, Yue Zhang

**Abstract**: Large Language Models (LLMs), such as ChatGPT and Bard, have revolutionized natural language understanding and generation. They possess deep language comprehension, human-like text generation capabilities, contextual awareness, and robust problem-solving skills, making them invaluable in various domains (e.g., search engines, customer support, translation). In the meantime, LLMs have also gained traction in the security community, revealing security vulnerabilities and showcasing their potential in security-related tasks. This paper explores the intersection of LLMs with security and privacy. Specifically, we investigate how LLMs positively impact security and privacy, potential risks and threats associated with their use, and inherent vulnerabilities within LLMs. Through a comprehensive literature review, the paper categorizes the papers into "The Good" (beneficial LLM applications), "The Bad" (offensive applications), and "The Ugly" (vulnerabilities of LLMs and their defenses). We have some interesting findings. For example, LLMs have proven to enhance code security (code vulnerability detection) and data privacy (data confidentiality protection), outperforming traditional methods. However, they can also be harnessed for various attacks (particularly user-level attacks) due to their human-like reasoning abilities. We have identified areas that require further research efforts. For example, Research on model and parameter extraction attacks is limited and often theoretical, hindered by LLM parameter scale and confidentiality. Safe instruction tuning, a recent development, requires more exploration. We hope that our work can shed light on the LLMs' potential to both bolster and jeopardize cybersecurity.



## **15. TrojFST: Embedding Trojans in Few-shot Prompt Tuning**

cs.LG

9 pages

**SubmitDate**: 2024-01-25    [abs](http://arxiv.org/abs/2312.10467v2) [paper-pdf](http://arxiv.org/pdf/2312.10467v2)

**Authors**: Mengxin Zheng, Jiaqi Xue, Xun Chen, YanShan Wang, Qian Lou, Lei Jiang

**Abstract**: Prompt-tuning has emerged as a highly effective approach for adapting a pre-trained language model (PLM) to handle new natural language processing tasks with limited input samples. However, the success of prompt-tuning has led to adversaries attempting backdoor attacks against this technique. Previous prompt-based backdoor attacks faced challenges when implemented through few-shot prompt-tuning, requiring either full-model fine-tuning or a large training dataset. We observe the difficulty in constructing a prompt-based backdoor using few-shot prompt-tuning, which involves freezing the PLM and tuning a soft prompt with a restricted set of input samples. This approach introduces an imbalanced poisoned dataset, making it susceptible to overfitting and lacking attention awareness. To address these challenges, we introduce TrojFST for backdoor attacks within the framework of few-shot prompt-tuning. TrojFST comprises three modules: balanced poison learning, selective token poisoning, and trojan-trigger attention. In comparison to previous prompt-based backdoor attacks, TrojFST demonstrates significant improvements, enhancing ASR $> 9\%$ and CDA by $> 4\%$ across various PLMs and a diverse set of downstream tasks.



## **16. Adaptive Text Watermark for Large Language Models**

cs.CL

**SubmitDate**: 2024-01-25    [abs](http://arxiv.org/abs/2401.13927v1) [paper-pdf](http://arxiv.org/pdf/2401.13927v1)

**Authors**: Yepeng Liu, Yuheng Bu

**Abstract**: The advancement of Large Language Models (LLMs) has led to increasing concerns about the misuse of AI-generated text, and watermarking for LLM-generated text has emerged as a potential solution. However, it is challenging to generate high-quality watermarked text while maintaining strong security, robustness, and the ability to detect watermarks without prior knowledge of the prompt or model. This paper proposes an adaptive watermarking strategy to address this problem. To improve the text quality and maintain robustness, we adaptively add watermarking to token distributions with high entropy measured using an auxiliary model and keep the low entropy token distributions untouched. For the sake of security and to further minimize the watermark's impact on text quality, instead of using a fixed green/red list generated from a random secret key, which can be vulnerable to decryption and forgery, we adaptively scale up the output logits in proportion based on the semantic embedding of previously generated text using a well designed semantic mapping model. Our experiments involving various LLMs demonstrate that our approach achieves comparable robustness performance to existing watermark methods. Additionally, the text generated by our method has perplexity comparable to that of \emph{un-watermarked} LLMs while maintaining security even under various attacks.



## **17. TrojanPuzzle: Covertly Poisoning Code-Suggestion Models**

cs.CR

**SubmitDate**: 2024-01-24    [abs](http://arxiv.org/abs/2301.02344v2) [paper-pdf](http://arxiv.org/pdf/2301.02344v2)

**Authors**: Hojjat Aghakhani, Wei Dai, Andre Manoel, Xavier Fernandes, Anant Kharkar, Christopher Kruegel, Giovanni Vigna, David Evans, Ben Zorn, Robert Sim

**Abstract**: With tools like GitHub Copilot, automatic code suggestion is no longer a dream in software engineering. These tools, based on large language models, are typically trained on massive corpora of code mined from unvetted public sources. As a result, these models are susceptible to data poisoning attacks where an adversary manipulates the model's training by injecting malicious data. Poisoning attacks could be designed to influence the model's suggestions at run time for chosen contexts, such as inducing the model into suggesting insecure code payloads. To achieve this, prior attacks explicitly inject the insecure code payload into the training data, making the poison data detectable by static analysis tools that can remove such malicious data from the training set. In this work, we demonstrate two novel attacks, COVERT and TROJANPUZZLE, that can bypass static analysis by planting malicious poison data in out-of-context regions such as docstrings. Our most novel attack, TROJANPUZZLE, goes one step further in generating less suspicious poison data by never explicitly including certain (suspicious) parts of the payload in the poison data, while still inducing a model that suggests the entire payload when completing code (i.e., outside docstrings). This makes TROJANPUZZLE robust against signature-based dataset-cleansing methods that can filter out suspicious sequences from the training data. Our evaluation against models of two sizes demonstrates that both COVERT and TROJANPUZZLE have significant implications for practitioners when selecting code used to train or tune code-suggestion models.



## **18. How Johnny Can Persuade LLMs to Jailbreak Them: Rethinking Persuasion to Challenge AI Safety by Humanizing LLMs**

cs.CL

14 pages of the main text, qualitative examples of jailbreaks may be  harmful in nature

**SubmitDate**: 2024-01-23    [abs](http://arxiv.org/abs/2401.06373v2) [paper-pdf](http://arxiv.org/pdf/2401.06373v2)

**Authors**: Yi Zeng, Hongpeng Lin, Jingwen Zhang, Diyi Yang, Ruoxi Jia, Weiyan Shi

**Abstract**: Most traditional AI safety research has approached AI models as machines and centered on algorithm-focused attacks developed by security experts. As large language models (LLMs) become increasingly common and competent, non-expert users can also impose risks during daily interactions. This paper introduces a new perspective to jailbreak LLMs as human-like communicators, to explore this overlooked intersection between everyday language interaction and AI safety. Specifically, we study how to persuade LLMs to jailbreak them. First, we propose a persuasion taxonomy derived from decades of social science research. Then, we apply the taxonomy to automatically generate interpretable persuasive adversarial prompts (PAP) to jailbreak LLMs. Results show that persuasion significantly increases the jailbreak performance across all risk categories: PAP consistently achieves an attack success rate of over $92\%$ on Llama 2-7b Chat, GPT-3.5, and GPT-4 in $10$ trials, surpassing recent algorithm-focused attacks. On the defense side, we explore various mechanisms against PAP and, found a significant gap in existing defenses, and advocate for more fundamental mitigation for highly interactive LLMs



## **19. Text Embedding Inversion Attacks on Multilingual Language Models**

cs.CL

13 pages

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2401.12192v1) [paper-pdf](http://arxiv.org/pdf/2401.12192v1)

**Authors**: Yiyi Chen, Heather Lent, Johannes Bjerva

**Abstract**: Representing textual information as real-numbered embeddings has become the norm in NLP. Moreover, with the rise of public interest in large language models (LLMs), Embeddings as a Service (EaaS) has rapidly gained traction as a business model. This is not without outstanding security risks, as previous research has demonstrated that sensitive data can be reconstructed from embeddings, even without knowledge of the underlying model that generated them. However, such work is limited by its sole focus on English, leaving all other languages vulnerable to attacks by malicious actors. %As many international and multilingual companies leverage EaaS, there is an urgent need for research into multilingual LLM security. To this end, this work investigates LLM security from the perspective of multilingual embedding inversion. Concretely, we define the problem of black-box multilingual and cross-lingual inversion attacks, with special attention to a cross-domain scenario. Our findings reveal that multilingual models are potentially more vulnerable to inversion attacks than their monolingual counterparts. This stems from the reduced data requirements for achieving comparable inversion performance in settings where the underlying language is not known a-priori. To our knowledge, this work is the first to delve into multilinguality within the context of inversion attacks, and our findings highlight the need for further investigation and enhanced defenses in the area of NLP Security.



## **20. All in How You Ask for It: Simple Black-Box Method for Jailbreak Attacks**

cs.CL

12 pages, 4 figures, 2 tables

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2401.09798v2) [paper-pdf](http://arxiv.org/pdf/2401.09798v2)

**Authors**: Kazuhiro Takemoto

**Abstract**: Large Language Models (LLMs) like ChatGPT face `jailbreak' challenges, where safeguards are bypassed to produce ethically harmful prompts. This study proposes a simple black-box method to effectively generate jailbreak prompts, overcoming the high complexity and computational costs associated with existing methods. The proposed technique iteratively rewrites harmful prompts into non-harmful expressions using the target LLM itself, based on the hypothesis that LLMs can directly sample expressions that bypass safeguards. Demonstrated through experiments with ChatGPT (GPT-3.5 and GPT-4) and Gemini-Pro, this method achieved an attack success rate of over 80% within an average of 5 iterations and remained effective despite model updates. The generated jailbreak prompts were naturally-worded and concise; moreover, they were difficult-to-defend. These results indicate that creating effective jailbreak prompts is simpler than previously considered, suggesting that black-box jailbreak attacks pose a more serious threat.



## **21. Jailbreaking GPT-4V via Self-Adversarial Attacks with System Prompts**

cs.CR

**SubmitDate**: 2024-01-20    [abs](http://arxiv.org/abs/2311.09127v2) [paper-pdf](http://arxiv.org/pdf/2311.09127v2)

**Authors**: Yuanwei Wu, Xiang Li, Yixin Liu, Pan Zhou, Lichao Sun

**Abstract**: Existing work on jailbreak Multimodal Large Language Models (MLLMs) has focused primarily on adversarial examples in model inputs, with less attention to vulnerabilities, especially in model API. To fill the research gap, we carry out the following work: 1) We discover a system prompt leakage vulnerability in GPT-4V. Through carefully designed dialogue, we successfully extract the internal system prompts of GPT-4V. This finding indicates potential exploitable security risks in MLLMs; 2) Based on the acquired system prompts, we propose a novel MLLM jailbreaking attack method termed SASP (Self-Adversarial Attack via System Prompt). By employing GPT-4 as a red teaming tool against itself, we aim to search for potential jailbreak prompts leveraging stolen system prompts. Furthermore, in pursuit of better performance, we also add human modification based on GPT-4's analysis, which further improves the attack success rate to 98.7\%; 3) We evaluated the effect of modifying system prompts to defend against jailbreaking attacks. Results show that appropriately designed system prompts can significantly reduce jailbreak success rates. Overall, our work provides new insights into enhancing MLLM security, demonstrating the important role of system prompts in jailbreaking. This finding could be leveraged to greatly facilitate jailbreak success rates while also holding the potential for defending against jailbreaks.



## **22. Universal Vulnerabilities in Large Language Models: In-context Learning Backdoor Attacks**

cs.CL

**SubmitDate**: 2024-01-20    [abs](http://arxiv.org/abs/2401.05949v3) [paper-pdf](http://arxiv.org/pdf/2401.05949v3)

**Authors**: Shuai Zhao, Meihuizi Jia, Luu Anh Tuan, Jinming Wen

**Abstract**: In-context learning, a paradigm bridging the gap between pre-training and fine-tuning, has demonstrated high efficacy in several NLP tasks, especially in few-shot settings. Unlike traditional fine-tuning methods, in-context learning adapts pre-trained models to unseen tasks without updating any parameters. Despite being widely applied, in-context learning is vulnerable to malicious attacks. In this work, we raise security concerns regarding this paradigm. Our studies demonstrate that an attacker can manipulate the behavior of large language models by poisoning the demonstration context, without the need for fine-tuning the model. Specifically, we have designed a new backdoor attack method, named ICLAttack, to target large language models based on in-context learning. Our method encompasses two types of attacks: poisoning demonstration examples and poisoning prompts, which can make models behave in accordance with predefined intentions. ICLAttack does not require additional fine-tuning to implant a backdoor, thus preserving the model's generality. Furthermore, the poisoned examples are correctly labeled, enhancing the natural stealth of our attack method. Extensive experimental results across several language models, ranging in size from 1.3B to 40B parameters, demonstrate the effectiveness of our attack method, exemplified by a high average attack success rate of 95.0% across the three datasets on OPT models. Our findings highlight the vulnerabilities of language models, and we hope this work will raise awareness of the possible security threats associated with in-context learning.



## **23. InferAligner: Inference-Time Alignment for Harmlessness through Cross-Model Guidance**

cs.CL

**SubmitDate**: 2024-01-20    [abs](http://arxiv.org/abs/2401.11206v1) [paper-pdf](http://arxiv.org/pdf/2401.11206v1)

**Authors**: Pengyu Wang, Dong Zhang, Linyang Li, Chenkun Tan, Xinghao Wang, Ke Ren, Botian Jiang, Xipeng Qiu

**Abstract**: With the rapid development of large language models (LLMs), they are not only used as general-purpose AI assistants but are also customized through further fine-tuning to meet the requirements of different applications. A pivotal factor in the success of current LLMs is the alignment process. Current alignment methods, such as supervised fine-tuning (SFT) and reinforcement learning from human feedback (RLHF), focus on training-time alignment and are often complex and cumbersome to implement. Therefore, we develop \textbf{InferAligner}, a novel inference-time alignment method that utilizes cross-model guidance for harmlessness alignment. InferAligner utilizes safety steering vectors extracted from safety-aligned model to modify the activations of the target model when responding to harmful inputs, thereby guiding the target model to provide harmless responses. Experimental results show that our method can be very effectively applied to domain-specific models in finance, medicine, and mathematics, as well as to multimodal large language models (MLLMs) such as LLaVA. It significantly diminishes the Attack Success Rate (ASR) of both harmful instructions and jailbreak attacks, while maintaining almost unchanged performance in downstream tasks.



## **24. Inducing High Energy-Latency of Large Vision-Language Models with Verbose Images**

cs.CV

Accepted by ICLR 2024

**SubmitDate**: 2024-01-20    [abs](http://arxiv.org/abs/2401.11170v1) [paper-pdf](http://arxiv.org/pdf/2401.11170v1)

**Authors**: Kuofeng Gao, Yang Bai, Jindong Gu, Shu-Tao Xia, Philip Torr, Zhifeng Li, Wei Liu

**Abstract**: Large vision-language models (VLMs) such as GPT-4 have achieved exceptional performance across various multi-modal tasks. However, the deployment of VLMs necessitates substantial energy consumption and computational resources. Once attackers maliciously induce high energy consumption and latency time (energy-latency cost) during inference of VLMs, it will exhaust computational resources. In this paper, we explore this attack surface about availability of VLMs and aim to induce high energy-latency cost during inference of VLMs. We find that high energy-latency cost during inference of VLMs can be manipulated by maximizing the length of generated sequences. To this end, we propose verbose images, with the goal of crafting an imperceptible perturbation to induce VLMs to generate long sentences during inference. Concretely, we design three loss objectives. First, a loss is proposed to delay the occurrence of end-of-sequence (EOS) token, where EOS token is a signal for VLMs to stop generating further tokens. Moreover, an uncertainty loss and a token diversity loss are proposed to increase the uncertainty over each generated token and the diversity among all tokens of the whole generated sequence, respectively, which can break output dependency at token-level and sequence-level. Furthermore, a temporal weight adjustment algorithm is proposed, which can effectively balance these losses. Extensive experiments demonstrate that our verbose images can increase the length of generated sequences by 7.87 times and 8.56 times compared to original images on MS-COCO and ImageNet datasets, which presents potential challenges for various applications. Our code is available at https://github.com/KuofengGao/Verbose_Images.



## **25. BadChain: Backdoor Chain-of-Thought Prompting for Large Language Models**

cs.CR

Accepted to ICLR2024

**SubmitDate**: 2024-01-20    [abs](http://arxiv.org/abs/2401.12242v1) [paper-pdf](http://arxiv.org/pdf/2401.12242v1)

**Authors**: Zhen Xiang, Fengqing Jiang, Zidi Xiong, Bhaskar Ramasubramanian, Radha Poovendran, Bo Li

**Abstract**: Large language models (LLMs) are shown to benefit from chain-of-thought (COT) prompting, particularly when tackling tasks that require systematic reasoning processes. On the other hand, COT prompting also poses new vulnerabilities in the form of backdoor attacks, wherein the model will output unintended malicious content under specific backdoor-triggered conditions during inference. Traditional methods for launching backdoor attacks involve either contaminating the training dataset with backdoored instances or directly manipulating the model parameters during deployment. However, these approaches are not practical for commercial LLMs that typically operate via API access. In this paper, we propose BadChain, the first backdoor attack against LLMs employing COT prompting, which does not require access to the training dataset or model parameters and imposes low computational overhead. BadChain leverages the inherent reasoning capabilities of LLMs by inserting a backdoor reasoning step into the sequence of reasoning steps of the model output, thereby altering the final response when a backdoor trigger exists in the query prompt. Empirically, we show the effectiveness of BadChain for two COT strategies across four LLMs (Llama2, GPT-3.5, PaLM2, and GPT-4) and six complex benchmark tasks encompassing arithmetic, commonsense, and symbolic reasoning. Moreover, we show that LLMs endowed with stronger reasoning capabilities exhibit higher susceptibility to BadChain, exemplified by a high average attack success rate of 97.0% across the six benchmark tasks on GPT-4. Finally, we propose two defenses based on shuffling and demonstrate their overall ineffectiveness against BadChain. Therefore, BadChain remains a severe threat to LLMs, underscoring the urgency for the development of robust and effective future defenses.



## **26. Pruning for Protection: Increasing Jailbreak Resistance in Aligned LLMs Without Fine-Tuning**

cs.LG

**SubmitDate**: 2024-01-19    [abs](http://arxiv.org/abs/2401.10862v1) [paper-pdf](http://arxiv.org/pdf/2401.10862v1)

**Authors**: Adib Hasan, Ileana Rugina, Alex Wang

**Abstract**: Large Language Models (LLMs) are vulnerable to `Jailbreaking' prompts, a type of attack that can coax these models into generating harmful and illegal content. In this paper, we show that pruning up to 20% of LLM parameters markedly increases their resistance to such attacks without additional training and without sacrificing their performance in standard benchmarks. Intriguingly, we discovered that the enhanced safety observed post-pruning correlates to the initial safety training level of the model, hinting that the effect of pruning could be more general and may hold for other LLM behaviors beyond safety. Additionally, we introduce a curated dataset of 225 harmful tasks across five categories, inserted into ten different Jailbreaking prompts, showing that pruning aids LLMs in concentrating attention on task-relevant tokens in jailbreaking prompts. Lastly, our experiments reveal that the prominent chat models, such as LLaMA-2 Chat, Vicuna, and Mistral Instruct exhibit high susceptibility to jailbreaking attacks, with some categories achieving nearly 70-100% success rate. These insights underline the potential of pruning as a generalizable approach for improving LLM safety, reliability, and potentially other desired behaviors.



## **27. Large Language Model Lateral Spear Phishing: A Comparative Study in Large-Scale Organizational Settings**

cs.CR

**SubmitDate**: 2024-01-18    [abs](http://arxiv.org/abs/2401.09727v1) [paper-pdf](http://arxiv.org/pdf/2401.09727v1)

**Authors**: Mazal Bethany, Athanasios Galiopoulos, Emet Bethany, Mohammad Bahrami Karkevandi, Nishant Vishwamitra, Peyman Najafirad

**Abstract**: The critical threat of phishing emails has been further exacerbated by the potential of LLMs to generate highly targeted, personalized, and automated spear phishing attacks. Two critical problems concerning LLM-facilitated phishing require further investigation: 1) Existing studies on lateral phishing lack specific examination of LLM integration for large-scale attacks targeting the entire organization, and 2) Current anti-phishing infrastructure, despite its extensive development, lacks the capability to prevent LLM-generated attacks, potentially impacting both employees and IT security incident management. However, the execution of such investigative studies necessitates a real-world environment, one that functions during regular business operations and mirrors the complexity of a large organizational infrastructure. This setting must also offer the flexibility required to facilitate a diverse array of experimental conditions, particularly the incorporation of phishing emails crafted by LLMs. This study is a pioneering exploration into the use of Large Language Models (LLMs) for the creation of targeted lateral phishing emails, targeting a large tier 1 university's operation and workforce of approximately 9,000 individuals over an 11-month period. It also evaluates the capability of email filtering infrastructure to detect such LLM-generated phishing attempts, providing insights into their effectiveness and identifying potential areas for improvement. Based on our findings, we propose machine learning-based detection techniques for such emails to detect LLM-generated phishing emails that were missed by the existing infrastructure, with an F1-score of 98.96.



## **28. MLLM-Protector: Ensuring MLLM's Safety without Hurting Performance**

cs.CR

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2401.02906v2) [paper-pdf](http://arxiv.org/pdf/2401.02906v2)

**Authors**: Renjie Pi, Tianyang Han, Yueqi Xie, Rui Pan, Qing Lian, Hanze Dong, Jipeng Zhang, Tong Zhang

**Abstract**: The deployment of multimodal large language models (MLLMs) has brought forth a unique vulnerability: susceptibility to malicious attacks through visual inputs. We delve into the novel challenge of defending MLLMs against such attacks. We discovered that images act as a "foreign language" that is not considered during alignment, which can make MLLMs prone to producing harmful responses. Unfortunately, unlike the discrete tokens considered in text-based LLMs, the continuous nature of image signals presents significant alignment challenges, which poses difficulty to thoroughly cover the possible scenarios. This vulnerability is exacerbated by the fact that open-source MLLMs are predominantly fine-tuned on limited image-text pairs that is much less than the extensive text-based pretraining corpus, which makes the MLLMs more prone to catastrophic forgetting of their original abilities during explicit alignment tuning. To tackle these challenges, we introduce MLLM-Protector, a plug-and-play strategy combining a lightweight harm detector and a response detoxifier. The harm detector's role is to identify potentially harmful outputs from the MLLM, while the detoxifier corrects these outputs to ensure the response stipulates to the safety standards. This approach effectively mitigates the risks posed by malicious visual inputs without compromising the model's overall performance. Our results demonstrate that MLLM-Protector offers a robust solution to a previously unaddressed aspect of MLLM security.



## **29. AttackEval: How to Evaluate the Effectiveness of Jailbreak Attacking on Large Language Models**

cs.CL

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2401.09002v1) [paper-pdf](http://arxiv.org/pdf/2401.09002v1)

**Authors**: Dong shu, Mingyu Jin, Suiyuan Zhu, Beichen Wang, Zihao Zhou, Chong Zhang, Yongfeng Zhang

**Abstract**: In our research, we pioneer a novel approach to evaluate the effectiveness of jailbreak attacks on Large Language Models (LLMs), such as GPT-4 and LLaMa2, diverging from traditional robustness-focused binary evaluations. Our study introduces two distinct evaluation frameworks: a coarse-grained evaluation and a fine-grained evaluation. Each framework, using a scoring range from 0 to 1, offers a unique perspective, enabling a more comprehensive and nuanced evaluation of attack effectiveness and empowering attackers to refine their attack prompts with greater understanding. Furthermore, we have developed a comprehensive ground truth dataset specifically tailored for jailbreak tasks. This dataset not only serves as a crucial benchmark for our current study but also establishes a foundational resource for future research, enabling consistent and comparative analyses in this evolving field. Upon meticulous comparison with traditional evaluation methods, we discovered that our evaluation aligns with the baseline's trend while offering a more profound and detailed assessment. We believe that by accurately evaluating the effectiveness of attack prompts in the Jailbreak task, our work lays a solid foundation for assessing a wider array of similar or even more complex tasks in the realm of prompt injection, potentially revolutionizing this field.



## **30. Whispering Pixels: Exploiting Uninitialized Register Accesses in Modern GPUs**

cs.CR

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2401.08881v1) [paper-pdf](http://arxiv.org/pdf/2401.08881v1)

**Authors**: Frederik Dermot Pustelnik, Xhani Marvin Saß, Jean-Pierre Seifert

**Abstract**: Graphic Processing Units (GPUs) have transcended their traditional use-case of rendering graphics and nowadays also serve as a powerful platform for accelerating ubiquitous, non-graphical rendering tasks. One prominent task is inference of neural networks, which process vast amounts of personal data, such as audio, text or images. Thus, GPUs became integral components for handling vast amounts of potentially confidential data, which has awakened the interest of security researchers. This lead to the discovery of various vulnerabilities in GPUs in recent years. In this paper, we uncover yet another vulnerability class in GPUs: We found that some GPU implementations lack proper register initialization routines before shader execution, leading to unintended register content leakage of previously executed shader kernels. We showcase the existence of the aforementioned vulnerability on products of 3 major vendors - Apple, NVIDIA and Qualcomm. The vulnerability poses unique challenges to an adversary due to opaque scheduling and register remapping algorithms present in the GPU firmware, complicating the reconstruction of leaked data. In order to illustrate the real-world impact of this flaw, we showcase how these challenges can be solved for attacking various workloads on the GPU. First, we showcase how uninitialized registers leak arbitrary pixel data processed by fragment shaders. We further implement information leakage attacks on intermediate data of Convolutional Neural Networks (CNNs) and present the attack's capability to leak and reconstruct the output of Large Language Models (LLMs).



## **31. IsamasRed: A Public Dataset Tracking Reddit Discussions on Israel-Hamas Conflict**

cs.SI

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2401.08202v1) [paper-pdf](http://arxiv.org/pdf/2401.08202v1)

**Authors**: Kai Chen, Zihao He, Keith Burghardt, Jingxin Zhang, Kristina Lerman

**Abstract**: The conflict between Israel and Palestinians significantly escalated after the October 7, 2023 Hamas attack, capturing global attention. To understand the public discourse on this conflict, we present a meticulously compiled dataset--IsamasRed--comprising nearly 400,000 conversations and over 8 million comments from Reddit, spanning from August 2023 to November 2023. We introduce an innovative keyword extraction framework leveraging a large language model to effectively identify pertinent keywords, ensuring a comprehensive data collection. Our initial analysis on the dataset, examining topics, controversy, emotional and moral language trends over time, highlights the emotionally charged and complex nature of the discourse. This dataset aims to enrich the understanding of online discussions, shedding light on the complex interplay between ideology, sentiment, and community engagement in digital spaces.



## **32. MGTBench: Benchmarking Machine-Generated Text Detection**

cs.CR

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2303.14822v3) [paper-pdf](http://arxiv.org/pdf/2303.14822v3)

**Authors**: Xinlei He, Xinyue Shen, Zeyuan Chen, Michael Backes, Yang Zhang

**Abstract**: Nowadays, powerful large language models (LLMs) such as ChatGPT have demonstrated revolutionary power in a variety of tasks. Consequently, the detection of machine-generated texts (MGTs) is becoming increasingly crucial as LLMs become more advanced and prevalent. These models have the ability to generate human-like language, making it challenging to discern whether a text is authored by a human or a machine. This raises concerns regarding authenticity, accountability, and potential bias. However, existing methods for detecting MGTs are evaluated using different model architectures, datasets, and experimental settings, resulting in a lack of a comprehensive evaluation framework that encompasses various methodologies. Furthermore, it remains unclear how existing detection methods would perform against powerful LLMs. In this paper, we fill this gap by proposing the first benchmark framework for MGT detection against powerful LLMs, named MGTBench. Extensive evaluations on public datasets with curated texts generated by various powerful LLMs such as ChatGPT-turbo and Claude demonstrate the effectiveness of different detection methods. Our ablation study shows that a larger number of words in general leads to better performance and most detection methods can achieve similar performance with much fewer training samples. Moreover, we delve into a more challenging task: text attribution. Our findings indicate that the model-based detection methods still perform well in the text attribution task. To investigate the robustness of different detection methods, we consider three adversarial attacks, namely paraphrasing, random spacing, and adversarial perturbations. We discover that these attacks can significantly diminish detection effectiveness, underscoring the critical need for the development of more robust detection methods.



## **33. Traces of Memorisation in Large Language Models for Code**

cs.CR

ICSE 2024 Research Track

**SubmitDate**: 2024-01-15    [abs](http://arxiv.org/abs/2312.11658v2) [paper-pdf](http://arxiv.org/pdf/2312.11658v2)

**Authors**: Ali Al-Kaswan, Maliheh Izadi, Arie van Deursen

**Abstract**: Large language models have gained significant popularity because of their ability to generate human-like text and potential applications in various fields, such as Software Engineering. Large language models for code are commonly trained on large unsanitised corpora of source code scraped from the internet. The content of these datasets is memorised and can be extracted by attackers with data extraction attacks. In this work, we explore memorisation in large language models for code and compare the rate of memorisation with large language models trained on natural language. We adopt an existing benchmark for natural language and construct a benchmark for code by identifying samples that are vulnerable to attack. We run both benchmarks against a variety of models, and perform a data extraction attack. We find that large language models for code are vulnerable to data extraction attacks, like their natural language counterparts. From the training data that was identified to be potentially extractable we were able to extract 47% from a CodeGen-Mono-16B code completion model. We also observe that models memorise more, as their parameter count grows, and that their pre-training data are also vulnerable to attack. We also find that data carriers are memorised at a higher rate than regular code or documentation and that different model architectures memorise different samples. Data leakage has severe outcomes, so we urge the research community to further investigate the extent of this phenomenon using a wider range of models and extraction techniques in order to build safeguards to mitigate this issue.



## **34. Authorship Obfuscation in Multilingual Machine-Generated Text Detection**

cs.CL

**SubmitDate**: 2024-01-15    [abs](http://arxiv.org/abs/2401.07867v1) [paper-pdf](http://arxiv.org/pdf/2401.07867v1)

**Authors**: Dominik Macko, Robert Moro, Adaku Uchendu, Ivan Srba, Jason Samuel Lucas, Michiharu Yamashita, Nafis Irtiza Tripto, Dongwon Lee, Jakub Simko, Maria Bielikova

**Abstract**: High-quality text generation capability of latest Large Language Models (LLMs) causes concerns about their misuse (e.g., in massive generation/spread of disinformation). Machine-generated text (MGT) detection is important to cope with such threats. However, it is susceptible to authorship obfuscation (AO) methods, such as paraphrasing, which can cause MGTs to evade detection. So far, this was evaluated only in monolingual settings. Thus, the susceptibility of recently proposed multilingual detectors is still unknown. We fill this gap by comprehensively benchmarking the performance of 10 well-known AO methods, attacking 37 MGT detection methods against MGTs in 11 languages (i.e., 10 $\times$ 37 $\times$ 11 = 4,070 combinations). We also evaluate the effect of data augmentation on adversarial robustness using obfuscated texts. The results indicate that all tested AO methods can cause detection evasion in all tested languages, where homoglyph attacks are especially successful.



## **35. Signed-Prompt: A New Approach to Prevent Prompt Injection Attacks Against LLM-Integrated Applications**

cs.CR

**SubmitDate**: 2024-01-15    [abs](http://arxiv.org/abs/2401.07612v1) [paper-pdf](http://arxiv.org/pdf/2401.07612v1)

**Authors**: Xuchen Suo

**Abstract**: The critical challenge of prompt injection attacks in Large Language Models (LLMs) integrated applications, a growing concern in the Artificial Intelligence (AI) field. Such attacks, which manipulate LLMs through natural language inputs, pose a significant threat to the security of these applications. Traditional defense strategies, including output and input filtering, as well as delimiter use, have proven inadequate. This paper introduces the 'Signed-Prompt' method as a novel solution. The study involves signing sensitive instructions within command segments by authorized users, enabling the LLM to discern trusted instruction sources. The paper presents a comprehensive analysis of prompt injection attack patterns, followed by a detailed explanation of the Signed-Prompt concept, including its basic architecture and implementation through both prompt engineering and fine-tuning of LLMs. Experiments demonstrate the effectiveness of the Signed-Prompt method, showing substantial resistance to various types of prompt injection attacks, thus validating its potential as a robust defense strategy in AI security.



## **36. Stability Analysis of ChatGPT-based Sentiment Analysis in AI Quality Assurance**

cs.CL

**SubmitDate**: 2024-01-15    [abs](http://arxiv.org/abs/2401.07441v1) [paper-pdf](http://arxiv.org/pdf/2401.07441v1)

**Authors**: Tinghui Ouyang, AprilPyone MaungMaung, Koichi Konishi, Yoshiki Seo, Isao Echizen

**Abstract**: In the era of large AI models, the complex architecture and vast parameters present substantial challenges for effective AI quality management (AIQM), e.g. large language model (LLM). This paper focuses on investigating the quality assurance of a specific LLM-based AI product--a ChatGPT-based sentiment analysis system. The study delves into stability issues related to both the operation and robustness of the expansive AI model on which ChatGPT is based. Experimental analysis is conducted using benchmark datasets for sentiment analysis. The results reveal that the constructed ChatGPT-based sentiment analysis system exhibits uncertainty, which is attributed to various operational factors. It demonstrated that the system also exhibits stability issues in handling conventional small text attacks involving robustness.



## **37. Advancing TTP Analysis: Harnessing the Power of Encoder-Only and Decoder-Only Language Models with Retrieval Augmented Generation**

cs.CR

**SubmitDate**: 2024-01-12    [abs](http://arxiv.org/abs/2401.00280v2) [paper-pdf](http://arxiv.org/pdf/2401.00280v2)

**Authors**: Reza Fayyazi, Rozhina Taghdimi, Shanchieh Jay Yang

**Abstract**: Tactics, Techniques, and Procedures (TTPs) outline the methods attackers use to exploit vulnerabilities. The interpretation of TTPs in the MITRE ATT&CK framework can be challenging for cybersecurity practitioners due to presumed expertise, complex dependencies, and inherent ambiguity. Meanwhile, advancements with Large Language Models (LLMs) have led to recent surge in studies exploring its uses in cybersecurity operations. This leads us to question how well encoder-only (e.g., RoBERTa) and decoder-only (e.g., GPT-3.5) LLMs can comprehend and summarize TTPs to inform analysts of the intended purposes (i.e., tactics) of a cyberattack procedure. The state-of-the-art LLMs have shown to be prone to hallucination by providing inaccurate information, which is problematic in critical domains like cybersecurity. Therefore, we propose the use of Retrieval Augmented Generation (RAG) techniques to extract relevant contexts for each cyberattack procedure for decoder-only LLMs (without fine-tuning). We further contrast such approach against supervised fine-tuning (SFT) of encoder-only LLMs. Our results reveal that both the direct-use of decoder-only LLMs (i.e., its pre-trained knowledge) and the SFT of encoder-only LLMs offer inaccurate interpretation of cyberattack procedures. Significant improvements are shown when RAG is used for decoder-only LLMs, particularly when directly relevant context is found. This study further sheds insights on the limitations and capabilities of using RAG for LLMs in interpreting TTPs.



## **38. Intention Analysis Prompting Makes Large Language Models A Good Jailbreak Defender**

cs.CL

9 pages, 5 figures

**SubmitDate**: 2024-01-12    [abs](http://arxiv.org/abs/2401.06561v1) [paper-pdf](http://arxiv.org/pdf/2401.06561v1)

**Authors**: Yuqi Zhang, Liang Ding, Lefei Zhang, Dacheng Tao

**Abstract**: Aligning large language models (LLMs) with human values, particularly in the face of stealthy and complex jailbreaks, presents a formidable challenge. In this study, we present a simple yet highly effective defense strategy, i.e., Intention Analysis Prompting (IAPrompt). The principle behind is to trigger LLMs' inherent self-correct and improve ability through a two-stage process: 1) essential intention analysis, and 2) policy-aligned response. Notably, IAPrompt is an inference-only method, thus could enhance the safety of LLMs without compromising their helpfulness. Extensive experiments on SAP200 and DAN benchmarks across Vicuna, ChatGLM, MPT, DeepSeek, and GPT-3.5 show that IAPrompt could consistently and significantly reduce the harmfulness in response (averagely -46.5% attack success rate) and maintain the general helpfulness. Further analyses present some insights into how our method works. To facilitate reproducibility, We release our code and scripts at: https://github.com/alphadl/SafeLLM_with_IntentionAnalysis



## **39. Towards Robust Pruning: An Adaptive Knowledge-Retention Pruning Strategy for Language Models**

cs.CL

**SubmitDate**: 2024-01-11    [abs](http://arxiv.org/abs/2310.13191v3) [paper-pdf](http://arxiv.org/pdf/2310.13191v3)

**Authors**: Jianwei Li, Qi Lei, Wei Cheng, Dongkuan Xu

**Abstract**: The pruning objective has recently extended beyond accuracy and sparsity to robustness in language models. Despite this, existing methods struggle to enhance robustness against adversarial attacks when continually increasing model sparsity and require a retraining process. As humans step into the era of large language models, these issues become increasingly prominent. This paper proposes that the robustness of language models is proportional to the extent of pre-trained knowledge they encompass. Accordingly, we introduce a post-training pruning strategy designed to faithfully replicate the embedding space and feature space of dense language models, aiming to conserve more pre-trained knowledge during the pruning process. In this setup, each layer's reconstruction error not only originates from itself but also includes cumulative error from preceding layers, followed by an adaptive rectification. Compared to other state-of-art baselines, our approach demonstrates a superior balance between accuracy, sparsity, robustness, and pruning cost with BERT on datasets SST2, IMDB, and AGNews, marking a significant stride towards robust pruning in language models.



## **40. LimeAttack: Local Explainable Method for Textual Hard-Label Adversarial Attack**

cs.CL

18 pages, 38th AAAI Main Track

**SubmitDate**: 2024-01-10    [abs](http://arxiv.org/abs/2308.00319v2) [paper-pdf](http://arxiv.org/pdf/2308.00319v2)

**Authors**: Hai Zhu, Zhaoqing Yang, Weiwei Shang, Yuren Wu

**Abstract**: Natural language processing models are vulnerable to adversarial examples. Previous textual adversarial attacks adopt gradients or confidence scores to calculate word importance ranking and generate adversarial examples. However, this information is unavailable in the real world. Therefore, we focus on a more realistic and challenging setting, named hard-label attack, in which the attacker can only query the model and obtain a discrete prediction label. Existing hard-label attack algorithms tend to initialize adversarial examples by random substitution and then utilize complex heuristic algorithms to optimize the adversarial perturbation. These methods require a lot of model queries and the attack success rate is restricted by adversary initialization. In this paper, we propose a novel hard-label attack algorithm named LimeAttack, which leverages a local explainable method to approximate word importance ranking, and then adopts beam search to find the optimal solution. Extensive experiments show that LimeAttack achieves the better attacking performance compared with existing hard-label attack under the same query budget. In addition, we evaluate the effectiveness of LimeAttack on large language models, and results indicate that adversarial examples remain a significant threat to large language models. The adversarial examples crafted by LimeAttack are highly transferable and effectively improve model robustness in adversarial training.



## **41. Jatmo: Prompt Injection Defense by Task-Specific Finetuning**

cs.CR

24 pages, 6 figures

**SubmitDate**: 2024-01-08    [abs](http://arxiv.org/abs/2312.17673v2) [paper-pdf](http://arxiv.org/pdf/2312.17673v2)

**Authors**: Julien Piet, Maha Alrashed, Chawin Sitawarin, Sizhe Chen, Zeming Wei, Elizabeth Sun, Basel Alomair, David Wagner

**Abstract**: Large Language Models (LLMs) are attracting significant research attention due to their instruction-following abilities, allowing users and developers to leverage LLMs for a variety of tasks. However, LLMs are vulnerable to prompt-injection attacks: a class of attacks that hijack the model's instruction-following abilities, changing responses to prompts to undesired, possibly malicious ones. In this work, we introduce Jatmo, a method for generating task-specific models resilient to prompt-injection attacks. Jatmo leverages the fact that LLMs can only follow instructions once they have undergone instruction tuning. It harnesses a teacher instruction-tuned model to generate a task-specific dataset, which is then used to fine-tune a base model (i.e., a non-instruction-tuned model). Jatmo only needs a task prompt and a dataset of inputs for the task: it uses the teacher model to generate outputs. For situations with no pre-existing datasets, Jatmo can use a single example, or in some cases none at all, to produce a fully synthetic dataset. Our experiments on seven tasks show that Jatmo models provide similar quality of outputs on their specific task as standard LLMs, while being resilient to prompt injections. The best attacks succeeded in less than 0.5% of cases against our models, versus 87% success rate against GPT-3.5-Turbo. We release Jatmo at https://github.com/wagner-group/prompt-injection-defense.



## **42. The Stronger the Diffusion Model, the Easier the Backdoor: Data Poisoning to Induce Copyright Breaches Without Adjusting Finetuning Pipeline**

cs.CR

This study reveals that by subtly inserting non-copyright-infringing  poisoning data into a diffusion model's training dataset, it's possible to  trigger the model to generate copyrighted content, highlighting  vulnerabilities in current copyright protection strategies

**SubmitDate**: 2024-01-07    [abs](http://arxiv.org/abs/2401.04136v1) [paper-pdf](http://arxiv.org/pdf/2401.04136v1)

**Authors**: Haonan Wang, Qianli Shen, Yao Tong, Yang Zhang, Kenji Kawaguchi

**Abstract**: The commercialization of diffusion models, renowned for their ability to generate high-quality images that are often indistinguishable from real ones, brings forth potential copyright concerns. Although attempts have been made to impede unauthorized access to copyrighted material during training and to subsequently prevent DMs from generating copyrighted images, the effectiveness of these solutions remains unverified. This study explores the vulnerabilities associated with copyright protection in DMs by introducing a backdoor data poisoning attack (SilentBadDiffusion) against text-to-image diffusion models. Our attack method operates without requiring access to or control over the diffusion model's training or fine-tuning processes; it merely involves the insertion of poisoning data into the clean training dataset. This data, comprising poisoning images equipped with prompts, is generated by leveraging the powerful capabilities of multimodal large language models and text-guided image inpainting techniques. Our experimental results and analysis confirm the method's effectiveness. By integrating a minor portion of non-copyright-infringing stealthy poisoning data into the clean dataset-rendering it free from suspicion-we can prompt the finetuned diffusion models to produce copyrighted content when activated by specific trigger prompts. These findings underline potential pitfalls in the prevailing copyright protection strategies and underscore the necessity for increased scrutiny and preventative measures against the misuse of DMs.



## **43. PromptBench: A Unified Library for Evaluation of Large Language Models**

cs.AI

An extension to PromptBench (arXiv:2306.04528) for unified evaluation  of LLMs using the same name; code: https://github.com/microsoft/promptbench

**SubmitDate**: 2024-01-05    [abs](http://arxiv.org/abs/2312.07910v2) [paper-pdf](http://arxiv.org/pdf/2312.07910v2)

**Authors**: Kaijie Zhu, Qinlin Zhao, Hao Chen, Jindong Wang, Xing Xie

**Abstract**: The evaluation of large language models (LLMs) is crucial to assess their performance and mitigate potential security risks. In this paper, we introduce PromptBench, a unified library to evaluate LLMs. It consists of several key components that are easily used and extended by researchers: prompt construction, prompt engineering, dataset and model loading, adversarial prompt attack, dynamic evaluation protocols, and analysis tools. PromptBench is designed to be an open, general, and flexible codebase for research purposes that can facilitate original study in creating new benchmarks, deploying downstream applications, and designing new evaluation protocols. The code is available at: https://github.com/microsoft/promptbench and will be continuously supported.



## **44. InstructTA: Instruction-Tuned Targeted Attack for Large Vision-Language Models**

cs.CV

**SubmitDate**: 2024-01-04    [abs](http://arxiv.org/abs/2312.01886v2) [paper-pdf](http://arxiv.org/pdf/2312.01886v2)

**Authors**: Xunguang Wang, Zhenlan Ji, Pingchuan Ma, Zongjie Li, Shuai Wang

**Abstract**: Large vision-language models (LVLMs) have demonstrated their incredible capability in image understanding and response generation. However, this rich visual interaction also makes LVLMs vulnerable to adversarial examples. In this paper, we formulate a novel and practical gray-box attack scenario that the adversary can only access the visual encoder of the victim LVLM, without the knowledge of its prompts (which are often proprietary for service providers and not publicly available) and its underlying large language model (LLM). This practical setting poses challenges to the cross-prompt and cross-model transferability of targeted adversarial attack, which aims to confuse the LVLM to output a response that is semantically similar to the attacker's chosen target text. To this end, we propose an instruction-tuned targeted attack (dubbed InstructTA) to deliver the targeted adversarial attack on LVLMs with high transferability. Initially, we utilize a public text-to-image generative model to "reverse" the target response into a target image, and employ GPT-4 to infer a reasonable instruction $\boldsymbol{p}^\prime$ from the target response. We then form a local surrogate model (sharing the same visual encoder with the victim LVLM) to extract instruction-aware features of an adversarial image example and the target image, and minimize the distance between these two features to optimize the adversarial example. To further improve the transferability, we augment the instruction $\boldsymbol{p}^\prime$ with instructions paraphrased from an LLM. Extensive experiments demonstrate the superiority of our proposed method in targeted attack performance and transferability.



## **45. Mining Temporal Attack Patterns from Cyberthreat Intelligence Reports**

cs.CR

A modified version of this pre-print is submitted to IEEE  Transactions on Software Engineering, and is under review

**SubmitDate**: 2024-01-03    [abs](http://arxiv.org/abs/2401.01883v1) [paper-pdf](http://arxiv.org/pdf/2401.01883v1)

**Authors**: Md Rayhanur Rahman, Brandon Wroblewski, Quinn Matthews, Brantley Morgan, Tim Menzies, Laurie Williams

**Abstract**: Defending from cyberattacks requires practitioners to operate on high-level adversary behavior. Cyberthreat intelligence (CTI) reports on past cyberattack incidents describe the chain of malicious actions with respect to time. To avoid repeating cyberattack incidents, practitioners must proactively identify and defend against recurring chain of actions - which we refer to as temporal attack patterns. Automatically mining the patterns among actions provides structured and actionable information on the adversary behavior of past cyberattacks. The goal of this paper is to aid security practitioners in prioritizing and proactive defense against cyberattacks by mining temporal attack patterns from cyberthreat intelligence reports. To this end, we propose ChronoCTI, an automated pipeline for mining temporal attack patterns from cyberthreat intelligence (CTI) reports of past cyberattacks. To construct ChronoCTI, we build the ground truth dataset of temporal attack patterns and apply state-of-the-art large language models, natural language processing, and machine learning techniques. We apply ChronoCTI on a set of 713 CTI reports, where we identify 124 temporal attack patterns - which we categorize into nine pattern categories. We identify that the most prevalent pattern category is to trick victim users into executing malicious code to initiate the attack, followed by bypassing the anti-malware system in the victim network. Based on the observed patterns, we advocate organizations to train users about cybersecurity best practices, introduce immutable operating systems with limited functionalities, and enforce multi-user authentications. Moreover, we advocate practitioners to leverage the automated mining capability of ChronoCTI and design countermeasures against the recurring attack patterns.



## **46. Safety and Performance, Why Not Both? Bi-Objective Optimized Model Compression against Heterogeneous Attacks Toward AI Software Deployment**

cs.AI

Accepted by IEEE Transactions on Software Engineering (TSE).  Camera-ready Version. arXiv admin note: substantial text overlap with  arXiv:2208.05969

**SubmitDate**: 2024-01-02    [abs](http://arxiv.org/abs/2401.00996v1) [paper-pdf](http://arxiv.org/pdf/2401.00996v1)

**Authors**: Jie Zhu, Leye Wang, Xiao Han, Anmin Liu, Tao Xie

**Abstract**: The size of deep learning models in artificial intelligence (AI) software is increasing rapidly, hindering the large-scale deployment on resource-restricted devices (e.g., smartphones). To mitigate this issue, AI software compression plays a crucial role, which aims to compress model size while keeping high performance. However, the intrinsic defects in a big model may be inherited by the compressed one. Such defects may be easily leveraged by adversaries, since a compressed model is usually deployed in a large number of devices without adequate protection. In this article, we aim to address the safe model compression problem from the perspective of safety-performance co-optimization. Specifically, inspired by the test-driven development (TDD) paradigm in software engineering, we propose a test-driven sparse training framework called SafeCompress. By simulating the attack mechanism as safety testing, SafeCompress can automatically compress a big model to a small one following the dynamic sparse training paradigm. Then, considering two kinds of representative and heterogeneous attack mechanisms, i.e., black-box membership inference attack and white-box membership inference attack, we develop two concrete instances called BMIA-SafeCompress and WMIA-SafeCompress. Further, we implement another instance called MMIA-SafeCompress by extending SafeCompress to defend against the occasion when adversaries conduct black-box and white-box membership inference attacks simultaneously. We conduct extensive experiments on five datasets for both computer vision and natural language processing tasks. The results show the effectiveness and generalizability of our framework. We also discuss how to adapt SafeCompress to other attacks besides membership inference attack, demonstrating the flexibility of SafeCompress.



## **47. Detection and Defense Against Prominent Attacks on Preconditioned LLM-Integrated Virtual Assistants**

cs.CR

Accepted to be published in the Proceedings of the 10th IEEE CSDE  2023, the Asia-Pacific Conference on Computer Science and Data Engineering  2023

**SubmitDate**: 2024-01-02    [abs](http://arxiv.org/abs/2401.00994v1) [paper-pdf](http://arxiv.org/pdf/2401.00994v1)

**Authors**: Chun Fai Chan, Daniel Wankit Yip, Aysan Esmradi

**Abstract**: The emergence of LLM (Large Language Model) integrated virtual assistants has brought about a rapid transformation in communication dynamics. During virtual assistant development, some developers prefer to leverage the system message, also known as an initial prompt or custom prompt, for preconditioning purposes. However, it is important to recognize that an excessive reliance on this functionality raises the risk of manipulation by malicious actors who can exploit it with carefully crafted prompts. Such malicious manipulation poses a significant threat, potentially compromising the accuracy and reliability of the virtual assistant's responses. Consequently, safeguarding the virtual assistants with detection and defense mechanisms becomes of paramount importance to ensure their safety and integrity. In this study, we explored three detection and defense mechanisms aimed at countering attacks that target the system message. These mechanisms include inserting a reference key, utilizing an LLM evaluator, and implementing a Self-Reminder. To showcase the efficacy of these mechanisms, they were tested against prominent attack techniques. Our findings demonstrate that the investigated mechanisms are capable of accurately identifying and counteracting the attacks. The effectiveness of these mechanisms underscores their potential in safeguarding the integrity and reliability of virtual assistants, reinforcing the importance of their implementation in real-world scenarios. By prioritizing the security of virtual assistants, organizations can maintain user trust, preserve the integrity of the application, and uphold the high standards expected in this era of transformative technologies.



## **48. A Novel Evaluation Framework for Assessing Resilience Against Prompt Injection Attacks in Large Language Models**

cs.CR

Accepted to be published in the Proceedings of The 10th IEEE CSDE  2023, the Asia-Pacific Conference on Computer Science and Data Engineering  2023

**SubmitDate**: 2024-01-02    [abs](http://arxiv.org/abs/2401.00991v1) [paper-pdf](http://arxiv.org/pdf/2401.00991v1)

**Authors**: Daniel Wankit Yip, Aysan Esmradi, Chun Fai Chan

**Abstract**: Prompt injection attacks exploit vulnerabilities in large language models (LLMs) to manipulate the model into unintended actions or generate malicious content. As LLM integrated applications gain wider adoption, they face growing susceptibility to such attacks. This study introduces a novel evaluation framework for quantifying the resilience of applications. The framework incorporates innovative techniques designed to ensure representativeness, interpretability, and robustness. To ensure the representativeness of simulated attacks on the application, a meticulous selection process was employed, resulting in 115 carefully chosen attacks based on coverage and relevance. For enhanced interpretability, a second LLM was utilized to evaluate the responses generated from these simulated attacks. Unlike conventional malicious content classifiers that provide only a confidence score, the LLM-based evaluation produces a score accompanied by an explanation, thereby enhancing interpretability. Subsequently, a resilience score is computed by assigning higher weights to attacks with greater impact, thus providing a robust measurement of the application resilience. To assess the framework's efficacy, it was applied on two LLMs, namely Llama2 and ChatGLM. Results revealed that Llama2, the newer model exhibited higher resilience compared to ChatGLM. This finding substantiates the effectiveness of the framework, aligning with the prevailing notion that newer models tend to possess greater resilience. Moreover, the framework exhibited exceptional versatility, requiring only minimal adjustments to accommodate emerging attack techniques and classifications, thereby establishing itself as an effective and practical solution. Overall, the framework offers valuable insights that empower organizations to make well-informed decisions to fortify their applications against potential threats from prompt injection.



## **49. Opening A Pandora's Box: Things You Should Know in the Era of Custom GPTs**

cs.CR

**SubmitDate**: 2023-12-31    [abs](http://arxiv.org/abs/2401.00905v1) [paper-pdf](http://arxiv.org/pdf/2401.00905v1)

**Authors**: Guanhong Tao, Siyuan Cheng, Zhuo Zhang, Junmin Zhu, Guangyu Shen, Xiangyu Zhang

**Abstract**: The emergence of large language models (LLMs) has significantly accelerated the development of a wide range of applications across various fields. There is a growing trend in the construction of specialized platforms based on LLMs, such as the newly introduced custom GPTs by OpenAI. While custom GPTs provide various functionalities like web browsing and code execution, they also introduce significant security threats. In this paper, we conduct a comprehensive analysis of the security and privacy issues arising from the custom GPT platform. Our systematic examination categorizes potential attack scenarios into three threat models based on the role of the malicious actor, and identifies critical data exchange channels in custom GPTs. Utilizing the STRIDE threat modeling framework, we identify 26 potential attack vectors, with 19 being partially or fully validated in real-world settings. Our findings emphasize the urgent need for robust security and privacy measures in the custom GPT ecosystem, especially in light of the forthcoming launch of the official GPT store by OpenAI.



## **50. Identifying and Mitigating the Security Risks of Generative AI**

cs.AI

**SubmitDate**: 2023-12-29    [abs](http://arxiv.org/abs/2308.14840v4) [paper-pdf](http://arxiv.org/pdf/2308.14840v4)

**Authors**: Clark Barrett, Brad Boyd, Elie Burzstein, Nicholas Carlini, Brad Chen, Jihye Choi, Amrita Roy Chowdhury, Mihai Christodorescu, Anupam Datta, Soheil Feizi, Kathleen Fisher, Tatsunori Hashimoto, Dan Hendrycks, Somesh Jha, Daniel Kang, Florian Kerschbaum, Eric Mitchell, John Mitchell, Zulfikar Ramzan, Khawaja Shams, Dawn Song, Ankur Taly, Diyi Yang

**Abstract**: Every major technical invention resurfaces the dual-use dilemma -- the new technology has the potential to be used for good as well as for harm. Generative AI (GenAI) techniques, such as large language models (LLMs) and diffusion models, have shown remarkable capabilities (e.g., in-context learning, code-completion, and text-to-image generation and editing). However, GenAI can be used just as well by attackers to generate new attacks and increase the velocity and efficacy of existing attacks.   This paper reports the findings of a workshop held at Google (co-organized by Stanford University and the University of Wisconsin-Madison) on the dual-use dilemma posed by GenAI. This paper is not meant to be comprehensive, but is rather an attempt to synthesize some of the interesting findings from the workshop. We discuss short-term and long-term goals for the community on this topic. We hope this paper provides both a launching point for a discussion on this important topic as well as interesting problems that the research community can work to address.



