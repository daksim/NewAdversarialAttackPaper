# Latest Adversarial Attack Papers
**update at 2025-01-21 09:53:01**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. The Effect of Similarity Measures on Accurate Stability Estimates for Local Surrogate Models in Text-based Explainable AI**

cs.LG

11 pages, 8 Tables (Minor edits for clarity and grammar)

**SubmitDate**: 2025-01-17    [abs](http://arxiv.org/abs/2406.15839v2) [paper-pdf](http://arxiv.org/pdf/2406.15839v2)

**Authors**: Christopher Burger, Charles Walter, Thai Le

**Abstract**: Recent work has investigated the vulnerability of local surrogate methods to adversarial perturbations on a machine learning (ML) model's inputs, where the explanation is manipulated while the meaning and structure of the original input remains similar under the complex model. Although weaknesses across many methods have been shown to exist, the reasons behind why remain little explored. Central to the concept of adversarial attacks on explainable AI (XAI) is the similarity measure used to calculate how one explanation differs from another. A poor choice of similarity measure can lead to erroneous conclusions on the efficacy of an XAI method. Too sensitive a measure results in exaggerated vulnerability, while too coarse understates its weakness. We investigate a variety of similarity measures designed for text-based ranked lists, including Kendall's Tau, Spearman's Footrule, and Rank-biased Overlap to determine how substantial changes in the type of measure or threshold of success affect the conclusions generated from common adversarial attack processes. Certain measures are found to be overly sensitive, resulting in erroneous estimates of stability.



## **2. Jailbreaking as a Reward Misspecification Problem**

cs.LG

**SubmitDate**: 2025-01-17    [abs](http://arxiv.org/abs/2406.14393v4) [paper-pdf](http://arxiv.org/pdf/2406.14393v4)

**Authors**: Zhihui Xie, Jiahui Gao, Lei Li, Zhenguo Li, Qi Liu, Lingpeng Kong

**Abstract**: The widespread adoption of large language models (LLMs) has raised concerns about their safety and reliability, particularly regarding their vulnerability to adversarial attacks. In this paper, we propose a novel perspective that attributes this vulnerability to reward misspecification during the alignment process. This misspecification occurs when the reward function fails to accurately capture the intended behavior, leading to misaligned model outputs. We introduce a metric ReGap to quantify the extent of reward misspecification and demonstrate its effectiveness and robustness in detecting harmful backdoor prompts. Building upon these insights, we present ReMiss, a system for automated red teaming that generates adversarial prompts in a reward-misspecified space. ReMiss achieves state-of-the-art attack success rates on the AdvBench benchmark against various target aligned LLMs while preserving the human readability of the generated prompts. Furthermore, these attacks on open-source models demonstrate high transferability to closed-source models like GPT-4o and out-of-distribution tasks from HarmBench. Detailed analysis highlights the unique advantages of the proposed reward misspecification objective compared to previous methods, offering new insights for improving LLM safety and robustness.



## **3. Michscan: Black-Box Neural Network Integrity Checking at Runtime Through Power Analysis**

cs.CR

11 pages, 7 figures. To appear in IEEE International Symposium on  Hardware Oriented Security and Trust (HOST) 2025. This material is based upon  work supported by the National Science Foundation under Grant No. 2245573

**SubmitDate**: 2025-01-17    [abs](http://arxiv.org/abs/2501.10174v1) [paper-pdf](http://arxiv.org/pdf/2501.10174v1)

**Authors**: Robi Paul, Michael Zuzak

**Abstract**: As neural networks are increasingly used for critical decision-making tasks, the threat of integrity attacks, where an adversary maliciously alters a model, has become a significant security and safety concern. These concerns are compounded by the use of licensed models, where end-users purchase third-party models with only black-box access to protect model intellectual property (IP). In such scenarios, conventional approaches to verify model integrity require knowledge of model parameters or cooperative model owners. To address this challenge, we propose Michscan, a methodology leveraging power analysis to verify the integrity of black-box TinyML neural networks designed for resource-constrained devices. Michscan is based on the observation that modifications to model parameters impact the instantaneous power consumption of the device. We leverage this observation to develop a runtime model integrity-checking methodology that employs correlational power analysis using a golden template or signature to mathematically quantify the likelihood of model integrity violations at runtime through the Mann-Whitney U-Test. Michscan operates in a black-box environment and does not require a cooperative or trustworthy model owner. We evaluated Michscan using an STM32F303RC microcontroller with an ARM Cortex-M4 running four TinyML models in the presence of three model integrity violations. Michscan successfully detected all integrity violations at runtime using power data from five inferences. All detected violations had a negligible probability P < 10^(-5) of being produced from an unmodified model (i.e., false positive).



## **4. Generative AI in Cybersecurity: A Comprehensive Review of LLM Applications and Vulnerabilities**

cs.CR

52 pages, 8 figures

**SubmitDate**: 2025-01-17    [abs](http://arxiv.org/abs/2405.12750v2) [paper-pdf](http://arxiv.org/pdf/2405.12750v2)

**Authors**: Mohamed Amine Ferrag, Fatima Alwahedi, Ammar Battah, Bilel Cherif, Abdechakour Mechri, Norbert Tihanyi, Tamas Bisztray, Merouane Debbah

**Abstract**: This paper provides a comprehensive review of the future of cybersecurity through Generative AI and Large Language Models (LLMs). We explore LLM applications across various domains, including hardware design security, intrusion detection, software engineering, design verification, cyber threat intelligence, malware detection, and phishing detection. We present an overview of LLM evolution and its current state, focusing on advancements in models such as GPT-4, GPT-3.5, Mixtral-8x7B, BERT, Falcon2, and LLaMA. Our analysis extends to LLM vulnerabilities, such as prompt injection, insecure output handling, data poisoning, DDoS attacks, and adversarial instructions. We delve into mitigation strategies to protect these models, providing a comprehensive look at potential attack scenarios and prevention techniques. Furthermore, we evaluate the performance of 42 LLM models in cybersecurity knowledge and hardware security, highlighting their strengths and weaknesses. We thoroughly evaluate cybersecurity datasets for LLM training and testing, covering the lifecycle from data creation to usage and identifying gaps for future research. In addition, we review new strategies for leveraging LLMs, including techniques like Half-Quadratic Quantization (HQQ), Reinforcement Learning with Human Feedback (RLHF), Direct Preference Optimization (DPO), Quantized Low-Rank Adapters (QLoRA), and Retrieval-Augmented Generation (RAG). These insights aim to enhance real-time cybersecurity defenses and improve the sophistication of LLM applications in threat detection and response. Our paper provides a foundational understanding and strategic direction for integrating LLMs into future cybersecurity frameworks, emphasizing innovation and robust model deployment to safeguard against evolving cyber threats.



## **5. CaFA: Cost-aware, Feasible Attacks With Database Constraints Against Neural Tabular Classifiers**

cs.CR

Accepted at IEEE S&P 2024

**SubmitDate**: 2025-01-17    [abs](http://arxiv.org/abs/2501.10013v1) [paper-pdf](http://arxiv.org/pdf/2501.10013v1)

**Authors**: Matan Ben-Tov, Daniel Deutch, Nave Frost, Mahmood Sharif

**Abstract**: This work presents CaFA, a system for Cost-aware Feasible Attacks for assessing the robustness of neural tabular classifiers against adversarial examples realizable in the problem space, while minimizing adversaries' effort. To this end, CaFA leverages TabPGD$-$an algorithm we set forth to generate adversarial perturbations suitable for tabular data$-$ and incorporates integrity constraints automatically mined by state-of-the-art database methods. After producing adversarial examples in the feature space via TabPGD, CaFA projects them on the mined constraints, leading, in turn, to better attack realizability. We tested CaFA with three datasets and two architectures and found, among others, that the constraints we use are of higher quality (measured via soundness and completeness) than ones employed in prior work. Moreover, CaFA achieves higher feasible success rates$-$i.e., it generates adversarial examples that are often misclassified while satisfying constraints$-$than prior attacks while simultaneously perturbing few features with lower magnitudes, thus saving effort and improving inconspicuousness. We open-source CaFA, hoping it will serve as a generic system enabling machine-learning engineers to assess their models' robustness against realizable attacks, thus advancing deployed models' trustworthiness.



## **6. Computing Optimization-Based Prompt Injections Against Closed-Weights Models By Misusing a Fine-Tuning API**

cs.CR

**SubmitDate**: 2025-01-16    [abs](http://arxiv.org/abs/2501.09798v1) [paper-pdf](http://arxiv.org/pdf/2501.09798v1)

**Authors**: Andrey Labunets, Nishit V. Pandya, Ashish Hooda, Xiaohan Fu, Earlence Fernandes

**Abstract**: We surface a new threat to closed-weight Large Language Models (LLMs) that enables an attacker to compute optimization-based prompt injections. Specifically, we characterize how an attacker can leverage the loss-like information returned from the remote fine-tuning interface to guide the search for adversarial prompts. The fine-tuning interface is hosted by an LLM vendor and allows developers to fine-tune LLMs for their tasks, thus providing utility, but also exposes enough information for an attacker to compute adversarial prompts. Through an experimental analysis, we characterize the loss-like values returned by the Gemini fine-tuning API and demonstrate that they provide a useful signal for discrete optimization of adversarial prompts using a greedy search algorithm. Using the PurpleLlama prompt injection benchmark, we demonstrate attack success rates between 65% and 82% on Google's Gemini family of LLMs. These attacks exploit the classic utility-security tradeoff - the fine-tuning interface provides a useful feature for developers but also exposes the LLMs to powerful attacks.



## **7. Adversarial-Ensemble Kolmogorov Arnold Networks for Enhancing Indoor Wi-Fi Positioning: A Defensive Approach Against Spoofing and Signal Manipulation Attacks**

cs.LG

**SubmitDate**: 2025-01-16    [abs](http://arxiv.org/abs/2501.09609v1) [paper-pdf](http://arxiv.org/pdf/2501.09609v1)

**Authors**: Mitul Goswami, Romit Chatterjee, Somnath Mahato, Prasant Kumar Pattnaik

**Abstract**: The research presents a study on enhancing the robustness of Wi-Fi-based indoor positioning systems against adversarial attacks. The goal is to improve the positioning accuracy and resilience of these systems under two attack scenarios: Wi-Fi Spoofing and Signal Strength Manipulation. Three models are developed and evaluated: a baseline model (M_Base), an adversarially trained robust model (M_Rob), and an ensemble model (M_Ens). All models utilize a Kolmogorov-Arnold Network (KAN) architecture. The robust model is trained with adversarially perturbed data, while the ensemble model combines predictions from both the base and robust models. Experimental results show that the robust model reduces positioning error by approximately 10% compared to the baseline, achieving 2.03 meters error under Wi-Fi spoofing and 2.00 meters under signal strength manipulation. The ensemble model further outperforms with errors of 2.01 meters and 1.975 meters for the respective attack types. This analysis highlights the effectiveness of adversarial training techniques in mitigating attack impacts. The findings underscore the importance of considering adversarial scenarios in developing indoor positioning systems, as improved resilience can significantly enhance the accuracy and reliability of such systems in mission-critical environments.



## **8. On the uncertainty principle of neural networks**

cs.LG

8 pages, 5 figures

**SubmitDate**: 2025-01-16    [abs](http://arxiv.org/abs/2205.01493v4) [paper-pdf](http://arxiv.org/pdf/2205.01493v4)

**Authors**: Jun-Jie Zhang, Dong-Xiao Zhang, Jian-Nan Chen, Long-Gang Pang, Deyu Meng

**Abstract**: In this study, we explore the inherent trade-off between accuracy and robustness in neural networks, drawing an analogy to the uncertainty principle in quantum mechanics. We propose that neural networks are subject to an uncertainty relation, which manifests as a fundamental limitation in their ability to simultaneously achieve high accuracy and robustness against adversarial attacks. Through mathematical proofs and empirical evidence, we demonstrate that this trade-off is a natural consequence of the sharp boundaries formed between different class concepts during training. Our findings reveal that the complementarity principle, a cornerstone of quantum physics, applies to neural networks, imposing fundamental limits on their capabilities in simultaneous learning of conjugate features. Meanwhile, our work suggests that achieving human-level intelligence through a single network architecture or massive datasets alone may be inherently limited. Our work provides new insights into the theoretical foundations of neural network vulnerability and opens up avenues for designing more robust neural network architectures.



## **9. Towards an End-to-End (E2E) Adversarial Learning and Application in the Physical World**

cs.CV

**SubmitDate**: 2025-01-16    [abs](http://arxiv.org/abs/2501.08258v2) [paper-pdf](http://arxiv.org/pdf/2501.08258v2)

**Authors**: Dudi Biton, Jacob Shams, Satoru Koda, Asaf Shabtai, Yuval Elovici, Ben Nassi

**Abstract**: The traditional learning process of patch-based adversarial attacks, conducted in the digital domain and then applied in the physical domain (e.g., via printed stickers), may suffer from reduced performance due to adversarial patches' limited transferability from the digital domain to the physical domain. Given that previous studies have considered using projectors to apply adversarial attacks, we raise the following question: can adversarial learning (i.e., patch generation) be performed entirely in the physical domain with a projector? In this work, we propose the Physical-domain Adversarial Patch Learning Augmentation (PAPLA) framework, a novel end-to-end (E2E) framework that converts adversarial learning from the digital domain to the physical domain using a projector. We evaluate PAPLA across multiple scenarios, including controlled laboratory settings and realistic outdoor environments, demonstrating its ability to ensure attack success compared to conventional digital learning-physical application (DL-PA) methods. We also analyze the impact of environmental factors, such as projection surface color, projector strength, ambient light, distance, and angle of the target object relative to the camera, on the effectiveness of projected patches. Finally, we demonstrate the feasibility of the attack against a parked car and a stop sign in a real-world outdoor environment. Our results show that under specific conditions, E2E adversarial learning in the physical domain eliminates the transferability issue and ensures evasion by object detectors. Finally, we provide insights into the challenges and opportunities of applying adversarial learning in the physical domain and explain where such an approach is more effective than using a sticker.



## **10. Direct Unlearning Optimization for Robust and Safe Text-to-Image Models**

cs.CV

This paper has been accepted for NeurIPS 2024

**SubmitDate**: 2025-01-16    [abs](http://arxiv.org/abs/2407.21035v2) [paper-pdf](http://arxiv.org/pdf/2407.21035v2)

**Authors**: Yong-Hyun Park, Sangdoo Yun, Jin-Hwa Kim, Junho Kim, Geonhui Jang, Yonghyun Jeong, Junghyo Jo, Gayoung Lee

**Abstract**: Recent advancements in text-to-image (T2I) models have unlocked a wide range of applications but also present significant risks, particularly in their potential to generate unsafe content. To mitigate this issue, researchers have developed unlearning techniques to remove the model's ability to generate potentially harmful content. However, these methods are easily bypassed by adversarial attacks, making them unreliable for ensuring the safety of generated images. In this paper, we propose Direct Unlearning Optimization (DUO), a novel framework for removing Not Safe For Work (NSFW) content from T2I models while preserving their performance on unrelated topics. DUO employs a preference optimization approach using curated paired image data, ensuring that the model learns to remove unsafe visual concepts while retaining unrelated features. Furthermore, we introduce an output-preserving regularization term to maintain the model's generative capabilities on safe content. Extensive experiments demonstrate that DUO can robustly defend against various state-of-the-art red teaming methods without significant performance degradation on unrelated topics, as measured by FID and CLIP scores. Our work contributes to the development of safer and more reliable T2I models, paving the way for their responsible deployment in both closed-source and open-source scenarios.



## **11. TPIA: Towards Target-specific Prompt Injection Attack against Code-oriented Large Language Models**

cs.CR

**SubmitDate**: 2025-01-16    [abs](http://arxiv.org/abs/2407.09164v4) [paper-pdf](http://arxiv.org/pdf/2407.09164v4)

**Authors**: Yuchen Yang, Hongwei Yao, Bingrun Yang, Yiling He, Yiming Li, Tianwei Zhang, Zhan Qin, Kui Ren, Chun Chen

**Abstract**: Recently, code-oriented large language models (Code LLMs) have been widely exploited to simplify and facilitate programming. With these tools, developers can easily generate the desired complete functional code based on incomplete code snippets and natural language prompts. Unfortunately, a few pioneering works revealed that these Code LLMs are vulnerable to backdoor and adversarial attacks. The former poisons the training data or model parameters, hijacking the LLMs to generate malicious code snippets when encountering the trigger. The latter crafts malicious adversarial input codes to reduce the quality of the generated codes. However, both attacks have some inherent limitations: backdoor attacks rely on the adversary's capability of controlling the model training process; adversarial attacks struggle with fulfilling specific malicious purposes. This paper presents a novel attack paradigm against Code LLMs, namely target-specific prompt injection attack (TPIA). TPIA generates non-functional perturbations containing the information of malicious instructions and inserts them into the victim's code context by spreading them into potentially used dependencies (e.g., packages or RAG's knowledge base). It induces the Code LLMs to generate attacker-specified malicious code snippets at the target location. In general, we compress the attacker-specified malicious objective into the perturbation by adversarial optimization based on greedy token search. We collect 13 representative malicious objectives to design 31 threat cases for three popular programming languages. We show that our TPIA can successfully attack three representative open-source Code LLMs (with an ASR of up to 97.9%) and two mainstream commercial Code LLM-integrated applications (with an ASR of over 90%) in all threat cases, using only a 12-token perturbation. Our work alerts a new practical threat of using Code LLMs.



## **12. Jodes: Efficient Oblivious Join in the Distributed Setting**

cs.CR

**SubmitDate**: 2025-01-16    [abs](http://arxiv.org/abs/2501.09334v1) [paper-pdf](http://arxiv.org/pdf/2501.09334v1)

**Authors**: Yilei Wang, Xiangdong Zeng, Sheng Wang, Feifei Li

**Abstract**: Trusted execution environment (TEE) has provided an isolated and secure environment for building cloud-based analytic systems, but it still suffers from access pattern leakages caused by side-channel attacks. To better secure the data, computation inside TEE enclave should be made oblivious, which introduces significant overhead and severely slows down the computation. A natural way to speed up is to build the analytic system with multiple servers in the distributed setting. However, this setting raises a new security concern -- the volumes of the transmissions among these servers can leak sensitive information to a network adversary. Existing works have designed specialized algorithms to address this concern, but their supports for equi-join, one of the most important but non-trivial database operators, are either inefficient, limited, or under a weak security assumption.   In this paper, we present Jodes, an efficient oblivious join algorithm in the distributed setting. Jodes prevents the leakage on both the network and enclave sides, supports a general equi-join operation, and provides a high security level protection that only publicizes the input sizes and the output size. Meanwhile, it achieves both communication cost and computation cost asymptotically superior to existing algorithms. To demonstrate the practicality of Jodes, we conduct experiments in the distributed setting comprising 16 servers. Empirical results show that Jodes achieves up to a sixfold performance improvement over state-of-the-art join algorithms.



## **13. Cooperative Decentralized Backdoor Attacks on Vertical Federated Learning**

cs.LG

This paper is currently under review in the IEEE/ACM Transactions on  Networking Special Issue on AI and Networking

**SubmitDate**: 2025-01-16    [abs](http://arxiv.org/abs/2501.09320v1) [paper-pdf](http://arxiv.org/pdf/2501.09320v1)

**Authors**: Seohyun Lee, Wenzhi Fang, Anindya Bijoy Das, Seyyedali Hosseinalipour, David J. Love, Christopher G. Brinton

**Abstract**: Federated learning (FL) is vulnerable to backdoor attacks, where adversaries alter model behavior on target classification labels by embedding triggers into data samples. While these attacks have received considerable attention in horizontal FL, they are less understood for vertical FL (VFL), where devices hold different features of the samples, and only the server holds the labels. In this work, we propose a novel backdoor attack on VFL which (i) does not rely on gradient information from the server and (ii) considers potential collusion among multiple adversaries for sample selection and trigger embedding. Our label inference model augments variational autoencoders with metric learning, which adversaries can train locally. A consensus process over the adversary graph topology determines which datapoints to poison. We further propose methods for trigger splitting across the adversaries, with an intensity-based implantation scheme skewing the server towards the trigger. Our convergence analysis reveals the impact of backdoor perturbations on VFL indicated by a stationarity gap for the trained model, which we verify empirically as well. We conduct experiments comparing our attack with recent backdoor VFL approaches, finding that ours obtains significantly higher success rates for the same main task performance despite not using server information. Additionally, our results verify the impact of collusion on attack performance.



## **14. Salient Information Preserving Adversarial Training Improves Clean and Robust Accuracy**

cs.CV

**SubmitDate**: 2025-01-15    [abs](http://arxiv.org/abs/2501.09086v1) [paper-pdf](http://arxiv.org/pdf/2501.09086v1)

**Authors**: Timothy Redgrave, Adam Czajka

**Abstract**: In this work we introduce Salient Information Preserving Adversarial Training (SIP-AT), an intuitive method for relieving the robustness-accuracy trade-off incurred by traditional adversarial training. SIP-AT uses salient image regions to guide the adversarial training process in such a way that fragile features deemed meaningful by an annotator remain unperturbed during training, allowing models to learn highly predictive non-robust features without sacrificing overall robustness. This technique is compatible with both human-based and automatically generated salience estimates, allowing SIP-AT to be used as a part of human-driven model development without forcing SIP-AT to be reliant upon additional human data. We perform experiments across multiple datasets and architectures and demonstrate that SIP-AT is able to boost the clean accuracy of models while maintaining a high degree of robustness against attacks at multiple epsilon levels. We complement our central experiments with an observational study measuring the rate at which human subjects successfully identify perturbed images. This study helps build a more intuitive understanding of adversarial attack strength and demonstrates the heightened importance of low-epsilon robustness. Our results demonstrate the efficacy of SIP-AT and provide valuable insight into the risks posed by adversarial samples of various strengths.



## **15. Improving Stability Estimates in Adversarial Explainable AI through Alternate Search Methods**

cs.LG

9 pages, 3 figures, 5 tables. arXiv admin note: text overlap with  arXiv:2406.15839

**SubmitDate**: 2025-01-15    [abs](http://arxiv.org/abs/2501.09006v1) [paper-pdf](http://arxiv.org/pdf/2501.09006v1)

**Authors**: Christopher Burger, Charles Walter

**Abstract**: Advances in the effectiveness of machine learning models have come at the cost of enormous complexity resulting in a poor understanding of how they function. Local surrogate methods have been used to approximate the workings of these complex models, but recent work has revealed their vulnerability to adversarial attacks where the explanation produced is appreciably different while the meaning and structure of the complex model's output remains similar. This prior work has focused on the existence of these weaknesses but not on their magnitude. Here we explore using an alternate search method with the goal of finding minimum viable perturbations, the fewest perturbations necessary to achieve a fixed similarity value between the original and altered text's explanation. Intuitively, a method that requires fewer perturbations to expose a given level of instability is inferior to one which requires more. This nuance allows for superior comparisons of the stability of explainability methods.



## **16. UIFV: Data Reconstruction Attack in Vertical Federated Learning**

cs.LG

**SubmitDate**: 2025-01-14    [abs](http://arxiv.org/abs/2406.12588v2) [paper-pdf](http://arxiv.org/pdf/2406.12588v2)

**Authors**: Jirui Yang, Peng Chen, Zhihui Lu, Qiang Duan, Yubing Bao

**Abstract**: Vertical Federated Learning (VFL) facilitates collaborative machine learning without the need for participants to share raw private data. However, recent studies have revealed privacy risks where adversaries might reconstruct sensitive features through data leakage during the learning process. Although data reconstruction methods based on gradient or model information are somewhat effective, they reveal limitations in VFL application scenarios. This is because these traditional methods heavily rely on specific model structures and/or have strict limitations on application scenarios. To address this, our study introduces the Unified InverNet Framework into VFL, which yields a novel and flexible approach (dubbed UIFV) that leverages intermediate feature data to reconstruct original data, instead of relying on gradients or model details. The intermediate feature data is the feature exchanged by different participants during the inference phase of VFL. Experiments on four datasets demonstrate that our methods significantly outperform state-of-the-art techniques in attack precision. Our work exposes severe privacy vulnerabilities within VFL systems that pose real threats to practical VFL applications and thus confirms the necessity of further enhancing privacy protection in the VFL architecture.



## **17. Cross-Modal Transferable Image-to-Video Attack on Video Quality Metrics**

cs.CV

Accepted for VISAPP 2025

**SubmitDate**: 2025-01-14    [abs](http://arxiv.org/abs/2501.08415v1) [paper-pdf](http://arxiv.org/pdf/2501.08415v1)

**Authors**: Georgii Gotin, Ekaterina Shumitskaya, Anastasia Antsiferova, Dmitriy Vatolin

**Abstract**: Recent studies have revealed that modern image and video quality assessment (IQA/VQA) metrics are vulnerable to adversarial attacks. An attacker can manipulate a video through preprocessing to artificially increase its quality score according to a certain metric, despite no actual improvement in visual quality. Most of the attacks studied in the literature are white-box attacks, while black-box attacks in the context of VQA have received less attention. Moreover, some research indicates a lack of transferability of adversarial examples generated for one model to another when applied to VQA. In this paper, we propose a cross-modal attack method, IC2VQA, aimed at exploring the vulnerabilities of modern VQA models. This approach is motivated by the observation that the low-level feature spaces of images and videos are similar. We investigate the transferability of adversarial perturbations across different modalities; specifically, we analyze how adversarial perturbations generated on a white-box IQA model with an additional CLIP module can effectively target a VQA model. The addition of the CLIP module serves as a valuable aid in increasing transferability, as the CLIP model is known for its effective capture of low-level semantics. Extensive experiments demonstrate that IC2VQA achieves a high success rate in attacking three black-box VQA models. We compare our method with existing black-box attack strategies, highlighting its superiority in terms of attack success within the same number of iterations and levels of attack strength. We believe that the proposed method will contribute to the deeper analysis of robust VQA metrics.



## **18. Exploring Robustness of LLMs to Sociodemographically-Conditioned Paraphrasing**

cs.CL

**SubmitDate**: 2025-01-14    [abs](http://arxiv.org/abs/2501.08276v1) [paper-pdf](http://arxiv.org/pdf/2501.08276v1)

**Authors**: Pulkit Arora, Akbar Karimi, Lucie Flek

**Abstract**: Large Language Models (LLMs) have shown impressive performance in various NLP tasks. However, there are concerns about their reliability in different domains of linguistic variations. Many works have proposed robustness evaluation measures for local adversarial attacks, but we need globally robust models unbiased to different language styles. We take a broader approach to explore a wider range of variations across sociodemographic dimensions to perform structured reliability tests on the reasoning capacity of language models. We extend the SocialIQA dataset to create diverse paraphrased sets conditioned on sociodemographic styles. The assessment aims to provide a deeper understanding of LLMs in (a) their capability of generating demographic paraphrases with engineered prompts and (b) their reasoning capabilities in real-world, complex language scenarios. We also explore measures such as perplexity, explainability, and ATOMIC performance of paraphrases for fine-grained reliability analysis of LLMs on these sets. We find that demographic-specific paraphrasing significantly impacts the performance of language models, indicating that the subtleties of language variations remain a significant challenge. The code and dataset will be made available for reproducibility and future research.



## **19. SoK: Design, Vulnerabilities, and Security Measures of Cryptocurrency Wallets**

cs.CR

**SubmitDate**: 2025-01-14    [abs](http://arxiv.org/abs/2307.12874v4) [paper-pdf](http://arxiv.org/pdf/2307.12874v4)

**Authors**: Yimika Erinle, Yathin Kethepalli, Yebo Feng, Jiahua Xu

**Abstract**: With the advent of decentralised digital currencies powered by blockchain technology, a new era of peer-to-peer transactions has commenced. The rapid growth of the cryptocurrency economy has led to increased use of transaction-enabling wallets, making them a focal point for security risks. As the frequency of wallet-related incidents rises, there is a critical need for a systematic approach to measure and evaluate these attacks, drawing lessons from past incidents to enhance wallet security. In response, we introduce a multi-dimensional design taxonomy for existing and novel wallets with various design decisions. We classify existing industry wallets based on this taxonomy, identify previously occurring vulnerabilities and discuss the security implications of design decisions. We also systematise threats to the wallet mechanism and analyse the adversary's goals, capabilities and required knowledge. We present a multi-layered attack framework and investigate 84 incidents between 2012 and 2024, accounting for $5.4B. Following this, we classify defence implementations for these attacks on the precautionary and remedial axes. We map the mechanism and design decisions to vulnerabilities, attacks, and possible defence methods to discuss various insights.



## **20. I Can Find You in Seconds! Leveraging Large Language Models for Code Authorship Attribution**

cs.SE

12 pages, 5 figures,

**SubmitDate**: 2025-01-14    [abs](http://arxiv.org/abs/2501.08165v1) [paper-pdf](http://arxiv.org/pdf/2501.08165v1)

**Authors**: Soohyeon Choi, Yong Kiam Tan, Mark Huasong Meng, Mohamed Ragab, Soumik Mondal, David Mohaisen, Khin Mi Mi Aung

**Abstract**: Source code authorship attribution is important in software forensics, plagiarism detection, and protecting software patch integrity. Existing techniques often rely on supervised machine learning, which struggles with generalization across different programming languages and coding styles due to the need for large labeled datasets. Inspired by recent advances in natural language authorship analysis using large language models (LLMs), which have shown exceptional performance without task-specific tuning, this paper explores the use of LLMs for source code authorship attribution.   We present a comprehensive study demonstrating that state-of-the-art LLMs can successfully attribute source code authorship across different languages. LLMs can determine whether two code snippets are written by the same author with zero-shot prompting, achieving a Matthews Correlation Coefficient (MCC) of 0.78, and can attribute code authorship from a small set of reference code snippets via few-shot learning, achieving MCC of 0.77. Additionally, LLMs show some adversarial robustness against misattribution attacks.   Despite these capabilities, we found that naive prompting of LLMs does not scale well with a large number of authors due to input token limitations. To address this, we propose a tournament-style approach for large-scale attribution. Evaluating this approach on datasets of C++ (500 authors, 26,355 samples) and Java (686 authors, 55,267 samples) code from GitHub, we achieve classification accuracy of up to 65% for C++ and 68.7% for Java using only one reference per author. These results open new possibilities for applying LLMs to code authorship attribution in cybersecurity and software engineering.



## **21. Set-Based Training for Neural Network Verification**

cs.LG

**SubmitDate**: 2025-01-14    [abs](http://arxiv.org/abs/2401.14961v3) [paper-pdf](http://arxiv.org/pdf/2401.14961v3)

**Authors**: Lukas Koller, Tobias Ladner, Matthias Althoff

**Abstract**: Neural networks are vulnerable to adversarial attacks, i.e., small input perturbations can significantly affect the outputs of a neural network. Therefore, to ensure safety of safety-critical environments, the robustness of a neural network must be formally verified against input perturbations, e.g., from noisy sensors. To improve the robustness of neural networks and thus simplify the formal verification, we present a novel set-based training procedure in which we compute the set of possible outputs given the set of possible inputs and compute for the first time a gradient set, i.e., each possible output has a different gradient. Therefore, we can directly reduce the size of the output enclosure by choosing gradients toward its center. Small output enclosures increase the robustness of a neural network and, at the same time, simplify its formal verification. The latter benefit is due to the fact that a larger size of propagated sets increases the conservatism of most verification methods. Our extensive evaluation demonstrates that set-based training produces robust neural networks with competitive performance, which can be verified using fast (polynomial-time) verification algorithms due to the reduced output set.



## **22. Gandalf the Red: Adaptive Security for LLMs**

cs.LG

Niklas Pfister, V\'aclav Volhejn and Manuel Knott contributed equally

**SubmitDate**: 2025-01-14    [abs](http://arxiv.org/abs/2501.07927v1) [paper-pdf](http://arxiv.org/pdf/2501.07927v1)

**Authors**: Niklas Pfister, Václav Volhejn, Manuel Knott, Santiago Arias, Julia Bazińska, Mykhailo Bichurin, Alan Commike, Janet Darling, Peter Dienes, Matthew Fiedler, David Haber, Matthias Kraft, Marco Lancini, Max Mathys, Damián Pascual-Ortiz, Jakub Podolak, Adrià Romero-López, Kyriacos Shiarlis, Andreas Signer, Zsolt Terek, Athanasios Theocharis, Daniel Timbrell, Samuel Trautwein, Samuel Watts, Natalie Wu, Mateo Rojas-Carulla

**Abstract**: Current evaluations of defenses against prompt attacks in large language model (LLM) applications often overlook two critical factors: the dynamic nature of adversarial behavior and the usability penalties imposed on legitimate users by restrictive defenses. We propose D-SEC (Dynamic Security Utility Threat Model), which explicitly separates attackers from legitimate users, models multi-step interactions, and rigorously expresses the security-utility in an optimizable form. We further address the shortcomings in existing evaluations by introducing Gandalf, a crowd-sourced, gamified red-teaming platform designed to generate realistic, adaptive attack datasets. Using Gandalf, we collect and release a dataset of 279k prompt attacks. Complemented by benign user data, our analysis reveals the interplay between security and utility, showing that defenses integrated in the LLM (e.g., system prompts) can degrade usability even without blocking requests. We demonstrate that restricted application domains, defense-in-depth, and adaptive defenses are effective strategies for building secure and useful LLM applications. Code is available at \href{https://github.com/lakeraai/dsec-gandalf}{\texttt{https://github.com/lakeraai/dsec-gandalf}}.



## **23. VENOM: Text-driven Unrestricted Adversarial Example Generation with Diffusion Models**

cs.CV

**SubmitDate**: 2025-01-14    [abs](http://arxiv.org/abs/2501.07922v1) [paper-pdf](http://arxiv.org/pdf/2501.07922v1)

**Authors**: Hui Kuurila-Zhang, Haoyu Chen, Guoying Zhao

**Abstract**: Adversarial attacks have proven effective in deceiving machine learning models by subtly altering input images, motivating extensive research in recent years. Traditional methods constrain perturbations within $l_p$-norm bounds, but advancements in Unrestricted Adversarial Examples (UAEs) allow for more complex, generative-model-based manipulations. Diffusion models now lead UAE generation due to superior stability and image quality over GANs. However, existing diffusion-based UAE methods are limited to using reference images and face challenges in generating Natural Adversarial Examples (NAEs) directly from random noise, often producing uncontrolled or distorted outputs. In this work, we introduce VENOM, the first text-driven framework for high-quality unrestricted adversarial examples generation through diffusion models. VENOM unifies image content generation and adversarial synthesis into a single reverse diffusion process, enabling high-fidelity adversarial examples without sacrificing attack success rate (ASR). To stabilize this process, we incorporate an adaptive adversarial guidance strategy with momentum, ensuring that the generated adversarial examples $x^*$ align with the distribution $p(x)$ of natural images. Extensive experiments demonstrate that VENOM achieves superior ASR and image quality compared to prior methods, marking a significant advancement in adversarial example generation and providing insights into model vulnerabilities for improved defense development.



## **24. Can Go AIs be adversarially robust?**

cs.LG

63 pages, AAAI 2025

**SubmitDate**: 2025-01-14    [abs](http://arxiv.org/abs/2406.12843v3) [paper-pdf](http://arxiv.org/pdf/2406.12843v3)

**Authors**: Tom Tseng, Euan McLean, Kellin Pelrine, Tony T. Wang, Adam Gleave

**Abstract**: Prior work found that superhuman Go AIs can be defeated by simple adversarial strategies, especially "cyclic" attacks. In this paper, we study whether adding natural countermeasures can achieve robustness in Go, a favorable domain for robustness since it benefits from incredible average-case capability and a narrow, innately adversarial setting. We test three defenses: adversarial training on hand-constructed positions, iterated adversarial training, and changing the network architecture. We find that though some of these defenses protect against previously discovered attacks, none withstand freshly trained adversaries. Furthermore, most of the reliably effective attacks these adversaries discover are different realizations of the same overall class of cyclic attacks. Our results suggest that building robust AI systems is challenging even with extremely superhuman systems in some of the most tractable settings, and highlight two key gaps: efficient generalization of defenses, and diversity in training. For interactive examples of attacks and a link to our codebase, see https://goattack.far.ai.



## **25. Don't Command, Cultivate: An Exploratory Study of System-2 Alignment**

cs.CL

In this version, the DPO and reinforcement learning methods have been  added

**SubmitDate**: 2025-01-14    [abs](http://arxiv.org/abs/2411.17075v5) [paper-pdf](http://arxiv.org/pdf/2411.17075v5)

**Authors**: Yuhang Wang, Yuxiang Zhang, Yanxu Zhu, Xinyan Wen, Jitao Sang

**Abstract**: The o1 system card identifies the o1 models as the most robust within OpenAI, with their defining characteristic being the progression from rapid, intuitive thinking to slower, more deliberate reasoning. This observation motivated us to investigate the influence of System-2 thinking patterns on model safety. In our preliminary research, we conducted safety evaluations of the o1 model, including complex jailbreak attack scenarios using adversarial natural language prompts and mathematical encoding prompts. Our findings indicate that the o1 model demonstrates relatively improved safety performance; however, it still exhibits vulnerabilities, particularly against jailbreak attacks employing mathematical encoding. Through detailed case analysis, we identified specific patterns in the o1 model's responses. We also explored the alignment of System-2 safety in open-source models using prompt engineering and supervised fine-tuning techniques. Experimental results show that some simple methods to encourage the model to carefully scrutinize user requests are beneficial for model safety. Additionally, we proposed a implementation plan for process supervision to enhance safety alignment. The implementation details and experimental results will be provided in future versions.



## **26. A Survey of Early Exit Deep Neural Networks in NLP**

cs.LG

**SubmitDate**: 2025-01-13    [abs](http://arxiv.org/abs/2501.07670v1) [paper-pdf](http://arxiv.org/pdf/2501.07670v1)

**Authors**: Divya Jyoti Bajpai, Manjesh Kumar Hanawal

**Abstract**: Deep Neural Networks (DNNs) have grown increasingly large in size to achieve state of the art performance across a wide range of tasks. However, their high computational requirements make them less suitable for resource-constrained applications. Also, real-world datasets often consist of a mixture of easy and complex samples, necessitating adaptive inference mechanisms that account for sample difficulty. Early exit strategies offer a promising solution by enabling adaptive inference, where simpler samples are classified using the initial layers of the DNN, thereby accelerating the overall inference process. By attaching classifiers at different layers, early exit methods not only reduce inference latency but also improve the model robustness against adversarial attacks. This paper presents a comprehensive survey of early exit methods and their applications in NLP.



## **27. SecAlign: Defending Against Prompt Injection with Preference Optimization**

cs.CR

Key words: prompt injection defense, LLM security, LLM-integrated  applications

**SubmitDate**: 2025-01-13    [abs](http://arxiv.org/abs/2410.05451v2) [paper-pdf](http://arxiv.org/pdf/2410.05451v2)

**Authors**: Sizhe Chen, Arman Zharmagambetov, Saeed Mahloujifar, Kamalika Chaudhuri, David Wagner, Chuan Guo

**Abstract**: Large language models (LLMs) are becoming increasingly prevalent in modern software systems, interfacing between the user and the Internet to assist with tasks that require advanced language understanding. To accomplish these tasks, the LLM often uses external data sources such as user documents, web retrieval, results from API calls, etc. This opens up new avenues for attackers to manipulate the LLM via prompt injection. Adversarial prompts can be injected into external data sources to override the system's intended instruction and instead execute a malicious instruction.   To mitigate this vulnerability, we propose a new defense called SecAlign based on the technique of preference optimization. Our defense first constructs a preference dataset with prompt-injected inputs, secure outputs (ones that respond to the legitimate instruction), and insecure outputs (ones that respond to the injection). We then perform preference optimization on this dataset to teach the LLM to prefer the secure output over the insecure one. This provides the first known method that reduces the success rates of various prompt injections to around 0%, even against attacks much more sophisticated than ones seen during training. This indicates our defense generalizes well against unknown and yet-to-come attacks. Also, our defended models are still practical with similar utility to the one before our defensive training. Our code is at https://github.com/facebookresearch/SecAlign



## **28. Exploring and Mitigating Adversarial Manipulation of Voting-Based Leaderboards**

cs.LG

**SubmitDate**: 2025-01-13    [abs](http://arxiv.org/abs/2501.07493v1) [paper-pdf](http://arxiv.org/pdf/2501.07493v1)

**Authors**: Yangsibo Huang, Milad Nasr, Anastasios Angelopoulos, Nicholas Carlini, Wei-Lin Chiang, Christopher A. Choquette-Choo, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Ken Ziyu Liu, Ion Stoica, Florian Tramer, Chiyuan Zhang

**Abstract**: It is now common to evaluate Large Language Models (LLMs) by having humans manually vote to evaluate model outputs, in contrast to typical benchmarks that evaluate knowledge or skill at some particular task. Chatbot Arena, the most popular benchmark of this type, ranks models by asking users to select the better response between two randomly selected models (without revealing which model was responsible for the generations). These platforms are widely trusted as a fair and accurate measure of LLM capabilities. In this paper, we show that if bot protection and other defenses are not implemented, these voting-based benchmarks are potentially vulnerable to adversarial manipulation. Specifically, we show that an attacker can alter the leaderboard (to promote their favorite model or demote competitors) at the cost of roughly a thousand votes (verified in a simulated, offline version of Chatbot Arena). Our attack consists of two steps: first, we show how an attacker can determine which model was used to generate a given reply with more than $95\%$ accuracy; and then, the attacker can use this information to consistently vote for (or against) a target model. Working with the Chatbot Arena developers, we identify, propose, and implement mitigations to improve the robustness of Chatbot Arena against adversarial manipulation, which, based on our analysis, substantially increases the cost of such attacks. Some of these defenses were present before our collaboration, such as bot protection with Cloudflare, malicious user detection, and rate limiting. Others, including reCAPTCHA and login are being integrated to strengthen the security in Chatbot Arena.



## **29. Agentic Copyright Watermarking against Adversarial Evidence Forgery with Purification-Agnostic Curriculum Proxy Learning**

cs.CV

**SubmitDate**: 2025-01-13    [abs](http://arxiv.org/abs/2409.01541v2) [paper-pdf](http://arxiv.org/pdf/2409.01541v2)

**Authors**: Erjin Bao, Ching-Chun Chang, Hanrui Wang, Isao Echizen

**Abstract**: With the proliferation of AI agents in various domains, protecting the ownership of AI models has become crucial due to the significant investment in their development. Unauthorized use and illegal distribution of these models pose serious threats to intellectual property, necessitating effective copyright protection measures. Model watermarking has emerged as a key technique to address this issue, embedding ownership information within models to assert rightful ownership during copyright disputes. This paper presents several contributions to model watermarking: a self-authenticating black-box watermarking protocol using hash techniques, a study on evidence forgery attacks using adversarial perturbations, a proposed defense involving a purification step to counter adversarial attacks, and a purification-agnostic curriculum proxy learning method to enhance watermark robustness and model performance. Experimental results demonstrate the effectiveness of these approaches in improving the security, reliability, and performance of watermarked models.



## **30. MOS-Attack: A Scalable Multi-objective Adversarial Attack Framework**

cs.LG

Under Review of CVPR 2025

**SubmitDate**: 2025-01-13    [abs](http://arxiv.org/abs/2501.07251v1) [paper-pdf](http://arxiv.org/pdf/2501.07251v1)

**Authors**: Ping Guo, Cheng Gong, Xi Lin, Fei Liu, Zhichao Lu, Qingfu Zhang, Zhenkun Wang

**Abstract**: Crafting adversarial examples is crucial for evaluating and enhancing the robustness of Deep Neural Networks (DNNs), presenting a challenge equivalent to maximizing a non-differentiable 0-1 loss function.   However, existing single objective methods, namely adversarial attacks focus on a surrogate loss function, do not fully harness the benefits of engaging multiple loss functions, as a result of insufficient understanding of their synergistic and conflicting nature.   To overcome these limitations, we propose the Multi-Objective Set-based Attack (MOS Attack), a novel adversarial attack framework leveraging multiple loss functions and automatically uncovering their interrelations.   The MOS Attack adopts a set-based multi-objective optimization strategy, enabling the incorporation of numerous loss functions without additional parameters.   It also automatically mines synergistic patterns among various losses, facilitating the generation of potent adversarial attacks with fewer objectives.   Extensive experiments have shown that our MOS Attack outperforms single-objective attacks. Furthermore, by harnessing the identified synergistic patterns, MOS Attack continues to show superior results with a reduced number of loss functions.



## **31. An Enhanced Zeroth-Order Stochastic Frank-Wolfe Framework for Constrained Finite-Sum Optimization**

cs.LG

35 pages, 4 figures, 3 tables

**SubmitDate**: 2025-01-13    [abs](http://arxiv.org/abs/2501.07201v1) [paper-pdf](http://arxiv.org/pdf/2501.07201v1)

**Authors**: Haishan Ye, Yinghui Huang, Hao Di, Xiangyu Chang

**Abstract**: We propose an enhanced zeroth-order stochastic Frank-Wolfe framework to address constrained finite-sum optimization problems, a structure prevalent in large-scale machine-learning applications. Our method introduces a novel double variance reduction framework that effectively reduces the gradient approximation variance induced by zeroth-order oracles and the stochastic sampling variance from finite-sum objectives. By leveraging this framework, our algorithm achieves significant improvements in query efficiency, making it particularly well-suited for high-dimensional optimization tasks. Specifically, for convex objectives, the algorithm achieves a query complexity of O(d \sqrt{n}/\epsilon ) to find an epsilon-suboptimal solution, where d is the dimensionality and n is the number of functions in the finite-sum objective. For non-convex objectives, it achieves a query complexity of O(d^{3/2}\sqrt{n}/\epsilon^2 ) without requiring the computation ofd partial derivatives at each iteration. These complexities are the best known among zeroth-order stochastic Frank-Wolfe algorithms that avoid explicit gradient calculations. Empirical experiments on convex and non-convex machine learning tasks, including sparse logistic regression, robust classification, and adversarial attacks on deep networks, validate the computational efficiency and scalability of our approach. Our algorithm demonstrates superior performance in both convergence rate and query complexity compared to existing methods.



## **32. Bitcoin Under Volatile Block Rewards: How Mempool Statistics Can Influence Bitcoin Mining**

cs.CR

**SubmitDate**: 2025-01-13    [abs](http://arxiv.org/abs/2411.11702v2) [paper-pdf](http://arxiv.org/pdf/2411.11702v2)

**Authors**: Roozbeh Sarenche, Alireza Aghabagherloo, Svetla Nikova, Bart Preneel

**Abstract**: The security of Bitcoin protocols is deeply dependent on the incentives provided to miners, which come from a combination of block rewards and transaction fees. As Bitcoin experiences more halving events, the protocol reward converges to zero, making transaction fees the primary source of miner rewards. This shift in Bitcoin's incentivization mechanism, which introduces volatility into block rewards, leads to the emergence of new security threats or intensifies existing ones. Previous security analyses of Bitcoin have either considered a fixed block reward model or a highly simplified volatile model, overlooking the complexities of Bitcoin's mempool behavior.   This paper presents a reinforcement learning-based tool to develop mining strategies under a more realistic volatile model. We employ the Asynchronous Advantage Actor-Critic (A3C) algorithm, which efficiently handles dynamic environments, such as the Bitcoin mempool, to derive near-optimal mining strategies when interacting with an environment that models the complexity of the Bitcoin mempool. This tool enables the analysis of adversarial mining strategies, such as selfish mining and undercutting, both before and after difficulty adjustments, providing insights into the effects of mining attacks in both the short and long term.   We revisit the Bitcoin security threshold presented in the WeRLman paper and demonstrate that the implicit predictability of valuable transaction arrivals in this model leads to an underestimation of the reported threshold. Additionally, we show that, while adversarial strategies like selfish mining under the fixed reward model incur an initial loss period of at least two weeks, the transition toward a transaction-fee era incentivizes mining pools to abandon honest mining for immediate profits. This incentive is expected to become more significant as the protocol reward approaches zero in the future.



## **33. Protego: Detecting Adversarial Examples for Vision Transformers via Intrinsic Capabilities**

cs.CV

Accepted by IEEE MetaCom 2024

**SubmitDate**: 2025-01-13    [abs](http://arxiv.org/abs/2501.07044v1) [paper-pdf](http://arxiv.org/pdf/2501.07044v1)

**Authors**: Jialin Wu, Kaikai Pan, Yanjiao Chen, Jiangyi Deng, Shengyuan Pang, Wenyuan Xu

**Abstract**: Transformer models have excelled in natural language tasks, prompting the vision community to explore their implementation in computer vision problems. However, these models are still influenced by adversarial examples. In this paper, we investigate the attack capabilities of six common adversarial attacks on three pretrained ViT models to reveal the vulnerability of ViT models. To understand and analyse the bias in neural network decisions when the input is adversarial, we use two visualisation techniques that are attention rollout and grad attention rollout. To prevent ViT models from adversarial attack, we propose Protego, a detection framework that leverages the transformer intrinsic capabilities to detection adversarial examples of ViT models. Nonetheless, this is challenging due to a diversity of attack strategies that may be adopted by adversaries. Inspired by the attention mechanism, we know that the token of prediction contains all the information from the input sample. Additionally, the attention region for adversarial examples differs from that of normal examples. Given these points, we can train a detector that achieves superior performance than existing detection methods to identify adversarial examples. Our experiments have demonstrated the high effectiveness of our detection method. For these six adversarial attack methods, our detector's AUC scores all exceed 0.95. Protego may advance investigations in metaverse security.



## **34. Exploring Transferability of Multimodal Adversarial Samples for Vision-Language Pre-training Models with Contrastive Learning**

cs.MM

**SubmitDate**: 2025-01-12    [abs](http://arxiv.org/abs/2308.12636v4) [paper-pdf](http://arxiv.org/pdf/2308.12636v4)

**Authors**: Youze Wang, Wenbo Hu, Yinpeng Dong, Hanwang Zhang, Hang Su, Richang Hong

**Abstract**: The integration of visual and textual data in Vision-Language Pre-training (VLP) models is crucial for enhancing vision-language understanding. However, the adversarial robustness of these models, especially in the alignment of image-text features, has not yet been sufficiently explored. In this paper, we introduce a novel gradient-based multimodal adversarial attack method, underpinned by contrastive learning, to improve the transferability of multimodal adversarial samples in VLP models. This method concurrently generates adversarial texts and images within imperceptive perturbation, employing both image-text and intra-modal contrastive loss. We evaluate the effectiveness of our approach on image-text retrieval and visual entailment tasks, using publicly available datasets in a black-box setting. Extensive experiments indicate a significant advancement over existing single-modal transfer-based adversarial attack methods and current multimodal adversarial attack approaches.



## **35. Mitigating Low-Frequency Bias: Feature Recalibration and Frequency Attention Regularization for Adversarial Robustness**

cs.CV

**SubmitDate**: 2025-01-12    [abs](http://arxiv.org/abs/2407.04016v2) [paper-pdf](http://arxiv.org/pdf/2407.04016v2)

**Authors**: Kejia Zhang, Juanjuan Weng, Yuanzheng Cai, Zhiming Luo, Shaozi Li

**Abstract**: Ensuring the robustness of deep neural networks against adversarial attacks remains a fundamental challenge in computer vision. While adversarial training (AT) has emerged as a promising defense strategy, our analysis reveals a critical limitation: AT-trained models exhibit a bias toward low-frequency features while neglecting high-frequency components. This bias is particularly concerning as each frequency component carries distinct and crucial information: low-frequency features encode fundamental structural patterns, while high-frequency features capture intricate details and textures. To address this limitation, we propose High-Frequency Feature Disentanglement and Recalibration (HFDR), a novel module that strategically separates and recalibrates frequency-specific features to capture latent semantic cues. We further introduce frequency attention regularization to harmonize feature extraction across the frequency spectrum and mitigate the inherent low-frequency bias of AT. Extensive experiments demonstrate our method's superior performance against white-box attacks and transfer attacks, while exhibiting strong generalization capabilities across diverse scenarios.



## **36. ModelShield: Adaptive and Robust Watermark against Model Extraction Attack**

cs.CR

**SubmitDate**: 2025-01-12    [abs](http://arxiv.org/abs/2405.02365v4) [paper-pdf](http://arxiv.org/pdf/2405.02365v4)

**Authors**: Kaiyi Pang, Tao Qi, Chuhan Wu, Minhao Bai, Minghu Jiang, Yongfeng Huang

**Abstract**: Large language models (LLMs) demonstrate general intelligence across a variety of machine learning tasks, thereby enhancing the commercial value of their intellectual property (IP). To protect this IP, model owners typically allow user access only in a black-box manner, however, adversaries can still utilize model extraction attacks to steal the model intelligence encoded in model generation. Watermarking technology offers a promising solution for defending against such attacks by embedding unique identifiers into the model-generated content. However, existing watermarking methods often compromise the quality of generated content due to heuristic alterations and lack robust mechanisms to counteract adversarial strategies, thus limiting their practicality in real-world scenarios. In this paper, we introduce an adaptive and robust watermarking method (named ModelShield) to protect the IP of LLMs. Our method incorporates a self-watermarking mechanism that allows LLMs to autonomously insert watermarks into their generated content to avoid the degradation of model content. We also propose a robust watermark detection mechanism capable of effectively identifying watermark signals under the interference of varying adversarial strategies. Besides, ModelShield is a plug-and-play method that does not require additional model training, enhancing its applicability in LLM deployments. Extensive evaluations on two real-world datasets and three LLMs demonstrate that our method surpasses existing methods in terms of defense effectiveness and robustness while significantly reducing the degradation of watermarking on the model-generated content.



## **37. ZOQO: Zero-Order Quantized Optimization**

cs.LG

Accepted to ICASSP 2025

**SubmitDate**: 2025-01-12    [abs](http://arxiv.org/abs/2501.06736v1) [paper-pdf](http://arxiv.org/pdf/2501.06736v1)

**Authors**: Noga Bar, Raja Giryes

**Abstract**: The increasing computational and memory demands in deep learning present significant challenges, especially in resource-constrained environments. We introduce a zero-order quantized optimization (ZOQO) method designed for training models with quantized parameters and operations. Our approach leverages zero-order approximations of the gradient sign and adapts the learning process to maintain the parameters' quantization without the need for full-precision gradient calculations. We demonstrate the effectiveness of ZOQO through experiments in fine-tuning of large language models and black-box adversarial attacks. Despite the limitations of zero-order and quantized operations training, our method achieves competitive performance compared to full-precision methods, highlighting its potential for low-resource environments.



## **38. Measuring the Robustness of Reference-Free Dialogue Evaluation Systems**

cs.CL

**SubmitDate**: 2025-01-12    [abs](http://arxiv.org/abs/2501.06728v1) [paper-pdf](http://arxiv.org/pdf/2501.06728v1)

**Authors**: Justin Vasselli, Adam Nohejl, Taro Watanabe

**Abstract**: Advancements in dialogue systems powered by large language models (LLMs) have outpaced the development of reliable evaluation metrics, particularly for diverse and creative responses. We present a benchmark for evaluating the robustness of reference-free dialogue metrics against four categories of adversarial attacks: speaker tag prefixes, static responses, ungrammatical responses, and repeated conversational context. We analyze metrics such as DialogRPT, UniEval, and PromptEval -- a prompt-based method leveraging LLMs -- across grounded and ungrounded datasets. By examining both their correlation with human judgment and susceptibility to adversarial attacks, we find that these two axes are not always aligned; metrics that appear to be equivalent when judged by traditional benchmarks may, in fact, vary in their scores of adversarial responses. These findings motivate the development of nuanced evaluation frameworks to address real-world dialogue challenges.



## **39. Towards Adversarially Robust Deep Metric Learning**

cs.LG

**SubmitDate**: 2025-01-12    [abs](http://arxiv.org/abs/2501.01025v2) [paper-pdf](http://arxiv.org/pdf/2501.01025v2)

**Authors**: Xiaopeng Ke

**Abstract**: Deep Metric Learning (DML) has shown remarkable successes in many domains by taking advantage of powerful deep neural networks. Deep neural networks are prone to adversarial attacks and could be easily fooled by adversarial examples. The current progress on this robustness issue is mainly about deep classification models but pays little attention to DML models. Existing works fail to thoroughly inspect the robustness of DML and neglect an important DML scenario, the clustering-based inference. In this work, we first point out the robustness issue of DML models in clustering-based inference scenarios. We find that, for the clustering-based inference, existing defenses designed DML are unable to be reused and the adaptions of defenses designed for deep classification models cannot achieve satisfactory robustness performance. To alleviate the hazard of adversarial examples, we propose a new defense, the Ensemble Adversarial Training (EAT), which exploits ensemble learning and adversarial training. EAT promotes the diversity of the ensemble, encouraging each model in the ensemble to have different robustness features, and employs a self-transferring mechanism to make full use of the robustness statistics of the whole ensemble in the update of every single model. We evaluate the EAT method on three widely-used datasets with two popular model architectures. The results show that the proposed EAT method greatly outperforms the adaptions of defenses designed for deep classification models.



## **40. Slot: Provenance-Driven APT Detection through Graph Reinforcement Learning**

cs.CR

**SubmitDate**: 2025-01-11    [abs](http://arxiv.org/abs/2410.17910v2) [paper-pdf](http://arxiv.org/pdf/2410.17910v2)

**Authors**: Wei Qiao, Yebo Feng, Teng Li, Zhuo Ma, Yulong Shen, JianFeng Ma, Yang Liu

**Abstract**: Advanced Persistent Threats (APTs) represent sophisticated cyberattacks characterized by their ability to remain undetected within the victim system for extended periods, aiming to exfiltrate sensitive data or disrupt operations. Existing detection approaches often struggle to effectively identify these complex threats, construct the attack chain for defense facilitation, or resist adversarial attacks. To overcome these challenges, we propose Slot, an advanced APT detection approach based on provenance graphs and graph reinforcement learning. Slot excels in uncovering multi-level hidden relationships, such as causal, contextual, and indirect connections, among system behaviors through provenance graph mining. By pioneering the integration of graph reinforcement learning, Slot dynamically adapts to new user activities and evolving attack strategies, enhancing its resilience against adversarial attacks. Additionally, Slot automatically constructs the attack chain according to detected attacks with clustering algorithms, providing precise identification of attack paths and facilitating the development of defense strategies. Evaluations with real-world datasets demonstrate Slot's outstanding accuracy, efficiency, adaptability, and robustness in APT detection, with most metrics surpassing state-of-the-art methods. Additionally, case studies conducted to assess Slot's effectiveness in supporting APT defense further establish it as a practical and reliable tool for cybersecurity protection.



## **41. LayerMix: Enhanced Data Augmentation through Fractal Integration for Robust Deep Learning**

cs.CV

**SubmitDate**: 2025-01-11    [abs](http://arxiv.org/abs/2501.04861v2) [paper-pdf](http://arxiv.org/pdf/2501.04861v2)

**Authors**: Hafiz Mughees Ahmad, Dario Morle, Afshin Rahimi

**Abstract**: Deep learning models have demonstrated remarkable performance across various computer vision tasks, yet their vulnerability to distribution shifts remains a critical challenge. Despite sophisticated neural network architectures, existing models often struggle to maintain consistent performance when confronted with Out-of-Distribution (OOD) samples, including natural corruptions, adversarial perturbations, and anomalous patterns. We introduce LayerMix, an innovative data augmentation approach that systematically enhances model robustness through structured fractal-based image synthesis. By meticulously integrating structural complexity into training datasets, our method generates semantically consistent synthetic samples that significantly improve neural network generalization capabilities. Unlike traditional augmentation techniques that rely on random transformations, LayerMix employs a structured mixing pipeline that preserves original image semantics while introducing controlled variability. Extensive experiments across multiple benchmark datasets, including CIFAR-10, CIFAR-100, ImageNet-200, and ImageNet-1K demonstrate LayerMixs superior performance in classification accuracy and substantially enhances critical Machine Learning (ML) safety metrics, including resilience to natural image corruptions, robustness against adversarial attacks, improved model calibration and enhanced prediction consistency. LayerMix represents a significant advancement toward developing more reliable and adaptable artificial intelligence systems by addressing the fundamental challenges of deep learning generalization. The code is available at https://github.com/ahmadmughees/layermix.



## **42. Effective Backdoor Mitigation in Vision-Language Models Depends on the Pre-training Objective**

cs.LG

Accepted at TMLR (https://openreview.net/forum?id=Conma3qnaT)

**SubmitDate**: 2025-01-11    [abs](http://arxiv.org/abs/2311.14948v4) [paper-pdf](http://arxiv.org/pdf/2311.14948v4)

**Authors**: Sahil Verma, Gantavya Bhatt, Avi Schwarzschild, Soumye Singhal, Arnav Mohanty Das, Chirag Shah, John P Dickerson, Pin-Yu Chen, Jeff Bilmes

**Abstract**: Despite the advanced capabilities of contemporary machine learning (ML) models, they remain vulnerable to adversarial and backdoor attacks. This vulnerability is particularly concerning in real-world deployments, where compromised models may exhibit unpredictable behavior in critical scenarios. Such risks are heightened by the prevalent practice of collecting massive, internet-sourced datasets for training multimodal models, as these datasets may harbor backdoors. Various techniques have been proposed to mitigate the effects of backdooring in multimodal models, such as CleanCLIP, which is the current state-of-the-art approach. In this work, we demonstrate that the efficacy of CleanCLIP in mitigating backdoors is highly dependent on the particular objective used during model pre-training. We observe that stronger pre-training objectives that lead to higher zero-shot classification performance correlate with harder to remove backdoors behaviors. We show this by training multimodal models on two large datasets consisting of 3 million (CC3M) and 6 million (CC6M) datapoints, under various pre-training objectives, followed by poison removal using CleanCLIP. We find that CleanCLIP, even with extensive hyperparameter tuning, is ineffective in poison removal when stronger pre-training objectives are used. Our findings underscore critical considerations for ML practitioners who train models using large-scale web-curated data and are concerned about potential backdoor threats.



## **43. Privacy-Preserving Distributed Defense Framework for DC Microgrids Against Exponentially Unbounded False Data Injection Attacks**

eess.SY

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.00588v2) [paper-pdf](http://arxiv.org/pdf/2501.00588v2)

**Authors**: Yi Zhang, Mohamadamin Rajabinezhad, Yichao Wang, Junbo Zhao, Shan Zuo

**Abstract**: This paper introduces a novel, fully distributed control framework for DC microgrids, enhancing resilience against exponentially unbounded false data injection (EU-FDI) attacks. Our framework features a consensus-based secondary control for each converter, effectively addressing these advanced threats. To further safeguard sensitive operational data, a privacy-preserving mechanism is incorporated into the control design, ensuring that critical information remains secure even under adversarial conditions. Rigorous Lyapunov stability analysis confirms the framework's ability to maintain critical DC microgrid operations like voltage regulation and load sharing under EU-FDI threats. The framework's practicality is validated through hardware-in-the-loop experiments, demonstrating its enhanced resilience and robust privacy protection against the complex challenges posed by quick variant FDI attacks.



## **44. Pixel Is Not A Barrier: An Effective Evasion Attack for Pixel-Domain Diffusion Models**

cs.CV

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2408.11810v2) [paper-pdf](http://arxiv.org/pdf/2408.11810v2)

**Authors**: Chun-Yen Shih, Li-Xuan Peng, Jia-Wei Liao, Ernie Chu, Cheng-Fu Chou, Jun-Cheng Chen

**Abstract**: Diffusion Models have emerged as powerful generative models for high-quality image synthesis, with many subsequent image editing techniques based on them. However, the ease of text-based image editing introduces significant risks, such as malicious editing for scams or intellectual property infringement. Previous works have attempted to safeguard images from diffusion-based editing by adding imperceptible perturbations. These methods are costly and specifically target prevalent Latent Diffusion Models (LDMs), while Pixel-domain Diffusion Models (PDMs) remain largely unexplored and robust against such attacks. Our work addresses this gap by proposing a novel attack framework, AtkPDM. AtkPDM is mainly composed of a feature representation attacking loss that exploits vulnerabilities in denoising UNets and a latent optimization strategy to enhance the naturalness of adversarial images. Extensive experiments demonstrate the effectiveness of our approach in attacking dominant PDM-based editing methods (e.g., SDEdit) while maintaining reasonable fidelity and robustness against common defense methods. Additionally, our framework is extensible to LDMs, achieving comparable performance to existing approaches.



## **45. Adversarial Detection by Approximation of Ensemble Boundary**

cs.LG

27 pages, 7 figures, 5 tables

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2211.10227v5) [paper-pdf](http://arxiv.org/pdf/2211.10227v5)

**Authors**: T. Windeatt

**Abstract**: Despite being effective in many application areas, Deep Neural Networks (DNNs) are vulnerable to being attacked. In object recognition, the attack takes the form of a small perturbation added to an image, that causes the DNN to misclassify, but to a human appears no different. Adversarial attacks lead to defences that are themselves subject to attack, and the attack/ defence strategies provide important information about the properties of DNNs. In this paper, a novel method of detecting adversarial attacks is proposed for an ensemble of Deep Neural Networks (DNNs) solving two-class pattern recognition problems. The ensemble is combined using Walsh coefficients which are capable of approximating Boolean functions and thereby controlling the decision boundary complexity. The hypothesis in this paper is that decision boundaries with high curvature allow adversarial perturbations to be found, but change the curvature of the decision boundary, which is then approximated in a different way by Walsh coefficients compared to the clean images. Besides controlling boundary complexity, the coefficients also measure the correlation with class labels, which may aid in understanding the learning and transferability properties of DNNs. While the experiments here use images, the proposed approach of modelling two-class ensemble decision boundaries could in principle be applied to any application area.



## **46. Beyond Optimal Fault Tolerance**

cs.DC

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.06044v1) [paper-pdf](http://arxiv.org/pdf/2501.06044v1)

**Authors**: Andrew Lewis-Pye, Tim Roughgarden

**Abstract**: The optimal fault-tolerance achievable by any protocol has been characterized in a wide range of settings. For example, for state machine replication (SMR) protocols operating in the partially synchronous setting, it is possible to simultaneously guarantee consistency against $\alpha$-bounded adversaries (i.e., adversaries that control less than an $\alpha$ fraction of the participants) and liveness against $\beta$-bounded adversaries if and only if $\alpha + 2\beta \leq 1$.   This paper characterizes to what extent "better-than-optimal" fault-tolerance guarantees are possible for SMR protocols when the standard consistency requirement is relaxed to allow a bounded number $r$ of consistency violations. We prove that bounding rollback is impossible without additional timing assumptions and investigate protocols that tolerate and recover from consistency violations whenever message delays around the time of an attack are bounded by a parameter $\Delta^*$ (which may be arbitrarily larger than the parameter $\Delta$ that bounds post-GST message delays in the partially synchronous model). Here, a protocol's fault-tolerance can be a non-constant function of $r$, and we prove, for each $r$, matching upper and lower bounds on the optimal ``recoverable fault-tolerance'' achievable by any SMR protocol. For example, for protocols that guarantee liveness against 1/3-bounded adversaries in the partially synchronous setting, a 5/9-bounded adversary can always cause one consistency violation but not two, and a 2/3-bounded adversary can always cause two consistency violations but not three. Our positive results are achieved through a generic ``recovery procedure'' that can be grafted on to any accountable SMR protocol and restores consistency following a violation while rolling back only transactions that were finalized in the previous $2\Delta^*$ timesteps.



## **47. Effective faking of verbal deception detection with target-aligned adversarial attacks**

cs.CL

preprint

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.05962v1) [paper-pdf](http://arxiv.org/pdf/2501.05962v1)

**Authors**: Bennett Kleinberg, Riccardo Loconte, Bruno Verschuere

**Abstract**: Background: Deception detection through analysing language is a promising avenue using both human judgments and automated machine learning judgments. For both forms of credibility assessment, automated adversarial attacks that rewrite deceptive statements to appear truthful pose a serious threat. Methods: We used a dataset of 243 truthful and 262 fabricated autobiographical stories in a deception detection task for humans and machine learning models. A large language model was tasked to rewrite deceptive statements so that they appear truthful. In Study 1, humans who made a deception judgment or used the detailedness heuristic and two machine learning models (a fine-tuned language model and a simple n-gram model) judged original or adversarial modifications of deceptive statements. In Study 2, we manipulated the target alignment of the modifications, i.e. tailoring the attack to whether the statements would be assessed by humans or computer models. Results: When adversarial modifications were aligned with their target, human (d=-0.07 and d=-0.04) and machine judgments (51% accuracy) dropped to the chance level. When the attack was not aligned with the target, both human heuristics judgments (d=0.30 and d=0.36) and machine learning predictions (63-78%) were significantly better than chance. Conclusions: Easily accessible language models can effectively help anyone fake deception detection efforts both by humans and machine learning models. Robustness against adversarial modifications for humans and machines depends on that target alignment. We close with suggestions on advancing deception research with adversarial attack designs.



## **48. Towards Backdoor Stealthiness in Model Parameter Space**

cs.CR

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.05928v1) [paper-pdf](http://arxiv.org/pdf/2501.05928v1)

**Authors**: Xiaoyun Xu, Zhuoran Liu, Stefanos Koffas, Stjepan Picek

**Abstract**: Recent research on backdoor stealthiness focuses mainly on indistinguishable triggers in input space and inseparable backdoor representations in feature space, aiming to circumvent backdoor defenses that examine these respective spaces. However, existing backdoor attacks are typically designed to resist a specific type of backdoor defense without considering the diverse range of defense mechanisms. Based on this observation, we pose a natural question: Are current backdoor attacks truly a real-world threat when facing diverse practical defenses?   To answer this question, we examine 12 common backdoor attacks that focus on input-space or feature-space stealthiness and 17 diverse representative defenses. Surprisingly, we reveal a critical blind spot: Backdoor attacks designed to be stealthy in input and feature spaces can be mitigated by examining backdoored models in parameter space. To investigate the underlying causes behind this common vulnerability, we study the characteristics of backdoor attacks in the parameter space. Notably, we find that input- and feature-space attacks introduce prominent backdoor-related neurons in parameter space, which are not thoroughly considered by current backdoor attacks. Taking comprehensive stealthiness into account, we propose a novel supply-chain attack called Grond. Grond limits the parameter changes by a simple yet effective module, Adversarial Backdoor Injection (ABI), which adaptively increases the parameter-space stealthiness during the backdoor injection. Extensive experiments demonstrate that Grond outperforms all 12 backdoor attacks against state-of-the-art (including adaptive) defenses on CIFAR-10, GTSRB, and a subset of ImageNet. In addition, we show that ABI consistently improves the effectiveness of common backdoor attacks.



## **49. Backdoor Attacks against No-Reference Image Quality Assessment Models via a Scalable Trigger**

cs.CV

Accept by AAAI 2025

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2412.07277v2) [paper-pdf](http://arxiv.org/pdf/2412.07277v2)

**Authors**: Yi Yu, Song Xia, Xun Lin, Wenhan Yang, Shijian Lu, Yap-peng Tan, Alex Kot

**Abstract**: No-Reference Image Quality Assessment (NR-IQA), responsible for assessing the quality of a single input image without using any reference, plays a critical role in evaluating and optimizing computer vision systems, e.g., low-light enhancement. Recent research indicates that NR-IQA models are susceptible to adversarial attacks, which can significantly alter predicted scores with visually imperceptible perturbations. Despite revealing vulnerabilities, these attack methods have limitations, including high computational demands, untargeted manipulation, limited practical utility in white-box scenarios, and reduced effectiveness in black-box scenarios. To address these challenges, we shift our focus to another significant threat and present a novel poisoning-based backdoor attack against NR-IQA (BAIQA), allowing the attacker to manipulate the IQA model's output to any desired target value by simply adjusting a scaling coefficient $\alpha$ for the trigger. We propose to inject the trigger in the discrete cosine transform (DCT) domain to improve the local invariance of the trigger for countering trigger diminishment in NR-IQA models due to widely adopted data augmentations. Furthermore, the universal adversarial perturbations (UAP) in the DCT space are designed as the trigger, to increase IQA model susceptibility to manipulation and improve attack effectiveness. In addition to the heuristic method for poison-label BAIQA (P-BAIQA), we explore the design of clean-label BAIQA (C-BAIQA), focusing on $\alpha$ sampling and image data refinement, driven by theoretical insights we reveal. Extensive experiments on diverse datasets and various NR-IQA models demonstrate the effectiveness of our attacks. Code can be found at https://github.com/yuyi-sd/BAIQA.



## **50. Image-based Multimodal Models as Intruders: Transferable Multimodal Attacks on Video-based MLLMs**

cs.CV

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.01042v2) [paper-pdf](http://arxiv.org/pdf/2501.01042v2)

**Authors**: Linhao Huang, Xue Jiang, Zhiqiang Wang, Wentao Mo, Xi Xiao, Bo Han, Yongjie Yin, Feng Zheng

**Abstract**: Video-based multimodal large language models (V-MLLMs) have shown vulnerability to adversarial examples in video-text multimodal tasks. However, the transferability of adversarial videos to unseen models--a common and practical real world scenario--remains unexplored. In this paper, we pioneer an investigation into the transferability of adversarial video samples across V-MLLMs. We find that existing adversarial attack methods face significant limitations when applied in black-box settings for V-MLLMs, which we attribute to the following shortcomings: (1) lacking generalization in perturbing video features, (2) focusing only on sparse key-frames, and (3) failing to integrate multimodal information. To address these limitations and deepen the understanding of V-MLLM vulnerabilities in black-box scenarios, we introduce the Image-to-Video MLLM (I2V-MLLM) attack. In I2V-MLLM, we utilize an image-based multimodal model (IMM) as a surrogate model to craft adversarial video samples. Multimodal interactions and temporal information are integrated to disrupt video representations within the latent space, improving adversarial transferability. In addition, a perturbation propagation technique is introduced to handle different unknown frame sampling strategies. Experimental results demonstrate that our method can generate adversarial examples that exhibit strong transferability across different V-MLLMs on multiple video-text multimodal tasks. Compared to white-box attacks on these models, our black-box attacks (using BLIP-2 as surrogate model) achieve competitive performance, with average attack success rates of 55.48% on MSVD-QA and 58.26% on MSRVTT-QA for VideoQA tasks, respectively. Our code will be released upon acceptance.



