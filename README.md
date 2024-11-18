# Latest Adversarial Attack Papers
**update at 2024-11-18 09:39:36**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. DiffPAD: Denoising Diffusion-based Adversarial Patch Decontamination**

cs.CV

Accepted to 2025 IEEE/CVF Winter Conference on Applications of  Computer Vision (WACV)

**SubmitDate**: 2024-11-14    [abs](http://arxiv.org/abs/2410.24006v2) [paper-pdf](http://arxiv.org/pdf/2410.24006v2)

**Authors**: Jia Fu, Xiao Zhang, Sepideh Pashami, Fatemeh Rahimian, Anders Holst

**Abstract**: In the ever-evolving adversarial machine learning landscape, developing effective defenses against patch attacks has become a critical challenge, necessitating reliable solutions to safeguard real-world AI systems. Although diffusion models have shown remarkable capacity in image synthesis and have been recently utilized to counter $\ell_p$-norm bounded attacks, their potential in mitigating localized patch attacks remains largely underexplored. In this work, we propose DiffPAD, a novel framework that harnesses the power of diffusion models for adversarial patch decontamination. DiffPAD first performs super-resolution restoration on downsampled input images, then adopts binarization, dynamic thresholding scheme and sliding window for effective localization of adversarial patches. Such a design is inspired by the theoretically derived correlation between patch size and diffusion restoration error that is generalized across diverse patch attack scenarios. Finally, DiffPAD applies inpainting techniques to the original input images with the estimated patch region being masked. By integrating closed-form solutions for super-resolution restoration and image inpainting into the conditional reverse sampling process of a pre-trained diffusion model, DiffPAD obviates the need for text guidance or fine-tuning. Through comprehensive experiments, we demonstrate that DiffPAD not only achieves state-of-the-art adversarial robustness against patch attacks but also excels in recovering naturalistic images without patch remnants. The source code is available at https://github.com/JasonFu1998/DiffPAD.



## **2. Are nuclear masks all you need for improved out-of-domain generalisation? A closer look at cancer classification in histopathology**

eess.IV

Poster at NeurIPS 2024

**SubmitDate**: 2024-11-14    [abs](http://arxiv.org/abs/2411.09373v1) [paper-pdf](http://arxiv.org/pdf/2411.09373v1)

**Authors**: Dhananjay Tomar, Alexander Binder, Andreas Kleppe

**Abstract**: Domain generalisation in computational histopathology is challenging because the images are substantially affected by differences among hospitals due to factors like fixation and staining of tissue and imaging equipment. We hypothesise that focusing on nuclei can improve the out-of-domain (OOD) generalisation in cancer detection. We propose a simple approach to improve OOD generalisation for cancer detection by focusing on nuclear morphology and organisation, as these are domain-invariant features critical in cancer detection. Our approach integrates original images with nuclear segmentation masks during training, encouraging the model to prioritise nuclei and their spatial arrangement. Going beyond mere data augmentation, we introduce a regularisation technique that aligns the representations of masks and original images. We show, using multiple datasets, that our method improves OOD generalisation and also leads to increased robustness to image corruptions and adversarial attacks. The source code is available at https://github.com/undercutspiky/SFL/



## **3. Enhancing generalization in high energy physics using white-box adversarial attacks**

hep-ph

10 pages, 4 figures, 8 tables, 3 algorithms, to be published in  Physical Review D (PRD), presented at the ML4Jets 2024 conference

**SubmitDate**: 2024-11-14    [abs](http://arxiv.org/abs/2411.09296v1) [paper-pdf](http://arxiv.org/pdf/2411.09296v1)

**Authors**: Franck Rothen, Samuel Klein, Matthew Leigh, Tobias Golling

**Abstract**: Machine learning is becoming increasingly popular in the context of particle physics. Supervised learning, which uses labeled Monte Carlo (MC) simulations, remains one of the most widely used methods for discriminating signals beyond the Standard Model. However, this paper suggests that supervised models may depend excessively on artifacts and approximations from Monte Carlo simulations, potentially limiting their ability to generalize well to real data. This study aims to enhance the generalization properties of supervised models by reducing the sharpness of local minima. It reviews the application of four distinct white-box adversarial attacks in the context of classifying Higgs boson decay signals. The attacks are divided into weight space attacks, and feature space attacks. To study and quantify the sharpness of different local minima this paper presents two analysis methods: gradient ascent and reduced Hessian eigenvalue analysis. The results show that white-box adversarial attacks significantly improve generalization performance, albeit with increased computational complexity.



## **4. BEARD: Benchmarking the Adversarial Robustness for Dataset Distillation**

cs.CV

15 pages, 6 figures

**SubmitDate**: 2024-11-14    [abs](http://arxiv.org/abs/2411.09265v1) [paper-pdf](http://arxiv.org/pdf/2411.09265v1)

**Authors**: Zheng Zhou, Wenquan Feng, Shuchang Lyu, Guangliang Cheng, Xiaowei Huang, Qi Zhao

**Abstract**: Dataset Distillation (DD) is an emerging technique that compresses large-scale datasets into significantly smaller synthesized datasets while preserving high test performance and enabling the efficient training of large models. However, current research primarily focuses on enhancing evaluation accuracy under limited compression ratios, often overlooking critical security concerns such as adversarial robustness. A key challenge in evaluating this robustness lies in the complex interactions between distillation methods, model architectures, and adversarial attack strategies, which complicate standardized assessments. To address this, we introduce BEARD, an open and unified benchmark designed to systematically assess the adversarial robustness of DD methods, including DM, IDM, and BACON. BEARD encompasses a variety of adversarial attacks (e.g., FGSM, PGD, C&W) on distilled datasets like CIFAR-10/100 and TinyImageNet. Utilizing an adversarial game framework, it introduces three key metrics: Robustness Ratio (RR), Attack Efficiency Ratio (AE), and Comprehensive Robustness-Efficiency Index (CREI). Our analysis includes unified benchmarks, various Images Per Class (IPC) settings, and the effects of adversarial training. Results are available on the BEARD Leaderboard, along with a library providing model and dataset pools to support reproducible research. Access the code at BEARD.



## **5. Injection Attacks Against End-to-End Encrypted Applications**

cs.CR

Published in IEEE Security and Privacy 2024

**SubmitDate**: 2024-11-14    [abs](http://arxiv.org/abs/2411.09228v1) [paper-pdf](http://arxiv.org/pdf/2411.09228v1)

**Authors**: Andrés Fábrega, Carolina Ortega Pérez, Armin Namavari, Ben Nassi, Rachit Agarwal, Thomas Ristenpart

**Abstract**: We explore an emerging threat model for end-to-end (E2E) encrypted applications: an adversary sends chosen messages to a target client, thereby "injecting" adversarial content into the application state. Such state is subsequently encrypted and synchronized to an adversarially-visible storage. By observing the lengths of the resulting cloud-stored ciphertexts, the attacker backs out confidential information. We investigate this injection threat model in the context of state-of-the-art encrypted messaging applications that support E2E encrypted backups. We show proof-of-concept attacks that can recover information about E2E encrypted messages or attachments sent via WhatsApp, assuming the ability to compromise the target user's Google or Apple account (which gives access to encrypted backups). We also show weaknesses in Signal's encrypted backup design that would allow injection attacks to infer metadata including a target user's number of contacts and conversations, should the adversary somehow obtain access to the user's encrypted Signal backup. While we do not believe our results should be of immediate concern for users of these messaging applications, our results do suggest that more work is needed to build tools that enjoy strong E2E security guarantees.



## **6. Transferable Adversarial Attacks against ASR**

eess.AS

IEEE SPL

**SubmitDate**: 2024-11-14    [abs](http://arxiv.org/abs/2411.09220v1) [paper-pdf](http://arxiv.org/pdf/2411.09220v1)

**Authors**: Xiaoxue Gao, Zexin Li, Yiming Chen, Cong Liu, Haizhou Li

**Abstract**: Given the extensive research and real-world applications of automatic speech recognition (ASR), ensuring the robustness of ASR models against minor input perturbations becomes a crucial consideration for maintaining their effectiveness in real-time scenarios. Previous explorations into ASR model robustness have predominantly revolved around evaluating accuracy on white-box settings with full access to ASR models. Nevertheless, full ASR model details are often not available in real-world applications. Therefore, evaluating the robustness of black-box ASR models is essential for a comprehensive understanding of ASR model resilience. In this regard, we thoroughly study the vulnerability of practical black-box attacks in cutting-edge ASR models and propose to employ two advanced time-domain-based transferable attacks alongside our differentiable feature extractor. We also propose a speech-aware gradient optimization approach (SAGO) for ASR, which forces mistranscription with minimal impact on human imperceptibility through voice activity detection rule and a speech-aware gradient-oriented optimizer. Our comprehensive experimental results reveal performance enhancements compared to baseline approaches across five models on two databases.



## **7. Infighting in the Dark: Multi-Labels Backdoor Attack in Federated Learning**

cs.CR

11 pages, 7 figures

**SubmitDate**: 2024-11-14    [abs](http://arxiv.org/abs/2409.19601v2) [paper-pdf](http://arxiv.org/pdf/2409.19601v2)

**Authors**: Ye Li, Yanchao Zhao, Chengcheng Zhu, Jiale Zhang

**Abstract**: Federated Learning (FL), a privacy-preserving decentralized machine learning framework, has been shown to be vulnerable to backdoor attacks. Current research primarily focuses on the Single-Label Backdoor Attack (SBA), wherein adversaries share a consistent target. However, a critical fact is overlooked: adversaries may be non-cooperative, have distinct targets, and operate independently, which exhibits a more practical scenario called Multi-Label Backdoor Attack (MBA). Unfortunately, prior works are ineffective in MBA scenario since non-cooperative attackers exclude each other. In this work, we conduct an in-depth investigation to uncover the inherent constraints of the exclusion: similar backdoor mappings are constructed for different targets, resulting in conflicts among backdoor functions. To address this limitation, we propose Mirage, the first non-cooperative MBA strategy in FL that allows attackers to inject effective and persistent backdoors into the global model without collusion by constructing in-distribution (ID) backdoor mapping. Specifically, we introduce an adversarial adaptation method to bridge the backdoor features and the target distribution in an ID manner. Additionally, we further leverage a constrained optimization method to ensure the ID mapping survives in the global training dynamics. Extensive evaluations demonstrate that Mirage outperforms various state-of-the-art attacks and bypasses existing defenses, achieving an average ASR greater than 97\% and maintaining over 90\% after 900 rounds. This work aims to alert researchers to this potential threat and inspire the design of effective defense mechanisms. Code has been made open-source.



## **8. LeapFrog: The Rowhammer Instruction Skip Attack**

cs.CR

Accepted at Hardware.io 2024

**SubmitDate**: 2024-11-14    [abs](http://arxiv.org/abs/2404.07878v2) [paper-pdf](http://arxiv.org/pdf/2404.07878v2)

**Authors**: Andrew Adiletta, M. Caner Tol, Kemal Derya, Berk Sunar, Saad Islam

**Abstract**: Since its inception, Rowhammer exploits have rapidly evolved into increasingly sophisticated threats compromising data integrity and the control flow integrity of victim processes. Nevertheless, it remains a challenge for an attacker to identify vulnerable targets (i.e., Rowhammer gadgets), understand the outcome of the attempted fault, and formulate an attack that yields useful results.   In this paper, we present a new type of Rowhammer gadget, called a LeapFrog gadget, which, when present in the victim code, allows an adversary to subvert code execution to bypass a critical piece of code (e.g., authentication check logic, encryption rounds, padding in security protocols). The LeapFrog gadget manifests when the victim code stores the Program Counter (PC) value in the user or kernel stack (e.g., a return address during a function call) which, when tampered with, repositions the return address to a location that bypasses a security-critical code pattern.   This research also presents a systematic process to identify LeapFrog gadgets. This methodology enables the automated detection of susceptible targets and the determination of optimal attack parameters. We first show the attack on a decision tree algorithm to show the potential implications. Secondly, we employ the attack on OpenSSL to bypass the encryption and reveal the plaintext. We then use our tools to scan the Open Quantum Safe library and report on the number of LeapFrog gadgets in the code. Lastly, we demonstrate this new attack vector through a practical demonstration in a client/server TLS handshake scenario, successfully inducing an instruction skip in a client application. Our findings extend the impact of Rowhammer attacks on control flow and contribute to developing more robust defenses against these increasingly sophisticated threats.



## **9. DROJ: A Prompt-Driven Attack against Large Language Models**

cs.CL

**SubmitDate**: 2024-11-14    [abs](http://arxiv.org/abs/2411.09125v1) [paper-pdf](http://arxiv.org/pdf/2411.09125v1)

**Authors**: Leyang Hu, Boran Wang

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional capabilities across various natural language processing tasks. Due to their training on internet-sourced datasets, LLMs can sometimes generate objectionable content, necessitating extensive alignment with human feedback to avoid such outputs. Despite massive alignment efforts, LLMs remain susceptible to adversarial jailbreak attacks, which usually are manipulated prompts designed to circumvent safety mechanisms and elicit harmful responses. Here, we introduce a novel approach, Directed Rrepresentation Optimization Jailbreak (DROJ), which optimizes jailbreak prompts at the embedding level to shift the hidden representations of harmful queries towards directions that are more likely to elicit affirmative responses from the model. Our evaluations on LLaMA-2-7b-chat model show that DROJ achieves a 100\% keyword-based Attack Success Rate (ASR), effectively preventing direct refusals. However, the model occasionally produces repetitive and non-informative responses. To mitigate this, we introduce a helpfulness system prompt that enhances the utility of the model's responses. Our code is available at https://github.com/Leon-Leyang/LLM-Safeguard.



## **10. Deciphering the Definition of Adversarial Robustness for post-hoc OOD Detectors**

cs.CR

**SubmitDate**: 2024-11-14    [abs](http://arxiv.org/abs/2406.15104v4) [paper-pdf](http://arxiv.org/pdf/2406.15104v4)

**Authors**: Peter Lorenz, Mario Fernandez, Jens Müller, Ullrich Köthe

**Abstract**: Detecting out-of-distribution (OOD) inputs is critical for safely deploying deep learning models in real-world scenarios. In recent years, many OOD detectors have been developed, and even the benchmarking has been standardized, i.e. OpenOOD. The number of post-hoc detectors is growing fast. They are showing an option to protect a pre-trained classifier against natural distribution shifts and claim to be ready for real-world scenarios. However, its effectiveness in dealing with adversarial examples (AdEx) has been neglected in most studies. In cases where an OOD detector includes AdEx in its experiments, the lack of uniform parameters for AdEx makes it difficult to accurately evaluate the performance of the OOD detector. This paper investigates the adversarial robustness of 16 post-hoc detectors against various evasion attacks. It also discusses a roadmap for adversarial defense in OOD detectors that would help adversarial robustness. We believe that level 1 (AdEx on a unified dataset) should be added to any OOD detector to see the limitations. The last level in the roadmap (defense against adaptive attacks) we added for integrity from an adversarial machine learning (AML) point of view, which we do not believe is the ultimate goal for OOD detectors.



## **11. Defending Large Language Models Against Attacks With Residual Stream Activation Analysis**

cs.CR

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2406.03230v4) [paper-pdf](http://arxiv.org/pdf/2406.03230v4)

**Authors**: Amelia Kawasaki, Andrew Davis, Houssam Abbas

**Abstract**: The widespread adoption of Large Language Models (LLMs), exemplified by OpenAI's ChatGPT, brings to the forefront the imperative to defend against adversarial threats on these models. These attacks, which manipulate an LLM's output by introducing malicious inputs, undermine the model's integrity and the trust users place in its outputs. In response to this challenge, our paper presents an innovative defensive strategy, given white box access to an LLM, that harnesses residual activation analysis between transformer layers of the LLM. We apply a novel methodology for analyzing distinctive activation patterns in the residual streams for attack prompt classification. We curate multiple datasets to demonstrate how this method of classification has high accuracy across multiple types of attack scenarios, including our newly-created attack dataset. Furthermore, we enhance the model's resilience by integrating safety fine-tuning techniques for LLMs in order to measure its effect on our capability to detect attacks. The results underscore the effectiveness of our approach in enhancing the detection and mitigation of adversarial inputs, advancing the security framework within which LLMs operate.



## **12. LLMStinger: Jailbreaking LLMs using RL fine-tuned LLMs**

cs.LG

Accepted at AAAI 2025

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2411.08862v1) [paper-pdf](http://arxiv.org/pdf/2411.08862v1)

**Authors**: Piyush Jha, Arnav Arora, Vijay Ganesh

**Abstract**: We introduce LLMStinger, a novel approach that leverages Large Language Models (LLMs) to automatically generate adversarial suffixes for jailbreak attacks. Unlike traditional methods, which require complex prompt engineering or white-box access, LLMStinger uses a reinforcement learning (RL) loop to fine-tune an attacker LLM, generating new suffixes based on existing attacks for harmful questions from the HarmBench benchmark. Our method significantly outperforms existing red-teaming approaches (we compared against 15 of the latest methods), achieving a +57.2% improvement in Attack Success Rate (ASR) on LLaMA2-7B-chat and a +50.3% ASR increase on Claude 2, both models known for their extensive safety measures. Additionally, we achieved a 94.97% ASR on GPT-3.5 and 99.4% on Gemma-2B-it, demonstrating the robustness and adaptability of LLMStinger across open and closed-source models.



## **13. On the Robustness of Neural Collapse and the Neural Collapse of Robustness**

cs.LG

Transactions on Machine Learning Research, 2024

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2311.07444v2) [paper-pdf](http://arxiv.org/pdf/2311.07444v2)

**Authors**: Jingtong Su, Ya Shi Zhang, Nikolaos Tsilivis, Julia Kempe

**Abstract**: Neural Collapse refers to the curious phenomenon in the end of training of a neural network, where feature vectors and classification weights converge to a very simple geometrical arrangement (a simplex). While it has been observed empirically in various cases and has been theoretically motivated, its connection with crucial properties of neural networks, like their generalization and robustness, remains unclear. In this work, we study the stability properties of these simplices. We find that the simplex structure disappears under small adversarial attacks, and that perturbed examples "leap" between simplex vertices. We further analyze the geometry of networks that are optimized to be robust against adversarial perturbations of the input, and find that Neural Collapse is a pervasive phenomenon in these cases as well, with clean and perturbed representations forming aligned simplices, and giving rise to a robust simple nearest-neighbor classifier. By studying the propagation of the amount of collapse inside the network, we identify novel properties of both robust and non-robust machine learning models, and show that earlier, unlike later layers maintain reliable simplices on perturbed data. Our code is available at https://github.com/JingtongSu/robust_neural_collapse .



## **14. Robust Optimal Power Flow Against Adversarial Attacks: A Tri-Level Optimization Approach**

eess.SY

This work has been submitted for possible publication

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2411.08618v1) [paper-pdf](http://arxiv.org/pdf/2411.08618v1)

**Authors**: Saman Mazaheri Khamaneh, Tong Wu

**Abstract**: In power systems, unpredictable events like extreme weather, equipment failures, and cyberattacks present significant challenges to ensuring safety and reliability. Ensuring resilience in the face of these uncertainties is crucial for reliable and efficient operations. This paper presents a tri-level optimization approach for robust power system operations that effectively address worst-case attacks. The first stage focuses on optimizing economic dispatch under normal operating conditions, aiming to minimize generation costs while maintaining the supply-demand balance. The second stage introduces an adversarial attack model, identifying worst-case scenarios that maximize the system's vulnerability by targeting distributed generation (DG). In the third stage, mitigation strategies are developed using fast-response energy storage systems (ESS) to minimize disruptions caused by these attacks. By integrating economic dispatch, vulnerability assessment, and mitigation into a unified framework, this approach provides a robust solution for enhancing power system resilience and safety against evolving adversarial threats. The approach is validated using the IEEE-33 node distribution system to demonstrate its effectiveness in achieving both cost efficiency and system resilience.



## **15. Target-driven Attack for Large Language Models**

cs.CL

12 pages, 7 figures. This work is an extension of the  arXiv:2404.07234 work. We propose new methods. 27th European Conference on  Artificial Intelligence 2024

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2411.07268v2) [paper-pdf](http://arxiv.org/pdf/2411.07268v2)

**Authors**: Chong Zhang, Mingyu Jin, Dong Shu, Taowen Wang, Dongfang Liu, Xiaobo Jin

**Abstract**: Current large language models (LLM) provide a strong foundation for large-scale user-oriented natural language tasks. Many users can easily inject adversarial text or instructions through the user interface, thus causing LLM model security challenges like the language model not giving the correct answer. Although there is currently a large amount of research on black-box attacks, most of these black-box attacks use random and heuristic strategies. It is unclear how these strategies relate to the success rate of attacks and thus effectively improve model robustness. To solve this problem, we propose our target-driven black-box attack method to maximize the KL divergence between the conditional probabilities of the clean text and the attack text to redefine the attack's goal. We transform the distance maximization problem into two convex optimization problems based on the attack goal to solve the attack text and estimate the covariance. Furthermore, the projected gradient descent algorithm solves the vector corresponding to the attack text. Our target-driven black-box attack approach includes two attack strategies: token manipulation and misinformation attack. Experimental results on multiple Large Language Models and datasets demonstrate the effectiveness of our attack method.



## **16. Confidence-aware Denoised Fine-tuning of Off-the-shelf Models for Certified Robustness**

cs.CV

26 pages; TMLR 2024; Code is available at  https://github.com/suhyeok24/FT-CADIS

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2411.08933v2) [paper-pdf](http://arxiv.org/pdf/2411.08933v2)

**Authors**: Suhyeok Jang, Seojin Kim, Jinwoo Shin, Jongheon Jeong

**Abstract**: The remarkable advances in deep learning have led to the emergence of many off-the-shelf classifiers, e.g., large pre-trained models. However, since they are typically trained on clean data, they remain vulnerable to adversarial attacks. Despite this vulnerability, their superior performance and transferability make off-the-shelf classifiers still valuable in practice, demanding further work to provide adversarial robustness for them in a post-hoc manner. A recently proposed method, denoised smoothing, leverages a denoiser model in front of the classifier to obtain provable robustness without additional training. However, the denoiser often creates hallucination, i.e., images that have lost the semantics of their originally assigned class, leading to a drop in robustness. Furthermore, its noise-and-denoise procedure introduces a significant distribution shift from the original distribution, causing the denoised smoothing framework to achieve sub-optimal robustness. In this paper, we introduce Fine-Tuning with Confidence-Aware Denoised Image Selection (FT-CADIS), a novel fine-tuning scheme to enhance the certified robustness of off-the-shelf classifiers. FT-CADIS is inspired by the observation that the confidence of off-the-shelf classifiers can effectively identify hallucinated images during denoised smoothing. Based on this, we develop a confidence-aware training objective to handle such hallucinated images and improve the stability of fine-tuning from denoised images. In this way, the classifier can be fine-tuned using only images that are beneficial for adversarial robustness. We also find that such a fine-tuning can be done by updating a small fraction of parameters of the classifier. Extensive experiments demonstrate that FT-CADIS has established the state-of-the-art certified robustness among denoised smoothing methods across all $\ell_2$-adversary radius in various benchmarks.



## **17. A Fully Local Last-Generated Rule in a Blockchain**

cs.CR

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2411.08439v1) [paper-pdf](http://arxiv.org/pdf/2411.08439v1)

**Authors**: Akira Sakurai, Kazuyuki Shudo

**Abstract**: An effective method for suppressing intentional forks in a blockchain is the last-generated rule, which selects the most recent chain as the main chain in the event of a chain tie. This rule helps invalidate blocks that are withheld by adversaries for a certain period. However, existing last-generated rules face an issue in that their applications to the system are not fully localized. In conservative cryptocurrency systems such as Bitcoin, it is desirable for methods to be applied in a fully local manner. In this paper, we propose a locally applicable last-generated rule. Our method is straightforward and is based on a relative time reference. By conservatively setting the upper bound for the clock skews $\Delta_{O_i}$ to 200 s, our proposed method reduces the proportion $\gamma$ of honest miners following the attacker during chain ties by more than 40% compared to existing local methods.



## **18. ADI: Adversarial Dominating Inputs in Vertical Federated Learning Systems**

cs.CR

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2201.02775v4) [paper-pdf](http://arxiv.org/pdf/2201.02775v4)

**Authors**: Qi Pang, Yuanyuan Yuan, Shuai Wang, Wenting Zheng

**Abstract**: Vertical federated learning (VFL) system has recently become prominent as a concept to process data distributed across many individual sources without the need to centralize it. Multiple participants collaboratively train models based on their local data in a privacy-aware manner. To date, VFL has become a de facto solution to securely learn a model among organizations, allowing knowledge to be shared without compromising privacy of any individuals. Despite the prosperous development of VFL systems, we find that certain inputs of a participant, named adversarial dominating inputs (ADIs), can dominate the joint inference towards the direction of the adversary's will and force other (victim) participants to make negligible contributions, losing rewards that are usually offered regarding the importance of their contributions in federated learning scenarios. We conduct a systematic study on ADIs by first proving their existence in typical VFL systems. We then propose gradient-based methods to synthesize ADIs of various formats and exploit common VFL systems. We further launch greybox fuzz testing, guided by the saliency score of ``victim'' participants, to perturb adversary-controlled inputs and systematically explore the VFL attack surface in a privacy-preserving manner. We conduct an in-depth study on the influence of critical parameters and settings in synthesizing ADIs. Our study reveals new VFL attack opportunities, promoting the identification of unknown threats before breaches and building more secure VFL systems.



## **19. "No Matter What You Do": Purifying GNN Models via Backdoor Unlearning**

cs.CR

18 pages, 12 figures, 9 tables

**SubmitDate**: 2024-11-13    [abs](http://arxiv.org/abs/2410.01272v2) [paper-pdf](http://arxiv.org/pdf/2410.01272v2)

**Authors**: Jiale Zhang, Chengcheng Zhu, Bosen Rao, Hao Sui, Xiaobing Sun, Bing Chen, Chunyi Zhou, Shouling Ji

**Abstract**: Recent studies have exposed that GNNs are vulnerable to several adversarial attacks, among which backdoor attack is one of the toughest. Similar to Deep Neural Networks (DNNs), backdoor attacks in GNNs lie in the fact that the attacker modifies a portion of graph data by embedding triggers and enforces the model to learn the trigger feature during the model training process. Despite the massive prior backdoor defense works on DNNs, defending against backdoor attacks in GNNs is largely unexplored, severely hindering the widespread application of GNNs in real-world tasks. To bridge this gap, we present GCleaner, the first backdoor mitigation method on GNNs. GCleaner can mitigate the presence of the backdoor logic within backdoored GNNs by reversing the backdoor learning procedure, aiming to restore the model performance to a level similar to that is directly trained on the original clean dataset. To achieve this objective, we ask: How to recover universal and hard backdoor triggers in GNNs? How to unlearn the backdoor trigger feature while maintaining the model performance? We conduct the graph trigger recovery via the explanation method to identify optimal trigger locations, facilitating the search of universal and hard backdoor triggers in the feature space of the backdoored model through maximal similarity. Subsequently, we introduce the backdoor unlearning mechanism, which combines knowledge distillation and gradient-based explainable knowledge for fine-grained backdoor erasure. Extensive experimental evaluations on four benchmark datasets demonstrate that GCleaner can reduce the backdoor attack success rate to 10% with only 1% of clean data, and has almost negligible degradation in model performance, which far outperforms the state-of-the-art (SOTA) defense methods.



## **20. Deceiving Question-Answering Models: A Hybrid Word-Level Adversarial Approach**

cs.CL

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.08248v1) [paper-pdf](http://arxiv.org/pdf/2411.08248v1)

**Authors**: Jiyao Li, Mingze Ni, Yongshun Gong, Wei Liu

**Abstract**: Deep learning underpins most of the currently advanced natural language processing (NLP) tasks such as textual classification, neural machine translation (NMT), abstractive summarization and question-answering (QA). However, the robustness of the models, particularly QA models, against adversarial attacks is a critical concern that remains insufficiently explored. This paper introduces QA-Attack (Question Answering Attack), a novel word-level adversarial strategy that fools QA models. Our attention-based attack exploits the customized attention mechanism and deletion ranking strategy to identify and target specific words within contextual passages. It creates deceptive inputs by carefully choosing and substituting synonyms, preserving grammatical integrity while misleading the model to produce incorrect responses. Our approach demonstrates versatility across various question types, particularly when dealing with extensive long textual inputs. Extensive experiments on multiple benchmark datasets demonstrate that QA-Attack successfully deceives baseline QA models and surpasses existing adversarial techniques regarding success rate, semantics changes, BLEU score, fluency and grammar error rate.



## **21. Adaptive Meta-Learning for Robust Deepfake Detection: A Multi-Agent Framework to Data Drift and Model Generalization**

cs.AI

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.08148v1) [paper-pdf](http://arxiv.org/pdf/2411.08148v1)

**Authors**: Dinesh Srivasthav P, Badri Narayan Subudhi

**Abstract**: Pioneering advancements in artificial intelligence, especially in genAI, have enabled significant possibilities for content creation, but also led to widespread misinformation and false content. The growing sophistication and realism of deepfakes is raising concerns about privacy invasion, identity theft, and has societal, business impacts, including reputational damage and financial loss. Many deepfake detectors have been developed to tackle this problem. Nevertheless, as for every AI model, the deepfake detectors face the wrath of lack of considerable generalization to unseen scenarios and cross-domain deepfakes. Besides, adversarial robustness is another critical challenge, as detectors drastically underperform to the slightest imperceptible change. Most state-of-the-art detectors are trained on static datasets and lack the ability to adapt to emerging deepfake attack trends. These three crucial challenges though hold paramount importance for reliability in practise, particularly in the deepfake domain, are also the problems with any other AI application. This paper proposes an adversarial meta-learning algorithm using task-specific adaptive sample synthesis and consistency regularization, in a refinement phase. By focussing on the classifier's strengths and weaknesses, it boosts both robustness and generalization of the model. Additionally, the paper introduces a hierarchical multi-agent retrieval-augmented generation workflow with a sample synthesis module to dynamically adapt the model to new data trends by generating custom deepfake samples. The paper further presents a framework integrating the meta-learning algorithm with the hierarchical multi-agent workflow, offering a holistic solution for enhancing generalization, robustness, and adaptability. Experimental results demonstrate the model's consistent performance across various datasets, outperforming the models in comparison.



## **22. Can adversarial attacks by large language models be attributed?**

cs.AI

7 pages, 1 figure

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.08003v1) [paper-pdf](http://arxiv.org/pdf/2411.08003v1)

**Authors**: Manuel Cebrian, Jan Arne Telle

**Abstract**: Attributing outputs from Large Language Models (LLMs) in adversarial settings-such as cyberattacks and disinformation-presents significant challenges that are likely to grow in importance. We investigate this attribution problem using formal language theory, specifically language identification in the limit as introduced by Gold and extended by Angluin. By modeling LLM outputs as formal languages, we analyze whether finite text samples can uniquely pinpoint the originating model. Our results show that due to the non-identifiability of certain language classes, under some mild assumptions about overlapping outputs from fine-tuned models it is theoretically impossible to attribute outputs to specific LLMs with certainty. This holds also when accounting for expressivity limitations of Transformer architectures. Even with direct model access or comprehensive monitoring, significant computational hurdles impede attribution efforts. These findings highlight an urgent need for proactive measures to mitigate risks posed by adversarial LLM use as their influence continues to expand.



## **23. IAE: Irony-based Adversarial Examples for Sentiment Analysis Systems**

cs.CL

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.07850v1) [paper-pdf](http://arxiv.org/pdf/2411.07850v1)

**Authors**: Xiaoyin Yi, Jiacheng Huang

**Abstract**: Adversarial examples, which are inputs deliberately perturbed with imperceptible changes to induce model errors, have raised serious concerns for the reliability and security of deep neural networks (DNNs). While adversarial attacks have been extensively studied in continuous data domains such as images, the discrete nature of text presents unique challenges. In this paper, we propose Irony-based Adversarial Examples (IAE), a method that transforms straightforward sentences into ironic ones to create adversarial text. This approach exploits the rhetorical device of irony, where the intended meaning is opposite to the literal interpretation, requiring a deeper understanding of context to detect. The IAE method is particularly challenging due to the need to accurately locate evaluation words, substitute them with appropriate collocations, and expand the text with suitable ironic elements while maintaining semantic coherence. Our research makes the following key contributions: (1) We introduce IAE, a strategy for generating textual adversarial examples using irony. This method does not rely on pre-existing irony corpora, making it a versatile tool for creating adversarial text in various NLP tasks. (2) We demonstrate that the performance of several state-of-the-art deep learning models on sentiment analysis tasks significantly deteriorates when subjected to IAE attacks. This finding underscores the susceptibility of current NLP systems to adversarial manipulation through irony. (3) We compare the impact of IAE on human judgment versus NLP systems, revealing that humans are less susceptible to the effects of irony in text.



## **24. Chain Association-based Attacking and Shielding Natural Language Processing Systems**

cs.CL

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.07843v1) [paper-pdf](http://arxiv.org/pdf/2411.07843v1)

**Authors**: Jiacheng Huang, Long Chen

**Abstract**: Association as a gift enables people do not have to mention something in completely straightforward words and allows others to understand what they intend to refer to. In this paper, we propose a chain association-based adversarial attack against natural language processing systems, utilizing the comprehension gap between humans and machines. We first generate a chain association graph for Chinese characters based on the association paradigm for building search space of potential adversarial examples. Then, we introduce an discrete particle swarm optimization algorithm to search for the optimal adversarial examples. We conduct comprehensive experiments and show that advanced natural language processing models and applications, including large language models, are vulnerable to our attack, while humans appear good at understanding the perturbed text. We also explore two methods, including adversarial training and associative graph-based recovery, to shield systems from chain association-based attack. Since a few examples that use some derogatory terms, this paper contains materials that may be offensive or upsetting to some people.



## **25. CausalDiff: Causality-Inspired Disentanglement via Diffusion Model for Adversarial Defense**

cs.CV

accepted by NeurIPS 2024

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2410.23091v3) [paper-pdf](http://arxiv.org/pdf/2410.23091v3)

**Authors**: Mingkun Zhang, Keping Bi, Wei Chen, Quanrun Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: Despite ongoing efforts to defend neural classifiers from adversarial attacks, they remain vulnerable, especially to unseen attacks. In contrast, humans are difficult to be cheated by subtle manipulations, since we make judgments only based on essential factors. Inspired by this observation, we attempt to model label generation with essential label-causative factors and incorporate label-non-causative factors to assist data generation. For an adversarial example, we aim to discriminate the perturbations as non-causative factors and make predictions only based on the label-causative factors. Concretely, we propose a casual diffusion model (CausalDiff) that adapts diffusion models for conditional data generation and disentangles the two types of casual factors by learning towards a novel casual information bottleneck objective. Empirically, CausalDiff has significantly outperformed state-of-the-art defense methods on various unseen attacks, achieving an average robustness of 86.39% (+4.01%) on CIFAR-10, 56.25% (+3.13%) on CIFAR-100, and 82.62% (+4.93%) on GTSRB (German Traffic Sign Recognition Benchmark).



## **26. Revisiting the Adversarial Robustness of Vision Language Models: a Multimodal Perspective**

cs.CV

17 pages, 13 figures

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2404.19287v3) [paper-pdf](http://arxiv.org/pdf/2404.19287v3)

**Authors**: Wanqi Zhou, Shuanghao Bai, Danilo P. Mandic, Qibin Zhao, Badong Chen

**Abstract**: Pretrained vision-language models (VLMs) like CLIP exhibit exceptional generalization across diverse downstream tasks. While recent studies reveal their vulnerability to adversarial attacks, research to date has primarily focused on enhancing the robustness of image encoders against image-based attacks, with defenses against text-based and multimodal attacks remaining largely unexplored. To this end, this work presents the first comprehensive study on improving the adversarial robustness of VLMs against attacks targeting image, text, and multimodal inputs. This is achieved by proposing multimodal contrastive adversarial training (MMCoA). Such an approach strengthens the robustness of both image and text encoders by aligning the clean text embeddings with adversarial image embeddings, and adversarial text embeddings with clean image embeddings. The robustness of the proposed MMCoA is examined against existing defense methods over image, text, and multimodal attacks on the CLIP model. Extensive experiments on 15 datasets across two tasks reveal the characteristics of different adversarial defense methods under distinct distribution shifts and dataset complexities across the three attack types. This paves the way for a unified framework of adversarial robustness against different modality attacks, opening up new possibilities for securing VLMs against multimodal attacks. The code is available at https://github.com/ElleZWQ/MMCoA.git.



## **27. Data-Driven Graph Switching for Cyber-Resilient Control in Microgrids**

eess.SY

Accepted in IEEE Design Methodologies Conference (DMC) 2024

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.07686v1) [paper-pdf](http://arxiv.org/pdf/2411.07686v1)

**Authors**: Suman Rath, Subham Sahoo

**Abstract**: Distributed microgrids are conventionally dependent on communication networks to achieve secondary control objectives. This dependence makes them vulnerable to stealth data integrity attacks (DIAs) where adversaries may perform manipulations via infected transmitters and repeaters to jeopardize stability. This paper presents a physics-guided, supervised Artificial Neural Network (ANN)-based framework that identifies communication-level cyberattacks in microgrids by analyzing whether incoming measurements will cause abnormal behavior of the secondary control layer. If abnormalities are detected, an iteration through possible spanning tree graph topologies that can be used to fulfill secondary control objectives is done. Then, a communication network topology that would not create secondary control abnormalities is identified and enforced for maximum stability. By altering the communication graph topology, the framework eliminates the dependence of the secondary control layer on inputs from compromised cyber devices helping it achieve resilience without instability. Several case studies are provided showcasing the robustness of the framework against False Data Injections and repeater-level Man-in-the-Middle attacks. To understand practical feasibility, robustness is also verified against larger microgrid sizes and in the presence of varying noise levels. Our findings indicate that performance can be affected when attempting scalability in the presence of noise. However, the framework operates robustly in low-noise settings.



## **28. Aligning Visual Contrastive learning models via Preference Optimization**

cs.CV

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.08923v1) [paper-pdf](http://arxiv.org/pdf/2411.08923v1)

**Authors**: Amirabbas Afzali, Borna Khodabandeh, Ali Rasekh, Mahyar JafariNodeh, Sepehr kazemi, Simon Gottschalk

**Abstract**: Contrastive learning models have demonstrated impressive abilities to capture semantic similarities by aligning representations in the embedding space. However, their performance can be limited by the quality of the training data and its inherent biases. While Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO) have been applied to generative models to align them with human preferences, their use in contrastive learning has yet to be explored. This paper introduces a novel method for training contrastive learning models using Preference Optimization (PO) to break down complex concepts. Our method systematically aligns model behavior with desired preferences, enhancing performance on the targeted task. In particular, we focus on enhancing model robustness against typographic attacks, commonly seen in contrastive models like CLIP. We further apply our method to disentangle gender understanding and mitigate gender biases, offering a more nuanced control over these sensitive attributes. Our experiments demonstrate that models trained using PO outperform standard contrastive learning techniques while retaining their ability to handle adversarial challenges and maintain accuracy on other downstream tasks. This makes our method well-suited for tasks requiring fairness, robustness, and alignment with specific preferences. We evaluate our method on several vision-language tasks, tackling challenges such as typographic attacks. Additionally, we explore the model's ability to disentangle gender concepts and mitigate gender bias, showcasing the versatility of our approach.



## **29. A Survey on Adversarial Machine Learning for Code Data: Realistic Threats, Countermeasures, and Interpretations**

cs.CR

Under a reviewing process since Sep. 3, 2024

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.07597v1) [paper-pdf](http://arxiv.org/pdf/2411.07597v1)

**Authors**: Yulong Yang, Haoran Fan, Chenhao Lin, Qian Li, Zhengyu Zhao, Chao Shen, Xiaohong Guan

**Abstract**: Code Language Models (CLMs) have achieved tremendous progress in source code understanding and generation, leading to a significant increase in research interests focused on applying CLMs to real-world software engineering tasks in recent years. However, in realistic scenarios, CLMs are exposed to potential malicious adversaries, bringing risks to the confidentiality, integrity, and availability of CLM systems. Despite these risks, a comprehensive analysis of the security vulnerabilities of CLMs in the extremely adversarial environment has been lacking. To close this research gap, we categorize existing attack techniques into three types based on the CIA triad: poisoning attacks (integrity \& availability infringement), evasion attacks (integrity infringement), and privacy attacks (confidentiality infringement). We have collected so far the most comprehensive (79) papers related to adversarial machine learning for CLM from the research fields of artificial intelligence, computer security, and software engineering. Our analysis covers each type of risk, examining threat model categorization, attack techniques, and countermeasures, while also introducing novel perspectives on eXplainable AI (XAI) and exploring the interconnections between different risks. Finally, we identify current challenges and future research opportunities. This study aims to provide a comprehensive roadmap for both researchers and practitioners and pave the way towards more reliable CLMs for practical applications.



## **30. Graph Agent Network: Empowering Nodes with Inference Capabilities for Adversarial Resilience**

cs.LG

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2306.06909v4) [paper-pdf](http://arxiv.org/pdf/2306.06909v4)

**Authors**: Ao Liu, Wenshan Li, Tao Li, Beibei Li, Guangquan Xu, Pan Zhou, Wengang Ma, Hanyuan Huang

**Abstract**: End-to-end training with global optimization have popularized graph neural networks (GNNs) for node classification, yet inadvertently introduced vulnerabilities to adversarial edge-perturbing attacks. Adversaries can exploit the inherent opened interfaces of GNNs' input and output, perturbing critical edges and thus manipulating the classification results. Current defenses, due to their persistent utilization of global-optimization-based end-to-end training schemes, inherently encapsulate the vulnerabilities of GNNs. This is specifically evidenced in their inability to defend against targeted secondary attacks. In this paper, we propose the Graph Agent Network (GAgN) to address the aforementioned vulnerabilities of GNNs. GAgN is a graph-structured agent network in which each node is designed as an 1-hop-view agent. Through the decentralized interactions between agents, they can learn to infer global perceptions to perform tasks including inferring embeddings, degrees and neighbor relationships for given nodes. This empowers nodes to filtering adversarial edges while carrying out classification tasks. Furthermore, agents' limited view prevents malicious messages from propagating globally in GAgN, thereby resisting global-optimization-based secondary attacks. We prove that single-hidden-layer multilayer perceptrons (MLPs) are theoretically sufficient to achieve these functionalities. Experimental results show that GAgN effectively implements all its intended capabilities and, compared to state-of-the-art defenses, achieves optimal classification accuracy on the perturbed datasets.



## **31. Fast Preemption: Forward-Backward Cascade Learning for Efficient and Transferable Proactive Adversarial Defense**

cs.CR

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2407.15524v4) [paper-pdf](http://arxiv.org/pdf/2407.15524v4)

**Authors**: Hanrui Wang, Ching-Chun Chang, Chun-Shien Lu, Isao Echizen

**Abstract**: Deep learning technology has brought convenience and advanced developments but has become untrustworthy due to its sensitivity to adversarial attacks. Attackers may utilize this sensitivity to manipulate predictions. To defend against such attacks, existing anti-adversarial methods typically counteract adversarial perturbations post-attack, while we have devised a proactive strategy that preempts by safeguarding media upfront, effectively neutralizing potential adversarial effects before the third-party attacks occur. This strategy, dubbed Fast Preemption, provides an efficient transferable preemptive defense by using different models for labeling inputs and learning crucial features. A forward-backward cascade learning algorithm is used to compute protective perturbations, starting with forward propagation optimization to achieve rapid convergence, followed by iterative backward propagation learning to alleviate overfitting. This strategy offers state-of-the-art transferability and protection across various systems. With the running of only three steps, our Fast Preemption framework outperforms benchmark training-time, test-time, and preemptive adversarial defenses. We have also devised the first, to our knowledge, effective white-box adaptive reversion attack and demonstrate that the protection added by our defense strategy is irreversible unless the backbone model, algorithm, and settings are fully compromised. This work provides a new direction to developing proactive defenses against adversarial attacks.



## **32. Rapid Response: Mitigating LLM Jailbreaks with a Few Examples**

cs.CL

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.07494v1) [paper-pdf](http://arxiv.org/pdf/2411.07494v1)

**Authors**: Alwin Peng, Julian Michael, Henry Sleight, Ethan Perez, Mrinank Sharma

**Abstract**: As large language models (LLMs) grow more powerful, ensuring their safety against misuse becomes crucial. While researchers have focused on developing robust defenses, no method has yet achieved complete invulnerability to attacks. We propose an alternative approach: instead of seeking perfect adversarial robustness, we develop rapid response techniques to look to block whole classes of jailbreaks after observing only a handful of attacks. To study this setting, we develop RapidResponseBench, a benchmark that measures a defense's robustness against various jailbreak strategies after adapting to a few observed examples. We evaluate five rapid response methods, all of which use jailbreak proliferation, where we automatically generate additional jailbreaks similar to the examples observed. Our strongest method, which fine-tunes an input classifier to block proliferated jailbreaks, reduces attack success rate by a factor greater than 240 on an in-distribution set of jailbreaks and a factor greater than 15 on an out-of-distribution set, having observed just one example of each jailbreaking strategy. Moreover, further studies suggest that the quality of proliferation model and number of proliferated examples play an key role in the effectiveness of this defense. Overall, our results highlight the potential of responding rapidly to novel jailbreaks to limit LLM misuse.



## **33. DrAttack: Prompt Decomposition and Reconstruction Makes Powerful LLM Jailbreakers**

cs.CR

**SubmitDate**: 2024-11-11    [abs](http://arxiv.org/abs/2402.16914v3) [paper-pdf](http://arxiv.org/pdf/2402.16914v3)

**Authors**: Xirui Li, Ruochen Wang, Minhao Cheng, Tianyi Zhou, Cho-Jui Hsieh

**Abstract**: The safety alignment of Large Language Models (LLMs) is vulnerable to both manual and automated jailbreak attacks, which adversarially trigger LLMs to output harmful content. However, current methods for jailbreaking LLMs, which nest entire harmful prompts, are not effective at concealing malicious intent and can be easily identified and rejected by well-aligned LLMs. This paper discovers that decomposing a malicious prompt into separated sub-prompts can effectively obscure its underlying malicious intent by presenting it in a fragmented, less detectable form, thereby addressing these limitations. We introduce an automatic prompt \textbf{D}ecomposition and \textbf{R}econstruction framework for jailbreak \textbf{Attack} (DrAttack). DrAttack includes three key components: (a) `Decomposition' of the original prompt into sub-prompts, (b) `Reconstruction' of these sub-prompts implicitly by in-context learning with semantically similar but harmless reassembling demo, and (c) a `Synonym Search' of sub-prompts, aiming to find sub-prompts' synonyms that maintain the original intent while jailbreaking LLMs. An extensive empirical study across multiple open-source and closed-source LLMs demonstrates that, with a significantly reduced number of queries, DrAttack obtains a substantial gain of success rate over prior SOTA prompt-only attackers. Notably, the success rate of 78.0\% on GPT-4 with merely 15 queries surpassed previous art by 33.1\%. The project is available at https://github.com/xirui-li/DrAttack.



## **34. The Inherent Adversarial Robustness of Analog In-Memory Computing**

cs.ET

**SubmitDate**: 2024-11-11    [abs](http://arxiv.org/abs/2411.07023v1) [paper-pdf](http://arxiv.org/pdf/2411.07023v1)

**Authors**: Corey Lammie, Julian Büchel, Athanasios Vasilopoulos, Manuel Le Gallo, Abu Sebastian

**Abstract**: A key challenge for Deep Neural Network (DNN) algorithms is their vulnerability to adversarial attacks. Inherently non-deterministic compute substrates, such as those based on Analog In-Memory Computing (AIMC), have been speculated to provide significant adversarial robustness when performing DNN inference. In this paper, we experimentally validate this conjecture for the first time on an AIMC chip based on Phase Change Memory (PCM) devices. We demonstrate higher adversarial robustness against different types of adversarial attacks when implementing an image classification network. Additional robustness is also observed when performing hardware-in-the-loop attacks, for which the attacker is assumed to have full access to the hardware. A careful study of the various noise sources indicate that a combination of stochastic noise sources (both recurrent and non-recurrent) are responsible for the adversarial robustness and that their type and magnitude disproportionately effects this property. Finally, it is demonstrated, via simulations, that when a much larger transformer network is used to implement a Natural Language Processing (NLP) task, additional robustness is still observed.



## **35. Computable Model-Independent Bounds for Adversarial Quantum Machine Learning**

cs.LG

21 pages, 9 figures

**SubmitDate**: 2024-11-11    [abs](http://arxiv.org/abs/2411.06863v1) [paper-pdf](http://arxiv.org/pdf/2411.06863v1)

**Authors**: Bacui Li, Tansu Alpcan, Chandra Thapa, Udaya Parampalli

**Abstract**: By leveraging the principles of quantum mechanics, QML opens doors to novel approaches in machine learning and offers potential speedup. However, machine learning models are well-documented to be vulnerable to malicious manipulations, and this susceptibility extends to the models of QML. This situation necessitates a thorough understanding of QML's resilience against adversarial attacks, particularly in an era where quantum computing capabilities are expanding. In this regard, this paper examines model-independent bounds on adversarial performance for QML. To the best of our knowledge, we introduce the first computation of an approximate lower bound for adversarial error when evaluating model resilience against sophisticated quantum-based adversarial attacks. Experimental results are compared to the computed bound, demonstrating the potential of QML models to achieve high robustness. In the best case, the experimental error is only 10% above the estimated bound, offering evidence of the inherent robustness of quantum models. This work not only advances our theoretical understanding of quantum model resilience but also provides a precise reference bound for the future development of robust QML algorithms.



## **36. Boosting the Targeted Transferability of Adversarial Examples via Salient Region & Weighted Feature Drop**

cs.IR

9 pages

**SubmitDate**: 2024-11-11    [abs](http://arxiv.org/abs/2411.06784v1) [paper-pdf](http://arxiv.org/pdf/2411.06784v1)

**Authors**: Shanjun Xu, Linghui Li, Kaiguo Yuan, Bingyu Li

**Abstract**: Deep neural networks can be vulnerable to adversarially crafted examples, presenting significant risks to practical applications. A prevalent approach for adversarial attacks relies on the transferability of adversarial examples, which are generated from a substitute model and leveraged to attack unknown black-box models. Despite various proposals aimed at improving transferability, the success of these attacks in targeted black-box scenarios is often hindered by the tendency for adversarial examples to overfit to the surrogate models. In this paper, we introduce a novel framework based on Salient region & Weighted Feature Drop (SWFD) designed to enhance the targeted transferability of adversarial examples. Drawing from the observation that examples with higher transferability exhibit smoother distributions in the deep-layer outputs, we propose the weighted feature drop mechanism to modulate activation values according to weights scaled by norm distribution, effectively addressing the overfitting issue when generating adversarial examples. Additionally, by leveraging salient region within the image to construct auxiliary images, our method enables the adversarial example's features to be transferred to the target category in a model-agnostic manner, thereby enhancing the transferability. Comprehensive experiments confirm that our approach outperforms state-of-the-art methods across diverse configurations. On average, the proposed SWFD raises the attack success rate for normally trained models and robust models by 16.31% and 7.06% respectively.



## **37. Beyond Text: Utilizing Vocal Cues to Improve Decision Making in LLMs for Robot Navigation Tasks**

cs.AI

30 pages, 7 figures

**SubmitDate**: 2024-11-11    [abs](http://arxiv.org/abs/2402.03494v3) [paper-pdf](http://arxiv.org/pdf/2402.03494v3)

**Authors**: Xingpeng Sun, Haoming Meng, Souradip Chakraborty, Amrit Singh Bedi, Aniket Bera

**Abstract**: While LLMs excel in processing text in these human conversations, they struggle with the nuances of verbal instructions in scenarios like social navigation, where ambiguity and uncertainty can erode trust in robotic and other AI systems. We can address this shortcoming by moving beyond text and additionally focusing on the paralinguistic features of these audio responses. These features are the aspects of spoken communication that do not involve the literal wording (lexical content) but convey meaning and nuance through how something is said. We present Beyond Text: an approach that improves LLM decision-making by integrating audio transcription along with a subsection of these features, which focus on the affect and more relevant in human-robot conversations.This approach not only achieves a 70.26% winning rate, outperforming existing LLMs by 22.16% to 48.30% (gemini-1.5-pro and gpt-3.5 respectively), but also enhances robustness against token manipulation adversarial attacks, highlighted by a 22.44% less decrease ratio than the text-only language model in winning rate. Beyond Text' marks an advancement in social robot navigation and broader Human-Robot interactions, seamlessly integrating text-based guidance with human-audio-informed language models.



## **38. Adversarial Detection with a Dynamically Stable System**

cs.AI

**SubmitDate**: 2024-11-11    [abs](http://arxiv.org/abs/2411.06666v1) [paper-pdf](http://arxiv.org/pdf/2411.06666v1)

**Authors**: Xiaowei Long, Jie Lin, Xiangyuan Yang

**Abstract**: Adversarial detection is designed to identify and reject maliciously crafted adversarial examples(AEs) which are generated to disrupt the classification of target models.   Presently, various input transformation-based methods have been developed on adversarial example detection, which typically rely on empirical experience and lead to unreliability against new attacks.   To address this issue, we propose and conduct a Dynamically Stable System (DSS), which can effectively detect the adversarial examples from normal examples according to the stability of input examples.   Particularly, in our paper, the generation of adversarial examples is considered as the perturbation process of a Lyapunov dynamic system, and we propose an example stability mechanism, in which a novel control term is added in adversarial example generation to ensure that the normal examples can achieve dynamic stability while the adversarial examples cannot achieve the stability.   Then, based on the proposed example stability mechanism, a Dynamically Stable System (DSS) is proposed, which can utilize the disruption and restoration actions to determine the stability of input examples and detect the adversarial examples through changes in the stability of the input examples.   In comparison with existing methods in three benchmark datasets(MNIST, CIFAR10, and CIFAR100), our evaluation results show that our proposed DSS can achieve ROC-AUC values of 99.83%, 97.81% and 94.47%, surpassing the state-of-the-art(SOTA) values of 97.35%, 91.10% and 93.49% in the other 7 methods.



## **39. Do Unlearning Methods Remove Information from Language Model Weights?**

cs.LG

**SubmitDate**: 2024-11-10    [abs](http://arxiv.org/abs/2410.08827v2) [paper-pdf](http://arxiv.org/pdf/2410.08827v2)

**Authors**: Aghyad Deeb, Fabien Roger

**Abstract**: Large Language Models' knowledge of how to perform cyber-security attacks, create bioweapons, and manipulate humans poses risks of misuse. Previous work has proposed methods to unlearn this knowledge. Historically, it has been unclear whether unlearning techniques are removing information from the model weights or just making it harder to access. To disentangle these two objectives, we propose an adversarial evaluation method to test for the removal of information from model weights: we give an attacker access to some facts that were supposed to be removed, and using those, the attacker tries to recover other facts from the same distribution that cannot be guessed from the accessible facts. We show that using fine-tuning on the accessible facts can recover 88% of the pre-unlearning accuracy when applied to current unlearning methods, revealing the limitations of these methods in removing information from the model weights.



## **40. HidePrint: Hiding the Radio Fingerprint via Random Noise**

cs.CR

**SubmitDate**: 2024-11-10    [abs](http://arxiv.org/abs/2411.06417v1) [paper-pdf](http://arxiv.org/pdf/2411.06417v1)

**Authors**: Gabriele Oligeri, Savio Sciancalepore

**Abstract**: Radio Frequency Fingerprinting (RFF) techniques allow a receiver to authenticate a transmitter by analyzing the physical layer of the radio spectrum. Although the vast majority of scientific contributions focus on improving the performance of RFF considering different parameters and scenarios, in this work, we consider RFF as an attack vector to identify and track a target device.   We propose, implement, and evaluate HidePrint, a solution to prevent tracking through RFF without affecting the quality of the communication link between the transmitter and the receiver. HidePrint hides the transmitter's fingerprint against an illegitimate eavesdropper by injecting controlled noise in the transmitted signal. We evaluate our solution against state-of-the-art image-based RFF techniques considering different adversarial models, different communication links (wired and wireless), and different configurations. Our results show that the injection of a Gaussian noise pattern with a standard deviation of (at least) 0.02 prevents device fingerprinting in all the considered scenarios, thus making the performance of the identification process indistinguishable from the random guess while affecting the Signal-to-Noise Ratio (SNR) of the received signal by only 0.1 dB. Moreover, we introduce selective radio fingerprint disclosure, a new technique that allows the transmitter to disclose the radio fingerprint to only a subset of intended receivers. This technique allows the transmitter to regain anonymity, thus preventing identification and tracking while allowing authorized receivers to authenticate the transmitter without affecting the quality of the transmitted signal.



## **41. Randomized Message-Interception Smoothing: Gray-box Certificates for Graph Neural Networks**

cs.LG

Accepted at NeurIPS 2022

**SubmitDate**: 2024-11-10    [abs](http://arxiv.org/abs/2301.02039v2) [paper-pdf](http://arxiv.org/pdf/2301.02039v2)

**Authors**: Yan Scholten, Jan Schuchardt, Simon Geisler, Aleksandar Bojchevski, Stephan Günnemann

**Abstract**: Randomized smoothing is one of the most promising frameworks for certifying the adversarial robustness of machine learning models, including Graph Neural Networks (GNNs). Yet, existing randomized smoothing certificates for GNNs are overly pessimistic since they treat the model as a black box, ignoring the underlying architecture. To remedy this, we propose novel gray-box certificates that exploit the message-passing principle of GNNs: We randomly intercept messages and carefully analyze the probability that messages from adversarially controlled nodes reach their target nodes. Compared to existing certificates, we certify robustness to much stronger adversaries that control entire nodes in the graph and can arbitrarily manipulate node features. Our certificates provide stronger guarantees for attacks at larger distances, as messages from farther-away nodes are more likely to get intercepted. We demonstrate the effectiveness of our method on various models and datasets. Since our gray-box certificates consider the underlying graph structure, we can significantly improve certifiable robustness by applying graph sparsification.



## **42. Robust Detection of LLM-Generated Text: A Comparative Analysis**

cs.CL

8 pages

**SubmitDate**: 2024-11-09    [abs](http://arxiv.org/abs/2411.06248v1) [paper-pdf](http://arxiv.org/pdf/2411.06248v1)

**Authors**: Yongye Su, Yuqing Wu

**Abstract**: The ability of large language models to generate complex texts allows them to be widely integrated into many aspects of life, and their output can quickly fill all network resources. As the impact of LLMs grows, it becomes increasingly important to develop powerful detectors for the generated text. This detector is essential to prevent the potential misuse of these technologies and to protect areas such as social media from the negative effects of false content generated by LLMS. The main goal of LLM-generated text detection is to determine whether text is generated by an LLM, which is a basic binary classification task. In our work, we mainly use three different classification methods based on open source datasets: traditional machine learning techniques such as logistic regression, k-means clustering, Gaussian Naive Bayes, support vector machines, and methods based on converters such as BERT, and finally algorithms that use LLMs to detect LLM-generated text. We focus on model generalization, potential adversarial attacks, and accuracy of model evaluation. Finally, the possible research direction in the future is proposed, and the current experimental results are summarized.



## **43. BM-PAW: A Profitable Mining Attack in the PoW-based Blockchain System**

cs.CR

21 pages, 4 figures

**SubmitDate**: 2024-11-09    [abs](http://arxiv.org/abs/2411.06187v1) [paper-pdf](http://arxiv.org/pdf/2411.06187v1)

**Authors**: Junjie Hu, Xunzhi Chen, Huan Yan, Na Ruan

**Abstract**: Mining attacks enable an adversary to procure a disproportionately large portion of mining rewards by deviating from honest mining practices within the PoW-based blockchain system. In this paper, we demonstrate that the security vulnerabilities of PoW-based blockchain extend beyond what these mining attacks initially reveal. We introduce a novel mining strategy, named BM-PAW, which yields superior rewards for both the attacker and the targeted pool compared to the state-of-the-art mining attack: PAW. Our analysis reveals that BM-PAW attackers are incentivized to offer appropriate bribe money to other targets, as they comply with the attacker's directives upon receiving payment. We find the BM-PAW attacker can circumvent the "miner's dilemma" through equilibrium analysis in a two-pool BM-PAW game scenario, wherein the outcome is determined by the attacker's mining power. We finally propose practical countermeasures to mitigate these novel pool attacks.



## **44. AI-Compass: A Comprehensive and Effective Multi-module Testing Tool for AI Systems**

cs.AI

**SubmitDate**: 2024-11-09    [abs](http://arxiv.org/abs/2411.06146v1) [paper-pdf](http://arxiv.org/pdf/2411.06146v1)

**Authors**: Zhiyu Zhu, Zhibo Jin, Hongsheng Hu, Minhui Xue, Ruoxi Sun, Seyit Camtepe, Praveen Gauravaram, Huaming Chen

**Abstract**: AI systems, in particular with deep learning techniques, have demonstrated superior performance for various real-world applications. Given the need for tailored optimization in specific scenarios, as well as the concerns related to the exploits of subsurface vulnerabilities, a more comprehensive and in-depth testing AI system becomes a pivotal topic. We have seen the emergence of testing tools in real-world applications that aim to expand testing capabilities. However, they often concentrate on ad-hoc tasks, rendering them unsuitable for simultaneously testing multiple aspects or components. Furthermore, trustworthiness issues arising from adversarial attacks and the challenge of interpreting deep learning models pose new challenges for developing more comprehensive and in-depth AI system testing tools. In this study, we design and implement a testing tool, \tool, to comprehensively and effectively evaluate AI systems. The tool extensively assesses multiple measurements towards adversarial robustness, model interpretability, and performs neuron analysis. The feasibility of the proposed testing tool is thoroughly validated across various modalities, including image classification, object detection, and text classification. Extensive experiments demonstrate that \tool is the state-of-the-art tool for a comprehensive assessment of the robustness and trustworthiness of AI systems. Our research sheds light on a general solution for AI systems testing landscape.



## **45. Robust Graph Neural Networks via Unbiased Aggregation**

cs.LG

NeurIPS 2024 poster. 28 pages, 14 figures

**SubmitDate**: 2024-11-09    [abs](http://arxiv.org/abs/2311.14934v2) [paper-pdf](http://arxiv.org/pdf/2311.14934v2)

**Authors**: Zhichao Hou, Ruiqi Feng, Tyler Derr, Xiaorui Liu

**Abstract**: The adversarial robustness of Graph Neural Networks (GNNs) has been questioned due to the false sense of security uncovered by strong adaptive attacks despite the existence of numerous defenses. In this work, we delve into the robustness analysis of representative robust GNNs and provide a unified robust estimation point of view to understand their robustness and limitations. Our novel analysis of estimation bias motivates the design of a robust and unbiased graph signal estimator. We then develop an efficient Quasi-Newton Iterative Reweighted Least Squares algorithm to solve the estimation problem, which is unfolded as robust unbiased aggregation layers in GNNs with theoretical guarantees. Our comprehensive experiments confirm the strong robustness of our proposed model under various scenarios, and the ablation study provides a deep understanding of its advantages. Our code is available at https://github.com/chris-hzc/RUNG.



## **46. Goal-guided Generative Prompt Injection Attack on Large Language Models**

cs.CR

11 pages, 6 figures

**SubmitDate**: 2024-11-09    [abs](http://arxiv.org/abs/2404.07234v4) [paper-pdf](http://arxiv.org/pdf/2404.07234v4)

**Authors**: Chong Zhang, Mingyu Jin, Qinkai Yu, Chengzhi Liu, Haochen Xue, Xiaobo Jin

**Abstract**: Current large language models (LLMs) provide a strong foundation for large-scale user-oriented natural language tasks. A large number of users can easily inject adversarial text or instructions through the user interface, thus causing LLMs model security challenges. Although there is currently a large amount of research on prompt injection attacks, most of these black-box attacks use heuristic strategies. It is unclear how these heuristic strategies relate to the success rate of attacks and thus effectively improve model robustness. To solve this problem, we redefine the goal of the attack: to maximize the KL divergence between the conditional probabilities of the clean text and the adversarial text. Furthermore, we prove that maximizing the KL divergence is equivalent to maximizing the Mahalanobis distance between the embedded representation $x$ and $x'$ of the clean text and the adversarial text when the conditional probability is a Gaussian distribution and gives a quantitative relationship on $x$ and $x'$. Then we designed a simple and effective goal-guided generative prompt injection strategy (G2PIA) to find an injection text that satisfies specific constraints to achieve the optimal attack effect approximately. It is particularly noteworthy that our attack method is a query-free black-box attack method with low computational cost. Experimental results on seven LLM models and four datasets show the effectiveness of our attack method.



## **47. Towards More Realistic Extraction Attacks: An Adversarial Perspective**

cs.CR

Presented at PrivateNLP@ACL2024

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2407.02596v2) [paper-pdf](http://arxiv.org/pdf/2407.02596v2)

**Authors**: Yash More, Prakhar Ganesh, Golnoosh Farnadi

**Abstract**: Language models are prone to memorizing parts of their training data which makes them vulnerable to extraction attacks. Existing research often examines isolated setups--such as evaluating extraction risks from a single model or with a fixed prompt design. However, a real-world adversary could access models across various sizes and checkpoints, as well as exploit prompt sensitivity, resulting in a considerably larger attack surface than previously studied. In this paper, we revisit extraction attacks from an adversarial perspective, focusing on how to leverage the brittleness of language models and the multi-faceted access to the underlying data. We find significant churn in extraction trends, i.e., even unintuitive changes to the prompt, or targeting smaller models and earlier checkpoints, can extract distinct information. By combining information from multiple attacks, our adversary is able to increase the extraction risks by up to $2 \times$. Furthermore, even with mitigation strategies like data deduplication, we find the same escalation of extraction risks against a real-world adversary. We conclude with a set of case studies, including detecting pre-training data, copyright violations, and extracting personally identifiable information, showing how our more realistic adversary can outperform existing adversaries in the literature.



## **48. A Survey of AI-Related Cyber Security Risks and Countermeasures in Mobility-as-a-Service**

cs.CR

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2411.05681v1) [paper-pdf](http://arxiv.org/pdf/2411.05681v1)

**Authors**: Kai-Fung Chu, Haiyue Yuan, Jinsheng Yuan, Weisi Guo, Nazmiye Balta-Ozkan, Shujun Li

**Abstract**: Mobility-as-a-Service (MaaS) integrates different transport modalities and can support more personalisation of travellers' journey planning based on their individual preferences, behaviours and wishes. To fully achieve the potential of MaaS, a range of AI (including machine learning and data mining) algorithms are needed to learn personal requirements and needs, to optimise journey planning of each traveller and all travellers as a whole, to help transport service operators and relevant governmental bodies to operate and plan their services, and to detect and prevent cyber attacks from various threat actors including dishonest and malicious travellers and transport operators. The increasing use of different AI and data processing algorithms in both centralised and distributed settings opens the MaaS ecosystem up to diverse cyber and privacy attacks at both the AI algorithm level and the connectivity surfaces. In this paper, we present the first comprehensive review on the coupling between AI-driven MaaS design and the diverse cyber security challenges related to cyber attacks and countermeasures. In particular, we focus on how current and emerging AI-facilitated privacy risks (profiling, inference, and third-party threats) and adversarial AI attacks (evasion, extraction, and gamification) may impact the MaaS ecosystem. These risks often combine novel attacks (e.g., inverse learning) with traditional attack vectors (e.g., man-in-the-middle attacks), exacerbating the risks for the wider participation actors and the emergence of new business models.



## **49. DeepDRK: Deep Dependency Regularized Knockoff for Feature Selection**

cs.LG

33 pages, 15 figures, 9 tables

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2402.17176v2) [paper-pdf](http://arxiv.org/pdf/2402.17176v2)

**Authors**: Hongyu Shen, Yici Yan, Zhizhen Zhao

**Abstract**: Model-X knockoff has garnered significant attention among various feature selection methods due to its guarantees for controlling the false discovery rate (FDR). Since its introduction in parametric design, knockoff techniques have evolved to handle arbitrary data distributions using deep learning-based generative models. However, we have observed limitations in the current implementations of the deep Model-X knockoff framework. Notably, the "swap property" that knockoffs require often faces challenges at the sample level, resulting in diminished selection power. To address these issues, we develop "Deep Dependency Regularized Knockoff (DeepDRK)," a distribution-free deep learning method that effectively balances FDR and power. In DeepDRK, we introduce a novel formulation of the knockoff model as a learning problem under multi-source adversarial attacks. By employing an innovative perturbation technique, we achieve lower FDR and higher power. Our model outperforms existing benchmarks across synthetic, semi-synthetic, and real-world datasets, particularly when sample sizes are small and data distributions are non-Gaussian.



## **50. Towards a Re-evaluation of Data Forging Attacks in Practice**

cs.CR

18 pages

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2411.05658v1) [paper-pdf](http://arxiv.org/pdf/2411.05658v1)

**Authors**: Mohamed Suliman, Anisa Halimi, Swanand Kadhe, Nathalie Baracaldo, Douglas Leith

**Abstract**: Data forging attacks provide counterfactual proof that a model was trained on a given dataset, when in fact, it was trained on another. These attacks work by forging (replacing) mini-batches with ones containing distinct training examples that produce nearly identical gradients. Data forging appears to break any potential avenues for data governance, as adversarial model owners may forge their training set from a dataset that is not compliant to one that is. Given these serious implications on data auditing and compliance, we critically analyse data forging from both a practical and theoretical point of view, finding that a key practical limitation of current attack methods makes them easily detectable by a verifier; namely that they cannot produce sufficiently identical gradients. Theoretically, we analyse the question of whether two distinct mini-batches can produce the same gradient. Generally, we find that while there may exist an infinite number of distinct mini-batches with real-valued training examples and labels that produce the same gradient, finding those that are within the allowed domain e.g. pixel values between 0-255 and one hot labels is a non trivial task. Our results call for the reevaluation of the strength of existing attacks, and for additional research into successful data forging, given the serious consequences it may have on machine learning and privacy.



