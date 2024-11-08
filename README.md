# Latest Adversarial Attack Papers
**update at 2024-11-08 16:09:02**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. FRACTURED-SORRY-Bench: Framework for Revealing Attacks in Conversational Turns Undermining Refusal Efficacy and Defenses over SORRY-Bench (Automated Multi-shot Jailbreaks)**

cs.CL

4 pages, 2 tables

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2408.16163v2) [paper-pdf](http://arxiv.org/pdf/2408.16163v2)

**Authors**: Aman Priyanshu, Supriti Vijay

**Abstract**: This paper introduces FRACTURED-SORRY-Bench, a framework for evaluating the safety of Large Language Models (LLMs) against multi-turn conversational attacks. Building upon the SORRY-Bench dataset, we propose a simple yet effective method for generating adversarial prompts by breaking down harmful queries into seemingly innocuous sub-questions. Our approach achieves a maximum increase of +46.22\% in Attack Success Rates (ASRs) across GPT-4, GPT-4o, GPT-4o-mini, and GPT-3.5-Turbo models compared to baseline methods. We demonstrate that this technique poses a challenge to current LLM safety measures and highlights the need for more robust defenses against subtle, multi-turn attacks.



## **2. Gradient Cuff: Detecting Jailbreak Attacks on Large Language Models by Exploring Refusal Loss Landscapes**

cs.CR

Accepted by NeurIPS 2024. Project page:  https://huggingface.co/spaces/TrustSafeAI/GradientCuff-Jailbreak-Defense

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2403.00867v3) [paper-pdf](http://arxiv.org/pdf/2403.00867v3)

**Authors**: Xiaomeng Hu, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Large Language Models (LLMs) are becoming a prominent generative AI tool, where the user enters a query and the LLM generates an answer. To reduce harm and misuse, efforts have been made to align these LLMs to human values using advanced training techniques such as Reinforcement Learning from Human Feedback (RLHF). However, recent studies have highlighted the vulnerability of LLMs to adversarial jailbreak attempts aiming at subverting the embedded safety guardrails. To address this challenge, this paper defines and investigates the Refusal Loss of LLMs and then proposes a method called Gradient Cuff to detect jailbreak attempts. Gradient Cuff exploits the unique properties observed in the refusal loss landscape, including functional values and its smoothness, to design an effective two-step detection strategy. Experimental results on two aligned LLMs (LLaMA-2-7B-Chat and Vicuna-7B-V1.5) and six types of jailbreak attacks (GCG, AutoDAN, PAIR, TAP, Base64, and LRL) show that Gradient Cuff can significantly improve the LLM's rejection capability for malicious jailbreak queries, while maintaining the model's performance for benign user queries by adjusting the detection threshold.



## **3. Attention Masks Help Adversarial Attacks to Bypass Safety Detectors**

cs.CR

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2411.04772v1) [paper-pdf](http://arxiv.org/pdf/2411.04772v1)

**Authors**: Yunfan Shi

**Abstract**: Despite recent research advancements in adversarial attack methods, current approaches against XAI monitors are still discoverable and slower. In this paper, we present an adaptive framework for attention mask generation to enable stealthy, explainable and efficient PGD image classification adversarial attack under XAI monitors. Specifically, we utilize mutation XAI mixture and multitask self-supervised X-UNet for attention mask generation to guide PGD attack. Experiments on MNIST (MLP), CIFAR-10 (AlexNet) have shown that our system can outperform benchmark PGD, Sparsefool and SOTA SINIFGSM in balancing among stealth, efficiency and explainability which is crucial for effectively fooling SOTA defense protected classifiers.



## **4. MISGUIDE: Security-Aware Attack Analytics for Smart Grid Load Frequency Control**

cs.CE

12 page journal

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2411.04731v1) [paper-pdf](http://arxiv.org/pdf/2411.04731v1)

**Authors**: Nur Imtiazul Haque, Prabin Mali, Mohammad Zakaria Haider, Mohammad Ashiqur Rahman, Sumit Paudyal

**Abstract**: Incorporating advanced information and communication technologies into smart grids (SGs) offers substantial operational benefits while increasing vulnerability to cyber threats like false data injection (FDI) attacks. Current SG attack analysis tools predominantly employ formal methods or adversarial machine learning (ML) techniques with rule-based bad data detectors to analyze the attack space. However, these attack analytics either generate simplistic attack vectors detectable by the ML-based anomaly detection models (ADMs) or fail to identify critical attack vectors from complex controller dynamics in a feasible time. This paper introduces MISGUIDE, a novel defense-aware attack analytics designed to extract verifiable multi-time slot-based FDI attack vectors from complex SG load frequency control dynamics and ADMs, utilizing the Gurobi optimizer. MISGUIDE can identify optimal (maliciously triggering under/over frequency relays in minimal time) and stealthy attack vectors. Using real-world load data, we validate the MISGUIDE-identified attack vectors through real-time hardware-in-the-loop (OPALRT) simulations of the IEEE 39-bus system.



## **5. Neural Fingerprints for Adversarial Attack Detection**

cs.CV

14 pages

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2411.04533v1) [paper-pdf](http://arxiv.org/pdf/2411.04533v1)

**Authors**: Haim Fisher, Moni Shahar, Yehezkel S. Resheff

**Abstract**: Deep learning models for image classification have become standard tools in recent years. A well known vulnerability of these models is their susceptibility to adversarial examples. These are generated by slightly altering an image of a certain class in a way that is imperceptible to humans but causes the model to classify it wrongly as another class. Many algorithms have been proposed to address this problem, falling generally into one of two categories: (i) building robust classifiers (ii) directly detecting attacked images. Despite the good performance of these detectors, we argue that in a white-box setting, where the attacker knows the configuration and weights of the network and the detector, they can overcome the detector by running many examples on a local copy, and sending only those that were not detected to the actual model. This problem is common in security applications where even a very good model is not sufficient to ensure safety. In this paper we propose to overcome this inherent limitation of any static defence with randomization. To do so, one must generate a very large family of detectors with consistent performance, and select one or more of them randomly for each input. For the individual detectors, we suggest the method of neural fingerprints. In the training phase, for each class we repeatedly sample a tiny random subset of neurons from certain layers of the network, and if their average is sufficiently different between clean and attacked images of the focal class they are considered a fingerprint and added to the detector bank. During test time, we sample fingerprints from the bank associated with the label predicted by the model, and detect attacks using a likelihood ratio test. We evaluate our detectors on ImageNet with different attack methods and model architectures, and show near-perfect detection with low rates of false detection.



## **6. Undermining Image and Text Classification Algorithms Using Adversarial Attacks**

cs.CR

Accepted for presentation at Electronic Imaging Conference 2025

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2411.03348v2) [paper-pdf](http://arxiv.org/pdf/2411.03348v2)

**Authors**: Langalibalele Lunga, Suhas Sreehari

**Abstract**: Machine learning models are prone to adversarial attacks, where inputs can be manipulated in order to cause misclassifications. While previous research has focused on techniques like Generative Adversarial Networks (GANs), there's limited exploration of GANs and Synthetic Minority Oversampling Technique (SMOTE) in text and image classification models to perform adversarial attacks. Our study addresses this gap by training various machine learning models and using GANs and SMOTE to generate additional data points aimed at attacking text classification models. Furthermore, we extend our investigation to face recognition models, training a Convolutional Neural Network(CNN) and subjecting it to adversarial attacks with fast gradient sign perturbations on key features identified by GradCAM, a technique used to highlight key image characteristics CNNs use in classification. Our experiments reveal a significant vulnerability in classification models. Specifically, we observe a 20 % decrease in accuracy for the top-performing text classification models post-attack, along with a 30 % decrease in facial recognition accuracy. This highlights the susceptibility of these models to manipulation of input data. Adversarial attacks not only compromise the security but also undermine the reliability of machine learning systems. By showcasing the impact of adversarial attacks on both text classification and face recognition models, our study underscores the urgent need for develop robust defenses against such vulnerabilities.



## **7. Game-Theoretic Defenses for Robust Conformal Prediction Against Adversarial Attacks in Medical Imaging**

cs.LG

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2411.04376v1) [paper-pdf](http://arxiv.org/pdf/2411.04376v1)

**Authors**: Rui Luo, Jie Bao, Zhixin Zhou, Chuangyin Dang

**Abstract**: Adversarial attacks pose significant threats to the reliability and safety of deep learning models, especially in critical domains such as medical imaging. This paper introduces a novel framework that integrates conformal prediction with game-theoretic defensive strategies to enhance model robustness against both known and unknown adversarial perturbations. We address three primary research questions: constructing valid and efficient conformal prediction sets under known attacks (RQ1), ensuring coverage under unknown attacks through conservative thresholding (RQ2), and determining optimal defensive strategies within a zero-sum game framework (RQ3). Our methodology involves training specialized defensive models against specific attack types and employing maximum and minimum classifiers to aggregate defenses effectively. Extensive experiments conducted on the MedMNIST datasets, including PathMNIST, OrganAMNIST, and TissueMNIST, demonstrate that our approach maintains high coverage guarantees while minimizing prediction set sizes. The game-theoretic analysis reveals that the optimal defensive strategy often converges to a singular robust model, outperforming uniform and simple strategies across all evaluated datasets. This work advances the state-of-the-art in uncertainty quantification and adversarial robustness, providing a reliable mechanism for deploying deep learning models in adversarial environments.



## **8. Towards Secured Smart Grid 2.0: Exploring Security Threats, Protection Models, and Challenges**

cs.NI

30 pages, 21 figures, 5 tables, accepted to appear in IEEE COMST

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2411.04365v1) [paper-pdf](http://arxiv.org/pdf/2411.04365v1)

**Authors**: Lan-Huong Nguyen, Van-Linh Nguyen, Ren-Hung Hwang, Jian-Jhih Kuo, Yu-Wen Chen, Chien-Chung Huang, Ping-I Pan

**Abstract**: Many nations are promoting the green transition in the energy sector to attain neutral carbon emissions by 2050. Smart Grid 2.0 (SG2) is expected to explore data-driven analytics and enhance communication technologies to improve the efficiency and sustainability of distributed renewable energy systems. These features are beyond smart metering and electric surplus distribution in conventional smart grids. Given the high dependence on communication networks to connect distributed microgrids in SG2, potential cascading failures of connectivity can cause disruption to data synchronization to the remote control systems. This paper reviews security threats and defense tactics for three stakeholders: power grid operators, communication network providers, and consumers. Through the survey, we found that SG2's stakeholders are particularly vulnerable to substation attacks/vandalism, malware/ransomware threats, blockchain vulnerabilities and supply chain breakdowns. Furthermore, incorporating artificial intelligence (AI) into autonomous energy management in distributed energy resources of SG2 creates new challenges. Accordingly, adversarial samples and false data injection on electricity reading and measurement sensors at power plants can fool AI-powered control functions and cause messy error-checking operations in energy storage, wrong energy estimation in electric vehicle charging, and even fraudulent transactions in peer-to-peer energy trading models. Scalable blockchain-based models, physical unclonable function, interoperable security protocols, and trustworthy AI models designed for managing distributed microgrids in SG2 are typical promising protection models for future research.



## **9. $B^4$: A Black-Box Scrubbing Attack on LLM Watermarks**

cs.CL

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2411.01222v3) [paper-pdf](http://arxiv.org/pdf/2411.01222v3)

**Authors**: Baizhou Huang, Xiao Pu, Xiaojun Wan

**Abstract**: Watermarking has emerged as a prominent technique for LLM-generated content detection by embedding imperceptible patterns. Despite supreme performance, its robustness against adversarial attacks remains underexplored. Previous work typically considers a grey-box attack setting, where the specific type of watermark is already known. Some even necessitates knowledge about hyperparameters of the watermarking method. Such prerequisites are unattainable in real-world scenarios. Targeting at a more realistic black-box threat model with fewer assumptions, we here propose $B^4$, a black-box scrubbing attack on watermarks. Specifically, we formulate the watermark scrubbing attack as a constrained optimization problem by capturing its objectives with two distributions, a Watermark Distribution and a Fidelity Distribution. This optimization problem can be approximately solved using two proxy distributions. Experimental results across 12 different settings demonstrate the superior performance of $B^4$ compared with other baselines.



## **10. Transferable Learned Image Compression-Resistant Adversarial Perturbations**

cs.CV

Accepted by BMVC 2024

**SubmitDate**: 2024-11-06    [abs](http://arxiv.org/abs/2401.03115v2) [paper-pdf](http://arxiv.org/pdf/2401.03115v2)

**Authors**: Yang Sui, Zhuohang Li, Ding Ding, Xiang Pan, Xiaozhong Xu, Shan Liu, Zhenzhong Chen

**Abstract**: Adversarial attacks can readily disrupt the image classification system, revealing the vulnerability of DNN-based recognition tasks. While existing adversarial perturbations are primarily applied to uncompressed images or compressed images by the traditional image compression method, i.e., JPEG, limited studies have investigated the robustness of models for image classification in the context of DNN-based image compression. With the rapid evolution of advanced image compression, DNN-based learned image compression has emerged as the promising approach for transmitting images in many security-critical applications, such as cloud-based face recognition and autonomous driving, due to its superior performance over traditional compression. Therefore, there is a pressing need to fully investigate the robustness of a classification system post-processed by learned image compression. To bridge this research gap, we explore the adversarial attack on a new pipeline that targets image classification models that utilize learned image compressors as pre-processing modules. Furthermore, to enhance the transferability of perturbations across various quality levels and architectures of learned image compression models, we introduce a saliency score-based sampling method to enable the fast generation of transferable perturbation. Extensive experiments with popular attack methods demonstrate the enhanced transferability of our proposed method when attacking images that have been post-processed with different learned image compression models.



## **11. DistriBlock: Identifying adversarial audio samples by leveraging characteristics of the output distribution**

cs.SD

Available at: https://proceedings.mlr.press/v244/pizarro24a.html

**SubmitDate**: 2024-11-06    [abs](http://arxiv.org/abs/2305.17000v8) [paper-pdf](http://arxiv.org/pdf/2305.17000v8)

**Authors**: Matías Pizarro, Dorothea Kolossa, Asja Fischer

**Abstract**: Adversarial attacks can mislead automatic speech recognition (ASR) systems into predicting an arbitrary target text, thus posing a clear security threat. To prevent such attacks, we propose DistriBlock, an efficient detection strategy applicable to any ASR system that predicts a probability distribution over output tokens in each time step. We measure a set of characteristics of this distribution: the median, maximum, and minimum over the output probabilities, the entropy of the distribution, as well as the Kullback-Leibler and the Jensen-Shannon divergence with respect to the distributions of the subsequent time step. Then, by leveraging the characteristics observed for both benign and adversarial data, we apply binary classifiers, including simple threshold-based classification, ensembles of such classifiers, and neural networks. Through extensive analysis across different state-of-the-art ASR systems and language data sets, we demonstrate the supreme performance of this approach, with a mean area under the receiver operating characteristic curve for distinguishing target adversarial examples against clean and noisy data of 99% and 97%, respectively. To assess the robustness of our method, we show that adaptive adversarial examples that can circumvent DistriBlock are much noisier, which makes them easier to detect through filtering and creates another avenue for preserving the system's robustness.



## **12. Reassessing Noise Augmentation Methods in the Context of Adversarial Speech**

eess.AS

**SubmitDate**: 2024-11-06    [abs](http://arxiv.org/abs/2409.01813v3) [paper-pdf](http://arxiv.org/pdf/2409.01813v3)

**Authors**: Karla Pizzi, Matías Pizarro, Asja Fischer

**Abstract**: In this study, we investigate if noise-augmented training can concurrently improve adversarial robustness in automatic speech recognition (ASR) systems. We conduct a comparative analysis of the adversarial robustness of four different state-of-the-art ASR architectures, where each of the ASR architectures is trained under three different augmentation conditions: one subject to background noise, speed variations, and reverberations, another subject to speed variations only, and a third without any form of data augmentation. The results demonstrate that noise augmentation not only improves model performance on noisy speech but also the model's robustness to adversarial attacks.



## **13. Robustifying automatic speech recognition by extracting slowly varying features**

eess.AS

**SubmitDate**: 2024-11-06    [abs](http://arxiv.org/abs/2112.07400v3) [paper-pdf](http://arxiv.org/pdf/2112.07400v3)

**Authors**: Matías Pizarro, Dorothea Kolossa, Asja Fischer

**Abstract**: In the past few years, it has been shown that deep learning systems are highly vulnerable under attacks with adversarial examples. Neural-network-based automatic speech recognition (ASR) systems are no exception. Targeted and untargeted attacks can modify an audio input signal in such a way that humans still recognise the same words, while ASR systems are steered to predict a different transcription. In this paper, we propose a defense mechanism against targeted adversarial attacks consisting in removing fast-changing features from the audio signals, either by applying slow feature analysis, a low-pass filter, or both, before feeding the input to the ASR system. We perform an empirical analysis of hybrid ASR models trained on data pre-processed in such a way. While the resulting models perform quite well on benign data, they are significantly more robust against targeted adversarial attacks: Our final, proposed model shows a performance on clean data similar to the baseline model, while being more than four times more robust.



## **14. Fundamental Limits of Routing Attack on Network Overload**

cs.NI

**SubmitDate**: 2024-11-06    [abs](http://arxiv.org/abs/2411.03749v1) [paper-pdf](http://arxiv.org/pdf/2411.03749v1)

**Authors**: Xinyu Wu, Eytan Modiano

**Abstract**: We quantify the threat of network adversaries to inducing \emph{network overload} through \emph{routing attacks}, where a subset of network nodes are hijacked by an adversary. We develop routing attacks on the hijacked nodes for two objectives related to overload: \emph{no-loss throughput minimization} and \emph{loss maximization}. The first objective attempts to identify a routing attack that minimizes the network's throughput that is guaranteed to survive. We develop a polynomial-time algorithm that can output the optimal routing attack in multi-hop networks with global information on the network's topology, and an algorithm with an approximation ratio of $2$ under partial information. The second objective attempts to maximize the throughput loss. We demonstrate that this problem is NP-hard, and develop two approximation algorithms with multiplicative and additive guarantees respectively in single-hop networks. We further investigate the adversary's optimal selection of nodes to hijack that can maximize network overload. We propose a heuristic polynomial-time algorithm to solve this NP-hard problem, and prove its optimality in special cases. We validate the near-optimal performance of the proposed algorithms over a wide range of network settings. Our results demonstrate that the proposed algorithms can accurately quantify the risk of overload given an arbitrary set of hijacked nodes and identify the critical nodes that should be protected against routing attacks.



## **15. The Early Bird Catches the Leak: Unveiling Timing Side Channels in LLM Serving Systems**

cs.CR

This work was submitted for review on Sept. 5, 2024, and the initial  version was uploaded to Arxiv on Sept. 30, 2024

**SubmitDate**: 2024-11-06    [abs](http://arxiv.org/abs/2409.20002v2) [paper-pdf](http://arxiv.org/pdf/2409.20002v2)

**Authors**: Linke Song, Zixuan Pang, Wenhao Wang, Zihao Wang, XiaoFeng Wang, Hongbo Chen, Wei Song, Yier Jin, Dan Meng, Rui Hou

**Abstract**: The wide deployment of Large Language Models (LLMs) has given rise to strong demands for optimizing their inference performance. Today's techniques serving this purpose primarily focus on reducing latency and improving throughput through algorithmic and hardware enhancements, while largely overlooking their privacy side effects, particularly in a multi-user environment. In our research, for the first time, we discovered a set of new timing side channels in LLM systems, arising from shared caches and GPU memory allocations, which can be exploited to infer both confidential system prompts and those issued by other users. These vulnerabilities echo security challenges observed in traditional computing systems, highlighting an urgent need to address potential information leakage in LLM serving infrastructures. In this paper, we report novel attack strategies designed to exploit such timing side channels inherent in LLM deployments, specifically targeting the Key-Value (KV) cache and semantic cache widely used to enhance LLM inference performance. Our approach leverages timing measurements and classification models to detect cache hits, allowing an adversary to infer private prompts with high accuracy. We also propose a token-by-token search algorithm to efficiently recover shared prompt prefixes in the caches, showing the feasibility of stealing system prompts and those produced by peer users. Our experimental studies on black-box testing of popular online LLM services demonstrate that such privacy risks are completely realistic, with significant consequences. Our findings underscore the need for robust mitigation to protect LLM systems against such emerging threats.



## **16. Formal Logic-guided Robust Federated Learning against Poisoning Attacks**

cs.CR

12 pages, 4 figures, 6 tables

**SubmitDate**: 2024-11-06    [abs](http://arxiv.org/abs/2411.03231v2) [paper-pdf](http://arxiv.org/pdf/2411.03231v2)

**Authors**: Dung Thuy Nguyen, Ziyan An, Taylor T. Johnson, Meiyi Ma, Kevin Leach

**Abstract**: Federated Learning (FL) offers a promising solution to the privacy concerns associated with centralized Machine Learning (ML) by enabling decentralized, collaborative learning. However, FL is vulnerable to various security threats, including poisoning attacks, where adversarial clients manipulate the training data or model updates to degrade overall model performance. Recognizing this threat, researchers have focused on developing defense mechanisms to counteract poisoning attacks in FL systems. However, existing robust FL methods predominantly focus on computer vision tasks, leaving a gap in addressing the unique challenges of FL with time series data. In this paper, we present FLORAL, a defense mechanism designed to mitigate poisoning attacks in federated learning for time-series tasks, even in scenarios with heterogeneous client data and a large number of adversarial participants. Unlike traditional model-centric defenses, FLORAL leverages logical reasoning to evaluate client trustworthiness by aligning their predictions with global time-series patterns, rather than relying solely on the similarity of client updates. Our approach extracts logical reasoning properties from clients, then hierarchically infers global properties, and uses these to verify client updates. Through formal logic verification, we assess the robustness of each client contribution, identifying deviations indicative of adversarial behavior. Experimental results on two datasets demonstrate the superior performance of our approach compared to existing baseline methods, highlighting its potential to enhance the robustness of FL to time series applications. Notably, FLORAL reduced the prediction error by 93.27% in the best-case scenario compared to the second-best baseline. Our code is available at https://anonymous.4open.science/r/FLORAL-Robust-FTS.



## **17. Benchmarking Vision Language Model Unlearning via Fictitious Facial Identity Dataset**

cs.CV

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2411.03554v1) [paper-pdf](http://arxiv.org/pdf/2411.03554v1)

**Authors**: Yingzi Ma, Jiongxiao Wang, Fei Wang, Siyuan Ma, Jiazhao Li, Xiujun Li, Furong Huang, Lichao Sun, Bo Li, Yejin Choi, Muhao Chen, Chaowei Xiao

**Abstract**: Machine unlearning has emerged as an effective strategy for forgetting specific information in the training data. However, with the increasing integration of visual data, privacy concerns in Vision Language Models (VLMs) remain underexplored. To address this, we introduce Facial Identity Unlearning Benchmark (FIUBench), a novel VLM unlearning benchmark designed to robustly evaluate the effectiveness of unlearning algorithms under the Right to be Forgotten setting. Specifically, we formulate the VLM unlearning task via constructing the Fictitious Facial Identity VQA dataset and apply a two-stage evaluation pipeline that is designed to precisely control the sources of information and their exposure levels. In terms of evaluation, since VLM supports various forms of ways to ask questions with the same semantic meaning, we also provide robust evaluation metrics including membership inference attacks and carefully designed adversarial privacy attacks to evaluate the performance of algorithms. Through the evaluation of four baseline VLM unlearning algorithms within FIUBench, we find that all methods remain limited in their unlearning performance, with significant trade-offs between model utility and forget quality. Furthermore, our findings also highlight the importance of privacy attacks for robust evaluations. We hope FIUBench will drive progress in developing more effective VLM unlearning algorithms.



## **18. Introducing Perturb-ability Score (PS) to Enhance Robustness Against Evasion Adversarial Attacks on ML-NIDS**

cs.CR

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2409.07448v2) [paper-pdf](http://arxiv.org/pdf/2409.07448v2)

**Authors**: Mohamed elShehaby, Ashraf Matrawy

**Abstract**: As network security threats continue to evolve, safeguarding Machine Learning (ML)-based Network Intrusion Detection Systems (NIDS) from adversarial attacks is crucial. This paper introduces the notion of feature perturb-ability and presents a novel Perturb-ability Score (PS) metric that identifies NIDS features susceptible to manipulation in the problem-space by an attacker. By quantifying a feature's susceptibility to perturbations within the problem-space, the PS facilitates the selection of features that are inherently more robust against evasion adversarial attacks on ML-NIDS during the feature selection phase. These features exhibit natural resilience to perturbations, as they are heavily constrained by the problem-space limitations and correlations of the NIDS domain. Furthermore, manipulating these features may either disrupt the malicious function of evasion adversarial attacks on NIDS or render the network traffic invalid for processing (or both). This proposed novel approach employs a fresh angle by leveraging network domain constraints as a defense mechanism against problem-space evasion adversarial attacks targeting ML-NIDS. We demonstrate the effectiveness of our PS-guided feature selection defense in enhancing NIDS robustness. Experimental results across various ML-based NIDS models and public datasets show that selecting only robust features (low-PS features) can maintain solid detection performance while significantly reducing vulnerability to evasion adversarial attacks. Additionally, our findings verify that the PS effectively identifies NIDS features highly vulnerable to problem-space perturbations.



## **19. Oblivious Defense in ML Models: Backdoor Removal without Detection**

cs.LG

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2411.03279v1) [paper-pdf](http://arxiv.org/pdf/2411.03279v1)

**Authors**: Shafi Goldwasser, Jonathan Shafer, Neekon Vafa, Vinod Vaikuntanathan

**Abstract**: As society grows more reliant on machine learning, ensuring the security of machine learning systems against sophisticated attacks becomes a pressing concern. A recent result of Goldwasser, Kim, Vaikuntanathan, and Zamir (2022) shows that an adversary can plant undetectable backdoors in machine learning models, allowing the adversary to covertly control the model's behavior. Backdoors can be planted in such a way that the backdoored machine learning model is computationally indistinguishable from an honest model without backdoors.   In this paper, we present strategies for defending against backdoors in ML models, even if they are undetectable. The key observation is that it is sometimes possible to provably mitigate or even remove backdoors without needing to detect them, using techniques inspired by the notion of random self-reducibility. This depends on properties of the ground-truth labels (chosen by nature), and not of the proposed ML model (which may be chosen by an attacker).   We give formal definitions for secure backdoor mitigation, and proceed to show two types of results. First, we show a "global mitigation" technique, which removes all backdoors from a machine learning model under the assumption that the ground-truth labels are close to a Fourier-heavy function. Second, we consider distributions where the ground-truth labels are close to a linear or polynomial function in $\mathbb{R}^n$. Here, we show "local mitigation" techniques, which remove backdoors with high probability for every inputs of interest, and are computationally cheaper than global mitigation. All of our constructions are black-box, so our techniques work without needing access to the model's representation (i.e., its code or parameters). Along the way we prove a simple result for robust mean estimation.



## **20. Gradient-Guided Conditional Diffusion Models for Private Image Reconstruction: Analyzing Adversarial Impacts of Differential Privacy and Denoising**

cs.CV

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2411.03053v1) [paper-pdf](http://arxiv.org/pdf/2411.03053v1)

**Authors**: Tao Huang, Jiayang Meng, Hong Chen, Guolong Zheng, Xu Yang, Xun Yi, Hua Wang

**Abstract**: We investigate the construction of gradient-guided conditional diffusion models for reconstructing private images, focusing on the adversarial interplay between differential privacy noise and the denoising capabilities of diffusion models. While current gradient-based reconstruction methods struggle with high-resolution images due to computational complexity and prior knowledge requirements, we propose two novel methods that require minimal modifications to the diffusion model's generation process and eliminate the need for prior knowledge. Our approach leverages the strong image generation capabilities of diffusion models to reconstruct private images starting from randomly generated noise, even when a small amount of differentially private noise has been added to the gradients. We also conduct a comprehensive theoretical analysis of the impact of differential privacy noise on the quality of reconstructed images, revealing the relationship among noise magnitude, the architecture of attacked models, and the attacker's reconstruction capability. Additionally, extensive experiments validate the effectiveness of our proposed methods and the accuracy of our theoretical findings, suggesting new directions for privacy risk auditing using conditional diffusion models.



## **21. Adversarial Markov Games: On Adaptive Decision-Based Attacks and Defenses**

cs.AI

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2312.13435v2) [paper-pdf](http://arxiv.org/pdf/2312.13435v2)

**Authors**: Ilias Tsingenopoulos, Vera Rimmer, Davy Preuveneers, Fabio Pierazzi, Lorenzo Cavallaro, Wouter Joosen

**Abstract**: Despite considerable efforts on making them robust, real-world ML-based systems remain vulnerable to decision based attacks, as definitive proofs of their operational robustness have so far proven intractable. The canonical approach in robustness evaluation calls for adaptive attacks, that is with complete knowledge of the defense and tailored to bypass it. In this study, we introduce a more expansive notion of being adaptive and show how attacks but also defenses can benefit by it and by learning from each other through interaction. We propose and evaluate a framework for adaptively optimizing black-box attacks and defenses against each other through the competitive game they form. To reliably measure robustness, it is important to evaluate against realistic and worst-case attacks. We thus augment both attacks and the evasive arsenal at their disposal through adaptive control, and observe that the same can be done for defenses, before we evaluate them first apart and then jointly under a multi-agent perspective. We demonstrate that active defenses, which control how the system responds, are a necessary complement to model hardening when facing decision-based attacks; then how these defenses can be circumvented by adaptive attacks, only to finally elicit active and adaptive defenses. We validate our observations through a wide theoretical and empirical investigation to confirm that AI-enabled adversaries pose a considerable threat to black-box ML-based systems, rekindling the proverbial arms race where defenses have to be AI-enabled too. Succinctly, we address the challenges posed by adaptive adversaries and develop adaptive defenses, thereby laying out effective strategies in ensuring the robustness of ML-based systems deployed in the real-world.



## **22. Flashy Backdoor: Real-world Environment Backdoor Attack on SNNs with DVS Cameras**

cs.CR

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2411.03022v1) [paper-pdf](http://arxiv.org/pdf/2411.03022v1)

**Authors**: Roberto Riaño, Gorka Abad, Stjepan Picek, Aitor Urbieta

**Abstract**: While security vulnerabilities in traditional Deep Neural Networks (DNNs) have been extensively studied, the susceptibility of Spiking Neural Networks (SNNs) to adversarial attacks remains mostly underexplored. Until now, the mechanisms to inject backdoors into SNN models have been limited to digital scenarios; thus, we present the first evaluation of backdoor attacks in real-world environments.   We begin by assessing the applicability of existing digital backdoor attacks and identifying their limitations for deployment in physical environments. To address each of the found limitations, we present three novel backdoor attack methods on SNNs, i.e., Framed, Strobing, and Flashy Backdoor. We also assess the effectiveness of traditional backdoor procedures and defenses adapted for SNNs, such as pruning, fine-tuning, and fine-pruning. The results show that while these procedures and defenses can mitigate some attacks, they often fail against stronger methods like Flashy Backdoor or sacrifice too much clean accuracy, rendering the models unusable.   Overall, all our methods can achieve up to a 100% Attack Success Rate while maintaining high clean accuracy in every tested dataset. Additionally, we evaluate the stealthiness of the triggers with commonly used metrics, finding them highly stealthy. Thus, we propose new alternatives more suited for identifying poisoned samples in these scenarios. Our results show that further research is needed to ensure the security of SNN-based systems against backdoor attacks and their safe application in real-world scenarios. The code, experiments, and results are available in our repository.



## **23. Region-Guided Attack on the Segment Anything Model (SAM)**

cs.CV

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2411.02974v1) [paper-pdf](http://arxiv.org/pdf/2411.02974v1)

**Authors**: Xiaoliang Liu, Furao Shen, Jian Zhao

**Abstract**: The Segment Anything Model (SAM) is a cornerstone of image segmentation, demonstrating exceptional performance across various applications, particularly in autonomous driving and medical imaging, where precise segmentation is crucial. However, SAM is vulnerable to adversarial attacks that can significantly impair its functionality through minor input perturbations. Traditional techniques, such as FGSM and PGD, are often ineffective in segmentation tasks due to their reliance on global perturbations that overlook spatial nuances. Recent methods like Attack-SAM-K and UAD have begun to address these challenges, but they frequently depend on external cues and do not fully leverage the structural interdependencies within segmentation processes. This limitation underscores the need for a novel adversarial strategy that exploits the unique characteristics of segmentation tasks. In response, we introduce the Region-Guided Attack (RGA), designed specifically for SAM. RGA utilizes a Region-Guided Map (RGM) to manipulate segmented regions, enabling targeted perturbations that fragment large segments and expand smaller ones, resulting in erroneous outputs from SAM. Our experiments demonstrate that RGA achieves high success rates in both white-box and black-box scenarios, emphasizing the need for robust defenses against such sophisticated attacks. RGA not only reveals SAM's vulnerabilities but also lays the groundwork for developing more resilient defenses against adversarial threats in image segmentation.



## **24. Bias in the Mirror: Are LLMs opinions robust to their own adversarial attacks ?**

cs.CL

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2410.13517v2) [paper-pdf](http://arxiv.org/pdf/2410.13517v2)

**Authors**: Virgile Rennard, Christos Xypolopoulos, Michalis Vazirgiannis

**Abstract**: Large language models (LLMs) inherit biases from their training data and alignment processes, influencing their responses in subtle ways. While many studies have examined these biases, little work has explored their robustness during interactions. In this paper, we introduce a novel approach where two instances of an LLM engage in self-debate, arguing opposing viewpoints to persuade a neutral version of the model. Through this, we evaluate how firmly biases hold and whether models are susceptible to reinforcing misinformation or shifting to harmful viewpoints. Our experiments span multiple LLMs of varying sizes, origins, and languages, providing deeper insights into bias persistence and flexibility across linguistic and cultural contexts.



## **25. Towards Efficient Transferable Preemptive Adversarial Defense**

cs.CR

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2407.15524v3) [paper-pdf](http://arxiv.org/pdf/2407.15524v3)

**Authors**: Hanrui Wang, Ching-Chun Chang, Chun-Shien Lu, Isao Echizen

**Abstract**: Deep learning technology has brought convenience and advanced developments but has become untrustworthy because of its sensitivity to inconspicuous perturbations (i.e., adversarial attacks). Attackers may utilize this sensitivity to manipulate predictions. To defend against such attacks, we have devised a proactive strategy for "attacking" the medias before it is attacked by the third party, so that when the protected medias are further attacked, the adversarial perturbations are automatically neutralized. This strategy, dubbed Fast Preemption, provides an efficient transferable preemptive defense by using different models for labeling inputs and learning crucial features. A forward-backward cascade learning algorithm is used to compute protective perturbations, starting with forward propagation optimization to achieve rapid convergence, followed by iterative backward propagation learning to alleviate overfitting. This strategy offers state-of-the-art transferability and protection across various systems. With the running of only three steps, our Fast Preemption framework outperforms benchmark training-time, test-time, and preemptive adversarial defenses. We have also devised the first to our knowledge effective white-box adaptive reversion attack and demonstrate that the protection added by our defense strategy is irreversible unless the backbone model, algorithm, and settings are fully compromised. This work provides a new direction to developing proactive defenses against adversarial attacks. The proposed methodology will be made available on GitHub.



## **26. Query-Efficient Adversarial Attack Against Vertical Federated Graph Learning**

cs.LG

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2411.02809v1) [paper-pdf](http://arxiv.org/pdf/2411.02809v1)

**Authors**: Jinyin Chen, Wenbo Mu, Luxin Zhang, Guohan Huang, Haibin Zheng, Yao Cheng

**Abstract**: Graph neural network (GNN) has captured wide attention due to its capability of graph representation learning for graph-structured data. However, the distributed data silos limit the performance of GNN. Vertical federated learning (VFL), an emerging technique to process distributed data, successfully makes GNN possible to handle the distributed graph-structured data. Despite the prosperous development of vertical federated graph learning (VFGL), the robustness of VFGL against the adversarial attack has not been explored yet. Although numerous adversarial attacks against centralized GNNs are proposed, their attack performance is challenged in the VFGL scenario. To the best of our knowledge, this is the first work to explore the adversarial attack against VFGL. A query-efficient hybrid adversarial attack framework is proposed to significantly improve the centralized adversarial attacks against VFGL, denoted as NA2, short for Neuron-based Adversarial Attack. Specifically, a malicious client manipulates its local training data to improve its contribution in a stealthy fashion. Then a shadow model is established based on the manipulated data to simulate the behavior of the server model in VFGL. As a result, the shadow model can improve the attack success rate of various centralized attacks with a few queries. Extensive experiments on five real-world benchmarks demonstrate that NA2 improves the performance of the centralized adversarial attacks against VFGL, achieving state-of-the-art performance even under potential adaptive defense where the defender knows the attack method. Additionally, we provide interpretable experiments of the effectiveness of NA2 via sensitive neurons identification and visualization of t-SNE.



## **27. TRANSPOSE: Transitional Approaches for Spatially-Aware LFI Resilient FSM Encoding**

cs.CR

14 pages, 11 figures

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2411.02798v1) [paper-pdf](http://arxiv.org/pdf/2411.02798v1)

**Authors**: Muhtadi Choudhury, Minyan Gao, Avinash Varna, Elad Peer, Domenic Forte

**Abstract**: Finite state machines (FSMs) regulate sequential circuits, including access to sensitive information and privileged CPU states. Courtesy of contemporary research on laser attacks, laser-based fault injection (LFI) is becoming even more precise where an adversary can thwart chip security by altering individual flip-flop (FF) values. Different laser models, e.g., bit flip, bit set, and bit reset, have been developed to appreciate LFI on practical targets. As traditional approaches may incorporate substantial overhead, state-based SPARSE and transition-based TAMED countermeasures were proposed in our prior work to improve FSM resiliency efficiently. TAMED overcame SPARSE's limitation of being too conservative, and generating multiple LFI resilient encodings for contemporary LFI models on demand. SPARSE, however, incorporated design layout information into its vulnerability estimation which makes its vulnerability estimation metric more accurate. In this paper, we extend TAMED by proposing a transition-based encoding CAD framework (TRANSPOSE), that incorporates spatial transitional vulnerability metrics to quantify design susceptibility of FSMs based on both the bit flip model and the set-reset models. TRANSPOSE also incorporates floorplan optimization into its framework to accommodate secure spatial inter-distance of FF-sensitive regions. All TRANSPOSE approaches are demonstrated on 5 multifarious benchmarks and outperform existing FSM encoding schemes/frameworks in terms of security and overhead.



## **28. Stochastic Monkeys at Play: Random Augmentations Cheaply Break LLM Safety Alignment**

cs.LG

Under peer review

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2411.02785v1) [paper-pdf](http://arxiv.org/pdf/2411.02785v1)

**Authors**: Jason Vega, Junsheng Huang, Gaokai Zhang, Hangoo Kang, Minjia Zhang, Gagandeep Singh

**Abstract**: Safety alignment of Large Language Models (LLMs) has recently become a critical objective of model developers. In response, a growing body of work has been investigating how safety alignment can be bypassed through various jailbreaking methods, such as adversarial attacks. However, these jailbreak methods can be rather costly or involve a non-trivial amount of creativity and effort, introducing the assumption that malicious users are high-resource or sophisticated. In this paper, we study how simple random augmentations to the input prompt affect safety alignment effectiveness in state-of-the-art LLMs, such as Llama 3 and Qwen 2. We perform an in-depth evaluation of 17 different models and investigate the intersection of safety under random augmentations with multiple dimensions: augmentation type, model size, quantization, fine-tuning-based defenses, and decoding strategies (e.g., sampling temperature). We show that low-resource and unsophisticated attackers, i.e. $\textit{stochastic monkeys}$, can significantly improve their chances of bypassing alignment with just 25 random augmentations per prompt.



## **29. Steering cooperation: Adversarial attacks on prisoner's dilemma in complex networks**

physics.soc-ph

19 pages, 4 figures

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2406.19692v4) [paper-pdf](http://arxiv.org/pdf/2406.19692v4)

**Authors**: Kazuhiro Takemoto

**Abstract**: This study examines the application of adversarial attack concepts to control the evolution of cooperation in the prisoner's dilemma game in complex networks. Specifically, it proposes a simple adversarial attack method that drives players' strategies towards a target state by adding small perturbations to social networks. The proposed method is evaluated on both model and real-world networks. Numerical simulations demonstrate that the proposed method can effectively promote cooperation with significantly smaller perturbations compared to other techniques. Additionally, this study shows that adversarial attacks can also be useful in inhibiting cooperation (promoting defection). The findings reveal that adversarial attacks on social networks can be potent tools for both promoting and inhibiting cooperation, opening new possibilities for controlling cooperative behavior in social systems while also highlighting potential risks.



## **30. Semantic-Aligned Adversarial Evolution Triangle for High-Transferability Vision-Language Attack**

cs.CV

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2411.02669v1) [paper-pdf](http://arxiv.org/pdf/2411.02669v1)

**Authors**: Xiaojun Jia, Sensen Gao, Qing Guo, Ke Ma, Yihao Huang, Simeng Qin, Yang Liu, Ivor Tsang Fellow, Xiaochun Cao

**Abstract**: Vision-language pre-training (VLP) models excel at interpreting both images and text but remain vulnerable to multimodal adversarial examples (AEs). Advancing the generation of transferable AEs, which succeed across unseen models, is key to developing more robust and practical VLP models. Previous approaches augment image-text pairs to enhance diversity within the adversarial example generation process, aiming to improve transferability by expanding the contrast space of image-text features. However, these methods focus solely on diversity around the current AEs, yielding limited gains in transferability. To address this issue, we propose to increase the diversity of AEs by leveraging the intersection regions along the adversarial trajectory during optimization. Specifically, we propose sampling from adversarial evolution triangles composed of clean, historical, and current adversarial examples to enhance adversarial diversity. We provide a theoretical analysis to demonstrate the effectiveness of the proposed adversarial evolution triangle. Moreover, we find that redundant inactive dimensions can dominate similarity calculations, distorting feature matching and making AEs model-dependent with reduced transferability. Hence, we propose to generate AEs in the semantic image-text feature contrast space, which can project the original feature space into a semantic corpus subspace. The proposed semantic-aligned subspace can reduce the image feature redundancy, thereby improving adversarial transferability. Extensive experiments across different datasets and models demonstrate that the proposed method can effectively improve adversarial transferability and outperform state-of-the-art adversarial attack methods. The code is released at https://github.com/jiaxiaojunQAQ/SA-AET.



## **31. Class-Conditioned Transformation for Enhanced Robust Image Classification**

cs.CV

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2303.15409v2) [paper-pdf](http://arxiv.org/pdf/2303.15409v2)

**Authors**: Tsachi Blau, Roy Ganz, Chaim Baskin, Michael Elad, Alex M. Bronstein

**Abstract**: Robust classification methods predominantly concentrate on algorithms that address a specific threat model, resulting in ineffective defenses against other threat models. Real-world applications are exposed to this vulnerability, as malicious attackers might exploit alternative threat models. In this work, we propose a novel test-time threat model agnostic algorithm that enhances Adversarial-Trained (AT) models. Our method operates through COnditional image transformation and DIstance-based Prediction (CODIP) and includes two main steps: First, we transform the input image into each dataset class, where the input image might be either clean or attacked. Next, we make a prediction based on the shortest transformed distance. The conditional transformation utilizes the perceptually aligned gradients property possessed by AT models and, as a result, eliminates the need for additional models or additional training. Moreover, it allows users to choose the desired balance between clean and robust accuracy without training. The proposed method achieves state-of-the-art results demonstrated through extensive experiments on various models, AT methods, datasets, and attack types. Notably, applying CODIP leads to substantial robust accuracy improvement of up to $+23\%$, $+20\%$, $+26\%$, and $+22\%$ on CIFAR10, CIFAR100, ImageNet and Flowers datasets, respectively.



## **32. Attacking Vision-Language Computer Agents via Pop-ups**

cs.CL

10 pages, preprint

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2411.02391v1) [paper-pdf](http://arxiv.org/pdf/2411.02391v1)

**Authors**: Yanzhe Zhang, Tao Yu, Diyi Yang

**Abstract**: Autonomous agents powered by large vision and language models (VLM) have demonstrated significant potential in completing daily computer tasks, such as browsing the web to book travel and operating desktop software, which requires agents to understand these interfaces. Despite such visual inputs becoming more integrated into agentic applications, what types of risks and attacks exist around them still remain unclear. In this work, we demonstrate that VLM agents can be easily attacked by a set of carefully designed adversarial pop-ups, which human users would typically recognize and ignore. This distraction leads agents to click these pop-ups instead of performing the tasks as usual. Integrating these pop-ups into existing agent testing environments like OSWorld and VisualWebArena leads to an attack success rate (the frequency of the agent clicking the pop-ups) of 86% on average and decreases the task success rate by 47%. Basic defense techniques such as asking the agent to ignore pop-ups or including an advertisement notice, are ineffective against the attack.



## **33. Swiper: a new paradigm for efficient weighted distributed protocols**

cs.DC

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2307.15561v2) [paper-pdf](http://arxiv.org/pdf/2307.15561v2)

**Authors**: Andrei Tonkikh, Luciano Freitas

**Abstract**: The majority of fault-tolerant distributed algorithms are designed assuming a nominal corruption model, in which at most a fraction $f_n$ of parties can be corrupted by the adversary. However, due to the infamous Sybil attack, nominal models are not sufficient to express the trust assumptions in open (i.e., permissionless) settings. Instead, permissionless systems typically operate in a weighted model, where each participant is associated with a weight and the adversary can corrupt a set of parties holding at most a fraction $f_w$ of the total weight.   In this paper, we suggest a simple way to transform a large class of protocols designed for the nominal model into the weighted model. To this end, we formalize and solve three novel optimization problems, which we collectively call the weight reduction problems, that allow us to map large real weights into small integer weights while preserving the properties necessary for the correctness of the protocols. In all cases, we manage to keep the sum of the integer weights to be at most linear in the number of parties, resulting in extremely efficient protocols for the weighted model. Moreover, we demonstrate that, on weight distributions that emerge in practice, the sum of the integer weights tends to be far from the theoretical worst case and, sometimes, even smaller than the number of participants.   While, for some protocols, our transformation requires an arbitrarily small reduction in resilience (i.e., $f_w = f_n - \epsilon$), surprisingly, for many important problems, we manage to obtain weighted solutions with the same resilience ($f_w = f_n$) as nominal ones. Notable examples include erasure-coded distributed storage and broadcast protocols, verifiable secret sharing, and asynchronous consensus.



## **34. Manipulation Facing Threats: Evaluating Physical Vulnerabilities in End-to-End Vision Language Action Models**

cs.CV

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2409.13174v2) [paper-pdf](http://arxiv.org/pdf/2409.13174v2)

**Authors**: Hao Cheng, Erjia Xiao, Chengyuan Yu, Zhao Yao, Jiahang Cao, Qiang Zhang, Jiaxu Wang, Mengshu Sun, Kaidi Xu, Jindong Gu, Renjing Xu

**Abstract**: Recently, driven by advancements in Multimodal Large Language Models (MLLMs), Vision Language Action Models (VLAMs) are being proposed to achieve better performance in open-vocabulary scenarios for robotic manipulation tasks. Since manipulation tasks involve direct interaction with the physical world, ensuring robustness and safety during the execution of this task is always a very critical issue. In this paper, by synthesizing current safety research on MLLMs and the specific application scenarios of the manipulation task in the physical world, we comprehensively evaluate VLAMs in the face of potential physical threats. Specifically, we propose the Physical Vulnerability Evaluating Pipeline (PVEP) that can incorporate as many visual modal physical threats as possible for evaluating the physical robustness of VLAMs. The physical threats in PVEP specifically include Out-of-Distribution, Typography-based Visual Prompts, and Adversarial Patch Attacks. By comparing the performance fluctuations of VLAMs before and after being attacked, we provide generalizable Analyses of how VLAMs respond to different physical security threats. Our project page is in this link: https://chaducheng.github.io/Manipulat-Facing-Threats/.



## **35. Alignment-Based Adversarial Training (ABAT) for Improving the Robustness and Accuracy of EEG-Based BCIs**

cs.HC

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2411.02094v1) [paper-pdf](http://arxiv.org/pdf/2411.02094v1)

**Authors**: Xiaoqing Chen, Ziwei Wang, Dongrui Wu

**Abstract**: Machine learning has achieved great success in electroencephalogram (EEG) based brain-computer interfaces (BCIs). Most existing BCI studies focused on improving the decoding accuracy, with only a few considering the adversarial security. Although many adversarial defense approaches have been proposed in other application domains such as computer vision, previous research showed that their direct extensions to BCIs degrade the classification accuracy on benign samples. This phenomenon greatly affects the applicability of adversarial defense approaches to EEG-based BCIs. To mitigate this problem, we propose alignment-based adversarial training (ABAT), which performs EEG data alignment before adversarial training. Data alignment aligns EEG trials from different domains to reduce their distribution discrepancies, and adversarial training further robustifies the classification boundary. The integration of data alignment and adversarial training can make the trained EEG classifiers simultaneously more accurate and more robust. Experiments on five EEG datasets from two different BCI paradigms (motor imagery classification, and event related potential recognition), three convolutional neural network classifiers (EEGNet, ShallowCNN and DeepCNN) and three different experimental settings (offline within-subject cross-block/-session classification, online cross-session classification, and pre-trained classifiers) demonstrated its effectiveness. It is very intriguing that adversarial attacks, which are usually used to damage BCI systems, can be used in ABAT to simultaneously improve the model accuracy and robustness.



## **36. Exploiting LLM Quantization**

cs.LG

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2405.18137v2) [paper-pdf](http://arxiv.org/pdf/2405.18137v2)

**Authors**: Kazuki Egashira, Mark Vero, Robin Staab, Jingxuan He, Martin Vechev

**Abstract**: Quantization leverages lower-precision weights to reduce the memory usage of large language models (LLMs) and is a key technique for enabling their deployment on commodity hardware. While LLM quantization's impact on utility has been extensively explored, this work for the first time studies its adverse effects from a security perspective. We reveal that widely used quantization methods can be exploited to produce a harmful quantized LLM, even though the full-precision counterpart appears benign, potentially tricking users into deploying the malicious quantized model. We demonstrate this threat using a three-staged attack framework: (i) first, we obtain a malicious LLM through fine-tuning on an adversarial task; (ii) next, we quantize the malicious model and calculate constraints that characterize all full-precision models that map to the same quantized model; (iii) finally, using projected gradient descent, we tune out the poisoned behavior from the full-precision model while ensuring that its weights satisfy the constraints computed in step (ii). This procedure results in an LLM that exhibits benign behavior in full precision but when quantized, it follows the adversarial behavior injected in step (i). We experimentally demonstrate the feasibility and severity of such an attack across three diverse scenarios: vulnerable code generation, content injection, and over-refusal attack. In practice, the adversary could host the resulting full-precision model on an LLM community hub such as Hugging Face, exposing millions of users to the threat of deploying its malicious quantized version on their devices.



## **37. DeSparsify: Adversarial Attack Against Token Sparsification Mechanisms in Vision Transformers**

cs.CV

18 pages, 6 figures

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2402.02554v2) [paper-pdf](http://arxiv.org/pdf/2402.02554v2)

**Authors**: Oryan Yehezkel, Alon Zolfi, Amit Baras, Yuval Elovici, Asaf Shabtai

**Abstract**: Vision transformers have contributed greatly to advancements in the computer vision domain, demonstrating state-of-the-art performance in diverse tasks (e.g., image classification, object detection). However, their high computational requirements grow quadratically with the number of tokens used. Token sparsification mechanisms have been proposed to address this issue. These mechanisms employ an input-dependent strategy, in which uninformative tokens are discarded from the computation pipeline, improving the model's efficiency. However, their dynamism and average-case assumption makes them vulnerable to a new threat vector - carefully crafted adversarial examples capable of fooling the sparsification mechanism, resulting in worst-case performance. In this paper, we present DeSparsify, an attack targeting the availability of vision transformers that use token sparsification mechanisms. The attack aims to exhaust the operating system's resources, while maintaining its stealthiness. Our evaluation demonstrates the attack's effectiveness on three token sparsification mechanisms and examines the attack's transferability between them and its effect on the GPU resources. To mitigate the impact of the attack, we propose various countermeasures.



## **38. LiDAttack: Robust Black-box Attack on LiDAR-based Object Detection**

cs.CV

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2411.01889v1) [paper-pdf](http://arxiv.org/pdf/2411.01889v1)

**Authors**: Jinyin Chen, Danxin Liao, Sheng Xiang, Haibin Zheng

**Abstract**: Since DNN is vulnerable to carefully crafted adversarial examples, adversarial attack on LiDAR sensors have been extensively studied. We introduce a robust black-box attack dubbed LiDAttack. It utilizes a genetic algorithm with a simulated annealing strategy to strictly limit the location and number of perturbation points, achieving a stealthy and effective attack. And it simulates scanning deviations, allowing it to adapt to dynamic changes in real world scenario variations. Extensive experiments are conducted on 3 datasets (i.e., KITTI, nuScenes, and self-constructed data) with 3 dominant object detection models (i.e., PointRCNN, PointPillar, and PV-RCNN++). The results reveal the efficiency of the LiDAttack when targeting a wide range of object detection models, with an attack success rate (ASR) up to 90%.



## **39. Learning predictable and robust neural representations by straightening image sequences**

cs.CV

Accepted at NeurIPS 2024

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2411.01777v1) [paper-pdf](http://arxiv.org/pdf/2411.01777v1)

**Authors**: Xueyan Niu, Cristina Savin, Eero P. Simoncelli

**Abstract**: Prediction is a fundamental capability of all living organisms, and has been proposed as an objective for learning sensory representations. Recent work demonstrates that in primate visual systems, prediction is facilitated by neural representations that follow straighter temporal trajectories than their initial photoreceptor encoding, which allows for prediction by linear extrapolation. Inspired by these experimental findings, we develop a self-supervised learning (SSL) objective that explicitly quantifies and promotes straightening. We demonstrate the power of this objective in training deep feedforward neural networks on smoothly-rendered synthetic image sequences that mimic commonly-occurring properties of natural videos. The learned model contains neural embeddings that are predictive, but also factorize the geometric, photometric, and semantic attributes of objects. The representations also prove more robust to noise and adversarial attacks compared to previous SSL methods that optimize for invariance to random augmentations. Moreover, these beneficial properties can be transferred to other training procedures by using the straightening objective as a regularizer, suggesting a broader utility for straightening as a principle for robust unsupervised learning.



## **40. Survey on Adversarial Attack and Defense for Medical Image Analysis: Methods and Challenges**

eess.IV

Accepted by ACM Computing Surveys (CSUR) (DOI:  https://doi.org/10.1145/3702638)

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2303.14133v2) [paper-pdf](http://arxiv.org/pdf/2303.14133v2)

**Authors**: Junhao Dong, Junxi Chen, Xiaohua Xie, Jianhuang Lai, Hao Chen

**Abstract**: Deep learning techniques have achieved superior performance in computer-aided medical image analysis, yet they are still vulnerable to imperceptible adversarial attacks, resulting in potential misdiagnosis in clinical practice. Oppositely, recent years have also witnessed remarkable progress in defense against these tailored adversarial examples in deep medical diagnosis systems. In this exposition, we present a comprehensive survey on recent advances in adversarial attacks and defenses for medical image analysis with a systematic taxonomy in terms of the application scenario. We also provide a unified framework for different types of adversarial attack and defense methods in the context of medical image analysis. For a fair comparison, we establish a new benchmark for adversarially robust medical diagnosis models obtained by adversarial training under various scenarios. To the best of our knowledge, this is the first survey paper that provides a thorough evaluation of adversarially robust medical diagnosis models. By analyzing qualitative and quantitative results, we conclude this survey with a detailed discussion of current challenges for adversarial attack and defense in medical image analysis systems to shed light on future research directions. Code is available on \href{https://github.com/tomvii/Adv_MIA}{\color{red}{GitHub}}.



## **41. A General Recipe for Contractive Graph Neural Networks -- Technical Report**

cs.LG

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2411.01717v1) [paper-pdf](http://arxiv.org/pdf/2411.01717v1)

**Authors**: Maya Bechler-Speicher, Moshe Eliasof

**Abstract**: Graph Neural Networks (GNNs) have gained significant popularity for learning representations of graph-structured data due to their expressive power and scalability. However, despite their success in domains such as social network analysis, recommendation systems, and bioinformatics, GNNs often face challenges related to stability, generalization, and robustness to noise and adversarial attacks. Regularization techniques have shown promise in addressing these challenges by controlling model complexity and improving robustness. Building on recent advancements in contractive GNN architectures, this paper presents a novel method for inducing contractive behavior in any GNN through SVD regularization. By deriving a sufficient condition for contractiveness in the update step and applying constraints on network parameters, we demonstrate the impact of SVD regularization on the Lipschitz constant of GNNs. Our findings highlight the role of SVD regularization in enhancing the stability and generalization of GNNs, contributing to the development of more robust graph-based learning algorithms dynamics.



## **42. UniGuard: Towards Universal Safety Guardrails for Jailbreak Attacks on Multimodal Large Language Models**

cs.CL

14 pages

**SubmitDate**: 2024-11-03    [abs](http://arxiv.org/abs/2411.01703v1) [paper-pdf](http://arxiv.org/pdf/2411.01703v1)

**Authors**: Sejoon Oh, Yiqiao Jin, Megha Sharma, Donghyun Kim, Eric Ma, Gaurav Verma, Srijan Kumar

**Abstract**: Multimodal large language models (MLLMs) have revolutionized vision-language understanding but are vulnerable to multimodal jailbreak attacks, where adversaries meticulously craft inputs to elicit harmful or inappropriate responses. We propose UniGuard, a novel multimodal safety guardrail that jointly considers the unimodal and cross-modal harmful signals. UniGuard is trained such that the likelihood of generating harmful responses in a toxic corpus is minimized, and can be seamlessly applied to any input prompt during inference with minimal computational costs. Extensive experiments demonstrate the generalizability of UniGuard across multiple modalities and attack strategies. It demonstrates impressive generalizability across multiple state-of-the-art MLLMs, including LLaVA, Gemini Pro, GPT-4, MiniGPT-4, and InstructBLIP, thereby broadening the scope of our solution.



## **43. Building the Self-Improvement Loop: Error Detection and Correction in Goal-Oriented Semantic Communications**

cs.NI

7 pages, 8 figures, this paper has been accepted for publication in  IEEE CSCN 2024

**SubmitDate**: 2024-11-03    [abs](http://arxiv.org/abs/2411.01544v1) [paper-pdf](http://arxiv.org/pdf/2411.01544v1)

**Authors**: Peizheng Li, Xinyi Lin, Adnan Aijaz

**Abstract**: Error detection and correction are essential for ensuring robust and reliable operation in modern communication systems, particularly in complex transmission environments. However, discussions on these topics have largely been overlooked in semantic communication (SemCom), which focuses on transmitting meaning rather than symbols, leading to significant improvements in communication efficiency. Despite these advantages, semantic errors -- stemming from discrepancies between transmitted and received meanings -- present a major challenge to system reliability. This paper addresses this gap by proposing a comprehensive framework for detecting and correcting semantic errors in SemCom systems. We formally define semantic error, detection, and correction mechanisms, and identify key sources of semantic errors. To address these challenges, we develop a Gaussian process (GP)-based method for latent space monitoring to detect errors, alongside a human-in-the-loop reinforcement learning (HITL-RL) approach to optimize semantic model configurations using user feedback. Experimental results validate the effectiveness of the proposed methods in mitigating semantic errors under various conditions, including adversarial attacks, input feature changes, physical channel variations, and user preference shifts. This work lays the foundation for more reliable and adaptive SemCom systems with robust semantic error management techniques.



## **44. Privacy-Preserving Customer Churn Prediction Model in the Context of Telecommunication Industry**

cs.LG

26 pages, 14 tables, 13 figures

**SubmitDate**: 2024-11-03    [abs](http://arxiv.org/abs/2411.01447v1) [paper-pdf](http://arxiv.org/pdf/2411.01447v1)

**Authors**: Joydeb Kumar Sana, M Sohel Rahman, M Saifur Rahman

**Abstract**: Data is the main fuel of a successful machine learning model. A dataset may contain sensitive individual records e.g. personal health records, financial data, industrial information, etc. Training a model using this sensitive data has become a new privacy concern when someone uses third-party cloud computing. Trained models also suffer privacy attacks which leads to the leaking of sensitive information of the training data. This study is conducted to preserve the privacy of training data in the context of customer churn prediction modeling for the telecommunications industry (TCI). In this work, we propose a framework for privacy-preserving customer churn prediction (PPCCP) model in the cloud environment. We have proposed a novel approach which is a combination of Generative Adversarial Networks (GANs) and adaptive Weight-of-Evidence (aWOE). Synthetic data is generated from GANs, and aWOE is applied on the synthetic training dataset before feeding the data to the classification algorithms. Our experiments were carried out using eight different machine learning (ML) classifiers on three openly accessible datasets from the telecommunication sector. We then evaluated the performance using six commonly employed evaluation metrics. In addition to presenting a data privacy analysis, we also performed a statistical significance test. The training and prediction processes achieve data privacy and the prediction classifiers achieve high prediction performance (87.1\% in terms of F-Measure for GANs-aWOE based Na\"{\i}ve Bayes model). In contrast to earlier studies, our suggested approach demonstrates a prediction enhancement of up to 28.9\% and 27.9\% in terms of accuracy and F-measure, respectively.



## **45. Enhancing Cyber-Resilience in Integrated Energy System Scheduling with Demand Response Using Deep Reinforcement Learning**

eess.SY

Accepted by Applied Energy, Manuscript ID: APEN-D-24-03080

**SubmitDate**: 2024-11-03    [abs](http://arxiv.org/abs/2311.17941v2) [paper-pdf](http://arxiv.org/pdf/2311.17941v2)

**Authors**: Yang Li, Wenjie Ma, Yuanzheng Li, Sen Li, Zhe Chen, Mohammad Shahidehpor

**Abstract**: Optimally scheduling multi-energy flow is an effective method to utilize renewable energy sources (RES) and improve the stability and economy of integrated energy systems (IES). However, the stable demand-supply of IES faces challenges from uncertainties that arise from RES and loads, as well as the increasing impact of cyber-attacks with advanced information and communication technologies adoption. To address these challenges, this paper proposes an innovative model-free resilience scheduling method based on state-adversarial deep reinforcement learning (DRL) for integrated demand response (IDR)-enabled IES. The proposed method designs an IDR program to explore the interaction ability of electricity-gas-heat flexible loads. Additionally, the state-adversarial Markov decision process (SA-MDP) model characterizes the energy scheduling problem of IES under cyber-attack, incorporating cyber-attacks as adversaries directly into the scheduling process. The state-adversarial soft actor-critic (SA-SAC) algorithm is proposed to mitigate the impact of cyber-attacks on the scheduling strategy, integrating adversarial training into the learning process to against cyber-attacks. Simulation results demonstrate that our method is capable of adequately addressing the uncertainties resulting from RES and loads, mitigating the impact of cyber-attacks on the scheduling strategy, and ensuring a stable demand supply for various energy sources. Moreover, the proposed method demonstrates resilience against cyber-attacks. Compared to the original soft actor-critic (SAC) algorithm, it achieves a 10% improvement in economic performance under cyber-attack scenarios.



## **46. High-Frequency Anti-DreamBooth: Robust Defense against Personalized Image Synthesis**

cs.CV

ECCV 2024 Workshop The Dark Side of Generative AIs and Beyond. Our  code is available at https://github.com/mti-lab/HF-ADB

**SubmitDate**: 2024-11-03    [abs](http://arxiv.org/abs/2409.08167v3) [paper-pdf](http://arxiv.org/pdf/2409.08167v3)

**Authors**: Takuto Onikubo, Yusuke Matsui

**Abstract**: Recently, text-to-image generative models have been misused to create unauthorized malicious images of individuals, posing a growing social problem. Previous solutions, such as Anti-DreamBooth, add adversarial noise to images to protect them from being used as training data for malicious generation. However, we found that the adversarial noise can be removed by adversarial purification methods such as DiffPure. Therefore, we propose a new adversarial attack method that adds strong perturbation on the high-frequency areas of images to make it more robust to adversarial purification. Our experiment showed that the adversarial images retained noise even after adversarial purification, hindering malicious image generation.



## **47. AED: An black-box NLP classifier model attacker**

cs.LG

**SubmitDate**: 2024-11-03    [abs](http://arxiv.org/abs/2112.11660v4) [paper-pdf](http://arxiv.org/pdf/2112.11660v4)

**Authors**: Yueyang Liu, Yan Huang, Zhipeng Cai

**Abstract**: Deep Neural Networks (DNNs) have been successful in solving real-world tasks in domains such as connected and automated vehicles, disease, and job hiring. However, their implications are far-reaching in critical application areas. Hence, there is a growing concern regarding the potential bias and robustness of these DNN models. A transparency and robust model is always demanded in high-stakes domains where reliability and safety are enforced, such as healthcare and finance. While most studies have focused on adversarial image attack scenarios, fewer studies have investigated the robustness of DNN models in natural language processing (NLP) due to their adversarial samples are difficult to generate. To address this gap, we propose a word-level NLP classifier attack model called "AED," which stands for Attention mechanism enabled post-model Explanation with Density peaks clustering algorithm for synonyms search and substitution. AED aims to test the robustness of NLP DNN models by interpretability their weaknesses and exploring alternative ways to optimize them. By identifying vulnerabilities and providing explanations, AED can help improve the reliability and safety of DNN models in critical application areas such as healthcare and automated transportation. Our experiment results demonstrate that compared with other existing models, AED can effectively generate adversarial examples that can fool the victim model while maintaining the original meaning of the input.



## **48. What Features in Prompts Jailbreak LLMs? Investigating the Mechanisms Behind Attacks**

cs.CR

**SubmitDate**: 2024-11-02    [abs](http://arxiv.org/abs/2411.03343v1) [paper-pdf](http://arxiv.org/pdf/2411.03343v1)

**Authors**: Nathalie Maria Kirch, Severin Field, Stephen Casper

**Abstract**: While `jailbreaks' have been central to research on the safety and reliability of LLMs (large language models), the underlying mechanisms behind these attacks are not well understood. Some prior works have used linear methods to analyze jailbreak prompts or model refusal. Here, however, we compare linear and nonlinear methods to study the features in prompts that contribute to successful jailbreaks. We do this by probing for jailbreak success based only on the portions of the latent representations corresponding to prompt tokens. First, we introduce a dataset of 10,800 jailbreak attempts from 35 attack methods. We then show that different jailbreaking methods work via different nonlinear features in prompts. Specifically, we find that while probes can distinguish between successful and unsuccessful jailbreaking prompts with a high degree of accuracy, they often transfer poorly to held-out attack methods. We also show that nonlinear probes can be used to mechanistically jailbreak the LLM by guiding the design of adversarial latent perturbations. These mechanistic jailbreaks are able to jailbreak Gemma-7B-IT more reliably than 34 of the 35 techniques that it was trained on. Ultimately, our results suggest that jailbreaks cannot be thoroughly understood in terms of universal or linear prompt features alone.



## **49. Understanding and Improving Adversarial Collaborative Filtering for Robust Recommendation**

cs.IR

To appear in NeurIPS 2024

**SubmitDate**: 2024-11-02    [abs](http://arxiv.org/abs/2410.22844v2) [paper-pdf](http://arxiv.org/pdf/2410.22844v2)

**Authors**: Kaike Zhang, Qi Cao, Yunfan Wu, Fei Sun, Huawei Shen, Xueqi Cheng

**Abstract**: Adversarial Collaborative Filtering (ACF), which typically applies adversarial perturbations at user and item embeddings through adversarial training, is widely recognized as an effective strategy for enhancing the robustness of Collaborative Filtering (CF) recommender systems against poisoning attacks. Besides, numerous studies have empirically shown that ACF can also improve recommendation performance compared to traditional CF. Despite these empirical successes, the theoretical understanding of ACF's effectiveness in terms of both performance and robustness remains unclear. To bridge this gap, in this paper, we first theoretically show that ACF can achieve a lower recommendation error compared to traditional CF with the same training epochs in both clean and poisoned data contexts. Furthermore, by establishing bounds for reductions in recommendation error during ACF's optimization process, we find that applying personalized magnitudes of perturbation for different users based on their embedding scales can further improve ACF's effectiveness. Building on these theoretical understandings, we propose Personalized Magnitude Adversarial Collaborative Filtering (PamaCF). Extensive experiments demonstrate that PamaCF effectively defends against various types of poisoning attacks while significantly enhancing recommendation performance.



## **50. Plentiful Jailbreaks with String Compositions**

cs.CL

NeurIPS SoLaR Workshop 2024

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2411.01084v1) [paper-pdf](http://arxiv.org/pdf/2411.01084v1)

**Authors**: Brian R. Y. Huang

**Abstract**: Large language models (LLMs) remain vulnerable to a slew of adversarial attacks and jailbreaking methods. One common approach employed by white-hat attackers, or \textit{red-teamers}, is to process model inputs and outputs using string-level obfuscations, which can include leetspeak, rotary ciphers, Base64, ASCII, and more. Our work extends these encoding-based attacks by unifying them in a framework of invertible string transformations. With invertibility, we can devise arbitrary \textit{string compositions}, defined as sequences of transformations, that we can encode and decode end-to-end programmatically. We devise a automated best-of-n attack that samples from a combinatorially large number of string compositions. Our jailbreaks obtain competitive attack success rates on several leading frontier models when evaluated on HarmBench, highlighting that encoding-based attacks remain a persistent vulnerability even in advanced LLMs.



