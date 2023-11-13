# Latest Adversarial Attack Papers
**update at 2023-11-13 15:25:12**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Triad: Trusted Timestamps in Untrusted Environments**

cs.CR

**SubmitDate**: 2023-11-10    [abs](http://arxiv.org/abs/2311.06156v1) [paper-pdf](http://arxiv.org/pdf/2311.06156v1)

**Authors**: Gabriel P. Fernandez, Andrey Brito, Christof Fetzer

**Abstract**: We aim to provide trusted time measurement mechanisms to applications and cloud infrastructure deployed in environments that could harbor potential adversaries, including the hardware infrastructure provider. Despite Trusted Execution Environments (TEEs) providing multiple security functionalities, timestamps from the Operating System are not covered. Nevertheless, some services require time for validating permissions or ordering events. To address that need, we introduce Triad, a trusted timestamp dispatcher of time readings. The solution provides trusted timestamps enforced by mutually supportive enclave-based clock servers that create a continuous trusted timeline. We leverage enclave properties such as forced exits and CPU-based counters to mitigate attacks on the server's timestamp counters. Triad produces trusted, confidential, monotonically-increasing timestamps with bounded error and desirable, non-trivial properties. Our implementation relies on Intel SGX and SCONE, allowing transparent usage. We evaluate Triad's error and behavior in multiple dimensions.



## **2. Fight Fire with Fire: Combating Adversarial Patch Attacks using Pattern-randomized Defensive Patches**

cs.CV

**SubmitDate**: 2023-11-10    [abs](http://arxiv.org/abs/2311.06122v1) [paper-pdf](http://arxiv.org/pdf/2311.06122v1)

**Authors**: Jianan Feng, Jiachun Li, Changqing Miao, Jianjun Huang, Wei You, Wenchang Shi, Bin Liang

**Abstract**: Object detection has found extensive applications in various tasks, but it is also susceptible to adversarial patch attacks. Existing defense methods often necessitate modifications to the target model or result in unacceptable time overhead. In this paper, we adopt a counterattack approach, following the principle of "fight fire with fire," and propose a novel and general methodology for defending adversarial attacks. We utilize an active defense strategy by injecting two types of defensive patches, canary and woodpecker, into the input to proactively probe or weaken potential adversarial patches without altering the target model. Moreover, inspired by randomization techniques employed in software security, we employ randomized canary and woodpecker injection patterns to defend against defense-aware attacks. The effectiveness and practicality of the proposed method are demonstrated through comprehensive experiments. The results illustrate that canary and woodpecker achieve high performance, even when confronted with unknown attack methods, while incurring limited time overhead. Furthermore, our method also exhibits sufficient robustness against defense-aware attacks, as evidenced by adaptive attack experiments.



## **3. Blades: A Unified Benchmark Suite for Byzantine Attacks and Defenses in Federated Learning**

cs.CR

**SubmitDate**: 2023-11-10    [abs](http://arxiv.org/abs/2206.05359v4) [paper-pdf](http://arxiv.org/pdf/2206.05359v4)

**Authors**: Shenghui Li, Edith Ngai, Fanghua Ye, Li Ju, Tianru Zhang, Thiemo Voigt

**Abstract**: Federated learning (FL) facilitates distributed training across different IoT and edge devices, safeguarding the privacy of their data. The inherent distributed structure of FL introduces vulnerabilities, especially from adversarial devices aiming to skew local updates to their advantage. Despite the plethora of research focusing on Byzantine-resilient FL, the academic community has yet to establish a comprehensive benchmark suite, pivotal for impartial assessment and comparison of different techniques. This paper presents Blades, a scalable, extensible, and easily configurable benchmark suite that supports researchers and developers in efficiently implementing and validating novel strategies against baseline algorithms in Byzantine-resilient FL. Blades contains built-in implementations of representative attack and defense strategies and offers a user-friendly interface that seamlessly integrates new ideas. Using Blades, we re-evaluate representative attacks and defenses on wide-ranging experimental configurations (approximately 1,500 trials in total). Through our extensive experiments, we gained new insights into FL robustness and highlighted previously overlooked limitations due to the absence of thorough evaluations and comparisons of baselines under various attack settings.



## **4. Practical Membership Inference Attacks against Fine-tuned Large Language Models via Self-prompt Calibration**

cs.CL

**SubmitDate**: 2023-11-10    [abs](http://arxiv.org/abs/2311.06062v1) [paper-pdf](http://arxiv.org/pdf/2311.06062v1)

**Authors**: Wenjie Fu, Huandong Wang, Chen Gao, Guanghua Liu, Yong Li, Tao Jiang

**Abstract**: Membership Inference Attacks (MIA) aim to infer whether a target data record has been utilized for model training or not. Prior attempts have quantified the privacy risks of language models (LMs) via MIAs, but there is still no consensus on whether existing MIA algorithms can cause remarkable privacy leakage on practical Large Language Models (LLMs). Existing MIAs designed for LMs can be classified into two categories: reference-free and reference-based attacks. They are both based on the hypothesis that training records consistently strike a higher probability of being sampled. Nevertheless, this hypothesis heavily relies on the overfitting of target models, which will be mitigated by multiple regularization methods and the generalization of LLMs. The reference-based attack seems to achieve promising effectiveness in LLMs, which measures a more reliable membership signal by comparing the probability discrepancy between the target model and the reference model. However, the performance of reference-based attack is highly dependent on a reference dataset that closely resembles the training dataset, which is usually inaccessible in the practical scenario. Overall, existing MIAs are unable to effectively unveil privacy leakage over practical fine-tuned LLMs that are overfitting-free and private. We propose a Membership Inference Attack based on Self-calibrated Probabilistic Variation (SPV-MIA). Specifically, since memorization in LLMs is inevitable during the training process and occurs before overfitting, we introduce a more reliable membership signal, probabilistic variation, which is based on memorization rather than overfitting. Furthermore, we introduce a self-prompt approach, which constructs the dataset to fine-tune the reference model by prompting the target LLM itself. In this manner, the adversary can collect a dataset with a similar distribution from public APIs.



## **5. Robust Adversarial Attacks Detection for Deep Learning based Relative Pose Estimation for Space Rendezvous**

cs.CV

**SubmitDate**: 2023-11-10    [abs](http://arxiv.org/abs/2311.05992v1) [paper-pdf](http://arxiv.org/pdf/2311.05992v1)

**Authors**: Ziwei Wang, Nabil Aouf, Jose Pizarro, Christophe Honvault

**Abstract**: Research on developing deep learning techniques for autonomous spacecraft relative navigation challenges is continuously growing in recent years. Adopting those techniques offers enhanced performance. However, such approaches also introduce heightened apprehensions regarding the trustability and security of such deep learning methods through their susceptibility to adversarial attacks. In this work, we propose a novel approach for adversarial attack detection for deep neural network-based relative pose estimation schemes based on the explainability concept. We develop for an orbital rendezvous scenario an innovative relative pose estimation technique adopting our proposed Convolutional Neural Network (CNN), which takes an image from the chaser's onboard camera and outputs accurately the target's relative position and rotation. We perturb seamlessly the input images using adversarial attacks that are generated by the Fast Gradient Sign Method (FGSM). The adversarial attack detector is then built based on a Long Short Term Memory (LSTM) network which takes the explainability measure namely SHapley Value from the CNN-based pose estimator and flags the detection of adversarial attacks when acting. Simulation results show that the proposed adversarial attack detector achieves a detection accuracy of 99.21%. Both the deep relative pose estimator and adversarial attack detector are then tested on real data captured from our laboratory-designed setup. The experimental results from our laboratory-designed setup demonstrate that the proposed adversarial attack detector achieves an average detection accuracy of 96.29%.



## **6. Resilient and constrained consensus against adversarial attacks: A distributed MPC framework**

eess.SY

**SubmitDate**: 2023-11-10    [abs](http://arxiv.org/abs/2311.05935v1) [paper-pdf](http://arxiv.org/pdf/2311.05935v1)

**Authors**: Henglai Wei, Kunwu Zhang, Hui Zhang, Yang Shi

**Abstract**: There has been a growing interest in realizing the resilient consensus of the multi-agent system (MAS) under cyber-attacks, which aims to achieve the consensus of normal agents (i.e., agents without attacks) in a network, depending on the neighboring information. The literature has developed mean-subsequence-reduced (MSR) algorithms for the MAS with F adversarial attacks and has shown that the consensus is achieved for the normal agents when the communication network is at least (2F+1)-robust. However, such a stringent requirement on the communication network needs to be relaxed to enable more practical applications. Our objective is, for the first time, to achieve less stringent conditions on the network, while ensuring the resilient consensus for the general linear MAS subject to control input constraints. In this work, we propose a distributed resilient consensus framework, consisting of a pre-designed consensus protocol and distributed model predictive control (DMPC) optimization, which can help significantly reduce the requirement on the network robustness and effectively handle the general linear constrained MAS under adversarial attacks. By employing a novel distributed adversarial attack detection mechanism based on the history information broadcast by neighbors and a convex set (i.e., resilience set), we can evaluate the reliability of communication links. Moreover, we show that the recursive feasibility of the associated DMPC optimization problem can be guaranteed. The proposed consensus protocol features the following properties: 1) by minimizing a group of control variables, the consensus performance is optimized; 2) the resilient consensus of the general linear constrained MAS subject to F-locally adversarial attacks is achieved when the communication network is (F+1)-robust. Finally, numerical simulation results are presented to verify the theoretical results.



## **7. On the (In)security of Peer-to-Peer Decentralized Machine Learning**

cs.CR

IEEE S&P'23 (Previous title: "On the Privacy of Decentralized Machine  Learning") + Fixed error in neighbors-discovery trick

**SubmitDate**: 2023-11-10    [abs](http://arxiv.org/abs/2205.08443v3) [paper-pdf](http://arxiv.org/pdf/2205.08443v3)

**Authors**: Dario Pasquini, Mathilde Raynal, Carmela Troncoso

**Abstract**: In this work, we carry out the first, in-depth, privacy analysis of Decentralized Learning -- a collaborative machine learning framework aimed at addressing the main limitations of federated learning. We introduce a suite of novel attacks for both passive and active decentralized adversaries. We demonstrate that, contrary to what is claimed by decentralized learning proposers, decentralized learning does not offer any security advantage over federated learning. Rather, it increases the attack surface enabling any user in the system to perform privacy attacks such as gradient inversion, and even gain full control over honest users' local model. We also show that, given the state of the art in protections, privacy-preserving configurations of decentralized learning require fully connected networks, losing any practical advantage over the federated setup and therefore completely defeating the objective of the decentralized approach.



## **8. Scale-MIA: A Scalable Model Inversion Attack against Secure Federated Learning via Latent Space Reconstruction**

cs.LG

**SubmitDate**: 2023-11-10    [abs](http://arxiv.org/abs/2311.05808v1) [paper-pdf](http://arxiv.org/pdf/2311.05808v1)

**Authors**: Shanghao Shi, Ning Wang, Yang Xiao, Chaoyu Zhang, Yi Shi, Y. Thomas Hou, Wenjing Lou

**Abstract**: Federated learning is known for its capability to safeguard participants' data privacy. However, recently emerged model inversion attacks (MIAs) have shown that a malicious parameter server can reconstruct individual users' local data samples through model updates. The state-of-the-art attacks either rely on computation-intensive search-based optimization processes to recover each input batch, making scaling difficult, or they involve the malicious parameter server adding extra modules before the global model architecture, rendering the attacks too conspicuous and easily detectable.   To overcome these limitations, we propose Scale-MIA, a novel MIA capable of efficiently and accurately recovering training samples of clients from the aggregated updates, even when the system is under the protection of a robust secure aggregation protocol. Unlike existing approaches treating models as black boxes, Scale-MIA recognizes the importance of the intricate architecture and inner workings of machine learning models. It identifies the latent space as the critical layer for breaching privacy and decomposes the complex recovery task into an innovative two-step process to reduce computation complexity. The first step involves reconstructing the latent space representations (LSRs) from the aggregated model updates using a closed-form inversion mechanism, leveraging specially crafted adversarial linear layers. In the second step, the whole input batches are recovered from the LSRs by feeding them into a fine-tuned generative decoder.   We implemented Scale-MIA on multiple commonly used machine learning models and conducted comprehensive experiments across various settings. The results demonstrate that Scale-MIA achieves excellent recovery performance on different datasets, exhibiting high reconstruction rates, accuracy, and attack efficiency on a larger scale compared to state-of-the-art MIAs.



## **9. What Distributions are Robust to Indiscriminate Poisoning Attacks for Linear Learners?**

cs.LG

NeurIPS 2023 camera-ready version, 39 pages

**SubmitDate**: 2023-11-09    [abs](http://arxiv.org/abs/2307.01073v2) [paper-pdf](http://arxiv.org/pdf/2307.01073v2)

**Authors**: Fnu Suya, Xiao Zhang, Yuan Tian, David Evans

**Abstract**: We study indiscriminate poisoning for linear learners where an adversary injects a few crafted examples into the training data with the goal of forcing the induced model to incur higher test error. Inspired by the observation that linear learners on some datasets are able to resist the best known attacks even without any defenses, we further investigate whether datasets can be inherently robust to indiscriminate poisoning attacks for linear learners. For theoretical Gaussian distributions, we rigorously characterize the behavior of an optimal poisoning attack, defined as the poisoning strategy that attains the maximum risk of the induced model at a given poisoning budget. Our results prove that linear learners can indeed be robust to indiscriminate poisoning if the class-wise data distributions are well-separated with low variance and the size of the constraint set containing all permissible poisoning points is also small. These findings largely explain the drastic variation in empirical attack performance of the state-of-the-art poisoning attacks on linear learners across benchmark datasets, making an important initial step towards understanding the underlying reasons some learning tasks are vulnerable to data poisoning attacks.



## **10. Robust Retraining-free GAN Fingerprinting via Personalized Normalization**

cs.CV

**SubmitDate**: 2023-11-09    [abs](http://arxiv.org/abs/2311.05478v1) [paper-pdf](http://arxiv.org/pdf/2311.05478v1)

**Authors**: Jianwei Fei, Zhihua Xia, Benedetta Tondi, Mauro Barni

**Abstract**: In recent years, there has been significant growth in the commercial applications of generative models, licensed and distributed by model developers to users, who in turn use them to offer services. In this scenario, there is a need to track and identify the responsible user in the presence of a violation of the license agreement or any kind of malicious usage. Although there are methods enabling Generative Adversarial Networks (GANs) to include invisible watermarks in the images they produce, generating a model with a different watermark, referred to as a fingerprint, for each user is time- and resource-consuming due to the need to retrain the model to include the desired fingerprint. In this paper, we propose a retraining-free GAN fingerprinting method that allows model developers to easily generate model copies with the same functionality but different fingerprints. The generator is modified by inserting additional Personalized Normalization (PN) layers whose parameters (scaling and bias) are generated by two dedicated shallow networks (ParamGen Nets) taking the fingerprint as input. A watermark decoder is trained simultaneously to extract the fingerprint from the generated images. The proposed method can embed different fingerprints inside the GAN by just changing the input of the ParamGen Nets and performing a feedforward pass, without finetuning or retraining. The performance of the proposed method in terms of robustness against both model-level and image-level attacks is also superior to the state-of-the-art.



## **11. ABIGX: A Unified Framework for eXplainable Fault Detection and Classification**

cs.LG

**SubmitDate**: 2023-11-09    [abs](http://arxiv.org/abs/2311.05316v1) [paper-pdf](http://arxiv.org/pdf/2311.05316v1)

**Authors**: Yue Zhuo, Jinchuan Qian, Zhihuan Song, Zhiqiang Ge

**Abstract**: For explainable fault detection and classification (FDC), this paper proposes a unified framework, ABIGX (Adversarial fault reconstruction-Based Integrated Gradient eXplanation). ABIGX is derived from the essentials of previous successful fault diagnosis methods, contribution plots (CP) and reconstruction-based contribution (RBC). It is the first explanation framework that provides variable contributions for the general FDC models. The core part of ABIGX is the adversarial fault reconstruction (AFR) method, which rethinks the FR from the perspective of adversarial attack and generalizes to fault classification models with a new fault index. For fault classification, we put forward a new problem of fault class smearing, which intrinsically hinders the correct explanation. We prove that ABIGX effectively mitigates this problem and outperforms the existing gradient-based explanation methods. For fault detection, we theoretically bridge ABIGX with conventional fault diagnosis methods by proving that CP and RBC are the linear specifications of ABIGX. The experiments evaluate the explanations of FDC by quantitative metrics and intuitive illustrations, the results of which show the general superiority of ABIGX to other advanced explanation methods.



## **12. Beyond Pretrained Features: Noisy Image Modeling Provides Adversarial Defense**

cs.CV

NeurIPS 2023

**SubmitDate**: 2023-11-09    [abs](http://arxiv.org/abs/2302.01056v3) [paper-pdf](http://arxiv.org/pdf/2302.01056v3)

**Authors**: Zunzhi You, Daochang Liu, Bohyung Han, Chang Xu

**Abstract**: Recent advancements in masked image modeling (MIM) have made it a prevailing framework for self-supervised visual representation learning. The MIM pretrained models, like most deep neural network methods, remain vulnerable to adversarial attacks, limiting their practical application, and this issue has received little research attention. In this paper, we investigate how this powerful self-supervised learning paradigm can provide adversarial robustness to downstream classifiers. During the exploration, we find that noisy image modeling (NIM), a simple variant of MIM that adopts denoising as the pre-text task, reconstructs noisy images surprisingly well despite severe corruption. Motivated by this observation, we propose an adversarial defense method, referred to as De^3, by exploiting the pretrained decoder for denoising. Through De^3, NIM is able to enhance adversarial robustness beyond providing pretrained features. Furthermore, we incorporate a simple modification, sampling the noise scale hyperparameter from random distributions, and enable the defense to achieve a better and tunable trade-off between accuracy and robustness. Experimental results demonstrate that, in terms of adversarial robustness, NIM is superior to MIM thanks to its effective denoising capability. Moreover, the defense provided by NIM achieves performance on par with adversarial training while offering the extra tunability advantage. Source code and models are available at https://github.com/youzunzhi/NIM-AdvDef.



## **13. Counter-Empirical Attacking based on Adversarial Reinforcement Learning for Time-Relevant Scoring System**

cs.LG

submitted to TKDE on 08-Jun-2022, receive the 1st round decision  (major revision) on 20-Apr-2023, submitted to TKDE 2nd time on 30-May-2023,  receive the 2nd round decision (major revision) on 30-Sep-2023, submitted to  TKDE 3rd time on 15-Oct-2023, now under review for the 3rd round of reviewing

**SubmitDate**: 2023-11-09    [abs](http://arxiv.org/abs/2311.05144v1) [paper-pdf](http://arxiv.org/pdf/2311.05144v1)

**Authors**: Xiangguo Sun, Hong Cheng, Hang Dong, Bo Qiao, Si Qin, Qingwei Lin

**Abstract**: Scoring systems are commonly seen for platforms in the era of big data. From credit scoring systems in financial services to membership scores in E-commerce shopping platforms, platform managers use such systems to guide users towards the encouraged activity pattern, and manage resources more effectively and more efficiently thereby. To establish such scoring systems, several "empirical criteria" are firstly determined, followed by dedicated top-down design for each factor of the score, which usually requires enormous effort to adjust and tune the scoring function in the new application scenario. What's worse, many fresh projects usually have no ground-truth or any experience to evaluate a reasonable scoring system, making the designing even harder. To reduce the effort of manual adjustment of the scoring function in every new scoring system, we innovatively study the scoring system from the preset empirical criteria without any ground truth, and propose a novel framework to improve the system from scratch. In this paper, we propose a "counter-empirical attacking" mechanism that can generate "attacking" behavior traces and try to break the empirical rules of the scoring system. Then an adversarial "enhancer" is applied to evaluate the scoring system and find the improvement strategy. By training the adversarial learning problem, a proper scoring function can be learned to be robust to the attacking activity traces that are trying to violate the empirical criteria. Extensive experiments have been conducted on two scoring systems including a shared computing resource platform and a financial credit system. The experimental results have validated the effectiveness of our proposed framework.



## **14. Byzantine-Robust Distributed Online Learning: Taming Adversarial Participants in An Adversarial Environment**

cs.LG

**SubmitDate**: 2023-11-09    [abs](http://arxiv.org/abs/2307.07980v2) [paper-pdf](http://arxiv.org/pdf/2307.07980v2)

**Authors**: Xingrong Dong, Zhaoxian Wu, Qing Ling, Zhi Tian

**Abstract**: This paper studies distributed online learning under Byzantine attacks. The performance of an online learning algorithm is often characterized by (adversarial) regret, which evaluates the quality of one-step-ahead decision-making when an environment provides adversarial losses, and a sublinear bound is preferred. But we prove that, even with a class of state-of-the-art robust aggregation rules, in an adversarial environment and in the presence of Byzantine participants, distributed online gradient descent can only achieve a linear adversarial regret bound, which is tight. This is the inevitable consequence of Byzantine attacks, even though we can control the constant of the linear adversarial regret to a reasonable level. Interestingly, when the environment is not fully adversarial so that the losses of the honest participants are i.i.d. (independent and identically distributed), we show that sublinear stochastic regret, in contrast to the aforementioned adversarial regret, is possible. We develop a Byzantine-robust distributed online momentum algorithm to attain such a sublinear stochastic regret bound. Extensive numerical experiments corroborate our theoretical analysis.



## **15. Familiarity-Based Open-Set Recognition Under Adversarial Attacks**

cs.CV

Published in: The 2nd Workshop and Challenges for Out-of-Distribution  Generalization in Computer Vision, ICCV 2023

**SubmitDate**: 2023-11-08    [abs](http://arxiv.org/abs/2311.05006v1) [paper-pdf](http://arxiv.org/pdf/2311.05006v1)

**Authors**: Philip Enevoldsen, Christian Gundersen, Nico Lang, Serge Belongie, Christian Igel

**Abstract**: Open-set recognition (OSR), the identification of novel categories, can be a critical component when deploying classification models in real-world applications. Recent work has shown that familiarity-based scoring rules such as the Maximum Softmax Probability (MSP) or the Maximum Logit Score (MLS) are strong baselines when the closed-set accuracy is high. However, one of the potential weaknesses of familiarity-based OSR are adversarial attacks. Here, we present gradient-based adversarial attacks on familiarity scores for both types of attacks, False Familiarity and False Novelty attacks, and evaluate their effectiveness in informed and uninformed settings on TinyImageNet.



## **16. Be Careful When Evaluating Explanations Regarding Ground Truth**

cs.CV

**SubmitDate**: 2023-11-08    [abs](http://arxiv.org/abs/2311.04813v1) [paper-pdf](http://arxiv.org/pdf/2311.04813v1)

**Authors**: Hubert Baniecki, Maciej Chrabaszcz, Andreas Holzinger, Bastian Pfeifer, Anna Saranti, Przemyslaw Biecek

**Abstract**: Evaluating explanations of image classifiers regarding ground truth, e.g. segmentation masks defined by human perception, primarily evaluates the quality of the models under consideration rather than the explanation methods themselves. Driven by this observation, we propose a framework for $\textit{jointly}$ evaluating the robustness of safety-critical systems that $\textit{combine}$ a deep neural network with an explanation method. These are increasingly used in real-world applications like medical image analysis or robotics. We introduce a fine-tuning procedure to (mis)align model$\unicode{x2013}$explanation pipelines with ground truth and use it to quantify the potential discrepancy between worst and best-case scenarios of human alignment. Experiments across various model architectures and post-hoc local interpretation methods provide insights into the robustness of vision transformers and the overall vulnerability of such AI systems to potential adversarial attacks.



## **17. VLAttack: Multimodal Adversarial Attacks on Vision-Language Tasks via Pre-trained Models**

cs.CR

Accepted by NeurIPS 2023

**SubmitDate**: 2023-11-08    [abs](http://arxiv.org/abs/2310.04655v2) [paper-pdf](http://arxiv.org/pdf/2310.04655v2)

**Authors**: Ziyi Yin, Muchao Ye, Tianrong Zhang, Tianyu Du, Jinguo Zhu, Han Liu, Jinghui Chen, Ting Wang, Fenglong Ma

**Abstract**: Vision-Language (VL) pre-trained models have shown their superiority on many multimodal tasks. However, the adversarial robustness of such models has not been fully explored. Existing approaches mainly focus on exploring the adversarial robustness under the white-box setting, which is unrealistic. In this paper, we aim to investigate a new yet practical task to craft image and text perturbations using pre-trained VL models to attack black-box fine-tuned models on different downstream tasks. Towards this end, we propose VLAttack to generate adversarial samples by fusing perturbations of images and texts from both single-modal and multimodal levels. At the single-modal level, we propose a new block-wise similarity attack (BSA) strategy to learn image perturbations for disrupting universal representations. Besides, we adopt an existing text attack strategy to generate text perturbations independent of the image-modal attack. At the multimodal level, we design a novel iterative cross-search attack (ICSA) method to update adversarial image-text pairs periodically, starting with the outputs from the single-modal level. We conduct extensive experiments to attack three widely-used VL pretrained models for six tasks on eight datasets. Experimental results show that the proposed VLAttack framework achieves the highest attack success rates on all tasks compared with state-of-the-art baselines, which reveals a significant blind spot in the deployment of pre-trained VL models. Codes will be released soon.



## **18. Evading Watermark based Detection of AI-Generated Content**

cs.LG

To appear in ACM Conference on Computer and Communications Security  (CCS), 2023

**SubmitDate**: 2023-11-08    [abs](http://arxiv.org/abs/2305.03807v5) [paper-pdf](http://arxiv.org/pdf/2305.03807v5)

**Authors**: Zhengyuan Jiang, Jinghuai Zhang, Neil Zhenqiang Gong

**Abstract**: A generative AI model can generate extremely realistic-looking content, posing growing challenges to the authenticity of information. To address the challenges, watermark has been leveraged to detect AI-generated content. Specifically, a watermark is embedded into an AI-generated content before it is released. A content is detected as AI-generated if a similar watermark can be decoded from it. In this work, we perform a systematic study on the robustness of such watermark-based AI-generated content detection. We focus on AI-generated images. Our work shows that an attacker can post-process a watermarked image via adding a small, human-imperceptible perturbation to it, such that the post-processed image evades detection while maintaining its visual quality. We show the effectiveness of our attack both theoretically and empirically. Moreover, to evade detection, our adversarial post-processing method adds much smaller perturbations to AI-generated images and thus better maintain their visual quality than existing popular post-processing methods such as JPEG compression, Gaussian blur, and Brightness/Contrast. Our work shows the insufficiency of existing watermark-based detection of AI-generated content, highlighting the urgent needs of new methods. Our code is publicly available: https://github.com/zhengyuan-jiang/WEvade.



## **19. Army of Thieves: Enhancing Black-Box Model Extraction via Ensemble based sample selection**

cs.LG

10 pages, 5 figures, paper accepted to WACV 2024

**SubmitDate**: 2023-11-08    [abs](http://arxiv.org/abs/2311.04588v1) [paper-pdf](http://arxiv.org/pdf/2311.04588v1)

**Authors**: Akshit Jindal, Vikram Goyal, Saket Anand, Chetan Arora

**Abstract**: Machine Learning (ML) models become vulnerable to Model Stealing Attacks (MSA) when they are deployed as a service. In such attacks, the deployed model is queried repeatedly to build a labelled dataset. This dataset allows the attacker to train a thief model that mimics the original model. To maximize query efficiency, the attacker has to select the most informative subset of data points from the pool of available data. Existing attack strategies utilize approaches like Active Learning and Semi-Supervised learning to minimize costs. However, in the black-box setting, these approaches may select sub-optimal samples as they train only one thief model. Depending on the thief model's capacity and the data it was pretrained on, the model might even select noisy samples that harm the learning process. In this work, we explore the usage of an ensemble of deep learning models as our thief model. We call our attack Army of Thieves(AOT) as we train multiple models with varying complexities to leverage the crowd's wisdom. Based on the ensemble's collective decision, uncertain samples are selected for querying, while the most confident samples are directly included in the training data. Our approach is the first one to utilize an ensemble of thief models to perform model extraction. We outperform the base approaches of existing state-of-the-art methods by at least 3% and achieve a 21% higher adversarial sample transferability than previous work for models trained on the CIFAR-10 dataset.



## **20. Constrained Adaptive Attacks: Realistic Evaluation of Adversarial Examples and Robust Training of Deep Neural Networks for Tabular Data**

cs.LG

**SubmitDate**: 2023-11-08    [abs](http://arxiv.org/abs/2311.04503v1) [paper-pdf](http://arxiv.org/pdf/2311.04503v1)

**Authors**: Thibault Simonetto, Salah Ghamizi, Antoine Desjardins, Maxime Cordy, Yves Le Traon

**Abstract**: State-of-the-art deep learning models for tabular data have recently achieved acceptable performance to be deployed in industrial settings. However, the robustness of these models remains scarcely explored. Contrary to computer vision, there is to date no realistic protocol to properly evaluate the adversarial robustness of deep tabular models due to intrinsic properties of tabular data such as categorical features, immutability, and feature relationship constraints. To fill this gap, we propose CAA, the first efficient evasion attack for constrained tabular deep learning models. CAA is an iterative parameter-free attack that combines gradient and search attacks to generate adversarial examples under constraints. We leverage CAA to build a benchmark of deep tabular models across three popular use cases: credit scoring, phishing and botnet attacks detection. Our benchmark supports ten threat models with increasing capabilities of the attacker, and reflects real-world attack scenarios for each use case. Overall, our results demonstrate how domain knowledge, adversarial training, and attack budgets impact the robustness assessment of deep tabular models and provide security practitioners with a set of recommendations to improve the robustness of deep tabular models against various evasion attack scenarios.



## **21. SyncBleed: A Realistic Threat Model and Mitigation Strategy for Zero-Involvement Pairing and Authentication (ZIPA)**

cs.CR

**SubmitDate**: 2023-11-08    [abs](http://arxiv.org/abs/2311.04433v1) [paper-pdf](http://arxiv.org/pdf/2311.04433v1)

**Authors**: Isaac Ahlgren, Jack West, Kyuin Lee, George K. Thiruvathukal, Neil Klingensmith

**Abstract**: Zero Involvement Pairing and Authentication (ZIPA) is a promising technique for auto-provisioning large networks of Internet-of-Things (IoT) devices. Presently, these networks use password-based authentication, which is difficult to scale to more than a handful of devices. To deal with this challenge, ZIPA enabled devices autonomously extract identical authentication or encryption keys from ambient environmental signals. However, during the key negotiation process, existing ZIPA systems leak information on a public wireless channel which can allow adversaries to learn the key. We demonstrate a passive attack called SyncBleed, which uses leaked information to reconstruct keys generated by ZIPA systems. To mitigate SyncBleed, we present TREVOR, an improved key generation technique that produces nearly identical bit sequences from environmental signals without leaking information. We demonstrate that TREVOR can generate keys from a variety of environmental signal types under 4 seconds, consistently achieving a 90-95% bit agreement rate across devices within various environmental sources.



## **22. Unveiling Safety Vulnerabilities of Large Language Models**

cs.CL

To be published in GEM workshop. Conference on Empirical Methods in  Natural Language Processing (EMNLP). 2023

**SubmitDate**: 2023-11-07    [abs](http://arxiv.org/abs/2311.04124v1) [paper-pdf](http://arxiv.org/pdf/2311.04124v1)

**Authors**: George Kour, Marcel Zalmanovici, Naama Zwerdling, Esther Goldbraich, Ora Nova Fandina, Ateret Anaby-Tavor, Orna Raz, Eitan Farchi

**Abstract**: As large language models become more prevalent, their possible harmful or inappropriate responses are a cause for concern. This paper introduces a unique dataset containing adversarial examples in the form of questions, which we call AttaQ, designed to provoke such harmful or inappropriate responses. We assess the efficacy of our dataset by analyzing the vulnerabilities of various models when subjected to it. Additionally, we introduce a novel automatic approach for identifying and naming vulnerable semantic regions - input semantic areas for which the model is likely to produce harmful outputs. This is achieved through the application of specialized clustering techniques that consider both the semantic similarity of the input attacks and the harmfulness of the model's responses. Automatically identifying vulnerable semantic regions enhances the evaluation of model weaknesses, facilitating targeted improvements to its safety mechanisms and overall reliability.



## **23. A Lightweight and Secure PUF-Based Authentication and Key-exchange Protocol for IoT Devices**

cs.CR

**SubmitDate**: 2023-11-07    [abs](http://arxiv.org/abs/2311.04078v1) [paper-pdf](http://arxiv.org/pdf/2311.04078v1)

**Authors**: Chandranshu Gupta, Gaurav Varshney

**Abstract**: The Internet of Things (IoT) has improved people's lives by seamlessly integrating into many facets of modern life and facilitating information sharing across platforms. Device Authentication and Key exchange are major challenges for the IoT. High computational resource requirements for cryptographic primitives and message transmission during Authentication make the existing methods like PKI and IBE not suitable for these resource constrained devices. PUF appears to offer a practical and economical security mechanism in place of typically sophisticated cryptosystems like PKI and IBE. PUF provides an unclonable and tamper sensitive unique signature based on the PUF chip by using manufacturing process variability. Therefore, in this study, we use lightweight bitwise XOR, hash function, and PUF to Authenticate IoT devices. Despite several studies employing the PUF to authenticate communication between IoT devices, to the authors' knowledge, existing solutions require intermediary gateway and internet capabilities by the IoT device to directly interact with a Server for Authentication and hence, are not scalable when the IoT device works on different technologies like BLE, Zigbee, etc. To address the aforementioned issue, we present a system in which the IoT device does not require a continuous active internet connection to communicate with the server in order to Authenticate itself. The results of a thorough security study are validated against adversarial attacks and PUF modeling attacks. For formal security validation, the AVISPA verification tool is also used. Performance study recommends this protocol's lightweight characteristics. The proposed protocol's acceptability and defenses against adversarial assaults are supported by a prototype developed with ESP32.



## **24. Structural Causal Models Reveal Confounder Bias in Linear Program Modelling**

cs.LG

Published at the 15th Asian Conference on Machine Learning (ACML  2023) Journal Track. Main paper: 19 pages, References: 2 pages, Supplement:  .5 page. Main paper: 3 figures, 3 tables, Supplement: 1 table

**SubmitDate**: 2023-11-07    [abs](http://arxiv.org/abs/2105.12697v6) [paper-pdf](http://arxiv.org/pdf/2105.12697v6)

**Authors**: Matej Zečević, Devendra Singh Dhami, Kristian Kersting

**Abstract**: The recent years have been marked by extended research on adversarial attacks, especially on deep neural networks. With this work we intend on posing and investigating the question of whether the phenomenon might be more general in nature, that is, adversarial-style attacks outside classical classification tasks. Specifically, we investigate optimization problems as they constitute a fundamental part of modern AI research. To this end, we consider the base class of optimizers namely Linear Programs (LPs). On our initial attempt of a na\"ive mapping between the formalism of adversarial examples and LPs, we quickly identify the key ingredients missing for making sense of a reasonable notion of adversarial examples for LPs. Intriguingly, the formalism of Pearl's notion to causality allows for the right description of adversarial like examples for LPs. Characteristically, we show the direct influence of the Structural Causal Model (SCM) onto the subsequent LP optimization, which ultimately exposes a notion of confounding in LPs (inherited by said SCM) that allows for adversarial-style attacks. We provide both the general proof formally alongside existential proofs of such intriguing LP-parameterizations based on SCM for three combinatorial problems, namely Linear Assignment, Shortest Path and a real world problem of energy systems.



## **25. FD-MIA: Efficient Attacks on Fairness-enhanced Models**

cs.LG

Under review

**SubmitDate**: 2023-11-07    [abs](http://arxiv.org/abs/2311.03865v1) [paper-pdf](http://arxiv.org/pdf/2311.03865v1)

**Authors**: Huan Tian, Guangsheng Zhang, Bo Liu, Tianqing Zhu, Ming Ding, Wanlei Zhou

**Abstract**: Previous studies have developed fairness methods for biased models that exhibit discriminatory behaviors towards specific subgroups. While these models have shown promise in achieving fair predictions, recent research has identified their potential vulnerability to score-based membership inference attacks (MIAs). In these attacks, adversaries can infer whether a particular data sample was used during training by analyzing the model's prediction scores. However, our investigations reveal that these score-based MIAs are ineffective when targeting fairness-enhanced models in binary classifications. The attack models trained to launch the MIAs degrade into simplistic threshold models, resulting in lower attack performance. Meanwhile, we observe that fairness methods often lead to prediction performance degradation for the majority subgroups of the training data. This raises the barrier to successful attacks and widens the prediction gaps between member and non-member data. Building upon these insights, we propose an efficient MIA method against fairness-enhanced models based on fairness discrepancy results (FD-MIA). It leverages the difference in the predictions from both the original and fairness-enhanced models and exploits the observed prediction gaps as attack clues. We also explore potential strategies for mitigating privacy leakages. Extensive experiments validate our findings and demonstrate the efficacy of the proposed method.



## **26. Detecting Language Model Attacks with Perplexity**

cs.CL

**SubmitDate**: 2023-11-07    [abs](http://arxiv.org/abs/2308.14132v3) [paper-pdf](http://arxiv.org/pdf/2308.14132v3)

**Authors**: Gabriel Alon, Michael Kamfonas

**Abstract**: A novel hack involving Large Language Models (LLMs) has emerged, exploiting adversarial suffixes to deceive models into generating perilous responses. Such jailbreaks can trick LLMs into providing intricate instructions to a malicious user for creating explosives, orchestrating a bank heist, or facilitating the creation of offensive content. By evaluating the perplexity of queries with adversarial suffixes using an open-source LLM (GPT-2), we found that they have exceedingly high perplexity values. As we explored a broad range of regular (non-adversarial) prompt varieties, we concluded that false positives are a significant challenge for plain perplexity filtering. A Light-GBM trained on perplexity and token length resolved the false positives and correctly detected most adversarial attacks in the test set.



## **27. Competence-Based Analysis of Language Models**

cs.CL

**SubmitDate**: 2023-11-07    [abs](http://arxiv.org/abs/2303.00333v3) [paper-pdf](http://arxiv.org/pdf/2303.00333v3)

**Authors**: Adam Davies, Jize Jiang, ChengXiang Zhai

**Abstract**: Despite the recent success of large, pretrained neural language models (LLMs) on a variety of prompting tasks, these models can be alarmingly brittle to small changes in inputs or application contexts. To better understand such behavior and motivate the design of more robust LLMs, we provide a causal formulation of linguistic competence in the context of LLMs and propose a general framework to study and measure LLM competence. Our framework, CALM (Competence-based Analysis of Language Models), establishes the first quantitative measure of LLM competence, which we study by damaging models' internal representations of various linguistic properties in the course of performing various tasks using causal probing and evaluating models' alignment under these interventions with a given causal model. We also develop a novel approach for performing causal probing interventions using gradient-based adversarial attacks, which can target a broader range of properties and representations than existing techniques. We carry out a case study of CALM using these interventions to analyze BERT and RoBERTa's competence across a variety of lexical inference tasks, showing that the CALM framework and competence metric can be valuable tools for explaining and predicting their behavior across these tasks.



## **28. Probabilistic Categorical Adversarial Attack & Adversarial Training**

cs.LG

**SubmitDate**: 2023-11-06    [abs](http://arxiv.org/abs/2210.09364v3) [paper-pdf](http://arxiv.org/pdf/2210.09364v3)

**Authors**: Han Xu, Pengfei He, Jie Ren, Yuxuan Wan, Zitao Liu, Hui Liu, Jiliang Tang

**Abstract**: The existence of adversarial examples brings huge concern for people to apply Deep Neural Networks (DNNs) in safety-critical tasks. However, how to generate adversarial examples with categorical data is an important problem but lack of extensive exploration. Previously established methods leverage greedy search method, which can be very time-consuming to conduct successful attack. This also limits the development of adversarial training and potential defenses for categorical data. To tackle this problem, we propose Probabilistic Categorical Adversarial Attack (PCAA), which transfers the discrete optimization problem to a continuous problem that can be solved efficiently by Projected Gradient Descent. In our paper, we theoretically analyze its optimality and time complexity to demonstrate its significant advantage over current greedy based attacks. Moreover, based on our attack, we propose an efficient adversarial training framework. Through a comprehensive empirical study, we justify the effectiveness of our proposed attack and defense algorithms.



## **29. On existence, uniqueness and scalability of adversarial robustness measures for AI classifiers**

stat.ML

16 pages, 3 figures

**SubmitDate**: 2023-11-06    [abs](http://arxiv.org/abs/2310.14421v2) [paper-pdf](http://arxiv.org/pdf/2310.14421v2)

**Authors**: Illia Horenko

**Abstract**: Simply-verifiable mathematical conditions for existence, uniqueness and explicit analytical computation of minimal adversarial paths (MAP) and minimal adversarial distances (MAD) for (locally) uniquely-invertible classifiers, for generalized linear models (GLM), and for entropic AI (EAI) are formulated and proven. Practical computation of MAP and MAD, their comparison and interpretations for various classes of AI tools (for neuronal networks, boosted random forests, GLM and EAI) are demonstrated on the common synthetic benchmarks: on a double Swiss roll spiral and its extensions, as well as on the two biomedical data problems (for the health insurance claim predictions, and for the heart attack lethality classification). On biomedical applications it is demonstrated how MAP provides unique minimal patient-specific risk-mitigating interventions in the predefined subsets of accessible control variables.



## **30. Preserving Privacy in GANs Against Membership Inference Attack**

cs.LG

**SubmitDate**: 2023-11-06    [abs](http://arxiv.org/abs/2311.03172v1) [paper-pdf](http://arxiv.org/pdf/2311.03172v1)

**Authors**: Mohammadhadi Shateri, Francisco Messina, Fabrice Labeau, Pablo Piantanida

**Abstract**: Generative Adversarial Networks (GANs) have been widely used for generating synthetic data for cases where there is a limited size real-world dataset or when data holders are unwilling to share their data samples. Recent works showed that GANs, due to overfitting and memorization, might leak information regarding their training data samples. This makes GANs vulnerable to Membership Inference Attacks (MIAs). Several defense strategies have been proposed in the literature to mitigate this privacy issue. Unfortunately, defense strategies based on differential privacy are proven to reduce extensively the quality of the synthetic data points. On the other hand, more recent frameworks such as PrivGAN and PAR-GAN are not suitable for small-size training datasets. In the present work, the overfitting in GANs is studied in terms of the discriminator, and a more general measure of overfitting based on the Bhattacharyya coefficient is defined. Then, inspired by Fano's inequality, our first defense mechanism against MIAs is proposed. This framework, which requires only a simple modification in the loss function of GANs, is referred to as the maximum entropy GAN or MEGAN and significantly improves the robustness of GANs to MIAs. As a second defense strategy, a more heuristic model based on minimizing the information leaked from generated samples about the training data points is presented. This approach is referred to as mutual information minimization GAN (MIMGAN) and uses a variational representation of the mutual information to minimize the information that a synthetic sample might leak about the whole training data set. Applying the proposed frameworks to some commonly used data sets against state-of-the-art MIAs reveals that the proposed methods can reduce the accuracy of the adversaries to the level of random guessing accuracy with a small reduction in the quality of the synthetic data samples.



## **31. Quantum-Error-Mitigated Detectable Byzantine Agreement with Dynamical Decoupling for Distributed Quantum Computing**

quant-ph

**SubmitDate**: 2023-11-06    [abs](http://arxiv.org/abs/2311.03097v1) [paper-pdf](http://arxiv.org/pdf/2311.03097v1)

**Authors**: Matthew Prest, Kuan-Cheng Chen

**Abstract**: In the burgeoning domain of distributed quantum computing, achieving consensus amidst adversarial settings remains a pivotal challenge. We introduce an enhancement to the Quantum Byzantine Agreement (QBA) protocol, uniquely incorporating advanced error mitigation techniques: Twirled Readout Error Extinction (T-REx) and dynamical decoupling (DD). Central to this refined approach is the utilization of a Noisy Intermediate Scale Quantum (NISQ) source device for heightened performance. Extensive tests on both simulated and real-world quantum devices, notably IBM's quantum computer, provide compelling evidence of the effectiveness of our T-REx and DD adaptations in mitigating prevalent quantum channel errors.   Subsequent to the entanglement distribution, our protocol adopts a verification method reminiscent of Quantum Key Distribution (QKD) schemes. The Commander then issues orders encoded in specific quantum states, like Retreat or Attack. In situations where received orders diverge, lieutenants engage in structured games to reconcile discrepancies. Notably, the frequency of these games is contingent upon the Commander's strategies and the overall network size. Our empirical findings underscore the enhanced resilience and effectiveness of the protocol in diverse scenarios. Nonetheless, scalability emerges as a concern with the growth of the network size. To sum up, our research illuminates the considerable potential of fortified quantum consensus systems in the NISQ era, highlighting the imperative for sustained research in bolstering quantum ecosystems.



## **32. SoK: Memorisation in machine learning**

cs.LG

**SubmitDate**: 2023-11-06    [abs](http://arxiv.org/abs/2311.03075v1) [paper-pdf](http://arxiv.org/pdf/2311.03075v1)

**Authors**: Dmitrii Usynin, Moritz Knolle, Georgios Kaissis

**Abstract**: Quantifying the impact of individual data samples on machine learning models is an open research problem. This is particularly relevant when complex and high-dimensional relationships have to be learned from a limited sample of the data generating distribution, such as in deep learning. It was previously shown that, in these cases, models rely not only on extracting patterns which are helpful for generalisation, but also seem to be required to incorporate some of the training data more or less as is, in a process often termed memorisation. This raises the question: if some memorisation is a requirement for effective learning, what are its privacy implications? In this work we unify a broad range of previous definitions and perspectives on memorisation in ML, discuss their interplay with model generalisation and their implications of these phenomena on data privacy. Moreover, we systematise methods allowing practitioners to detect the occurrence of memorisation or quantify it and contextualise our findings in a broad range of ML learning settings. Finally, we discuss memorisation in the context of privacy attacks, differential privacy (DP) and adversarial actors.



## **33. Can LLMs Follow Simple Rules?**

cs.AI

Project website: https://eecs.berkeley.edu/~normanmu/llm_rules

**SubmitDate**: 2023-11-06    [abs](http://arxiv.org/abs/2311.04235v1) [paper-pdf](http://arxiv.org/pdf/2311.04235v1)

**Authors**: Norman Mu, Sarah Chen, Zifan Wang, Sizhe Chen, David Karamardian, Lulwa Aljeraisy, Dan Hendrycks, David Wagner

**Abstract**: As Large Language Models (LLMs) are deployed with increasing real-world responsibilities, it is important to be able to specify and constrain the behavior of these systems in a reliable manner. Model developers may wish to set explicit rules for the model, such as "do not generate abusive content", but these may be circumvented by jailbreaking techniques. Evaluating how well LLMs follow developer-provided rules in the face of adversarial inputs typically requires manual review, which slows down monitoring and methods development. To address this issue, we propose Rule-following Language Evaluation Scenarios (RuLES), a programmatic framework for measuring rule-following ability in LLMs. RuLES consists of 15 simple text scenarios in which the model is instructed to obey a set of rules in natural language while interacting with the human user. Each scenario has a concise evaluation program to determine whether the model has broken any rules in a conversation. Through manual exploration of model behavior in our scenarios, we identify 6 categories of attack strategies and collect two suites of test cases: one consisting of unique conversations from manual testing and one that systematically implements strategies from the 6 categories. Across various popular proprietary and open models such as GPT-4 and Llama 2, we find that all models are susceptible to a wide variety of adversarial hand-crafted user inputs, though GPT-4 is the best-performing model. Additionally, we evaluate open models under gradient-based attacks and find significant vulnerabilities. We propose RuLES as a challenging new setting for research into exploring and defending against both manual and automatic attacks on LLMs.



## **34. DP-DCAN: Differentially Private Deep Contrastive Autoencoder Network for Single-cell Clustering**

cs.LG

**SubmitDate**: 2023-11-06    [abs](http://arxiv.org/abs/2311.03410v1) [paper-pdf](http://arxiv.org/pdf/2311.03410v1)

**Authors**: Huifa Li, Jie Fu, Zhili Chen, Xiaomin Yang, Haitao Liu, Xinpeng Ling

**Abstract**: Single-cell RNA sequencing (scRNA-seq) is important to transcriptomic analysis of gene expression. Recently, deep learning has facilitated the analysis of high-dimensional single-cell data. Unfortunately, deep learning models may leak sensitive information about users. As a result, Differential Privacy (DP) is increasingly used to protect privacy. However, existing DP methods usually perturb whole neural networks to achieve differential privacy, and hence result in great performance overheads. To address this challenge, in this paper, we take advantage of the uniqueness of the autoencoder that it outputs only the dimension-reduced vector in the middle of the network, and design a Differentially Private Deep Contrastive Autoencoder Network (DP-DCAN) by partial network perturbation for single-cell clustering. Since only partial network is added with noise, the performance improvement is obvious and twofold: one part of network is trained with less noise due to a bigger privacy budget, and the other part is trained without any noise. Experimental results of six datasets have verified that DP-DCAN is superior to the traditional DP scheme with whole network perturbation. Moreover, DP-DCAN demonstrates strong robustness to adversarial attacks. The code is available at https://github.com/LFD-byte/DP-DCAN.



## **35. MAWSEO: Adversarial Wiki Search Poisoning for Illicit Online Promotion**

cs.CR

**SubmitDate**: 2023-11-06    [abs](http://arxiv.org/abs/2304.11300v2) [paper-pdf](http://arxiv.org/pdf/2304.11300v2)

**Authors**: Zilong Lin, Zhengyi Li, Xiaojing Liao, XiaoFeng Wang, Xiaozhong Liu

**Abstract**: As a prominent instance of vandalism edits, Wiki search poisoning for illicit promotion is a cybercrime in which the adversary aims at editing Wiki articles to promote illicit businesses through Wiki search results of relevant queries. In this paper, we report a study that, for the first time, shows that such stealthy blackhat SEO on Wiki can be automated. Our technique, called MAWSEO, employs adversarial revisions to achieve real-world cybercriminal objectives, including rank boosting, vandalism detection evasion, topic relevancy, semantic consistency, user awareness (but not alarming) of promotional content, etc. Our evaluation and user study demonstrate that MAWSEO is capable of effectively and efficiently generating adversarial vandalism edits, which can bypass state-of-the-art built-in Wiki vandalism detectors, and also get promotional content through to Wiki users without triggering their alarms. In addition, we investigated potential defense, including coherence based detection and adversarial training of vandalism detection, against our attack in the Wiki ecosystem.



## **36. CT-GAT: Cross-Task Generative Adversarial Attack based on Transferability**

cs.CL

Accepted to EMNLP 2023 main conference Corrected the header error in  Figure 3

**SubmitDate**: 2023-11-05    [abs](http://arxiv.org/abs/2310.14265v2) [paper-pdf](http://arxiv.org/pdf/2310.14265v2)

**Authors**: Minxuan Lv, Chengwei Dai, Kun Li, Wei Zhou, Songlin Hu

**Abstract**: Neural network models are vulnerable to adversarial examples, and adversarial transferability further increases the risk of adversarial attacks. Current methods based on transferability often rely on substitute models, which can be impractical and costly in real-world scenarios due to the unavailability of training data and the victim model's structural details. In this paper, we propose a novel approach that directly constructs adversarial examples by extracting transferable features across various tasks. Our key insight is that adversarial transferability can extend across different tasks. Specifically, we train a sequence-to-sequence generative model named CT-GAT using adversarial sample data collected from multiple tasks to acquire universal adversarial features and generate adversarial examples for different tasks. We conduct experiments on ten distinct datasets, and the results demonstrate that our method achieves superior attack performance with small cost.



## **37. DeepZero: Scaling up Zeroth-Order Optimization for Deep Model Training**

cs.LG

**SubmitDate**: 2023-11-05    [abs](http://arxiv.org/abs/2310.02025v2) [paper-pdf](http://arxiv.org/pdf/2310.02025v2)

**Authors**: Aochuan Chen, Yimeng Zhang, Jinghan Jia, James Diffenderfer, Jiancheng Liu, Konstantinos Parasyris, Yihua Zhang, Zheng Zhang, Bhavya Kailkhura, Sijia Liu

**Abstract**: Zeroth-order (ZO) optimization has become a popular technique for solving machine learning (ML) problems when first-order (FO) information is difficult or impossible to obtain. However, the scalability of ZO optimization remains an open problem: Its use has primarily been limited to relatively small-scale ML problems, such as sample-wise adversarial attack generation. To our best knowledge, no prior work has demonstrated the effectiveness of ZO optimization in training deep neural networks (DNNs) without a significant decrease in performance. To overcome this roadblock, we develop DeepZero, a principled ZO deep learning (DL) framework that can scale ZO optimization to DNN training from scratch through three primary innovations. First, we demonstrate the advantages of coordinate-wise gradient estimation (CGE) over randomized vector-wise gradient estimation in training accuracy and computational efficiency. Second, we propose a sparsity-induced ZO training protocol that extends the model pruning methodology using only finite differences to explore and exploit the sparse DL prior in CGE. Third, we develop the methods of feature reuse and forward parallelization to advance the practical implementations of ZO training. Our extensive experiments show that DeepZero achieves state-of-the-art (SOTA) accuracy on ResNet-20 trained on CIFAR-10, approaching FO training performance for the first time. Furthermore, we show the practical utility of DeepZero in applications of certified adversarial defense and DL-based partial differential equation error correction, achieving 10-20% improvement over SOTA. We believe our results will inspire future research on scalable ZO optimization and contribute to advancing DL with black box.



## **38. Exploring Transferability of Multimodal Adversarial Samples for Vision-Language Pre-training Models with Contrastive Learning**

cs.MM

**SubmitDate**: 2023-11-05    [abs](http://arxiv.org/abs/2308.12636v2) [paper-pdf](http://arxiv.org/pdf/2308.12636v2)

**Authors**: Youze Wang, Wenbo Hu, Yinpeng Dong, Hanwang Zhang, Richang Hong

**Abstract**: Vision-language pre-training models (VLP) are vulnerable, especially to multimodal adversarial samples, which can be crafted by adding imperceptible perturbations on both original images and texts. However, under the black-box setting, there have been no works to explore the transferability of multimodal adversarial attacks against the VLP models. In this work, we take CLIP as the surrogate model and propose a gradient-based multimodal attack method to generate transferable adversarial samples against the VLP models. By applying the gradient to optimize the adversarial images and adversarial texts simultaneously, our method can better search for and attack the vulnerable images and text information pairs. To improve the transferability of the attack, we utilize contrastive learning including image-text contrastive learning and intra-modal contrastive learning to have a more generalized understanding of the underlying data distribution and mitigate the overfitting of the surrogate model so that the generated multimodal adversarial samples have a higher transferability for VLP models. Extensive experiments validate the effectiveness of the proposed method.



## **39. Unfolding Local Growth Rate Estimates for (Almost) Perfect Adversarial Detection**

cs.CV

accepted at VISAPP23

**SubmitDate**: 2023-11-04    [abs](http://arxiv.org/abs/2212.06776v3) [paper-pdf](http://arxiv.org/pdf/2212.06776v3)

**Authors**: Peter Lorenz, Margret Keuper, Janis Keuper

**Abstract**: Convolutional neural networks (CNN) define the state-of-the-art solution on many perceptual tasks. However, current CNN approaches largely remain vulnerable against adversarial perturbations of the input that have been crafted specifically to fool the system while being quasi-imperceptible to the human eye. In recent years, various approaches have been proposed to defend CNNs against such attacks, for example by model hardening or by adding explicit defence mechanisms. Thereby, a small "detector" is included in the network and trained on the binary classification task of distinguishing genuine data from data containing adversarial perturbations. In this work, we propose a simple and light-weight detector, which leverages recent findings on the relation between networks' local intrinsic dimensionality (LID) and adversarial attacks. Based on a re-interpretation of the LID measure and several simple adaptations, we surpass the state-of-the-art on adversarial detection by a significant margin and reach almost perfect results in terms of F1-score for several networks and datasets. Sources available at: https://github.com/adverML/multiLID



## **40. MTS-DVGAN: Anomaly Detection in Cyber-Physical Systems using a Dual Variational Generative Adversarial Network**

cs.CR

27 pages, 14 figures, 8 tables. Accepted by Computers & Security

**SubmitDate**: 2023-11-04    [abs](http://arxiv.org/abs/2311.02378v1) [paper-pdf](http://arxiv.org/pdf/2311.02378v1)

**Authors**: Haili Sun, Yan Huang, Lansheng Han, Cai Fu, Hongle Liu, Xiang Long

**Abstract**: Deep generative models are promising in detecting novel cyber-physical attacks, mitigating the vulnerability of Cyber-physical systems (CPSs) without relying on labeled information. Nonetheless, these generative models face challenges in identifying attack behaviors that closely resemble normal data, or deviate from the normal data distribution but are in close proximity to the manifold of the normal cluster in latent space. To tackle this problem, this article proposes a novel unsupervised dual variational generative adversarial model named MST-DVGAN, to perform anomaly detection in multivariate time series data for CPS security. The central concept is to enhance the model's discriminative capability by widening the distinction between reconstructed abnormal samples and their normal counterparts. Specifically, we propose an augmented module by imposing contrastive constraints on the reconstruction process to obtain a more compact embedding. Then, by exploiting the distribution property and modeling the normal patterns of multivariate time series, a variational autoencoder is introduced to force the generative adversarial network (GAN) to generate diverse samples. Furthermore, two augmented loss functions are designed to extract essential characteristics in a self-supervised manner through mutual guidance between the augmented samples and original samples. Finally, a specific feature center loss is introduced for the generator network to enhance its stability. Empirical experiments are conducted on three public datasets, namely SWAT, WADI and NSL_KDD. Comparing with the state-of-the-art methods, the evaluation results show that the proposed MTS-DVGAN is more stable and can achieve consistent performance improvement.



## **41. From Trojan Horses to Castle Walls: Unveiling Bilateral Backdoor Effects in Diffusion Models**

cs.LG

10 pages, 6 figures, 7 tables

**SubmitDate**: 2023-11-04    [abs](http://arxiv.org/abs/2311.02373v1) [paper-pdf](http://arxiv.org/pdf/2311.02373v1)

**Authors**: Zhuoshi Pan, Yuguang Yao, Gaowen Liu, Bingquan Shen, H. Vicky Zhao, Ramana Rao Kompella, Sijia Liu

**Abstract**: While state-of-the-art diffusion models (DMs) excel in image generation, concerns regarding their security persist. Earlier research highlighted DMs' vulnerability to backdoor attacks, but these studies placed stricter requirements than conventional methods like 'BadNets' in image classification. This is because the former necessitates modifications to the diffusion sampling and training procedures. Unlike the prior work, we investigate whether generating backdoor attacks in DMs can be as simple as BadNets, i.e., by only contaminating the training dataset without tampering the original diffusion process. In this more realistic backdoor setting, we uncover bilateral backdoor effects that not only serve an adversarial purpose (compromising the functionality of DMs) but also offer a defensive advantage (which can be leveraged for backdoor defense). Specifically, we find that a BadNets-like backdoor attack remains effective in DMs for producing incorrect images (misaligned with the intended text conditions), and thereby yielding incorrect predictions when DMs are used as classifiers. Meanwhile, backdoored DMs exhibit an increased ratio of backdoor triggers, a phenomenon we refer to as `trigger amplification', among the generated images. We show that this latter insight can be used to enhance the detection of backdoor-poisoned training data. Even under a low backdoor poisoning ratio, studying the backdoor effects of DMs is also valuable for designing anti-backdoor image classifiers. Last but not least, we establish a meaningful linkage between backdoor attacks and the phenomenon of data replications by exploring DMs' inherent data memorization tendencies. The codes of our work are available at https://github.com/OPTML-Group/BiBadDiff.



## **42. Secure compilation of rich smart contracts on poor UTXO blockchains**

cs.CR

**SubmitDate**: 2023-11-04    [abs](http://arxiv.org/abs/2305.09545v2) [paper-pdf](http://arxiv.org/pdf/2305.09545v2)

**Authors**: Massimo Bartoletti, Riccardo Marchesin, Roberto Zunino

**Abstract**: Most blockchain platforms from Ethereum onwards render smart contracts as stateful reactive objects that update their state and transfer crypto-assets in response to transactions. A drawback of this design is that when users submit a transaction, they cannot predict in which state it will be executed. This exposes them to transaction-ordering attacks, a widespread class of attacks where adversaries with the power to construct blocks of transactions can extract value from smart contracts (the so-called MEV attacks). The UTXO model is an alternative blockchain design that thwarts these attacks by requiring new transactions to spend past ones: since transactions have unique identifiers, reordering attacks are ineffective. Currently, the blockchains following the UTXO model either provide contracts with limited expressiveness (Bitcoin), or require complex run-time environments (Cardano). We present ILLUM, an Intermediate-Level Language for the UTXO Model. ILLUM can express real-world smart contracts, e.g. those found in Decentralized Finance. We define a compiler from ILLUM to a bare-bone UTXO blockchain with loop-free scripts. Our compilation target only requires minimal extensions to Bitcoin Script: in particular, we exploit covenants, a mechanism for preserving scripts along chains of transactions. We prove the security of our compiler: namely, any attack targeting the compiled contract is also observable at the ILLUM level. Hence, the compiler does not introduce new vulnerabilities that were not already present in the source ILLUM contract. Finally, we discuss the suitability of ILLUM as a compilation target for higher-level contract languages.



## **43. Generative Adversarial Networks to infer velocity components in rotating turbulent flows**

physics.flu-dyn

**SubmitDate**: 2023-11-03    [abs](http://arxiv.org/abs/2301.07541v2) [paper-pdf](http://arxiv.org/pdf/2301.07541v2)

**Authors**: Tianyi Li, Michele Buzzicotti, Luca Biferale, Fabio Bonaccorso

**Abstract**: Inference problems for two-dimensional snapshots of rotating turbulent flows are studied. We perform a systematic quantitative benchmark of point-wise and statistical reconstruction capabilities of the linear Extended Proper Orthogonal Decomposition (EPOD) method, a non-linear Convolutional Neural Network (CNN) and a Generative Adversarial Network (GAN). We attack the important task of inferring one velocity component out of the measurement of a second one, and two cases are studied: (I) both components lay in the plane orthogonal to the rotation axis and (II) one of the two is parallel to the rotation axis. We show that EPOD method works well only for the former case where both components are strongly correlated, while CNN and GAN always outperform EPOD both concerning point-wise and statistical reconstructions. For case (II), when the input and output data are weakly correlated, all methods fail to reconstruct faithfully the point-wise information. In this case, only GAN is able to reconstruct the field in a statistical sense. The analysis is performed using both standard validation tools based on $L_2$ spatial distance between the prediction and the ground truth and more sophisticated multi-scale analysis using wavelet decomposition. Statistical validation is based on standard Jensen-Shannon divergence between the probability density functions, spectral properties and multi-scale flatness.



## **44. HINT: Healthy Influential-Noise based Training to Defend against Data Poisoning Attacks**

cs.LG

**SubmitDate**: 2023-11-03    [abs](http://arxiv.org/abs/2309.08549v2) [paper-pdf](http://arxiv.org/pdf/2309.08549v2)

**Authors**: Minh-Hao Van, Alycia N. Carey, Xintao Wu

**Abstract**: While numerous defense methods have been proposed to prohibit potential poisoning attacks from untrusted data sources, most research works only defend against specific attacks, which leaves many avenues for an adversary to exploit. In this work, we propose an efficient and robust training approach to defend against data poisoning attacks based on influence functions, named Healthy Influential-Noise based Training. Using influence functions, we craft healthy noise that helps to harden the classification model against poisoning attacks without significantly affecting the generalization ability on test data. In addition, our method can perform effectively when only a subset of the training data is modified, instead of the current method of adding noise to all examples that has been used in several previous works. We conduct comprehensive evaluations over two image datasets with state-of-the-art poisoning attacks under different realistic attack scenarios. Our empirical results show that HINT can efficiently protect deep learning models against the effect of both untargeted and targeted poisoning attacks.



## **45. Adaptive Data Analysis in a Balanced Adversarial Model**

cs.LG

Accepted to NeurIPS 2023 (Spotlight)

**SubmitDate**: 2023-11-03    [abs](http://arxiv.org/abs/2305.15452v2) [paper-pdf](http://arxiv.org/pdf/2305.15452v2)

**Authors**: Kobbi Nissim, Uri Stemmer, Eliad Tsfadia

**Abstract**: In adaptive data analysis, a mechanism gets $n$ i.i.d. samples from an unknown distribution $D$, and is required to provide accurate estimations to a sequence of adaptively chosen statistical queries with respect to $D$. Hardt and Ullman (FOCS 2014) and Steinke and Ullman (COLT 2015) showed that in general, it is computationally hard to answer more than $\Theta(n^2)$ adaptive queries, assuming the existence of one-way functions.   However, these negative results strongly rely on an adversarial model that significantly advantages the adversarial analyst over the mechanism, as the analyst, who chooses the adaptive queries, also chooses the underlying distribution $D$. This imbalance raises questions with respect to the applicability of the obtained hardness results -- an analyst who has complete knowledge of the underlying distribution $D$ would have little need, if at all, to issue statistical queries to a mechanism which only holds a finite number of samples from $D$.   We consider more restricted adversaries, called \emph{balanced}, where each such adversary consists of two separated algorithms: The \emph{sampler} who is the entity that chooses the distribution and provides the samples to the mechanism, and the \emph{analyst} who chooses the adaptive queries, but has no prior knowledge of the underlying distribution (and hence has no a priori advantage with respect to the mechanism). We improve the quality of previous lower bounds by revisiting them using an efficient \emph{balanced} adversary, under standard public-key cryptography assumptions. We show that these stronger hardness assumptions are unavoidable in the sense that any computationally bounded \emph{balanced} adversary that has the structure of all known attacks, implies the existence of public-key cryptography.



## **46. The Alignment Problem in Context**

cs.LG

**SubmitDate**: 2023-11-03    [abs](http://arxiv.org/abs/2311.02147v1) [paper-pdf](http://arxiv.org/pdf/2311.02147v1)

**Authors**: Raphaël Millière

**Abstract**: A core challenge in the development of increasingly capable AI systems is to make them safe and reliable by ensuring their behaviour is consistent with human values. This challenge, known as the alignment problem, does not merely apply to hypothetical future AI systems that may pose catastrophic risks; it already applies to current systems, such as large language models, whose potential for harm is rapidly increasing. In this paper, I assess whether we are on track to solve the alignment problem for large language models, and what that means for the safety of future AI systems. I argue that existing strategies for alignment are insufficient, because large language models remain vulnerable to adversarial attacks that can reliably elicit unsafe behaviour. I offer an explanation of this lingering vulnerability on which it is not simply a contingent limitation of current language models, but has deep technical ties to a crucial aspect of what makes these models useful and versatile in the first place -- namely, their remarkable aptitude to learn "in context" directly from user instructions. It follows that the alignment problem is not only unsolved for current AI systems, but may be intrinsically difficult to solve without severely undermining their capabilities. Furthermore, this assessment raises concerns about the prospect of ensuring the safety of future and more capable AI systems.



## **47. Bucks for Buckets (B4B): Active Defenses Against Stealing Encoders**

cs.LG

Accepted at NeurIPS2023

**SubmitDate**: 2023-11-03    [abs](http://arxiv.org/abs/2310.08571v2) [paper-pdf](http://arxiv.org/pdf/2310.08571v2)

**Authors**: Jan Dubiński, Stanisław Pawlak, Franziska Boenisch, Tomasz Trzciński, Adam Dziedzic

**Abstract**: Machine Learning as a Service (MLaaS) APIs provide ready-to-use and high-utility encoders that generate vector representations for given inputs. Since these encoders are very costly to train, they become lucrative targets for model stealing attacks during which an adversary leverages query access to the API to replicate the encoder locally at a fraction of the original training costs. We propose Bucks for Buckets (B4B), the first active defense that prevents stealing while the attack is happening without degrading representation quality for legitimate API users. Our defense relies on the observation that the representations returned to adversaries who try to steal the encoder's functionality cover a significantly larger fraction of the embedding space than representations of legitimate users who utilize the encoder to solve a particular downstream task.vB4B leverages this to adaptively adjust the utility of the returned representations according to a user's coverage of the embedding space. To prevent adaptive adversaries from eluding our defense by simply creating multiple user accounts (sybils), B4B also individually transforms each user's representations. This prevents the adversary from directly aggregating representations over multiple accounts to create their stolen encoder copy. Our active defense opens a new path towards securely sharing and democratizing encoders over public APIs.



## **48. Efficient Black-Box Adversarial Attacks on Neural Text Detectors**

cs.CL

Accepted at ICNLSP 2023

**SubmitDate**: 2023-11-03    [abs](http://arxiv.org/abs/2311.01873v1) [paper-pdf](http://arxiv.org/pdf/2311.01873v1)

**Authors**: Vitalii Fishchuk, Daniel Braun

**Abstract**: Neural text detectors are models trained to detect whether a given text was generated by a language model or written by a human. In this paper, we investigate three simple and resource-efficient strategies (parameter tweaking, prompt engineering, and character-level mutations) to alter texts generated by GPT-3.5 that are unsuspicious or unnoticeable for humans but cause misclassification by neural text detectors. The results show that especially parameter tweaking and character-level mutations are effective strategies.



## **49. Adversarial Attacks against Binary Similarity Systems**

cs.CR

**SubmitDate**: 2023-11-03    [abs](http://arxiv.org/abs/2303.11143v2) [paper-pdf](http://arxiv.org/pdf/2303.11143v2)

**Authors**: Gianluca Capozzi, Daniele Cono D'Elia, Giuseppe Antonio Di Luna, Leonardo Querzoni

**Abstract**: In recent years, binary analysis gained traction as a fundamental approach to inspect software and guarantee its security. Due to the exponential increase of devices running software, much research is now moving towards new autonomous solutions based on deep learning models, as they have been showing state-of-the-art performances in solving binary analysis problems. One of the hot topics in this context is binary similarity, which consists in determining if two functions in assembly code are compiled from the same source code. However, it is unclear how deep learning models for binary similarity behave in an adversarial context. In this paper, we study the resilience of binary similarity models against adversarial examples, showing that they are susceptible to both targeted and untargeted attacks (w.r.t. similarity goals) performed by black-box and white-box attackers. In more detail, we extensively test three current state-of-the-art solutions for binary similarity against two black-box greedy attacks, including a new technique that we call Spatial Greedy, and one white-box attack in which we repurpose a gradient-guided strategy used in attacks to image classifiers.



## **50. Adversarial Attacks on Cooperative Multi-agent Bandits**

cs.LG

**SubmitDate**: 2023-11-03    [abs](http://arxiv.org/abs/2311.01698v1) [paper-pdf](http://arxiv.org/pdf/2311.01698v1)

**Authors**: Jinhang Zuo, Zhiyao Zhang, Xuchuang Wang, Cheng Chen, Shuai Li, John C. S. Lui, Mohammad Hajiesmaili, Adam Wierman

**Abstract**: Cooperative multi-agent multi-armed bandits (CMA2B) consider the collaborative efforts of multiple agents in a shared multi-armed bandit game. We study latent vulnerabilities exposed by this collaboration and consider adversarial attacks on a few agents with the goal of influencing the decisions of the rest. More specifically, we study adversarial attacks on CMA2B in both homogeneous settings, where agents operate with the same arm set, and heterogeneous settings, where agents have distinct arm sets. In the homogeneous setting, we propose attack strategies that, by targeting just one agent, convince all agents to select a particular target arm $T-o(T)$ times while incurring $o(T)$ attack costs in $T$ rounds. In the heterogeneous setting, we prove that a target arm attack requires linear attack costs and propose attack strategies that can force a maximum number of agents to suffer linear regrets while incurring sublinear costs and only manipulating the observations of a few target agents. Numerical experiments validate the effectiveness of our proposed attack strategies.



