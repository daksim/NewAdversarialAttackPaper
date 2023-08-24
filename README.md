# Latest Adversarial Attack Papers
**update at 2023-08-24 13:07:38**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. On-Manifold Projected Gradient Descent**

cs.LG

**SubmitDate**: 2023-08-23    [abs](http://arxiv.org/abs/2308.12279v1) [paper-pdf](http://arxiv.org/pdf/2308.12279v1)

**Authors**: Aaron Mahler, Tyrus Berry, Tom Stephens, Harbir Antil, Michael Merritt, Jeanie Schreiber, Ioannis Kevrekidis

**Abstract**: This work provides a computable, direct, and mathematically rigorous approximation to the differential geometry of class manifolds for high-dimensional data, along with nonlinear projections from input space onto these class manifolds. The tools are applied to the setting of neural network image classifiers, where we generate novel, on-manifold data samples, and implement a projected gradient descent algorithm for on-manifold adversarial training. The susceptibility of neural networks (NNs) to adversarial attack highlights the brittle nature of NN decision boundaries in input space. Introducing adversarial examples during training has been shown to reduce the susceptibility of NNs to adversarial attack; however, it has also been shown to reduce the accuracy of the classifier if the examples are not valid examples for that class. Realistic "on-manifold" examples have been previously generated from class manifolds in the latent of an autoencoder. Our work explores these phenomena in a geometric and computational setting that is much closer to the raw, high-dimensional input space than can be provided by VAE or other black box dimensionality reductions. We employ conformally invariant diffusion maps (CIDM) to approximate class manifolds in diffusion coordinates, and develop the Nystr\"{o}m projection to project novel points onto class manifolds in this setting. On top of the manifold approximation, we leverage the spectral exterior calculus (SEC) to determine geometric quantities such as tangent vectors of the manifold. We use these tools to obtain adversarial examples that reside on a class manifold, yet fool a classifier. These misclassifications then become explainable in terms of human-understandable manipulations within the data, by expressing the on-manifold adversary in the semantic basis on the manifold.



## **2. Sample Complexity of Robust Learning against Evasion Attacks**

cs.LG

DPhil (PhD) Thesis - University of Oxford

**SubmitDate**: 2023-08-23    [abs](http://arxiv.org/abs/2308.12054v1) [paper-pdf](http://arxiv.org/pdf/2308.12054v1)

**Authors**: Pascale Gourdeau

**Abstract**: It is becoming increasingly important to understand the vulnerability of machine learning models to adversarial attacks. One of the fundamental problems in adversarial machine learning is to quantify how much training data is needed in the presence of evasion attacks, where data is corrupted at test time. In this thesis, we work with the exact-in-the-ball notion of robustness and study the feasibility of adversarially robust learning from the perspective of learning theory, considering sample complexity.   We first explore the setting where the learner has access to random examples only, and show that distributional assumptions are essential. We then focus on learning problems with distributions on the input data that satisfy a Lipschitz condition and show that robustly learning monotone conjunctions has sample complexity at least exponential in the adversary's budget (the maximum number of bits it can perturb on each input). However, if the adversary is restricted to perturbing $O(\log n)$ bits, then one can robustly learn conjunctions and decision lists w.r.t. log-Lipschitz distributions.   We then study learning models where the learner is given more power. We first consider local membership queries, where the learner can query the label of points near the training sample. We show that, under the uniform distribution, the exponential dependence on the adversary's budget to robustly learn conjunctions remains inevitable. We then introduce a local equivalence query oracle, which returns whether the hypothesis and target concept agree in a given region around a point in the training sample, and a counterexample if it exists. We show that if the query radius is equal to the adversary's budget, we can develop robust empirical risk minimization algorithms in the distribution-free setting. We give general query complexity upper and lower bounds, as well as for concrete concept classes.



## **3. Phase-shifted Adversarial Training**

cs.LG

Conference on Uncertainty in Artificial Intelligence, 2023 (UAI 2023)

**SubmitDate**: 2023-08-23    [abs](http://arxiv.org/abs/2301.04785v3) [paper-pdf](http://arxiv.org/pdf/2301.04785v3)

**Authors**: Yeachan Kim, Seongyeon Kim, Ihyeok Seo, Bonggun Shin

**Abstract**: Adversarial training has been considered an imperative component for safely deploying neural network-based applications to the real world. To achieve stronger robustness, existing methods primarily focus on how to generate strong attacks by increasing the number of update steps, regularizing the models with the smoothed loss function, and injecting the randomness into the attack. Instead, we analyze the behavior of adversarial training through the lens of response frequency. We empirically discover that adversarial training causes neural networks to have low convergence to high-frequency information, resulting in highly oscillated predictions near each data. To learn high-frequency contents efficiently and effectively, we first prove that a universal phenomenon of frequency principle, i.e., \textit{lower frequencies are learned first}, still holds in adversarial training. Based on that, we propose phase-shifted adversarial training (PhaseAT) in which the model learns high-frequency components by shifting these frequencies to the low-frequency range where the fast convergence occurs. For evaluations, we conduct the experiments on CIFAR-10 and ImageNet with the adaptive attack carefully designed for reliable evaluation. Comprehensive results show that PhaseAT significantly improves the convergence for high-frequency information. This results in improved adversarial robustness by enabling the model to have smoothed predictions near each data.



## **4. Designing an attack-defense game: how to increase robustness of financial transaction models via a competition**

cs.LG

**SubmitDate**: 2023-08-23    [abs](http://arxiv.org/abs/2308.11406v2) [paper-pdf](http://arxiv.org/pdf/2308.11406v2)

**Authors**: Alexey Zaytsev, Alex Natekin, Evgeni Vorsin, Valerii Smirnov, Georgii Smirnov, Oleg Sidorshin, Alexander Senin, Alexander Dudin, Dmitry Berestnev

**Abstract**: Given the escalating risks of malicious attacks in the finance sector and the consequential severe damage, a thorough understanding of adversarial strategies and robust defense mechanisms for machine learning models is critical. The threat becomes even more severe with the increased adoption in banks more accurate, but potentially fragile neural networks. We aim to investigate the current state and dynamics of adversarial attacks and defenses for neural network models that use sequential financial data as the input.   To achieve this goal, we have designed a competition that allows realistic and detailed investigation of problems in modern financial transaction data. The participants compete directly against each other, so possible attacks and defenses are examined in close-to-real-life conditions. Our main contributions are the analysis of the competition dynamics that answers the questions on how important it is to conceal a model from malicious users, how long does it take to break it, and what techniques one should use to make it more robust, and introduction additional way to attack models or increase their robustness.   Our analysis continues with a meta-study on the used approaches with their power, numerical experiments, and accompanied ablations studies. We show that the developed attacks and defenses outperform existing alternatives from the literature while being practical in terms of execution, proving the validity of the competition as a tool for uncovering vulnerabilities of machine learning models and mitigating them in various domains.



## **5. Does Physical Adversarial Example Really Matter to Autonomous Driving? Towards System-Level Effect of Adversarial Object Evasion Attack**

cs.CR

Accepted by ICCV 2023

**SubmitDate**: 2023-08-23    [abs](http://arxiv.org/abs/2308.11894v1) [paper-pdf](http://arxiv.org/pdf/2308.11894v1)

**Authors**: Ningfei Wang, Yunpeng Luo, Takami Sato, Kaidi Xu, Qi Alfred Chen

**Abstract**: In autonomous driving (AD), accurate perception is indispensable to achieving safe and secure driving. Due to its safety-criticality, the security of AD perception has been widely studied. Among different attacks on AD perception, the physical adversarial object evasion attacks are especially severe. However, we find that all existing literature only evaluates their attack effect at the targeted AI component level but not at the system level, i.e., with the entire system semantics and context such as the full AD pipeline. Thereby, this raises a critical research question: can these existing researches effectively achieve system-level attack effects (e.g., traffic rule violations) in the real-world AD context? In this work, we conduct the first measurement study on whether and how effectively the existing designs can lead to system-level effects, especially for the STOP sign-evasion attacks due to their popularity and severity. Our evaluation results show that all the representative prior works cannot achieve any system-level effects. We observe two design limitations in the prior works: 1) physical model-inconsistent object size distribution in pixel sampling and 2) lack of vehicle plant model and AD system model consideration. Then, we propose SysAdv, a novel system-driven attack design in the AD context and our evaluation results show that the system-level effects can be significantly improved, i.e., the violation rate increases by around 70%.



## **6. Adversarial Training Using Feedback Loops**

cs.LG

**SubmitDate**: 2023-08-23    [abs](http://arxiv.org/abs/2308.11881v1) [paper-pdf](http://arxiv.org/pdf/2308.11881v1)

**Authors**: Ali Haisam Muhammad Rafid, Adrian Sandu

**Abstract**: Deep neural networks (DNN) have found wide applicability in numerous fields due to their ability to accurately learn very complex input-output relations. Despite their accuracy and extensive use, DNNs are highly susceptible to adversarial attacks due to limited generalizability. For future progress in the field, it is essential to build DNNs that are robust to any kind of perturbations to the data points. In the past, many techniques have been proposed to robustify DNNs using first-order derivative information of the network.   This paper proposes a new robustification approach based on control theory. A neural network architecture that incorporates feedback control, named Feedback Neural Networks, is proposed. The controller is itself a neural network, which is trained using regular and adversarial data such as to stabilize the system outputs. The novel adversarial training approach based on the feedback control architecture is called Feedback Looped Adversarial Training (FLAT). Numerical results on standard test problems empirically show that our FLAT method is more effective than the state-of-the-art to guard against adversarial attacks.



## **7. Measuring Equality in Machine Learning Security Defenses: A Case Study in Speech Recognition**

cs.LG

Accepted to AISec'23

**SubmitDate**: 2023-08-23    [abs](http://arxiv.org/abs/2302.08973v6) [paper-pdf](http://arxiv.org/pdf/2302.08973v6)

**Authors**: Luke E. Richards, Edward Raff, Cynthia Matuszek

**Abstract**: Over the past decade, the machine learning security community has developed a myriad of defenses for evasion attacks. An understudied question in that community is: for whom do these defenses defend? This work considers common approaches to defending learned systems and how security defenses result in performance inequities across different sub-populations. We outline appropriate parity metrics for analysis and begin to answer this question through empirical results of the fairness implications of machine learning security methods. We find that many methods that have been proposed can cause direct harm, like false rejection and unequal benefits from robustness training. The framework we propose for measuring defense equality can be applied to robustly trained models, preprocessing-based defenses, and rejection methods. We identify a set of datasets with a user-centered application and a reasonable computational cost suitable for case studies in measuring the equality of defenses. In our case study of speech command recognition, we show how such adversarial training and augmentation have non-equal but complex protections for social subgroups across gender, accent, and age in relation to user coverage. We present a comparison of equality between two rejection-based defenses: randomized smoothing and neural rejection, finding randomized smoothing more equitable due to the sampling mechanism for minority groups. This represents the first work examining the disparity in the adversarial robustness in the speech domain and the fairness evaluation of rejection-based defenses.



## **8. SEA: Shareable and Explainable Attribution for Query-based Black-box Attacks**

cs.LG

**SubmitDate**: 2023-08-23    [abs](http://arxiv.org/abs/2308.11845v1) [paper-pdf](http://arxiv.org/pdf/2308.11845v1)

**Authors**: Yue Gao, Ilia Shumailov, Kassem Fawaz

**Abstract**: Machine Learning (ML) systems are vulnerable to adversarial examples, particularly those from query-based black-box attacks. Despite various efforts to detect and prevent such attacks, there is a need for a more comprehensive approach to logging, analyzing, and sharing evidence of attacks. While classic security benefits from well-established forensics and intelligence sharing, Machine Learning is yet to find a way to profile its attackers and share information about them. In response, this paper introduces SEA, a novel ML security system to characterize black-box attacks on ML systems for forensic purposes and to facilitate human-explainable intelligence sharing. SEA leverages the Hidden Markov Models framework to attribute the observed query sequence to known attacks. It thus understands the attack's progression rather than just focusing on the final adversarial examples. Our evaluations reveal that SEA is effective at attack attribution, even on their second occurrence, and is robust to adaptive strategies designed to evade forensics analysis. Interestingly, SEA's explanations of the attack behavior allow us even to fingerprint specific minor implementation bugs in attack libraries. For example, we discover that the SignOPT and Square attacks implementation in ART v1.14 sends over 50% specific zero difference queries. We thoroughly evaluate SEA on a variety of settings and demonstrate that it can recognize the same attack's second occurrence with 90+% Top-1 and 95+% Top-3 accuracy.



## **9. Ceci n'est pas une pomme: Adversarial Illusions in Multi-Modal Embeddings**

cs.CR

**SubmitDate**: 2023-08-22    [abs](http://arxiv.org/abs/2308.11804v1) [paper-pdf](http://arxiv.org/pdf/2308.11804v1)

**Authors**: Eugene Bagdasaryan, Vitaly Shmatikov

**Abstract**: Multi-modal encoders map images, sounds, texts, videos, etc. into a single embedding space, aligning representations across modalities (e.g., associate an image of a dog with a barking sound). We show that multi-modal embeddings can be vulnerable to an attack we call "adversarial illusions." Given an input in any modality, an adversary can perturb it so as to make its embedding close to that of an arbitrary, adversary-chosen input in another modality. Illusions thus enable the adversary to align any image with any text, any text with any sound, etc.   Adversarial illusions exploit proximity in the embedding space and are thus agnostic to downstream tasks. Using ImageBind embeddings, we demonstrate how adversarially aligned inputs, generated without knowledge of specific downstream tasks, mislead image generation, text generation, and zero-shot classification.



## **10. Multi-Instance Adversarial Attack on GNN-Based Malicious Domain Detection**

cs.CR

To Appear in the 45th IEEE Symposium on Security and Privacy (IEEE  S\&P 2024), May 20-23, 2024

**SubmitDate**: 2023-08-22    [abs](http://arxiv.org/abs/2308.11754v1) [paper-pdf](http://arxiv.org/pdf/2308.11754v1)

**Authors**: Mahmoud Nazzal, Issa Khalil, Abdallah Khreishah, NhatHai Phan, Yao Ma

**Abstract**: Malicious domain detection (MDD) is an open security challenge that aims to detect if an Internet domain is associated with cyber-attacks. Among many approaches to this problem, graph neural networks (GNNs) are deemed highly effective. GNN-based MDD uses DNS logs to represent Internet domains as nodes in a maliciousness graph (DMG) and trains a GNN to infer their maliciousness by leveraging identified malicious domains. Since this method relies on accessible DNS logs to construct DMGs, it exposes a vulnerability for adversaries to manipulate their domain nodes' features and connections within DMGs. Existing research mainly concentrates on threat models that manipulate individual attacker nodes. However, adversaries commonly generate multiple domains to achieve their goals economically and avoid detection. Their objective is to evade discovery across as many domains as feasible. In this work, we call the attack that manipulates several nodes in the DMG concurrently a multi-instance evasion attack. We present theoretical and empirical evidence that the existing single-instance evasion techniques for are inadequate to launch multi-instance evasion attacks against GNN-based MDDs. Therefore, we introduce MintA, an inference-time multi-instance adversarial attack on GNN-based MDDs. MintA enhances node and neighborhood evasiveness through optimized perturbations and operates successfully with only black-box access to the target model, eliminating the need for knowledge about the model's specifics or non-adversary nodes. We formulate an optimization challenge for MintA, achieving an approximate solution. Evaluating MintA on a leading GNN-based MDD technique with real-world data showcases an attack success rate exceeding 80%. These findings act as a warning for security experts, underscoring GNN-based MDDs' susceptibility to practical attacks that can undermine their effectiveness and benefits.



## **11. Security Analysis of the Consumer Remote SIM Provisioning Protocol**

cs.CR

35 pages, 9 figures, Associated ProVerif model files located at  https://github.com/peltona/rsp_model

**SubmitDate**: 2023-08-22    [abs](http://arxiv.org/abs/2211.15323v2) [paper-pdf](http://arxiv.org/pdf/2211.15323v2)

**Authors**: Abu Shohel Ahmed, Aleksi Peltonen, Mohit Sethi, Tuomas Aura

**Abstract**: Remote SIM provisioning (RSP) for consumer devices is the protocol specified by the GSM Association for downloading SIM profiles into a secure element in a mobile device. The process is commonly known as eSIM, and it is expected to replace removable SIM cards. The security of the protocol is critical because the profile includes the credentials with which the mobile device will authenticate to the mobile network. In this paper, we present a formal security analysis of the consumer RSP protocol. We model the multi-party protocol in applied pi calculus, define formal security goals, and verify them in ProVerif. The analysis shows that the consumer RSP protocol protects against a network adversary when all the intended participants are honest. However, we also model the protocol in realistic partial compromise scenarios where the adversary controls a legitimate participant or communication channel. The security failures in the partial compromise scenarios reveal weaknesses in the protocol design. The most important observation is that the security of RSP depends unnecessarily on it being encapsulated in a TLS tunnel. Also, the lack of pre-established identifiers means that a compromised download server anywhere in the world or a compromised secure element can be used for attacks against RSP between honest participants. Additionally, the lack of reliable methods for verifying user intent can lead to serious security failures. Based on the findings, we recommend practical improvements to RSP implementations, future versions of the specification, and mobile operator processes to increase the robustness of eSIM security.



## **12. Evading Watermark based Detection of AI-Generated Content**

cs.LG

To appear in ACM Conference on Computer and Communications Security  (CCS), 2023

**SubmitDate**: 2023-08-22    [abs](http://arxiv.org/abs/2305.03807v3) [paper-pdf](http://arxiv.org/pdf/2305.03807v3)

**Authors**: Zhengyuan Jiang, Jinghuai Zhang, Neil Zhenqiang Gong

**Abstract**: A generative AI model can generate extremely realistic-looking content, posing growing challenges to the authenticity of information. To address the challenges, watermark has been leveraged to detect AI-generated content. Specifically, a watermark is embedded into an AI-generated content before it is released. A content is detected as AI-generated if a similar watermark can be decoded from it. In this work, we perform a systematic study on the robustness of such watermark-based AI-generated content detection. We focus on AI-generated images. Our work shows that an attacker can post-process a watermarked image via adding a small, human-imperceptible perturbation to it, such that the post-processed image evades detection while maintaining its visual quality. We show the effectiveness of our attack both theoretically and empirically. Moreover, to evade detection, our adversarial post-processing method adds much smaller perturbations to AI-generated images and thus better maintain their visual quality than existing popular post-processing methods such as JPEG compression, Gaussian blur, and Brightness/Contrast. Our work shows the insufficiency of existing watermark-based detection of AI-generated content, highlighting the urgent needs of new methods. Our code is publicly available: https://github.com/zhengyuan-jiang/WEvade.



## **13. Robustness of SAM: Segment Anything Under Corruptions and Beyond**

cs.CV

The first work evaluates the robustness of SAM under various  corruptions such as style transfer, local occlusion, and adversarial patch  attack

**SubmitDate**: 2023-08-22    [abs](http://arxiv.org/abs/2306.07713v2) [paper-pdf](http://arxiv.org/pdf/2306.07713v2)

**Authors**: Yu Qiao, Chaoning Zhang, Taegoo Kang, Donghun Kim, Shehbaz Tariq, Chenshuang Zhang, Choong Seon Hong

**Abstract**: Segment anything model (SAM), as the name suggests, is claimed to be capable of cutting out any object and demonstrates impressive zero-shot transfer performance with the guidance of a prompt. However, there is currently a lack of comprehensive evaluation regarding its robustness under various corruptions. Understanding SAM's robustness across different corruption scenarios is crucial for its real-world deployment. Prior works show that SAM is biased towards texture (style) rather than shape, motivated by which we start by investigating SAM's robustness against style transfer, which is synthetic corruption. Following the interpretation of the corruption's effect as style change, we proceed to conduct a comprehensive evaluation of the SAM for its robustness against 15 types of common corruption. These corruptions mainly fall into categories such as digital, noise, weather, and blur. Within each of these corruption categories, we explore 5 severity levels to simulate real-world corruption scenarios. Beyond the corruptions, we further assess its robustness regarding local occlusion and local adversarial patch attacks in images. To the best of our knowledge, our work is the first of its kind to evaluate the robustness of SAM under style change, local occlusion, and local adversarial patch attacks. Considering that patch attacks visible to human eyes are easily detectable, we also assess SAM's robustness against adversarial perturbations that are imperceptible to human eyes. Overall, this work provides a comprehensive empirical study on SAM's robustness, evaluating its performance under various corruptions and extending the assessment to critical aspects like local occlusion, local patch attacks, and imperceptible adversarial perturbations, which yields valuable insights into SAM's practical applicability and effectiveness in addressing real-world challenges.



## **14. CrowdGuard: Federated Backdoor Detection in Federated Learning**

cs.CR

To appear in the Network and Distributed System Security (NDSS)  Symposium 2024. Phillip Rieger and Torsten Krau{\ss} contributed equally to  this contribution. 19 pages, 8 figures, 5 tables, 4 algorithms, 5 equations

**SubmitDate**: 2023-08-22    [abs](http://arxiv.org/abs/2210.07714v3) [paper-pdf](http://arxiv.org/pdf/2210.07714v3)

**Authors**: Phillip Rieger, Torsten Krauß, Markus Miettinen, Alexandra Dmitrienko, Ahmad-Reza Sadeghi

**Abstract**: Federated Learning (FL) is a promising approach enabling multiple clients to train Deep Neural Networks (DNNs) collaboratively without sharing their local training data. However, FL is susceptible to backdoor (or targeted poisoning) attacks. These attacks are initiated by malicious clients who seek to compromise the learning process by introducing specific behaviors into the learned model that can be triggered by carefully crafted inputs. Existing FL safeguards have various limitations: They are restricted to specific data distributions or reduce the global model accuracy due to excluding benign models or adding noise, are vulnerable to adaptive defense-aware adversaries, or require the server to access local models, allowing data inference attacks.   This paper presents a novel defense mechanism, CrowdGuard, that effectively mitigates backdoor attacks in FL and overcomes the deficiencies of existing techniques. It leverages clients' feedback on individual models, analyzes the behavior of neurons in hidden layers, and eliminates poisoned models through an iterative pruning scheme. CrowdGuard employs a server-located stacked clustering scheme to enhance its resilience to rogue client feedback. The evaluation results demonstrate that CrowdGuard achieves a 100% True-Positive-Rate and True-Negative-Rate across various scenarios, including IID and non-IID data distributions. Additionally, CrowdGuard withstands adaptive adversaries while preserving the original performance of protected models. To ensure confidentiality, CrowdGuard uses a secure and privacy-preserving architecture leveraging Trusted Execution Environments (TEEs) on both client and server sides.



## **15. LDP-Feat: Image Features with Local Differential Privacy**

cs.CV

11 pages, 4 figures, to be published in International Conference on  Computer Vision (ICCV) 2023

**SubmitDate**: 2023-08-22    [abs](http://arxiv.org/abs/2308.11223v1) [paper-pdf](http://arxiv.org/pdf/2308.11223v1)

**Authors**: Francesco Pittaluga, Bingbing Zhuang

**Abstract**: Modern computer vision services often require users to share raw feature descriptors with an untrusted server. This presents an inherent privacy risk, as raw descriptors may be used to recover the source images from which they were extracted. To address this issue, researchers recently proposed privatizing image features by embedding them within an affine subspace containing the original feature as well as adversarial feature samples. In this paper, we propose two novel inversion attacks to show that it is possible to (approximately) recover the original image features from these embeddings, allowing us to recover privacy-critical image content. In light of such successes and the lack of theoretical privacy guarantees afforded by existing visual privacy methods, we further propose the first method to privatize image features via local differential privacy, which, unlike prior approaches, provides a guaranteed bound for privacy leakage regardless of the strength of the attacks. In addition, our method yields strong performance in visual localization as a downstream task while enjoying the privacy guarantee.



## **16. Boosting Adversarial Transferability by Block Shuffle and Rotation**

cs.CV

**SubmitDate**: 2023-08-22    [abs](http://arxiv.org/abs/2308.10299v2) [paper-pdf](http://arxiv.org/pdf/2308.10299v2)

**Authors**: Kunyu Wang, Xuanran He, Wenxuan Wang, Xiaosen Wang

**Abstract**: Adversarial examples mislead deep neural networks with imperceptible perturbations and have brought significant threats to deep learning. An important aspect is their transferability, which refers to their ability to deceive other models, thus enabling attacks in the black-box setting. Though various methods have been proposed to boost transferability, the performance still falls short compared with white-box attacks. In this work, we observe that existing input transformation based attacks, one of the mainstream transfer-based attacks, result in different attention heatmaps on various models, which might limit the transferability. We also find that breaking the intrinsic relation of the image can disrupt the attention heatmap of the original image. Based on this finding, we propose a novel input transformation based attack called block shuffle and rotation (BSR). Specifically, BSR splits the input image into several blocks, then randomly shuffles and rotates these blocks to construct a set of new images for gradient calculation. Empirical evaluations on the ImageNet dataset demonstrate that BSR could achieve significantly better transferability than the existing input transformation based methods under single-model and ensemble-model settings. Combining BSR with the current input transformation method can further improve the transferability, which significantly outperforms the state-of-the-art methods.



## **17. Adversarial Attacks on Code Models with Discriminative Graph Patterns**

cs.SE

**SubmitDate**: 2023-08-22    [abs](http://arxiv.org/abs/2308.11161v1) [paper-pdf](http://arxiv.org/pdf/2308.11161v1)

**Authors**: Thanh-Dat Nguyen, Yang Zhou, Xuan Bach D. Le, Patanamon, Thongtanunam, David Lo

**Abstract**: Pre-trained language models of code are now widely used in various software engineering tasks such as code generation, code completion, vulnerability detection, etc. This, in turn, poses security and reliability risks to these models. One of the important threats is \textit{adversarial attacks}, which can lead to erroneous predictions and largely affect model performance on downstream tasks. Current adversarial attacks on code models usually adopt fixed sets of program transformations, such as variable renaming and dead code insertion, leading to limited attack effectiveness. To address the aforementioned challenges, we propose a novel adversarial attack framework, GraphCodeAttack, to better evaluate the robustness of code models. Given a target code model, GraphCodeAttack automatically mines important code patterns, which can influence the model's decisions, to perturb the structure of input code to the model. To do so, GraphCodeAttack uses a set of input source codes to probe the model's outputs and identifies the \textit{discriminative} ASTs patterns that can influence the model decisions. GraphCodeAttack then selects appropriate AST patterns, concretizes the selected patterns as attacks, and inserts them as dead code into the model's input program. To effectively synthesize attacks from AST patterns, GraphCodeAttack uses a separate pre-trained code model to fill in the ASTs with concrete code snippets. We evaluate the robustness of two popular code models (e.g., CodeBERT and GraphCodeBERT) against our proposed approach on three tasks: Authorship Attribution, Vulnerability Prediction, and Clone Detection. The experimental results suggest that our proposed approach significantly outperforms state-of-the-art approaches in attacking code models such as CARROT and ALERT.



## **18. Distributed Black-box Attack against Image Classification Cloud Services**

cs.LG

8 pages, 10 figures

**SubmitDate**: 2023-08-21    [abs](http://arxiv.org/abs/2210.16371v3) [paper-pdf](http://arxiv.org/pdf/2210.16371v3)

**Authors**: Han Wu, Sareh Rowlands, Johan Wahlstrom

**Abstract**: Black-box adversarial attacks can fool image classifiers into misclassifying images without requiring access to model structure and weights. Recent studies have reported attack success rates of over 95% with less than 1,000 queries. The question then arises of whether black-box attacks have become a real threat against IoT devices that rely on cloud APIs to achieve image classification. To shed some light on this, note that prior research has primarily focused on increasing the success rate and reducing the number of queries. However, another crucial factor for black-box attacks against cloud APIs is the time required to perform the attack. This paper applies black-box attacks directly to cloud APIs rather than to local models, thereby avoiding mistakes made in prior research that applied the perturbation before image encoding and pre-processing. Further, we exploit load balancing to enable distributed black-box attacks that can reduce the attack time by a factor of about five for both local search and gradient estimation methods.



## **19. A Man-in-the-Middle Attack against Object Detection Systems**

cs.RO

6 pages, 7 figures

**SubmitDate**: 2023-08-21    [abs](http://arxiv.org/abs/2208.07174v3) [paper-pdf](http://arxiv.org/pdf/2208.07174v3)

**Authors**: Han Wu, Sareh Rowlands, Johan Wahlstrom

**Abstract**: Object detection systems using deep learning models have become increasingly popular in robotics thanks to the rising power of CPUs and GPUs in embedded systems. However, these models are susceptible to adversarial attacks. While some attacks are limited by strict assumptions on access to the detection system, we propose a novel hardware attack inspired by Man-in-the-Middle attacks in cryptography. This attack generates an Universal Adversarial Perturbation (UAP) and then inject the perturbation between the USB camera and the detection system via a hardware attack. Besides, prior research is misled by an evaluation metric that measures the model accuracy rather than the attack performance. In combination with our proposed evaluation metrics, we significantly increases the strength of adversarial perturbations. These findings raise serious concerns for applications of deep learning models in safety-critical systems, such as autonomous driving.



## **20. Spear and Shield: Adversarial Attacks and Defense Methods for Model-Based Link Prediction on Continuous-Time Dynamic Graphs**

cs.LG

**SubmitDate**: 2023-08-21    [abs](http://arxiv.org/abs/2308.10779v1) [paper-pdf](http://arxiv.org/pdf/2308.10779v1)

**Authors**: Dongjin Lee, Juho Lee, Kijung Shin

**Abstract**: Real-world graphs are dynamic, constantly evolving with new interactions, such as financial transactions in financial networks. Temporal Graph Neural Networks (TGNNs) have been developed to effectively capture the evolving patterns in dynamic graphs. While these models have demonstrated their superiority, being widely adopted in various important fields, their vulnerabilities against adversarial attacks remain largely unexplored. In this paper, we propose T-SPEAR, a simple and effective adversarial attack method for link prediction on continuous-time dynamic graphs, focusing on investigating the vulnerabilities of TGNNs. Specifically, before the training procedure of a victim model, which is a TGNN for link prediction, we inject edge perturbations to the data that are unnoticeable in terms of the four constraints we propose, and yet effective enough to cause malfunction of the victim model. Moreover, we propose a robust training approach T-SHIELD to mitigate the impact of adversarial attacks. By using edge filtering and enforcing temporal smoothness to node embeddings, we enhance the robustness of the victim model. Our experimental study shows that T-SPEAR significantly degrades the victim model's performance on link prediction tasks, and even more, our attacks are transferable to other TGNNs, which differ from the victim model assumed by the attacker. Moreover, we demonstrate that T-SHIELD effectively filters out adversarial edges and exhibits robustness against adversarial attacks, surpassing the link prediction performance of the naive TGNN by up to 11.2% under T-SPEAR.



## **21. Adversarial Attacks and Defenses for Semantic Communication in Vehicular Metaverses**

cs.CR

**SubmitDate**: 2023-08-21    [abs](http://arxiv.org/abs/2306.03528v2) [paper-pdf](http://arxiv.org/pdf/2306.03528v2)

**Authors**: Jiawen Kang, Jiayi He, Hongyang Du, Zehui Xiong, Zhaohui Yang, Xumin Huang, Shengli Xie

**Abstract**: For vehicular metaverses, one of the ultimate user-centric goals is to optimize the immersive experience and Quality of Service (QoS) for users on board. Semantic Communication (SemCom) has been introduced as a revolutionary paradigm that significantly eases communication resource pressure for vehicular metaverse applications to achieve this goal. SemCom enables high-quality and ultra-efficient vehicular communication, even with explosively increasing data traffic among vehicles. In this article, we propose a hierarchical SemCom-enabled vehicular metaverses framework consisting of the global metaverse, local metaverses, SemCom module, and resource pool. The global and local metaverses are brand-new concepts from the metaverse's distribution standpoint. Considering the QoS of users, this article explores the potential security vulnerabilities of the proposed framework. To that purpose, this study highlights a specific security risk to the framework's SemCom module and offers a viable defense solution, so encouraging community researchers to focus more on vehicular metaverse security. Finally, we provide an overview of the open issues of secure SemCom in the vehicular metaverses, notably pointing out potential future research directions.



## **22. Boosting Adversarial Attack with Similar Target**

cs.CV

**SubmitDate**: 2023-08-21    [abs](http://arxiv.org/abs/2308.10743v1) [paper-pdf](http://arxiv.org/pdf/2308.10743v1)

**Authors**: Shuo Zhang, Ziruo Wang, Zikai Zhou, Huanran Chen

**Abstract**: Deep neural networks are vulnerable to adversarial examples, posing a threat to the models' applications and raising security concerns. An intriguing property of adversarial examples is their strong transferability. Several methods have been proposed to enhance transferability, including ensemble attacks which have demonstrated their efficacy. However, prior approaches simply average logits, probabilities, or losses for model ensembling, lacking a comprehensive analysis of how and why model ensembling significantly improves transferability. In this paper, we propose a similar targeted attack method named Similar Target~(ST). By promoting cosine similarity between the gradients of each model, our method regularizes the optimization direction to simultaneously attack all surrogate models. This strategy has been proven to enhance generalization ability. Experimental results on ImageNet validate the effectiveness of our approach in improving adversarial transferability. Our method outperforms state-of-the-art attackers on 18 discriminative classifiers and adversarially trained models.



## **23. SALSy: Security-Aware Layout Synthesis**

cs.CR

**SubmitDate**: 2023-08-21    [abs](http://arxiv.org/abs/2308.06201v2) [paper-pdf](http://arxiv.org/pdf/2308.06201v2)

**Authors**: Mohammad Eslami, Tiago Perez, Samuel Pagliarini

**Abstract**: Integrated Circuits (ICs) are the target of diverse attacks during their lifetime. Fabrication-time attacks, such as the insertion of Hardware Trojans, can give an adversary access to privileged data and/or the means to corrupt the IC's internal computation. Post-fabrication attacks, where the end-user takes a malicious role, also attempt to obtain privileged information through means such as fault injection and probing. Taking these threats into account and at the same time, this paper proposes a methodology for Security-Aware Layout Synthesis (SALSy), such that ICs can be designed with security in mind in the same manner as power-performance-area (PPA) metrics are considered today, a concept known as security closure. Furthermore, the trade-offs between PPA and security are considered and a chip is fabricated in a 65nm CMOS commercial technology for validation purposes - a feature not seen in previous research on security closure. Measurements on the fabricated ICs indicate that SALSy promotes a modest increase in power in order to achieve significantly improved security metrics.



## **24. On the Adversarial Robustness of Multi-Modal Foundation Models**

cs.LG

ICCV AROW 2023

**SubmitDate**: 2023-08-21    [abs](http://arxiv.org/abs/2308.10741v1) [paper-pdf](http://arxiv.org/pdf/2308.10741v1)

**Authors**: Christian Schlarmann, Matthias Hein

**Abstract**: Multi-modal foundation models combining vision and language models such as Flamingo or GPT-4 have recently gained enormous interest. Alignment of foundation models is used to prevent models from providing toxic or harmful output. While malicious users have successfully tried to jailbreak foundation models, an equally important question is if honest users could be harmed by malicious third-party content. In this paper we show that imperceivable attacks on images in order to change the caption output of a multi-modal foundation model can be used by malicious content providers to harm honest users e.g. by guiding them to malicious websites or broadcast fake information. This indicates that countermeasures to adversarial attacks should be used by any deployed multi-modal foundation model.



## **25. Model-free False Data Injection Attack in Networked Control Systems: A Feedback Optimization Approach**

math.OC

**SubmitDate**: 2023-08-21    [abs](http://arxiv.org/abs/2212.07633v2) [paper-pdf](http://arxiv.org/pdf/2212.07633v2)

**Authors**: Xiaoyu Luo, Chongrong Fang, Jianping He, Chengcheng Zhao, Dario Paccagnan

**Abstract**: Security issues have gathered growing interest within the control systems community, as physical components and communication networks are increasingly vulnerable to cyber attacks. In this context, recent literature has studied increasingly sophisticated \emph{false data injection} attacks, with the aim to design mitigative measures that improve the systems' security. Notably, data-driven attack strategies -- whereby the system dynamics is oblivious to the adversary -- have received increasing attention. However, many of the existing works on the topic rely on the implicit assumption of linear system dynamics, significantly limiting their scope. Contrary to that, in this work we design and analyze \emph{truly} model-free false data injection attack that applies to general linear and nonlinear systems. More specifically, we aim at designing an injected signal that steers the output of the system toward a (maliciously chosen) trajectory. We do so by designing a zeroth-order feedback optimization policy and jointly use probing signals for real-time measurements. We then characterize the quality of the proposed model-free attack through its optimality gap, which is affected by the dimensions of the attack signal, the number of iterations performed, and the convergence rate of the system. Finally, we extend the proposed attack scheme to the systems with internal noise. Extensive simulations show the effectiveness of the proposed attack scheme.



## **26. Measuring the Effect of Causal Disentanglement on the Adversarial Robustness of Neural Network Models**

cs.LG

12 pages, 3 figures

**SubmitDate**: 2023-08-21    [abs](http://arxiv.org/abs/2308.10708v1) [paper-pdf](http://arxiv.org/pdf/2308.10708v1)

**Authors**: Preben M. Ness, Dusica Marijan, Sunanda Bose

**Abstract**: Causal Neural Network models have shown high levels of robustness to adversarial attacks as well as an increased capacity for generalisation tasks such as few-shot learning and rare-context classification compared to traditional Neural Networks. This robustness is argued to stem from the disentanglement of causal and confounder input signals. However, no quantitative study has yet measured the level of disentanglement achieved by these types of causal models or assessed how this relates to their adversarial robustness.   Existing causal disentanglement metrics are not applicable to deterministic models trained on real-world datasets. We, therefore, utilise metrics of content/style disentanglement from the field of Computer Vision to measure different aspects of the causal disentanglement for four state-of-the-art causal Neural Network models. By re-implementing these models with a common ResNet18 architecture we are able to fairly measure their adversarial robustness on three standard image classification benchmarking datasets under seven common white-box attacks. We find a strong association (r=0.820, p=0.001) between the degree to which models decorrelate causal and confounder signals and their adversarial robustness. Additionally, we find a moderate negative association between the pixel-level information content of the confounder signal and adversarial robustness (r=-0.597, p=0.040).



## **27. Transferable Attack for Semantic Segmentation**

cs.CV

Source code is available at: https://github.com/anucvers/TASS

**SubmitDate**: 2023-08-21    [abs](http://arxiv.org/abs/2307.16572v2) [paper-pdf](http://arxiv.org/pdf/2307.16572v2)

**Authors**: Mengqi He, Jing Zhang, Zhaoyuan Yang, Mingyi He, Nick Barnes, Yuchao Dai

**Abstract**: We analysis performance of semantic segmentation models wrt. adversarial attacks, and observe that the adversarial examples generated from a source model fail to attack the target models. i.e The conventional attack methods, such as PGD and FGSM, do not transfer well to target models, making it necessary to study the transferable attacks, especially transferable attacks for semantic segmentation. We find two main factors to achieve transferable attack. Firstly, the attack should come with effective data augmentation and translation-invariant features to deal with unseen models. Secondly, stabilized optimization strategies are needed to find the optimal attack direction. Based on the above observations, we propose an ensemble attack for semantic segmentation to achieve more effective attacks with higher transferability. The source code and experimental results are publicly available via our project page: https://github.com/anucvers/TASS.



## **28. FLARE: Fingerprinting Deep Reinforcement Learning Agents using Universal Adversarial Masks**

cs.LG

Will appear in the proceedings of ACSAC 2023; 13 pages, 5 figures, 7  tables

**SubmitDate**: 2023-08-21    [abs](http://arxiv.org/abs/2307.14751v2) [paper-pdf](http://arxiv.org/pdf/2307.14751v2)

**Authors**: Buse G. A. Tekgul, N. Asokan

**Abstract**: We propose FLARE, the first fingerprinting mechanism to verify whether a suspected Deep Reinforcement Learning (DRL) policy is an illegitimate copy of another (victim) policy. We first show that it is possible to find non-transferable, universal adversarial masks, i.e., perturbations, to generate adversarial examples that can successfully transfer from a victim policy to its modified versions but not to independently trained policies. FLARE employs these masks as fingerprints to verify the true ownership of stolen DRL policies by measuring an action agreement value over states perturbed via such masks. Our empirical evaluations show that FLARE is effective (100% action agreement on stolen copies) and does not falsely accuse independent policies (no false positives). FLARE is also robust to model modification attacks and cannot be easily evaded by more informed adversaries without negatively impacting agent performance. We also show that not all universal adversarial masks are suitable candidates for fingerprints due to the inherent characteristics of DRL policies. The spatio-temporal dynamics of DRL problems and sequential decision-making process make characterizing the decision boundary of DRL policies more difficult, as well as searching for universal masks that capture the geometry of it.



## **29. Improving the Transferability of Adversarial Examples with Arbitrary Style Transfer**

cs.CV

10 pages, 2 figures, accepted by the 31st ACM International  Conference on Multimedia (MM '23)

**SubmitDate**: 2023-08-21    [abs](http://arxiv.org/abs/2308.10601v1) [paper-pdf](http://arxiv.org/pdf/2308.10601v1)

**Authors**: Zhijin Ge, Fanhua Shang, Hongying Liu, Yuanyuan Liu, Liang Wan, Wei Feng, Xiaosen Wang

**Abstract**: Deep neural networks are vulnerable to adversarial examples crafted by applying human-imperceptible perturbations on clean inputs. Although many attack methods can achieve high success rates in the white-box setting, they also exhibit weak transferability in the black-box setting. Recently, various methods have been proposed to improve adversarial transferability, in which the input transformation is one of the most effective methods. In this work, we notice that existing input transformation-based works mainly adopt the transformed data in the same domain for augmentation. Inspired by domain generalization, we aim to further improve the transferability using the data augmented from different domains. Specifically, a style transfer network can alter the distribution of low-level visual features in an image while preserving semantic content for humans. Hence, we propose a novel attack method named Style Transfer Method (STM) that utilizes a proposed arbitrary style transfer network to transform the images into different domains. To avoid inconsistent semantic information of stylized images for the classification network, we fine-tune the style transfer network and mix up the generated images added by random noise with the original images to maintain semantic consistency and boost input diversity. Extensive experimental results on the ImageNet-compatible dataset show that our proposed method can significantly improve the adversarial transferability on either normally trained models or adversarially trained models than state-of-the-art input transformation-based attacks. Code is available at: https://github.com/Zhijin-Ge/STM.



## **30. Single-User Injection for Invisible Shilling Attack against Recommender Systems**

cs.IR

CIKM 2023. 10 pages, 5 figures

**SubmitDate**: 2023-08-21    [abs](http://arxiv.org/abs/2308.10467v1) [paper-pdf](http://arxiv.org/pdf/2308.10467v1)

**Authors**: Chengzhi Huang, Hui Li

**Abstract**: Recommendation systems (RS) are crucial for alleviating the information overload problem. Due to its pivotal role in guiding users to make decisions, unscrupulous parties are lured to launch attacks against RS to affect the decisions of normal users and gain illegal profits. Among various types of attacks, shilling attack is one of the most subsistent and profitable attacks. In shilling attack, an adversarial party injects a number of well-designed fake user profiles into the system to mislead RS so that the attack goal can be achieved. Although existing shilling attack methods have achieved promising results, they all adopt the attack paradigm of multi-user injection, where some fake user profiles are required. This paper provides the first study of shilling attack in an extremely limited scenario: only one fake user profile is injected into the victim RS to launch shilling attacks (i.e., single-user injection). We propose a novel single-user injection method SUI-Attack for invisible shilling attack. SUI-Attack is a graph based attack method that models shilling attack as a node generation task over the user-item bipartite graph of the victim RS, and it constructs the fake user profile by generating user features and edges that link the fake user to items. Extensive experiments demonstrate that SUI-Attack can achieve promising attack results in single-user injection. In addition to its attack power, SUI-Attack increases the stealthiness of shilling attack and reduces the risk of being detected. We provide our implementation at: https://github.com/KDEGroup/SUI-Attack.



## **31. Adaptive Local Steps Federated Learning with Differential Privacy Driven by Convergence Analysis**

cs.LG

**SubmitDate**: 2023-08-21    [abs](http://arxiv.org/abs/2308.10457v1) [paper-pdf](http://arxiv.org/pdf/2308.10457v1)

**Authors**: Xinpeng Ling, Jie Fu, Zhili Chen

**Abstract**: Federated Learning (FL) is a distributed machine learning technique that allows model training among multiple devices or organizations without sharing data. However, while FL ensures that the raw data is not directly accessible to external adversaries, adversaries can still obtain some statistical information about the data through differential attacks. Differential Privacy (DP) has been proposed, which adds noise to the model or gradients to prevent adversaries from inferring private information from the transmitted parameters. We reconsider the framework of differential privacy federated learning in resource-constrained scenarios (privacy budget and communication resources). We analyze the convergence of federated learning with differential privacy (DPFL) on resource-constrained scenarios and propose an Adaptive Local Steps Differential Privacy Federated Learning (ALS-DPFL) algorithm. We experiment our algorithm on the FashionMNIST and Cifar-10 datasets and achieve quite good performance relative to previous work.



## **32. Quantum Query Lower Bounds for Key Recovery Attacks on the Even-Mansour Cipher**

quant-ph

**SubmitDate**: 2023-08-21    [abs](http://arxiv.org/abs/2308.10418v1) [paper-pdf](http://arxiv.org/pdf/2308.10418v1)

**Authors**: Akinori Kawachi, Yuki Naito

**Abstract**: The Even-Mansour (EM) cipher is one of the famous constructions for a block cipher. Kuwakado and Morii demonstrated that a quantum adversary can recover its $n$-bit secret keys only with $O(n)$ nonadaptive quantum queries. While the security of the EM cipher and its variants is well-understood for classical adversaries, very little is currently known of their quantum security. Towards a better understanding of the quantum security, or the limits of quantum adversaries for the EM cipher, we study the quantum query complexity for the key recovery of the EM cipher and prove every quantum algorithm requires $\Omega(n)$ quantum queries for the key recovery even if it is allowed to make adaptive queries. Therefore, the quantum attack of Kuwakado and Morii has the optimal query complexity up to a constant factor, and we cannot asymptotically improve it even with adaptive quantum queries.



## **33. HoSNN: Adversarially-Robust Homeostatic Spiking Neural Networks with Adaptive Firing Thresholds**

cs.NE

**SubmitDate**: 2023-08-20    [abs](http://arxiv.org/abs/2308.10373v1) [paper-pdf](http://arxiv.org/pdf/2308.10373v1)

**Authors**: Hejia Geng, Peng Li

**Abstract**: Spiking neural networks (SNNs) offer promise for efficient and powerful neurally inspired computation. Common to other types of neural networks, however, SNNs face the severe issue of vulnerability to adversarial attacks. We present the first study that draws inspiration from neural homeostasis to develop a bio-inspired solution that counters the susceptibilities of SNNs to adversarial onslaughts. At the heart of our approach is a novel threshold-adapting leaky integrate-and-fire (TA-LIF) neuron model, which we adopt to construct the proposed adversarially robust homeostatic SNN (HoSNN). Distinct from traditional LIF models, our TA-LIF model incorporates a self-stabilizing dynamic thresholding mechanism, curtailing adversarial noise propagation and safeguarding the robustness of HoSNNs in an unsupervised manner. Theoretical analysis is presented to shed light on the stability and convergence properties of the TA-LIF neurons, underscoring their superior dynamic robustness under input distributional shifts over traditional LIF neurons. Remarkably, without explicit adversarial training, our HoSNNs demonstrate inherent robustness on CIFAR-10, with accuracy improvements to 72.6% and 54.19% against FGSM and PGD attacks, up from 20.97% and 0.6%, respectively. Furthermore, with minimal FGSM adversarial training, our HoSNNs surpass previous models by 29.99% under FGSM and 47.83% under PGD attacks on CIFAR-10. Our findings offer a new perspective on harnessing biological principles for bolstering SNNs adversarial robustness and defense, paving the way to more resilient neuromorphic computing.



## **34. Hiding Backdoors within Event Sequence Data via Poisoning Attacks**

cs.LG

**SubmitDate**: 2023-08-20    [abs](http://arxiv.org/abs/2308.10201v1) [paper-pdf](http://arxiv.org/pdf/2308.10201v1)

**Authors**: Elizaveta Kovtun, Alina Ermilova, Dmitry Berestnev, Alexey Zaytsev

**Abstract**: The financial industry relies on deep learning models for making important decisions. This adoption brings new danger, as deep black-box models are known to be vulnerable to adversarial attacks. In computer vision, one can shape the output during inference by performing an adversarial attack called poisoning via introducing a backdoor into the model during training. For sequences of financial transactions of a customer, insertion of a backdoor is harder to perform, as models operate over a more complex discrete space of sequences, and systematic checks for insecurities occur. We provide a method to introduce concealed backdoors, creating vulnerabilities without altering their functionality for uncontaminated data. To achieve this, we replace a clean model with a poisoned one that is aware of the availability of a backdoor and utilize this knowledge. Our most difficult for uncovering attacks include either additional supervised detection step of poisoned data activated during the test or well-hidden model weight modifications. The experimental study provides insights into how these effects vary across different datasets, architectures, and model components. Alternative methods and baselines, such as distillation-type regularization, are also explored but found to be less efficient. Conducted on three open transaction datasets and architectures, including LSTM, CNN, and Transformer, our findings not only illuminate the vulnerabilities in contemporary models but also can drive the construction of more robust systems.



## **35. Intriguing Properties of Text-guided Diffusion Models**

cs.CV

Project page: https://sage-diffusion.github.io/

**SubmitDate**: 2023-08-19    [abs](http://arxiv.org/abs/2306.00974v4) [paper-pdf](http://arxiv.org/pdf/2306.00974v4)

**Authors**: Qihao Liu, Adam Kortylewski, Yutong Bai, Song Bai, Alan Yuille

**Abstract**: Text-guided diffusion models (TDMs) are widely applied but can fail unexpectedly. Common failures include: (i) natural-looking text prompts generating images with the wrong content, or (ii) different random samples of the latent variables that generate vastly different, and even unrelated, outputs despite being conditioned on the same text prompt. In this work, we aim to study and understand the failure modes of TDMs in more detail. To achieve this, we propose SAGE, an adversarial attack on TDMs that uses image classifiers as surrogate loss functions, to search over the discrete prompt space and the high-dimensional latent space of TDMs to automatically discover unexpected behaviors and failure cases in the image generation. We make several technical contributions to ensure that SAGE finds failure cases of the diffusion model, rather than the classifier, and verify this in a human study. Our study reveals four intriguing properties of TDMs that have not been systematically studied before: (1) We find a variety of natural text prompts producing images that fail to capture the semantics of input texts. We categorize these failures into ten distinct types based on the underlying causes. (2) We find samples in the latent space (which are not outliers) that lead to distorted images independent of the text prompt, suggesting that parts of the latent space are not well-structured. (3) We also find latent samples that lead to natural-looking images which are unrelated to the text prompt, implying a potential misalignment between the latent and prompt spaces. (4) By appending a single adversarial token embedding to an input prompt we can generate a variety of specified target objects, while only minimally affecting the CLIP score. This demonstrates the fragility of language representations and raises potential safety concerns. Project page: https://sage-diffusion.github.io/



## **36. A Comparison of Adversarial Learning Techniques for Malware Detection**

cs.CR

**SubmitDate**: 2023-08-19    [abs](http://arxiv.org/abs/2308.09958v1) [paper-pdf](http://arxiv.org/pdf/2308.09958v1)

**Authors**: Pavla Louthánová, Matouš Kozák, Martin Jureček, Mark Stamp

**Abstract**: Machine learning has proven to be a useful tool for automated malware detection, but machine learning models have also been shown to be vulnerable to adversarial attacks. This article addresses the problem of generating adversarial malware samples, specifically malicious Windows Portable Executable files. We summarize and compare work that has focused on adversarial machine learning for malware detection. We use gradient-based, evolutionary algorithm-based, and reinforcement-based methods to generate adversarial samples, and then test the generated samples against selected antivirus products. We compare the selected methods in terms of accuracy and practical applicability. The results show that applying optimized modifications to previously detected malware can lead to incorrect classification of the file as benign. It is also known that generated malware samples can be successfully used against detection models other than those used to generate them and that using combinations of generators can create new samples that evade detection. Experiments show that the Gym-malware generator, which uses a reinforcement learning approach, has the greatest practical potential. This generator achieved an average sample generation time of 5.73 seconds and the highest average evasion rate of 44.11%. Using the Gym-malware generator in combination with itself improved the evasion rate to 58.35%.



## **37. Black-box Adversarial Attacks against Dense Retrieval Models: A Multi-view Contrastive Learning Method**

cs.IR

Accept by CIKM2023, 10 pages

**SubmitDate**: 2023-08-19    [abs](http://arxiv.org/abs/2308.09861v1) [paper-pdf](http://arxiv.org/pdf/2308.09861v1)

**Authors**: Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Wei Chen, Yixing Fan, Xueqi Cheng

**Abstract**: Neural ranking models (NRMs) and dense retrieval (DR) models have given rise to substantial improvements in overall retrieval performance. In addition to their effectiveness, and motivated by the proven lack of robustness of deep learning-based approaches in other areas, there is growing interest in the robustness of deep learning-based approaches to the core retrieval problem. Adversarial attack methods that have so far been developed mainly focus on attacking NRMs, with very little attention being paid to the robustness of DR models. In this paper, we introduce the adversarial retrieval attack (AREA) task. The AREA task is meant to trick DR models into retrieving a target document that is outside the initial set of candidate documents retrieved by the DR model in response to a query. We consider the decision-based black-box adversarial setting, which is realistic in real-world search engines. To address the AREA task, we first employ existing adversarial attack methods designed for NRMs. We find that the promising results that have previously been reported on attacking NRMs, do not generalize to DR models: these methods underperform a simple term spamming method. We attribute the observed lack of generalizability to the interaction-focused architecture of NRMs, which emphasizes fine-grained relevance matching. DR models follow a different representation-focused architecture that prioritizes coarse-grained representations. We propose to formalize attacks on DR models as a contrastive learning problem in a multi-view representation space. The core idea is to encourage the consistency between each view representation of the target document and its corresponding viewer via view-wise supervision signals. Experimental results demonstrate that the proposed method can significantly outperform existing attack strategies in misleading the DR model with small indiscernible text perturbations.



## **38. RAIN: RegulArization on Input and Network for Black-Box Domain Adaptation**

cs.CV

Accepted by IJCAI 2023

**SubmitDate**: 2023-08-19    [abs](http://arxiv.org/abs/2208.10531v4) [paper-pdf](http://arxiv.org/pdf/2208.10531v4)

**Authors**: Qucheng Peng, Zhengming Ding, Lingjuan Lyu, Lichao Sun, Chen Chen

**Abstract**: Source-Free domain adaptation transits the source-trained model towards target domain without exposing the source data, trying to dispel these concerns about data privacy and security. However, this paradigm is still at risk of data leakage due to adversarial attacks on the source model. Hence, the Black-Box setting only allows to use the outputs of source model, but still suffers from overfitting on the source domain more severely due to source model's unseen weights. In this paper, we propose a novel approach named RAIN (RegulArization on Input and Network) for Black-Box domain adaptation from both input-level and network-level regularization. For the input-level, we design a new data augmentation technique as Phase MixUp, which highlights task-relevant objects in the interpolations, thus enhancing input-level regularization and class consistency for target models. For network-level, we develop a Subnetwork Distillation mechanism to transfer knowledge from the target subnetwork to the full target network via knowledge distillation, which thus alleviates overfitting on the source domain by learning diverse target representations. Extensive experiments show that our method achieves state-of-the-art performance on several cross-domain benchmarks under both single- and multi-source black-box domain adaptation.



## **39. Backdoor Mitigation by Correcting the Distribution of Neural Activations**

cs.LG

**SubmitDate**: 2023-08-18    [abs](http://arxiv.org/abs/2308.09850v1) [paper-pdf](http://arxiv.org/pdf/2308.09850v1)

**Authors**: Xi Li, Zhen Xiang, David J. Miller, George Kesidis

**Abstract**: Backdoor (Trojan) attacks are an important type of adversarial exploit against deep neural networks (DNNs), wherein a test instance is (mis)classified to the attacker's target class whenever the attacker's backdoor trigger is present. In this paper, we reveal and analyze an important property of backdoor attacks: a successful attack causes an alteration in the distribution of internal layer activations for backdoor-trigger instances, compared to that for clean instances. Even more importantly, we find that instances with the backdoor trigger will be correctly classified to their original source classes if this distribution alteration is corrected. Based on our observations, we propose an efficient and effective method that achieves post-training backdoor mitigation by correcting the distribution alteration using reverse-engineered triggers. Notably, our method does not change any trainable parameters of the DNN, but achieves generally better mitigation performance than existing methods that do require intensive DNN parameter tuning. It also efficiently detects test instances with the trigger, which may help to catch adversarial entities in the act of exploiting the backdoor.



## **40. Hard No-Box Adversarial Attack on Skeleton-Based Human Action Recognition with Skeleton-Motion-Informed Gradient**

cs.CV

Camera-ready version for ICCV 2023

**SubmitDate**: 2023-08-18    [abs](http://arxiv.org/abs/2308.05681v2) [paper-pdf](http://arxiv.org/pdf/2308.05681v2)

**Authors**: Zhengzhi Lu, He Wang, Ziyi Chang, Guoan Yang, Hubert P. H. Shum

**Abstract**: Recently, methods for skeleton-based human activity recognition have been shown to be vulnerable to adversarial attacks. However, these attack methods require either the full knowledge of the victim (i.e. white-box attacks), access to training data (i.e. transfer-based attacks) or frequent model queries (i.e. black-box attacks). All their requirements are highly restrictive, raising the question of how detrimental the vulnerability is. In this paper, we show that the vulnerability indeed exists. To this end, we consider a new attack task: the attacker has no access to the victim model or the training data or labels, where we coin the term hard no-box attack. Specifically, we first learn a motion manifold where we define an adversarial loss to compute a new gradient for the attack, named skeleton-motion-informed (SMI) gradient. Our gradient contains information of the motion dynamics, which is different from existing gradient-based attack methods that compute the loss gradient assuming each dimension in the data is independent. The SMI gradient can augment many gradient-based attack methods, leading to a new family of no-box attack methods. Extensive evaluation and comparison show that our method imposes a real threat to existing classifiers. They also show that the SMI gradient improves the transferability and imperceptibility of adversarial samples in both no-box and transfer-based black-box settings.



## **41. MONA: An Efficient and Scalable Strategy for Targeted k-Nodes Collapse**

cs.SI

5 pages, 6 figures, 1 table, 5 algorithms

**SubmitDate**: 2023-08-18    [abs](http://arxiv.org/abs/2308.09601v1) [paper-pdf](http://arxiv.org/pdf/2308.09601v1)

**Authors**: Yuqian Lv, Bo Zhou, Jinhuan Wang, Shanqing Yu, Qi Xuan

**Abstract**: The concept of k-core plays an important role in measuring the cohesiveness and engagement of a network. And recent studies have shown the vulnerability of k-core under adversarial attacks. However, there are few researchers concentrating on the vulnerability of individual nodes within k-core. Therefore, in this paper, we attempt to study Targeted k-Nodes Collapse Problem (TNsCP), which focuses on removing a minimal size set of edges to make multiple target k-nodes collapse. For this purpose, we first propose a novel algorithm named MOD for candidate reduction. Then we introduce an efficient strategy named MONA, based on MOD, to address TNsCP. Extensive experiments validate the effectiveness and scalability of MONA compared to several baselines. An open-source implementation is available at https://github.com/Yocenly/MONA.



## **42. Compensating Removed Frequency Components: Thwarting Voice Spectrum Reduction Attacks**

cs.CR

Accepted by 2024 Network and Distributed System Security Symposium  (NDSS'24)

**SubmitDate**: 2023-08-18    [abs](http://arxiv.org/abs/2308.09546v1) [paper-pdf](http://arxiv.org/pdf/2308.09546v1)

**Authors**: Shu Wang, Kun Sun, Qi Li

**Abstract**: Automatic speech recognition (ASR) provides diverse audio-to-text services for humans to communicate with machines. However, recent research reveals ASR systems are vulnerable to various malicious audio attacks. In particular, by removing the non-essential frequency components, a new spectrum reduction attack can generate adversarial audios that can be perceived by humans but cannot be correctly interpreted by ASR systems. It raises a new challenge for content moderation solutions to detect harmful content in audio and video available on social media platforms. In this paper, we propose an acoustic compensation system named ACE to counter the spectrum reduction attacks over ASR systems. Our system design is based on two observations, namely, frequency component dependencies and perturbation sensitivity. First, since the Discrete Fourier Transform computation inevitably introduces spectral leakage and aliasing effects to the audio frequency spectrum, the frequency components with similar frequencies will have a high correlation. Thus, considering the intrinsic dependencies between neighboring frequency components, it is possible to recover more of the original audio by compensating for the removed components based on the remaining ones. Second, since the removed components in the spectrum reduction attacks can be regarded as an inverse of adversarial noise, the attack success rate will decrease when the adversarial audio is replayed in an over-the-air scenario. Hence, we can model the acoustic propagation process to add over-the-air perturbations into the attacked audio. We implement a prototype of ACE and the experiments show ACE can effectively reduce up to 87.9% of ASR inference errors caused by spectrum reduction attacks. Also, by analyzing residual errors, we summarize six general types of ASR inference errors and investigate the error causes and potential mitigation solutions.



## **43. Robust Evaluation of Diffusion-Based Adversarial Purification**

cs.CV

Accepted by ICCV 2023, Oral presentation

**SubmitDate**: 2023-08-18    [abs](http://arxiv.org/abs/2303.09051v2) [paper-pdf](http://arxiv.org/pdf/2303.09051v2)

**Authors**: Minjong Lee, Dongwoo Kim

**Abstract**: We question the current evaluation practice on diffusion-based purification methods. Diffusion-based purification methods aim to remove adversarial effects from an input data point at test time. The approach gains increasing attention as an alternative to adversarial training due to the disentangling between training and testing. Well-known white-box attacks are often employed to measure the robustness of the purification. However, it is unknown whether these attacks are the most effective for the diffusion-based purification since the attacks are often tailored for adversarial training. We analyze the current practices and provide a new guideline for measuring the robustness of purification methods against adversarial attacks. Based on our analysis, we further propose a new purification strategy improving robustness compared to the current diffusion-based purification methods.



## **44. REAP: A Large-Scale Realistic Adversarial Patch Benchmark**

cs.CV

ICCV 2023. Code and benchmark can be found at  https://github.com/wagner-group/reap-benchmark

**SubmitDate**: 2023-08-18    [abs](http://arxiv.org/abs/2212.05680v2) [paper-pdf](http://arxiv.org/pdf/2212.05680v2)

**Authors**: Nabeel Hingun, Chawin Sitawarin, Jerry Li, David Wagner

**Abstract**: Machine learning models are known to be susceptible to adversarial perturbation. One famous attack is the adversarial patch, a sticker with a particularly crafted pattern that makes the model incorrectly predict the object it is placed on. This attack presents a critical threat to cyber-physical systems that rely on cameras such as autonomous cars. Despite the significance of the problem, conducting research in this setting has been difficult; evaluating attacks and defenses in the real world is exceptionally costly while synthetic data are unrealistic. In this work, we propose the REAP (REalistic Adversarial Patch) benchmark, a digital benchmark that allows the user to evaluate patch attacks on real images, and under real-world conditions. Built on top of the Mapillary Vistas dataset, our benchmark contains over 14,000 traffic signs. Each sign is augmented with a pair of geometric and lighting transformations, which can be used to apply a digitally generated patch realistically onto the sign. Using our benchmark, we perform the first large-scale assessments of adversarial patch attacks under realistic conditions. Our experiments suggest that adversarial patch attacks may present a smaller threat than previously believed and that the success rate of an attack on simpler digital simulations is not predictive of its actual effectiveness in practice. We release our benchmark publicly at https://github.com/wagner-group/reap-benchmark.



## **45. Attacking logo-based phishing website detectors with adversarial perturbations**

cs.CR

To appear in ESORICS 2023

**SubmitDate**: 2023-08-18    [abs](http://arxiv.org/abs/2308.09392v1) [paper-pdf](http://arxiv.org/pdf/2308.09392v1)

**Authors**: Jehyun Lee, Zhe Xin, Melanie Ng Pei See, Kanav Sabharwal, Giovanni Apruzzese, Dinil Mon Divakaran

**Abstract**: Recent times have witnessed the rise of anti-phishing schemes powered by deep learning (DL). In particular, logo-based phishing detectors rely on DL models from Computer Vision to identify logos of well-known brands on webpages, to detect malicious webpages that imitate a given brand. For instance, Siamese networks have demonstrated notable performance for these tasks, enabling the corresponding anti-phishing solutions to detect even "zero-day" phishing webpages. In this work, we take the next step of studying the robustness of logo-based phishing detectors against adversarial ML attacks. We propose a novel attack exploiting generative adversarial perturbations to craft "adversarial logos" that evade phishing detectors. We evaluate our attacks through: (i) experiments on datasets containing real logos, to evaluate the robustness of state-of-the-art phishing detectors; and (ii) user studies to gauge whether our adversarial logos can deceive human eyes. The results show that our proposed attack is capable of crafting perturbed logos subtle enough to evade various DL models-achieving an evasion rate of up to 95%. Moreover, users are not able to spot significant differences between generated adversarial logos and original ones.



## **46. Gradient-Based Word Substitution for Obstinate Adversarial Examples Generation in Language Models**

cs.CL

19 pages

**SubmitDate**: 2023-08-18    [abs](http://arxiv.org/abs/2307.12507v2) [paper-pdf](http://arxiv.org/pdf/2307.12507v2)

**Authors**: Yimu Wang, Peng Shi, Hongyang Zhang

**Abstract**: In this paper, we study the problem of generating obstinate (over-stability) adversarial examples by word substitution in NLP, where input text is meaningfully changed but the model's prediction does not, even though it should. Previous word substitution approaches have predominantly focused on manually designed antonym-based strategies for generating obstinate adversarial examples, which hinders its application as these strategies can only find a subset of obstinate adversarial examples and require human efforts. To address this issue, in this paper, we introduce a novel word substitution method named GradObstinate, a gradient-based approach that automatically generates obstinate adversarial examples without any constraints on the search space or the need for manual design principles. To empirically evaluate the efficacy of GradObstinate, we conduct comprehensive experiments on five representative models (Electra, ALBERT, Roberta, DistillBERT, and CLIP) finetuned on four NLP benchmarks (SST-2, MRPC, SNLI, and SQuAD) and a language-grounding benchmark (MSCOCO). Extensive experiments show that our proposed GradObstinate generates more powerful obstinate adversarial examples, exhibiting a higher attack success rate compared to antonym-based methods. Furthermore, to show the transferability of obstinate word substitutions found by GradObstinate, we replace the words in four representative NLP benchmarks with their obstinate substitutions. Notably, obstinate substitutions exhibit a high success rate when transferred to other models in black-box settings, including even GPT-3 and ChatGPT. Examples of obstinate adversarial examples found by GradObstinate are available at https://huggingface.co/spaces/anonauthors/SecretLanguage.



## **47. Among Us: Adversarially Robust Collaborative Perception by Consensus**

cs.RO

Accepted by ICCV 2023

**SubmitDate**: 2023-08-18    [abs](http://arxiv.org/abs/2303.09495v3) [paper-pdf](http://arxiv.org/pdf/2303.09495v3)

**Authors**: Yiming Li, Qi Fang, Jiamu Bai, Siheng Chen, Felix Juefei-Xu, Chen Feng

**Abstract**: Multiple robots could perceive a scene (e.g., detect objects) collaboratively better than individuals, although easily suffer from adversarial attacks when using deep learning. This could be addressed by the adversarial defense, but its training requires the often-unknown attacking mechanism. Differently, we propose ROBOSAC, a novel sampling-based defense strategy generalizable to unseen attackers. Our key idea is that collaborative perception should lead to consensus rather than dissensus in results compared to individual perception. This leads to our hypothesize-and-verify framework: perception results with and without collaboration from a random subset of teammates are compared until reaching a consensus. In such a framework, more teammates in the sampled subset often entail better perception performance but require longer sampling time to reject potential attackers. Thus, we derive how many sampling trials are needed to ensure the desired size of an attacker-free subset, or equivalently, the maximum size of such a subset that we can successfully sample within a given number of trials. We validate our method on the task of collaborative 3D object detection in autonomous driving scenarios.



## **48. Targeted Adversarial Attacks on Wind Power Forecasts**

cs.LG

21 pages, including appendix, 12 figures

**SubmitDate**: 2023-08-17    [abs](http://arxiv.org/abs/2303.16633v2) [paper-pdf](http://arxiv.org/pdf/2303.16633v2)

**Authors**: René Heinrich, Christoph Scholz, Stephan Vogt, Malte Lehna

**Abstract**: In recent years, researchers proposed a variety of deep learning models for wind power forecasting. These models predict the wind power generation of wind farms or entire regions more accurately than traditional machine learning algorithms or physical models. However, latest research has shown that deep learning models can often be manipulated by adversarial attacks. Since wind power forecasts are essential for the stability of modern power systems, it is important to protect them from this threat. In this work, we investigate the vulnerability of two different forecasting models to targeted, semi-targeted, and untargeted adversarial attacks. We consider a Long Short-Term Memory (LSTM) network for predicting the power generation of individual wind farms and a Convolutional Neural Network (CNN) for forecasting the wind power generation throughout Germany. Moreover, we propose the Total Adversarial Robustness Score (TARS), an evaluation metric for quantifying the robustness of regression models to targeted and semi-targeted adversarial attacks. It assesses the impact of attacks on the model's performance, as well as the extent to which the attacker's goal was achieved, by assigning a score between 0 (very vulnerable) and 1 (very robust). In our experiments, the LSTM forecasting model was fairly robust and achieved a TARS value of over 0.78 for all adversarial attacks investigated. The CNN forecasting model only achieved TARS values below 0.10 when trained ordinarily, and was thus very vulnerable. Yet, its robustness could be significantly improved by adversarial training, which always resulted in a TARS above 0.46.



## **49. Recent Latest Message Driven GHOST: Balancing Dynamic Availability With Asynchrony Resilience**

cs.DC

**SubmitDate**: 2023-08-17    [abs](http://arxiv.org/abs/2302.11326v3) [paper-pdf](http://arxiv.org/pdf/2302.11326v3)

**Authors**: Francesco D'Amato, Luca Zanolini

**Abstract**: Dynamic participation has recently become a crucial requirement for devising permissionless consensus protocols. This notion, originally formalized by Pass and Shi (ASIACRYPT 2017) through their "sleepy model", captures the essence of a system's ability to handle participants joining or leaving during a protocol execution. A dynamically available consensus protocol preserves safety and liveness while allowing dynamic participation. Blockchain protocols, such as Bitcoin's consensus protocol, have implicitly adopted this concept. In the context of Ethereum's consensus protocol, Gasper, Neu, Tas, and Tse (S&P 2021) presented an attack against LMD-GHOST -- the component of Gasper designed to ensure dynamic availability. Consequently, LMD-GHOST results unable to fulfill its intended function of providing dynamic availability for the protocol. Despite attempts to mitigate this issue, the modified protocol still does not achieve dynamic availability, highlighting the need for more secure dynamically available protocols. In this work, we present RLMD-GHOST, a synchronous consensus protocol that not only ensures dynamic availability but also maintains safety during bounded periods of asynchrony. This protocol is particularly appealing for practical systems where strict synchrony assumptions may not always hold, contrary to general assumptions in standard synchronous protocols. Additionally, we present the "generalized sleepy model", within which our results are proven. Building upon the original sleepy model proposed by Pass and Shi, our model extends it with more generalized and stronger constraints on the corruption and sleepiness power of the adversary. This approach allows us to explore a wide range of dynamic participation regimes, spanning from complete dynamic participation to no dynamic participation, i.e., with every participant online.



## **50. A Survey on Malware Detection with Graph Representation Learning**

cs.CR

Preprint, submitted to ACM Computing Surveys on March 2023. For any  suggestions or improvements, please contact me directly by e-mail

**SubmitDate**: 2023-08-17    [abs](http://arxiv.org/abs/2303.16004v2) [paper-pdf](http://arxiv.org/pdf/2303.16004v2)

**Authors**: Tristan Bilot, Nour El Madhoun, Khaldoun Al Agha, Anis Zouaoui

**Abstract**: Malware detection has become a major concern due to the increasing number and complexity of malware. Traditional detection methods based on signatures and heuristics are used for malware detection, but unfortunately, they suffer from poor generalization to unknown attacks and can be easily circumvented using obfuscation techniques. In recent years, Machine Learning (ML) and notably Deep Learning (DL) achieved impressive results in malware detection by learning useful representations from data and have become a solution preferred over traditional methods. More recently, the application of such techniques on graph-structured data has achieved state-of-the-art performance in various domains and demonstrates promising results in learning more robust representations from malware. Yet, no literature review focusing on graph-based deep learning for malware detection exists. In this survey, we provide an in-depth literature review to summarize and unify existing works under the common approaches and architectures. We notably demonstrate that Graph Neural Networks (GNNs) reach competitive results in learning robust embeddings from malware represented as expressive graph structures, leading to an efficient detection by downstream classifiers. This paper also reviews adversarial attacks that are utilized to fool graph-based detection methods. Challenges and future research directions are discussed at the end of the paper.



