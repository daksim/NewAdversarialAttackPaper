# Latest Adversarial Attack Papers
**update at 2025-01-02 10:09:47**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Adversarial Attack and Defense for LoRa Device Identification and Authentication via Deep Learning**

cs.NI

**SubmitDate**: 2024-12-30    [abs](http://arxiv.org/abs/2412.21164v1) [paper-pdf](http://arxiv.org/pdf/2412.21164v1)

**Authors**: Yalin E. Sagduyu, Tugba Erpek

**Abstract**: LoRa provides long-range, energy-efficient communications in Internet of Things (IoT) applications that rely on Low-Power Wide-Area Network (LPWAN) capabilities. Despite these merits, concerns persist regarding the security of LoRa networks, especially in situations where device identification and authentication are imperative to secure the reliable access to the LoRa networks. This paper explores a deep learning (DL) approach to tackle these concerns, focusing on two critical tasks, namely (i) identifying LoRa devices and (ii) classifying them to legitimate and rogue devices. Deep neural networks (DNNs), encompassing both convolutional and feedforward neural networks, are trained for these tasks using actual LoRa signal data. In this setting, the adversaries may spoof rogue LoRa signals through the kernel density estimation (KDE) method based on legitimate device signals that are received by the adversaries. Two cases are considered, (i) training two separate classifiers, one for each of the two tasks, and (ii) training a multi-task classifier for both tasks. The vulnerabilities of the resulting DNNs to manipulations in input samples are studied in form of untargeted and targeted adversarial attacks using the Fast Gradient Sign Method (FGSM). Individual and common perturbations are considered against single-task and multi-task classifiers for the LoRa signal analysis. To provide resilience against such attacks, a defense approach is presented by increasing the robustness of classifiers with adversarial training. Results quantify how vulnerable LoRa signal classification tasks are to adversarial attacks and emphasize the need to fortify IoT applications against these subtle yet effective threats.



## **2. BridgePure: Revealing the Fragility of Black-box Data Protection**

cs.LG

26 pages,13 figures

**SubmitDate**: 2024-12-30    [abs](http://arxiv.org/abs/2412.21061v1) [paper-pdf](http://arxiv.org/pdf/2412.21061v1)

**Authors**: Yihan Wang, Yiwei Lu, Xiao-Shan Gao, Gautam Kamath, Yaoliang Yu

**Abstract**: Availability attacks, or unlearnable examples, are defensive techniques that allow data owners to modify their datasets in ways that prevent unauthorized machine learning models from learning effectively while maintaining the data's intended functionality. It has led to the release of popular black-box tools for users to upload personal data and receive protected counterparts. In this work, we show such black-box protections can be substantially bypassed if a small set of unprotected in-distribution data is available. Specifically, an adversary can (1) easily acquire (unprotected, protected) pairs by querying the black-box protections with the unprotected dataset; and (2) train a diffusion bridge model to build a mapping. This mapping, termed BridgePure, can effectively remove the protection from any previously unseen data within the same distribution. Under this threat model, our method demonstrates superior purification performance on classification and style mimicry tasks, exposing critical vulnerabilities in black-box data protection.



## **3. RobustBlack: Challenging Black-Box Adversarial Attacks on State-of-the-Art Defenses**

cs.LG

**SubmitDate**: 2024-12-30    [abs](http://arxiv.org/abs/2412.20987v1) [paper-pdf](http://arxiv.org/pdf/2412.20987v1)

**Authors**: Mohamed Djilani, Salah Ghamizi, Maxime Cordy

**Abstract**: Although adversarial robustness has been extensively studied in white-box settings, recent advances in black-box attacks (including transfer- and query-based approaches) are primarily benchmarked against weak defenses, leaving a significant gap in the evaluation of their effectiveness against more recent and moderate robust models (e.g., those featured in the Robustbench leaderboard). In this paper, we question this lack of attention from black-box attacks to robust models. We establish a framework to evaluate the effectiveness of recent black-box attacks against both top-performing and standard defense mechanisms, on the ImageNet dataset. Our empirical evaluation reveals the following key findings: (1) the most advanced black-box attacks struggle to succeed even against simple adversarially trained models; (2) robust models that are optimized to withstand strong white-box attacks, such as AutoAttack, also exhibits enhanced resilience against black-box attacks; and (3) robustness alignment between the surrogate models and the target model plays a key factor in the success rate of transfer-based attacks



## **4. GASLITEing the Retrieval: Exploring Vulnerabilities in Dense Embedding-based Search**

cs.CR

**SubmitDate**: 2024-12-30    [abs](http://arxiv.org/abs/2412.20953v1) [paper-pdf](http://arxiv.org/pdf/2412.20953v1)

**Authors**: Matan Ben-Tov, Mahmood Sharif

**Abstract**: Dense embedding-based text retrieval$\unicode{x2013}$retrieval of relevant passages from corpora via deep learning encodings$\unicode{x2013}$has emerged as a powerful method attaining state-of-the-art search results and popularizing the use of Retrieval Augmented Generation (RAG). Still, like other search methods, embedding-based retrieval may be susceptible to search-engine optimization (SEO) attacks, where adversaries promote malicious content by introducing adversarial passages to corpora. To faithfully assess and gain insights into the susceptibility of such systems to SEO, this work proposes the GASLITE attack, a mathematically principled gradient-based search method for generating adversarial passages without relying on the corpus content or modifying the model. Notably, GASLITE's passages (1) carry adversary-chosen information while (2) achieving high retrieval ranking for a selected query distribution when inserted to corpora. We use GASLITE to extensively evaluate retrievers' robustness, testing nine advanced models under varied threat models, while focusing on realistic adversaries targeting queries on a specific concept (e.g., a public figure). We found GASLITE consistently outperformed baselines by $\geq$140% success rate, in all settings. Particularly, adversaries using GASLITE require minimal effort to manipulate search results$\unicode{x2013}$by injecting a negligible amount of adversarial passages ($\leq$0.0001% of the corpus), they could make them visible in the top-10 results for 61-100% of unseen concept-specific queries against most evaluated models. Inspecting variance in retrievers' robustness, we identify key factors that may contribute to models' susceptibility to SEO, including specific properties in the embedding space's geometry.



## **5. Two Heads Are Better Than One: Averaging along Fine-Tuning to Improve Targeted Transferability**

cs.CV

9 pages, 6 figures, accepted by 2025ICASSP

**SubmitDate**: 2024-12-30    [abs](http://arxiv.org/abs/2412.20807v1) [paper-pdf](http://arxiv.org/pdf/2412.20807v1)

**Authors**: Hui Zeng, Sanshuai Cui, Biwei Chen, Anjie Peng

**Abstract**: With much longer optimization time than that of untargeted attacks notwithstanding, the transferability of targeted attacks is still far from satisfactory. Recent studies reveal that fine-tuning an existing adversarial example (AE) in feature space can efficiently boost its targeted transferability. However, existing fine-tuning schemes only utilize the endpoint and ignore the valuable information in the fine-tuning trajectory. Noting that the vanilla fine-tuning trajectory tends to oscillate around the periphery of a flat region of the loss surface, we propose averaging over the fine-tuning trajectory to pull the crafted AE towards a more centered region. We compare the proposed method with existing fine-tuning schemes by integrating them with state-of-the-art targeted attacks in various attacking scenarios. Experimental results uphold the superiority of the proposed method in boosting targeted transferability. The code is available at github.com/zengh5/Avg_FT.



## **6. DV-FSR: A Dual-View Target Attack Framework for Federated Sequential Recommendation**

cs.CR

I am requesting the withdrawal of my paper due to identified errors  that require significant revision

**SubmitDate**: 2024-12-30    [abs](http://arxiv.org/abs/2409.07500v2) [paper-pdf](http://arxiv.org/pdf/2409.07500v2)

**Authors**: Qitao Qin, Yucong Luo, Mingyue Cheng, Qingyang Mao, Chenyi Lei

**Abstract**: Federated recommendation (FedRec) preserves user privacy by enabling decentralized training of personalized models, but this architecture is inherently vulnerable to adversarial attacks. Significant research has been conducted on targeted attacks in FedRec systems, motivated by commercial and social influence considerations. However, much of this work has largely overlooked the differential robustness of recommendation models. Moreover, our empirical findings indicate that existing targeted attack methods achieve only limited effectiveness in Federated Sequential Recommendation (FSR) tasks. Driven by these observations, we focus on investigating targeted attacks in FSR and propose a novel dualview attack framework, named DV-FSR. This attack method uniquely combines a sampling-based explicit strategy with a contrastive learning-based implicit gradient strategy to orchestrate a coordinated attack. Additionally, we introduce a specific defense mechanism tailored for targeted attacks in FSR, aiming to evaluate the mitigation effects of the attack method we proposed. Extensive experiments validate the effectiveness of our proposed approach on representative sequential models.



## **7. Sample Correlation for Fingerprinting Deep Face Recognition**

cs.CV

**SubmitDate**: 2024-12-30    [abs](http://arxiv.org/abs/2412.20768v1) [paper-pdf](http://arxiv.org/pdf/2412.20768v1)

**Authors**: Jiyang Guan, Jian Liang, Yanbo Wang, Ran He

**Abstract**: Face recognition has witnessed remarkable advancements in recent years, thanks to the development of deep learning techniques.However, an off-the-shelf face recognition model as a commercial service could be stolen by model stealing attacks, posing great threats to the rights of the model owner.Model fingerprinting, as a model stealing detection method, aims to verify whether a suspect model is stolen from the victim model, gaining more and more attention nowadays.Previous methods always utilize transferable adversarial examples as the model fingerprint, but this method is known to be sensitive to adversarial defense and transfer learning techniques.To address this issue, we consider the pairwise relationship between samples instead and propose a novel yet simple model stealing detection method based on SAmple Correlation (SAC).Specifically, we present SAC-JC that selects JPEG compressed samples as model inputs and calculates the correlation matrix among their model outputs.Extensive results validate that SAC successfully defends against various model stealing attacks in deep face recognition, encompassing face verification and face emotion recognition, exhibiting the highest performance in terms of AUC, p-value and F1 score.Furthermore, we extend our evaluation of SAC-JC to object recognition datasets including Tiny-ImageNet and CIFAR10, which also demonstrates the superior performance of SAC-JC to previous methods.The code will be available at \url{https://github.com/guanjiyang/SAC_JC}.



## **8. Enhancing Privacy in Federated Learning through Quantum Teleportation Integration**

quant-ph

**SubmitDate**: 2024-12-30    [abs](http://arxiv.org/abs/2412.20762v1) [paper-pdf](http://arxiv.org/pdf/2412.20762v1)

**Authors**: Koffka Khan

**Abstract**: Federated learning enables collaborative model training across multiple clients without sharing raw data, thereby enhancing privacy. However, the exchange of model updates can still expose sensitive information. Quantum teleportation, a process that transfers quantum states between distant locations without physical transmission of the particles themselves, has recently been implemented in real-world networks. This position paper explores the potential of integrating quantum teleportation into federated learning frameworks to bolster privacy. By leveraging quantum entanglement and the no-cloning theorem, quantum teleportation ensures that data remains secure during transmission, as any eavesdropping attempt would be detectable. We propose a novel architecture where quantum teleportation facilitates the secure exchange of model parameters and gradients among clients and servers. This integration aims to mitigate risks associated with data leakage and adversarial attacks inherent in classical federated learning setups. We also discuss the practical challenges of implementing such a system, including the current limitations of quantum network infrastructure and the need for hybrid quantum-classical protocols. Our analysis suggests that, despite these challenges, the convergence of quantum communication technologies and federated learning presents a promising avenue for achieving unprecedented levels of privacy in distributed machine learning.



## **9. Unsupervised dense retrieval with conterfactual contrastive learning**

cs.IR

arXiv admin note: text overlap with arXiv:2107.07773 by other authors

**SubmitDate**: 2024-12-30    [abs](http://arxiv.org/abs/2412.20756v1) [paper-pdf](http://arxiv.org/pdf/2412.20756v1)

**Authors**: Haitian Chen, Qingyao Ai, Xiao Wang, Yiqun Liu, Fen Lin, Qin Liu

**Abstract**: Efficiently retrieving a concise set of candidates from a large document corpus remains a pivotal challenge in Information Retrieval (IR). Neural retrieval models, particularly dense retrieval models built with transformers and pretrained language models, have been popular due to their superior performance. However, criticisms have also been raised on their lack of explainability and vulnerability to adversarial attacks. In response to these challenges, we propose to improve the robustness of dense retrieval models by enhancing their sensitivity of fine-graned relevance signals. A model achieving sensitivity in this context should exhibit high variances when documents' key passages determining their relevance to queries have been modified, while maintaining low variances for other changes in irrelevant passages. This sensitivity allows a dense retrieval model to produce robust results with respect to attacks that try to promote documents without actually increasing their relevance. It also makes it possible to analyze which part of a document is actually relevant to a query, and thus improve the explainability of the retrieval model. Motivated by causality and counterfactual analysis, we propose a series of counterfactual regularization methods based on game theory and unsupervised learning with counterfactual passages. Experiments show that, our method can extract key passages without reliance on the passage-level relevance annotations. Moreover, the regularized dense retrieval models exhibit heightened robustness against adversarial attacks, surpassing the state-of-the-art anti-attack methods.



## **10. Graph Neural Networks for Next-Generation-IoT: Recent Advances and Open Challenges**

cs.IT

28 pages, 15 figures, and 6 tables. Submitted for publication

**SubmitDate**: 2024-12-30    [abs](http://arxiv.org/abs/2412.20634v1) [paper-pdf](http://arxiv.org/pdf/2412.20634v1)

**Authors**: Nguyen Xuan Tung, Le Tung Giang, Bui Duc Son, Seon Geun Jeong, Trinh Van Chien, Won Joo Hwang, Lajos Hanzo

**Abstract**: Graph Neural Networks (GNNs) have emerged as a critical tool for optimizing and managing the complexities of the Internet of Things (IoT) in next-generation networks. This survey presents a comprehensive exploration of how GNNs may be harnessed in 6G IoT environments, focusing on key challenges and opportunities through a series of open questions. We commence with an exploration of GNN paradigms and the roles of node, edge, and graph-level tasks in solving wireless networking problems and highlight GNNs' ability to overcome the limitations of traditional optimization methods. This guidance enhances problem-solving efficiency across various next-generation (NG) IoT scenarios. Next, we provide a detailed discussion of the application of GNN in advanced NG enabling technologies, including massive MIMO, reconfigurable intelligent surfaces, satellites, THz, mobile edge computing (MEC), and ultra-reliable low latency communication (URLLC). We then delve into the challenges posed by adversarial attacks, offering insights into defense mechanisms to secure GNN-based NG-IoT networks. Next, we examine how GNNs can be integrated with future technologies like integrated sensing and communication (ISAC), satellite-air-ground-sea integrated networks (SAGSIN), and quantum computing. Our findings highlight the transformative potential of GNNs in improving efficiency, scalability, and security within NG-IoT systems, paving the way for future advances. Finally, we propose a set of design guidelines to facilitate the development of efficient, scalable, and secure GNN models tailored for NG IoT applications.



## **11. ErasableMask: A Robust and Erasable Privacy Protection Scheme against Black-box Face Recognition Models**

cs.CV

**SubmitDate**: 2024-12-29    [abs](http://arxiv.org/abs/2412.17038v3) [paper-pdf](http://arxiv.org/pdf/2412.17038v3)

**Authors**: Sipeng Shen, Yunming Zhang, Dengpan Ye, Xiuwen Shi, Long Tang, Haoran Duan, Jiacheng Deng, Ziyi Liu

**Abstract**: While face recognition (FR) models have brought remarkable convenience in face verification and identification, they also pose substantial privacy risks to the public. Existing facial privacy protection schemes usually adopt adversarial examples to disrupt face verification of FR models. However, these schemes often suffer from weak transferability against black-box FR models and permanently damage the identifiable information that cannot fulfill the requirements of authorized operations such as forensics and authentication. To address these limitations, we propose ErasableMask, a robust and erasable privacy protection scheme against black-box FR models. Specifically, via rethinking the inherent relationship between surrogate FR models, ErasableMask introduces a novel meta-auxiliary attack, which boosts black-box transferability by learning more general features in a stable and balancing optimization strategy. It also offers a perturbation erasion mechanism that supports the erasion of semantic perturbations in protected face without degrading image quality. To further improve performance, ErasableMask employs a curriculum learning strategy to mitigate optimization conflicts between adversarial attack and perturbation erasion. Extensive experiments on the CelebA-HQ and FFHQ datasets demonstrate that ErasableMask achieves the state-of-the-art performance in transferability, achieving over 72% confidence on average in commercial FR systems. Moreover, ErasableMask also exhibits outstanding perturbation erasion performance, achieving over 90% erasion success rate.



## **12. Real-time Fake News from Adversarial Feedback**

cs.CL

**SubmitDate**: 2024-12-29    [abs](http://arxiv.org/abs/2410.14651v2) [paper-pdf](http://arxiv.org/pdf/2410.14651v2)

**Authors**: Sanxing Chen, Yukun Huang, Bhuwan Dhingra

**Abstract**: We show that existing evaluations for fake news detection based on conventional sources, such as claims on fact-checking websites, result in high accuracies over time for LLM-based detectors -- even after their knowledge cutoffs. This suggests that recent popular fake news from such sources can be easily detected due to pre-training and retrieval corpus contamination or increasingly salient shallow patterns. Instead, we argue that a proper fake news detection dataset should test a model's ability to reason factually about the current world by retrieving and reading related evidence. To this end, we develop a novel pipeline that leverages natural language feedback from a RAG-based detector to iteratively modify real-time news into deceptive fake news that challenges LLMs. Our iterative rewrite decreases the binary classification ROC-AUC by an absolute 17.5 percent for a strong RAG-based GPT-4o detector. Our experiments reveal the important role of RAG in both detecting and generating fake news, as retrieval-free LLM detectors are vulnerable to unseen events and adversarial attacks, while feedback from RAG detection helps discover more deceitful patterns in fake news.



## **13. Optimal and Feasible Contextuality-based Randomness Generation**

quant-ph

21 pages, 8 figures

**SubmitDate**: 2024-12-28    [abs](http://arxiv.org/abs/2412.20126v1) [paper-pdf](http://arxiv.org/pdf/2412.20126v1)

**Authors**: Yuan Liu, Ravishankar Ramanathan

**Abstract**: Semi-device-independent (SDI) randomness generation protocols based on Kochen-Specker contextuality offer the attractive features of compact devices, high rates, and ease of experimental implementation over fully device-independent (DI) protocols. Here, we investigate this paradigm and derive four results to improve the state-of-art. Firstly, we introduce a family of simple, experimentally feasible orthogonality graphs (measurement compatibility structures) for which the maximum violation of the corresponding non-contextuality inequalities allows to certify the maximum amount of $\log_2 d$ bits from a qu$d$it system with projective measurements for $d \geq 3$. We analytically derive the Lovasz theta and fractional packing number for this graph family, and thereby prove their utility for optimal randomness generation in both randomness expansion and amplification tasks. Secondly, a central additional assumption in contextuality-based protocols over fully DI ones, is that the measurements are repeatable and satisfy an intended compatibility structure. We frame a relaxation of this condition in terms of $\epsilon$-orthogonality graphs for a parameter $\epsilon > 0$, and derive quantum correlations that allow to certify randomness for arbitrary relaxation $\epsilon \in [0,1)$. Thirdly, it is well known that a single qubit is non-contextual, i.e., the qubit correlations can be explained by a non-contextual hidden variable (NCHV) model. We show however that a single qubit is \textit{almost} contextual, in that there exist qubit correlations that cannot be explained by $\epsilon$-ontologically faithful NCHV models for small $\epsilon > 0$. Finally, we point out possible attacks by quantum and general consistent (non-signalling) adversaries for certain classes of contextuality tests over and above those considered in DI scenarios.



## **14. On the Validity of Traditional Vulnerability Scoring Systems for Adversarial Attacks against LLMs**

cs.CR

101 pages, 3 figures

**SubmitDate**: 2024-12-28    [abs](http://arxiv.org/abs/2412.20087v1) [paper-pdf](http://arxiv.org/pdf/2412.20087v1)

**Authors**: Atmane Ayoub Mansour Bahar, Ahmad Samer Wazan

**Abstract**: This research investigates the effectiveness of established vulnerability metrics, such as the Common Vulnerability Scoring System (CVSS), in evaluating attacks against Large Language Models (LLMs), with a focus on Adversarial Attacks (AAs). The study explores the influence of both general and specific metric factors in determining vulnerability scores, providing new perspectives on potential enhancements to these metrics.   This study adopts a quantitative approach, calculating and comparing the coefficient of variation of vulnerability scores across 56 adversarial attacks on LLMs. The attacks, sourced from various research papers, and obtained through online databases, were evaluated using multiple vulnerability metrics. Scores were determined by averaging the values assessed by three distinct LLMs. The results indicate that existing scoring-systems yield vulnerability scores with minimal variation across different attacks, suggesting that many of the metric factors are inadequate for assessing adversarial attacks on LLMs. This is particularly true for context-specific factors or those with predefined value sets, such as those in CVSS. These findings support the hypothesis that current vulnerability metrics, especially those with rigid values, are limited in evaluating AAs on LLMs, highlighting the need for the development of more flexible, generalized metrics tailored to such attacks.   This research offers a fresh analysis of the effectiveness and applicability of established vulnerability metrics, particularly in the context of Adversarial Attacks on Large Language Models, both of which have gained significant attention in recent years. Through extensive testing and calculations, the study underscores the limitations of these metrics and opens up new avenues for improving and refining vulnerability assessment frameworks specifically tailored for LLMs.



## **15. B-AVIBench: Towards Evaluating the Robustness of Large Vision-Language Model on Black-box Adversarial Visual-Instructions**

cs.CV

Accepted by IEEE Transactions on Information Forensics & Security

**SubmitDate**: 2024-12-28    [abs](http://arxiv.org/abs/2403.09346v2) [paper-pdf](http://arxiv.org/pdf/2403.09346v2)

**Authors**: Hao Zhang, Wenqi Shao, Hong Liu, Yongqiang Ma, Ping Luo, Yu Qiao, Nanning Zheng, Kaipeng Zhang

**Abstract**: Large Vision-Language Models (LVLMs) have shown significant progress in responding well to visual-instructions from users. However, these instructions, encompassing images and text, are susceptible to both intentional and inadvertent attacks. Despite the critical importance of LVLMs' robustness against such threats, current research in this area remains limited. To bridge this gap, we introduce B-AVIBench, a framework designed to analyze the robustness of LVLMs when facing various Black-box Adversarial Visual-Instructions (B-AVIs), including four types of image-based B-AVIs, ten types of text-based B-AVIs, and nine types of content bias B-AVIs (such as gender, violence, cultural, and racial biases, among others). We generate 316K B-AVIs encompassing five categories of multimodal capabilities (ten tasks) and content bias. We then conduct a comprehensive evaluation involving 14 open-source LVLMs to assess their performance. B-AVIBench also serves as a convenient tool for practitioners to evaluate the robustness of LVLMs against B-AVIs. Our findings and extensive experimental results shed light on the vulnerabilities of LVLMs, and highlight that inherent biases exist even in advanced closed-source LVLMs like GeminiProVision and GPT-4V. This underscores the importance of enhancing the robustness, security, and fairness of LVLMs. The source code and benchmark are available at https://github.com/zhanghao5201/B-AVIBench.



## **16. A Robust Adversarial Ensemble with Causal (Feature Interaction) Interpretations for Image Classification**

cs.CV

**SubmitDate**: 2024-12-28    [abs](http://arxiv.org/abs/2412.20025v1) [paper-pdf](http://arxiv.org/pdf/2412.20025v1)

**Authors**: Chunheng Zhao, Pierluigi Pisu, Gurcan Comert, Negash Begashaw, Varghese Vaidyan, Nina Christine Hubig

**Abstract**: Deep learning-based discriminative classifiers, despite their remarkable success, remain vulnerable to adversarial examples that can mislead model predictions. While adversarial training can enhance robustness, it fails to address the intrinsic vulnerability stemming from the opaque nature of these black-box models. We present a deep ensemble model that combines discriminative features with generative models to achieve both high accuracy and adversarial robustness. Our approach integrates a bottom-level pre-trained discriminative network for feature extraction with a top-level generative classification network that models adversarial input distributions through a deep latent variable model. Using variational Bayes, our model achieves superior robustness against white-box adversarial attacks without adversarial training. Extensive experiments on CIFAR-10 and CIFAR-100 demonstrate our model's superior adversarial robustness. Through evaluations using counterfactual metrics and feature interaction-based metrics, we establish correlations between model interpretability and adversarial robustness. Additionally, preliminary results on Tiny-ImageNet validate our approach's scalability to more complex datasets, offering a practical solution for developing robust image classification models.



## **17. Adversarial Robustness for Deep Learning-based Wildfire Detection Models**

cs.CV

**SubmitDate**: 2024-12-28    [abs](http://arxiv.org/abs/2412.20006v1) [paper-pdf](http://arxiv.org/pdf/2412.20006v1)

**Authors**: Ryo Ide, Lei Yang

**Abstract**: Smoke detection using Deep Neural Networks (DNNs) is an effective approach for early wildfire detection. However, because smoke is temporally and spatially anomalous, there are limitations in collecting sufficient training data. This raises overfitting and bias concerns in existing DNN-based wildfire detection models. Thus, we introduce WARP (Wildfire Adversarial Robustness Procedure), the first model-agnostic framework for evaluating the adversarial robustness of DNN-based wildfire detection models. WARP addresses limitations in smoke image diversity using global and local adversarial attack methods. The global attack method uses image-contextualized Gaussian noise, while the local attack method uses patch noise injection, tailored to address critical aspects of wildfire detection. Leveraging WARP's model-agnostic capabilities, we assess the adversarial robustness of real-time Convolutional Neural Networks (CNNs) and Transformers. The analysis revealed valuable insights into the models' limitations. Specifically, the global attack method demonstrates that the Transformer model has more than 70\% precision degradation than the CNN against global noise. In contrast, the local attack method shows that both models are susceptible to cloud image injections when detecting smoke-positive instances, suggesting a need for model improvements through data augmentation. WARP's comprehensive robustness analysis contributed to the development of wildfire-specific data augmentation strategies, marking a step toward practicality.



## **18. Standard-Deviation-Inspired Regularization for Improving Adversarial Robustness**

cs.LG

**SubmitDate**: 2024-12-27    [abs](http://arxiv.org/abs/2412.19947v1) [paper-pdf](http://arxiv.org/pdf/2412.19947v1)

**Authors**: Olukorede Fakorede, Modeste Atsague, Jin Tian

**Abstract**: Adversarial Training (AT) has been demonstrated to improve the robustness of deep neural networks (DNNs) against adversarial attacks. AT is a min-max optimization procedure where in adversarial examples are generated to train a more robust DNN. The inner maximization step of AT increases the losses of inputs with respect to their actual classes. The outer minimization involves minimizing the losses on the adversarial examples obtained from the inner maximization. This work proposes a standard-deviation-inspired (SDI) regularization term to improve adversarial robustness and generalization. We argue that the inner maximization in AT is similar to minimizing a modified standard deviation of the model's output probabilities. Moreover, we suggest that maximizing this modified standard deviation can complement the outer minimization of the AT framework. To support our argument, we experimentally show that the SDI measure can be used to craft adversarial examples. Additionally, we demonstrate that combining the SDI regularization term with existing AT variants enhances the robustness of DNNs against stronger attacks, such as CW and Auto-attack, and improves generalization.



## **19. A High Dimensional Statistical Model for Adversarial Training: Geometry and Trade-Offs**

stat.ML

**SubmitDate**: 2024-12-27    [abs](http://arxiv.org/abs/2402.05674v3) [paper-pdf](http://arxiv.org/pdf/2402.05674v3)

**Authors**: Kasimir Tanner, Matteo Vilucchio, Bruno Loureiro, Florent Krzakala

**Abstract**: This work investigates adversarial training in the context of margin-based linear classifiers in the high-dimensional regime where the dimension $d$ and the number of data points $n$ diverge with a fixed ratio $\alpha = n / d$. We introduce a tractable mathematical model where the interplay between the data and adversarial attacker geometries can be studied, while capturing the core phenomenology observed in the adversarial robustness literature. Our main theoretical contribution is an exact asymptotic description of the sufficient statistics for the adversarial empirical risk minimiser, under generic convex and non-increasing losses for a Block Feature Model. Our result allow us to precisely characterise which directions in the data are associated with a higher generalisation/robustness trade-off, as defined by a robustness and a usefulness metric. We show that the the presence of multiple different feature types is crucial to the high sample complexity performances of adversarial training. In particular, we unveil the existence of directions which can be defended without penalising accuracy. Finally, we show the advantage of defending non-robust features during training, identifying a uniform protection as an inherently effective defence mechanism.



## **20. Enhancing Adversarial Robustness of Deep Neural Networks Through Supervised Contrastive Learning**

cs.LG

8 pages, 11 figures

**SubmitDate**: 2024-12-27    [abs](http://arxiv.org/abs/2412.19747v1) [paper-pdf](http://arxiv.org/pdf/2412.19747v1)

**Authors**: Longwei Wang, Navid Nayyem, Abdullah Rakin

**Abstract**: Adversarial attacks exploit the vulnerabilities of convolutional neural networks by introducing imperceptible perturbations that lead to misclassifications, exposing weaknesses in feature representations and decision boundaries. This paper presents a novel framework combining supervised contrastive learning and margin-based contrastive loss to enhance adversarial robustness. Supervised contrastive learning improves the structure of the feature space by clustering embeddings of samples within the same class and separating those from different classes. Margin-based contrastive loss, inspired by support vector machines, enforces explicit constraints to create robust decision boundaries with well-defined margins. Experiments on the CIFAR-100 dataset with a ResNet-18 backbone demonstrate robustness performance improvements in adversarial accuracy under Fast Gradient Sign Method attacks.



## **21. Gröbner Basis Cryptanalysis of Ciminion and Hydra**

cs.CR

**SubmitDate**: 2024-12-27    [abs](http://arxiv.org/abs/2405.05040v3) [paper-pdf](http://arxiv.org/pdf/2405.05040v3)

**Authors**: Matthias Johann Steiner

**Abstract**: Ciminion and Hydra are two recently introduced symmetric key Pseudo-Random Functions for Multi-Party Computation applications. For efficiency both primitives utilize quadratic permutations at round level. Therefore, polynomial system solving-based attacks pose a serious threat to these primitives. For Ciminion, we construct a quadratic degree reverse lexicographic (DRL) Gr\"obner basis for the iterated polynomial model via linear transformations. With the Gr\"obner basis we can simplify cryptanalysis since we do not need to impose genericity assumptions anymore to derive complexity estimations. For Hydra, with the help of a computer algebra program like SageMath we construct a DRL Gr\"obner basis for the iterated model via linear transformations and a linear change of coordinates. In the Hydra proposal it was claimed that $r_\mathcal{H} = 31$ rounds are sufficient to provide $128$ bits of security against Gr\"obner basis attacks for an ideal adversary with $\omega = 2$. However, via our Hydra Gr\"obner basis standard term order conversion to a lexicographic (LEX) Gr\"obner basis requires just $126$ bits with $\omega = 2$. Moreover, via a dedicated polynomial system solving technique up to $r_\mathcal{H} = 33$ rounds can be attacked below $128$ bits for an ideal adversary.



## **22. Attribution for Enhanced Explanation with Transferable Adversarial eXploration**

cs.AI

**SubmitDate**: 2024-12-27    [abs](http://arxiv.org/abs/2412.19523v1) [paper-pdf](http://arxiv.org/pdf/2412.19523v1)

**Authors**: Zhiyu Zhu, Jiayu Zhang, Zhibo Jin, Huaming Chen, Jianlong Zhou, Fang Chen

**Abstract**: The interpretability of deep neural networks is crucial for understanding model decisions in various applications, including computer vision. AttEXplore++, an advanced framework built upon AttEXplore, enhances attribution by incorporating transferable adversarial attack methods such as MIG and GRA, significantly improving the accuracy and robustness of model explanations. We conduct extensive experiments on five models, including CNNs (Inception-v3, ResNet-50, VGG16) and vision transformers (MaxViT-T, ViT-B/16), using the ImageNet dataset. Our method achieves an average performance improvement of 7.57\% over AttEXplore and 32.62\% compared to other state-of-the-art interpretability algorithms. Using insertion and deletion scores as evaluation metrics, we show that adversarial transferability plays a vital role in enhancing attribution results. Furthermore, we explore the impact of randomness, perturbation rate, noise amplitude, and diversity probability on attribution performance, demonstrating that AttEXplore++ provides more stable and reliable explanations across various models. We release our code at: https://anonymous.4open.science/r/ATTEXPLOREP-8435/



## **23. An Engorgio Prompt Makes Large Language Model Babble on**

cs.CR

**SubmitDate**: 2024-12-27    [abs](http://arxiv.org/abs/2412.19394v1) [paper-pdf](http://arxiv.org/pdf/2412.19394v1)

**Authors**: Jianshuo Dong, Ziyuan Zhang, Qingjie Zhang, Han Qiu, Tianwei Zhang, Hao Wang, Hewu Li, Qi Li, Chao Zhang, Ke Xu

**Abstract**: Auto-regressive large language models (LLMs) have yielded impressive performance in many real-world tasks. However, the new paradigm of these LLMs also exposes novel threats. In this paper, we explore their vulnerability to inference cost attacks, where a malicious user crafts Engorgio prompts to intentionally increase the computation cost and latency of the inference process. We design Engorgio, a novel methodology, to efficiently generate adversarial Engorgio prompts to affect the target LLM's service availability. Engorgio has the following two technical contributions. (1) We employ a parameterized distribution to track LLMs' prediction trajectory. (2) Targeting the auto-regressive nature of LLMs' inference process, we propose novel loss functions to stably suppress the appearance of the <EOS> token, whose occurrence will interrupt the LLM's generation process. We conduct extensive experiments on 13 open-sourced LLMs with parameters ranging from 125M to 30B. The results show that Engorgio prompts can successfully induce LLMs to generate abnormally long outputs (i.e., roughly 2-13$\times$ longer to reach 90%+ of the output length limit) in a white-box scenario and our real-world experiment demonstrates Engergio's threat to LLM service with limited computing resources. The code is accessible at https://github.com/jianshuod/Engorgio-prompt.



## **24. Quantum-Inspired Weight-Constrained Neural Network: Reducing Variable Numbers by 100x Compared to Standard Neural Networks**

quant-ph

13 pages, 5 figures. Comments are welcome

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2412.19355v1) [paper-pdf](http://arxiv.org/pdf/2412.19355v1)

**Authors**: Shaozhi Li, M Sabbir Salek, Binayyak Roy, Yao Wang, Mashrur Chowdhury

**Abstract**: Although quantum machine learning has shown great promise, the practical application of quantum computers remains constrained in the noisy intermediate-scale quantum era. To take advantage of quantum machine learning, we investigate the underlying mathematical principles of these quantum models and adapt them to classical machine learning frameworks. Specifically, we develop a classical weight-constrained neural network that generates weights based on quantum-inspired insights. We find that this approach can reduce the number of variables in a classical neural network by a factor of 135 while preserving its learnability. In addition, we develop a dropout method to enhance the robustness of quantum machine learning models, which are highly susceptible to adversarial attacks. This technique can also be applied to improve the adversarial resilience of the classical weight-constrained neural network, which is essential for industry applications, such as self-driving vehicles. Our work offers a novel approach to reduce the complexity of large classical neural networks, addressing a critical challenge in machine learning.



## **25. Federated Hybrid Training and Self-Adversarial Distillation: Towards Robust Edge Networks**

cs.CV

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2412.19354v1) [paper-pdf](http://arxiv.org/pdf/2412.19354v1)

**Authors**: Yu Qiao, Apurba Adhikary, Kitae Kim, Eui-Nam Huh, Zhu Han, Choong Seon Hong

**Abstract**: Federated learning (FL) is a distributed training technology that enhances data privacy in mobile edge networks by allowing data owners to collaborate without transmitting raw data to the edge server. However, data heterogeneity and adversarial attacks pose challenges to develop an unbiased and robust global model for edge deployment. To address this, we propose Federated hyBrid Adversarial training and self-adversarial disTillation (FedBAT), a new framework designed to improve both robustness and generalization of the global model. FedBAT seamlessly integrates hybrid adversarial training and self-adversarial distillation into the conventional FL framework from data augmentation and feature distillation perspectives. From a data augmentation perspective, we propose hybrid adversarial training to defend against adversarial attacks by balancing accuracy and robustness through a weighted combination of standard and adversarial training. From a feature distillation perspective, we introduce a novel augmentation-invariant adversarial distillation method that aligns local adversarial features of augmented images with their corresponding unbiased global clean features. This alignment can effectively mitigate bias from data heterogeneity while enhancing both the robustness and generalization of the global model. Extensive experimental results across multiple datasets demonstrate that FedBAT yields comparable or superior performance gains in improving robustness while maintaining accuracy compared to several baselines.



## **26. xSRL: Safety-Aware Explainable Reinforcement Learning -- Safety as a Product of Explainability**

cs.AI

Accepted to 24th International Conference on Autonomous Agents and  Multiagent Systems (AAMAS 2025)

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2412.19311v1) [paper-pdf](http://arxiv.org/pdf/2412.19311v1)

**Authors**: Risal Shahriar Shefin, Md Asifur Rahman, Thai Le, Sarra Alqahtani

**Abstract**: Reinforcement learning (RL) has shown great promise in simulated environments, such as games, where failures have minimal consequences. However, the deployment of RL agents in real-world systems such as autonomous vehicles, robotics, UAVs, and medical devices demands a higher level of safety and transparency, particularly when facing adversarial threats. Safe RL algorithms have been developed to address these concerns by optimizing both task performance and safety constraints. However, errors are inevitable, and when they occur, it is essential that the RL agents can also explain their actions to human operators. This makes trust in the safety mechanisms of RL systems crucial for effective deployment. Explainability plays a key role in building this trust by providing clear, actionable insights into the agent's decision-making process, ensuring that safety-critical decisions are well understood. While machine learning (ML) has seen significant advances in interpretability and visualization, explainability methods for RL remain limited. Current tools fail to address the dynamic, sequential nature of RL and its needs to balance task performance with safety constraints over time. The re-purposing of traditional ML methods, such as saliency maps, is inadequate for safety-critical RL applications where mistakes can result in severe consequences. To bridge this gap, we propose xSRL, a framework that integrates both local and global explanations to provide a comprehensive understanding of RL agents' behavior. xSRL also enables developers to identify policy vulnerabilities through adversarial attacks, offering tools to debug and patch agents without retraining. Our experiments and user studies demonstrate xSRL's effectiveness in increasing safety in RL systems, making them more reliable and trustworthy for real-world deployment. Code is available at https://github.com/risal-shefin/xSRL.



## **27. Game-Theoretically Secure Distributed Protocols for Fair Allocation in Coalitional Games**

cs.GT

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2412.19192v1) [paper-pdf](http://arxiv.org/pdf/2412.19192v1)

**Authors**: T-H. Hubert Chan, Qipeng Kuang, Quan Xue

**Abstract**: We consider game-theoretically secure distributed protocols for coalition games that approximate the Shapley value with small multiplicative error. Since all known existing approximation algorithms for the Shapley value are randomized, it is a challenge to design efficient distributed protocols among mutually distrusted players when there is no central authority to generate unbiased randomness. The game-theoretic notion of maximin security has been proposed to offer guarantees to an honest player's reward even if all other players are susceptible to an adversary.   Permutation sampling is often used in approximation algorithms for the Shapley value. A previous work in 1994 by Zlotkin et al. proposed a simple constant-round distributed permutation generation protocol based on commitment scheme, but it is vulnerable to rushing attacks. The protocol, however, can detect such attacks.   In this work, we model the limited resources of an adversary by a violation budget that determines how many times it can perform such detectable attacks. Therefore, by repeating the number of permutation samples, an honest player's reward can be guaranteed to be close to its Shapley value. We explore both high probability and expected maximin security. We obtain an upper bound on the number of permutation samples for high probability maximin security, even with an unknown violation budget. Furthermore, we establish a matching lower bound for the weaker notion of expected maximin security in specific permutation generation protocols. We have also performed experiments on both synthetic and real data to empirically verify our results.



## **28. TSCheater: Generating High-Quality Tibetan Adversarial Texts via Visual Similarity**

cs.CL

Camera-Ready Version; Accepted at ICASSP 2025

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2412.02371v3) [paper-pdf](http://arxiv.org/pdf/2412.02371v3)

**Authors**: Xi Cao, Quzong Gesang, Yuan Sun, Nuo Qun, Tashi Nyima

**Abstract**: Language models based on deep neural networks are vulnerable to textual adversarial attacks. While rich-resource languages like English are receiving focused attention, Tibetan, a cross-border language, is gradually being studied due to its abundant ancient literature and critical language strategy. Currently, there are several Tibetan adversarial text generation methods, but they do not fully consider the textual features of Tibetan script and overestimate the quality of generated adversarial texts. To address this issue, we propose a novel Tibetan adversarial text generation method called TSCheater, which considers the characteristic of Tibetan encoding and the feature that visually similar syllables have similar semantics. This method can also be transferred to other abugidas, such as Devanagari script. We utilize a self-constructed Tibetan syllable visual similarity database called TSVSDB to generate substitution candidates and adopt a greedy algorithm-based scoring mechanism to determine substitution order. After that, we conduct the method on eight victim language models. Experimentally, TSCheater outperforms existing methods in attack effectiveness, perturbation magnitude, semantic similarity, visual similarity, and human acceptance. Finally, we construct the first Tibetan adversarial robustness evaluation benchmark called AdvTS, which is generated by existing methods and proofread by humans.



## **29. DiffPatch: Generating Customizable Adversarial Patches using Diffusion Model**

cs.CV

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2412.01440v2) [paper-pdf](http://arxiv.org/pdf/2412.01440v2)

**Authors**: Zhixiang Wang, Guangnan Ye, Xiaosen Wang, Siheng Chen, Zhibo Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Physical adversarial patches printed on clothing can easily allow individuals to evade person detectors. However, most existing adversarial patch generation methods prioritize attack effectiveness over stealthiness, resulting in patches that are aesthetically unpleasing. Although existing methods using generative adversarial networks or diffusion models can produce more natural-looking patches, they often struggle to balance stealthiness with attack effectiveness and lack flexibility for user customization. To address these challenges, we propose a novel diffusion-based customizable patch generation framework termed DiffPatch, specifically tailored for creating naturalistic and customizable adversarial patches. Our approach enables users to utilize a reference image as the source, rather than starting from random noise, and incorporates masks to craft naturalistic patches of various shapes, not limited to squares. To prevent the original semantics from being lost during the diffusion process, we employ Null-text inversion to map random noise samples to a single input image and generate patches through Incomplete Diffusion Optimization (IDO). Notably, while maintaining a natural appearance, our method achieves a comparable attack performance to state-of-the-art non-naturalistic patches when using similarly sized attacks. Using DiffPatch, we have created a physical adversarial T-shirt dataset, AdvPatch-1K, specifically targeting YOLOv5s. This dataset includes over a thousand images across diverse scenarios, validating the effectiveness of our attack in real-world environments. Moreover, it provides a valuable resource for future research.



## **30. Provable Robust Saliency-based Explanations**

cs.LG

Accepted to NeurIPS 2024

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2212.14106v4) [paper-pdf](http://arxiv.org/pdf/2212.14106v4)

**Authors**: Chao Chen, Chenghua Guo, Rufeng Chen, Guixiang Ma, Ming Zeng, Xiangwen Liao, Xi Zhang, Sihong Xie

**Abstract**: To foster trust in machine learning models, explanations must be faithful and stable for consistent insights. Existing relevant works rely on the $\ell_p$ distance for stability assessment, which diverges from human perception. Besides, existing adversarial training (AT) associated with intensive computations may lead to an arms race. To address these challenges, we introduce a novel metric to assess the stability of top-$k$ salient features. We introduce R2ET which trains for stable explanation by efficient and effective regularizer, and analyze R2ET by multi-objective optimization to prove numerical and statistical stability of explanations. Moreover, theoretical connections between R2ET and certified robustness justify R2ET's stability in all attacks. Extensive experiments across various data modalities and model architectures show that R2ET achieves superior stability against stealthy attacks, and generalizes effectively across different explanation methods.



## **31. Imperceptible Adversarial Attacks on Point Clouds Guided by Point-to-Surface Field**

cs.CV

Accepted by ICASSP 2025

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2412.19015v1) [paper-pdf](http://arxiv.org/pdf/2412.19015v1)

**Authors**: Keke Tang, Weiyao Ke, Weilong Peng, Xiaofei Wang, Ziyong Du, Zhize Wu, Peican Zhu, Zhihong Tian

**Abstract**: Adversarial attacks on point clouds are crucial for assessing and improving the adversarial robustness of 3D deep learning models. Traditional solutions strictly limit point displacement during attacks, making it challenging to balance imperceptibility with adversarial effectiveness. In this paper, we attribute the inadequate imperceptibility of adversarial attacks on point clouds to deviations from the underlying surface. To address this, we introduce a novel point-to-surface (P2S) field that adjusts adversarial perturbation directions by dragging points back to their original underlying surface. Specifically, we use a denoising network to learn the gradient field of the logarithmic density function encoding the shape's surface, and apply a distance-aware adjustment to perturbation directions during attacks, thereby enhancing imperceptibility. Extensive experiments show that adversarial attacks guided by our P2S field are more imperceptible, outperforming state-of-the-art methods.



## **32. Bridging Interpretability and Robustness Using LIME-Guided Model Refinement**

cs.LG

10 pages, 15 figures

**SubmitDate**: 2024-12-25    [abs](http://arxiv.org/abs/2412.18952v1) [paper-pdf](http://arxiv.org/pdf/2412.18952v1)

**Authors**: Navid Nayyem, Abdullah Rakin, Longwei Wang

**Abstract**: This paper explores the intricate relationship between interpretability and robustness in deep learning models. Despite their remarkable performance across various tasks, deep learning models often exhibit critical vulnerabilities, including susceptibility to adversarial attacks, over-reliance on spurious correlations, and a lack of transparency in their decision-making processes. To address these limitations, we propose a novel framework that leverages Local Interpretable Model-Agnostic Explanations (LIME) to systematically enhance model robustness. By identifying and mitigating the influence of irrelevant or misleading features, our approach iteratively refines the model, penalizing reliance on these features during training. Empirical evaluations on multiple benchmark datasets demonstrate that LIME-guided refinement not only improves interpretability but also significantly enhances resistance to adversarial perturbations and generalization to out-of-distribution data.



## **33. Improving Integrated Gradient-based Transferable Adversarial Examples by Refining the Integration Path**

cs.CR

Accepted by AAAI 2025

**SubmitDate**: 2024-12-25    [abs](http://arxiv.org/abs/2412.18844v1) [paper-pdf](http://arxiv.org/pdf/2412.18844v1)

**Authors**: Yuchen Ren, Zhengyu Zhao, Chenhao Lin, Bo Yang, Lu Zhou, Zhe Liu, Chao Shen

**Abstract**: Transferable adversarial examples are known to cause threats in practical, black-box attack scenarios. A notable approach to improving transferability is using integrated gradients (IG), originally developed for model interpretability. In this paper, we find that existing IG-based attacks have limited transferability due to their naive adoption of IG in model interpretability. To address this limitation, we focus on the IG integration path and refine it in three aspects: multiplicity, monotonicity, and diversity, supported by theoretical analyses. We propose the Multiple Monotonic Diversified Integrated Gradients (MuMoDIG) attack, which can generate highly transferable adversarial examples on different CNN and ViT models and defenses. Experiments validate that MuMoDIG outperforms the latest IG-based attack by up to 37.3\% and other state-of-the-art attacks by 8.4\%. In general, our study reveals that migrating established techniques to improve transferability may require non-trivial efforts. Code is available at \url{https://github.com/RYC-98/MuMoDIG}.



## **34. Distortion-Aware Adversarial Attacks on Bounding Boxes of Object Detectors**

cs.CV

**SubmitDate**: 2024-12-25    [abs](http://arxiv.org/abs/2412.18815v1) [paper-pdf](http://arxiv.org/pdf/2412.18815v1)

**Authors**: Pham Phuc, Son Vuong, Khang Nguyen, Tuan Dang

**Abstract**: Deep learning-based object detection has become ubiquitous in the last decade due to its high accuracy in many real-world applications. With this growing trend, these models are interested in being attacked by adversaries, with most of the results being on classifiers, which do not match the context of practical object detection. In this work, we propose a novel method to fool object detectors, expose the vulnerability of state-of-the-art detectors, and promote later works to build more robust detectors to adversarial examples. Our method aims to generate adversarial images by perturbing object confidence scores during training, which is crucial in predicting confidence for each class in the testing phase. Herein, we provide a more intuitive technique to embed additive noises based on detected objects' masks and the training loss with distortion control over the original image by leveraging the gradient of iterative images. To verify the proposed method, we perform adversarial attacks against different object detectors, including the most recent state-of-the-art models like YOLOv8, Faster R-CNN, RetinaNet, and Swin Transformer. We also evaluate our technique on MS COCO 2017 and PASCAL VOC 2012 datasets and analyze the trade-off between success attack rate and image distortion. Our experiments show that the achievable success attack rate is up to $100$\% and up to $98$\% when performing white-box and black-box attacks, respectively. The source code and relevant documentation for this work are available at the following link: https://github.com/anonymous20210106/attack_detector



## **35. Protective Perturbations against Unauthorized Data Usage in Diffusion-based Image Generation**

cs.CV

**SubmitDate**: 2024-12-25    [abs](http://arxiv.org/abs/2412.18791v1) [paper-pdf](http://arxiv.org/pdf/2412.18791v1)

**Authors**: Sen Peng, Jijia Yang, Mingyue Wang, Jianfei He, Xiaohua Jia

**Abstract**: Diffusion-based text-to-image models have shown immense potential for various image-related tasks. However, despite their prominence and popularity, customizing these models using unauthorized data also brings serious privacy and intellectual property issues. Existing methods introduce protective perturbations based on adversarial attacks, which are applied to the customization samples. In this systematization of knowledge, we present a comprehensive survey of protective perturbation methods designed to prevent unauthorized data usage in diffusion-based image generation. We establish the threat model and categorize the downstream tasks relevant to these methods, providing a detailed analysis of their designs. We also propose a completed evaluation framework for these perturbation techniques, aiming to advance research in this field.



## **36. Attack-in-the-Chain: Bootstrapping Large Language Models for Attacks Against Black-box Neural Ranking Models**

cs.IR

Accepted by AAAI25

**SubmitDate**: 2024-12-25    [abs](http://arxiv.org/abs/2412.18770v1) [paper-pdf](http://arxiv.org/pdf/2412.18770v1)

**Authors**: Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng

**Abstract**: Neural ranking models (NRMs) have been shown to be highly effective in terms of retrieval performance. Unfortunately, they have also displayed a higher degree of sensitivity to attacks than previous generation models. To help expose and address this lack of robustness, we introduce a novel ranking attack framework named Attack-in-the-Chain, which tracks interactions between large language models (LLMs) and NRMs based on chain-of-thought (CoT) prompting to generate adversarial examples under black-box settings. Our approach starts by identifying anchor documents with higher ranking positions than the target document as nodes in the reasoning chain. We then dynamically assign the number of perturbation words to each node and prompt LLMs to execute attacks. Finally, we verify the attack performance of all nodes at each reasoning step and proceed to generate the next reasoning step. Empirical results on two web search benchmarks show the effectiveness of our method.



## **37. Token Highlighter: Inspecting and Mitigating Jailbreak Prompts for Large Language Models**

cs.CR

Accepted by AAAI 2025. Project page:  https://huggingface.co/spaces/TrustSafeAI/Token-Highlighter

**SubmitDate**: 2024-12-25    [abs](http://arxiv.org/abs/2412.18171v2) [paper-pdf](http://arxiv.org/pdf/2412.18171v2)

**Authors**: Xiaomeng Hu, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Large Language Models (LLMs) are increasingly being integrated into services such as ChatGPT to provide responses to user queries. To mitigate potential harm and prevent misuse, there have been concerted efforts to align the LLMs with human values and legal compliance by incorporating various techniques, such as Reinforcement Learning from Human Feedback (RLHF), into the training of the LLMs. However, recent research has exposed that even aligned LLMs are susceptible to adversarial manipulations known as Jailbreak Attacks. To address this challenge, this paper proposes a method called Token Highlighter to inspect and mitigate the potential jailbreak threats in the user query. Token Highlighter introduced a concept called Affirmation Loss to measure the LLM's willingness to answer the user query. It then uses the gradient of Affirmation Loss for each token in the user query to locate the jailbreak-critical tokens. Further, Token Highlighter exploits our proposed Soft Removal technique to mitigate the jailbreak effects of critical tokens via shrinking their token embeddings. Experimental results on two aligned LLMs (LLaMA-2 and Vicuna-V1.5) demonstrate that the proposed method can effectively defend against a variety of Jailbreak Attacks while maintaining competent performance on benign questions of the AlpacaEval benchmark. In addition, Token Highlighter is a cost-effective and interpretable defense because it only needs to query the protected LLM once to compute the Affirmation Loss and can highlight the critical tokens upon refusal.



## **38. Evaluating the Adversarial Robustness of Detection Transformers**

cs.CV

**SubmitDate**: 2024-12-25    [abs](http://arxiv.org/abs/2412.18718v1) [paper-pdf](http://arxiv.org/pdf/2412.18718v1)

**Authors**: Amirhossein Nazeri, Chunheng Zhao, Pierluigi Pisu

**Abstract**: Robust object detection is critical for autonomous driving and mobile robotics, where accurate detection of vehicles, pedestrians, and obstacles is essential for ensuring safety. Despite the advancements in object detection transformers (DETRs), their robustness against adversarial attacks remains underexplored. This paper presents a comprehensive evaluation of DETR model and its variants under both white-box and black-box adversarial attacks, using the MS-COCO and KITTI datasets to cover general and autonomous driving scenarios. We extend prominent white-box attack methods (FGSM, PGD, and CW) to assess DETR vulnerability, demonstrating that DETR models are significantly susceptible to adversarial attacks, similar to traditional CNN-based detectors. Our extensive transferability analysis reveals high intra-network transferability among DETR variants, but limited cross-network transferability to CNN-based models. Additionally, we propose a novel untargeted attack designed specifically for DETR, exploiting its intermediate loss functions to induce misclassification with minimal perturbations. Visualizations of self-attention feature maps provide insights into how adversarial attacks affect the internal representations of DETR models. These findings reveal critical vulnerabilities in detection transformers under standard adversarial attacks, emphasizing the need for future research to enhance the robustness of transformer-based object detectors in safety-critical applications.



## **39. SurvAttack: Black-Box Attack On Survival Models through Ontology-Informed EHR Perturbation**

cs.LG

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.18706v1) [paper-pdf](http://arxiv.org/pdf/2412.18706v1)

**Authors**: Mohsen Nayebi Kerdabadi, Arya Hadizadeh Moghaddam, Bin Liu, Mei Liu, Zijun Yao

**Abstract**: Survival analysis (SA) models have been widely studied in mining electronic health records (EHRs), particularly in forecasting the risk of critical conditions for prioritizing high-risk patients. However, their vulnerability to adversarial attacks is much less explored in the literature. Developing black-box perturbation algorithms and evaluating their impact on state-of-the-art survival models brings two benefits to medical applications. First, it can effectively evaluate the robustness of models in pre-deployment testing. Also, exploring how subtle perturbations would result in significantly different outcomes can provide counterfactual insights into the clinical interpretation of model prediction. In this work, we introduce SurvAttack, a novel black-box adversarial attack framework leveraging subtle clinically compatible, and semantically consistent perturbations on longitudinal EHRs to degrade survival models' predictive performance. We specifically develop a greedy algorithm to manipulate medical codes with various adversarial actions throughout a patient's medical history. Then, these adversarial actions are prioritized using a composite scoring strategy based on multi-aspect perturbation quality, including saliency, perturbation stealthiness, and clinical meaningfulness. The proposed adversarial EHR perturbation algorithm is then used in an efficient SA-specific strategy to attack a survival model when estimating the temporal ranking of survival urgency for patients. To demonstrate the significance of our work, we conduct extensive experiments, including baseline comparisons, explainability analysis, and case studies. The experimental results affirm our research's effectiveness in illustrating the vulnerabilities of patient survival models, model interpretation, and ultimately contributing to healthcare quality.



## **40. Adversarial Attack Against Images Classification based on Generative Adversarial Networks**

cs.CV

7 pages, 6 figures

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.16662v2) [paper-pdf](http://arxiv.org/pdf/2412.16662v2)

**Authors**: Yahe Yang

**Abstract**: Adversarial attacks on image classification systems have always been an important problem in the field of machine learning, and generative adversarial networks (GANs), as popular models in the field of image generation, have been widely used in various novel scenarios due to their powerful generative capabilities. However, with the popularity of generative adversarial networks, the misuse of fake image technology has raised a series of security problems, such as malicious tampering with other people's photos and videos, and invasion of personal privacy. Inspired by the generative adversarial networks, this work proposes a novel adversarial attack method, aiming to gain insight into the weaknesses of the image classification system and improve its anti-attack ability. Specifically, the generative adversarial networks are used to generate adversarial samples with small perturbations but enough to affect the decision-making of the classifier, and the adversarial samples are generated through the adversarial learning of the training generator and the classifier. From extensive experiment analysis, we evaluate the effectiveness of the method on a classical image classification dataset, and the results show that our model successfully deceives a variety of advanced classifiers while maintaining the naturalness of adversarial samples.



## **41. An Empirical Analysis of Federated Learning Models Subject to Label-Flipping Adversarial Attack**

cs.LG

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.18507v1) [paper-pdf](http://arxiv.org/pdf/2412.18507v1)

**Authors**: Kunal Bhatnagar, Sagana Chattanathan, Angela Dang, Bhargav Eranki, Ronnit Rana, Charan Sridhar, Siddharth Vedam, Angie Yao, Mark Stamp

**Abstract**: In this paper, we empirically analyze adversarial attacks on selected federated learning models. The specific learning models considered are Multinominal Logistic Regression (MLR), Support Vector Classifier (SVC), Multilayer Perceptron (MLP), Convolution Neural Network (CNN), %Recurrent Neural Network (RNN), Random Forest, XGBoost, and Long Short-Term Memory (LSTM). For each model, we simulate label-flipping attacks, experimenting extensively with 10 federated clients and 100 federated clients. We vary the percentage of adversarial clients from 10% to 100% and, simultaneously, the percentage of labels flipped by each adversarial client is also varied from 10% to 100%. Among other results, we find that models differ in their inherent robustness to the two vectors in our label-flipping attack, i.e., the percentage of adversarial clients, and the percentage of labels flipped by each adversarial client. We discuss the potential practical implications of our results.



## **42. Prompted Contextual Vectors for Spear-Phishing Detection**

cs.LG

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2402.08309v3) [paper-pdf](http://arxiv.org/pdf/2402.08309v3)

**Authors**: Daniel Nahmias, Gal Engelberg, Dan Klein, Asaf Shabtai

**Abstract**: Spear-phishing attacks present a significant security challenge, with large language models (LLMs) escalating the threat by generating convincing emails and facilitating target reconnaissance. To address this, we propose a detection approach based on a novel document vectorization method that utilizes an ensemble of LLMs to create representation vectors. By prompting LLMs to reason and respond to human-crafted questions, we quantify the presence of common persuasion principles in the email's content, producing prompted contextual document vectors for a downstream supervised machine learning model. We evaluate our method using a unique dataset generated by a proprietary system that automates target reconnaissance and spear-phishing email creation. Our method achieves a 91\% F1 score in identifying LLM-generated spear-phishing emails, with the training set comprising only traditional phishing and benign emails. Key contributions include a novel document vectorization method utilizing LLM reasoning, a publicly available dataset of high-quality spear-phishing emails, and the demonstrated effectiveness of our method in detecting such emails. This methodology can be utilized for various document classification tasks, particularly in adversarial problem domains.



## **43. Unveiling the Threat of Fraud Gangs to Graph Neural Networks: Multi-Target Graph Injection Attacks against GNN-Based Fraud Detectors**

cs.LG

19 pages, 5 figures, 12 tables, The 39th AAAI Conference on  Artificial Intelligence (AAAI 2025)

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.18370v1) [paper-pdf](http://arxiv.org/pdf/2412.18370v1)

**Authors**: Jinhyeok Choi, Heehyeon Kim, Joyce Jiyoung Whang

**Abstract**: Graph neural networks (GNNs) have emerged as an effective tool for fraud detection, identifying fraudulent users, and uncovering malicious behaviors. However, attacks against GNN-based fraud detectors and their risks have rarely been studied, thereby leaving potential threats unaddressed. Recent findings suggest that frauds are increasingly organized as gangs or groups. In this work, we design attack scenarios where fraud gangs aim to make their fraud nodes misclassified as benign by camouflaging their illicit activities in collusion. Based on these scenarios, we study adversarial attacks against GNN-based fraud detectors by simulating attacks of fraud gangs in three real-world fraud cases: spam reviews, fake news, and medical insurance frauds. We define these attacks as multi-target graph injection attacks and propose MonTi, a transformer-based Multi-target one-Time graph injection attack model. MonTi simultaneously generates attributes and edges of all attack nodes with a transformer encoder, capturing interdependencies between attributes and edges more effectively than most existing graph injection attack methods that generate these elements sequentially. Additionally, MonTi adaptively allocates the degree budget for each attack node to explore diverse injection structures involving target, candidate, and attack nodes, unlike existing methods that fix the degree budget across all attack nodes. Experiments show that MonTi outperforms the state-of-the-art graph injection attack methods on five real-world graphs.



## **44. Hypergraph Attacks via Injecting Homogeneous Nodes into Elite Hyperedges**

cs.LG

9 pages, The 39th Annual AAAI Conference on Artificial  Intelligence(2025)

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.18365v1) [paper-pdf](http://arxiv.org/pdf/2412.18365v1)

**Authors**: Meixia He, Peican Zhu, Keke Tang, Yangming Guo

**Abstract**: Recent studies have shown that Hypergraph Neural Networks (HGNNs) are vulnerable to adversarial attacks. Existing approaches focus on hypergraph modification attacks guided by gradients, overlooking node spanning in the hypergraph and the group identity of hyperedges, thereby resulting in limited attack performance and detectable attacks. In this manuscript, we present a novel framework, i.e., Hypergraph Attacks via Injecting Homogeneous Nodes into Elite Hyperedges (IE-Attack), to tackle these challenges. Initially, utilizing the node spanning in the hypergraph, we propose the elite hyperedges sampler to identify hyperedges to be injected. Subsequently, a node generator utilizing Kernel Density Estimation (KDE) is proposed to generate the homogeneous node with the group identity of hyperedges. Finally, by injecting the homogeneous node into elite hyperedges, IE-Attack improves the attack performance and enhances the imperceptibility of attacks. Extensive experiments are conducted on five authentic datasets to validate the effectiveness of IE-Attack and the corresponding superiority to state-of-the-art methods.



## **45. Level Up with ML Vulnerability Identification: Leveraging Domain Constraints in Feature Space for Robust Android Malware Detection**

cs.LG

The paper was accepted by ACM Transactions on Privacy and Security on  2 December 2024

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2205.15128v4) [paper-pdf](http://arxiv.org/pdf/2205.15128v4)

**Authors**: Hamid Bostani, Zhengyu Zhao, Zhuoran Liu, Veelasha Moonsamy

**Abstract**: Machine Learning (ML) promises to enhance the efficacy of Android Malware Detection (AMD); however, ML models are vulnerable to realistic evasion attacks--crafting realizable Adversarial Examples (AEs) that satisfy Android malware domain constraints. To eliminate ML vulnerabilities, defenders aim to identify susceptible regions in the feature space where ML models are prone to deception. The primary approach to identifying vulnerable regions involves investigating realizable AEs, but generating these feasible apps poses a challenge. For instance, previous work has relied on generating either feature-space norm-bounded AEs or problem-space realizable AEs in adversarial hardening. The former is efficient but lacks full coverage of vulnerable regions while the latter can uncover these regions by satisfying domain constraints but is known to be time-consuming. To address these limitations, we propose an approach to facilitate the identification of vulnerable regions. Specifically, we introduce a new interpretation of Android domain constraints in the feature space, followed by a novel technique that learns them. Our empirical evaluations across various evasion attacks indicate effective detection of AEs using learned domain constraints, with an average of 89.6%. Furthermore, extensive experiments on different Android malware detectors demonstrate that utilizing our learned domain constraints in Adversarial Training (AT) outperforms other AT-based defenses that rely on norm-bounded AEs or state-of-the-art non-uniform perturbations. Finally, we show that retraining a malware detector with a wide variety of feature-space realizable AEs results in a 77.9% robustness improvement against realizable AEs generated by unknown problem-space transformations, with up to 70x faster training than using problem-space realizable AEs.



## **46. Can Large Language Models Improve the Adversarial Robustness of Graph Neural Networks?**

cs.LG

accepted by KDD 2025

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2408.08685v3) [paper-pdf](http://arxiv.org/pdf/2408.08685v3)

**Authors**: Zhongjian Zhang, Xiao Wang, Huichi Zhou, Yue Yu, Mengmei Zhang, Cheng Yang, Chuan Shi

**Abstract**: Graph neural networks (GNNs) are vulnerable to adversarial attacks, especially for topology perturbations, and many methods that improve the robustness of GNNs have received considerable attention. Recently, we have witnessed the significant success of large language models (LLMs), leading many to explore the great potential of LLMs on GNNs. However, they mainly focus on improving the performance of GNNs by utilizing LLMs to enhance the node features. Therefore, we ask: Will the robustness of GNNs also be enhanced with the powerful understanding and inference capabilities of LLMs? By presenting the empirical results, we find that despite that LLMs can improve the robustness of GNNs, there is still an average decrease of 23.1% in accuracy, implying that the GNNs remain extremely vulnerable against topology attacks. Therefore, another question is how to extend the capabilities of LLMs on graph adversarial robustness. In this paper, we propose an LLM-based robust graph structure inference framework, LLM4RGNN, which distills the inference capabilities of GPT-4 into a local LLM for identifying malicious edges and an LM-based edge predictor for finding missing important edges, so as to recover a robust graph structure. Extensive experiments demonstrate that LLM4RGNN consistently improves the robustness across various GNNs. Even in some cases where the perturbation ratio increases to 40%, the accuracy of GNNs is still better than that on the clean graph. The source code can be found in https://github.com/zhongjian-zhang/LLM4RGNN.



## **47. On the Effectiveness of Adversarial Training on Malware Classifiers**

cs.LG

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.18218v1) [paper-pdf](http://arxiv.org/pdf/2412.18218v1)

**Authors**: Hamid Bostani, Jacopo Cortellazzi, Daniel Arp, Fabio Pierazzi, Veelasha Moonsamy, Lorenzo Cavallaro

**Abstract**: Adversarial Training (AT) has been widely applied to harden learning-based classifiers against adversarial evasive attacks. However, its effectiveness in identifying and strengthening vulnerable areas of the model's decision space while maintaining high performance on clean data of malware classifiers remains an under-explored area. In this context, the robustness that AT achieves has often been assessed against unrealistic or weak adversarial attacks, which negatively affect performance on clean data and are arguably no longer threats. Previous work seems to suggest robustness is a task-dependent property of AT. We instead argue it is a more complex problem that requires exploring AT and the intertwined roles played by certain factors within data, feature representations, classifiers, and robust optimization settings, as well as proper evaluation factors, such as the realism of evasion attacks, to gain a true sense of AT's effectiveness. In our paper, we address this gap by systematically exploring the role such factors have in hardening malware classifiers through AT. Contrary to recent prior work, a key observation of our research and extensive experiments confirm the hypotheses that all such factors influence the actual effectiveness of AT, as demonstrated by the varying degrees of success from our empirical analysis. We identify five evaluation pitfalls that affect state-of-the-art studies and summarize our insights in ten takeaways to draw promising research directions toward better understanding the factors' settings under which adversarial training works at best.



## **48. Robustness-aware Automatic Prompt Optimization**

cs.CL

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.18196v1) [paper-pdf](http://arxiv.org/pdf/2412.18196v1)

**Authors**: Zeru Shi, Zhenting Wang, Yongye Su, Weidi Luo, Fan Yang, Yongfeng Zhang

**Abstract**: The performance of Large Language Models (LLMs) is based on the quality of the prompts and the semantic and structural integrity information of the input data. However, current prompt generation methods primarily focus on generating prompts for clean input data, often overlooking the impact of perturbed inputs on prompt performance. To address this limitation, we propose BATprompt (By Adversarial Training prompt), a novel method for prompt generation designed to withstand input perturbations (such as typos in the input). Inspired by adversarial training techniques, BATprompt demonstrates strong performance on a variety of perturbed tasks through a two-step process: adversarial perturbation and iterative optimization on unperturbed input via LLM. Unlike conventional adversarial attack methods, BATprompt avoids reliance on real gradients or model parameters. Instead, it leverages the advanced reasoning, language understanding and self reflection capabilities of LLMs to simulate gradients, guiding the generation of adversarial perturbations and optimizing prompt performance. In our experiments, we evaluate BATprompt on multiple datasets across both language understanding and generation tasks. The results indicate that BATprompt outperforms existing prompt generation methods, delivering superior robustness and performance under diverse perturbation scenarios.



## **49. Sparse-PGD: A Unified Framework for Sparse Adversarial Perturbations Generation**

cs.LG

Extended version. Codes are available at  https://github.com/CityU-MLO/sPGD

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2405.05075v3) [paper-pdf](http://arxiv.org/pdf/2405.05075v3)

**Authors**: Xuyang Zhong, Chen Liu

**Abstract**: This work studies sparse adversarial perturbations, including both unstructured and structured ones. We propose a framework based on a white-box PGD-like attack method named Sparse-PGD to effectively and efficiently generate such perturbations. Furthermore, we combine Sparse-PGD with a black-box attack to comprehensively and more reliably evaluate the models' robustness against unstructured and structured sparse adversarial perturbations. Moreover, the efficiency of Sparse-PGD enables us to conduct adversarial training to build robust models against various sparse perturbations. Extensive experiments demonstrate that our proposed attack algorithm exhibits strong performance in different scenarios. More importantly, compared with other robust models, our adversarially trained model demonstrates state-of-the-art robustness against various sparse attacks.



## **50. AEIOU: A Unified Defense Framework against NSFW Prompts in Text-to-Image Models**

cs.CR

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.18123v1) [paper-pdf](http://arxiv.org/pdf/2412.18123v1)

**Authors**: Yiming Wang, Jiahao Chen, Qingming Li, Xing Yang, Shouling Ji

**Abstract**: As text-to-image (T2I) models continue to advance and gain widespread adoption, their associated safety issues are becoming increasingly prominent. Malicious users often exploit these models to generate Not-Safe-for-Work (NSFW) images using harmful or adversarial prompts, highlighting the critical need for robust safeguards to ensure the integrity and compliance of model outputs. Current internal safeguards frequently degrade image quality, while external detection methods often suffer from low accuracy and inefficiency.   In this paper, we introduce AEIOU, a defense framework that is Adaptable, Efficient, Interpretable, Optimizable, and Unified against NSFW prompts in T2I models. AEIOU extracts NSFW features from the hidden states of the model's text encoder, utilizing the separable nature of these features to detect NSFW prompts. The detection process is efficient, requiring minimal inference time. AEIOU also offers real-time interpretation of results and supports optimization through data augmentation techniques. The framework is versatile, accommodating various T2I architectures. Our extensive experiments show that AEIOU significantly outperforms both commercial and open-source moderation tools, achieving over 95% accuracy across all datasets and improving efficiency by at least tenfold. It effectively counters adaptive attacks and excels in few-shot and multi-label scenarios.



