# Latest Adversarial Attack Papers
**update at 2023-08-30 11:18:19**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Masquerade: Simple and Lightweight Transaction Reordering Mitigation in Blockchains**

cs.CR

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2308.15347v1) [paper-pdf](http://arxiv.org/pdf/2308.15347v1)

**Authors**: Arti Vedula, Shaileshh Bojja Venkatakrishnan, Abhishek Gupta

**Abstract**: Blockchains offer strong security gurarantees, but cannot protect users against the ordering of transactions. Players such as miners, bots and validators can reorder various transactions and reap significant profits, called the Maximal Extractable Value (MEV). In this paper, we propose an MEV aware protocol design called Masquerade, and show that it will increase user satisfaction and confidence in the system. We propose a strict per-transaction level of ordering to ensure that a transaction is committed either way even if it is revealed. In this protocol, we introduce the notion of a "token" to mitigate the actions taken by an adversary in an attack scenario. Such tokens can be purchased voluntarily by users, who can then choose to include the token numbers in their transactions. If the users include the token in their transactions, then our protocol requires the block-builder to order the transactions strictly according to token numbers. We show through extensive simulations that this reduces the probability that the adversaries can benefit from MEV transactions as compared to existing current practices.



## **2. Imperceptible Adversarial Attack on Deep Neural Networks from Image Boundary**

cs.LG

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2308.15344v1) [paper-pdf](http://arxiv.org/pdf/2308.15344v1)

**Authors**: Fahad Alrasheedi, Xin Zhong

**Abstract**: Although Deep Neural Networks (DNNs), such as the convolutional neural networks (CNN) and Vision Transformers (ViTs), have been successfully applied in the field of computer vision, they are demonstrated to be vulnerable to well-sought Adversarial Examples (AEs) that can easily fool the DNNs. The research in AEs has been active, and many adversarial attacks and explanations have been proposed since they were discovered in 2014. The mystery of the AE's existence is still an open question, and many studies suggest that DNN training algorithms have blind spots. The salient objects usually do not overlap with boundaries; hence, the boundaries are not the DNN model's attention. Nevertheless, recent studies show that the boundaries can dominate the behavior of the DNN models. Hence, this study aims to look at the AEs from a different perspective and proposes an imperceptible adversarial attack that systemically attacks the input image boundary for finding the AEs. The experimental results have shown that the proposed boundary attacking method effectively attacks six CNN models and the ViT using only 32% of the input image content (from the boundaries) with an average success rate (SR) of 95.2% and an average peak signal-to-noise ratio of 41.37 dB. Correlation analyses are conducted, including the relation between the adversarial boundary's width and the SR and how the adversarial boundary changes the DNN model's attention. This paper's discoveries can potentially advance the understanding of AEs and provide a different perspective on how AEs can be constructed.



## **3. Longest-chain Attacks: Difficulty Adjustment and Timestamp Verifiability**

cs.CR

A short version appears at MobiHoc23 as a poster

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2308.15312v1) [paper-pdf](http://arxiv.org/pdf/2308.15312v1)

**Authors**: Tzuo Hann Law, Selman Erol, Lewis Tseng

**Abstract**: We study an adversary who attacks a Proof-of-Work (POW) blockchain by selfishly constructing an alternative longest chain. We characterize optimal strategies employed by the adversary when a difficulty adjustment rule al\`a Bitcoin applies. As time (namely the times-tamp specified in each block) in most permissionless POW blockchains is somewhat subjective, we focus on two extreme scenarios: when time is completely verifiable, and when it is completely unverifiable. We conclude that an adversary who faces a difficulty adjustment rule will find a longest-chain attack very challenging when timestamps are verifiable. POW blockchains with frequent difficulty adjustments relative to time reporting flexibility will be substantially more vulnerable to longest-chain attacks. Our main fining provides guidance on the design of difficulty adjustment rules and demonstrates the importance of timestamp verifiability.



## **4. A Classification-Guided Approach for Adversarial Attacks against Neural Machine Translation**

cs.CL

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2308.15246v1) [paper-pdf](http://arxiv.org/pdf/2308.15246v1)

**Authors**: Sahar Sadrizadeh, Ljiljana Dolamic, Pascal Frossard

**Abstract**: Neural Machine Translation (NMT) models have been shown to be vulnerable to adversarial attacks, wherein carefully crafted perturbations of the input can mislead the target model. In this paper, we introduce ACT, a novel adversarial attack framework against NMT systems guided by a classifier. In our attack, the adversary aims to craft meaning-preserving adversarial examples whose translations by the NMT model belong to a different class than the original translations in the target language. Unlike previous attacks, our new approach has a more substantial effect on the translation by altering the overall meaning, which leads to a different class determined by a classifier. To evaluate the robustness of NMT models to this attack, we propose enhancements to existing black-box word-replacement-based attacks by incorporating output translations of the target NMT model and the output logits of a classifier within the attack process. Extensive experiments in various settings, including a comparison with existing untargeted attacks, demonstrate that the proposed attack is considerably more successful in altering the class of the output translation and has more effect on the translation. This new paradigm can show the vulnerabilities of NMT systems by focusing on the class of translation rather than the mere translation quality as studied traditionally.



## **5. Can We Rely on AI?**

math.NA

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2308.15092v1) [paper-pdf](http://arxiv.org/pdf/2308.15092v1)

**Authors**: Desmond J. Higham

**Abstract**: Over the last decade, adversarial attack algorithms have revealed instabilities in deep learning tools. These algorithms raise issues regarding safety, reliability and interpretability in artificial intelligence; especially in high risk settings. From a practical perspective, there has been a war of escalation between those developing attack and defence strategies. At a more theoretical level, researchers have also studied bigger picture questions concerning the existence and computability of attacks. Here we give a brief overview of the topic, focusing on aspects that are likely to be of interest to researchers in applied and computational mathematics.



## **6. Advancing Adversarial Robustness Through Adversarial Logit Update**

cs.LG

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2308.15072v1) [paper-pdf](http://arxiv.org/pdf/2308.15072v1)

**Authors**: Hao Xuan, Peican Zhu, Xingyu Li

**Abstract**: Deep Neural Networks are susceptible to adversarial perturbations. Adversarial training and adversarial purification are among the most widely recognized defense strategies. Although these methods have different underlying logic, both rely on absolute logit values to generate label predictions. In this study, we theoretically analyze the logit difference around successful adversarial attacks from a theoretical point of view and propose a new principle, namely Adversarial Logit Update (ALU), to infer adversarial sample's labels. Based on ALU, we introduce a new classification paradigm that utilizes pre- and post-purification logit differences for model's adversarial robustness boost. Without requiring adversarial or additional data for model training, our clean data synthesis model can be easily applied to various pre-trained models for both adversarial sample detection and ALU-based data classification. Extensive experiments on both CIFAR-10, CIFAR-100, and tiny-ImageNet datasets show that even with simple components, the proposed solution achieves superior robustness performance compared to state-of-the-art methods against a wide range of adversarial attacks. Our python implementation is submitted in our Supplementary document and will be published upon the paper's acceptance.



## **7. Double Public Key Signing Function Oracle Attack on EdDSA Software Implementations**

cs.CR

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2308.15009v1) [paper-pdf](http://arxiv.org/pdf/2308.15009v1)

**Authors**: Sam Grierson, Konstantinos Chalkias, William J Buchanan

**Abstract**: EdDSA is a standardised elliptic curve digital signature scheme introduced to overcome some of the issues prevalent in the more established ECDSA standard. Due to the EdDSA standard specifying that the EdDSA signature be deterministic, if the signing function were to be used as a public key signing oracle for the attacker, the unforgeability notion of security of the scheme can be broken. This paper describes an attack against some of the most popular EdDSA implementations, which results in an adversary recovering the private key used during signing. With this recovered secret key, an adversary can sign arbitrary messages that would be seen as valid by the EdDSA verification function. A list of libraries with vulnerable APIs at the time of publication is provided. Furthermore, this paper provides two suggestions for securing EdDSA signing APIs against this vulnerability while it additionally discusses failed attempts to solve the issue.



## **8. Stealthy Backdoor Attack for Code Models**

cs.CR

18 pages, Under review of IEEE Transactions on Software Engineering

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2301.02496v2) [paper-pdf](http://arxiv.org/pdf/2301.02496v2)

**Authors**: Zhou Yang, Bowen Xu, Jie M. Zhang, Hong Jin Kang, Jieke Shi, Junda He, David Lo

**Abstract**: Code models, such as CodeBERT and CodeT5, offer general-purpose representations of code and play a vital role in supporting downstream automated software engineering tasks. Most recently, code models were revealed to be vulnerable to backdoor attacks. A code model that is backdoor-attacked can behave normally on clean examples but will produce pre-defined malicious outputs on examples injected with triggers that activate the backdoors. Existing backdoor attacks on code models use unstealthy and easy-to-detect triggers. This paper aims to investigate the vulnerability of code models with stealthy backdoor attacks. To this end, we propose AFRAIDOOR (Adversarial Feature as Adaptive Backdoor). AFRAIDOOR achieves stealthiness by leveraging adversarial perturbations to inject adaptive triggers into different inputs. We evaluate AFRAIDOOR on three widely adopted code models (CodeBERT, PLBART and CodeT5) and two downstream tasks (code summarization and method name prediction). We find that around 85% of adaptive triggers in AFRAIDOOR bypass the detection in the defense process. By contrast, only less than 12% of the triggers from previous work bypass the defense. When the defense method is not applied, both AFRAIDOOR and baselines have almost perfect attack success rates. However, once a defense is applied, the success rates of baselines decrease dramatically to 10.47% and 12.06%, while the success rate of AFRAIDOOR are 77.05% and 92.98% on the two tasks. Our finding exposes security weaknesses in code models under stealthy backdoor attacks and shows that the state-of-the-art defense method cannot provide sufficient protection. We call for more research efforts in understanding security threats to code models and developing more effective countermeasures.



## **9. WSAM: Visual Explanations from Style Augmentation as Adversarial Attacker and Their Influence in Image Classification**

cs.CV

8 pages, 10 figures

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2308.14995v1) [paper-pdf](http://arxiv.org/pdf/2308.14995v1)

**Authors**: Felipe Moreno-Vera, Edgar Medina, Jorge Poco

**Abstract**: Currently, style augmentation is capturing attention due to convolutional neural networks (CNN) being strongly biased toward recognizing textures rather than shapes. Most existing styling methods either perform a low-fidelity style transfer or a weak style representation in the embedding vector. This paper outlines a style augmentation algorithm using stochastic-based sampling with noise addition to improving randomization on a general linear transformation for style transfer. With our augmentation strategy, all models not only present incredible robustness against image stylizing but also outperform all previous methods and surpass the state-of-the-art performance for the STL-10 dataset. In addition, we present an analysis of the model interpretations under different style variations. At the same time, we compare comprehensive experiments demonstrating the performance when applied to deep neural architectures in training settings.



## **10. Randomized Line-to-Row Mapping for Low-Overhead Rowhammer Mitigations**

cs.CR

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2308.14907v1) [paper-pdf](http://arxiv.org/pdf/2308.14907v1)

**Authors**: Anish Saxena, Saurav Mathur, Moinuddin Qureshi

**Abstract**: Modern systems mitigate Rowhammer using victim refresh, which refreshes the two neighbours of an aggressor row when it encounters a specified number of activations. Unfortunately, complex attack patterns like Half-Double break victim-refresh, rendering current systems vulnerable. Instead, recently proposed secure Rowhammer mitigations rely on performing mitigative action on the aggressor rather than the victims. Such schemes employ mitigative actions such as row-migration or access-control and include AQUA, SRS, and Blockhammer. While these schemes incur only modest slowdowns at Rowhammer thresholds of few thousand, they incur prohibitive slowdowns (15%-600%) for lower thresholds that are likely in the near future. The goal of our paper is to make secure Rowhammer mitigations practical at such low thresholds.   Our paper provides the key insights that benign application encounter thousands of hot rows (receiving more activations than the threshold) due to the memory mapping, which places spatially proximate lines in the same row to maximize row-buffer hitrate. Unfortunately, this causes row to receive activations for many frequently used lines. We propose Rubix, which breaks the spatial correlation in the line-to-row mapping by using an encrypted address to access the memory, reducing the likelihood of hot rows by 2 to 3 orders of magnitude. To aid row-buffer hits, Rubix randomizes a group of 1-4 lines. We also propose Rubix-D, which dynamically changes the line-to-row mapping. Rubix-D minimizes hot-rows and makes it much harder for an adversary to learn the spatial neighbourhood of a row. Rubix reduces the slowdown of AQUA (from 15% to 1%), SRS (from 60% to 2%), and Blockhammer (from 600% to 3%) while incurring a storage of less than 1 Kilobyte.



## **11. A Stochastic Surveillance Stackelberg Game: Co-Optimizing Defense Placement and Patrol Strategy**

eess.SY

8 pages, 1 figure, jointly submitted to the IEEE Control Systems  Letters and the 2024 American Control Conference

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2308.14714v1) [paper-pdf](http://arxiv.org/pdf/2308.14714v1)

**Authors**: Yohan John, Gilberto Diaz-Garcia, Xiaoming Duan, Jason R. Marden, Francesco Bullo

**Abstract**: Stochastic patrol routing is known to be advantageous in adversarial settings; however, the optimal choice of stochastic routing strategy is dependent on a model of the adversary. Duan et al. formulated a Stackelberg game for the worst-case scenario, i.e., a surveillance agent confronted with an omniscient attacker [IEEE TCNS, 8(2), 769-80, 2021]. In this article, we extend their formulation to accommodate heterogeneous defenses at the various nodes of the graph. We derive an upper bound on the value of the game. We identify methods for computing effective patrol strategies for certain classes of graphs. Finally, we leverage the heterogeneous defense formulation to develop novel defense placement algorithms that complement the patrol strategies.



## **12. Adversarial Attacks on Foundational Vision Models**

cs.CV

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2308.14597v1) [paper-pdf](http://arxiv.org/pdf/2308.14597v1)

**Authors**: Nathan Inkawhich, Gwendolyn McDonald, Ryan Luley

**Abstract**: Rapid progress is being made in developing large, pretrained, task-agnostic foundational vision models such as CLIP, ALIGN, DINOv2, etc. In fact, we are approaching the point where these models do not have to be finetuned downstream, and can simply be used in zero-shot or with a lightweight probing head. Critically, given the complexity of working at this scale, there is a bottleneck where relatively few organizations in the world are executing the training then sharing the models on centralized platforms such as HuggingFace and torch.hub. The goal of this work is to identify several key adversarial vulnerabilities of these models in an effort to make future designs more robust. Intuitively, our attacks manipulate deep feature representations to fool an out-of-distribution (OOD) detector which will be required when using these open-world-aware models to solve closed-set downstream tasks. Our methods reliably make in-distribution (ID) images (w.r.t. a downstream task) be predicted as OOD and vice versa while existing in extremely low-knowledge-assumption threat models. We show our attacks to be potent in whitebox and blackbox settings, as well as when transferred across foundational model types (e.g., attack DINOv2 with CLIP)! This work is only just the beginning of a long journey towards adversarially robust foundational vision models.



## **13. ReMAV: Reward Modeling of Autonomous Vehicles for Finding Likely Failure Events**

cs.AI

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2308.14550v1) [paper-pdf](http://arxiv.org/pdf/2308.14550v1)

**Authors**: Aizaz Sharif, Dusica Marijan

**Abstract**: Autonomous vehicles are advanced driving systems that are well known for being vulnerable to various adversarial attacks, compromising the vehicle's safety, and posing danger to other road users. Rather than actively training complex adversaries by interacting with the environment, there is a need to first intelligently find and reduce the search space to only those states where autonomous vehicles are found less confident. In this paper, we propose a blackbox testing framework ReMAV using offline trajectories first to analyze the existing behavior of autonomous vehicles and determine appropriate thresholds for finding the probability of failure events. Our reward modeling technique helps in creating a behavior representation that allows us to highlight regions of likely uncertain behavior even when the baseline autonomous vehicle is performing well. This approach allows for more efficient testing without the need for computational and inefficient active adversarial learning techniques. We perform our experiments in a high-fidelity urban driving environment using three different driving scenarios containing single and multi-agent interactions. Our experiment shows 35%, 23%, 48%, and 50% increase in occurrences of vehicle collision, road objects collision, pedestrian collision, and offroad steering events respectively by the autonomous vehicle under test, demonstrating a significant increase in failure events. We also perform a comparative analysis with prior testing frameworks and show that they underperform in terms of training-testing efficiency, finding total infractions, and simulation steps to identify the first failure compared to our approach. The results show that the proposed framework can be used to understand existing weaknesses of the autonomous vehicles under test in order to only attack those regions, starting with the simplistic perturbation models.



## **14. Efficient Decision-based Black-box Patch Attacks on Video Recognition**

cs.CV

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2303.11917v2) [paper-pdf](http://arxiv.org/pdf/2303.11917v2)

**Authors**: Kaixun Jiang, Zhaoyu Chen, Hao Huang, Jiafeng Wang, Dingkang Yang, Bo Li, Yan Wang, Wenqiang Zhang

**Abstract**: Although Deep Neural Networks (DNNs) have demonstrated excellent performance, they are vulnerable to adversarial patches that introduce perceptible and localized perturbations to the input. Generating adversarial patches on images has received much attention, while adversarial patches on videos have not been well investigated. Further, decision-based attacks, where attackers only access the predicted hard labels by querying threat models, have not been well explored on video models either, even if they are practical in real-world video recognition scenes. The absence of such studies leads to a huge gap in the robustness assessment for video models. To bridge this gap, this work first explores decision-based patch attacks on video models. We analyze that the huge parameter space brought by videos and the minimal information returned by decision-based models both greatly increase the attack difficulty and query burden. To achieve a query-efficient attack, we propose a spatial-temporal differential evolution (STDE) framework. First, STDE introduces target videos as patch textures and only adds patches on keyframes that are adaptively selected by temporal difference. Second, STDE takes minimizing the patch area as the optimization objective and adopts spatialtemporal mutation and crossover to search for the global optimum without falling into the local optimum. Experiments show STDE has demonstrated state-of-the-art performance in terms of threat, efficiency and imperceptibility. Hence, STDE has the potential to be a powerful tool for evaluating the robustness of video recognition models.



## **15. Mitigating the source-side channel vulnerability by characterization of photon statistics**

quant-ph

Comments and suggestions are welcomed

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2308.14402v1) [paper-pdf](http://arxiv.org/pdf/2308.14402v1)

**Authors**: Tanya Sharma, Ayan Biswas, Jayanth Ramakrishnan, Pooja Chandravanshi, Ravindra P. Singh

**Abstract**: Quantum key distribution (QKD) theoretically offers unconditional security. Unfortunately, the gap between theory and practice threatens side-channel attacks on practical QKD systems. Many well-known QKD protocols use weak coherent laser pulses to encode the quantum information. These sources differ from ideal single photon sources and follow Poisson statistics. Many protocols, such as decoy state and coincidence detection protocols, rely on monitoring the photon statistics to detect any information leakage. The accurate measurement and characterization of photon statistics enable the detection of adversarial attacks and the estimation of secure key rates, strengthening the overall security of the QKD system. We have rigorously characterized our source to estimate the mean photon number employing multiple detectors for comparison against measurements made with a single detector. Furthermore, we have also studied intensity fluctuations to help identify and mitigate any potential information leakage due to state preparation flaws. We aim to bridge the gap between theory and practice to achieve information-theoretic security.



## **16. QEVSEC: Quick Electric Vehicle SEcure Charging via Dynamic Wireless Power Transfer**

cs.CR

6 pages, conference

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2205.10292v3) [paper-pdf](http://arxiv.org/pdf/2205.10292v3)

**Authors**: Tommaso Bianchi, Surudhi Asokraj, Alessandro Brighente, Mauro Conti, Radha Poovendran

**Abstract**: Dynamic Wireless Power Transfer (DWPT) can be used for on-demand recharging of Electric Vehicles (EV) while driving. However, DWPT raises numerous security and privacy concerns. Recently, researchers demonstrated that DWPT systems are vulnerable to adversarial attacks. In an EV charging scenario, an attacker can prevent the authorized customer from charging, obtain a free charge by billing a victim user and track a target vehicle. State-of-the-art authentication schemes relying on centralized solutions are either vulnerable to various attacks or have high computational complexity, making them unsuitable for a dynamic scenario. In this paper, we propose Quick Electric Vehicle SEcure Charging (QEVSEC), a novel, secure, and efficient authentication protocol for the dynamic charging of EVs. Our idea for QEVSEC originates from multiple vulnerabilities we found in the state-of-the-art protocol that allows tracking of user activity and is susceptible to replay attacks. Based on these observations, the proposed protocol solves these issues and achieves lower computational complexity by using only primitive cryptographic operations in a very short message exchange. QEVSEC provides scalability and a reduced cost in each iteration, thus lowering the impact on the power needed from the grid.



## **17. Hiding Visual Information via Obfuscating Adversarial Perturbations**

cs.CV

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2209.15304v4) [paper-pdf](http://arxiv.org/pdf/2209.15304v4)

**Authors**: Zhigang Su, Dawei Zhou, Nannan Wangu, Decheng Li, Zhen Wang, Xinbo Gao

**Abstract**: Growing leakage and misuse of visual information raise security and privacy concerns, which promotes the development of information protection. Existing adversarial perturbations-based methods mainly focus on the de-identification against deep learning models. However, the inherent visual information of the data has not been well protected. In this work, inspired by the Type-I adversarial attack, we propose an adversarial visual information hiding method to protect the visual privacy of data. Specifically, the method generates obfuscating adversarial perturbations to obscure the visual information of the data. Meanwhile, it maintains the hidden objectives to be correctly predicted by models. In addition, our method does not modify the parameters of the applied model, which makes it flexible for different scenarios. Experimental results on the recognition and classification tasks demonstrate that the proposed method can effectively hide visual information and hardly affect the performances of models. The code is available in the supplementary material.



## **18. Detecting Language Model Attacks with Perplexity**

cs.CL

**SubmitDate**: 2023-08-27    [abs](http://arxiv.org/abs/2308.14132v1) [paper-pdf](http://arxiv.org/pdf/2308.14132v1)

**Authors**: Gabriel Alon, Michael Kamfonas

**Abstract**: A novel hack involving Large Language Models (LLMs) has emerged, leveraging adversarial suffixes to trick models into generating perilous responses. This method has garnered considerable attention from reputable media outlets such as the New York Times and Wired, thereby influencing public perception regarding the security and safety of LLMs. In this study, we advocate the utilization of perplexity as one of the means to recognize such potential attacks. The underlying concept behind these hacks revolves around appending an unusually constructed string of text to a harmful query that would otherwise be blocked. This maneuver confuses the protective mechanisms and tricks the model into generating a forbidden response. Such scenarios could result in providing detailed instructions to a malicious user for constructing explosives or orchestrating a bank heist. Our investigation demonstrates the feasibility of employing perplexity, a prevalent natural language processing metric, to detect these adversarial tactics before generating a forbidden response. By evaluating the perplexity of queries with and without such adversarial suffixes using an open-source LLM, we discovered that nearly 90 percent were above a perplexity of 1000. This contrast underscores the efficacy of perplexity for detecting this type of exploit.



## **19. Fairness and Privacy in Voice Biometrics:A Study of Gender Influences Using wav2vec 2.0**

eess.AS

7 pages

**SubmitDate**: 2023-08-27    [abs](http://arxiv.org/abs/2308.14049v1) [paper-pdf](http://arxiv.org/pdf/2308.14049v1)

**Authors**: Oubaida Chouchane, Michele Panariello, Chiara Galdi, Massimiliano Todisco, Nicholas Evans

**Abstract**: This study investigates the impact of gender information on utility, privacy, and fairness in voice biometric systems, guided by the General Data Protection Regulation (GDPR) mandates, which underscore the need for minimizing the processing and storage of private and sensitive data, and ensuring fairness in automated decision-making systems. We adopt an approach that involves the fine-tuning of the wav2vec 2.0 model for speaker verification tasks, evaluating potential gender-related privacy vulnerabilities in the process. Gender influences during the fine-tuning process were employed to enhance fairness and privacy in order to emphasise or obscure gender information within the speakers' embeddings. Results from VoxCeleb datasets indicate our adversarial model increases privacy against uninformed attacks, yet slightly diminishes speaker verification performance compared to the non-adversarial model. However, the model's efficacy reduces against informed attacks. Analysis of system performance was conducted to identify potential gender biases, thus highlighting the need for further research to understand and improve the delicate interplay between utility, privacy, and equity in voice biometric systems.



## **20. Device-Independent Quantum Key Distribution Based on the Mermin-Peres Magic Square Game**

quant-ph

**SubmitDate**: 2023-08-27    [abs](http://arxiv.org/abs/2308.14037v1) [paper-pdf](http://arxiv.org/pdf/2308.14037v1)

**Authors**: Yi-Zheng Zhen, Yingqiu Mao, Yu-Zhe Zhang, Feihu Xu, Barry C. Sanders

**Abstract**: Device-independent quantum key distribution (DIQKD) is information-theoretically secure against adversaries who possess a scalable quantum computer and who have supplied malicious key-establishment systems; however, the DIQKD key rate is currently too low. Consequently, we devise a DIQKD scheme based on the quantum nonlocal Mermin-Peres magic square game: our scheme asymptotically delivers DIQKD against collective attacks, even with noise. Our scheme outperforms DIQKD using the Clauser-Horne-Shimony-Holt game with respect to the number of game rounds, albeit not number of entangled pairs, provided that both state visibility and detection efficiency are high enough.



## **21. A semantic backdoor attack against Graph Convolutional Networks**

cs.LG

**SubmitDate**: 2023-08-26    [abs](http://arxiv.org/abs/2302.14353v4) [paper-pdf](http://arxiv.org/pdf/2302.14353v4)

**Authors**: Jiazhu Dai, Zhipeng Xiong

**Abstract**: Graph convolutional networks (GCNs) have been very effective in addressing the issue of various graph-structured related tasks. However, recent research has shown that GCNs are vulnerable to a new type of threat called a backdoor attack, where the adversary can inject a hidden backdoor into GCNs so that the attacked model performs well on benign samples, but its prediction will be maliciously changed to the attacker-specified target label if the hidden backdoor is activated by the attacker-defined trigger. A semantic backdoor attack is a new type of backdoor attack on deep neural networks (DNNs), where a naturally occurring semantic feature of samples can serve as a backdoor trigger such that the infected DNN models will misclassify testing samples containing the predefined semantic feature even without the requirement of modifying the testing samples. Since the backdoor trigger is a naturally occurring semantic feature of the samples, semantic backdoor attacks are more imperceptible and pose a new and serious threat. In this paper, we investigate whether such semantic backdoor attacks are possible for GCNs and propose a semantic backdoor attack against GCNs (SBAG) under the context of graph classification to reveal the existence of this security vulnerability in GCNs. SBAG uses a certain type of node in the samples as a backdoor trigger and injects a hidden backdoor into GCN models by poisoning training data. The backdoor will be activated, and the GCN models will give malicious classification results specified by the attacker even on unmodified samples as long as the samples contain enough trigger nodes. We evaluate SBAG on four graph datasets and the experimental results indicate that SBAG is effective.



## **22. Active learning for fast and slow modeling attacks on Arbiter PUFs**

cs.CR

**SubmitDate**: 2023-08-25    [abs](http://arxiv.org/abs/2308.13645v1) [paper-pdf](http://arxiv.org/pdf/2308.13645v1)

**Authors**: Vincent Dumoulin, Wenjing Rao, Natasha Devroye

**Abstract**: Modeling attacks, in which an adversary uses machine learning techniques to model a hardware-based Physically Unclonable Function (PUF) pose a great threat to the viability of these hardware security primitives. In most modeling attacks, a random subset of challenge-response-pairs (CRPs) are used as the labeled data for the machine learning algorithm. Here, for the arbiter-PUF, a delay based PUF which may be viewed as a linear threshold function with random weights (due to manufacturing imperfections), we investigate the role of active learning in Support Vector Machine (SVM) learning. We focus on challenge selection to help SVM algorithm learn ``fast'' and learn ``slow''. Our methods construct challenges rather than relying on a sample pool of challenges as in prior work. Using active learning to learn ``fast'' (less CRPs revealed, higher accuracies) may help manufacturers learn the manufactured PUFs more efficiently, or may form a more powerful attack when the attacker may query the PUF for CRPs at will. Using active learning to select challenges from which learning is ``slow'' (low accuracy despite a large number of revealed CRPs) may provide a basis for slowing down attackers who are limited to overhearing CRPs.



## **23. Unveiling the Role of Message Passing in Dual-Privacy Preservation on GNNs**

cs.LG

CIKM 2023

**SubmitDate**: 2023-08-25    [abs](http://arxiv.org/abs/2308.13513v1) [paper-pdf](http://arxiv.org/pdf/2308.13513v1)

**Authors**: Tianyi Zhao, Hui Hu, Lu Cheng

**Abstract**: Graph Neural Networks (GNNs) are powerful tools for learning representations on graphs, such as social networks. However, their vulnerability to privacy inference attacks restricts their practicality, especially in high-stake domains. To address this issue, privacy-preserving GNNs have been proposed, focusing on preserving node and/or link privacy. This work takes a step back and investigates how GNNs contribute to privacy leakage. Through theoretical analysis and simulations, we identify message passing under structural bias as the core component that allows GNNs to \textit{propagate} and \textit{amplify} privacy leakage. Building upon these findings, we propose a principled privacy-preserving GNN framework that effectively safeguards both node and link privacy, referred to as dual-privacy preservation. The framework comprises three major modules: a Sensitive Information Obfuscation Module that removes sensitive information from node embeddings, a Dynamic Structure Debiasing Module that dynamically corrects the structural bias, and an Adversarial Learning Module that optimizes the privacy-utility trade-off. Experimental results on four benchmark datasets validate the effectiveness of the proposed model in protecting both node and link privacy while preserving high utility for downstream tasks, such as node classification.



## **24. Overcoming Adversarial Attacks for Human-in-the-Loop Applications**

cs.LG

New Frontiers in Adversarial Machine Learning, ICML 2022

**SubmitDate**: 2023-08-25    [abs](http://arxiv.org/abs/2306.05952v2) [paper-pdf](http://arxiv.org/pdf/2306.05952v2)

**Authors**: Ryan McCoppin, Marla Kennedy, Platon Lukyanenko, Sean Kennedy

**Abstract**: Including human analysis has the potential to positively affect the robustness of Deep Neural Networks and is relatively unexplored in the Adversarial Machine Learning literature. Neural network visual explanation maps have been shown to be prone to adversarial attacks. Further research is needed in order to select robust visualizations of explanations for the image analyst to evaluate a given model. These factors greatly impact Human-In-The-Loop (HITL) evaluation tools due to their reliance on adversarial images, including explanation maps and measurements of robustness. We believe models of human visual attention may improve interpretability and robustness of human-machine imagery analysis systems. Our challenge remains, how can HITL evaluation be robust in this adversarial landscape?



## **25. Defensive Few-shot Learning**

cs.CV

Accepted to IEEE Transactions on Pattern Analysis and Machine  Intelligence (TPAMI) 2022

**SubmitDate**: 2023-08-25    [abs](http://arxiv.org/abs/1911.06968v2) [paper-pdf](http://arxiv.org/pdf/1911.06968v2)

**Authors**: Wenbin Li, Lei Wang, Xingxing Zhang, Lei Qi, Jing Huo, Yang Gao, Jiebo Luo

**Abstract**: This paper investigates a new challenging problem called defensive few-shot learning in order to learn a robust few-shot model against adversarial attacks. Simply applying the existing adversarial defense methods to few-shot learning cannot effectively solve this problem. This is because the commonly assumed sample-level distribution consistency between the training and test sets can no longer be met in the few-shot setting. To address this situation, we develop a general defensive few-shot learning (DFSL) framework to answer the following two key questions: (1) how to transfer adversarial defense knowledge from one sample distribution to another? (2) how to narrow the distribution gap between clean and adversarial examples under the few-shot setting? To answer the first question, we propose an episode-based adversarial training mechanism by assuming a task-level distribution consistency to better transfer the adversarial defense knowledge. As for the second question, within each few-shot task, we design two kinds of distribution consistency criteria to narrow the distribution gap between clean and adversarial examples from the feature-wise and prediction-wise perspectives, respectively. Extensive experiments demonstrate that the proposed framework can effectively make the existing few-shot models robust against adversarial attacks. Code is available at https://github.com/WenbinLee/DefensiveFSL.git.



## **26. Feature Unlearning for Pre-trained GANs and VAEs**

cs.CV

**SubmitDate**: 2023-08-25    [abs](http://arxiv.org/abs/2303.05699v3) [paper-pdf](http://arxiv.org/pdf/2303.05699v3)

**Authors**: Saemi Moon, Seunghyuk Cho, Dongwoo Kim

**Abstract**: We tackle the problem of feature unlearning from a pre-trained image generative model: GANs and VAEs. Unlike a common unlearning task where an unlearning target is a subset of the training set, we aim to unlearn a specific feature, such as hairstyle from facial images, from the pre-trained generative models. As the target feature is only presented in a local region of an image, unlearning the entire image from the pre-trained model may result in losing other details in the remaining region of the image. To specify which features to unlearn, we collect randomly generated images that contain the target features. We then identify a latent representation corresponding to the target feature and then use the representation to fine-tune the pre-trained model. Through experiments on MNIST and CelebA datasets, we show that target features are successfully removed while keeping the fidelity of the original models. Further experiments with an adversarial attack show that the unlearned model is more robust under the presence of malicious parties.



## **27. Why Does Little Robustness Help? Understanding and Improving Adversarial Transferability from Surrogate Training**

cs.LG

Accepted by IEEE Symposium on Security and Privacy (Oakland) 2024; 21  pages, 11 figures, 13 tables

**SubmitDate**: 2023-08-25    [abs](http://arxiv.org/abs/2307.07873v4) [paper-pdf](http://arxiv.org/pdf/2307.07873v4)

**Authors**: Yechao Zhang, Shengshan Hu, Leo Yu Zhang, Junyu Shi, Minghui Li, Xiaogeng Liu, Wei Wan, Hai Jin

**Abstract**: Adversarial examples (AEs) for DNNs have been shown to be transferable: AEs that successfully fool white-box surrogate models can also deceive other black-box models with different architectures. Although a bunch of empirical studies have provided guidance on generating highly transferable AEs, many of these findings lack explanations and even lead to inconsistent advice. In this paper, we take a further step towards understanding adversarial transferability, with a particular focus on surrogate aspects. Starting from the intriguing little robustness phenomenon, where models adversarially trained with mildly perturbed adversarial samples can serve as better surrogates, we attribute it to a trade-off between two predominant factors: model smoothness and gradient similarity. Our investigations focus on their joint effects, rather than their separate correlations with transferability. Through a series of theoretical and empirical analyses, we conjecture that the data distribution shift in adversarial training explains the degradation of gradient similarity. Building on these insights, we explore the impacts of data augmentation and gradient regularization on transferability and identify that the trade-off generally exists in the various training mechanisms, thus building a comprehensive blueprint for the regulation mechanism behind transferability. Finally, we provide a general route for constructing better surrogates to boost transferability which optimizes both model smoothness and gradient similarity simultaneously, e.g., the combination of input gradient regularization and sharpness-aware minimization (SAM), validated by extensive experiments. In summary, we call for attention to the united impacts of these two factors for launching effective transfer attacks, rather than optimizing one while ignoring the other, and emphasize the crucial role of manipulating surrogate models.



## **28. Face Encryption via Frequency-Restricted Identity-Agnostic Attacks**

cs.CV

I noticed something missing in the article's description in  subsection 3.2, so I'd like to undo it and re-finalize and describe it

**SubmitDate**: 2023-08-25    [abs](http://arxiv.org/abs/2308.05983v3) [paper-pdf](http://arxiv.org/pdf/2308.05983v3)

**Authors**: Xin Dong, Rui Wang, Siyuan Liang, Aishan Liu, Lihua Jing

**Abstract**: Billions of people are sharing their daily live images on social media everyday. However, malicious collectors use deep face recognition systems to easily steal their biometric information (e.g., faces) from these images. Some studies are being conducted to generate encrypted face photos using adversarial attacks by introducing imperceptible perturbations to reduce face information leakage. However, existing studies need stronger black-box scenario feasibility and more natural visual appearances, which challenge the feasibility of privacy protection. To address these problems, we propose a frequency-restricted identity-agnostic (FRIA) framework to encrypt face images from unauthorized face recognition without access to personal information. As for the weak black-box scenario feasibility, we obverse that representations of the average feature in multiple face recognition models are similar, thus we propose to utilize the average feature via the crawled dataset from the Internet as the target to guide the generation, which is also agnostic to identities of unknown face recognition systems; in nature, the low-frequency perturbations are more visually perceptible by the human vision system. Inspired by this, we restrict the perturbation in the low-frequency facial regions by discrete cosine transform to achieve the visual naturalness guarantee. Extensive experiments on several face recognition models demonstrate that our FRIA outperforms other state-of-the-art methods in generating more natural encrypted faces while attaining high black-box attack success rates of 96%. In addition, we validate the efficacy of FRIA using real-world black-box commercial API, which reveals the potential of FRIA in practice. Our codes can be found in https://github.com/XinDong10/FRIA.



## **29. Evaluating the Vulnerabilities in ML systems in terms of adversarial attacks**

cs.LG

**SubmitDate**: 2023-08-24    [abs](http://arxiv.org/abs/2308.12918v1) [paper-pdf](http://arxiv.org/pdf/2308.12918v1)

**Authors**: John Harshith, Mantej Singh Gill, Madhan Jothimani

**Abstract**: There have been recent adversarial attacks that are difficult to find. These new adversarial attacks methods may pose challenges to current deep learning cyber defense systems and could influence the future defense of cyberattacks. The authors focus on this domain in this research paper. They explore the consequences of vulnerabilities in AI systems. This includes discussing how they might arise, differences between randomized and adversarial examples and also potential ethical implications of vulnerabilities. Moreover, it is important to train the AI systems appropriately when they are in testing phase and getting them ready for broader use.



## **30. Near Optimal Adversarial Attack on UCB Bandits**

cs.LG

Appeared at ICML 2023 AdvML Workshop

**SubmitDate**: 2023-08-24    [abs](http://arxiv.org/abs/2008.09312v6) [paper-pdf](http://arxiv.org/pdf/2008.09312v6)

**Authors**: Shiliang Zuo

**Abstract**: I study a stochastic multi-arm bandit problem where rewards are subject to adversarial corruption. I propose a novel attack strategy that manipulates a learner employing the UCB algorithm into pulling some non-optimal target arm $T - o(T)$ times with a cumulative cost that scales as $\widehat{O}(\sqrt{\log T})$, where $T$ is the number of rounds. I also prove the first lower bound on the cumulative attack cost. The lower bound matches the upper bound up to $O(\log \log T)$ factors, showing the proposed attack strategy to be near optimal.



## **31. Fast Adversarial Training with Smooth Convergence**

cs.LG

**SubmitDate**: 2023-08-24    [abs](http://arxiv.org/abs/2308.12857v1) [paper-pdf](http://arxiv.org/pdf/2308.12857v1)

**Authors**: Mengnan Zhao, Lihe Zhang, Yuqiu Kong, Baocai Yin

**Abstract**: Fast adversarial training (FAT) is beneficial for improving the adversarial robustness of neural networks. However, previous FAT work has encountered a significant issue known as catastrophic overfitting when dealing with large perturbation budgets, \ie the adversarial robustness of models declines to near zero during training.   To address this, we analyze the training process of prior FAT work and observe that catastrophic overfitting is accompanied by the appearance of loss convergence outliers.   Therefore, we argue a moderately smooth loss convergence process will be a stable FAT process that solves catastrophic overfitting.   To obtain a smooth loss convergence process, we propose a novel oscillatory constraint (dubbed ConvergeSmooth) to limit the loss difference between adjacent epochs. The convergence stride of ConvergeSmooth is introduced to balance convergence and smoothing. Likewise, we design weight centralization without introducing additional hyperparameters other than the loss balance coefficient.   Our proposed methods are attack-agnostic and thus can improve the training stability of various FAT techniques.   Extensive experiments on popular datasets show that the proposed methods efficiently avoid catastrophic overfitting and outperform all previous FAT methods. Code is available at \url{https://github.com/FAT-CS/ConvergeSmooth}.



## **32. Unifying Gradients to Improve Real-world Robustness for Deep Networks**

stat.ML

**SubmitDate**: 2023-08-24    [abs](http://arxiv.org/abs/2208.06228v2) [paper-pdf](http://arxiv.org/pdf/2208.06228v2)

**Authors**: Yingwen Wu, Sizhe Chen, Kun Fang, Xiaolin Huang

**Abstract**: The wide application of deep neural networks (DNNs) demands an increasing amount of attention to their real-world robustness, i.e., whether a DNN resists black-box adversarial attacks, among which score-based query attacks (SQAs) are most threatening since they can effectively hurt a victim network with the only access to model outputs. Defending against SQAs requires a slight but artful variation of outputs due to the service purpose for users, who share the same output information with SQAs. In this paper, we propose a real-world defense by Unifying Gradients (UniG) of different data so that SQAs could only probe a much weaker attack direction that is similar for different samples. Since such universal attack perturbations have been validated as less aggressive than the input-specific perturbations, UniG protects real-world DNNs by indicating attackers a twisted and less informative attack direction. We implement UniG efficiently by a Hadamard product module which is plug-and-play. According to extensive experiments on 5 SQAs, 2 adaptive attacks and 7 defense baselines, UniG significantly improves real-world robustness without hurting clean accuracy on CIFAR10 and ImageNet. For instance, UniG maintains a model of 77.80% accuracy under 2500-query Square attack while the state-of-the-art adversarially-trained model only has 67.34% on CIFAR10. Simultaneously, UniG outperforms all compared baselines in terms of clean accuracy and achieves the smallest modification of the model output. The code is released at https://github.com/snowien/UniG-pytorch.



## **33. Universal Soldier: Using Universal Adversarial Perturbations for Detecting Backdoor Attacks**

cs.LG

**SubmitDate**: 2023-08-24    [abs](http://arxiv.org/abs/2302.00747v3) [paper-pdf](http://arxiv.org/pdf/2302.00747v3)

**Authors**: Xiaoyun Xu, Oguzhan Ersoy, Stjepan Picek

**Abstract**: Deep learning models achieve excellent performance in numerous machine learning tasks. Yet, they suffer from security-related issues such as adversarial examples and poisoning (backdoor) attacks. A deep learning model may be poisoned by training with backdoored data or by modifying inner network parameters. Then, a backdoored model performs as expected when receiving a clean input, but it misclassifies when receiving a backdoored input stamped with a pre-designed pattern called "trigger". Unfortunately, it is difficult to distinguish between clean and backdoored models without prior knowledge of the trigger. This paper proposes a backdoor detection method by utilizing a special type of adversarial attack, universal adversarial perturbation (UAP), and its similarities with a backdoor trigger. We observe an intuitive phenomenon: UAPs generated from backdoored models need fewer perturbations to mislead the model than UAPs from clean models. UAPs of backdoored models tend to exploit the shortcut from all classes to the target class, built by the backdoor trigger. We propose a novel method called Universal Soldier for Backdoor detection (USB) and reverse engineering potential backdoor triggers via UAPs. Experiments on 345 models trained on several datasets show that USB effectively detects the injected backdoor and provides comparable or better results than state-of-the-art methods.



## **34. Don't Look into the Sun: Adversarial Solarization Attacks on Image Classifiers**

cs.CV

**SubmitDate**: 2023-08-24    [abs](http://arxiv.org/abs/2308.12661v1) [paper-pdf](http://arxiv.org/pdf/2308.12661v1)

**Authors**: Paul Gavrikov, Janis Keuper

**Abstract**: Assessing the robustness of deep neural networks against out-of-distribution inputs is crucial, especially in safety-critical domains like autonomous driving, but also in safety systems where malicious actors can digitally alter inputs to circumvent safety guards. However, designing effective out-of-distribution tests that encompass all possible scenarios while preserving accurate label information is a challenging task. Existing methodologies often entail a compromise between variety and constraint levels for attacks and sometimes even both. In a first step towards a more holistic robustness evaluation of image classification models, we introduce an attack method based on image solarization that is conceptually straightforward yet avoids jeopardizing the global structure of natural images independent of the intensity. Through comprehensive evaluations of multiple ImageNet models, we demonstrate the attack's capacity to degrade accuracy significantly, provided it is not integrated into the training augmentations. Interestingly, even then, no full immunity to accuracy deterioration is achieved. In other settings, the attack can often be simplified into a black-box attack with model-independent parameters. Defenses against other corruptions do not consistently extend to be effective against our specific attack.   Project website: https://github.com/paulgavrikov/adversarial_solarization



## **35. Exploring Transferability of Multimodal Adversarial Samples for Vision-Language Pre-training Models with Contrastive Learning**

cs.MM

**SubmitDate**: 2023-08-24    [abs](http://arxiv.org/abs/2308.12636v1) [paper-pdf](http://arxiv.org/pdf/2308.12636v1)

**Authors**: Youze Wang, Wenbo Hu, Yinpeng Dong, Richang Hong

**Abstract**: Vision-language pre-training models (VLP) are vulnerable, especially to multimodal adversarial samples, which can be crafted by adding imperceptible perturbations on both original images and texts. However, under the black-box setting, there have been no works to explore the transferability of multimodal adversarial attacks against the VLP models. In this work, we take CLIP as the surrogate model and propose a gradient-based multimodal attack method to generate transferable adversarial samples against the VLP models. By applying the gradient to optimize the adversarial images and adversarial texts simultaneously, our method can better search for and attack the vulnerable images and text information pairs. To improve the transferability of the attack, we utilize contrastive learning including image-text contrastive learning and intra-modal contrastive learning to have a more generalized understanding of the underlying data distribution and mitigate the overfitting of the surrogate model so that the generated multimodal adversarial samples have a higher transferability for VLP models. Extensive experiments validate the effectiveness of the proposed method.



## **36. PromptBench: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts**

cs.CL

Technical report; updated with new experiments and related work; 27  pages; code is at: https://github.com/microsoft/promptbench

**SubmitDate**: 2023-08-24    [abs](http://arxiv.org/abs/2306.04528v3) [paper-pdf](http://arxiv.org/pdf/2306.04528v3)

**Authors**: Kaijie Zhu, Jindong Wang, Jiaheng Zhou, Zichen Wang, Hao Chen, Yidong Wang, Linyi Yang, Wei Ye, Neil Zhenqiang Gong, Yue Zhang, Xing Xie

**Abstract**: The increasing reliance on Large Language Models (LLMs) across academia and industry necessitates a comprehensive understanding of their robustness to prompts. In response to this vital need, we introduce PromptBench, a robustness benchmark designed to measure LLMs' resilience to adversarial prompts. This study uses a plethora of adversarial textual attacks targeting prompts across multiple levels: character, word, sentence, and semantic. These prompts are then employed in diverse tasks, such as sentiment analysis, natural language inference, reading comprehension, machine translation, and math problem-solving. Our study generates 4,032 adversarial prompts, meticulously evaluated over 8 tasks and 13 datasets, with 567,084 test samples in total. Our findings demonstrate that contemporary LLMs are vulnerable to adversarial prompts. Furthermore, we present comprehensive analysis to understand the mystery behind prompt robustness and its transferability. We then offer insightful robustness analysis and pragmatic recommendations for prompt composition, beneficial to both researchers and everyday users. We make our code, prompts, and methodologies to generate adversarial prompts publicly accessible, thereby enabling and encouraging collaborative exploration in this pivotal field: https://github.com/microsoft/promptbench.



## **37. Towards an Accurate and Secure Detector against Adversarial Perturbations**

cs.CV

**SubmitDate**: 2023-08-24    [abs](http://arxiv.org/abs/2305.10856v2) [paper-pdf](http://arxiv.org/pdf/2305.10856v2)

**Authors**: Chao Wang, Shuren Qi, Zhiqiu Huang, Yushu Zhang, Rushi Lan, Xiaochun Cao

**Abstract**: The vulnerability of deep neural networks to adversarial perturbations has been widely perceived in the computer vision community. From a security perspective, it poses a critical risk for modern vision systems, e.g., the popular Deep Learning as a Service (DLaaS) frameworks. For protecting off-the-shelf deep models while not modifying them, current algorithms typically detect adversarial patterns through discriminative decomposition of natural-artificial data. However, these decompositions are biased towards frequency or spatial discriminability, thus failing to capture adversarial patterns comprehensively. More seriously, successful defense-aware (secondary) adversarial attack (i.e., evading the detector as well as fooling the model) is practical under the assumption that the adversary is fully aware of the detector (i.e., the Kerckhoffs's principle). Motivated by such facts, we propose an accurate and secure adversarial example detector, relying on a spatial-frequency discriminative decomposition with secret keys. It expands the above works on two aspects: 1) the introduced Krawtchouk basis provides better spatial-frequency discriminability and thereby is more suitable for capturing adversarial patterns than the common trigonometric or wavelet basis; 2) the extensive parameters for decomposition are generated by a pseudo-random function with secret keys, hence blocking the defense-aware adversarial attack. Theoretical and numerical analysis demonstrates the increased accuracy and security of our detector with respect to a number of state-of-the-art algorithms.



## **38. A Huber Loss Minimization Approach to Byzantine Robust Federated Learning**

cs.LG

**SubmitDate**: 2023-08-24    [abs](http://arxiv.org/abs/2308.12581v1) [paper-pdf](http://arxiv.org/pdf/2308.12581v1)

**Authors**: Puning Zhao, Fei Yu, Zhiguo Wan

**Abstract**: Federated learning systems are susceptible to adversarial attacks. To combat this, we introduce a novel aggregator based on Huber loss minimization, and provide a comprehensive theoretical analysis. Under independent and identically distributed (i.i.d) assumption, our approach has several advantages compared to existing methods. Firstly, it has optimal dependence on $\epsilon$, which stands for the ratio of attacked clients. Secondly, our approach does not need precise knowledge of $\epsilon$. Thirdly, it allows different clients to have unequal data sizes. We then broaden our analysis to include non-i.i.d data, such that clients have slightly different distributions.



## **39. Adversarial Training Using Feedback Loops**

cs.LG

**SubmitDate**: 2023-08-24    [abs](http://arxiv.org/abs/2308.11881v2) [paper-pdf](http://arxiv.org/pdf/2308.11881v2)

**Authors**: Ali Haisam Muhammad Rafid, Adrian Sandu

**Abstract**: Deep neural networks (DNN) have found wide applicability in numerous fields due to their ability to accurately learn very complex input-output relations. Despite their accuracy and extensive use, DNNs are highly susceptible to adversarial attacks due to limited generalizability. For future progress in the field, it is essential to build DNNs that are robust to any kind of perturbations to the data points. In the past, many techniques have been proposed to robustify DNNs using first-order derivative information of the network.   This paper proposes a new robustification approach based on control theory. A neural network architecture that incorporates feedback control, named Feedback Neural Networks, is proposed. The controller is itself a neural network, which is trained using regular and adversarial data such as to stabilize the system outputs. The novel adversarial training approach based on the feedback control architecture is called Feedback Looped Adversarial Training (FLAT). Numerical results on standard test problems empirically show that our FLAT method is more effective than the state-of-the-art to guard against adversarial attacks.



## **40. BadVFL: Backdoor Attacks in Vertical Federated Learning**

cs.LG

Accepted for publication at the 45th IEEE Symposium on Security &  Privacy (S&P 2024). Please cite accordingly

**SubmitDate**: 2023-08-23    [abs](http://arxiv.org/abs/2304.08847v2) [paper-pdf](http://arxiv.org/pdf/2304.08847v2)

**Authors**: Mohammad Naseri, Yufei Han, Emiliano De Cristofaro

**Abstract**: Federated learning (FL) enables multiple parties to collaboratively train a machine learning model without sharing their data; rather, they train their own model locally and send updates to a central server for aggregation. Depending on how the data is distributed among the participants, FL can be classified into Horizontal (HFL) and Vertical (VFL). In VFL, the participants share the same set of training instances but only host a different and non-overlapping subset of the whole feature space. Whereas in HFL, each participant shares the same set of features while the training set is split into locally owned training data subsets.   VFL is increasingly used in applications like financial fraud detection; nonetheless, very little work has analyzed its security. In this paper, we focus on robustness in VFL, in particular, on backdoor attacks, whereby an adversary attempts to manipulate the aggregate model during the training process to trigger misclassifications. Performing backdoor attacks in VFL is more challenging than in HFL, as the adversary i) does not have access to the labels during training and ii) cannot change the labels as she only has access to the feature embeddings. We present a first-of-its-kind clean-label backdoor attack in VFL, which consists of two phases: a label inference and a backdoor phase. We demonstrate the effectiveness of the attack on three different datasets, investigate the factors involved in its success, and discuss countermeasures to mitigate its impact.



## **41. BaDExpert: Extracting Backdoor Functionality for Accurate Backdoor Input Detection**

cs.CR

**SubmitDate**: 2023-08-23    [abs](http://arxiv.org/abs/2308.12439v1) [paper-pdf](http://arxiv.org/pdf/2308.12439v1)

**Authors**: Tinghao Xie, Xiangyu Qi, Ping He, Yiming Li, Jiachen T. Wang, Prateek Mittal

**Abstract**: We present a novel defense, against backdoor attacks on Deep Neural Networks (DNNs), wherein adversaries covertly implant malicious behaviors (backdoors) into DNNs. Our defense falls within the category of post-development defenses that operate independently of how the model was generated. The proposed defense is built upon a novel reverse engineering approach that can directly extract backdoor functionality of a given backdoored model to a backdoor expert model. The approach is straightforward -- finetuning the backdoored model over a small set of intentionally mislabeled clean samples, such that it unlearns the normal functionality while still preserving the backdoor functionality, and thus resulting in a model (dubbed a backdoor expert model) that can only recognize backdoor inputs. Based on the extracted backdoor expert model, we show the feasibility of devising highly accurate backdoor input detectors that filter out the backdoor inputs during model inference. Further augmented by an ensemble strategy with a finetuned auxiliary model, our defense, BaDExpert (Backdoor Input Detection with Backdoor Expert), effectively mitigates 16 SOTA backdoor attacks while minimally impacting clean utility. The effectiveness of BaDExpert has been verified on multiple datasets (CIFAR10, GTSRB and ImageNet) across various model architectures (ResNet, VGG, MobileNetV2 and Vision Transformer).



## **42. On-Manifold Projected Gradient Descent**

cs.LG

**SubmitDate**: 2023-08-23    [abs](http://arxiv.org/abs/2308.12279v1) [paper-pdf](http://arxiv.org/pdf/2308.12279v1)

**Authors**: Aaron Mahler, Tyrus Berry, Tom Stephens, Harbir Antil, Michael Merritt, Jeanie Schreiber, Ioannis Kevrekidis

**Abstract**: This work provides a computable, direct, and mathematically rigorous approximation to the differential geometry of class manifolds for high-dimensional data, along with nonlinear projections from input space onto these class manifolds. The tools are applied to the setting of neural network image classifiers, where we generate novel, on-manifold data samples, and implement a projected gradient descent algorithm for on-manifold adversarial training. The susceptibility of neural networks (NNs) to adversarial attack highlights the brittle nature of NN decision boundaries in input space. Introducing adversarial examples during training has been shown to reduce the susceptibility of NNs to adversarial attack; however, it has also been shown to reduce the accuracy of the classifier if the examples are not valid examples for that class. Realistic "on-manifold" examples have been previously generated from class manifolds in the latent of an autoencoder. Our work explores these phenomena in a geometric and computational setting that is much closer to the raw, high-dimensional input space than can be provided by VAE or other black box dimensionality reductions. We employ conformally invariant diffusion maps (CIDM) to approximate class manifolds in diffusion coordinates, and develop the Nystr\"{o}m projection to project novel points onto class manifolds in this setting. On top of the manifold approximation, we leverage the spectral exterior calculus (SEC) to determine geometric quantities such as tangent vectors of the manifold. We use these tools to obtain adversarial examples that reside on a class manifold, yet fool a classifier. These misclassifications then become explainable in terms of human-understandable manipulations within the data, by expressing the on-manifold adversary in the semantic basis on the manifold.



## **43. LCANets++: Robust Audio Classification using Multi-layer Neural Networks with Lateral Competition**

cs.SD

This work has been submitted to the IEEE for possible publication.  Copyright may be transferred without notice, after which this version may no  longer be accessible

**SubmitDate**: 2023-08-23    [abs](http://arxiv.org/abs/2308.12882v1) [paper-pdf](http://arxiv.org/pdf/2308.12882v1)

**Authors**: Sayanton V. Dibbo, Juston S. Moore, Garrett T. Kenyon, Michael A. Teti

**Abstract**: Audio classification aims at recognizing audio signals, including speech commands or sound events. However, current audio classifiers are susceptible to perturbations and adversarial attacks. In addition, real-world audio classification tasks often suffer from limited labeled data. To help bridge these gaps, previous work developed neuro-inspired convolutional neural networks (CNNs) with sparse coding via the Locally Competitive Algorithm (LCA) in the first layer (i.e., LCANets) for computer vision. LCANets learn in a combination of supervised and unsupervised learning, reducing dependency on labeled samples. Motivated by the fact that auditory cortex is also sparse, we extend LCANets to audio recognition tasks and introduce LCANets++, which are CNNs that perform sparse coding in multiple layers via LCA. We demonstrate that LCANets++ are more robust than standard CNNs and LCANets against perturbations, e.g., background noise, as well as black-box and white-box attacks, e.g., evasion and fast gradient sign (FGSM) attacks.



## **44. Sample Complexity of Robust Learning against Evasion Attacks**

cs.LG

DPhil (PhD) Thesis - University of Oxford

**SubmitDate**: 2023-08-23    [abs](http://arxiv.org/abs/2308.12054v1) [paper-pdf](http://arxiv.org/pdf/2308.12054v1)

**Authors**: Pascale Gourdeau

**Abstract**: It is becoming increasingly important to understand the vulnerability of machine learning models to adversarial attacks. One of the fundamental problems in adversarial machine learning is to quantify how much training data is needed in the presence of evasion attacks, where data is corrupted at test time. In this thesis, we work with the exact-in-the-ball notion of robustness and study the feasibility of adversarially robust learning from the perspective of learning theory, considering sample complexity.   We first explore the setting where the learner has access to random examples only, and show that distributional assumptions are essential. We then focus on learning problems with distributions on the input data that satisfy a Lipschitz condition and show that robustly learning monotone conjunctions has sample complexity at least exponential in the adversary's budget (the maximum number of bits it can perturb on each input). However, if the adversary is restricted to perturbing $O(\log n)$ bits, then one can robustly learn conjunctions and decision lists w.r.t. log-Lipschitz distributions.   We then study learning models where the learner is given more power. We first consider local membership queries, where the learner can query the label of points near the training sample. We show that, under the uniform distribution, the exponential dependence on the adversary's budget to robustly learn conjunctions remains inevitable. We then introduce a local equivalence query oracle, which returns whether the hypothesis and target concept agree in a given region around a point in the training sample, and a counterexample if it exists. We show that if the query radius is equal to the adversary's budget, we can develop robust empirical risk minimization algorithms in the distribution-free setting. We give general query complexity upper and lower bounds, as well as for concrete concept classes.



## **45. Phase-shifted Adversarial Training**

cs.LG

Conference on Uncertainty in Artificial Intelligence, 2023 (UAI 2023)

**SubmitDate**: 2023-08-23    [abs](http://arxiv.org/abs/2301.04785v3) [paper-pdf](http://arxiv.org/pdf/2301.04785v3)

**Authors**: Yeachan Kim, Seongyeon Kim, Ihyeok Seo, Bonggun Shin

**Abstract**: Adversarial training has been considered an imperative component for safely deploying neural network-based applications to the real world. To achieve stronger robustness, existing methods primarily focus on how to generate strong attacks by increasing the number of update steps, regularizing the models with the smoothed loss function, and injecting the randomness into the attack. Instead, we analyze the behavior of adversarial training through the lens of response frequency. We empirically discover that adversarial training causes neural networks to have low convergence to high-frequency information, resulting in highly oscillated predictions near each data. To learn high-frequency contents efficiently and effectively, we first prove that a universal phenomenon of frequency principle, i.e., \textit{lower frequencies are learned first}, still holds in adversarial training. Based on that, we propose phase-shifted adversarial training (PhaseAT) in which the model learns high-frequency components by shifting these frequencies to the low-frequency range where the fast convergence occurs. For evaluations, we conduct the experiments on CIFAR-10 and ImageNet with the adaptive attack carefully designed for reliable evaluation. Comprehensive results show that PhaseAT significantly improves the convergence for high-frequency information. This results in improved adversarial robustness by enabling the model to have smoothed predictions near each data.



## **46. Designing an attack-defense game: how to increase robustness of financial transaction models via a competition**

cs.LG

**SubmitDate**: 2023-08-23    [abs](http://arxiv.org/abs/2308.11406v2) [paper-pdf](http://arxiv.org/pdf/2308.11406v2)

**Authors**: Alexey Zaytsev, Alex Natekin, Evgeni Vorsin, Valerii Smirnov, Georgii Smirnov, Oleg Sidorshin, Alexander Senin, Alexander Dudin, Dmitry Berestnev

**Abstract**: Given the escalating risks of malicious attacks in the finance sector and the consequential severe damage, a thorough understanding of adversarial strategies and robust defense mechanisms for machine learning models is critical. The threat becomes even more severe with the increased adoption in banks more accurate, but potentially fragile neural networks. We aim to investigate the current state and dynamics of adversarial attacks and defenses for neural network models that use sequential financial data as the input.   To achieve this goal, we have designed a competition that allows realistic and detailed investigation of problems in modern financial transaction data. The participants compete directly against each other, so possible attacks and defenses are examined in close-to-real-life conditions. Our main contributions are the analysis of the competition dynamics that answers the questions on how important it is to conceal a model from malicious users, how long does it take to break it, and what techniques one should use to make it more robust, and introduction additional way to attack models or increase their robustness.   Our analysis continues with a meta-study on the used approaches with their power, numerical experiments, and accompanied ablations studies. We show that the developed attacks and defenses outperform existing alternatives from the literature while being practical in terms of execution, proving the validity of the competition as a tool for uncovering vulnerabilities of machine learning models and mitigating them in various domains.



## **47. Does Physical Adversarial Example Really Matter to Autonomous Driving? Towards System-Level Effect of Adversarial Object Evasion Attack**

cs.CR

Accepted by ICCV 2023

**SubmitDate**: 2023-08-23    [abs](http://arxiv.org/abs/2308.11894v1) [paper-pdf](http://arxiv.org/pdf/2308.11894v1)

**Authors**: Ningfei Wang, Yunpeng Luo, Takami Sato, Kaidi Xu, Qi Alfred Chen

**Abstract**: In autonomous driving (AD), accurate perception is indispensable to achieving safe and secure driving. Due to its safety-criticality, the security of AD perception has been widely studied. Among different attacks on AD perception, the physical adversarial object evasion attacks are especially severe. However, we find that all existing literature only evaluates their attack effect at the targeted AI component level but not at the system level, i.e., with the entire system semantics and context such as the full AD pipeline. Thereby, this raises a critical research question: can these existing researches effectively achieve system-level attack effects (e.g., traffic rule violations) in the real-world AD context? In this work, we conduct the first measurement study on whether and how effectively the existing designs can lead to system-level effects, especially for the STOP sign-evasion attacks due to their popularity and severity. Our evaluation results show that all the representative prior works cannot achieve any system-level effects. We observe two design limitations in the prior works: 1) physical model-inconsistent object size distribution in pixel sampling and 2) lack of vehicle plant model and AD system model consideration. Then, we propose SysAdv, a novel system-driven attack design in the AD context and our evaluation results show that the system-level effects can be significantly improved, i.e., the violation rate increases by around 70%.



## **48. Measuring Equality in Machine Learning Security Defenses: A Case Study in Speech Recognition**

cs.LG

Accepted to AISec'23

**SubmitDate**: 2023-08-23    [abs](http://arxiv.org/abs/2302.08973v6) [paper-pdf](http://arxiv.org/pdf/2302.08973v6)

**Authors**: Luke E. Richards, Edward Raff, Cynthia Matuszek

**Abstract**: Over the past decade, the machine learning security community has developed a myriad of defenses for evasion attacks. An understudied question in that community is: for whom do these defenses defend? This work considers common approaches to defending learned systems and how security defenses result in performance inequities across different sub-populations. We outline appropriate parity metrics for analysis and begin to answer this question through empirical results of the fairness implications of machine learning security methods. We find that many methods that have been proposed can cause direct harm, like false rejection and unequal benefits from robustness training. The framework we propose for measuring defense equality can be applied to robustly trained models, preprocessing-based defenses, and rejection methods. We identify a set of datasets with a user-centered application and a reasonable computational cost suitable for case studies in measuring the equality of defenses. In our case study of speech command recognition, we show how such adversarial training and augmentation have non-equal but complex protections for social subgroups across gender, accent, and age in relation to user coverage. We present a comparison of equality between two rejection-based defenses: randomized smoothing and neural rejection, finding randomized smoothing more equitable due to the sampling mechanism for minority groups. This represents the first work examining the disparity in the adversarial robustness in the speech domain and the fairness evaluation of rejection-based defenses.



## **49. SEA: Shareable and Explainable Attribution for Query-based Black-box Attacks**

cs.LG

**SubmitDate**: 2023-08-23    [abs](http://arxiv.org/abs/2308.11845v1) [paper-pdf](http://arxiv.org/pdf/2308.11845v1)

**Authors**: Yue Gao, Ilia Shumailov, Kassem Fawaz

**Abstract**: Machine Learning (ML) systems are vulnerable to adversarial examples, particularly those from query-based black-box attacks. Despite various efforts to detect and prevent such attacks, there is a need for a more comprehensive approach to logging, analyzing, and sharing evidence of attacks. While classic security benefits from well-established forensics and intelligence sharing, Machine Learning is yet to find a way to profile its attackers and share information about them. In response, this paper introduces SEA, a novel ML security system to characterize black-box attacks on ML systems for forensic purposes and to facilitate human-explainable intelligence sharing. SEA leverages the Hidden Markov Models framework to attribute the observed query sequence to known attacks. It thus understands the attack's progression rather than just focusing on the final adversarial examples. Our evaluations reveal that SEA is effective at attack attribution, even on their second occurrence, and is robust to adaptive strategies designed to evade forensics analysis. Interestingly, SEA's explanations of the attack behavior allow us even to fingerprint specific minor implementation bugs in attack libraries. For example, we discover that the SignOPT and Square attacks implementation in ART v1.14 sends over 50% specific zero difference queries. We thoroughly evaluate SEA on a variety of settings and demonstrate that it can recognize the same attack's second occurrence with 90+% Top-1 and 95+% Top-3 accuracy.



## **50. Ceci n'est pas une pomme: Adversarial Illusions in Multi-Modal Embeddings**

cs.CR

**SubmitDate**: 2023-08-22    [abs](http://arxiv.org/abs/2308.11804v1) [paper-pdf](http://arxiv.org/pdf/2308.11804v1)

**Authors**: Eugene Bagdasaryan, Vitaly Shmatikov

**Abstract**: Multi-modal encoders map images, sounds, texts, videos, etc. into a single embedding space, aligning representations across modalities (e.g., associate an image of a dog with a barking sound). We show that multi-modal embeddings can be vulnerable to an attack we call "adversarial illusions." Given an input in any modality, an adversary can perturb it so as to make its embedding close to that of an arbitrary, adversary-chosen input in another modality. Illusions thus enable the adversary to align any image with any text, any text with any sound, etc.   Adversarial illusions exploit proximity in the embedding space and are thus agnostic to downstream tasks. Using ImageBind embeddings, we demonstrate how adversarially aligned inputs, generated without knowledge of specific downstream tasks, mislead image generation, text generation, and zero-shot classification.



