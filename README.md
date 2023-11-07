# Latest Adversarial Attack Papers
**update at 2023-11-07 10:17:15**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. On existence, uniqueness and scalability of adversarial robustness measures for AI classifiers**

stat.ML

16 pages, 3 figures

**SubmitDate**: 2023-11-06    [abs](http://arxiv.org/abs/2310.14421v2) [paper-pdf](http://arxiv.org/pdf/2310.14421v2)

**Authors**: Illia Horenko

**Abstract**: Simply-verifiable mathematical conditions for existence, uniqueness and explicit analytical computation of minimal adversarial paths (MAP) and minimal adversarial distances (MAD) for (locally) uniquely-invertible classifiers, for generalized linear models (GLM), and for entropic AI (EAI) are formulated and proven. Practical computation of MAP and MAD, their comparison and interpretations for various classes of AI tools (for neuronal networks, boosted random forests, GLM and EAI) are demonstrated on the common synthetic benchmarks: on a double Swiss roll spiral and its extensions, as well as on the two biomedical data problems (for the health insurance claim predictions, and for the heart attack lethality classification). On biomedical applications it is demonstrated how MAP provides unique minimal patient-specific risk-mitigating interventions in the predefined subsets of accessible control variables.



## **2. Preserving Privacy in GANs Against Membership Inference Attack**

cs.LG

**SubmitDate**: 2023-11-06    [abs](http://arxiv.org/abs/2311.03172v1) [paper-pdf](http://arxiv.org/pdf/2311.03172v1)

**Authors**: Mohammadhadi Shateri, Francisco Messina, Fabrice Labeau, Pablo Piantanida

**Abstract**: Generative Adversarial Networks (GANs) have been widely used for generating synthetic data for cases where there is a limited size real-world dataset or when data holders are unwilling to share their data samples. Recent works showed that GANs, due to overfitting and memorization, might leak information regarding their training data samples. This makes GANs vulnerable to Membership Inference Attacks (MIAs). Several defense strategies have been proposed in the literature to mitigate this privacy issue. Unfortunately, defense strategies based on differential privacy are proven to reduce extensively the quality of the synthetic data points. On the other hand, more recent frameworks such as PrivGAN and PAR-GAN are not suitable for small-size training datasets. In the present work, the overfitting in GANs is studied in terms of the discriminator, and a more general measure of overfitting based on the Bhattacharyya coefficient is defined. Then, inspired by Fano's inequality, our first defense mechanism against MIAs is proposed. This framework, which requires only a simple modification in the loss function of GANs, is referred to as the maximum entropy GAN or MEGAN and significantly improves the robustness of GANs to MIAs. As a second defense strategy, a more heuristic model based on minimizing the information leaked from generated samples about the training data points is presented. This approach is referred to as mutual information minimization GAN (MIMGAN) and uses a variational representation of the mutual information to minimize the information that a synthetic sample might leak about the whole training data set. Applying the proposed frameworks to some commonly used data sets against state-of-the-art MIAs reveals that the proposed methods can reduce the accuracy of the adversaries to the level of random guessing accuracy with a small reduction in the quality of the synthetic data samples.



## **3. Quantum-Error-Mitigated Detectable Byzantine Agreement with Dynamical Decoupling for Distributed Quantum Computing**

quant-ph

**SubmitDate**: 2023-11-06    [abs](http://arxiv.org/abs/2311.03097v1) [paper-pdf](http://arxiv.org/pdf/2311.03097v1)

**Authors**: Matthew Prest, Kuan-Cheng Chen

**Abstract**: In the burgeoning domain of distributed quantum computing, achieving consensus amidst adversarial settings remains a pivotal challenge. We introduce an enhancement to the Quantum Byzantine Agreement (QBA) protocol, uniquely incorporating advanced error mitigation techniques: Twirled Readout Error Extinction (T-REx) and dynamical decoupling (DD). Central to this refined approach is the utilization of a Noisy Intermediate Scale Quantum (NISQ) source device for heightened performance. Extensive tests on both simulated and real-world quantum devices, notably IBM's quantum computer, provide compelling evidence of the effectiveness of our T-REx and DD adaptations in mitigating prevalent quantum channel errors.   Subsequent to the entanglement distribution, our protocol adopts a verification method reminiscent of Quantum Key Distribution (QKD) schemes. The Commander then issues orders encoded in specific quantum states, like Retreat or Attack. In situations where received orders diverge, lieutenants engage in structured games to reconcile discrepancies. Notably, the frequency of these games is contingent upon the Commander's strategies and the overall network size. Our empirical findings underscore the enhanced resilience and effectiveness of the protocol in diverse scenarios. Nonetheless, scalability emerges as a concern with the growth of the network size. To sum up, our research illuminates the considerable potential of fortified quantum consensus systems in the NISQ era, highlighting the imperative for sustained research in bolstering quantum ecosystems.



## **4. SoK: Memorisation in machine learning**

cs.LG

**SubmitDate**: 2023-11-06    [abs](http://arxiv.org/abs/2311.03075v1) [paper-pdf](http://arxiv.org/pdf/2311.03075v1)

**Authors**: Dmitrii Usynin, Moritz Knolle, Georgios Kaissis

**Abstract**: Quantifying the impact of individual data samples on machine learning models is an open research problem. This is particularly relevant when complex and high-dimensional relationships have to be learned from a limited sample of the data generating distribution, such as in deep learning. It was previously shown that, in these cases, models rely not only on extracting patterns which are helpful for generalisation, but also seem to be required to incorporate some of the training data more or less as is, in a process often termed memorisation. This raises the question: if some memorisation is a requirement for effective learning, what are its privacy implications? In this work we unify a broad range of previous definitions and perspectives on memorisation in ML, discuss their interplay with model generalisation and their implications of these phenomena on data privacy. Moreover, we systematise methods allowing practitioners to detect the occurrence of memorisation or quantify it and contextualise our findings in a broad range of ML learning settings. Finally, we discuss memorisation in the context of privacy attacks, differential privacy (DP) and adversarial actors.



## **5. MAWSEO: Adversarial Wiki Search Poisoning for Illicit Online Promotion**

cs.CR

**SubmitDate**: 2023-11-06    [abs](http://arxiv.org/abs/2304.11300v2) [paper-pdf](http://arxiv.org/pdf/2304.11300v2)

**Authors**: Zilong Lin, Zhengyi Li, Xiaojing Liao, XiaoFeng Wang, Xiaozhong Liu

**Abstract**: As a prominent instance of vandalism edits, Wiki search poisoning for illicit promotion is a cybercrime in which the adversary aims at editing Wiki articles to promote illicit businesses through Wiki search results of relevant queries. In this paper, we report a study that, for the first time, shows that such stealthy blackhat SEO on Wiki can be automated. Our technique, called MAWSEO, employs adversarial revisions to achieve real-world cybercriminal objectives, including rank boosting, vandalism detection evasion, topic relevancy, semantic consistency, user awareness (but not alarming) of promotional content, etc. Our evaluation and user study demonstrate that MAWSEO is capable of effectively and efficiently generating adversarial vandalism edits, which can bypass state-of-the-art built-in Wiki vandalism detectors, and also get promotional content through to Wiki users without triggering their alarms. In addition, we investigated potential defense, including coherence based detection and adversarial training of vandalism detection, against our attack in the Wiki ecosystem.



## **6. Detecting Language Model Attacks with Perplexity**

cs.CL

**SubmitDate**: 2023-11-05    [abs](http://arxiv.org/abs/2308.14132v2) [paper-pdf](http://arxiv.org/pdf/2308.14132v2)

**Authors**: Gabriel Alon, Michael Kamfonas

**Abstract**: A novel hack involving Large Language Models (LLMs) has emerged, leveraging adversarial suffixes to trick models into generating perilous responses. This method has garnered considerable attention from reputable media outlets such as the New York Times and Wired, thereby influencing public perception regarding the security and safety of LLMs. In this study, we advocate the utilization of perplexity as one of the means to recognize such potential attacks. The underlying concept behind these hacks revolves around appending an unusually constructed string of text to a harmful query that would otherwise be blocked. This maneuver confuses the protective mechanisms and tricks the model into generating a forbidden response. Such scenarios could result in providing detailed instructions to a malicious user for constructing explosives or orchestrating a bank heist. Our investigation demonstrates the feasibility of employing perplexity, a prevalent natural language processing metric, to detect these adversarial tactics before generating a forbidden response. By evaluating the perplexity of queries with and without such adversarial suffixes using an open-source LLM, we discovered that nearly 90 percent were above a perplexity of 1000. This contrast underscores the efficacy of perplexity for detecting this type of exploit.



## **7. CT-GAT: Cross-Task Generative Adversarial Attack based on Transferability**

cs.CL

Accepted to EMNLP 2023 main conference Corrected the header error in  Figure 3

**SubmitDate**: 2023-11-05    [abs](http://arxiv.org/abs/2310.14265v2) [paper-pdf](http://arxiv.org/pdf/2310.14265v2)

**Authors**: Minxuan Lv, Chengwei Dai, Kun Li, Wei Zhou, Songlin Hu

**Abstract**: Neural network models are vulnerable to adversarial examples, and adversarial transferability further increases the risk of adversarial attacks. Current methods based on transferability often rely on substitute models, which can be impractical and costly in real-world scenarios due to the unavailability of training data and the victim model's structural details. In this paper, we propose a novel approach that directly constructs adversarial examples by extracting transferable features across various tasks. Our key insight is that adversarial transferability can extend across different tasks. Specifically, we train a sequence-to-sequence generative model named CT-GAT using adversarial sample data collected from multiple tasks to acquire universal adversarial features and generate adversarial examples for different tasks. We conduct experiments on ten distinct datasets, and the results demonstrate that our method achieves superior attack performance with small cost.



## **8. DeepZero: Scaling up Zeroth-Order Optimization for Deep Model Training**

cs.LG

**SubmitDate**: 2023-11-05    [abs](http://arxiv.org/abs/2310.02025v2) [paper-pdf](http://arxiv.org/pdf/2310.02025v2)

**Authors**: Aochuan Chen, Yimeng Zhang, Jinghan Jia, James Diffenderfer, Jiancheng Liu, Konstantinos Parasyris, Yihua Zhang, Zheng Zhang, Bhavya Kailkhura, Sijia Liu

**Abstract**: Zeroth-order (ZO) optimization has become a popular technique for solving machine learning (ML) problems when first-order (FO) information is difficult or impossible to obtain. However, the scalability of ZO optimization remains an open problem: Its use has primarily been limited to relatively small-scale ML problems, such as sample-wise adversarial attack generation. To our best knowledge, no prior work has demonstrated the effectiveness of ZO optimization in training deep neural networks (DNNs) without a significant decrease in performance. To overcome this roadblock, we develop DeepZero, a principled ZO deep learning (DL) framework that can scale ZO optimization to DNN training from scratch through three primary innovations. First, we demonstrate the advantages of coordinate-wise gradient estimation (CGE) over randomized vector-wise gradient estimation in training accuracy and computational efficiency. Second, we propose a sparsity-induced ZO training protocol that extends the model pruning methodology using only finite differences to explore and exploit the sparse DL prior in CGE. Third, we develop the methods of feature reuse and forward parallelization to advance the practical implementations of ZO training. Our extensive experiments show that DeepZero achieves state-of-the-art (SOTA) accuracy on ResNet-20 trained on CIFAR-10, approaching FO training performance for the first time. Furthermore, we show the practical utility of DeepZero in applications of certified adversarial defense and DL-based partial differential equation error correction, achieving 10-20% improvement over SOTA. We believe our results will inspire future research on scalable ZO optimization and contribute to advancing DL with black box.



## **9. Exploring Transferability of Multimodal Adversarial Samples for Vision-Language Pre-training Models with Contrastive Learning**

cs.MM

**SubmitDate**: 2023-11-05    [abs](http://arxiv.org/abs/2308.12636v2) [paper-pdf](http://arxiv.org/pdf/2308.12636v2)

**Authors**: Youze Wang, Wenbo Hu, Yinpeng Dong, Hanwang Zhang, Richang Hong

**Abstract**: Vision-language pre-training models (VLP) are vulnerable, especially to multimodal adversarial samples, which can be crafted by adding imperceptible perturbations on both original images and texts. However, under the black-box setting, there have been no works to explore the transferability of multimodal adversarial attacks against the VLP models. In this work, we take CLIP as the surrogate model and propose a gradient-based multimodal attack method to generate transferable adversarial samples against the VLP models. By applying the gradient to optimize the adversarial images and adversarial texts simultaneously, our method can better search for and attack the vulnerable images and text information pairs. To improve the transferability of the attack, we utilize contrastive learning including image-text contrastive learning and intra-modal contrastive learning to have a more generalized understanding of the underlying data distribution and mitigate the overfitting of the surrogate model so that the generated multimodal adversarial samples have a higher transferability for VLP models. Extensive experiments validate the effectiveness of the proposed method.



## **10. Unfolding Local Growth Rate Estimates for (Almost) Perfect Adversarial Detection**

cs.CV

accepted at VISAPP23

**SubmitDate**: 2023-11-04    [abs](http://arxiv.org/abs/2212.06776v3) [paper-pdf](http://arxiv.org/pdf/2212.06776v3)

**Authors**: Peter Lorenz, Margret Keuper, Janis Keuper

**Abstract**: Convolutional neural networks (CNN) define the state-of-the-art solution on many perceptual tasks. However, current CNN approaches largely remain vulnerable against adversarial perturbations of the input that have been crafted specifically to fool the system while being quasi-imperceptible to the human eye. In recent years, various approaches have been proposed to defend CNNs against such attacks, for example by model hardening or by adding explicit defence mechanisms. Thereby, a small "detector" is included in the network and trained on the binary classification task of distinguishing genuine data from data containing adversarial perturbations. In this work, we propose a simple and light-weight detector, which leverages recent findings on the relation between networks' local intrinsic dimensionality (LID) and adversarial attacks. Based on a re-interpretation of the LID measure and several simple adaptations, we surpass the state-of-the-art on adversarial detection by a significant margin and reach almost perfect results in terms of F1-score for several networks and datasets. Sources available at: https://github.com/adverML/multiLID



## **11. MTS-DVGAN: Anomaly Detection in Cyber-Physical Systems using a Dual Variational Generative Adversarial Network**

cs.CR

27 pages, 14 figures, 8 tables. Accepted by Computers & Security

**SubmitDate**: 2023-11-04    [abs](http://arxiv.org/abs/2311.02378v1) [paper-pdf](http://arxiv.org/pdf/2311.02378v1)

**Authors**: Haili Sun, Yan Huang, Lansheng Han, Cai Fu, Hongle Liu, Xiang Long

**Abstract**: Deep generative models are promising in detecting novel cyber-physical attacks, mitigating the vulnerability of Cyber-physical systems (CPSs) without relying on labeled information. Nonetheless, these generative models face challenges in identifying attack behaviors that closely resemble normal data, or deviate from the normal data distribution but are in close proximity to the manifold of the normal cluster in latent space. To tackle this problem, this article proposes a novel unsupervised dual variational generative adversarial model named MST-DVGAN, to perform anomaly detection in multivariate time series data for CPS security. The central concept is to enhance the model's discriminative capability by widening the distinction between reconstructed abnormal samples and their normal counterparts. Specifically, we propose an augmented module by imposing contrastive constraints on the reconstruction process to obtain a more compact embedding. Then, by exploiting the distribution property and modeling the normal patterns of multivariate time series, a variational autoencoder is introduced to force the generative adversarial network (GAN) to generate diverse samples. Furthermore, two augmented loss functions are designed to extract essential characteristics in a self-supervised manner through mutual guidance between the augmented samples and original samples. Finally, a specific feature center loss is introduced for the generator network to enhance its stability. Empirical experiments are conducted on three public datasets, namely SWAT, WADI and NSL_KDD. Comparing with the state-of-the-art methods, the evaluation results show that the proposed MTS-DVGAN is more stable and can achieve consistent performance improvement.



## **12. From Trojan Horses to Castle Walls: Unveiling Bilateral Backdoor Effects in Diffusion Models**

cs.LG

10 pages, 6 figures, 7 tables

**SubmitDate**: 2023-11-04    [abs](http://arxiv.org/abs/2311.02373v1) [paper-pdf](http://arxiv.org/pdf/2311.02373v1)

**Authors**: Zhuoshi Pan, Yuguang Yao, Gaowen Liu, Bingquan Shen, H. Vicky Zhao, Ramana Rao Kompella, Sijia Liu

**Abstract**: While state-of-the-art diffusion models (DMs) excel in image generation, concerns regarding their security persist. Earlier research highlighted DMs' vulnerability to backdoor attacks, but these studies placed stricter requirements than conventional methods like 'BadNets' in image classification. This is because the former necessitates modifications to the diffusion sampling and training procedures. Unlike the prior work, we investigate whether generating backdoor attacks in DMs can be as simple as BadNets, i.e., by only contaminating the training dataset without tampering the original diffusion process. In this more realistic backdoor setting, we uncover bilateral backdoor effects that not only serve an adversarial purpose (compromising the functionality of DMs) but also offer a defensive advantage (which can be leveraged for backdoor defense). Specifically, we find that a BadNets-like backdoor attack remains effective in DMs for producing incorrect images (misaligned with the intended text conditions), and thereby yielding incorrect predictions when DMs are used as classifiers. Meanwhile, backdoored DMs exhibit an increased ratio of backdoor triggers, a phenomenon we refer to as `trigger amplification', among the generated images. We show that this latter insight can be used to enhance the detection of backdoor-poisoned training data. Even under a low backdoor poisoning ratio, studying the backdoor effects of DMs is also valuable for designing anti-backdoor image classifiers. Last but not least, we establish a meaningful linkage between backdoor attacks and the phenomenon of data replications by exploring DMs' inherent data memorization tendencies. The codes of our work are available at https://github.com/OPTML-Group/BiBadDiff.



## **13. Secure compilation of rich smart contracts on poor UTXO blockchains**

cs.CR

**SubmitDate**: 2023-11-04    [abs](http://arxiv.org/abs/2305.09545v2) [paper-pdf](http://arxiv.org/pdf/2305.09545v2)

**Authors**: Massimo Bartoletti, Riccardo Marchesin, Roberto Zunino

**Abstract**: Most blockchain platforms from Ethereum onwards render smart contracts as stateful reactive objects that update their state and transfer crypto-assets in response to transactions. A drawback of this design is that when users submit a transaction, they cannot predict in which state it will be executed. This exposes them to transaction-ordering attacks, a widespread class of attacks where adversaries with the power to construct blocks of transactions can extract value from smart contracts (the so-called MEV attacks). The UTXO model is an alternative blockchain design that thwarts these attacks by requiring new transactions to spend past ones: since transactions have unique identifiers, reordering attacks are ineffective. Currently, the blockchains following the UTXO model either provide contracts with limited expressiveness (Bitcoin), or require complex run-time environments (Cardano). We present ILLUM, an Intermediate-Level Language for the UTXO Model. ILLUM can express real-world smart contracts, e.g. those found in Decentralized Finance. We define a compiler from ILLUM to a bare-bone UTXO blockchain with loop-free scripts. Our compilation target only requires minimal extensions to Bitcoin Script: in particular, we exploit covenants, a mechanism for preserving scripts along chains of transactions. We prove the security of our compiler: namely, any attack targeting the compiled contract is also observable at the ILLUM level. Hence, the compiler does not introduce new vulnerabilities that were not already present in the source ILLUM contract. Finally, we discuss the suitability of ILLUM as a compilation target for higher-level contract languages.



## **14. Generative Adversarial Networks to infer velocity components in rotating turbulent flows**

physics.flu-dyn

**SubmitDate**: 2023-11-03    [abs](http://arxiv.org/abs/2301.07541v2) [paper-pdf](http://arxiv.org/pdf/2301.07541v2)

**Authors**: Tianyi Li, Michele Buzzicotti, Luca Biferale, Fabio Bonaccorso

**Abstract**: Inference problems for two-dimensional snapshots of rotating turbulent flows are studied. We perform a systematic quantitative benchmark of point-wise and statistical reconstruction capabilities of the linear Extended Proper Orthogonal Decomposition (EPOD) method, a non-linear Convolutional Neural Network (CNN) and a Generative Adversarial Network (GAN). We attack the important task of inferring one velocity component out of the measurement of a second one, and two cases are studied: (I) both components lay in the plane orthogonal to the rotation axis and (II) one of the two is parallel to the rotation axis. We show that EPOD method works well only for the former case where both components are strongly correlated, while CNN and GAN always outperform EPOD both concerning point-wise and statistical reconstructions. For case (II), when the input and output data are weakly correlated, all methods fail to reconstruct faithfully the point-wise information. In this case, only GAN is able to reconstruct the field in a statistical sense. The analysis is performed using both standard validation tools based on $L_2$ spatial distance between the prediction and the ground truth and more sophisticated multi-scale analysis using wavelet decomposition. Statistical validation is based on standard Jensen-Shannon divergence between the probability density functions, spectral properties and multi-scale flatness.



## **15. HINT: Healthy Influential-Noise based Training to Defend against Data Poisoning Attacks**

cs.LG

**SubmitDate**: 2023-11-03    [abs](http://arxiv.org/abs/2309.08549v2) [paper-pdf](http://arxiv.org/pdf/2309.08549v2)

**Authors**: Minh-Hao Van, Alycia N. Carey, Xintao Wu

**Abstract**: While numerous defense methods have been proposed to prohibit potential poisoning attacks from untrusted data sources, most research works only defend against specific attacks, which leaves many avenues for an adversary to exploit. In this work, we propose an efficient and robust training approach to defend against data poisoning attacks based on influence functions, named Healthy Influential-Noise based Training. Using influence functions, we craft healthy noise that helps to harden the classification model against poisoning attacks without significantly affecting the generalization ability on test data. In addition, our method can perform effectively when only a subset of the training data is modified, instead of the current method of adding noise to all examples that has been used in several previous works. We conduct comprehensive evaluations over two image datasets with state-of-the-art poisoning attacks under different realistic attack scenarios. Our empirical results show that HINT can efficiently protect deep learning models against the effect of both untargeted and targeted poisoning attacks.



## **16. Adaptive Data Analysis in a Balanced Adversarial Model**

cs.LG

Accepted to NeurIPS 2023 (Spotlight)

**SubmitDate**: 2023-11-03    [abs](http://arxiv.org/abs/2305.15452v2) [paper-pdf](http://arxiv.org/pdf/2305.15452v2)

**Authors**: Kobbi Nissim, Uri Stemmer, Eliad Tsfadia

**Abstract**: In adaptive data analysis, a mechanism gets $n$ i.i.d. samples from an unknown distribution $D$, and is required to provide accurate estimations to a sequence of adaptively chosen statistical queries with respect to $D$. Hardt and Ullman (FOCS 2014) and Steinke and Ullman (COLT 2015) showed that in general, it is computationally hard to answer more than $\Theta(n^2)$ adaptive queries, assuming the existence of one-way functions.   However, these negative results strongly rely on an adversarial model that significantly advantages the adversarial analyst over the mechanism, as the analyst, who chooses the adaptive queries, also chooses the underlying distribution $D$. This imbalance raises questions with respect to the applicability of the obtained hardness results -- an analyst who has complete knowledge of the underlying distribution $D$ would have little need, if at all, to issue statistical queries to a mechanism which only holds a finite number of samples from $D$.   We consider more restricted adversaries, called \emph{balanced}, where each such adversary consists of two separated algorithms: The \emph{sampler} who is the entity that chooses the distribution and provides the samples to the mechanism, and the \emph{analyst} who chooses the adaptive queries, but has no prior knowledge of the underlying distribution (and hence has no a priori advantage with respect to the mechanism). We improve the quality of previous lower bounds by revisiting them using an efficient \emph{balanced} adversary, under standard public-key cryptography assumptions. We show that these stronger hardness assumptions are unavoidable in the sense that any computationally bounded \emph{balanced} adversary that has the structure of all known attacks, implies the existence of public-key cryptography.



## **17. The Alignment Problem in Context**

cs.LG

**SubmitDate**: 2023-11-03    [abs](http://arxiv.org/abs/2311.02147v1) [paper-pdf](http://arxiv.org/pdf/2311.02147v1)

**Authors**: Raphaël Millière

**Abstract**: A core challenge in the development of increasingly capable AI systems is to make them safe and reliable by ensuring their behaviour is consistent with human values. This challenge, known as the alignment problem, does not merely apply to hypothetical future AI systems that may pose catastrophic risks; it already applies to current systems, such as large language models, whose potential for harm is rapidly increasing. In this paper, I assess whether we are on track to solve the alignment problem for large language models, and what that means for the safety of future AI systems. I argue that existing strategies for alignment are insufficient, because large language models remain vulnerable to adversarial attacks that can reliably elicit unsafe behaviour. I offer an explanation of this lingering vulnerability on which it is not simply a contingent limitation of current language models, but has deep technical ties to a crucial aspect of what makes these models useful and versatile in the first place -- namely, their remarkable aptitude to learn "in context" directly from user instructions. It follows that the alignment problem is not only unsolved for current AI systems, but may be intrinsically difficult to solve without severely undermining their capabilities. Furthermore, this assessment raises concerns about the prospect of ensuring the safety of future and more capable AI systems.



## **18. Bucks for Buckets (B4B): Active Defenses Against Stealing Encoders**

cs.LG

Accepted at NeurIPS2023

**SubmitDate**: 2023-11-03    [abs](http://arxiv.org/abs/2310.08571v2) [paper-pdf](http://arxiv.org/pdf/2310.08571v2)

**Authors**: Jan Dubiński, Stanisław Pawlak, Franziska Boenisch, Tomasz Trzciński, Adam Dziedzic

**Abstract**: Machine Learning as a Service (MLaaS) APIs provide ready-to-use and high-utility encoders that generate vector representations for given inputs. Since these encoders are very costly to train, they become lucrative targets for model stealing attacks during which an adversary leverages query access to the API to replicate the encoder locally at a fraction of the original training costs. We propose Bucks for Buckets (B4B), the first active defense that prevents stealing while the attack is happening without degrading representation quality for legitimate API users. Our defense relies on the observation that the representations returned to adversaries who try to steal the encoder's functionality cover a significantly larger fraction of the embedding space than representations of legitimate users who utilize the encoder to solve a particular downstream task.vB4B leverages this to adaptively adjust the utility of the returned representations according to a user's coverage of the embedding space. To prevent adaptive adversaries from eluding our defense by simply creating multiple user accounts (sybils), B4B also individually transforms each user's representations. This prevents the adversary from directly aggregating representations over multiple accounts to create their stolen encoder copy. Our active defense opens a new path towards securely sharing and democratizing encoders over public APIs.



## **19. Efficient Black-Box Adversarial Attacks on Neural Text Detectors**

cs.CL

Accepted at ICNLSP 2023

**SubmitDate**: 2023-11-03    [abs](http://arxiv.org/abs/2311.01873v1) [paper-pdf](http://arxiv.org/pdf/2311.01873v1)

**Authors**: Vitalii Fishchuk, Daniel Braun

**Abstract**: Neural text detectors are models trained to detect whether a given text was generated by a language model or written by a human. In this paper, we investigate three simple and resource-efficient strategies (parameter tweaking, prompt engineering, and character-level mutations) to alter texts generated by GPT-3.5 that are unsuspicious or unnoticeable for humans but cause misclassification by neural text detectors. The results show that especially parameter tweaking and character-level mutations are effective strategies.



## **20. Adversarial Attacks against Binary Similarity Systems**

cs.CR

**SubmitDate**: 2023-11-03    [abs](http://arxiv.org/abs/2303.11143v2) [paper-pdf](http://arxiv.org/pdf/2303.11143v2)

**Authors**: Gianluca Capozzi, Daniele Cono D'Elia, Giuseppe Antonio Di Luna, Leonardo Querzoni

**Abstract**: In recent years, binary analysis gained traction as a fundamental approach to inspect software and guarantee its security. Due to the exponential increase of devices running software, much research is now moving towards new autonomous solutions based on deep learning models, as they have been showing state-of-the-art performances in solving binary analysis problems. One of the hot topics in this context is binary similarity, which consists in determining if two functions in assembly code are compiled from the same source code. However, it is unclear how deep learning models for binary similarity behave in an adversarial context. In this paper, we study the resilience of binary similarity models against adversarial examples, showing that they are susceptible to both targeted and untargeted attacks (w.r.t. similarity goals) performed by black-box and white-box attackers. In more detail, we extensively test three current state-of-the-art solutions for binary similarity against two black-box greedy attacks, including a new technique that we call Spatial Greedy, and one white-box attack in which we repurpose a gradient-guided strategy used in attacks to image classifiers.



## **21. Adversarial Attacks on Cooperative Multi-agent Bandits**

cs.LG

**SubmitDate**: 2023-11-03    [abs](http://arxiv.org/abs/2311.01698v1) [paper-pdf](http://arxiv.org/pdf/2311.01698v1)

**Authors**: Jinhang Zuo, Zhiyao Zhang, Xuchuang Wang, Cheng Chen, Shuai Li, John C. S. Lui, Mohammad Hajiesmaili, Adam Wierman

**Abstract**: Cooperative multi-agent multi-armed bandits (CMA2B) consider the collaborative efforts of multiple agents in a shared multi-armed bandit game. We study latent vulnerabilities exposed by this collaboration and consider adversarial attacks on a few agents with the goal of influencing the decisions of the rest. More specifically, we study adversarial attacks on CMA2B in both homogeneous settings, where agents operate with the same arm set, and heterogeneous settings, where agents have distinct arm sets. In the homogeneous setting, we propose attack strategies that, by targeting just one agent, convince all agents to select a particular target arm $T-o(T)$ times while incurring $o(T)$ attack costs in $T$ rounds. In the heterogeneous setting, we prove that a target arm attack requires linear attack costs and propose attack strategies that can force a maximum number of agents to suffer linear regrets while incurring sublinear costs and only manipulating the observations of a few target agents. Numerical experiments validate the effectiveness of our proposed attack strategies.



## **22. Universal Perturbation-based Secret Key-Controlled Data Hiding**

cs.CR

18 pages, 8 tables, 10 figures

**SubmitDate**: 2023-11-03    [abs](http://arxiv.org/abs/2311.01696v1) [paper-pdf](http://arxiv.org/pdf/2311.01696v1)

**Authors**: Donghua Wang, Wen Yao, Tingsong Jiang, Xiaoqian Chen

**Abstract**: Deep neural networks (DNNs) are demonstrated to be vulnerable to universal perturbation, a single quasi-perceptible perturbation that can deceive the DNN on most images. However, the previous works are focused on using universal perturbation to perform adversarial attacks, while the potential usability of universal perturbation as data carriers in data hiding is less explored, especially for the key-controlled data hiding method. In this paper, we propose a novel universal perturbation-based secret key-controlled data-hiding method, realizing data hiding with a single universal perturbation and data decoding with the secret key-controlled decoder. Specifically, we optimize a single universal perturbation, which serves as a data carrier that can hide multiple secret images and be added to most cover images. Then, we devise a secret key-controlled decoder to extract different secret images from the single container image constructed by the universal perturbation by using different secret keys. Moreover, a suppress loss function is proposed to prevent the secret image from leakage. Furthermore, we adopt a robust module to boost the decoder's capability against corruption. Finally, A co-joint optimization strategy is proposed to find the optimal universal perturbation and decoder. Extensive experiments are conducted on different datasets to demonstrate the effectiveness of the proposed method. Additionally, the physical test performed on platforms (e.g., WeChat and Twitter) verifies the usability of the proposed method in practice.



## **23. Robust Adversarial Reinforcement Learning via Bounded Rationality Curricula**

cs.LG

Under review

**SubmitDate**: 2023-11-03    [abs](http://arxiv.org/abs/2311.01642v1) [paper-pdf](http://arxiv.org/pdf/2311.01642v1)

**Authors**: Aryaman Reddi, Maximilian Tölle, Jan Peters, Georgia Chalvatzaki, Carlo D'Eramo

**Abstract**: Robustness against adversarial attacks and distribution shifts is a long-standing goal of Reinforcement Learning (RL). To this end, Robust Adversarial Reinforcement Learning (RARL) trains a protagonist against destabilizing forces exercised by an adversary in a competitive zero-sum Markov game, whose optimal solution, i.e., rational strategy, corresponds to a Nash equilibrium. However, finding Nash equilibria requires facing complex saddle point optimization problems, which can be prohibitive to solve, especially for high-dimensional control. In this paper, we propose a novel approach for adversarial RL based on entropy regularization to ease the complexity of the saddle point optimization problem. We show that the solution of this entropy-regularized problem corresponds to a Quantal Response Equilibrium (QRE), a generalization of Nash equilibria that accounts for bounded rationality, i.e., agents sometimes play random actions instead of optimal ones. Crucially, the connection between the entropy-regularized objective and QRE enables free modulation of the rationality of the agents by simply tuning the temperature coefficient. We leverage this insight to propose our novel algorithm, Quantal Adversarial RL (QARL), which gradually increases the rationality of the adversary in a curriculum fashion until it is fully rational, easing the complexity of the optimization problem while retaining robustness. We provide extensive evidence of QARL outperforming RARL and recent baselines across several MuJoCo locomotion and navigation problems in overall performance and robustness.



## **24. Assist Is Just as Important as the Goal: Image Resurfacing to Aid Model's Robust Prediction**

cs.CV

**SubmitDate**: 2023-11-02    [abs](http://arxiv.org/abs/2311.01563v1) [paper-pdf](http://arxiv.org/pdf/2311.01563v1)

**Authors**: Abhijith Sharma, Phil Munz, Apurva Narayan

**Abstract**: Adversarial patches threaten visual AI models in the real world. The number of patches in a patch attack is variable and determines the attack's potency in a specific environment. Most existing defenses assume a single patch in the scene, and the multiple patch scenarios are shown to overcome them. This paper presents a model-agnostic defense against patch attacks based on total variation for image resurfacing (TVR). The TVR is an image-cleansing method that processes images to remove probable adversarial regions. TVR can be utilized solely or augmented with a defended model, providing multi-level security for robust prediction. TVR nullifies the influence of patches in a single image scan with no prior assumption on the number of patches in the scene. We validate TVR on the ImageNet-Patch benchmark dataset and with real-world physical objects, demonstrating its ability to mitigate patch attack.



## **25. E(2) Equivariant Neural Networks for Robust Galaxy Morphology Classification**

astro-ph.GA

10 pages, 4 figures, 3 tables, Accepted to the Machine Learning and  the Physical Sciences Workshop at NeurIPS 2023

**SubmitDate**: 2023-11-02    [abs](http://arxiv.org/abs/2311.01500v1) [paper-pdf](http://arxiv.org/pdf/2311.01500v1)

**Authors**: Sneh Pandya, Purvik Patel, Franc O, Jonathan Blazek

**Abstract**: We propose the use of group convolutional neural network architectures (GCNNs) equivariant to the 2D Euclidean group, $E(2)$, for the task of galaxy morphology classification by utilizing symmetries of the data present in galaxy images as an inductive bias in the architecture. We conduct robustness studies by introducing artificial perturbations via Poisson noise insertion and one-pixel adversarial attacks to simulate the effects of limited observational capabilities. We train, validate, and test GCNNs equivariant to discrete subgroups of $E(2)$ - the cyclic and dihedral groups of order $N$ - on the Galaxy10 DECals dataset and find that GCNNs achieve higher classification accuracy and are consistently more robust than their non-equivariant counterparts, with an architecture equivariant to the group $D_{16}$ achieving a $95.52 \pm 0.18\%$ test-set accuracy. We also find that the model loses $<6\%$ accuracy on a $50\%$-noise dataset and all GCNNs are less susceptible to one-pixel perturbations than an identically constructed CNN. Our code is publicly available at https://github.com/snehjp2/GCNNMorphology.



## **26. Like an Open Book? Read Neural Network Architecture with Simple Power Analysis on 32-bit Microcontrollers**

cs.CR

Accepted CARDIS 2023; ANR PICTURE PROJECT (ANR-20-CE39-0013)

**SubmitDate**: 2023-11-02    [abs](http://arxiv.org/abs/2311.01344v1) [paper-pdf](http://arxiv.org/pdf/2311.01344v1)

**Authors**: Raphael Joud, Pierre-Alain Moellic, Simon Pontie, Jean-Baptiste Rigaud

**Abstract**: Model extraction is a growing concern for the security of AI systems. For deep neural network models, the architecture is the most important information an adversary aims to recover. Being a sequence of repeated computation blocks, neural network models deployed on edge-devices will generate distinctive side-channel leakages. The latter can be exploited to extract critical information when targeted platforms are physically accessible. By combining theoretical knowledge about deep learning practices and analysis of a widespread implementation library (ARM CMSIS-NN), our purpose is to answer this critical question: how far can we extract architecture information by simply examining an EM side-channel trace? For the first time, we propose an extraction methodology for traditional MLP and CNN models running on a high-end 32-bit microcontroller (Cortex-M7) that relies only on simple pattern recognition analysis. Despite few challenging cases, we claim that, contrary to parameters extraction, the complexity of the attack is relatively low and we highlight the urgent need for practicable protections that could fit the strong memory and latency requirements of such platforms.



## **27. Towards Evaluating Transfer-based Attacks Systematically, Practically, and Fairly**

cs.LG

Accepted by NeurIPS 2023

**SubmitDate**: 2023-11-02    [abs](http://arxiv.org/abs/2311.01323v1) [paper-pdf](http://arxiv.org/pdf/2311.01323v1)

**Authors**: Qizhang Li, Yiwen Guo, Wangmeng Zuo, Hao Chen

**Abstract**: The adversarial vulnerability of deep neural networks (DNNs) has drawn great attention due to the security risk of applying these models in real-world applications. Based on transferability of adversarial examples, an increasing number of transfer-based methods have been developed to fool black-box DNN models whose architecture and parameters are inaccessible. Although tremendous effort has been exerted, there still lacks a standardized benchmark that could be taken advantage of to compare these methods systematically, fairly, and practically. Our investigation shows that the evaluation of some methods needs to be more reasonable and more thorough to verify their effectiveness, to avoid, for example, unfair comparison and insufficient consideration of possible substitute/victim models. Therefore, we establish a transfer-based attack benchmark (TA-Bench) which implements 30+ methods. In this paper, we evaluate and compare them comprehensively on 25 popular substitute/victim models on ImageNet. New insights about the effectiveness of these methods are gained and guidelines for future evaluations are provided. Code at: https://github.com/qizhangli/TA-Bench.



## **28. Improving Adversarial Transferability via Intermediate-level Perturbation Decay**

cs.LG

Accepted by NeurIPS 2023

**SubmitDate**: 2023-11-02    [abs](http://arxiv.org/abs/2304.13410v3) [paper-pdf](http://arxiv.org/pdf/2304.13410v3)

**Authors**: Qizhang Li, Yiwen Guo, Wangmeng Zuo, Hao Chen

**Abstract**: Intermediate-level attacks that attempt to perturb feature representations following an adversarial direction drastically have shown favorable performance in crafting transferable adversarial examples. Existing methods in this category are normally formulated with two separate stages, where a directional guide is required to be determined at first and the scalar projection of the intermediate-level perturbation onto the directional guide is enlarged thereafter. The obtained perturbation deviates from the guide inevitably in the feature space, and it is revealed in this paper that such a deviation may lead to sub-optimal attack. To address this issue, we develop a novel intermediate-level method that crafts adversarial examples within a single stage of optimization. In particular, the proposed method, named intermediate-level perturbation decay (ILPD), encourages the intermediate-level perturbation to be in an effective adversarial direction and to possess a great magnitude simultaneously. In-depth discussion verifies the effectiveness of our method. Experimental results show that it outperforms state-of-the-arts by large margins in attacking various victim models on ImageNet (+10.07% on average) and CIFAR-10 (+3.88% on average). Our code is at https://github.com/qizhangli/ILPD-attack.



## **29. Understanding and Improving Ensemble Adversarial Defense**

cs.LG

**SubmitDate**: 2023-11-02    [abs](http://arxiv.org/abs/2310.18477v2) [paper-pdf](http://arxiv.org/pdf/2310.18477v2)

**Authors**: Yian Deng, Tingting Mu

**Abstract**: The strategy of ensemble has become popular in adversarial defense, which trains multiple base classifiers to defend against adversarial attacks in a cooperative manner. Despite the empirical success, theoretical explanations on why an ensemble of adversarially trained classifiers is more robust than single ones remain unclear. To fill in this gap, we develop a new error theory dedicated to understanding ensemble adversarial defense, demonstrating a provable 0-1 loss reduction on challenging sample sets in an adversarial defense scenario. Guided by this theory, we propose an effective approach to improve ensemble adversarial defense, named interactive global adversarial training (iGAT). The proposal includes (1) a probabilistic distributing rule that selectively allocates to different base classifiers adversarial examples that are globally challenging to the ensemble, and (2) a regularization term to rescue the severest weaknesses of the base classifiers. Being tested over various existing ensemble adversarial defense techniques, iGAT is capable of boosting their performance by increases up to 17% evaluated using CIFAR10 and CIFAR100 datasets under both white-box and black-box attacks.



## **30. Detection Defenses: An Empty Promise against Adversarial Patch Attacks on Optical Flow**

cs.CV

Accepted to WACV 2024

**SubmitDate**: 2023-11-02    [abs](http://arxiv.org/abs/2310.17403v2) [paper-pdf](http://arxiv.org/pdf/2310.17403v2)

**Authors**: Erik Scheurer, Jenny Schmalfuss, Alexander Lis, Andrés Bruhn

**Abstract**: Adversarial patches undermine the reliability of optical flow predictions when placed in arbitrary scene locations. Therefore, they pose a realistic threat to real-world motion detection and its downstream applications. Potential remedies are defense strategies that detect and remove adversarial patches, but their influence on the underlying motion prediction has not been investigated. In this paper, we thoroughly examine the currently available detect-and-remove defenses ILP and LGS for a wide selection of state-of-the-art optical flow methods, and illuminate their side effects on the quality and robustness of the final flow predictions. In particular, we implement defense-aware attacks to investigate whether current defenses are able to withstand attacks that take the defense mechanism into account. Our experiments yield two surprising results: Detect-and-remove defenses do not only lower the optical flow quality on benign scenes, in doing so, they also harm the robustness under patch attacks for all tested optical flow methods except FlowNetC. As currently employed detect-and-remove defenses fail to deliver the promised adversarial robustness for optical flow, they evoke a false sense of security. The code is available at https://github.com/cv-stuttgart/DetectionDefenses.



## **31. Boosting Adversarial Transferability by Achieving Flat Local Maxima**

cs.CV

Accepted by the Neural Information Processing Systems (NeurIPS 2023)

**SubmitDate**: 2023-11-02    [abs](http://arxiv.org/abs/2306.05225v2) [paper-pdf](http://arxiv.org/pdf/2306.05225v2)

**Authors**: Zhijin Ge, Hongying Liu, Xiaosen Wang, Fanhua Shang, Yuanyuan Liu

**Abstract**: Transfer-based attack adopts the adversarial examples generated on the surrogate model to attack various models, making it applicable in the physical world and attracting increasing interest. Recently, various adversarial attacks have emerged to boost adversarial transferability from different perspectives. In this work, inspired by the observation that flat local minima are correlated with good generalization, we assume and empirically validate that adversarial examples at a flat local region tend to have good transferability by introducing a penalized gradient norm to the original loss function. Since directly optimizing the gradient regularization norm is computationally expensive and intractable for generating adversarial examples, we propose an approximation optimization method to simplify the gradient update of the objective function. Specifically, we randomly sample an example and adopt a first-order procedure to approximate the curvature of Hessian/vector product, which makes computing more efficient by interpolating two neighboring gradients. Meanwhile, in order to obtain a more stable gradient direction, we randomly sample multiple examples and average the gradients of these examples to reduce the variance due to random sampling during the iterative process. Extensive experimental results on the ImageNet-compatible dataset show that the proposed method can generate adversarial examples at flat local regions, and significantly improve the adversarial transferability on either normally trained models or adversarially trained models than the state-of-the-art attacks. Our codes are available at: https://github.com/Trustworthy-AI-Group/PGN.



## **32. Tensor Trust: Interpretable Prompt Injection Attacks from an Online Game**

cs.LG

**SubmitDate**: 2023-11-02    [abs](http://arxiv.org/abs/2311.01011v1) [paper-pdf](http://arxiv.org/pdf/2311.01011v1)

**Authors**: Sam Toyer, Olivia Watkins, Ethan Adrian Mendes, Justin Svegliato, Luke Bailey, Tiffany Wang, Isaac Ong, Karim Elmaaroufi, Pieter Abbeel, Trevor Darrell, Alan Ritter, Stuart Russell

**Abstract**: While Large Language Models (LLMs) are increasingly being used in real-world applications, they remain vulnerable to prompt injection attacks: malicious third party prompts that subvert the intent of the system designer. To help researchers study this problem, we present a dataset of over 126,000 prompt injection attacks and 46,000 prompt-based "defenses" against prompt injection, all created by players of an online game called Tensor Trust. To the best of our knowledge, this is currently the largest dataset of human-generated adversarial examples for instruction-following LLMs. The attacks in our dataset have a lot of easily interpretable stucture, and shed light on the weaknesses of LLMs. We also use the dataset to create a benchmark for resistance to two types of prompt injection, which we refer to as prompt extraction and prompt hijacking. Our benchmark results show that many models are vulnerable to the attack strategies in the Tensor Trust dataset. Furthermore, we show that some attack strategies from the dataset generalize to deployed LLM-based applications, even though they have a very different set of constraints to the game. We release all data and source code at https://tensortrust.ai/paper



## **33. Private Graph Extraction via Feature Explanations**

cs.LG

Accepted at PETS 2023

**SubmitDate**: 2023-11-02    [abs](http://arxiv.org/abs/2206.14724v2) [paper-pdf](http://arxiv.org/pdf/2206.14724v2)

**Authors**: Iyiola E. Olatunji, Mandeep Rathee, Thorben Funke, Megha Khosla

**Abstract**: Privacy and interpretability are two important ingredients for achieving trustworthy machine learning. We study the interplay of these two aspects in graph machine learning through graph reconstruction attacks. The goal of the adversary here is to reconstruct the graph structure of the training data given access to model explanations. Based on the different kinds of auxiliary information available to the adversary, we propose several graph reconstruction attacks. We show that additional knowledge of post-hoc feature explanations substantially increases the success rate of these attacks. Further, we investigate in detail the differences between attack performance with respect to three different classes of explanation methods for graph neural networks: gradient-based, perturbation-based, and surrogate model-based methods. While gradient-based explanations reveal the most in terms of the graph structure, we find that these explanations do not always score high in utility. For the other two classes of explanations, privacy leakage increases with an increase in explanation utility. Finally, we propose a defense based on a randomized response mechanism for releasing the explanations, which substantially reduces the attack success rate. Our code is available at https://github.com/iyempissy/graph-stealing-attacks-with-explanation



## **34. Adversary ML Resilience in Autonomous Driving Through Human Centered Perception Mechanisms**

cs.CV

15 pages, 17 figures

**SubmitDate**: 2023-11-02    [abs](http://arxiv.org/abs/2311.01478v1) [paper-pdf](http://arxiv.org/pdf/2311.01478v1)

**Authors**: Aakriti Shah

**Abstract**: Physical adversarial attacks on road signs are continuously exploiting vulnerabilities in modern day autonomous vehicles (AVs) and impeding their ability to correctly classify what type of road sign they encounter. Current models cannot generalize input data well, resulting in overfitting or underfitting. In overfitting, the model memorizes the input data but cannot generalize to new scenarios. In underfitting, the model does not learn enough of the input data to accurately classify these road signs. This paper explores the resilience of autonomous driving systems against three main physical adversarial attacks (tape, graffiti, illumination), specifically targeting object classifiers. Several machine learning models were developed and evaluated on two distinct datasets: road signs (stop signs, speed limit signs, traffic lights, and pedestrian crosswalk signs) and geometric shapes (octagons, circles, squares, and triangles). The study compared algorithm performance under different conditions, including clean and adversarial training and testing on these datasets. To build robustness against attacks, defense techniques like adversarial training and transfer learning were implemented. Results demonstrated transfer learning models played a crucial role in performance by allowing knowledge gained from shape training to improve generalizability of road sign classification, despite the datasets being completely different. The paper suggests future research directions, including human-in-the-loop validation, security analysis, real-world testing, and explainable AI for transparency. This study aims to contribute to improving security and robustness of object classifiers in autonomous vehicles and mitigating adversarial example impacts on driving systems.



## **35. MIST: Defending Against Membership Inference Attacks Through Membership-Invariant Subspace Training**

cs.CR

**SubmitDate**: 2023-11-02    [abs](http://arxiv.org/abs/2311.00919v1) [paper-pdf](http://arxiv.org/pdf/2311.00919v1)

**Authors**: Jiacheng Li, Ninghui Li, Bruno Ribeiro

**Abstract**: In Member Inference (MI) attacks, the adversary try to determine whether an instance is used to train a machine learning (ML) model. MI attacks are a major privacy concern when using private data to train ML models. Most MI attacks in the literature take advantage of the fact that ML models are trained to fit the training data well, and thus have very low loss on training instances. Most defenses against MI attacks therefore try to make the model fit the training data less well. Doing so, however, generally results in lower accuracy. We observe that training instances have different degrees of vulnerability to MI attacks. Most instances will have low loss even when not included in training. For these instances, the model can fit them well without concerns of MI attacks. An effective defense only needs to (possibly implicitly) identify instances that are vulnerable to MI attacks and avoids overfitting them. A major challenge is how to achieve such an effect in an efficient training process. Leveraging two distinct recent advancements in representation learning: counterfactually-invariant representations and subspace learning methods, we introduce a novel Membership-Invariant Subspace Training (MIST) method to defend against MI attacks. MIST avoids overfitting the vulnerable instances without significant impact on other instances. We have conducted extensive experimental studies, comparing MIST with various other state-of-the-art (SOTA) MI defenses against several SOTA MI attacks. We find that MIST outperforms other defenses while resulting in minimal reduction in testing accuracy.



## **36. Optimal Cost Constrained Adversarial Attacks For Multiple Agent Systems**

cs.LG

Submitted to ICCASP2024

**SubmitDate**: 2023-11-01    [abs](http://arxiv.org/abs/2311.00859v1) [paper-pdf](http://arxiv.org/pdf/2311.00859v1)

**Authors**: Ziqing Lu, Guanlin Liu, Lifeng Cai, Weiyu Xu

**Abstract**: Finding optimal adversarial attack strategies is an important topic in reinforcement learning and the Markov decision process. Previous studies usually assume one all-knowing coordinator (attacker) for whom attacking different recipient (victim) agents incurs uniform costs. However, in reality, instead of using one limitless central attacker, the attacks often need to be performed by distributed attack agents. We formulate the problem of performing optimal adversarial agent-to-agent attacks using distributed attack agents, in which we impose distinct cost constraints on each different attacker-victim pair. We propose an optimal method integrating within-step static constrained attack-resource allocation optimization and between-step dynamic programming to achieve the optimal adversarial attack in a multi-agent system. Our numerical results show that the proposed attacks can significantly reduce the rewards received by the attacked agents.



## **37. Robustness Tests for Automatic Machine Translation Metrics with Adversarial Attacks**

cs.CL

Accepted in Findings of EMNLP 2023

**SubmitDate**: 2023-11-01    [abs](http://arxiv.org/abs/2311.00508v1) [paper-pdf](http://arxiv.org/pdf/2311.00508v1)

**Authors**: Yichen Huang, Timothy Baldwin

**Abstract**: We investigate MT evaluation metric performance on adversarially-synthesized texts, to shed light on metric robustness. We experiment with word- and character-level attacks on three popular machine translation metrics: BERTScore, BLEURT, and COMET. Our human experiments validate that automatic metrics tend to overpenalize adversarially-degraded translations. We also identify inconsistencies in BERTScore ratings, where it judges the original sentence and the adversarially-degraded one as similar, while judging the degraded translation as notably worse than the original with respect to the reference. We identify patterns of brittleness that motivate more robust metric development.



## **38. Improving Robustness for Vision Transformer with a Simple Dynamic Scanning Augmentation**

cs.CV

Accepted in Neurocomputing

**SubmitDate**: 2023-11-01    [abs](http://arxiv.org/abs/2311.00441v1) [paper-pdf](http://arxiv.org/pdf/2311.00441v1)

**Authors**: Shashank Kotyan, Danilo Vasconcellos Vargas

**Abstract**: Vision Transformer (ViT) has demonstrated promising performance in computer vision tasks, comparable to state-of-the-art neural networks. Yet, this new type of deep neural network architecture is vulnerable to adversarial attacks limiting its capabilities in terms of robustness. This article presents a novel contribution aimed at further improving the accuracy and robustness of ViT, particularly in the face of adversarial attacks. We propose an augmentation technique called `Dynamic Scanning Augmentation' that leverages dynamic input sequences to adaptively focus on different patches, thereby maintaining performance and robustness. Our detailed investigations reveal that this adaptability to the input sequence induces significant changes in the attention mechanism of ViT, even for the same image. We introduce four variations of Dynamic Scanning Augmentation, outperforming ViT in terms of both robustness to adversarial attacks and accuracy against natural images, with one variant showing comparable results. By integrating our augmentation technique, we observe a substantial increase in ViT's robustness, improving it from $17\%$ to $92\%$ measured across different types of adversarial attacks. These findings, together with other comprehensive tests, indicate that Dynamic Scanning Augmentation enhances accuracy and robustness by promoting a more adaptive type of attention. In conclusion, this work contributes to the ongoing research on Vision Transformers by introducing Dynamic Scanning Augmentation as a technique for improving the accuracy and robustness of ViT. The observed results highlight the potential of this approach in advancing computer vision tasks and merit further exploration in future studies.



## **39. NEO-KD: Knowledge-Distillation-Based Adversarial Training for Robust Multi-Exit Neural Networks**

cs.LG

10 pages, 4 figures, accepted by 37th Conference on Neural  Information Processing Systems (NeurIPS 2023)

**SubmitDate**: 2023-11-01    [abs](http://arxiv.org/abs/2311.00428v1) [paper-pdf](http://arxiv.org/pdf/2311.00428v1)

**Authors**: Seokil Ham, Jungwuk Park, Dong-Jun Han, Jaekyun Moon

**Abstract**: While multi-exit neural networks are regarded as a promising solution for making efficient inference via early exits, combating adversarial attacks remains a challenging problem. In multi-exit networks, due to the high dependency among different submodels, an adversarial example targeting a specific exit not only degrades the performance of the target exit but also reduces the performance of all other exits concurrently. This makes multi-exit networks highly vulnerable to simple adversarial attacks. In this paper, we propose NEO-KD, a knowledge-distillation-based adversarial training strategy that tackles this fundamental challenge based on two key contributions. NEO-KD first resorts to neighbor knowledge distillation to guide the output of the adversarial examples to tend to the ensemble outputs of neighbor exits of clean data. NEO-KD also employs exit-wise orthogonal knowledge distillation for reducing adversarial transferability across different submodels. The result is a significantly improved robustness against adversarial attacks. Experimental results on various datasets/models show that our method achieves the best adversarial accuracy with reduced computation budgets, compared to the baselines relying on existing adversarial training or knowledge distillation techniques for multi-exit networks.



## **40. Adversarial Examples in the Physical World: A Survey**

cs.CV

Adversarial examples, physical-world scenarios, attacks and defenses

**SubmitDate**: 2023-11-01    [abs](http://arxiv.org/abs/2311.01473v1) [paper-pdf](http://arxiv.org/pdf/2311.01473v1)

**Authors**: Jiakai Wang, Donghua Wang, Jin Hu, Siyang Wu, Tingsong Jiang, Wen Yao, Aishan Liu, Xianglong Liu

**Abstract**: Deep neural networks (DNNs) have demonstrated high vulnerability to adversarial examples. Besides the attacks in the digital world, the practical implications of adversarial examples in the physical world present significant challenges and safety concerns. However, current research on physical adversarial examples (PAEs) lacks a comprehensive understanding of their unique characteristics, leading to limited significance and understanding. In this paper, we address this gap by thoroughly examining the characteristics of PAEs within a practical workflow encompassing training, manufacturing, and re-sampling processes. By analyzing the links between physical adversarial attacks, we identify manufacturing and re-sampling as the primary sources of distinct attributes and particularities in PAEs. Leveraging this knowledge, we develop a comprehensive analysis and classification framework for PAEs based on their specific characteristics, covering over 100 studies on physical-world adversarial examples. Furthermore, we investigate defense strategies against PAEs and identify open challenges and opportunities for future research. We aim to provide a fresh, thorough, and systematic understanding of PAEs, thereby promoting the development of robust adversarial learning and its application in open-world scenarios.



## **41. Dynamics-aware Adversarial Attack of Adaptive Neural Networks**

cs.CV

arXiv admin note: text overlap with arXiv:2112.09428

**SubmitDate**: 2023-11-01    [abs](http://arxiv.org/abs/2210.08159v3) [paper-pdf](http://arxiv.org/pdf/2210.08159v3)

**Authors**: An Tao, Yueqi Duan, Yingqi Wang, Jiwen Lu, Jie Zhou

**Abstract**: In this paper, we investigate the dynamics-aware adversarial attack problem of adaptive neural networks. Most existing adversarial attack algorithms are designed under a basic assumption -- the network architecture is fixed throughout the attack process. However, this assumption does not hold for many recently proposed adaptive neural networks, which adaptively deactivate unnecessary execution units based on inputs to improve computational efficiency. It results in a serious issue of lagged gradient, making the learned attack at the current step ineffective due to the architecture change afterward. To address this issue, we propose a Leaded Gradient Method (LGM) and show the significant effects of the lagged gradient. More specifically, we reformulate the gradients to be aware of the potential dynamic changes of network architectures, so that the learned attack better "leads" the next step than the dynamics-unaware methods when network architecture changes dynamically. Extensive experiments on representative types of adaptive neural networks for both 2D images and 3D point clouds show that our LGM achieves impressive adversarial attack performance compared with the dynamic-unaware attack methods. Code is available at https://github.com/antao97/LGM.



## **42. LFAA: Crafting Transferable Targeted Adversarial Examples with Low-Frequency Perturbations**

cs.CV

**SubmitDate**: 2023-11-01    [abs](http://arxiv.org/abs/2310.20175v2) [paper-pdf](http://arxiv.org/pdf/2310.20175v2)

**Authors**: Kunyu Wang, Juluan Shi, Wenxuan Wang

**Abstract**: Deep neural networks are susceptible to adversarial attacks, which pose a significant threat to their security and reliability in real-world applications. The most notable adversarial attacks are transfer-based attacks, where an adversary crafts an adversarial example to fool one model, which can also fool other models. While previous research has made progress in improving the transferability of untargeted adversarial examples, the generation of targeted adversarial examples that can transfer between models remains a challenging task. In this work, we present a novel approach to generate transferable targeted adversarial examples by exploiting the vulnerability of deep neural networks to perturbations on high-frequency components of images. We observe that replacing the high-frequency component of an image with that of another image can mislead deep models, motivating us to craft perturbations containing high-frequency information to achieve targeted attacks. To this end, we propose a method called Low-Frequency Adversarial Attack (\name), which trains a conditional generator to generate targeted adversarial perturbations that are then added to the low-frequency component of the image. Extensive experiments on ImageNet demonstrate that our proposed approach significantly outperforms state-of-the-art methods, improving targeted attack success rates by a margin from 3.2\% to 15.5\%.



## **43. Magmaw: Modality-Agnostic Adversarial Attacks on Machine Learning-Based Wireless Communication Systems**

cs.CR

**SubmitDate**: 2023-11-01    [abs](http://arxiv.org/abs/2311.00207v1) [paper-pdf](http://arxiv.org/pdf/2311.00207v1)

**Authors**: Jung-Woo Chang, Ke Sun, Nasimeh Heydaribeni, Seira Hidano, Xinyu Zhang, Farinaz Koushanfar

**Abstract**: Machine Learning (ML) has been instrumental in enabling joint transceiver optimization by merging all physical layer blocks of the end-to-end wireless communication systems. Although there have been a number of adversarial attacks on ML-based wireless systems, the existing methods do not provide a comprehensive view including multi-modality of the source data, common physical layer components, and wireless domain constraints. This paper proposes Magmaw, the first black-box attack methodology capable of generating universal adversarial perturbations for any multimodal signal transmitted over a wireless channel. We further introduce new objectives for adversarial attacks on ML-based downstream applications. The resilience of the attack to the existing widely used defense methods of adversarial training and perturbation signal subtraction is experimentally verified. For proof-of-concept evaluation, we build a real-time wireless attack platform using a software-defined radio system. Experimental results demonstrate that Magmaw causes significant performance degradation even in the presence of the defense mechanisms. Surprisingly, Magmaw is also effective against encrypted communication channels and conventional communications.



## **44. Robust Safety Classifier for Large Language Models: Adversarial Prompt Shield**

cs.CL

11 pages, 2 figures

**SubmitDate**: 2023-10-31    [abs](http://arxiv.org/abs/2311.00172v1) [paper-pdf](http://arxiv.org/pdf/2311.00172v1)

**Authors**: Jinhwa Kim, Ali Derakhshan, Ian G. Harris

**Abstract**: Large Language Models' safety remains a critical concern due to their vulnerability to adversarial attacks, which can prompt these systems to produce harmful responses. In the heart of these systems lies a safety classifier, a computational model trained to discern and mitigate potentially harmful, offensive, or unethical outputs. However, contemporary safety classifiers, despite their potential, often fail when exposed to inputs infused with adversarial noise. In response, our study introduces the Adversarial Prompt Shield (APS), a lightweight model that excels in detection accuracy and demonstrates resilience against adversarial prompts. Additionally, we propose novel strategies for autonomously generating adversarial training datasets, named Bot Adversarial Noisy Dialogue (BAND) datasets. These datasets are designed to fortify the safety classifier's robustness, and we investigate the consequences of incorporating adversarial examples into the training process. Through evaluations involving Large Language Models, we demonstrate that our classifier has the potential to decrease the attack success rate resulting from adversarial attacks by up to 60%. This advancement paves the way for the next generation of more reliable and resilient conversational agents.



## **45. Amoeba: Circumventing ML-supported Network Censorship via Adversarial Reinforcement Learning**

cs.CR

**SubmitDate**: 2023-10-31    [abs](http://arxiv.org/abs/2310.20469v1) [paper-pdf](http://arxiv.org/pdf/2310.20469v1)

**Authors**: Haoyu Liu, Alec F. Diallo, Paul Patras

**Abstract**: Embedding covert streams into a cover channel is a common approach to circumventing Internet censorship, due to censors' inability to examine encrypted information in otherwise permitted protocols (Skype, HTTPS, etc.). However, recent advances in machine learning (ML) enable detecting a range of anti-censorship systems by learning distinct statistical patterns hidden in traffic flows. Therefore, designing obfuscation solutions able to generate traffic that is statistically similar to innocuous network activity, in order to deceive ML-based classifiers at line speed, is difficult.   In this paper, we formulate a practical adversarial attack strategy against flow classifiers as a method for circumventing censorship. Specifically, we cast the problem of finding adversarial flows that will be misclassified as a sequence generation task, which we solve with Amoeba, a novel reinforcement learning algorithm that we design. Amoeba works by interacting with censoring classifiers without any knowledge of their model structure, but by crafting packets and observing the classifiers' decisions, in order to guide the sequence generation process. Our experiments using data collected from two popular anti-censorship systems demonstrate that Amoeba can effectively shape adversarial flows that have on average 94% attack success rate against a range of ML algorithms. In addition, we show that these adversarial flows are robust in different network environments and possess transferability across various ML models, meaning that once trained against one, our agent can subvert other censoring classifiers without retraining.



## **46. On Extracting Specialized Code Abilities from Large Language Models: A Feasibility Study**

cs.SE

13 pages

**SubmitDate**: 2023-10-31    [abs](http://arxiv.org/abs/2303.03012v4) [paper-pdf](http://arxiv.org/pdf/2303.03012v4)

**Authors**: Zongjie Li, Chaozheng Wang, Pingchuan Ma, Chaowei Liu, Shuai Wang, Daoyuan Wu, Cuiyun Gao, Yang Liu

**Abstract**: Recent advances in large language models (LLMs) significantly boost their usage in software engineering. However, training a well-performing LLM demands a substantial workforce for data collection and annotation. Moreover, training datasets may be proprietary or partially open, and the process often requires a costly GPU cluster. The intellectual property value of commercial LLMs makes them attractive targets for imitation attacks, but creating an imitation model with comparable parameters still incurs high costs. This motivates us to explore a practical and novel direction: slicing commercial black-box LLMs using medium-sized backbone models. In this paper, we explore the feasibility of launching imitation attacks on LLMs to extract their specialized code abilities, such as"code synthesis" and "code translation." We systematically investigate the effectiveness of launching code ability extraction attacks under different code-related tasks with multiple query schemes, including zero-shot, in-context, and Chain-of-Thought. We also design response checks to refine the outputs, leading to an effective imitation training process. Our results show promising outcomes, demonstrating that with a reasonable number of queries, attackers can train a medium-sized backbone model to replicate specialized code behaviors similar to the target LLMs. We summarize our findings and insights to help researchers better understand the threats posed by imitation attacks, including revealing a practical attack surface for generating adversarial code examples against LLMs.



## **47. Robust nonparametric regression based on deep ReLU neural networks**

stat.ME

40 pages

**SubmitDate**: 2023-10-31    [abs](http://arxiv.org/abs/2310.20294v1) [paper-pdf](http://arxiv.org/pdf/2310.20294v1)

**Authors**: Juntong Chen

**Abstract**: In this paper, we consider robust nonparametric regression using deep neural networks with ReLU activation function. While several existing theoretically justified methods are geared towards robustness against identical heavy-tailed noise distributions, the rise of adversarial attacks has emphasized the importance of safeguarding estimation procedures against systematic contamination. We approach this statistical issue by shifting our focus towards estimating conditional distributions. To address it robustly, we introduce a novel estimation procedure based on $\ell$-estimation. Under a mild model assumption, we establish general non-asymptotic risk bounds for the resulting estimators, showcasing their robustness against contamination, outliers, and model misspecification. We then delve into the application of our approach using deep ReLU neural networks. When the model is well-specified and the regression function belongs to an $\alpha$-H\"older class, employing $\ell$-type estimation on suitable networks enables the resulting estimators to achieve the minimax optimal rate of convergence. Additionally, we demonstrate that deep $\ell$-type estimators can circumvent the curse of dimensionality by assuming the regression function closely resembles the composition of several H\"older functions. To attain this, new deep fully-connected ReLU neural networks have been designed to approximate this composition class. This approximation result can be of independent interest.



## **48. CFDP: Common Frequency Domain Pruning**

cs.CV

CVPR ECV 2023 Accepted Paper

**SubmitDate**: 2023-10-31    [abs](http://arxiv.org/abs/2306.04147v2) [paper-pdf](http://arxiv.org/pdf/2306.04147v2)

**Authors**: Samir Khaki, Weihan Luo

**Abstract**: As the saying goes, sometimes less is more -- and when it comes to neural networks, that couldn't be more true. Enter pruning, the art of selectively trimming away unnecessary parts of a network to create a more streamlined, efficient architecture. In this paper, we introduce a novel end-to-end pipeline for model pruning via the frequency domain. This work aims to shed light on the interoperability of intermediate model outputs and their significance beyond the spatial domain. Our method, dubbed Common Frequency Domain Pruning (CFDP) aims to extrapolate common frequency characteristics defined over the feature maps to rank the individual channels of a layer based on their level of importance in learning the representation. By harnessing the power of CFDP, we have achieved state-of-the-art results on CIFAR-10 with GoogLeNet reaching an accuracy of 95.25%, that is, +0.2% from the original model. We also outperform all benchmarks and match the original model's performance on ImageNet, using only 55% of the trainable parameters and 60% of the FLOPs. In addition to notable performances, models produced via CFDP exhibit robustness to a variety of configurations including pruning from untrained neural architectures, and resistance to adversarial attacks. The implementation code can be found at https://github.com/Skhaki18/CFDP.



## **49. BERT Lost Patience Won't Be Robust to Adversarial Slowdown**

cs.LG

Accepted to NeurIPS 2023 [Poster]

**SubmitDate**: 2023-10-31    [abs](http://arxiv.org/abs/2310.19152v2) [paper-pdf](http://arxiv.org/pdf/2310.19152v2)

**Authors**: Zachary Coalson, Gabriel Ritter, Rakesh Bobba, Sanghyun Hong

**Abstract**: In this paper, we systematically evaluate the robustness of multi-exit language models against adversarial slowdown. To audit their robustness, we design a slowdown attack that generates natural adversarial text bypassing early-exit points. We use the resulting WAFFLE attack as a vehicle to conduct a comprehensive evaluation of three multi-exit mechanisms with the GLUE benchmark against adversarial slowdown. We then show our attack significantly reduces the computational savings provided by the three methods in both white-box and black-box settings. The more complex a mechanism is, the more vulnerable it is to adversarial slowdown. We also perform a linguistic analysis of the perturbed text inputs, identifying common perturbation patterns that our attack generates, and comparing them with standard adversarial text attacks. Moreover, we show that adversarial training is ineffective in defeating our slowdown attack, but input sanitization with a conversational model, e.g., ChatGPT, can remove perturbations effectively. This result suggests that future work is needed for developing efficient yet robust multi-exit models. Our code is available at: https://github.com/ztcoalson/WAFFLE



## **50. Is Robustness Transferable across Languages in Multilingual Neural Machine Translation?**

cs.AI

**SubmitDate**: 2023-10-31    [abs](http://arxiv.org/abs/2310.20162v1) [paper-pdf](http://arxiv.org/pdf/2310.20162v1)

**Authors**: Leiyu Pan, Supryadi, Deyi Xiong

**Abstract**: Robustness, the ability of models to maintain performance in the face of perturbations, is critical for developing reliable NLP systems. Recent studies have shown promising results in improving the robustness of models through adversarial training and data augmentation. However, in machine translation, most of these studies have focused on bilingual machine translation with a single translation direction. In this paper, we investigate the transferability of robustness across different languages in multilingual neural machine translation. We propose a robustness transfer analysis protocol and conduct a series of experiments. In particular, we use character-, word-, and multi-level noises to attack the specific translation direction of the multilingual neural machine translation model and evaluate the robustness of other translation directions. Our findings demonstrate that the robustness gained in one translation direction can indeed transfer to other translation directions. Additionally, we empirically find scenarios where robustness to character-level noise and word-level noise is more likely to transfer.



