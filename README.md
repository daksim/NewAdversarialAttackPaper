# Latest Adversarial Attack Papers
**update at 2022-06-23 06:31:29**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Using EBGAN for Anomaly Intrusion Detection**

cs.CR

**SubmitDate**: 2022-06-21    [paper-pdf](http://arxiv.org/pdf/2206.10400v1)

**Authors**: Yi Cui, Wenfeng Shen, Jian Zhang, Weijia Lu, Chuang Liu, Lin Sun, Si Chen

**Abstracts**: As an active network security protection scheme, intrusion detection system (IDS) undertakes the important responsibility of detecting network attacks in the form of malicious network traffic. Intrusion detection technology is an important part of IDS. At present, many scholars have carried out extensive research on intrusion detection technology. However, developing an efficient intrusion detection method for massive network traffic data is still difficult. Since Generative Adversarial Networks (GANs) have powerful modeling capabilities for complex high-dimensional data, they provide new ideas for addressing this problem. In this paper, we put forward an EBGAN-based intrusion detection method, IDS-EBGAN, that classifies network records as normal traffic or malicious traffic. The generator in IDS-EBGAN is responsible for converting the original malicious network traffic in the training set into adversarial malicious examples. This is because we want to use adversarial learning to improve the ability of discriminator to detect malicious traffic. At the same time, the discriminator adopts Autoencoder model. During testing, IDS-EBGAN uses reconstruction error of discriminator to classify traffic records.



## **2. Problem-Space Evasion Attacks in the Android OS: a Survey**

cs.CR

**SubmitDate**: 2022-06-21    [paper-pdf](http://arxiv.org/pdf/2205.14576v2)

**Authors**: Harel Berger, Chen Hajaj, Amit Dvir

**Abstracts**: Android is the most popular OS worldwide. Therefore, it is a target for various kinds of malware. As a countermeasure, the security community works day and night to develop appropriate Android malware detection systems, with ML-based or DL-based systems considered as some of the most common types. Against these detection systems, intelligent adversaries develop a wide set of evasion attacks, in which an attacker slightly modifies a malware sample to evade its target detection system. In this survey, we address problem-space evasion attacks in the Android OS, where attackers manipulate actual APKs, rather than their extracted feature vector. We aim to explore this kind of attacks, frequently overlooked by the research community due to a lack of knowledge of the Android domain, or due to focusing on general mathematical evasion attacks - i.e., feature-space evasion attacks. We discuss the different aspects of problem-space evasion attacks, using a new taxonomy, which focuses on key ingredients of each problem-space attack, such as the attacker model, the attacker's mode of operation, and the functional assessment of post-attack applications.



## **3. Certifiably Robust Policy Learning against Adversarial Communication in Multi-agent Systems**

cs.LG

**SubmitDate**: 2022-06-21    [paper-pdf](http://arxiv.org/pdf/2206.10158v1)

**Authors**: Yanchao Sun, Ruijie Zheng, Parisa Hassanzadeh, Yongyuan Liang, Soheil Feizi, Sumitra Ganesh, Furong Huang

**Abstracts**: Communication is important in many multi-agent reinforcement learning (MARL) problems for agents to share information and make good decisions. However, when deploying trained communicative agents in a real-world application where noise and potential attackers exist, the safety of communication-based policies becomes a severe issue that is underexplored. Specifically, if communication messages are manipulated by malicious attackers, agents relying on untrustworthy communication may take unsafe actions that lead to catastrophic consequences. Therefore, it is crucial to ensure that agents will not be misled by corrupted communication, while still benefiting from benign communication. In this work, we consider an environment with $N$ agents, where the attacker may arbitrarily change the communication from any $C<\frac{N-1}{2}$ agents to a victim agent. For this strong threat model, we propose a certifiable defense by constructing a message-ensemble policy that aggregates multiple randomly ablated message sets. Theoretical analysis shows that this message-ensemble policy can utilize benign communication while being certifiably robust to adversarial communication, regardless of the attacking algorithm. Experiments in multiple environments verify that our defense significantly improves the robustness of trained policies against various types of attacks.



## **4. ProML: A Decentralised Platform for Provenance Management of Machine Learning Software Systems**

cs.SE

Accepted as full paper in ECSA 2022 conference. To be presented

**SubmitDate**: 2022-06-21    [paper-pdf](http://arxiv.org/pdf/2206.10110v1)

**Authors**: Nguyen Khoi Tran, Bushra Sabir, M. Ali Babar, Nini Cui, Mehran Abolhasan, Justin Lipman

**Abstracts**: Large-scale Machine Learning (ML) based Software Systems are increasingly developed by distributed teams situated in different trust domains. Insider threats can launch attacks from any domain to compromise ML assets (models and datasets). Therefore, practitioners require information about how and by whom ML assets were developed to assess their quality attributes such as security, safety, and fairness. Unfortunately, it is challenging for ML teams to access and reconstruct such historical information of ML assets (ML provenance) because it is generally fragmented across distributed ML teams and threatened by the same adversaries that attack ML assets. This paper proposes ProML, a decentralised platform that leverages blockchain and smart contracts to empower distributed ML teams to jointly manage a single source of truth about circulated ML assets' provenance without relying on a third party, which is vulnerable to insider threats and presents a single point of failure. We propose a novel architectural approach called Artefact-as-a-State-Machine to leverage blockchain transactions and smart contracts for managing ML provenance information and introduce a user-driven provenance capturing mechanism to integrate existing scripts and tools to ProML without compromising participants' control over their assets and toolchains. We evaluate the performance and overheads of ProML by benchmarking a proof-of-concept system on a global blockchain. Furthermore, we assessed ProML's security against a threat model of a distributed ML workflow.



## **5. Make Some Noise: Reliable and Efficient Single-Step Adversarial Training**

cs.LG

**SubmitDate**: 2022-06-20    [paper-pdf](http://arxiv.org/pdf/2202.01181v2)

**Authors**: Pau de Jorge, Adel Bibi, Riccardo Volpi, Amartya Sanyal, Philip H. S. Torr, Grégory Rogez, Puneet K. Dokania

**Abstracts**: Recently, Wong et al. showed that adversarial training with single-step FGSM leads to a characteristic failure mode named catastrophic overfitting (CO), in which a model becomes suddenly vulnerable to multi-step attacks. They showed that adding a random perturbation prior to FGSM (RS-FGSM) seemed to be sufficient to prevent CO. However, Andriushchenko and Flammarion observed that RS-FGSM still leads to CO for larger perturbations, and proposed an expensive regularizer (GradAlign) to avoid CO. In this work, we methodically revisit the role of noise and clipping in single-step adversarial training. Contrary to previous intuitions, we find that using a stronger noise around the clean sample combined with not clipping is highly effective in avoiding CO for large perturbation radii. Based on these observations, we then propose Noise-FGSM (N-FGSM) that, while providing the benefits of single-step adversarial training, does not suffer from CO. Empirical analyses on a large suite of experiments show that N-FGSM is able to match or surpass the performance of previous single-step methods while achieving a 3$\times$ speed-up. Code can be found in https://github.com/pdejorge/N-FGSM



## **6. Diversified Adversarial Attacks based on Conjugate Gradient Method**

cs.LG

**SubmitDate**: 2022-06-20    [paper-pdf](http://arxiv.org/pdf/2206.09628v1)

**Authors**: Keiichiro Yamamura, Haruki Sato, Nariaki Tateiwa, Nozomi Hata, Toru Mitsutake, Issa Oe, Hiroki Ishikura, Katsuki Fujisawa

**Abstracts**: Deep learning models are vulnerable to adversarial examples, and adversarial attacks used to generate such examples have attracted considerable research interest. Although existing methods based on the steepest descent have achieved high attack success rates, ill-conditioned problems occasionally reduce their performance. To address this limitation, we utilize the conjugate gradient (CG) method, which is effective for this type of problem, and propose a novel attack algorithm inspired by the CG method, named the Auto Conjugate Gradient (ACG) attack. The results of large-scale evaluation experiments conducted on the latest robust models show that, for most models, ACG was able to find more adversarial examples with fewer iterations than the existing SOTA algorithm Auto-PGD (APGD). We investigated the difference in search performance between ACG and APGD in terms of diversification and intensification, and define a measure called Diversity Index (DI) to quantify the degree of diversity. From the analysis of the diversity using this index, we show that the more diverse search of the proposed method remarkably improves its attack success rate.



## **7. On the Limitations of Stochastic Pre-processing Defenses**

cs.LG

**SubmitDate**: 2022-06-19    [paper-pdf](http://arxiv.org/pdf/2206.09491v1)

**Authors**: Yue Gao, Ilia Shumailov, Kassem Fawaz, Nicolas Papernot

**Abstracts**: Defending against adversarial examples remains an open problem. A common belief is that randomness at inference increases the cost of finding adversarial inputs. An example of such a defense is to apply a random transformation to inputs prior to feeding them to the model. In this paper, we empirically and theoretically investigate such stochastic pre-processing defenses and demonstrate that they are flawed. First, we show that most stochastic defenses are weaker than previously thought; they lack sufficient randomness to withstand even standard attacks like projected gradient descent. This casts doubt on a long-held assumption that stochastic defenses invalidate attacks designed to evade deterministic defenses and force attackers to integrate the Expectation over Transformation (EOT) concept. Second, we show that stochastic defenses confront a trade-off between adversarial robustness and model invariance; they become less effective as the defended model acquires more invariance to their randomization. Future work will need to decouple these two effects. Our code is available in the supplementary material.



## **8. A Universal Adversarial Policy for Text Classifiers**

cs.LG

Accepted for publication in Neural Networks (2022), see  https://doi.org/10.1016/j.neunet.2022.06.018

**SubmitDate**: 2022-06-19    [paper-pdf](http://arxiv.org/pdf/2206.09458v1)

**Authors**: Gallil Maimon, Lior Rokach

**Abstracts**: Discovering the existence of universal adversarial perturbations had large theoretical and practical impacts on the field of adversarial learning. In the text domain, most universal studies focused on adversarial prefixes which are added to all texts. However, unlike the vision domain, adding the same perturbation to different inputs results in noticeably unnatural inputs. Therefore, we introduce a new universal adversarial setup - a universal adversarial policy, which has many advantages of other universal attacks but also results in valid texts - thus making it relevant in practice. We achieve this by learning a single search policy over a predefined set of semantics preserving text alterations, on many texts. This formulation is universal in that the policy is successful in finding adversarial examples on new texts efficiently. Our approach uses text perturbations which were extensively shown to produce natural attacks in the non-universal setup (specific synonym replacements). We suggest a strong baseline approach for this formulation which uses reinforcement learning. It's ability to generalise (from as few as 500 training texts) shows that universal adversarial patterns exist in the text domain as well.



## **9. JPEG Compression-Resistant Low-Mid Adversarial Perturbation against Unauthorized Face Recognition System**

cs.CV

**SubmitDate**: 2022-06-19    [paper-pdf](http://arxiv.org/pdf/2206.09410v1)

**Authors**: Jiaming Zhang, Qi Yi, Jitao Sang

**Abstracts**: It has been observed that the unauthorized use of face recognition system raises privacy problems. Using adversarial perturbations provides one possible solution to address this issue. A critical issue to exploit adversarial perturbation against unauthorized face recognition system is that: The images uploaded to the web need to be processed by JPEG compression, which weakens the effectiveness of adversarial perturbation. Existing JPEG compression-resistant methods fails to achieve a balance among compression resistance, transferability, and attack effectiveness. To this end, we propose a more natural solution called low frequency adversarial perturbation (LFAP). Instead of restricting the adversarial perturbations, we turn to regularize the source model to employing more low-frequency features by adversarial training. Moreover, to better influence model in different frequency components, we proposed the refined low-mid frequency adversarial perturbation (LMFAP) considering the mid frequency components as the productive complement. We designed a variety of settings in this study to simulate the real-world application scenario, including cross backbones, supervisory heads, training datasets and testing datasets. Quantitative and qualitative experimental results validate the effectivenss of proposed solutions.



## **10. Towards Adversarial Attack on Vision-Language Pre-training Models**

cs.LG

**SubmitDate**: 2022-06-19    [paper-pdf](http://arxiv.org/pdf/2206.09391v1)

**Authors**: Jiaming Zhang, Qi Yi, Jitao Sang

**Abstracts**: While vision-language pre-training model (VLP) has shown revolutionary improvements on various vision-language (V+L) tasks, the studies regarding its adversarial robustness remain largely unexplored. This paper studied the adversarial attack on popular VLP models and V+L tasks. First, we analyzed the performance of adversarial attacks under different settings. By examining the influence of different perturbed objects and attack targets, we concluded some key observations as guidance on both designing strong multimodal adversarial attack and constructing robust VLP models. Second, we proposed a novel multimodal attack method on the VLP models called Collaborative Multimodal Adversarial Attack (Co-Attack), which collectively carries out the attacks on the image modality and the text modality. Experimental results demonstrated that the proposed method achieves improved attack performances on different V+L downstream tasks and VLP models. The analysis observations and novel attack method hopefully provide new understanding into the adversarial robustness of VLP models, so as to contribute their safe and reliable deployment in more real-world scenarios.



## **11. Efficient and Transferable Adversarial Examples from Bayesian Neural Networks**

cs.LG

Accepted at UAI 2022

**SubmitDate**: 2022-06-18    [paper-pdf](http://arxiv.org/pdf/2011.05074v4)

**Authors**: Martin Gubri, Maxime Cordy, Mike Papadakis, Yves Le Traon, Koushik Sen

**Abstracts**: An established way to improve the transferability of black-box evasion attacks is to craft the adversarial examples on an ensemble-based surrogate to increase diversity. We argue that transferability is fundamentally related to uncertainty. Based on a state-of-the-art Bayesian Deep Learning technique, we propose a new method to efficiently build a surrogate by sampling approximately from the posterior distribution of neural network weights, which represents the belief about the value of each parameter. Our extensive experiments on ImageNet, CIFAR-10 and MNIST show that our approach improves the success rates of four state-of-the-art attacks significantly (up to 83.2 percentage points), in both intra-architecture and inter-architecture transferability. On ImageNet, our approach can reach 94% of success rate while reducing training computations from 11.6 to 2.4 exaflops, compared to an ensemble of independently trained DNNs. Our vanilla surrogate achieves 87.5% of the time higher transferability than three test-time techniques designed for this purpose. Our work demonstrates that the way to train a surrogate has been overlooked, although it is an important element of transfer-based attacks. We are, therefore, the first to review the effectiveness of several training methods in increasing transferability. We provide new directions to better understand the transferability phenomenon and offer a simple but strong baseline for future work.



## **12. DECK: Model Hardening for Defending Pervasive Backdoors**

cs.CR

**SubmitDate**: 2022-06-18    [paper-pdf](http://arxiv.org/pdf/2206.09272v1)

**Authors**: Guanhong Tao, Yingqi Liu, Siyuan Cheng, Shengwei An, Zhuo Zhang, Qiuling Xu, Guangyu Shen, Xiangyu Zhang

**Abstracts**: Pervasive backdoors are triggered by dynamic and pervasive input perturbations. They can be intentionally injected by attackers or naturally exist in normally trained models. They have a different nature from the traditional static and localized backdoors that can be triggered by perturbing a small input area with some fixed pattern, e.g., a patch with solid color. Existing defense techniques are highly effective for traditional backdoors. However, they may not work well for pervasive backdoors, especially regarding backdoor removal and model hardening. In this paper, we propose a novel model hardening technique against pervasive backdoors, including both natural and injected backdoors. We develop a general pervasive attack based on an encoder-decoder architecture enhanced with a special transformation layer. The attack can model a wide range of existing pervasive backdoor attacks and quantify them by class distances. As such, using the samples derived from our attack in adversarial training can harden a model against these backdoor vulnerabilities. Our evaluation on 9 datasets with 15 model structures shows that our technique can enlarge class distances by 59.65% on average with less than 1% accuracy degradation and no robustness loss, outperforming five hardening techniques such as adversarial training, universal adversarial training, MOTH, etc. It can reduce the attack success rate of six pervasive backdoor attacks from 99.06% to 1.94%, surpassing seven state-of-the-art backdoor removal techniques.



## **13. On the Role of Generalization in Transferability of Adversarial Examples**

cs.LG

**SubmitDate**: 2022-06-18    [paper-pdf](http://arxiv.org/pdf/2206.09238v1)

**Authors**: Yilin Wang, Farzan Farnia

**Abstracts**: Black-box adversarial attacks designing adversarial examples for unseen neural networks (NNs) have received great attention over the past years. While several successful black-box attack schemes have been proposed in the literature, the underlying factors driving the transferability of black-box adversarial examples still lack a thorough understanding. In this paper, we aim to demonstrate the role of the generalization properties of the substitute classifier used for generating adversarial examples in the transferability of the attack scheme to unobserved NN classifiers. To do this, we apply the max-min adversarial example game framework and show the importance of the generalization properties of the substitute NN in the success of the black-box attack scheme in application to different NN classifiers. We prove theoretical generalization bounds on the difference between the attack transferability rates on training and test samples. Our bounds suggest that a substitute NN with better generalization behavior could result in more transferable adversarial examples. In addition, we show that standard operator norm-based regularization methods could improve the transferability of the designed adversarial examples. We support our theoretical results by performing several numerical experiments showing the role of the substitute network's generalization in generating transferable adversarial examples. Our empirical results indicate the power of Lipschitz regularization methods in improving the transferability of adversarial examples.



## **14. Measuring Lower Bounds of Local Differential Privacy via Adversary Instantiations in Federated Learning**

cs.CR

15 pages, 7 figures

**SubmitDate**: 2022-06-18    [paper-pdf](http://arxiv.org/pdf/2206.09122v1)

**Authors**: Marin Matsumoto, Tsubasa Takahashi, Seng Pei Liew, Masato Oguchi

**Abstracts**: Local differential privacy (LDP) gives a strong privacy guarantee to be used in a distributed setting like federated learning (FL). LDP mechanisms in FL protect a client's gradient by randomizing it on the client; however, how can we interpret the privacy level given by the randomization? Moreover, what types of attacks can we mitigate in practice? To answer these questions, we introduce an empirical privacy test by measuring the lower bounds of LDP. The privacy test estimates how an adversary predicts if a reported randomized gradient was crafted from a raw gradient $g_1$ or $g_2$. We then instantiate six adversaries in FL under LDP to measure empirical LDP at various attack surfaces, including a worst-case attack that reaches the theoretical upper bound of LDP. The empirical privacy test with the adversary instantiations enables us to interpret LDP more intuitively and discuss relaxation of the privacy parameter until a particular instantiated attack surfaces. We also demonstrate numerical observations of the measured privacy in these adversarial settings, and the worst-case attack is not realistic in FL. In the end, we also discuss the possible relaxation of privacy levels in FL under LDP.



## **15. Existence and Minimax Theorems for Adversarial Surrogate Risks in Binary Classification**

cs.LG

37 pages, 1 Figure

**SubmitDate**: 2022-06-18    [paper-pdf](http://arxiv.org/pdf/2206.09098v1)

**Authors**: Natalie S. Frank

**Abstracts**: Adversarial training is one of the most popular methods for training methods robust to adversarial attacks, however, it is not well-understood from a theoretical perspective. We prove and existence, regularity, and minimax theorems for adversarial surrogate risks. Our results explain some empirical observations on adversarial robustness from prior work and suggest new directions in algorithm development. Furthermore, our results extend previously known existence and minimax theorems for the adversarial classification risk to surrogate risks.



## **16. Comment on Transferability and Input Transformation with Additive Noise**

cs.LG

**SubmitDate**: 2022-06-18    [paper-pdf](http://arxiv.org/pdf/2206.09075v1)

**Authors**: Hoki Kim, Jinseong Park, Jaewook Lee

**Abstracts**: Adversarial attacks have verified the existence of the vulnerability of neural networks. By adding small perturbations to a benign example, adversarial attacks successfully generate adversarial examples that lead misclassification of deep learning models. More importantly, an adversarial example generated from a specific model can also deceive other models without modification. We call this phenomenon ``transferability". Here, we analyze the relationship between transferability and input transformation with additive noise by mathematically proving that the modified optimization can produce more transferable adversarial examples.



## **17. Learning Generative Deception Strategies in Combinatorial Masking Games**

cs.GT

GameSec 2021

**SubmitDate**: 2022-06-17    [paper-pdf](http://arxiv.org/pdf/2109.11637v2)

**Authors**: Junlin Wu, Charles Kamhoua, Murat Kantarcioglu, Yevgeniy Vorobeychik

**Abstracts**: Deception is a crucial tool in the cyberdefence repertoire, enabling defenders to leverage their informational advantage to reduce the likelihood of successful attacks. One way deception can be employed is through obscuring, or masking, some of the information about how systems are configured, increasing attacker's uncertainty about their targets. We present a novel game-theoretic model of the resulting defender-attacker interaction, where the defender chooses a subset of attributes to mask, while the attacker responds by choosing an exploit to execute. The strategies of both players have combinatorial structure with complex informational dependencies, and therefore even representing these strategies is not trivial. First, we show that the problem of computing an equilibrium of the resulting zero-sum defender-attacker game can be represented as a linear program with a combinatorial number of system configuration variables and constraints, and develop a constraint generation approach for solving this problem. Next, we present a novel highly scalable approach for approximately solving such games by representing the strategies of both players as neural networks. The key idea is to represent the defender's mixed strategy using a deep neural network generator, and then using alternating gradient-descent-ascent algorithm, analogous to the training of Generative Adversarial Networks. Our experiments, as well as a case study, demonstrate the efficacy of the proposed approach.



## **18. Is Multi-Modal Necessarily Better? Robustness Evaluation of Multi-modal Fake News Detection**

cs.AI

**SubmitDate**: 2022-06-17    [paper-pdf](http://arxiv.org/pdf/2206.08788v1)

**Authors**: Jinyin Chen, Chengyu Jia, Haibin Zheng, Ruoxi Chen, Chenbo Fu

**Abstracts**: The proliferation of fake news and its serious negative social influence push fake news detection methods to become necessary tools for web managers. Meanwhile, the multi-media nature of social media makes multi-modal fake news detection popular for its ability to capture more modal features than uni-modal detection methods. However, current literature on multi-modal detection is more likely to pursue the detection accuracy but ignore the robustness of the detector. To address this problem, we propose a comprehensive robustness evaluation of multi-modal fake news detectors. In this work, we simulate the attack methods of malicious users and developers, i.e., posting fake news and injecting backdoors. Specifically, we evaluate multi-modal detectors with five adversarial and two backdoor attack methods. Experiment results imply that: (1) The detection performance of the state-of-the-art detectors degrades significantly under adversarial attacks, even worse than general detectors; (2) Most multi-modal detectors are more vulnerable when subjected to attacks on visual modality than textual modality; (3) Popular events' images will cause significant degradation to the detectors when they are subjected to backdoor attacks; (4) The performance of these detectors under multi-modal attacks is worse than under uni-modal attacks; (5) Defensive methods will improve the robustness of the multi-modal detectors.



## **19. Detecting Adversarial Examples in Batches -- a geometrical approach**

cs.LG

Submitted to AdvML workshop at ICML2022

**SubmitDate**: 2022-06-17    [paper-pdf](http://arxiv.org/pdf/2206.08738v1)

**Authors**: Danush Kumar Venkatesh, Peter Steinbach

**Abstracts**: Many deep learning methods have successfully solved complex tasks in computer vision and speech recognition applications. Nonetheless, the robustness of these models has been found to be vulnerable to perturbed inputs or adversarial examples, which are imperceptible to the human eye, but lead the model to erroneous output decisions. In this study, we adapt and introduce two geometric metrics, density and coverage, and evaluate their use in detecting adversarial samples in batches of unseen data. We empirically study these metrics using MNIST and two real-world biomedical datasets from MedMNIST, subjected to two different adversarial attacks. Our experiments show promising results for both metrics to detect adversarial examples. We believe that his work can lay the ground for further study on these metrics' use in deployed machine learning systems to monitor for possible attacks by adversarial examples or related pathologies such as dataset shift.



## **20. Minimum Noticeable Difference based Adversarial Privacy Preserving Image Generation**

cs.CV

**SubmitDate**: 2022-06-17    [paper-pdf](http://arxiv.org/pdf/2206.08638v1)

**Authors**: Wen Sun, Jian Jin, Weisi Lin

**Abstracts**: Deep learning models are found to be vulnerable to adversarial examples, as wrong predictions can be caused by small perturbation in input for deep learning models. Most of the existing works of adversarial image generation try to achieve attacks for most models, while few of them make efforts on guaranteeing the perceptual quality of the adversarial examples. High quality adversarial examples matter for many applications, especially for the privacy preserving. In this work, we develop a framework based on the Minimum Noticeable Difference (MND) concept to generate adversarial privacy preserving images that have minimum perceptual difference from the clean ones but are able to attack deep learning models. To achieve this, an adversarial loss is firstly proposed to make the deep learning models attacked by the adversarial images successfully. Then, a perceptual quality-preserving loss is developed by taking the magnitude of perturbation and perturbation-caused structural and gradient changes into account, which aims to preserve high perceptual quality for adversarial image generation. To the best of our knowledge, this is the first work on exploring quality-preserving adversarial image generation based on the MND concept for privacy preserving. To evaluate its performance in terms of perceptual quality, the deep models on image classification and face recognition are tested with the proposed method and several anchor methods in this work. Extensive experimental results demonstrate that the proposed MND framework is capable of generating adversarial images with remarkably improved performance metrics (e.g., PSNR, SSIM, and MOS) than that generated with the anchor methods.



## **21. Query-Efficient and Scalable Black-Box Adversarial Attacks on Discrete Sequential Data via Bayesian Optimization**

cs.LG

ICML 2022; Codes at  https://github.com/snu-mllab/DiscreteBlockBayesAttack

**SubmitDate**: 2022-06-17    [paper-pdf](http://arxiv.org/pdf/2206.08575v1)

**Authors**: Deokjae Lee, Seungyong Moon, Junhyeok Lee, Hyun Oh Song

**Abstracts**: We focus on the problem of adversarial attacks against models on discrete sequential data in the black-box setting where the attacker aims to craft adversarial examples with limited query access to the victim model. Existing black-box attacks, mostly based on greedy algorithms, find adversarial examples using pre-computed key positions to perturb, which severely limits the search space and might result in suboptimal solutions. To this end, we propose a query-efficient black-box attack using Bayesian optimization, which dynamically computes important positions using an automatic relevance determination (ARD) categorical kernel. We introduce block decomposition and history subsampling techniques to improve the scalability of Bayesian optimization when an input sequence becomes long. Moreover, we develop a post-optimization algorithm that finds adversarial examples with smaller perturbation size. Experiments on natural language and protein classification tasks demonstrate that our method consistently achieves higher attack success rate with significant reduction in query count and modification rate compared to the previous state-of-the-art methods.



## **22. Adversarial Attack and Defense for Non-Parametric Two-Sample Tests**

cs.LG

Accepted by ICML 2022

**SubmitDate**: 2022-06-17    [paper-pdf](http://arxiv.org/pdf/2202.03077v2)

**Authors**: Xilie Xu, Jingfeng Zhang, Feng Liu, Masashi Sugiyama, Mohan Kankanhalli

**Abstracts**: Non-parametric two-sample tests (TSTs) that judge whether two sets of samples are drawn from the same distribution, have been widely used in the analysis of critical data. People tend to employ TSTs as trusted basic tools and rarely have any doubt about their reliability. This paper systematically uncovers the failure mode of non-parametric TSTs through adversarial attacks and then proposes corresponding defense strategies. First, we theoretically show that an adversary can upper-bound the distributional shift which guarantees the attack's invisibility. Furthermore, we theoretically find that the adversary can also degrade the lower bound of a TST's test power, which enables us to iteratively minimize the test criterion in order to search for adversarial pairs. To enable TST-agnostic attacks, we propose an ensemble attack (EA) framework that jointly minimizes the different types of test criteria. Second, to robustify TSTs, we propose a max-min optimization that iteratively generates adversarial pairs to train the deep kernels. Extensive experiments on both simulated and real-world datasets validate the adversarial vulnerabilities of non-parametric TSTs and the effectiveness of our proposed defense. Source code is available at https://github.com/GodXuxilie/Robust-TST.git.



## **23. A Unified Evaluation of Textual Backdoor Learning: Frameworks and Benchmarks**

cs.LG

19 pages

**SubmitDate**: 2022-06-17    [paper-pdf](http://arxiv.org/pdf/2206.08514v1)

**Authors**: Ganqu Cui, Lifan Yuan, Bingxiang He, Yangyi Chen, Zhiyuan Liu, Maosong Sun

**Abstracts**: Textual backdoor attacks are a kind of practical threat to NLP systems. By injecting a backdoor in the training phase, the adversary could control model predictions via predefined triggers. As various attack and defense models have been proposed, it is of great significance to perform rigorous evaluations. However, we highlight two issues in previous backdoor learning evaluations: (1) The differences between real-world scenarios (e.g. releasing poisoned datasets or models) are neglected, and we argue that each scenario has its own constraints and concerns, thus requires specific evaluation protocols; (2) The evaluation metrics only consider whether the attacks could flip the models' predictions on poisoned samples and retain performances on benign samples, but ignore that poisoned samples should also be stealthy and semantic-preserving. To address these issues, we categorize existing works into three practical scenarios in which attackers release datasets, pre-trained models, and fine-tuned models respectively, then discuss their unique evaluation methodologies. On metrics, to completely evaluate poisoned samples, we use grammar error increase and perplexity difference for stealthiness, along with text similarity for validity. After formalizing the frameworks, we develop an open-source toolkit OpenBackdoor to foster the implementations and evaluations of textual backdoor learning. With this toolkit, we perform extensive experiments to benchmark attack and defense models under the suggested paradigm. To facilitate the underexplored defenses against poisoned datasets, we further propose CUBE, a simple yet strong clustering-based defense baseline. We hope that our frameworks and benchmarks could serve as the cornerstones for future model development and evaluations.



## **24. I Know What You Trained Last Summer: A Survey on Stealing Machine Learning Models and Defences**

cs.LG

Under review at ACM Computing Surveys

**SubmitDate**: 2022-06-16    [paper-pdf](http://arxiv.org/pdf/2206.08451v1)

**Authors**: Daryna Oliynyk, Rudolf Mayer, Andreas Rauber

**Abstracts**: Machine Learning-as-a-Service (MLaaS) has become a widespread paradigm, making even the most complex machine learning models available for clients via e.g. a pay-per-query principle. This allows users to avoid time-consuming processes of data collection, hyperparameter tuning, and model training. However, by giving their customers access to the (predictions of their) models, MLaaS providers endanger their intellectual property, such as sensitive training data, optimised hyperparameters, or learned model parameters. Adversaries can create a copy of the model with (almost) identical behavior using the the prediction labels only. While many variants of this attack have been described, only scattered defence strategies have been proposed, addressing isolated threats. This raises the necessity for a thorough systematisation of the field of model stealing, to arrive at a comprehensive understanding why these attacks are successful, and how they could be holistically defended against. We address this by categorising and comparing model stealing attacks, assessing their performance, and exploring corresponding defence techniques in different settings. We propose a taxonomy for attack and defence approaches, and provide guidelines on how to select the right attack or defence strategy based on the goal and available resources. Finally, we analyse which defences are rendered less effective by current attack strategies.



## **25. Boosting the Adversarial Transferability of Surrogate Model with Dark Knowledge**

cs.LG

26 pages, 5 figures

**SubmitDate**: 2022-06-16    [paper-pdf](http://arxiv.org/pdf/2206.08316v1)

**Authors**: Dingcheng Yang, Zihao Xiao, Wenjian Yu

**Abstracts**: Deep neural networks (DNNs) for image classification are known to be vulnerable to adversarial examples. And, the adversarial examples have transferability, which means an adversarial example for a DNN model can fool another black-box model with a non-trivial probability. This gave birth of the transfer-based adversarial attack where the adversarial examples generated by a pretrained or known model (called surrogate model) are used to conduct black-box attack. There are some work on how to generate the adversarial examples from a given surrogate model to achieve better transferability. However, training a special surrogate model to generate adversarial examples with better transferability is relatively under-explored. In this paper, we propose a method of training a surrogate model with abundant dark knowledge to boost the adversarial transferability of the adversarial examples generated by the surrogate model. This trained surrogate model is named dark surrogate model (DSM), and the proposed method to train DSM consists of two key components: a teacher model extracting dark knowledge and providing soft labels, and the mixing augmentation skill which enhances the dark knowledge of training data. Extensive experiments have been conducted to show that the proposed method can substantially improve the adversarial transferability of surrogate model across different architectures of surrogate model and optimizers for generating adversarial examples. We also show that the proposed method can be applied to other scenarios of transfer-based attack that contain dark knowledge, like face verification.



## **26. Adversarial Patch Attacks and Defences in Vision-Based Tasks: A Survey**

cs.CV

A. Sharma and Y. Bian share equal contribution

**SubmitDate**: 2022-06-16    [paper-pdf](http://arxiv.org/pdf/2206.08304v1)

**Authors**: Abhijith Sharma, Yijun Bian, Phil Munz, Apurva Narayan

**Abstracts**: Adversarial attacks in deep learning models, especially for safety-critical systems, are gaining more and more attention in recent years, due to the lack of trust in the security and robustness of AI models. Yet the more primitive adversarial attacks might be physically infeasible or require some resources that are hard to access like the training data, which motivated the emergence of patch attacks. In this survey, we provide a comprehensive overview to cover existing techniques of adversarial patch attacks, aiming to help interested researchers quickly catch up with the progress in this field. We also discuss existing techniques for developing detection and defences against adversarial patches, aiming to help the community better understand this field and its applications in the real world.



## **27. Adversarial Robustness of Graph-based Anomaly Detection**

cs.CR

arXiv admin note: substantial text overlap with arXiv:2106.09989

**SubmitDate**: 2022-06-16    [paper-pdf](http://arxiv.org/pdf/2206.08260v1)

**Authors**: Yulin Zhu, Yuni Lai, Kaifa Zhao, Xiapu Luo, Mingquan Yuan, Jian Ren, Kai Zhou

**Abstracts**: Graph-based anomaly detection is becoming prevalent due to the powerful representation abilities of graphs as well as recent advances in graph mining techniques. These GAD tools, however, expose a new attacking surface, ironically due to their unique advantage of being able to exploit the relations among data. That is, attackers now can manipulate those relations (i.e., the structure of the graph) to allow target nodes to evade detection or degenerate the classification performance of the detection. In this paper, we exploit this vulnerability by designing the structural poisoning attacks to a FeXtra-based GAD system termed OddBall as well as the black box attacks against GCN-based GAD systems by attacking the imbalanced lienarized GCN ( LGCN ). Specifically, we formulate the attack against OddBall and LGCN as a one-level optimization problem by incorporating different regression techniques, where the key technical challenge is to efficiently solve the problem in a discrete domain. We propose a novel attack method termed BinarizedAttack based on gradient descent. Comparing to prior arts, BinarizedAttack can better use the gradient information, making it particularly suitable for solving discrete optimization problems, thus opening the door to studying a new type of attack against security analytic tools that rely on graph data.



## **28. Adversarial attacks on voter model dynamics in complex networks**

physics.soc-ph

7 pages, 5 figures

**SubmitDate**: 2022-06-16    [paper-pdf](http://arxiv.org/pdf/2111.09561v2)

**Authors**: Katsumi Chiyomaru, Kazuhiro Takemoto

**Abstracts**: This study investigates adversarial attacks conducted to distort voter model dynamics in complex networks. Specifically, a simple adversarial attack method is proposed to hold the state of opinions of an individual closer to the target state in the voter model dynamics. This indicates that even when one opinion is the majority, the vote outcome can be inverted (i.e., the outcome can lean toward the other opinion) by adding extremely small (hard-to-detect) perturbations strategically generated in social networks. Adversarial attacks are relatively more effective in complex (large and dense) networks. These results indicate that opinion dynamics can be unknowingly distorted.



## **29. Adversarial Privacy Protection on Speech Enhancement**

cs.SD

5 pages, 6 figures

**SubmitDate**: 2022-06-16    [paper-pdf](http://arxiv.org/pdf/2206.08170v1)

**Authors**: Mingyu Dong, Diqun Yan, Rangding Wang

**Abstracts**: Speech is easily leaked imperceptibly, such as being recorded by mobile phones in different situations. Private content in speech may be maliciously extracted through speech enhancement technology. Speech enhancement technology has developed rapidly along with deep neural networks (DNNs), but adversarial examples can cause DNNs to fail. In this work, we propose an adversarial method to degrade speech enhancement systems. Experimental results show that generated adversarial examples can erase most content information in original examples or replace it with target speech content through speech enhancement. The word error rate (WER) between an enhanced original example and enhanced adversarial example recognition result can reach 89.0%. WER of target attack between enhanced adversarial example and target example is low to 33.75% . Adversarial perturbation can bring the rate of change to the original example to more than 1.4430. This work can prevent the malicious extraction of speech.



## **30. Detecting Adversarial Examples Is (Nearly) As Hard As Classifying Them**

cs.LG

ICML 2022 (Long Talk)

**SubmitDate**: 2022-06-16    [paper-pdf](http://arxiv.org/pdf/2107.11630v2)

**Authors**: Florian Tramèr

**Abstracts**: Making classifiers robust to adversarial examples is hard. Thus, many defenses tackle the seemingly easier task of detecting perturbed inputs. We show a barrier towards this goal. We prove a general hardness reduction between detection and classification of adversarial examples: given a robust detector for attacks at distance {\epsilon} (in some metric), we can build a similarly robust (but inefficient) classifier for attacks at distance {\epsilon}/2. Our reduction is computationally inefficient, and thus cannot be used to build practical classifiers. Instead, it is a useful sanity check to test whether empirical detection results imply something much stronger than the authors presumably anticipated. To illustrate, we revisit 13 detector defenses. For 11/13 cases, we show that the claimed detection results would imply an inefficient classifier with robustness far beyond the state-of-the-art.



## **31. Adversarial Attacks on Gaussian Process Bandits**

stat.ML

Accepted to ICML 2022

**SubmitDate**: 2022-06-16    [paper-pdf](http://arxiv.org/pdf/2110.08449v3)

**Authors**: Eric Han, Jonathan Scarlett

**Abstracts**: Gaussian processes (GP) are a widely-adopted tool used to sequentially optimize black-box functions, where evaluations are costly and potentially noisy. Recent works on GP bandits have proposed to move beyond random noise and devise algorithms robust to adversarial attacks. This paper studies this problem from the attacker's perspective, proposing various adversarial attack methods with differing assumptions on the attacker's strength and prior information. Our goal is to understand adversarial attacks on GP bandits from theoretical and practical perspectives. We focus primarily on targeted attacks on the popular GP-UCB algorithm and a related elimination-based algorithm, based on adversarially perturbing the function $f$ to produce another function $\tilde{f}$ whose optima are in some target region $\mathcal{R}_{\rm target}$. Based on our theoretical analysis, we devise both white-box attacks (known $f$) and black-box attacks (unknown $f$), with the former including a Subtraction attack and Clipping attack, and the latter including an Aggressive subtraction attack. We demonstrate that adversarial attacks on GP bandits can succeed in forcing the algorithm towards $\mathcal{R}_{\rm target}$ even with a low attack budget, and we test our attacks' effectiveness on a diverse range of objective functions.



## **32. Analysis and Extensions of Adversarial Training for Video Classification**

cs.CV

**SubmitDate**: 2022-06-16    [paper-pdf](http://arxiv.org/pdf/2206.07953v1)

**Authors**: Kaleab A. Kinfu, René Vidal

**Abstracts**: Adversarial training (AT) is a simple yet effective defense against adversarial attacks to image classification systems, which is based on augmenting the training set with attacks that maximize the loss. However, the effectiveness of AT as a defense for video classification has not been thoroughly studied. Our first contribution is to show that generating optimal attacks for video requires carefully tuning the attack parameters, especially the step size. Notably, we show that the optimal step size varies linearly with the attack budget. Our second contribution is to show that using a smaller (sub-optimal) attack budget at training time leads to a more robust performance at test time. Based on these findings, we propose three defenses against attacks with variable attack budgets. The first one, Adaptive AT, is a technique where the attack budget is drawn from a distribution that is adapted as training iterations proceed. The second, Curriculum AT, is a technique where the attack budget is increased as training iterations proceed. The third, Generative AT, further couples AT with a denoising generative adversarial network to boost robust performance. Experiments on the UCF101 dataset demonstrate that the proposed methods improve adversarial robustness against multiple attack types.



## **33. A Comprehensive Test Pattern Generation Approach Exploiting SAT Attack for Logic Locking**

cs.CR

10 pages, 8 figures

**SubmitDate**: 2022-06-16    [paper-pdf](http://arxiv.org/pdf/2204.11307v2)

**Authors**: Yadi Zhong, Ujjwal Guin

**Abstracts**: The need for reducing manufacturing defect escape in today's safety-critical applications requires increased fault coverage. However, generating a test set using commercial automatic test pattern generation (ATPG) tools that lead to zero-defect escape is still an open problem. It is challenging to detect all stuck-at faults to reach 100% fault coverage. In parallel, the hardware security community has been actively involved in developing solutions for logic locking to prevent IP piracy. Locks (e.g., XOR gates) are inserted in different locations of the netlist so that an adversary cannot determine the secret key. Unfortunately, the Boolean satisfiability (SAT) based attack, introduced in [1], can break different logic locking schemes in minutes. In this paper, we propose a novel test pattern generation approach using the powerful SAT attack on logic locking. A stuck-at fault is modeled as a locked gate with a secret key. Our modeling of stuck-at faults preserves the property of fault activation and propagation. We show that the input pattern that determines the key is a test for the stuck-at fault. We propose two different approaches for test pattern generation. First, a single stuck-at fault is targeted, and a corresponding locked circuit with one key bit is created. This approach generates one test pattern per fault. Second, we consider a group of faults and convert the circuit to its locked version with multiple key bits. The inputs obtained from the SAT tool are the test set for detecting this group of faults. Our approach is able to find test patterns for hard-to-detect faults that were previously failed in commercial ATPG tools. The proposed test pattern generation approach can efficiently detect redundant faults present in a circuit. We demonstrate the effectiveness of the approach on ITC'99 benchmarks. The results show that we can achieve a perfect fault coverage reaching 100%.



## **34. Architectural Backdoors in Neural Networks**

cs.LG

**SubmitDate**: 2022-06-15    [paper-pdf](http://arxiv.org/pdf/2206.07840v1)

**Authors**: Mikel Bober-Irizar, Ilia Shumailov, Yiren Zhao, Robert Mullins, Nicolas Papernot

**Abstracts**: Machine learning is vulnerable to adversarial manipulation. Previous literature has demonstrated that at the training stage attackers can manipulate data and data sampling procedures to control model behaviour. A common attack goal is to plant backdoors i.e. force the victim model to learn to recognise a trigger known only by the adversary. In this paper, we introduce a new class of backdoor attacks that hide inside model architectures i.e. in the inductive bias of the functions used to train. These backdoors are simple to implement, for instance by publishing open-source code for a backdoored model architecture that others will reuse unknowingly. We demonstrate that model architectural backdoors represent a real threat and, unlike other approaches, can survive a complete re-training from scratch. We formalise the main construction principles behind architectural backdoors, such as a link between the input and the output, and describe some possible protections against them. We evaluate our attacks on computer vision benchmarks of different scales and demonstrate the underlying vulnerability is pervasive in a variety of training settings.



## **35. Search-Based Testing Approach for Deep Reinforcement Learning Agents**

cs.SE

**SubmitDate**: 2022-06-15    [paper-pdf](http://arxiv.org/pdf/2206.07813v1)

**Authors**: Amirhossein Zolfagharian, Manel Abdellatif, Lionel Briand, Mojtaba Bagherzadeh, Ramesh S

**Abstracts**: Deep Reinforcement Learning (DRL) algorithms have been increasingly employed during the last decade to solve various decision-making problems such as autonomous driving and robotics. However, these algorithms have faced great challenges when deployed in safety-critical environments since they often exhibit erroneous behaviors that can lead to potentially critical errors. One way to assess the safety of DRL agents is to test them to detect possible faults leading to critical failures during their execution. This raises the question of how we can efficiently test DRL policies to ensure their correctness and adherence to safety requirements. Most existing works on testing DRL agents use adversarial attacks that perturb states or actions of the agent. However, such attacks often lead to unrealistic states of the environment. Their main goal is to test the robustness of DRL agents rather than testing the compliance of agents' policies with respect to requirements. Due to the huge state space of DRL environments, the high cost of test execution, and the black-box nature of DRL algorithms, the exhaustive testing of DRL agents is impossible. In this paper, we propose a Search-based Testing Approach of Reinforcement Learning Agents (STARLA) to test the policy of a DRL agent by effectively searching for failing executions of the agent within a limited testing budget. We use machine learning models and a dedicated genetic algorithm to narrow the search towards faulty episodes. We apply STARLA on a Deep-Q-Learning agent which is widely used as a benchmark and show that it significantly outperforms Random Testing by detecting more faults related to the agent's policy. We also investigate how to extract rules that characterize faulty episodes of the DRL agent using our search results. Such rules can be used to understand the conditions under which the agent fails and thus assess its deployment risks.



## **36. Poison Forensics: Traceback of Data Poisoning Attacks in Neural Networks**

cs.CR

18 pages

**SubmitDate**: 2022-06-15    [paper-pdf](http://arxiv.org/pdf/2110.06904v2)

**Authors**: Shawn Shan, Arjun Nitin Bhagoji, Haitao Zheng, Ben Y. Zhao

**Abstracts**: In adversarial machine learning, new defenses against attacks on deep learning systems are routinely broken soon after their release by more powerful attacks. In this context, forensic tools can offer a valuable complement to existing defenses, by tracing back a successful attack to its root cause, and offering a path forward for mitigation to prevent similar attacks in the future.   In this paper, we describe our efforts in developing a forensic traceback tool for poison attacks on deep neural networks. We propose a novel iterative clustering and pruning solution that trims "innocent" training samples, until all that remains is the set of poisoned data responsible for the attack. Our method clusters training samples based on their impact on model parameters, then uses an efficient data unlearning method to prune innocent clusters. We empirically demonstrate the efficacy of our system on three types of dirty-label (backdoor) poison attacks and three types of clean-label poison attacks, across domains of computer vision and malware classification. Our system achieves over 98.4% precision and 96.8% recall across all attacks. We also show that our system is robust against four anti-forensics measures specifically designed to attack it.



## **37. Learn to Adapt: Robust Drift Detection in Security Domain**

cs.CR

**SubmitDate**: 2022-06-15    [paper-pdf](http://arxiv.org/pdf/2206.07581v1)

**Authors**: Aditya Kuppa, Nhien-An Le-Khac

**Abstracts**: Deploying robust machine learning models has to account for concept drifts arising due to the dynamically changing and non-stationary nature of data. Addressing drifts is particularly imperative in the security domain due to the ever-evolving threat landscape and lack of sufficiently labeled training data at the deployment time leading to performance degradation. Recently proposed concept drift detection methods in literature tackle this problem by identifying the changes in feature/data distributions and periodically retraining the models to learn new concepts. While these types of strategies should absolutely be conducted when possible, they are not robust towards attacker-induced drifts and suffer from a delay in detecting new attacks. We aim to address these shortcomings in this work. we propose a robust drift detector that not only identifies drifted samples but also discovers new classes as they arrive in an on-line fashion. We evaluate the proposed method with two security-relevant data sets -- network intrusion data set released in 2018 and APT Command and Control dataset combined with web categorization data. Our evaluation shows that our drifting detection method is not only highly accurate but also robust towards adversarial drifts and discovers new classes from drifted samples.



## **38. Creating a Secure Underlay for the Internet**

cs.NI

Usenix Security 2022

**SubmitDate**: 2022-06-15    [paper-pdf](http://arxiv.org/pdf/2206.06879v2)

**Authors**: Henry Birge-Lee, Joel Wanner, Grace Cimaszewski, Jonghoon Kwon, Liang Wang, Francois Wirz, Prateek Mittal, Adrian Perrig, Yixin Sun

**Abstracts**: Adversaries can exploit inter-domain routing vulnerabilities to intercept communication and compromise the security of critical Internet applications. Meanwhile the deployment of secure routing solutions such as Border Gateway Protocol Security (BGPsec) and Scalability, Control and Isolation On Next-generation networks (SCION) are still limited. How can we leverage emerging secure routing backbones and extend their security properties to the broader Internet?   We design and deploy an architecture to bootstrap secure routing. Our key insight is to abstract the secure routing backbone as a virtual Autonomous System (AS), called Secure Backbone AS (SBAS). While SBAS appears as one AS to the Internet, it is a federated network where routes are exchanged between participants using a secure backbone. SBAS makes BGP announcements for its customers' IP prefixes at multiple locations (referred to as Points of Presence or PoPs) allowing traffic from non-participating hosts to be routed to a nearby SBAS PoP (where it is then routed over the secure backbone to the true prefix owner). In this manner, we are the first to integrate a federated secure non-BGP routing backbone with the BGP-speaking Internet.   We present a real-world deployment of our architecture that uses SCIONLab to emulate the secure backbone and the PEERING framework to make BGP announcements to the Internet. A combination of real-world attacks and Internet-scale simulations shows that SBAS substantially reduces the threat of routing attacks. Finally, we survey network operators to better understand optimal governance and incentive models.



## **39. TPC: Transformation-Specific Smoothing for Point Cloud Models**

cs.CV

Accepted as a conference paper at ICML 2022

**SubmitDate**: 2022-06-15    [paper-pdf](http://arxiv.org/pdf/2201.12733v3)

**Authors**: Wenda Chu, Linyi Li, Bo Li

**Abstracts**: Point cloud models with neural network architectures have achieved great success and have been widely used in safety-critical applications, such as Lidar-based recognition systems in autonomous vehicles. However, such models are shown vulnerable to adversarial attacks which aim to apply stealthy semantic transformations such as rotation and tapering to mislead model predictions. In this paper, we propose a transformation-specific smoothing framework TPC, which provides tight and scalable robustness guarantees for point cloud models against semantic transformation attacks. We first categorize common 3D transformations into three categories: additive (e.g., shearing), composable (e.g., rotation), and indirectly composable (e.g., tapering), and we present generic robustness certification strategies for all categories respectively. We then specify unique certification protocols for a range of specific semantic transformations and their compositions. Extensive experiments on several common 3D transformations show that TPC significantly outperforms the state of the art. For example, our framework boosts the certified accuracy against twisting transformation along z-axis (within 20$^\circ$) from 20.3$\%$ to 83.8$\%$. Codes and models are available at https://github.com/Qianhewu/Point-Cloud-Smoothing.



## **40. Towards Understanding Adversarial Robustness of Optical Flow Networks**

cs.CV

CVPR 2022

**SubmitDate**: 2022-06-15    [paper-pdf](http://arxiv.org/pdf/2103.16255v3)

**Authors**: Simon Schrodi, Tonmoy Saikia, Thomas Brox

**Abstracts**: Recent work demonstrated the lack of robustness of optical flow networks to physical patch-based adversarial attacks. The possibility to physically attack a basic component of automotive systems is a reason for serious concerns. In this paper, we analyze the cause of the problem and show that the lack of robustness is rooted in the classical aperture problem of optical flow estimation in combination with bad choices in the details of the network architecture. We show how these mistakes can be rectified in order to make optical flow networks robust to physical patch-based attacks. Additionally, we take a look at global white-box attacks in the scope of optical flow. We find that targeted white-box attacks can be crafted to bias flow estimation models towards any desired output, but this requires access to the input images and model weights. However, in the case of universal attacks, we find that optical flow networks are robust. Code is available at https://github.com/lmb-freiburg/understanding_flow_robustness.



## **41. Hardening DNNs against Transfer Attacks during Network Compression using Greedy Adversarial Pruning**

cs.LG

4 pages, 2 figures

**SubmitDate**: 2022-06-15    [paper-pdf](http://arxiv.org/pdf/2206.07406v1)

**Authors**: Jonah O'Brien Weiss, Tiago Alves, Sandip Kundu

**Abstracts**: The prevalence and success of Deep Neural Network (DNN) applications in recent years have motivated research on DNN compression, such as pruning and quantization. These techniques accelerate model inference, reduce power consumption, and reduce the size and complexity of the hardware necessary to run DNNs, all with little to no loss in accuracy. However, since DNNs are vulnerable to adversarial inputs, it is important to consider the relationship between compression and adversarial robustness. In this work, we investigate the adversarial robustness of models produced by several irregular pruning schemes and by 8-bit quantization. Additionally, while conventional pruning removes the least important parameters in a DNN, we investigate the effect of an unconventional pruning method: removing the most important model parameters based on the gradient on adversarial inputs. We call this method Greedy Adversarial Pruning (GAP) and we find that this pruning method results in models that are resistant to transfer attacks from their uncompressed counterparts.



## **42. Morphence-2.0: Evasion-Resilient Moving Target Defense Powered by Out-of-Distribution Detection**

cs.CR

13 pages, 6 figures, 2 tables. arXiv admin note: substantial text  overlap with arXiv:2108.13952

**SubmitDate**: 2022-06-15    [paper-pdf](http://arxiv.org/pdf/2206.07321v1)

**Authors**: Abderrahmen Amich, Ata Kaboudi, Birhanu Eshete

**Abstracts**: Evasion attacks against machine learning models often succeed via iterative probing of a fixed target model, whereby an attack that succeeds once will succeed repeatedly. One promising approach to counter this threat is making a model a moving target against adversarial inputs. To this end, we introduce Morphence-2.0, a scalable moving target defense (MTD) powered by out-of-distribution (OOD) detection to defend against adversarial examples. By regularly moving the decision function of a model, Morphence-2.0 makes it significantly challenging for repeated or correlated attacks to succeed. Morphence-2.0 deploys a pool of models generated from a base model in a manner that introduces sufficient randomness when it responds to prediction queries. Via OOD detection, Morphence-2.0 is equipped with a scheduling approach that assigns adversarial examples to robust decision functions and benign samples to an undefended accurate models. To ensure repeated or correlated attacks fail, the deployed pool of models automatically expires after a query budget is reached and the model pool is seamlessly replaced by a new model pool generated in advance. We evaluate Morphence-2.0 on two benchmark image classification datasets (MNIST and CIFAR10) against 4 reference attacks (3 white-box and 1 black-box). Morphence-2.0 consistently outperforms prior defenses while preserving accuracy on clean data and reducing attack transferability. We also show that, when powered by OOD detection, Morphence-2.0 is able to precisely make an input-based movement of the model's decision function that leads to higher prediction accuracy on both adversarial and benign queries.



## **43. Autoregressive Perturbations for Data Poisoning**

cs.LG

22 pages, 13 figures. Code available at  https://github.com/psandovalsegura/autoregressive-poisoning

**SubmitDate**: 2022-06-15    [paper-pdf](http://arxiv.org/pdf/2206.03693v2)

**Authors**: Pedro Sandoval-Segura, Vasu Singla, Jonas Geiping, Micah Goldblum, Tom Goldstein, David W. Jacobs

**Abstracts**: The prevalence of data scraping from social media as a means to obtain datasets has led to growing concerns regarding unauthorized use of data. Data poisoning attacks have been proposed as a bulwark against scraping, as they make data "unlearnable" by adding small, imperceptible perturbations. Unfortunately, existing methods require knowledge of both the target architecture and the complete dataset so that a surrogate network can be trained, the parameters of which are used to generate the attack. In this work, we introduce autoregressive (AR) poisoning, a method that can generate poisoned data without access to the broader dataset. The proposed AR perturbations are generic, can be applied across different datasets, and can poison different architectures. Compared to existing unlearnable methods, our AR poisons are more resistant against common defenses such as adversarial training and strong data augmentations. Our analysis further provides insight into what makes an effective data poison.



## **44. Fast and Reliable Evaluation of Adversarial Robustness with Minimum-Margin Attack**

cs.LG

**SubmitDate**: 2022-06-15    [paper-pdf](http://arxiv.org/pdf/2206.07314v1)

**Authors**: Ruize Gao, Jiongxiao Wang, Kaiwen Zhou, Feng Liu, Binghui Xie, Gang Niu, Bo Han, James Cheng

**Abstracts**: The AutoAttack (AA) has been the most reliable method to evaluate adversarial robustness when considerable computational resources are available. However, the high computational cost (e.g., 100 times more than that of the project gradient descent attack) makes AA infeasible for practitioners with limited computational resources, and also hinders applications of AA in the adversarial training (AT). In this paper, we propose a novel method, minimum-margin (MM) attack, to fast and reliably evaluate adversarial robustness. Compared with AA, our method achieves comparable performance but only costs 3% of the computational time in extensive experiments. The reliability of our method lies in that we evaluate the quality of adversarial examples using the margin between two targets that can precisely identify the most adversarial example. The computational efficiency of our method lies in an effective Sequential TArget Ranking Selection (STARS) method, ensuring that the cost of the MM attack is independent of the number of classes. The MM attack opens a new way for evaluating adversarial robustness and provides a feasible and reliable way to generate high-quality adversarial examples in AT.



## **45. Human Eyes Inspired Recurrent Neural Networks are More Robust Against Adversarial Noises**

cs.CV

**SubmitDate**: 2022-06-15    [paper-pdf](http://arxiv.org/pdf/2206.07282v1)

**Authors**: Minkyu Choi, Yizhen Zhang, Kuan Han, Xiaokai Wang, Zhongming Liu

**Abstracts**: Compared to human vision, computer vision based on convolutional neural networks (CNN) are more vulnerable to adversarial noises. This difference is likely attributable to how the eyes sample visual input and how the brain processes retinal samples through its dorsal and ventral visual pathways, which are under-explored for computer vision. Inspired by the brain, we design recurrent neural networks, including an input sampler that mimics the human retina, a dorsal network that guides where to look next, and a ventral network that represents the retinal samples. Taking these modules together, the models learn to take multiple glances at an image, attend to a salient part at each glance, and accumulate the representation over time to recognize the image. We test such models for their robustness against a varying level of adversarial noises with a special focus on the effect of different input sampling strategies. Our findings suggest that retinal foveation and sampling renders a model more robust against adversarial noises, and the model may correct itself from an attack when it is given a longer time to take more glances at an image. In conclusion, robust visual recognition can benefit from the combined use of three brain-inspired mechanisms: retinal transformation, attention guided eye movement, and recurrent processing, as opposed to feedforward-only CNNs.



## **46. Can Linear Programs Have Adversarial Examples? A Causal Perspective**

cs.LG

Main paper: 9 pages, References: 2 page, Supplement: 2 pages. Main  paper: 2 figures, 3 tables, Supplement: 1 figure, 1 table

**SubmitDate**: 2022-06-14    [paper-pdf](http://arxiv.org/pdf/2105.12697v5)

**Authors**: Matej Zečević, Devendra Singh Dhami, Kristian Kersting

**Abstracts**: The recent years have been marked by extended research on adversarial attacks, especially on deep neural networks. With this work we intend on posing and investigating the question of whether the phenomenon might be more general in nature, that is, adversarial-style attacks outside classification. Specifically, we investigate optimization problems starting with Linear Programs (LPs). We start off by demonstrating the shortcoming of a naive mapping between the formalism of adversarial examples and LPs, to then reveal how we can provide the missing piece -- intriguingly, through the Pearlian notion of Causality. Characteristically, we show the direct influence of the Structural Causal Model (SCM) onto the subsequent LP optimization, which ultimately exposes a notion of confounding in LPs (inherited by said SCM) that allows for adversarial-style attacks. We provide both the general proof formally alongside existential proofs of such intriguing LP-parameterizations based on SCM for three combinatorial problems, namely Linear Assignment, Shortest Path and a real world problem of energy systems.



## **47. Defending Observation Attacks in Deep Reinforcement Learning via Detection and Denoising**

cs.LG

**SubmitDate**: 2022-06-14    [paper-pdf](http://arxiv.org/pdf/2206.07188v1)

**Authors**: Zikang Xiong, Joe Eappen, He Zhu, Suresh Jagannathan

**Abstracts**: Neural network policies trained using Deep Reinforcement Learning (DRL) are well-known to be susceptible to adversarial attacks. In this paper, we consider attacks manifesting as perturbations in the observation space managed by the external environment. These attacks have been shown to downgrade policy performance significantly. We focus our attention on well-trained deterministic and stochastic neural network policies in the context of continuous control benchmarks subject to four well-studied observation space adversarial attacks. To defend against these attacks, we propose a novel defense strategy using a detect-and-denoise schema. Unlike previous adversarial training approaches that sample data in adversarial scenarios, our solution does not require sampling data in an environment under attack, thereby greatly reducing risk during training. Detailed experimental results show that our technique is comparable with state-of-the-art adversarial training approaches.



## **48. Proximal Splitting Adversarial Attacks for Semantic Segmentation**

cs.LG

Code available at:  https://github.com/jeromerony/alma_prox_segmentation

**SubmitDate**: 2022-06-14    [paper-pdf](http://arxiv.org/pdf/2206.07179v1)

**Authors**: Jérôme Rony, Jean-Christophe Pesquet, Ismail Ben Ayed

**Abstracts**: Classification has been the focal point of research on adversarial attacks, but only a few works investigate methods suited to denser prediction tasks, such as semantic segmentation. The methods proposed in these works do not accurately solve the adversarial segmentation problem and, therefore, are overoptimistic in terms of size of the perturbations required to fool models. Here, we propose a white-box attack for these models based on a proximal splitting to produce adversarial perturbations with much smaller $\ell_1$, $\ell_2$, or $\ell_\infty$ norms. Our attack can handle large numbers of constraints within a nonconvex minimization framework via an Augmented Lagrangian approach, coupled with adaptive constraint scaling and masking strategies. We demonstrate that our attack significantly outperforms previously proposed ones, as well as classification attacks that we adapted for segmentation, providing a first comprehensive benchmark for this dense task. Our results push current limits concerning robustness evaluations in segmentation tasks.



## **49. FENCE: Feasible Evasion Attacks on Neural Networks in Constrained Environments**

cs.CR

35 pages

**SubmitDate**: 2022-06-14    [paper-pdf](http://arxiv.org/pdf/1909.10480v4)

**Authors**: Alesia Chernikova, Alina Oprea

**Abstracts**: As advances in Deep Neural Networks (DNNs) demonstrate unprecedented levels of performance in many critical applications, their vulnerability to attacks is still an open question. We consider evasion attacks at testing time against Deep Learning in constrained environments, in which dependencies between features need to be satisfied. These situations may arise naturally in tabular data or may be the result of feature engineering in specific application domains, such as threat detection in cyber security. We propose a general iterative gradient-based framework called FENCE for crafting evasion attacks that take into consideration the specifics of constrained domains and application requirements. We apply it against Feed-Forward Neural Networks trained for two cyber security applications: network traffic botnet classification and malicious domain classification, to generate feasible adversarial examples. We extensively evaluate the success rate and performance of our attacks, compare their improvement over several baselines, and analyze factors that impact the attack success rate, including the optimization objective and the data imbalance. We show that with minimal effort (e.g., generating 12 additional network connections), an attacker can change the model's prediction from the Malicious class to Benign and evade the classifier. We show that models trained on datasets with higher imbalance are more vulnerable to our FENCE attacks. Finally, we demonstrate the potential of performing adversarial training in constrained domains to increase the model resilience against these evasion attacks.



## **50. A Layered Reference Model for Penetration Testing with Reinforcement Learning and Attack Graphs**

cs.CR

**SubmitDate**: 2022-06-14    [paper-pdf](http://arxiv.org/pdf/2206.06934v1)

**Authors**: Tyler Cody

**Abstracts**: This paper considers key challenges to using reinforcement learning (RL) with attack graphs to automate penetration testing in real-world applications from a systems perspective. RL approaches to automated penetration testing are actively being developed, but there is no consensus view on the representation of computer networks with which RL should be interacting. Moreover, there are significant open challenges to how those representations can be grounded to the real networks where RL solution methods are applied. This paper elaborates on representation and grounding using topic challenges of interacting with real networks in real-time, emulating realistic adversary behavior, and handling unstable, evolving networks. These challenges are both practical and mathematical, and they directly concern the reliability and dependability of penetration testing systems. This paper proposes a layered reference model to help organize related research and engineering efforts. The presented layered reference model contrasts traditional models of attack graph workflows because it is not scoped to a sequential, feed-forward generation and analysis process, but to broader aspects of lifecycle and continuous deployment. Researchers and practitioners can use the presented layered reference model as a first-principles outline to help orient the systems engineering of their penetration testing systems.



