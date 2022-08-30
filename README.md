# Latest Adversarial Attack Papers
**update at 2022-08-31 06:31:29**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Demystifying Arch-hints for Model Extraction: An Attack in Unified Memory System**

cs.CR

**SubmitDate**: 2022-08-29    [paper-pdf](http://arxiv.org/pdf/2208.13720v1)

**Authors**: Zhendong Wang, Xiaoming Zeng, Xulong Tang, Danfeng Zhang, Xing Hu, Yang Hu

**Abstracts**: The deep neural network (DNN) models are deemed confidential due to their unique value in expensive training efforts, privacy-sensitive training data, and proprietary network characteristics. Consequently, the model value raises incentive for adversary to steal the model for profits, such as the representative model extraction attack. Emerging attack can leverage timing-sensitive architecture-level events (i.e., Arch-hints) disclosed in hardware platforms to extract DNN model layer information accurately. In this paper, we take the first step to uncover the root cause of such Arch-hints and summarize the principles to identify them. We then apply these principles to emerging Unified Memory (UM) management system and identify three new Arch-hints caused by UM's unique data movement patterns. We then develop a new extraction attack, UMProbe. We also create the first DNN benchmark suite in UM and utilize the benchmark suite to evaluate UMProbe. Our evaluation shows that UMProbe can extract the layer sequence with an accuracy of 95% for almost all victim test models, which thus calls for more attention to the DNN security in UM system.



## **2. Understanding the Limits of Poisoning Attacks in Episodic Reinforcement Learning**

cs.LG

Accepted at International Joint Conferences on Artificial  Intelligence (IJCAI) 2022

**SubmitDate**: 2022-08-29    [paper-pdf](http://arxiv.org/pdf/2208.13663v1)

**Authors**: Anshuka Rangi, Haifeng Xu, Long Tran-Thanh, Massimo Franceschetti

**Abstracts**: To understand the security threats to reinforcement learning (RL) algorithms, this paper studies poisoning attacks to manipulate \emph{any} order-optimal learning algorithm towards a targeted policy in episodic RL and examines the potential damage of two natural types of poisoning attacks, i.e., the manipulation of \emph{reward} and \emph{action}. We discover that the effect of attacks crucially depend on whether the rewards are bounded or unbounded. In bounded reward settings, we show that only reward manipulation or only action manipulation cannot guarantee a successful attack. However, by combining reward and action manipulation, the adversary can manipulate any order-optimal learning algorithm to follow any targeted policy with $\tilde{\Theta}(\sqrt{T})$ total attack cost, which is order-optimal, without any knowledge of the underlying MDP. In contrast, in unbounded reward settings, we show that reward manipulation attacks are sufficient for an adversary to successfully manipulate any order-optimal learning algorithm to follow any targeted policy using $\tilde{O}(\sqrt{T})$ amount of contamination. Our results reveal useful insights about what can or cannot be achieved by poisoning attacks, and are set to spur more works on the design of robust RL algorithms.



## **3. HAT4RD: Hierarchical Adversarial Training for Rumor Detection on Social Media**

cs.CL

**SubmitDate**: 2022-08-29    [paper-pdf](http://arxiv.org/pdf/2110.00425v2)

**Authors**: Shiwen Ni, Jiawen Li, Hung-Yu Kao

**Abstracts**: With the development of social media, social communication has changed. While this facilitates people's communication and access to information, it also provides an ideal platform for spreading rumors. In normal or critical situations, rumors will affect people's judgment and even endanger social security. However, natural language is high-dimensional and sparse, and the same rumor may be expressed in hundreds of ways on social media. As such, the robustness and generalization of the current rumor detection model are put into question. We proposed a novel \textbf{h}ierarchical \textbf{a}dversarial \textbf{t}raining method for \textbf{r}umor \textbf{d}etection (HAT4RD) on social media. Specifically, HAT4RD is based on gradient ascent by adding adversarial perturbations to the embedding layers of post-level and event-level modules to deceive the detector. At the same time, the detector uses stochastic gradient descent to minimize the adversarial risk to learn a more robust model. In this way, the post-level and event-level sample spaces are enhanced, and we have verified the robustness of our model under a variety of adversarial attacks. Moreover, visual experiments indicate that the proposed model drifts into an area with a flat loss landscape, leading to better generalization. We evaluate our proposed method on three public rumors datasets from two commonly used social platforms (Twitter and Weibo). Experiment results demonstrate that our model achieves better results than state-of-the-art methods.



## **4. Towards Both Accurate and Robust Neural Networks without Extra Data**

cs.CV

**SubmitDate**: 2022-08-29    [paper-pdf](http://arxiv.org/pdf/2103.13124v2)

**Authors**: Faqiang Liu, Rong Zhao

**Abstracts**: Deep neural networks have achieved remarkable performance in various applications but are extremely vulnerable to adversarial perturbation. The most representative and promising methods that can enhance model robustness, such as adversarial training and its variants, substantially degrade model accuracy on benign samples, limiting practical utility. Although incorporating extra training data can alleviate the trade-off to a certain extent, it remains unsolved to achieve both robustness and accuracy under limited training data. Here, we demonstrate the feasibility of overcoming the trade-off, by developing an adversarial feature stacking (AFS) model, which combines multiple independent feature extractors with varied levels of robustness and accuracy. Theoretical analysis is further conducted, and general principles for the selection of basic feature extractors are provided. We evaluate the AFS model on CIFAR-10 and CIFAR-100 datasets with strong adaptive attack methods, significantly advancing the state-of-the-art in terms of the trade-off. The AFS model achieves a benign accuracy improvement of ~6% on CIFAR-10 and ~10% on CIFAR-100 with comparable or even stronger robustness than the state-of-the-art adversarial training methods.



## **5. Tricking the Hashing Trick: A Tight Lower Bound on the Robustness of CountSketch to Adaptive Inputs**

cs.DS

**SubmitDate**: 2022-08-28    [paper-pdf](http://arxiv.org/pdf/2207.00956v2)

**Authors**: Edith Cohen, Jelani Nelson, Tamás Sarlós, Uri Stemmer

**Abstracts**: CountSketch and Feature Hashing (the "hashing trick") are popular randomized dimensionality reduction methods that support recovery of $\ell_2$-heavy hitters (keys $i$ where $v_i^2 > \epsilon \|\boldsymbol{v}\|_2^2$) and approximate inner products. When the inputs are {\em not adaptive} (do not depend on prior outputs), classic estimators applied to a sketch of size $O(\ell/\epsilon)$ are accurate for a number of queries that is exponential in $\ell$. When inputs are adaptive, however, an adversarial input can be constructed after $O(\ell)$ queries with the classic estimator and the best known robust estimator only supports $\tilde{O}(\ell^2)$ queries. In this work we show that this quadratic dependence is in a sense inherent: We design an attack that after $O(\ell^2)$ queries produces an adversarial input vector whose sketch is highly biased. Our attack uses "natural" non-adaptive inputs (only the final adversarial input is chosen adaptively) and universally applies with any correct estimator, including one that is unknown to the attacker. In that, we expose inherent vulnerability of this fundamental method.



## **6. Categorical composable cryptography: extended version**

cs.CR

Extended version of arXiv:2105.05949 which appeared in FoSSaCS 2022

**SubmitDate**: 2022-08-28    [paper-pdf](http://arxiv.org/pdf/2208.13232v1)

**Authors**: Anne Broadbent, Martti Karvonen

**Abstracts**: We formalize the simulation paradigm of cryptography in terms of category theory and show that protocols secure against abstract attacks form a symmetric monoidal category, thus giving an abstract model of composable security definitions in cryptography. Our model is able to incorporate computational security, set-up assumptions and various attack models such as colluding or independently acting subsets of adversaries in a modular, flexible fashion. W We conclude by using string diagrams to rederive the security of the one-time pad and no-go results concerning the limits of bipartite and tripartite cryptography, ruling out e.g., composable commitments and broadcasting. On the way, we exhibit two categorical constructions of resource theories that might be of independent interest: one capturing resources shared among multiple parties and one capturing resource conversions that succeed asymptotically.



## **7. Categorical composable cryptography**

cs.CR

Updated to match the proceedings version

**SubmitDate**: 2022-08-28    [paper-pdf](http://arxiv.org/pdf/2105.05949v3)

**Authors**: Anne Broadbent, Martti Karvonen

**Abstracts**: We formalize the simulation paradigm of cryptography in terms of category theory and show that protocols secure against abstract attacks form a symmetric monoidal category, thus giving an abstract model of composable security definitions in cryptography. Our model is able to incorporate computational security, set-up assumptions and various attack models such as colluding or independently acting subsets of adversaries in a modular, flexible fashion. We conclude by using string diagrams to rederive the security of the one-time pad and no-go results concerning the limits of bipartite and tripartite cryptography, ruling out e.g., composable commitments and broadcasting.



## **8. Self-Supervised Adversarial Example Detection by Disentangled Representation**

cs.CV

to appear in TrustCom 2022

**SubmitDate**: 2022-08-28    [paper-pdf](http://arxiv.org/pdf/2105.03689v4)

**Authors**: Zhaoxi Zhang, Leo Yu Zhang, Xufei Zheng, Jinyu Tian, Jiantao Zhou

**Abstracts**: Deep learning models are known to be vulnerable to adversarial examples that are elaborately designed for malicious purposes and are imperceptible to the human perceptual system. Autoencoder, when trained solely over benign examples, has been widely used for (self-supervised) adversarial detection based on the assumption that adversarial examples yield larger reconstruction errors. However, because lacking adversarial examples in its training and the too strong generalization ability of autoencoder, this assumption does not always hold true in practice. To alleviate this problem, we explore how to detect adversarial examples with disentangled label/semantic features under the autoencoder structure. Specifically, we propose Disentangled Representation-based Reconstruction (DRR). In DRR, we train an autoencoder over both correctly paired label/semantic features and incorrectly paired label/semantic features to reconstruct benign and counterexamples. This mimics the behavior of adversarial examples and can reduce the unnecessary generalization ability of autoencoder. We compare our method with the state-of-the-art self-supervised detection methods under different adversarial attacks and different victim models, and it exhibits better performance in various metrics (area under the ROC curve, true positive rate, and true negative rate) for most attack settings. Though DRR is initially designed for visual tasks only, we demonstrate that it can be easily extended for natural language tasks as well. Notably, different from other autoencoder-based detectors, our method can provide resistance to the adaptive adversary.



## **9. Cross-domain Cross-architecture Black-box Attacks on Fine-tuned Models with Transferred Evolutionary Strategies**

cs.LG

To appear in CIKM 2022

**SubmitDate**: 2022-08-28    [paper-pdf](http://arxiv.org/pdf/2208.13182v1)

**Authors**: Yinghua Zhang, Yangqiu Song, Kun Bai, Qiang Yang

**Abstracts**: Fine-tuning can be vulnerable to adversarial attacks. Existing works about black-box attacks on fine-tuned models (BAFT) are limited by strong assumptions. To fill the gap, we propose two novel BAFT settings, cross-domain and cross-domain cross-architecture BAFT, which only assume that (1) the target model for attacking is a fine-tuned model, and (2) the source domain data is known and accessible. To successfully attack fine-tuned models under both settings, we propose to first train an adversarial generator against the source model, which adopts an encoder-decoder architecture and maps a clean input to an adversarial example. Then we search in the low-dimensional latent space produced by the encoder of the adversarial generator. The search is conducted under the guidance of the surrogate gradient obtained from the source model. Experimental results on different domains and different network architectures demonstrate that the proposed attack method can effectively and efficiently attack the fine-tuned models.



## **10. Improved and Interpretable Defense to Transferred Adversarial Examples by Jacobian Norm with Selective Input Gradient Regularization**

cs.LG

Under review

**SubmitDate**: 2022-08-28    [paper-pdf](http://arxiv.org/pdf/2207.13036v3)

**Authors**: Deyin Liu, Lin Wu, Lingqiao Liu, Haifeng Zhao, Farid Boussaid, Mohammed Bennamoun

**Abstracts**: Deep neural networks (DNNs) are known to be vulnerable to adversarial examples that are crafted with imperceptible perturbations, i.e., a small change in an input image can induce a mis-classification, and thus threatens the reliability of deep learning based deployment systems. Adversarial training (AT) is often adopted to improve robustness through training a mixture of corrupted and clean data. However, most of AT based methods are ineffective in dealing with transferred adversarial examples which are generated to fool a wide spectrum of defense models, and thus cannot satisfy the generalization requirement raised in real-world scenarios. Moreover, adversarially training a defense model in general cannot produce interpretable predictions towards the inputs with perturbations, whilst a highly interpretable robust model is required by different domain experts to understand the behaviour of a DNN. In this work, we propose a novel approach based on Jacobian norm and Selective Input Gradient Regularization (J-SIGR), which suggests the linearized robustness through Jacobian normalization and also regularizes the perturbation-based saliency maps to imitate the model's interpretable predictions. As such, we achieve both the improved defense and high interpretability of DNNs. Finally, we evaluate our method across different architectures against powerful adversarial attacks. Experiments demonstrate that the proposed J-SIGR confers improved robustness against transferred adversarial attacks, and we also show that the predictions from the neural network are easy to interpret.



## **11. Covariate Balancing Methods for Randomized Controlled Trials Are Not Adversarially Robust**

econ.EM

12 pages, double column, 4 figures

**SubmitDate**: 2022-08-28    [paper-pdf](http://arxiv.org/pdf/2110.13262v3)

**Authors**: Hossein Babaei, Sina Alemohammad, Richard Baraniuk

**Abstracts**: The first step towards investigating the effectiveness of a treatment via a randomized trial is to split the population into control and treatment groups then compare the average response of the treatment group receiving the treatment to the control group receiving the placebo.   In order to ensure that the difference between the two groups is caused only by the treatment, it is crucial that the control and the treatment groups have similar statistics. Indeed, the validity and reliability of a trial are determined by the similarity of two groups' statistics. Covariate balancing methods increase the similarity between the distributions of the two groups' covariates. However, often in practice, there are not enough samples to accurately estimate the groups' covariate distributions. In this paper, we empirically show that covariate balancing with the Standardized Means Difference (SMD) covariate balancing measure, as well as Pocock's sequential treatment assignment method, are susceptible to worst-case treatment assignments. Worst-case treatment assignments are those admitted by the covariate balance measure, but result in highest possible ATE estimation errors. We developed an adversarial attack to find adversarial treatment assignment for any given trial. Then, we provide an index to measure how close the given trial is to the worst-case. To this end, we provide an optimization-based algorithm, namely Adversarial Treatment ASsignment in TREatment Effect Trials (ATASTREET), to find the adversarial treatment assignments.



## **12. Overcoming Data Availability Attacks in Blockchain Systems: Short Code-Length LDPC Code Design for Coded Merkle Tree**

cs.IT

18 pages, 7 figures, 3 tables, accepted at IEEE Transactions on  Communications (TCOM) 2022. This version reflects comments from reviewers at  TCOM

**SubmitDate**: 2022-08-27    [paper-pdf](http://arxiv.org/pdf/2108.13332v3)

**Authors**: Debarnab Mitra, Lev Tauz, Lara Dolecek

**Abstracts**: Light nodes are clients in blockchain systems that only store a small portion of the blockchain ledger. In certain blockchains, light nodes are vulnerable to a data availability (DA) attack where a malicious node makes the light nodes accept an invalid block by hiding the invalid portion of the block from the nodes in the system. Recently, a technique based on LDPC codes called Coded Merkle Tree was proposed by Yu et al. that enables light nodes to detect a DA attack by randomly requesting/sampling portions of the block from the malicious node. However, light nodes fail to detect a DA attack with high probability if a malicious node hides a small stopping set of the LDPC code. In this paper, we demonstrate that a suitable co-design of specialized LDPC codes and the light node sampling strategy leads to a high probability of detection of DA attacks. We consider different adversary models based on their computational capabilities of finding stopping sets. For the different adversary models, we provide new specialized LDPC code constructions and coupled light node sampling strategies and demonstrate that they lead to a higher probability of detection of DA attacks compared to approaches proposed in earlier literature.



## **13. Cooperative Distributed State Estimation: Resilient Topologies against Smart Spoofers**

cs.CR

**SubmitDate**: 2022-08-27    [paper-pdf](http://arxiv.org/pdf/1909.04172v4)

**Authors**: Mostafa Safi

**Abstracts**: A network of observers is considered, where through asynchronous (with bounded delay) communications, they cooperatively estimate the states of a Linear Time-Invariant (LTI) system. In such a setting, a new type of adversary might affect the observation process by impersonating the identity of the regular node, which is a violation of communication authenticity. These adversaries also inherit the capabilities of Byzantine nodes, making them more powerful threats called smart spoofers. We show how asynchronous networks are vulnerable to smart spoofing attack. In the estimation scheme considered in this paper, information flows from the sets of source nodes, which can detect a portion of the state variables each, to the other follower nodes. The regular nodes, to avoid being misguided by the threats, distributively filter the extreme values received from the nodes in their neighborhood. Topological conditions based on strong robustness are proposed to guarantee the convergence. Two simulation scenarios are provided to verify the results.



## **14. SA: Sliding attack for synthetic speech detection with resistance to clipping and self-splicing**

cs.SD

12 pages, Neurocomputing

**SubmitDate**: 2022-08-27    [paper-pdf](http://arxiv.org/pdf/2208.13066v1)

**Authors**: Deng JiaCheng, Dong Li, Yan Diqun, Wang Rangding, Zeng Jiaming

**Abstracts**: Deep neural networks are vulnerable to adversarial examples that mislead models with imperceptible perturbations. In audio, although adversarial examples have achieved incredible attack success rates on white-box settings and black-box settings, most existing adversarial attacks are constrained by the input length. A More practical scenario is that the adversarial examples must be clipped or self-spliced and input into the black-box model. Therefore, it is necessary to explore how to improve transferability in different input length settings. In this paper, we take the synthetic speech detection task as an example and consider two representative SOTA models. We observe that the gradients of fragments with the same sample value are similar in different models via analyzing the gradients obtained by feeding samples into the model after cropping or self-splicing. Inspired by the above observation, we propose a new adversarial attack method termed sliding attack. Specifically, we make each sampling point aware of gradients at different locations, which can simulate the situation where adversarial examples are input to black-box models with varying input lengths. Therefore, instead of using the current gradient directly in each iteration of the gradient calculation, we go through the following three steps. First, we extract subsegments of different lengths using sliding windows. We then augment the subsegments with data from the adjacent domains. Finally, we feed the sub-segments into different models to obtain aggregate gradients to update adversarial examples. Empirical results demonstrate that our method could significantly improve the transferability of adversarial examples after clipping or self-splicing. Besides, our method could also enhance the transferability between models based on different features.



## **15. Adversarial Robustness for Tabular Data through Cost and Utility Awareness**

cs.LG

* authors contributed equally

**SubmitDate**: 2022-08-27    [paper-pdf](http://arxiv.org/pdf/2208.13058v1)

**Authors**: Klim Kireev, Bogdan Kulynych, Carmela Troncoso

**Abstracts**: Many machine learning problems use data in the tabular domains. Adversarial examples can be especially damaging for these applications. Yet, existing works on adversarial robustness mainly focus on machine-learning models in the image and text domains. We argue that due to the differences between tabular data and images or text, existing threat models are inappropriate for tabular domains. These models do not capture that cost can be more important than imperceptibility, nor that the adversary could ascribe different value to the utility obtained from deploying different adversarial examples. We show that due to these differences the attack and defence methods used for images and text cannot be directly applied to the tabular setup. We address these issues by proposing new cost and utility-aware threat models tailored to the adversarial capabilities and constraints of attackers targeting tabular domains. We introduce a framework that enables us to design attack and defence mechanisms which result in models protected against cost or utility-aware adversaries, e.g., adversaries constrained by a certain dollar budget. We show that our approach is effective on three tabular datasets corresponding to applications for which adversarial examples can have economic and social implications.



## **16. TrojViT: Trojan Insertion in Vision Transformers**

cs.LG

9 pages, 4 figures, 9 tables

**SubmitDate**: 2022-08-27    [paper-pdf](http://arxiv.org/pdf/2208.13049v1)

**Authors**: Mengxin Zheng, Qian Lou, Lei Jiang

**Abstracts**: Vision Transformers (ViTs) have demonstrated the state-of-the-art performance in various vision-related tasks. The success of ViTs motivates adversaries to perform backdoor attacks on ViTs. Although the vulnerability of traditional CNNs to backdoor attacks is well-known, backdoor attacks on ViTs are seldom-studied. Compared to CNNs capturing pixel-wise local features by convolutions, ViTs extract global context information through patches and attentions. Na\"ively transplanting CNN-specific backdoor attacks to ViTs yields only a low clean data accuracy and a low attack success rate. In this paper, we propose a stealth and practical ViT-specific backdoor attack $TrojViT$. Rather than an area-wise trigger used by CNN-specific backdoor attacks, TrojViT generates a patch-wise trigger designed to build a Trojan composed of some vulnerable bits on the parameters of a ViT stored in DRAM memory through patch salience ranking and attention-target loss. TrojViT further uses minimum-tuned parameter update to reduce the bit number of the Trojan. Once the attacker inserts the Trojan into the ViT model by flipping the vulnerable bits, the ViT model still produces normal inference accuracy with benign inputs. But when the attacker embeds a trigger into an input, the ViT model is forced to classify the input to a predefined target class. We show that flipping only few vulnerable bits identified by TrojViT on a ViT model using the well-known RowHammer can transform the model into a backdoored one. We perform extensive experiments of multiple datasets on various ViT models. TrojViT can classify $99.64\%$ of test images to a target class by flipping $345$ bits on a ViT for ImageNet.



## **17. SoK: Decentralized Finance (DeFi) Incidents**

cs.CR

**SubmitDate**: 2022-08-27    [paper-pdf](http://arxiv.org/pdf/2208.13035v1)

**Authors**: Liyi Zhou, Xihan Xiong, Jens Ernstberger, Stefanos Chaliasos, Zhipeng Wang, Ye Wang, Kaihua Qin, Roger Wattenhofer, Dawn Song, Arthur Gervais

**Abstracts**: Within just four years, the blockchain-based Decentralized Finance (DeFi) ecosystem has accumulated a peak total value locked (TVL) of more than 253 billion USD. This surge in DeFi's popularity has, unfortunately, been accompanied by many impactful incidents. According to our data, users, liquidity providers, speculators, and protocol operators suffered a total loss of at least 3.24 USD from Apr 30, 2018 to Apr 30, 2022. Given the blockchain's transparency and increasing incident frequency, two questions arise: How can we systematically measure, evaluate, and compare DeFi incidents? How can we learn from past attacks to strengthen DeFi security?   In this paper, we introduce a common reference frame to systematically evaluate and compare DeFi incidents. We investigate 77 academic papers, 30 audit reports, and 181 real-world incidents. Our open data reveals several gaps between academia and the practitioners' community. For example, few academic papers address "price oracle attacks" and "permissonless interactions", while our data suggests that they are the two most frequent incident types (15% and 10.5% correspondingly). We also investigate potential defenses, and find that: (i) 103 (56%) of the attacks are not executed atomically, granting a rescue time frame for defenders; (ii) SoTA bytecode similarity analysis can at least detect 31 vulnerable/23 adversarial contracts; and (iii) 33 (15.3%) of the adversaries leak potentially identifiable information by interacting with centralized exchanges.



## **18. Overparameterized (robust) models from computational constraints**

cs.LG

**SubmitDate**: 2022-08-27    [paper-pdf](http://arxiv.org/pdf/2208.12926v1)

**Authors**: Sanjam Garg, Somesh Jha, Saeed Mahloujifar, Mohammad Mahmoody, Mingyuan Wang

**Abstracts**: Overparameterized models with millions of parameters have been hugely successful. In this work, we ask: can the need for large models be, at least in part, due to the \emph{computational} limitations of the learner? Additionally, we ask, is this situation exacerbated for \emph{robust} learning? We show that this indeed could be the case. We show learning tasks for which computationally bounded learners need \emph{significantly more} model parameters than what information-theoretic learners need. Furthermore, we show that even more model parameters could be necessary for robust learning. In particular, for computationally bounded learners, we extend the recent result of Bubeck and Sellke [NeurIPS'2021] which shows that robust models might need more parameters, to the computational regime and show that bounded learners could provably need an even larger number of parameters. Then, we address the following related question: can we hope to remedy the situation for robust computationally bounded learning by restricting \emph{adversaries} to also be computationally bounded for sake of obtaining models with fewer parameters? Here again, we show that this could be possible. Specifically, building on the work of Garg, Jha, Mahloujifar, and Mahmoody [ALT'2020], we demonstrate a learning task that can be learned efficiently and robustly against a computationally bounded attacker, while to be robust against an information-theoretic attacker requires the learner to utilize significantly more parameters.



## **19. Bitcoin's Latency--Security Analysis Made Simple**

cs.CR

**SubmitDate**: 2022-08-27    [paper-pdf](http://arxiv.org/pdf/2203.06357v3)

**Authors**: Dongning Guo, Ling Ren

**Abstracts**: Simple closed-form upper and lower bounds are developed for the security of the Nakamoto consensus as a function of the confirmation depth, the honest and adversarial block mining rates, and an upper bound on the block propagation delay. The bounds are exponential in the confirmation depth and apply regardless of the adversary's attack strategy. The gap between the upper and lower bounds is small for Bitcoin's parameters. For example, assuming an average block interval of 10 minutes, a network delay bound of ten seconds, and 10% adversarial mining power, the widely used 6-block confirmation rule yields a safety violation between 0.11% and 0.35% probability.



## **20. Network-Level Adversaries in Federated Learning**

cs.CR

12 pages. Appearing at IEEE CNS 2022

**SubmitDate**: 2022-08-27    [paper-pdf](http://arxiv.org/pdf/2208.12911v1)

**Authors**: Giorgio Severi, Matthew Jagielski, Gökberk Yar, Yuxuan Wang, Alina Oprea, Cristina Nita-Rotaru

**Abstracts**: Federated learning is a popular strategy for training models on distributed, sensitive data, while preserving data privacy. Prior work identified a range of security threats on federated learning protocols that poison the data or the model. However, federated learning is a networked system where the communication between clients and server plays a critical role for the learning task performance. We highlight how communication introduces another vulnerability surface in federated learning and study the impact of network-level adversaries on training federated learning models. We show that attackers dropping the network traffic from carefully selected clients can significantly decrease model accuracy on a target population. Moreover, we show that a coordinated poisoning campaign from a few clients can amplify the dropping attacks. Finally, we develop a server-side defense which mitigates the impact of our attacks by identifying and up-sampling clients likely to positively contribute towards target accuracy. We comprehensively evaluate our attacks and defenses on three datasets, assuming encrypted communication channels and attackers with partial visibility of the network.



## **21. Adversarial Relighting Against Face Recognition**

cs.CV

**SubmitDate**: 2022-08-27    [paper-pdf](http://arxiv.org/pdf/2108.07920v4)

**Authors**: Qian Zhang, Qing Guo, Ruijun Gao, Felix Juefei-Xu, Hongkai Yu, Wei Feng

**Abstracts**: Deep face recognition (FR) has achieved significantly high accuracy on several challenging datasets and fosters successful real-world applications, even showing high robustness to the illumination variation that is usually regarded as a main threat to the FR system. However, in the real world, illumination variation caused by diverse lighting conditions cannot be fully covered by the limited face dataset. In this paper, we study the threat of lighting against FR from a new angle, i.e., adversarial attack, and identify a new task, i.e., adversarial relighting. Given a face image, adversarial relighting aims to produce a naturally relighted counterpart while fooling the state-of-the-art deep FR methods. To this end, we first propose the physical modelbased adversarial relighting attack (ARA) denoted as albedoquotient-based adversarial relighting attack (AQ-ARA). It generates natural adversarial light under the physical lighting model and guidance of FR systems and synthesizes adversarially relighted face images. Moreover, we propose the auto-predictive adversarial relighting attack (AP-ARA) by training an adversarial relighting network (ARNet) to automatically predict the adversarial light in a one-step manner according to different input faces, allowing efficiency-sensitive applications. More importantly, we propose to transfer the above digital attacks to physical ARA (PhyARA) through a precise relighting device, making the estimated adversarial lighting condition reproducible in the real world. We validate our methods on three state-of-the-art deep FR methods, i.e., FaceNet, ArcFace, and CosFace, on two public datasets. The extensive and insightful results demonstrate our work can generate realistic adversarial relighted face images fooling face recognition tasks easily, revealing the threat of specific light directions and strengths.



## **22. ATTRITION: Attacking Static Hardware Trojan Detection Techniques Using Reinforcement Learning**

cs.CR

To Appear in 2022 ACM SIGSAC Conference on Computer and  Communications Security (CCS), November 2022

**SubmitDate**: 2022-08-26    [paper-pdf](http://arxiv.org/pdf/2208.12897v1)

**Authors**: Vasudev Gohil, Hao Guo, Satwik Patnaik, Jeyavijayan, Rajendran

**Abstracts**: Stealthy hardware Trojans (HTs) inserted during the fabrication of integrated circuits can bypass the security of critical infrastructures. Although researchers have proposed many techniques to detect HTs, several limitations exist, including: (i) a low success rate, (ii) high algorithmic complexity, and (iii) a large number of test patterns. Furthermore, the most pertinent drawback of prior detection techniques stems from an incorrect evaluation methodology, i.e., they assume that an adversary inserts HTs randomly. Such inappropriate adversarial assumptions enable detection techniques to claim high HT detection accuracy, leading to a "false sense of security." Unfortunately, to the best of our knowledge, despite more than a decade of research on detecting HTs inserted during fabrication, there have been no concerted efforts to perform a systematic evaluation of HT detection techniques.   In this paper, we play the role of a realistic adversary and question the efficacy of HT detection techniques by developing an automated, scalable, and practical attack framework, ATTRITION, using reinforcement learning (RL). ATTRITION evades eight detection techniques across two HT detection categories, showcasing its agnostic behavior. ATTRITION achieves average attack success rates of $47\times$ and $211\times$ compared to randomly inserted HTs against state-of-the-art HT detection techniques. We demonstrate ATTRITION's ability to evade detection techniques by evaluating designs ranging from the widely-used academic suites to larger designs such as the open-source MIPS and mor1kx processors to AES and a GPS module. Additionally, we showcase the impact of ATTRITION-generated HTs through two case studies (privilege escalation and kill switch) on the mor1kx processor. We envision that our work, along with our released HT benchmarks and models, fosters the development of better HT detection techniques.



## **23. SoftHebb: Bayesian Inference in Unsupervised Hebbian Soft Winner-Take-All Networks**

cs.LG

**SubmitDate**: 2022-08-26    [paper-pdf](http://arxiv.org/pdf/2107.05747v3)

**Authors**: Timoleon Moraitis, Dmitry Toichkin, Adrien Journé, Yansong Chua, Qinghai Guo

**Abstracts**: Hebbian plasticity in winner-take-all (WTA) networks is highly attractive for neuromorphic on-chip learning, owing to its efficient, local, unsupervised, and on-line nature. Moreover, its biological plausibility may help overcome important limitations of artificial algorithms, such as their susceptibility to adversarial attacks and long training time. However, Hebbian WTA learning has found little use in machine learning (ML), likely because it has been missing an optimization theory compatible with deep learning (DL). Here we show rigorously that WTA networks constructed by standard DL elements, combined with a Hebbian-like plasticity that we derive, maintain a Bayesian generative model of the data. Importantly, without any supervision, our algorithm, SoftHebb, minimizes cross-entropy, i.e. a common loss function in supervised DL. We show this theoretically and in practice. The key is a "soft" WTA where there is no absolute "hard" winner neuron. Strikingly, in shallow-network comparisons with backpropagation (BP), SoftHebb shows advantages beyond its Hebbian efficiency. Namely, it converges faster and is significantly more robust to noise and adversarial attacks. Notably, attacks that maximally confuse SoftHebb are also confusing to the human eye, potentially linking human perceptual robustness, with Hebbian WTA circuits of cortex. Finally, SoftHebb can generate synthetic objects as interpolations of real object classes. All in all, Hebbian efficiency, theoretical underpinning, cross-entropy-minimization, and surprising empirical advantages, suggest that SoftHebb may inspire highly neuromorphic and radically different, but practical and advantageous learning algorithms and hardware accelerators.



## **24. What Does the Gradient Tell When Attacking the Graph Structure**

cs.LG

**SubmitDate**: 2022-08-26    [paper-pdf](http://arxiv.org/pdf/2208.12815v1)

**Authors**: Zihan Liu, Ge Wang, Yun Luo, Stan Z. Li

**Abstracts**: Recent studies have proven that graph neural networks are vulnerable to adversarial attacks. Attackers can rely solely on the training labels to disrupt the performance of the agnostic victim model by edge perturbations. Researchers observe that the saliency-based attackers tend to add edges rather than delete them, which is previously explained by the fact that adding edges pollutes the nodes' features by aggregation while removing edges only leads to some loss of information. In this paper, we further prove that the attackers perturb graphs by adding inter-class edges, which also manifests as a reduction in the homophily of the perturbed graph. From this point of view, saliency-based attackers still have room for improvement in capability and imperceptibility. The message passing of the GNN-based surrogate model leads to the oversmoothing of nodes connected by inter-class edges, preventing attackers from obtaining the distinctiveness of node features. To solve this issue, we introduce a multi-hop aggregated message passing to preserve attribute differences between nodes. In addition, we propose a regularization term to restrict the homophily variance to enhance the attack imperceptibility. Experiments verify that our proposed surrogate model improves the attacker's versatility and the regularization term helps to limit the homophily of the perturbed graph.



## **25. Robust Prototypical Few-Shot Organ Segmentation with Regularized Neural-ODEs**

cs.CV

**SubmitDate**: 2022-08-26    [paper-pdf](http://arxiv.org/pdf/2208.12428v1)

**Authors**: Prashant Pandey, Mustafa Chasmai, Tanuj Sur, Brejesh Lall

**Abstracts**: Despite the tremendous progress made by deep learning models in image semantic segmentation, they typically require large annotated examples, and increasing attention is being diverted to problem settings like Few-Shot Learning (FSL) where only a small amount of annotation is needed for generalisation to novel classes. This is especially seen in medical domains where dense pixel-level annotations are expensive to obtain. In this paper, we propose Regularized Prototypical Neural Ordinary Differential Equation (R-PNODE), a method that leverages intrinsic properties of Neural-ODEs, assisted and enhanced by additional cluster and consistency losses to perform Few-Shot Segmentation (FSS) of organs. R-PNODE constrains support and query features from the same classes to lie closer in the representation space thereby improving the performance over the existing Convolutional Neural Network (CNN) based FSS methods. We further demonstrate that while many existing Deep CNN based methods tend to be extremely vulnerable to adversarial attacks, R-PNODE exhibits increased adversarial robustness for a wide array of these attacks. We experiment with three publicly available multi-organ segmentation datasets in both in-domain and cross-domain FSS settings to demonstrate the efficacy of our method. In addition, we perform experiments with seven commonly used adversarial attacks in various settings to demonstrate R-PNODE's robustness. R-PNODE outperforms the baselines for FSS by significant margins and also shows superior performance for a wide array of attacks varying in intensity and design.



## **26. SNAP: Efficient Extraction of Private Properties with Poisoning**

cs.LG

27 pages, 13 figures

**SubmitDate**: 2022-08-25    [paper-pdf](http://arxiv.org/pdf/2208.12348v1)

**Authors**: Harsh Chaudhari, John Abascal, Alina Oprea, Matthew Jagielski, Florian Tramèr, Jonathan Ullman

**Abstracts**: Property inference attacks allow an adversary to extract global properties of the training dataset from a machine learning model. Such attacks have privacy implications for data owners who share their datasets to train machine learning models. Several existing approaches for property inference attacks against deep neural networks have been proposed, but they all rely on the attacker training a large number of shadow models, which induces large computational overhead.   In this paper, we consider the setting of property inference attacks in which the attacker can poison a subset of the training dataset and query the trained target model. Motivated by our theoretical analysis of model confidences under poisoning, we design an efficient property inference attack, SNAP, which obtains higher attack success and requires lower amounts of poisoning than the state-of-the-art poisoning-based property inference attack by Mahloujifar et al. For example, on the Census dataset, SNAP achieves 34% higher success rate than Mahloujifar et al. while being 56.5x faster. We also extend our attack to determine if a certain property is present at all in training, and estimate the exact proportion of a property of interest efficiently. We evaluate our attack on several properties of varying proportions from four datasets, and demonstrate SNAP's generality and effectiveness.



## **27. Semantic Preserving Adversarial Attack Generation with Autoencoder and Genetic Algorithm**

cs.LG

8 pages conference paper, accepted for publication in IEEE GLOBECOM  2022

**SubmitDate**: 2022-08-25    [paper-pdf](http://arxiv.org/pdf/2208.12230v1)

**Authors**: Xinyi Wang, Simon Yusuf Enoch, Dong Seong Kim

**Abstracts**: Widely used deep learning models are found to have poor robustness. Little noises can fool state-of-the-art models into making incorrect predictions. While there is a great deal of high-performance attack generation methods, most of them directly add perturbations to original data and measure them using L_p norms; this can break the major structure of data, thus, creating invalid attacks. In this paper, we propose a black-box attack, which, instead of modifying original data, modifies latent features of data extracted by an autoencoder; then, we measure noises in semantic space to protect the semantics of data. We trained autoencoders on MNIST and CIFAR-10 datasets and found optimal adversarial perturbations using a genetic algorithm. Our approach achieved a 100% attack success rate on the first 100 data of MNIST and CIFAR-10 datasets with less perturbation than FGSM.



## **28. Passive Triangulation Attack on ORide**

cs.CR

**SubmitDate**: 2022-08-25    [paper-pdf](http://arxiv.org/pdf/2208.12216v1)

**Authors**: Shyam Murthy, Srinivas Vivek

**Abstracts**: Privacy preservation in Ride Hailing Services is intended to protect privacy of drivers and riders. ORide is one of the early RHS proposals published at USENIX Security Symposium 2017. In the ORide protocol, riders and drivers, operating in a zone, encrypt their locations using a Somewhat Homomorphic Encryption scheme (SHE) and forward them to the Service Provider (SP). SP homomorphically computes the squared Euclidean distance between riders and available drivers. Rider receives the encrypted distances and selects the optimal rider after decryption. In order to prevent a triangulation attack, SP randomly permutes the distances before sending them to the rider. In this work, we use propose a passive attack that uses triangulation to determine coordinates of all participating drivers whose permuted distances are available from the points of view of multiple honest-but-curious adversary riders. An attack on ORide was published at SAC 2021. The same paper proposes a countermeasure using noisy Euclidean distances to thwart their attack. We extend our attack to determine locations of drivers when given their permuted and noisy Euclidean distances from multiple points of reference, where the noise perturbation comes from a uniform distribution. We conduct experiments with different number of drivers and for different perturbation values. Our experiments show that we can determine locations of all drivers participating in the ORide protocol. For the perturbed distance version of the ORide protocol, our algorithm reveals locations of about 25% to 50% of participating drivers. Our algorithm runs in time polynomial in number of drivers.



## **29. Automatic Mapping of Unstructured Cyber Threat Intelligence: An Experimental Study**

cs.CR

2022 IEEE 33rd International Symposium on Software Reliability  Engineering (ISSRE)

**SubmitDate**: 2022-08-25    [paper-pdf](http://arxiv.org/pdf/2208.12144v1)

**Authors**: Vittorio Orbinato, Mariarosaria Barbaraci, Roberto Natella, Domenico Cotroneo

**Abstracts**: Proactive approaches to security, such as adversary emulation, leverage information about threat actors and their techniques (Cyber Threat Intelligence, CTI). However, most CTI still comes in unstructured forms (i.e., natural language), such as incident reports and leaked documents. To support proactive security efforts, we present an experimental study on the automatic classification of unstructured CTI into attack techniques using machine learning (ML). We contribute with two new datasets for CTI analysis, and we evaluate several ML models, including both traditional and deep learning-based ones. We present several lessons learned about how ML can perform at this task, which classifiers perform best and under which conditions, which are the main causes of classification errors, and the challenges ahead for CTI analysis.



## **30. ECG-ATK-GAN: Robustness against Adversarial Attacks on ECGs using Conditional Generative Adversarial Networks**

eess.SP

Accepted to MICCAI2022 Applications of Medical AI (AMAI) Workshop

**SubmitDate**: 2022-08-25    [paper-pdf](http://arxiv.org/pdf/2110.09983v3)

**Authors**: Khondker Fariha Hossain, Sharif Amit Kamran, Alireza Tavakkoli, Xingjun Ma

**Abstracts**: Automating arrhythmia detection from ECG requires a robust and trusted system that retains high accuracy under electrical disturbances. Many machine learning approaches have reached human-level performance in classifying arrhythmia from ECGs. However, these architectures are vulnerable to adversarial attacks, which can misclassify ECG signals by decreasing the model's accuracy. Adversarial attacks are small crafted perturbations injected in the original data which manifest the out-of-distribution shifts in signal to misclassify the correct class. Thus, security concerns arise for false hospitalization and insurance fraud abusing these perturbations. To mitigate this problem, we introduce the first novel Conditional Generative Adversarial Network (GAN), robust against adversarial attacked ECG signals and retaining high accuracy. Our architecture integrates a new class-weighted objective function for adversarial perturbation identification and new blocks for discerning and combining out-of-distribution shifts in signals in the learning process for accurately classifying various arrhythmia types. Furthermore, we benchmark our architecture on six different white and black-box attacks and compare them with other recently proposed arrhythmia classification models on two publicly available ECG arrhythmia datasets. The experiment confirms that our model is more robust against such adversarial attacks for classifying arrhythmia with high accuracy.



## **31. A Perturbation Resistant Transformation and Classification System for Deep Neural Networks**

cs.CV

12 pages, 4 figures

**SubmitDate**: 2022-08-25    [paper-pdf](http://arxiv.org/pdf/2208.11839v1)

**Authors**: Nathaniel Dean, Dilip Sarkar

**Abstracts**: Deep convolutional neural networks accurately classify a diverse range of natural images, but may be easily deceived when designed, imperceptible perturbations are embedded in the images. In this paper, we design a multi-pronged training, input transformation, and image ensemble system that is attack agnostic and not easily estimated. Our system incorporates two novel features. The first is a transformation layer that computes feature level polynomial kernels from class-level training data samples and iteratively updates input image copies at inference time based on their feature kernel differences to create an ensemble of transformed inputs. The second is a classification system that incorporates the prediction of the undefended network with a hard vote on the ensemble of filtered images. Our evaluations on the CIFAR10 dataset show our system improves the robustness of an undefended network against a variety of bounded and unbounded white-box attacks under different distance metrics, while sacrificing little accuracy on clean images. Against adaptive full-knowledge attackers creating end-to-end attacks, our system successfully augments the existing robustness of adversarially trained networks, for which our methods are most effectively applied.



## **32. A New Kind of Adversarial Example**

cs.CV

**SubmitDate**: 2022-08-25    [paper-pdf](http://arxiv.org/pdf/2208.02430v2)

**Authors**: Ali Borji

**Abstracts**: Almost all adversarial attacks are formulated to add an imperceptible perturbation to an image in order to fool a model. Here, we consider the opposite which is adversarial examples that can fool a human but not a model. A large enough and perceptible perturbation is added to an image such that a model maintains its original decision, whereas a human will most likely make a mistake if forced to decide (or opt not to decide at all). Existing targeted attacks can be reformulated to synthesize such adversarial examples. Our proposed attack, dubbed NKE, is similar in essence to the fooling images, but is more efficient since it uses gradient descent instead of evolutionary algorithms. It also offers a new and unified perspective into the problem of adversarial vulnerability. Experimental results over MNIST and CIFAR-10 datasets show that our attack is quite efficient in fooling deep neural networks. Code is available at https://github.com/aliborji/NKE.



## **33. Attacking Neural Binary Function Detection**

cs.CR

18 pages

**SubmitDate**: 2022-08-24    [paper-pdf](http://arxiv.org/pdf/2208.11667v1)

**Authors**: Joshua Bundt, Michael Davinroy, Ioannis Agadakos, Alina Oprea, William Robertson

**Abstracts**: Binary analyses based on deep neural networks (DNNs), or neural binary analyses (NBAs), have become a hotly researched topic in recent years. DNNs have been wildly successful at pushing the performance and accuracy envelopes in the natural language and image processing domains. Thus, DNNs are highly promising for solving binary analysis problems that are typically hard due to a lack of complete information resulting from the lossy compilation process. Despite this promise, it is unclear that the prevailing strategy of repurposing embeddings and model architectures originally developed for other problem domains is sound given the adversarial contexts under which binary analysis often operates.   In this paper, we empirically demonstrate that the current state of the art in neural function boundary detection is vulnerable to both inadvertent and deliberate adversarial attacks. We proceed from the insight that current generation NBAs are built upon embeddings and model architectures intended to solve syntactic problems. We devise a simple, reproducible, and scalable black-box methodology for exploring the space of inadvertent attacks - instruction sequences that could be emitted by common compiler toolchains and configurations - that exploits this syntactic design focus. We then show that these inadvertent misclassifications can be exploited by an attacker, serving as the basis for a highly effective black-box adversarial example generation process. We evaluate this methodology against two state-of-the-art neural function boundary detectors: XDA and DeepDi. We conclude with an analysis of the evaluation data and recommendations for how future research might avoid succumbing to similar attacks.



## **34. Adversarial Driving: Attacking End-to-End Autonomous Driving**

cs.CV

7 pages, 6 figures

**SubmitDate**: 2022-08-24    [paper-pdf](http://arxiv.org/pdf/2103.09151v3)

**Authors**: Han Wu, Syed Yunas, Sareh Rowlands, Wenjie Ruan, Johan Wahlstrom

**Abstracts**: As the research in deep neural networks advances, deep convolutional networks become feasible for automated driving tasks. There is an emerging trend of employing end-to-end models in the automation of driving tasks. However, previous research unveils that deep neural networks are vulnerable to adversarial attacks in classification tasks. While for regression tasks such as autonomous driving, the effect of these attacks remains rarely explored. In this research, we devise two white-box targeted attacks against end-to-end autonomous driving systems. The driving model takes an image as input and outputs the steering angle. Our attacks can manipulate the behavior of the autonomous driving system only by perturbing the input image. Both attacks can be initiated in real-time on CPUs without employing GPUs. This research aims to raise concerns over applications of end-to-end models in safety-critical systems.



## **35. Unrestricted Black-box Adversarial Attack Using GAN with Limited Queries**

cs.CV

Accepted to the ECCV 2022 Workshop on Adversarial Robustness in the  Real World

**SubmitDate**: 2022-08-24    [paper-pdf](http://arxiv.org/pdf/2208.11613v1)

**Authors**: Dongbin Na, Sangwoo Ji, Jong Kim

**Abstracts**: Adversarial examples are inputs intentionally generated for fooling a deep neural network. Recent studies have proposed unrestricted adversarial attacks that are not norm-constrained. However, the previous unrestricted attack methods still have limitations to fool real-world applications in a black-box setting. In this paper, we present a novel method for generating unrestricted adversarial examples using GAN where an attacker can only access the top-1 final decision of a classification model. Our method, Latent-HSJA, efficiently leverages the advantages of a decision-based attack in the latent space and successfully manipulates the latent vectors for fooling the classification model.   With extensive experiments, we demonstrate that our proposed method is efficient in evaluating the robustness of classification models with limited queries in a black-box setting. First, we demonstrate that our targeted attack method is query-efficient to produce unrestricted adversarial examples for a facial identity recognition model that contains 307 identities. Then, we demonstrate that the proposed method can also successfully attack a real-world celebrity recognition service.



## **36. Robustness of the Tangle 2.0 Consensus**

cs.DC

**SubmitDate**: 2022-08-24    [paper-pdf](http://arxiv.org/pdf/2208.08254v2)

**Authors**: Bing-Yang Lin, Daria Dziubałtowska, Piotr Macek, Andreas Penzkofer, Sebastian Müller

**Abstracts**: In this paper, we investigate the performance of the Tangle 2.0 consensus protocol in a Byzantine environment. We use an agent-based simulation model that incorporates the main features of the Tangle 2.0 consensus protocol. Our experimental results demonstrate that the Tangle 2.0 protocol is robust to the bait-and-switch attack up to the theoretical upper bound of the adversary's 33% voting weight. We further show that the common coin mechanism in Tangle 2.0 is necessary for robustness against powerful adversaries. Moreover, the experimental results confirm that the protocol can achieve around 1s confirmation time in typical scenarios and that the confirmation times of non-conflicting transactions are not affected by the presence of conflicts.



## **37. LPF-Defense: 3D Adversarial Defense based on Frequency Analysis**

cs.CV

15 pages, 7 figures

**SubmitDate**: 2022-08-24    [paper-pdf](http://arxiv.org/pdf/2202.11287v2)

**Authors**: Hanieh Naderi, Kimia Noorbakhsh, Arian Etemadi, Shohreh Kasaei

**Abstracts**: Although 3D point cloud classification has recently been widely deployed in different application scenarios, it is still very vulnerable to adversarial attacks. This increases the importance of robust training of 3D models in the face of adversarial attacks. Based on our analysis on the performance of existing adversarial attacks, more adversarial perturbations are found in the mid and high-frequency components of input data. Therefore, by suppressing the high-frequency content in the training phase, the models robustness against adversarial examples is improved. Experiments showed that the proposed defense method decreases the success rate of six attacks on PointNet, PointNet++ ,, and DGCNN models. In particular, improvements are achieved with an average increase of classification accuracy by 3.8 % on drop100 attack and 4.26 % on drop200 attack compared to the state-of-the-art methods. The method also improves models accuracy on the original dataset compared to other available methods.



## **38. Trace and Detect Adversarial Attacks on CNNs using Feature Response Maps**

cs.CV

13 pages, 6 figures

**SubmitDate**: 2022-08-24    [paper-pdf](http://arxiv.org/pdf/2208.11436v1)

**Authors**: Mohammadreza Amirian, Friedhelm Schwenker, Thilo Stadelmann

**Abstracts**: The existence of adversarial attacks on convolutional neural networks (CNN) questions the fitness of such models for serious applications. The attacks manipulate an input image such that misclassification is evoked while still looking normal to a human observer -- they are thus not easily detectable. In a different context, backpropagated activations of CNN hidden layers -- "feature responses" to a given input -- have been helpful to visualize for a human "debugger" what the CNN "looks at" while computing its output. In this work, we propose a novel detection method for adversarial examples to prevent attacks. We do so by tracking adversarial perturbations in feature responses, allowing for automatic detection using average local spatial entropy. The method does not alter the original network architecture and is fully human-interpretable. Experiments confirm the validity of our approach for state-of-the-art attacks on large-scale models trained on ImageNet.



## **39. Towards an Awareness of Time Series Anomaly Detection Models' Adversarial Vulnerability**

cs.LG

Part of Proceedings of the 31st ACM International Conference on  Information and Knowledge Management (CIKM '22)

**SubmitDate**: 2022-08-24    [paper-pdf](http://arxiv.org/pdf/2208.11264v1)

**Authors**: Shahroz Tariq, Binh M. Le, Simon S. Woo

**Abstracts**: Time series anomaly detection is extensively studied in statistics, economics, and computer science. Over the years, numerous methods have been proposed for time series anomaly detection using deep learning-based methods. Many of these methods demonstrate state-of-the-art performance on benchmark datasets, giving the false impression that these systems are robust and deployable in many practical and industrial real-world scenarios. In this paper, we demonstrate that the performance of state-of-the-art anomaly detection methods is degraded substantially by adding only small adversarial perturbations to the sensor data. We use different scoring metrics such as prediction errors, anomaly, and classification scores over several public and private datasets ranging from aerospace applications, server machines, to cyber-physical systems in power plants. Under well-known adversarial attacks from Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) methods, we demonstrate that state-of-the-art deep neural networks (DNNs) and graph neural networks (GNNs) methods, which claim to be robust against anomalies and have been possibly integrated in real-life systems, have their performance drop to as low as 0%. To the best of our understanding, we demonstrate, for the first time, the vulnerabilities of anomaly detection systems against adversarial attacks. The overarching goal of this research is to raise awareness towards the adversarial vulnerabilities of time series anomaly detectors.



## **40. ObfuNAS: A Neural Architecture Search-based DNN Obfuscation Approach**

cs.CR

9 pages

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2208.08569v2)

**Authors**: Tong Zhou, Shaolei Ren, Xiaolin Xu

**Abstracts**: Malicious architecture extraction has been emerging as a crucial concern for deep neural network (DNN) security. As a defense, architecture obfuscation is proposed to remap the victim DNN to a different architecture. Nonetheless, we observe that, with only extracting an obfuscated DNN architecture, the adversary can still retrain a substitute model with high performance (e.g., accuracy), rendering the obfuscation techniques ineffective. To mitigate this under-explored vulnerability, we propose ObfuNAS, which converts the DNN architecture obfuscation into a neural architecture search (NAS) problem. Using a combination of function-preserving obfuscation strategies, ObfuNAS ensures that the obfuscated DNN architecture can only achieve lower accuracy than the victim. We validate the performance of ObfuNAS with open-source architecture datasets like NAS-Bench-101 and NAS-Bench-301. The experimental results demonstrate that ObfuNAS can successfully find the optimal mask for a victim model within a given FLOPs constraint, leading up to 2.6% inference accuracy degradation for attackers with only 0.14x FLOPs overhead. The code is available at: https://github.com/Tongzhou0101/ObfuNAS.



## **41. Auditing Membership Leakages of Multi-Exit Networks**

cs.CR

Accepted by CCS 2022

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2208.11180v1)

**Authors**: Zheng Li, Yiyong Liu, Xinlei He, Ning Yu, Michael Backes, Yang Zhang

**Abstracts**: Relying on the fact that not all inputs require the same amount of computation to yield a confident prediction, multi-exit networks are gaining attention as a prominent approach for pushing the limits of efficient deployment. Multi-exit networks endow a backbone model with early exits, allowing to obtain predictions at intermediate layers of the model and thus save computation time and/or energy. However, current various designs of multi-exit networks are only considered to achieve the best trade-off between resource usage efficiency and prediction accuracy, the privacy risks stemming from them have never been explored. This prompts the need for a comprehensive investigation of privacy risks in multi-exit networks.   In this paper, we perform the first privacy analysis of multi-exit networks through the lens of membership leakages. In particular, we first leverage the existing attack methodologies to quantify the multi-exit networks' vulnerability to membership leakages. Our experimental results show that multi-exit networks are less vulnerable to membership leakages and the exit (number and depth) attached to the backbone model is highly correlated with the attack performance. Furthermore, we propose a hybrid attack that exploits the exit information to improve the performance of existing attacks. We evaluate membership leakage threat caused by our hybrid attack under three different adversarial setups, ultimately arriving at a model-free and data-free adversary. These results clearly demonstrate that our hybrid attacks are very broadly applicable, thereby the corresponding risks are much more severe than shown by existing membership inference attacks. We further present a defense mechanism called TimeGuard specifically for multi-exit networks and show that TimeGuard mitigates the newly proposed attacks perfectly.



## **42. Adversarial Speaker Distillation for Countermeasure Model on Automatic Speaker Verification**

cs.SD

Accepted by ISCA SPSC 2022

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2203.17031v5)

**Authors**: Yen-Lun Liao, Xuanjun Chen, Chung-Che Wang, Jyh-Shing Roger Jang

**Abstracts**: The countermeasure (CM) model is developed to protect ASV systems from spoof attacks and prevent resulting personal information leakage in Automatic Speaker Verification (ASV) system. Based on practicality and security considerations, the CM model is usually deployed on edge devices, which have more limited computing resources and storage space than cloud-based systems, confining the model size under a limitation. To better trade off the CM model sizes and performance, we proposed an adversarial speaker distillation method, which is an improved version of knowledge distillation method combined with generalized end-to-end (GE2E) pre-training and adversarial fine-tuning. In the evaluation phase of the ASVspoof 2021 Logical Access task, our proposed adversarial speaker distillation ResNetSE (ASD-ResNetSE) model reaches 0.2695 min t-DCF and 3.54\% EER. ASD-ResNetSE only used 22.5\% of parameters and 19.4\% of multiply and accumulate operands of ResNetSE model.



## **43. Privacy Enhancement for Cloud-Based Few-Shot Learning**

cs.LG

14 pages, 13 figures, 3 tables. Preprint. Accepted in IEEE WCCI 2022  International Joint Conference on Neural Networks (IJCNN)

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2205.07864v2)

**Authors**: Archit Parnami, Muhammad Usama, Liyue Fan, Minwoo Lee

**Abstracts**: Requiring less data for accurate models, few-shot learning has shown robustness and generality in many application domains. However, deploying few-shot models in untrusted environments may inflict privacy concerns, e.g., attacks or adversaries that may breach the privacy of user-supplied data. This paper studies the privacy enhancement for the few-shot learning in an untrusted environment, e.g., the cloud, by establishing a novel privacy-preserved embedding space that preserves the privacy of data and maintains the accuracy of the model. We examine the impact of various image privacy methods such as blurring, pixelization, Gaussian noise, and differentially private pixelization (DP-Pix) on few-shot image classification and propose a method that learns privacy-preserved representation through the joint loss. The empirical results show how privacy-performance trade-off can be negotiated for privacy-enhanced few-shot learning.



## **44. A Comprehensive Study of Real-Time Object Detection Networks Across Multiple Domains: A Survey**

cs.CV

Published in Transactions on Machine Learning Research (TMLR) with  Survey Certification

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2208.10895v1)

**Authors**: Elahe Arani, Shruthi Gowda, Ratnajit Mukherjee, Omar Magdy, Senthilkumar Kathiresan, Bahram Zonooz

**Abstracts**: Deep neural network based object detectors are continuously evolving and are used in a multitude of applications, each having its own set of requirements. While safety-critical applications need high accuracy and reliability, low-latency tasks need resource and energy-efficient networks. Real-time detectors, which are a necessity in high-impact real-world applications, are continuously proposed, but they overemphasize the improvements in accuracy and speed while other capabilities such as versatility, robustness, resource and energy efficiency are omitted. A reference benchmark for existing networks does not exist, nor does a standard evaluation guideline for designing new networks, which results in ambiguous and inconsistent comparisons. We, thus, conduct a comprehensive study on multiple real-time detectors (anchor-, keypoint-, and transformer-based) on a wide range of datasets and report results on an extensive set of metrics. We also study the impact of variables such as image size, anchor dimensions, confidence thresholds, and architecture layers on the overall performance. We analyze the robustness of detection networks against distribution shifts, natural corruptions, and adversarial attacks. Also, we provide a calibration analysis to gauge the reliability of the predictions. Finally, to highlight the real-world impact, we conduct two unique case studies, on autonomous driving and healthcare applications. To further gauge the capability of networks in critical real-time applications, we report the performance after deploying the detection networks on edge devices. Our extensive empirical study can act as a guideline for the industrial community to make an informed choice on the existing networks. We also hope to inspire the research community towards a new direction in the design and evaluation of networks that focuses on a bigger and holistic overview for a far-reaching impact.



## **45. Transferability Ranking of Adversarial Examples**

cs.LG

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2208.10878v1)

**Authors**: Mosh Levy, Yuval Elovici, Yisroel Mirsky

**Abstracts**: Adversarial examples can be used to maliciously and covertly change a model's prediction. It is known that an adversarial example designed for one model can transfer to other models as well. This poses a major threat because it means that attackers can target systems in a blackbox manner.   In the domain of transferability, researchers have proposed ways to make attacks more transferable and to make models more robust to transferred examples. However, to the best of our knowledge, there are no works which propose a means for ranking the transferability of an adversarial example in the perspective of a blackbox attacker. This is an important task because an attacker is likely to use only a select set of examples, and therefore will want to select the samples which are most likely to transfer.   In this paper we suggest a method for ranking the transferability of adversarial examples without access to the victim's model. To accomplish this, we define and estimate the expected transferability of a sample given limited information about the victim. We also explore practical scenarios: where the adversary can select the best sample to attack and where the adversary must use a specific sample but can choose different perturbations. Through our experiments, we found that our ranking method can increase an attacker's success rate by up to 80% compared to the baseline (random selection without ranking).



## **46. Complete Traceability Multimedia Fingerprinting Codes Resistant to Averaging Attack and Adversarial Noise with Optimal Rate**

cs.IT

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2108.09015v4)

**Authors**: Ilya Vorobyev

**Abstracts**: In this paper we consider complete traceability multimedia fingerprinting codes resistant to averaging attacks and adversarial noise. Recently it was shown that there are no such codes for the case of an arbitrary linear attack. However, for the case of averaging attacks complete traceability multimedia fingerprinting codes of exponential cardinality resistant to constant adversarial noise were constructed in 2020 by Egorova et al. We continue this work and provide an improved lower bound on the rate of these codes.



## **47. Evaluating Machine Unlearning via Epistemic Uncertainty**

cs.LG

Rejected at ECML 2021. Even though the paper was rejected, we want to  "publish" it on arxiv, since we believe that it is nevertheless interesting  to investigate the connections between unlearning and uncertainty

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2208.10836v1)

**Authors**: Alexander Becker, Thomas Liebig

**Abstracts**: There has been a growing interest in Machine Unlearning recently, primarily due to legal requirements such as the General Data Protection Regulation (GDPR) and the California Consumer Privacy Act. Thus, multiple approaches were presented to remove the influence of specific target data points from a trained model. However, when evaluating the success of unlearning, current approaches either use adversarial attacks or compare their results to the optimal solution, which usually incorporates retraining from scratch. We argue that both ways are insufficient in practice. In this work, we present an evaluation metric for Machine Unlearning algorithms based on epistemic uncertainty. This is the first definition of a general evaluation metric for Machine Unlearning to our best knowledge.



## **48. UKP-SQuARE v2 Explainability and Adversarial Attacks for Trustworthy QA**

cs.CL

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2208.09316v2)

**Authors**: Rachneet Sachdeva, Haritz Puerto, Tim Baumgärtner, Sewin Tariverdian, Hao Zhang, Kexin Wang, Hossain Shaikh Saadi, Leonardo F. R. Ribeiro, Iryna Gurevych

**Abstracts**: Question Answering (QA) systems are increasingly deployed in applications where they support real-world decisions. However, state-of-the-art models rely on deep neural networks, which are difficult to interpret by humans. Inherently interpretable models or post hoc explainability methods can help users to comprehend how a model arrives at its prediction and, if successful, increase their trust in the system. Furthermore, researchers can leverage these insights to develop new methods that are more accurate and less biased. In this paper, we introduce SQuARE v2, the new version of SQuARE, to provide an explainability infrastructure for comparing models based on methods such as saliency maps and graph-based explanations. While saliency maps are useful to inspect the importance of each input token for the model's prediction, graph-based explanations from external Knowledge Graphs enable the users to verify the reasoning behind the model prediction. In addition, we provide multiple adversarial attacks to compare the robustness of QA models. With these explainability methods and adversarial attacks, we aim to ease the research on trustworthy QA models. SQuARE is available on https://square.ukp-lab.de.



## **49. SoK: Certified Robustness for Deep Neural Networks**

cs.LG

To appear at 2023 IEEE Symposium on Security and Privacy (SP); 14  pages for the main text; benchmark & tool website:  http://sokcertifiedrobustness.github.io/

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2009.04131v7)

**Authors**: Linyi Li, Tao Xie, Bo Li

**Abstracts**: Great advances in deep neural networks (DNNs) have led to state-of-the-art performance on a wide range of tasks. However, recent studies have shown that DNNs are vulnerable to adversarial attacks, which have brought great concerns when deploying these models to safety-critical applications such as autonomous driving. Different defense approaches have been proposed against adversarial attacks, including: a) empirical defenses, which can usually be adaptively attacked again without providing robustness certification; and b) certifiably robust approaches, which consist of robustness verification providing the lower bound of robust accuracy against any attacks under certain conditions and corresponding robust training approaches. In this paper, we systematize certifiably robust approaches and related practical and theoretical implications and findings. We also provide the first comprehensive benchmark on existing robustness verification and training approaches on different datasets. In particular, we 1) provide a taxonomy for the robustness verification and training approaches, as well as summarize the methodologies for representative algorithms, 2) reveal the characteristics, strengths, limitations, and fundamental connections among these approaches, 3) discuss current research progresses, theoretical barriers, main challenges, and future directions for certifiably robust approaches for DNNs, and 4) provide an open-sourced unified platform to evaluate 20+ representative certifiably robust approaches.



## **50. Adversarial Vulnerability of Temporal Feature Networks for Object Detection**

cs.CV

Accepted for publication at ECCV 2022 SAIAD workshop

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2208.10773v1)

**Authors**: Svetlana Pavlitskaya, Nikolai Polley, Michael Weber, J. Marius Zöllner

**Abstracts**: Taking into account information across the temporal domain helps to improve environment perception in autonomous driving. However, it has not been studied so far whether temporally fused neural networks are vulnerable to deliberately generated perturbations, i.e. adversarial attacks, or whether temporal history is an inherent defense against them. In this work, we study whether temporal feature networks for object detection are vulnerable to universal adversarial attacks. We evaluate attacks of two types: imperceptible noise for the whole image and locally-bound adversarial patch. In both cases, perturbations are generated in a white-box manner using PGD. Our experiments confirm, that attacking even a portion of a temporal input suffices to fool the network. We visually assess generated perturbations to gain insights into the functioning of attacks. To enhance the robustness, we apply adversarial training using 5-PGD. Our experiments on KITTI and nuScenes datasets demonstrate, that a model robustified via K-PGD is able to withstand the studied attacks while keeping the mAP-based performance comparable to that of an unattacked model.



