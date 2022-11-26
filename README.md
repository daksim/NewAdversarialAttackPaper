# Latest Adversarial Attack Papers
**update at 2022-11-26 11:39:11**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Query Efficient Cross-Dataset Transferable Black-Box Attack on Action Recognition**

cs.CV

**SubmitDate**: 2022-11-23    [abs](http://arxiv.org/abs/2211.13171v1) [paper-pdf](http://arxiv.org/pdf/2211.13171v1)

**Authors**: Rohit Gupta, Naveed Akhtar, Gaurav Kumar Nayak, Ajmal Mian, Mubarak Shah

**Abstract**: Black-box adversarial attacks present a realistic threat to action recognition systems. Existing black-box attacks follow either a query-based approach where an attack is optimized by querying the target model, or a transfer-based approach where attacks are generated using a substitute model. While these methods can achieve decent fooling rates, the former tends to be highly query-inefficient while the latter assumes extensive knowledge of the black-box model's training data. In this paper, we propose a new attack on action recognition that addresses these shortcomings by generating perturbations to disrupt the features learned by a pre-trained substitute model to reduce the number of queries. By using a nearly disjoint dataset to train the substitute model, our method removes the requirement that the substitute model be trained using the same dataset as the target model, and leverages queries to the target model to retain the fooling rate benefits provided by query-based methods. This ultimately results in attacks which are more transferable than conventional black-box attacks. Through extensive experiments, we demonstrate highly query-efficient black-box attacks with the proposed framework. Our method achieves 8% and 12% higher deception rates compared to state-of-the-art query-based and transfer-based attacks, respectively.



## **2. Adversarial Attacks are a Surprisingly Strong Baseline for Poisoning Few-Shot Meta-Learners**

cs.LG

Accepted at I Can't Believe It's Not Better Workshop, Neurips 2022

**SubmitDate**: 2022-11-23    [abs](http://arxiv.org/abs/2211.12990v1) [paper-pdf](http://arxiv.org/pdf/2211.12990v1)

**Authors**: Elre T. Oldewage, John Bronskill, Richard E. Turner

**Abstract**: This paper examines the robustness of deployed few-shot meta-learning systems when they are fed an imperceptibly perturbed few-shot dataset. We attack amortized meta-learners, which allows us to craft colluding sets of inputs that are tailored to fool the system's learning algorithm when used as training data. Jointly crafted adversarial inputs might be expected to synergistically manipulate a classifier, allowing for very strong data-poisoning attacks that would be hard to detect. We show that in a white box setting, these attacks are very successful and can cause the target model's predictions to become worse than chance. However, in opposition to the well-known transferability of adversarial examples in general, the colluding sets do not transfer well to different classifiers. We explore two hypotheses to explain this: 'overfitting' by the attack, and mismatch between the model on which the attack is generated and that to which the attack is transferred. Regardless of the mitigation strategies suggested by these hypotheses, the colluding inputs transfer no better than adversarial inputs that are generated independently in the usual way.



## **3. Visual Information Hiding Based on Obfuscating Adversarial Perturbations**

cs.CV

**SubmitDate**: 2022-11-23    [abs](http://arxiv.org/abs/2209.15304v2) [paper-pdf](http://arxiv.org/pdf/2209.15304v2)

**Authors**: Zhigang Su, Dawei Zhou, Decheng Liu, Nannan Wang, Zhen Wang, Xinbo Gao

**Abstract**: Growing leakage and misuse of visual information raise security and privacy concerns, which promotes the development of information protection. Existing adversarial perturbations-based methods mainly focus on the de-identification against deep learning models. However, the inherent visual information of the data has not been well protected. In this work, inspired by the Type-I adversarial attack, we propose an adversarial visual information hiding method to protect the visual privacy of data. Specifically, the method generates obfuscating adversarial perturbations to obscure the visual information of the data. Meanwhile, it maintains the hidden objectives to be correctly predicted by models. In addition, our method does not modify the parameters of the applied model, which makes it flexible for different scenarios. Experimental results on the recognition and classification tasks demonstrate that the proposed method can effectively hide visual information and hardly affect the performances of models. The code is available in the supplementary material.



## **4. Privacy-Enhancing Optical Embeddings for Lensless Classification**

cs.CV

30 pages, 27 figures, under review, for code, see  https://github.com/ebezzam/LenslessClassification. arXiv admin note:  substantial text overlap with arXiv:2206.01429

**SubmitDate**: 2022-11-23    [abs](http://arxiv.org/abs/2211.12864v1) [paper-pdf](http://arxiv.org/pdf/2211.12864v1)

**Authors**: Eric Bezzam, Martin Vetterli, Matthieu Simeoni

**Abstract**: Lensless imaging can provide visual privacy due to the highly multiplexed characteristic of its measurements. However, this alone is a weak form of security, as various adversarial attacks can be designed to invert the one-to-many scene mapping of such cameras. In this work, we enhance the privacy provided by lensless imaging by (1) downsampling at the sensor and (2) using a programmable mask with variable patterns as our optical encoder. We build a prototype from a low-cost LCD and Raspberry Pi components, for a total cost of around 100 USD. This very low price point allows our system to be deployed and leveraged in a broad range of applications. In our experiments, we first demonstrate the viability and reconfigurability of our system by applying it to various classification tasks: MNIST, CelebA (face attributes), and CIFAR10. By jointly optimizing the mask pattern and a digital classifier in an end-to-end fashion, low-dimensional, privacy-enhancing embeddings are learned directly at the sensor. Secondly, we show how the proposed system, through variable mask patterns, can thwart adversaries that attempt to invert the system (1) via plaintext attacks or (2) in the event of camera parameters leaks. We demonstrate the defense of our system to both risks, with 55% and 26% drops in image quality metrics for attacks based on model-based convex optimization and generative neural networks respectively. We open-source a wave propagation and camera simulator needed for end-to-end optimization, the training software, and a library for interfacing with the camera.



## **5. Byzantine Multiple Access Channels -- Part I: Reliable Communication**

cs.IT

This supercedes Part I of arxiv:1904.11925

**SubmitDate**: 2022-11-23    [abs](http://arxiv.org/abs/2211.12769v1) [paper-pdf](http://arxiv.org/pdf/2211.12769v1)

**Authors**: Neha Sangwan, Mayank Bakshi, Bikash Kumar Dey, Vinod M. Prabhakaran

**Abstract**: We study communication over a Multiple Access Channel (MAC) where users can possibly be adversarial. When all users are non-adversarial, we want their messages to be decoded reliably. When a user behaves adversarially, we require that the honest users' messages be decoded reliably. An adversarial user can mount an attack by sending any input into the channel rather than following the protocol. It turns out that the $2$-user MAC capacity region follows from the point-to-point Arbitrarily Varying Channel (AVC) capacity. For the $3$-user MAC in which at most one user may be malicious, we characterize the capacity region for deterministic codes and randomized codes (where each user shares an independent random secret key with the receiver). These results are then generalized for the $k$-user MAC where the adversary may control all users in one out of a collection of given subsets.



## **6. Reliable Robustness Evaluation via Automatically Constructed Attack Ensembles**

cs.LG

**SubmitDate**: 2022-11-23    [abs](http://arxiv.org/abs/2211.12713v1) [paper-pdf](http://arxiv.org/pdf/2211.12713v1)

**Authors**: Shengcai Liu, Fu Peng, Ke Tang

**Abstract**: Attack Ensemble (AE), which combines multiple attacks together, provides a reliable way to evaluate adversarial robustness. In practice, AEs are often constructed and tuned by human experts, which however tends to be sub-optimal and time-consuming. In this work, we present AutoAE, a conceptually simple approach for automatically constructing AEs. In brief, AutoAE repeatedly adds the attack and its iteration steps to the ensemble that maximizes ensemble improvement per additional iteration consumed. We show theoretically that AutoAE yields AEs provably within a constant factor of the optimal for a given defense. We then use AutoAE to construct two AEs for $l_{\infty}$ and $l_2$ attacks, and apply them without any tuning or adaptation to 45 top adversarial defenses on the RobustBench leaderboard. In all except one cases we achieve equal or better (often the latter) robustness evaluation than existing AEs, and notably, in 29 cases we achieve better robustness evaluation than the best known one. Such performance of AutoAE shows itself as a reliable evaluation protocol for adversarial robustness, which further indicates the huge potential of automatic AE construction. Code is available at \url{https://github.com/LeegerPENG/AutoAE}.



## **7. Benchmarking Adversarially Robust Quantum Machine Learning at Scale**

quant-ph

10 pages, 5 Figures

**SubmitDate**: 2022-11-23    [abs](http://arxiv.org/abs/2211.12681v1) [paper-pdf](http://arxiv.org/pdf/2211.12681v1)

**Authors**: Maxwell T. West, Sarah M. Erfani, Christopher Leckie, Martin Sevior, Lloyd C. L. Hollenberg, Muhammad Usman

**Abstract**: Machine learning (ML) methods such as artificial neural networks are rapidly becoming ubiquitous in modern science, technology and industry. Despite their accuracy and sophistication, neural networks can be easily fooled by carefully designed malicious inputs known as adversarial attacks. While such vulnerabilities remain a serious challenge for classical neural networks, the extent of their existence is not fully understood in the quantum ML setting. In this work, we benchmark the robustness of quantum ML networks, such as quantum variational classifiers (QVC), at scale by performing rigorous training for both simple and complex image datasets and through a variety of high-end adversarial attacks. Our results show that QVCs offer a notably enhanced robustness against classical adversarial attacks by learning features which are not detected by the classical neural networks, indicating a possible quantum advantage for ML tasks. Contrarily, and remarkably, the converse is not true, with attacks on quantum networks also capable of deceiving classical neural networks. By combining quantum and classical network outcomes, we propose a novel adversarial attack detection technology. Traditionally quantum advantage in ML systems has been sought through increased accuracy or algorithmic speed-up, but our work has revealed the potential for a new kind of quantum advantage through superior robustness of ML models, whose practical realisation will address serious security concerns and reliability issues of ML algorithms employed in a myriad of applications including autonomous vehicles, cybersecurity, and surveillance robotic systems.



## **8. Membership Inference Attacks via Adversarial Examples**

cs.LG

Trustworthy and Socially Responsible Machine Learning (TSRML 2022)  co-located with NeurIPS 2022

**SubmitDate**: 2022-11-23    [abs](http://arxiv.org/abs/2207.13572v2) [paper-pdf](http://arxiv.org/pdf/2207.13572v2)

**Authors**: Hamid Jalalzai, Elie Kadoche, Rémi Leluc, Vincent Plassier

**Abstract**: The raise of machine learning and deep learning led to significant improvement in several domains. This change is supported by both the dramatic rise in computation power and the collection of large datasets. Such massive datasets often include personal data which can represent a threat to privacy. Membership inference attacks are a novel direction of research which aims at recovering training data used by a learning algorithm. In this paper, we develop a mean to measure the leakage of training data leveraging a quantity appearing as a proxy of the total variation of a trained model near its training samples. We extend our work by providing a novel defense mechanism. Our contributions are supported by empirical evidence through convincing numerical experiments.



## **9. Improving Robust Generalization by Direct PAC-Bayesian Bound Minimization**

cs.LG

**SubmitDate**: 2022-11-22    [abs](http://arxiv.org/abs/2211.12624v1) [paper-pdf](http://arxiv.org/pdf/2211.12624v1)

**Authors**: Zifan Wang, Nan Ding, Tomer Levinboim, Xi Chen, Radu Soricut

**Abstract**: Recent research in robust optimization has shown an overfitting-like phenomenon in which models trained against adversarial attacks exhibit higher robustness on the training set compared to the test set. Although previous work provided theoretical explanations for this phenomenon using a robust PAC-Bayesian bound over the adversarial test error, related algorithmic derivations are at best only loosely connected to this bound, which implies that there is still a gap between their empirical success and our understanding of adversarial robustness theory. To close this gap, in this paper we consider a different form of the robust PAC-Bayesian bound and directly minimize it with respect to the model posterior. The derivation of the optimal solution connects PAC-Bayesian learning to the geometry of the robust loss surface through a Trace of Hessian (TrH) regularizer that measures the surface flatness. In practice, we restrict the TrH regularizer to the top layer only, which results in an analytical solution to the bound whose computational cost does not depend on the network depth. Finally, we evaluate our TrH regularization approach over CIFAR-10/100 and ImageNet using Vision Transformers (ViT) and compare against baseline adversarial robustness algorithms. Experimental results show that TrH regularization leads to improved ViT robustness that either matches or surpasses previous state-of-the-art approaches while at the same time requires less memory and computational cost.



## **10. Diagnostics for Deep Neural Networks with Automated Copy/Paste Attacks**

cs.LG

**SubmitDate**: 2022-11-22    [abs](http://arxiv.org/abs/2211.10024v2) [paper-pdf](http://arxiv.org/pdf/2211.10024v2)

**Authors**: Stephen Casper, Kaivalya Hariharan, Dylan Hadfield-Menell

**Abstract**: Deep neural networks (DNNs) are powerful, but they can make mistakes that pose significant risks. A model performing well on a test set does not imply safety in deployment, so it is important to have additional tools to understand its flaws. Adversarial examples can help reveal weaknesses, but they are often difficult for a human to interpret or draw generalizable, actionable conclusions from. Some previous works have addressed this by studying human-interpretable attacks. We build on these with three contributions. First, we introduce a method termed Search for Natural Adversarial Features Using Embeddings (SNAFUE) which offers a fully-automated method for finding "copy/paste" attacks in which one natural image can be pasted into another in order to induce an unrelated misclassification. Second, we use this to red team an ImageNet classifier and identify hundreds of easily-describable sets of vulnerabilities. Third, we compare this approach with other interpretability tools by attempting to rediscover trojans. Our results suggest that SNAFUE can be useful for interpreting DNNs and generating adversarial data for them. Code is available at https://github.com/thestephencasper/snafue



## **11. Attacking Image Splicing Detection and Localization Algorithms Using Synthetic Traces**

eess.IV

**SubmitDate**: 2022-11-22    [abs](http://arxiv.org/abs/2211.12314v1) [paper-pdf](http://arxiv.org/pdf/2211.12314v1)

**Authors**: Shengbang Fang, Matthew C Stamm

**Abstract**: Recent advances in deep learning have enabled forensics researchers to develop a new class of image splicing detection and localization algorithms. These algorithms identify spliced content by detecting localized inconsistencies in forensic traces using Siamese neural networks, either explicitly during analysis or implicitly during training. At the same time, deep learning has enabled new forms of anti-forensic attacks, such as adversarial examples and generative adversarial network (GAN) based attacks. Thus far, however, no anti-forensic attack has been demonstrated against image splicing detection and localization algorithms. In this paper, we propose a new GAN-based anti-forensic attack that is able to fool state-of-the-art splicing detection and localization algorithms such as EXIF-Net, Noiseprint, and Forensic Similarity Graphs. This attack operates by adversarially training an anti-forensic generator against a set of Siamese neural networks so that it is able to create synthetic forensic traces. Under analysis, these synthetic traces appear authentic and are self-consistent throughout an image. Through a series of experiments, we demonstrate that our attack is capable of fooling forensic splicing detection and localization algorithms without introducing visually detectable artifacts into an attacked image. Additionally, we demonstrate that our attack outperforms existing alternative attack approaches. %



## **12. Jointly Attacking Graph Neural Network and its Explanations**

cs.LG

Accepted by ICDE 2023 (39th IEEE International Conference on Data  Engineering)

**SubmitDate**: 2022-11-22    [abs](http://arxiv.org/abs/2108.03388v2) [paper-pdf](http://arxiv.org/pdf/2108.03388v2)

**Authors**: Wenqi Fan, Wei Jin, Xiaorui Liu, Han Xu, Xianfeng Tang, Suhang Wang, Qing Li, Jiliang Tang, Jianping Wang, Charu Aggarwal

**Abstract**: Graph Neural Networks (GNNs) have boosted the performance for many graph-related tasks. Despite the great success, recent studies have shown that GNNs are highly vulnerable to adversarial attacks, where adversaries can mislead the GNNs' prediction by modifying graphs. On the other hand, the explanation of GNNs (GNNExplainer) provides a better understanding of a trained GNN model by generating a small subgraph and features that are most influential for its prediction. In this paper, we first perform empirical studies to validate that GNNExplainer can act as an inspection tool and have the potential to detect the adversarial perturbations for graphs. This finding motivates us to further initiate a new problem investigation: Whether a graph neural network and its explanations can be jointly attacked by modifying graphs with malicious desires? It is challenging to answer this question since the goals of adversarial attacks and bypassing the GNNExplainer essentially contradict each other. In this work, we give a confirmative answer to this question by proposing a novel attack framework (GEAttack), which can attack both a GNN model and its explanations by simultaneously exploiting their vulnerabilities. Extensive experiments on two explainers (GNNExplainer and PGExplainer) under various real-world datasets demonstrate the effectiveness of the proposed method.



## **13. PointCA: Evaluating the Robustness of 3D Point Cloud Completion Models Against Adversarial Examples**

cs.CV

Accepted by the 37th AAAI Conference on Artificial Intelligence  (AAAI-23)

**SubmitDate**: 2022-11-22    [abs](http://arxiv.org/abs/2211.12294v1) [paper-pdf](http://arxiv.org/pdf/2211.12294v1)

**Authors**: Shengshan Hu, Junwei Zhang, Wei Liu, Junhui Hou, Minghui Li, Leo Yu Zhang, Hai Jin, Lichao Sun

**Abstract**: Point cloud completion, as the upstream procedure of 3D recognition and segmentation, has become an essential part of many tasks such as navigation and scene understanding. While various point cloud completion models have demonstrated their powerful capabilities, their robustness against adversarial attacks, which have been proven to be fatally malicious towards deep neural networks, remains unknown. In addition, existing attack approaches towards point cloud classifiers cannot be applied to the completion models due to different output forms and attack purposes. In order to evaluate the robustness of the completion models, we propose PointCA, the first adversarial attack against 3D point cloud completion models. PointCA can generate adversarial point clouds that maintain high similarity with the original ones, while being completed as another object with totally different semantic information. Specifically, we minimize the representation discrepancy between the adversarial example and the target point set to jointly explore the adversarial point clouds in the geometry space and the feature space. Furthermore, to launch a stealthier attack, we innovatively employ the neighbourhood density information to tailor the perturbation constraint, leading to geometry-aware and distribution-adaptive modifications for each point. Extensive experiments against different premier point cloud completion networks show that PointCA can cause a performance degradation from 77.9% to 16.7%, with the structure chamfer distance kept below 0.01. We conclude that existing completion models are severely vulnerable to adversarial examples, and state-of-the-art defenses for point cloud classification will be partially invalid when applied to incomplete and uneven point cloud data.



## **14. SoK: Inference Attacks and Defenses in Human-Centered Wireless Sensing**

cs.CR

**SubmitDate**: 2022-11-22    [abs](http://arxiv.org/abs/2211.12087v1) [paper-pdf](http://arxiv.org/pdf/2211.12087v1)

**Authors**: Wei Sun, Tingjun Chen, Neil Gong

**Abstract**: Human-centered wireless sensing aims to understand the fine-grained environment and activities of a human using the diverse wireless signals around her. The wireless sensing community has demonstrated the superiority of such techniques in many applications such as smart homes, human-computer interactions, and smart cities. Like many other technologies, wireless sensing is also a double-edged sword. While the sensed information about a human can be used for many good purposes such as enhancing life quality, an adversary can also abuse it to steal private information about the human (e.g., location, living habits, and behavioral biometric characteristics). However, the literature lacks a systematic understanding of the privacy vulnerabilities of wireless sensing and the defenses against them.   In this work, we aim to bridge this gap. First, we propose a framework to systematize wireless sensing-based inference attacks. Our framework consists of three key steps: deploying a sniffing device, sniffing wireless signals, and inferring private information. Our framework can be used to guide the design of new inference attacks since different attacks can instantiate these three steps differently. Second, we propose a defense-in-depth framework to systematize defenses against such inference attacks. The prevention component of our framework aims to prevent inference attacks via obfuscating the wireless signals around a human, while the detection component aims to detect and respond to attacks. Third, based on our attack and defense frameworks, we identify gaps in the existing literature and discuss future research directions.



## **15. How Fraudster Detection Contributes to Robust Recommendation**

cs.IR

**SubmitDate**: 2022-11-22    [abs](http://arxiv.org/abs/2211.11534v2) [paper-pdf](http://arxiv.org/pdf/2211.11534v2)

**Authors**: Yuni Lai, Kai Zhou

**Abstract**: The adversarial robustness of recommendation systems under node injection attacks has received considerable research attention. Recently, a robust recommendation system GraphRfi was proposed, and it was shown that GraphRfi could successfully mitigate the effects of injected fake users in the system. Unfortunately, we demonstrate that GraphRfi is still vulnerable to attacks due to the supervised nature of its fraudster detection component. Specifically, we propose a new attack metaC against GraphRfi, and further analyze why GraphRfi fails under such an attack. Based on the insights we obtained from the vulnerability analysis, we build a new robust recommendation system PDR by re-designing the fraudster detection component. Comprehensive experiments show that our defense approach outperforms other benchmark methods under attacks. Overall, our research demonstrates an effective framework of integrating fraudster detection into recommendation to achieve adversarial robustness.



## **16. Modeling Resources in Permissionless Longest-chain Total-order Broadcast**

cs.CR

**SubmitDate**: 2022-11-22    [abs](http://arxiv.org/abs/2211.12050v1) [paper-pdf](http://arxiv.org/pdf/2211.12050v1)

**Authors**: Sarah Azouvi, Christian Cachin, Duc V. Le, Marko Vukolic, Luca Zanolini

**Abstract**: Blockchain protocols implement total-order broadcast in a permissionless setting, where processes can freely join and leave. In such a setting, to safeguard against Sybil attacks, correct processes rely on cryptographic proofs tied to a particular type of resource to make them eligible to order transactions. For example, in the case of Proof-of-Work (PoW), this resource is computation, and the proof is a solution to a computationally hard puzzle. Conversely, in Proof-of-Stake (PoS), the resource corresponds to the number of coins that every process in the system owns, and a secure lottery selects a process for participation proportionally to its coin holdings.   Although many resource-based blockchain protocols are formally proven secure in the literature, the existing security proofs fail to demonstrate why particular types of resources cause the blockchain protocols to be vulnerable to distinct classes of attacks. For instance, PoS systems are more vulnerable to long-range attacks, where an adversary corrupts past processes to re-write the history, than Proof-of-Work and Proof-of-Storage systems. Proof-of-Storage-based and Proof-of-Stake-based protocols are both more susceptible to private double-spending attacks than Proof-of-Work-based protocols; in this case, an adversary mines its chain in secret without sharing its blocks with the rest of the processes until the end of the attack.   In this paper, we formally characterize the properties of resources through an abstraction called resource allocator and give a framework for understanding longest-chain consensus protocols based on different underlying resources. In addition, we use this resource allocator to demonstrate security trade-offs between various resources focusing on well-known attacks (e.g., the long-range attack and nothing-at-stake attacks).



## **17. QueryNet: Attack by Multi-Identity Surrogates**

cs.LG

QueryNet reduces queries by about an order of magnitude against SOTA  black-box attacks

**SubmitDate**: 2022-11-22    [abs](http://arxiv.org/abs/2105.15010v4) [paper-pdf](http://arxiv.org/pdf/2105.15010v4)

**Authors**: Sizhe Chen, Zhehao Huang, Qinghua Tao, Xiaolin Huang

**Abstract**: Deep Neural Networks (DNNs) are acknowledged as vulnerable to adversarial attacks, while the existing black-box attacks require extensive queries on the victim DNN to achieve high success rates. For query-efficiency, surrogate models of the victim are used to generate transferable Adversarial Examples (AEs) because of their Gradient Similarity (GS), i.e., surrogates' attack gradients are similar to the victim's ones. However, it is generally neglected to exploit their similarity on outputs, namely the Prediction Similarity (PS), to filter out inefficient queries by surrogates without querying the victim. To jointly utilize and also optimize surrogates' GS and PS, we develop QueryNet, a unified attack framework that can significantly reduce queries. QueryNet creatively attacks by multi-identity surrogates, i.e., crafts several AEs for one sample by different surrogates, and also uses surrogates to decide on the most promising AE for the query. After that, the victim's query feedback is accumulated to optimize not only surrogates' parameters but also their architectures, enhancing both the GS and the PS. Although QueryNet has no access to pre-trained surrogates' prior, it reduces queries by averagely about an order of magnitude compared to alternatives within an acceptable time, according to our comprehensive experiments: 11 victims (including two commercial models) on MNIST/CIFAR10/ImageNet, allowing only 8-bit image queries, and no access to the victim's training data. The code is available at https://github.com/Sizhe-Chen/QueryNet.



## **18. Addressing Mistake Severity in Neural Networks with Semantic Knowledge**

cs.LG

**SubmitDate**: 2022-11-21    [abs](http://arxiv.org/abs/2211.11880v1) [paper-pdf](http://arxiv.org/pdf/2211.11880v1)

**Authors**: Natalie Abreu, Nathan Vaska, Victoria Helus

**Abstract**: Robustness in deep neural networks and machine learning algorithms in general is an open research challenge. In particular, it is difficult to ensure algorithmic performance is maintained on out-of-distribution inputs or anomalous instances that cannot be anticipated at training time. Embodied agents will be deployed in these conditions, and are likely to make incorrect predictions. An agent will be viewed as untrustworthy unless it can maintain its performance in dynamic environments. Most robust training techniques aim to improve model accuracy on perturbed inputs; as an alternate form of robustness, we aim to reduce the severity of mistakes made by neural networks in challenging conditions. We leverage current adversarial training methods to generate targeted adversarial attacks during the training process in order to increase the semantic similarity between a model's predictions and true labels of misclassified instances. Results demonstrate that our approach performs better with respect to mistake severity compared to standard and adversarially trained models. We also find an intriguing role that non-robust features play with regards to semantic similarity.



## **19. Voice Spoofing Countermeasures: Taxonomy, State-of-the-art, experimental analysis of generalizability, open challenges, and the way forward**

eess.AS

**SubmitDate**: 2022-11-21    [abs](http://arxiv.org/abs/2210.00417v2) [paper-pdf](http://arxiv.org/pdf/2210.00417v2)

**Authors**: Awais Khan, Khalid Mahmood Malik, James Ryan, Mikul Saravanan

**Abstract**: Malicious actors may seek to use different voice-spoofing attacks to fool ASV systems and even use them for spreading misinformation. Various countermeasures have been proposed to detect these spoofing attacks. Due to the extensive work done on spoofing detection in automated speaker verification (ASV) systems in the last 6-7 years, there is a need to classify the research and perform qualitative and quantitative comparisons on state-of-the-art countermeasures. Additionally, no existing survey paper has reviewed integrated solutions to voice spoofing evaluation and speaker verification, adversarial/antiforensics attacks on spoofing countermeasures, and ASV itself, or unified solutions to detect multiple attacks using a single model. Further, no work has been done to provide an apples-to-apples comparison of published countermeasures in order to assess their generalizability by evaluating them across corpora. In this work, we conduct a review of the literature on spoofing detection using hand-crafted features, deep learning, end-to-end, and universal spoofing countermeasure solutions to detect speech synthesis (SS), voice conversion (VC), and replay attacks. Additionally, we also review integrated solutions to voice spoofing evaluation and speaker verification, adversarial and anti-forensics attacks on voice countermeasures, and ASV. The limitations and challenges of the existing spoofing countermeasures are also presented. We report the performance of these countermeasures on several datasets and evaluate them across corpora. For the experiments, we employ the ASVspoof2019 and VSDC datasets along with GMM, SVM, CNN, and CNN-GRU classifiers. (For reproduceability of the results, the code of the test bed can be found in our GitHub Repository.



## **20. Backdoor Attacks on Multiagent Collaborative Systems**

cs.MA

11 pages

**SubmitDate**: 2022-11-21    [abs](http://arxiv.org/abs/2211.11455v1) [paper-pdf](http://arxiv.org/pdf/2211.11455v1)

**Authors**: Shuo Chen, Yue Qiu, Jie Zhang

**Abstract**: Backdoor attacks on reinforcement learning implant a backdoor in a victim agent's policy. Once the victim observes the trigger signal, it will switch to the abnormal mode and fail its task. Most of the attacks assume the adversary can arbitrarily modify the victim's observations, which may not be practical. One work proposes to let one adversary agent use its actions to affect its opponent in two-agent competitive games, so that the opponent quickly fails after observing certain trigger actions. However, in multiagent collaborative systems, agents may not always be able to observe others. When and how much the adversary agent can affect others are uncertain, and we want the adversary agent to trigger others for as few times as possible. To solve this problem, we first design a novel training framework to produce auxiliary rewards that measure the extent to which the other agents'observations being affected. Then we use the auxiliary rewards to train a trigger policy which enables the adversary agent to efficiently affect the others' observations. Given these affected observations, we further train the other agents to perform abnormally. Extensive experiments demonstrate that the proposed method enables the adversary agent to lure the others into the abnormal mode with only a few actions.



## **21. SPIN: Simulated Poisoning and Inversion Network for Federated Learning-Based 6G Vehicular Networks**

cs.LG

6 pages, 4 figures

**SubmitDate**: 2022-11-21    [abs](http://arxiv.org/abs/2211.11321v1) [paper-pdf](http://arxiv.org/pdf/2211.11321v1)

**Authors**: Sunder Ali Khowaja, Parus Khuwaja, Kapal Dev, Angelos Antonopoulos

**Abstract**: The applications concerning vehicular networks benefit from the vision of beyond 5G and 6G technologies such as ultra-dense network topologies, low latency, and high data rates. Vehicular networks have always faced data privacy preservation concerns, which lead to the advent of distributed learning techniques such as federated learning. Although federated learning has solved data privacy preservation issues to some extent, the technique is quite vulnerable to model inversion and model poisoning attacks. We assume that the design of defense mechanism and attacks are two sides of the same coin. Designing a method to reduce vulnerability requires the attack to be effective and challenging with real-world implications. In this work, we propose simulated poisoning and inversion network (SPIN) that leverages the optimization approach for reconstructing data from a differential model trained by a vehicular node and intercepted when transmitted to roadside unit (RSU). We then train a generative adversarial network (GAN) to improve the generation of data with each passing round and global update from the RSU, accordingly. Evaluation results show the qualitative and quantitative effectiveness of the proposed approach. The attack initiated by SPIN can reduce up to 22% accuracy on publicly available datasets while just using a single attacker. We assume that revealing the simulation of such attacks would help us find its defense mechanism in an effective manner.



## **22. Understanding the Vulnerability of Skeleton-based Human Activity Recognition via Black-box Attack**

cs.CV

arXiv admin note: substantial text overlap with arXiv:2103.05266

**SubmitDate**: 2022-11-21    [abs](http://arxiv.org/abs/2211.11312v1) [paper-pdf](http://arxiv.org/pdf/2211.11312v1)

**Authors**: Yunfeng Diao, He Wang, Tianjia Shao, Yong-Liang Yang, Kun Zhou, David Hogg

**Abstract**: Human Activity Recognition (HAR) has been employed in a wide range of applications, e.g. self-driving cars, where safety and lives are at stake. Recently, the robustness of existing skeleton-based HAR methods has been questioned due to their vulnerability to adversarial attacks, which causes concerns considering the scale of the implication. However, the proposed attacks require the full-knowledge of the attacked classifier, which is overly restrictive. In this paper, we show such threats indeed exist, even when the attacker only has access to the input/output of the model. To this end, we propose the very first black-box adversarial attack approach in skeleton-based HAR called BASAR. BASAR explores the interplay between the classification boundary and the natural motion manifold. To our best knowledge, this is the first time data manifold is introduced in adversarial attacks on time series. Via BASAR, we find on-manifold adversarial samples are extremely deceitful and rather common in skeletal motions, in contrast to the common belief that adversarial samples only exist off-manifold. Through exhaustive evaluation, we show that BASAR can deliver successful attacks across classifiers, datasets, and attack modes. By attack, BASAR helps identify the potential causes of the model vulnerability and provides insights on possible improvements. Finally, to mitigate the newly identified threat, we propose a new adversarial training approach by leveraging the sophisticated distributions of on/off-manifold adversarial samples, called mixed manifold-based adversarial training (MMAT). MMAT can successfully help defend against adversarial attacks without compromising classification accuracy.



## **23. Boosting the Transferability of Adversarial Attacks with Global Momentum Initialization**

cs.CV

**SubmitDate**: 2022-11-21    [abs](http://arxiv.org/abs/2211.11236v1) [paper-pdf](http://arxiv.org/pdf/2211.11236v1)

**Authors**: Jiafeng Wang, Zhaoyu Chen, Kaixun Jiang, Dingkang Yang, Lingyi Hong, Yan Wang, Wenqiang Zhang

**Abstract**: Deep neural networks are vulnerable to adversarial examples, which attach human invisible perturbations to benign inputs. Simultaneously, adversarial examples exhibit transferability under different models, which makes practical black-box attacks feasible. However, existing methods are still incapable of achieving desired transfer attack performance. In this work, from the perspective of gradient optimization and consistency, we analyze and discover the gradient elimination phenomenon as well as the local momentum optimum dilemma. To tackle these issues, we propose Global Momentum Initialization (GI) to suppress gradient elimination and help search for the global optimum. Specifically, we perform gradient pre-convergence before the attack and carry out a global search during the pre-convergence stage. Our method can be easily combined with almost all existing transfer methods, and we improve the success rate of transfer attacks significantly by an average of 6.4% under various advanced defense mechanisms compared to state-of-the-art methods. Eventually, we achieve an attack success rate of 95.4%, fully illustrating the insecurity of existing defense mechanisms.



## **24. Deep Composite Face Image Attacks: Generation, Vulnerability and Detection**

cs.CV

**SubmitDate**: 2022-11-20    [abs](http://arxiv.org/abs/2211.11039v1) [paper-pdf](http://arxiv.org/pdf/2211.11039v1)

**Authors**: Jag Mohan Singh, Raghavendra Ramachandra

**Abstract**: Face manipulation attacks have drawn the attention of biometric researchers because of their vulnerability to Face Recognition Systems (FRS). This paper proposes a novel scheme to generate Composite Face Image Attacks (CFIA) based on the Generative Adversarial Networks (GANs). Given the face images from contributory data subjects, the proposed CFIA method will independently generate the segmented facial attributes, then blend them using transparent masks to generate the CFIA samples. { The primary motivation for CFIA is to utilize deep learning to generate facial attribute-based composite attacks, which has been explored relatively less in the current literature.} We generate $14$ different combinations of facial attributes resulting in $14$ unique CFIA samples for each pair of contributory data subjects. Extensive experiments are carried out on our newly generated CFIA dataset consisting of 1000 unique identities with 2000 bona fide samples and 14000 CFIA samples, thus resulting in an overall 16000 face image samples. We perform a sequence of experiments to benchmark the vulnerability of CFIA to automatic FRS (based on both deep-learning and commercial-off-the-shelf (COTS). We introduced a new metric named Generalized Morphing Attack Potential (GMAP) to benchmark the vulnerability effectively. Additional experiments are performed to compute the perceptual quality of the generated CFIA samples. Finally, the CFIA detection performance is presented using three different Face Morphing Attack Detection (MAD) algorithms. The proposed CFIA method indicates good perceptual quality based on the obtained results. Further, { FRS is vulnerable to CFIA} (much higher than SOTA), making it difficult to detect by human observers and automatic detection algorithms. Lastly, we performed experiments to detect the CFIA samples using three different detection techniques automatically.



## **25. Adversarial Cheap Talk**

cs.LG

**SubmitDate**: 2022-11-20    [abs](http://arxiv.org/abs/2211.11030v1) [paper-pdf](http://arxiv.org/pdf/2211.11030v1)

**Authors**: Chris Lu, Timon Willi, Alistair Letcher, Jakob Foerster

**Abstract**: Adversarial attacks in reinforcement learning (RL) often assume highly-privileged access to the victim's parameters, environment, or data. Instead, this paper proposes a novel adversarial setting called a Cheap Talk MDP in which an Adversary can merely append deterministic messages to the Victim's observation, resulting in a minimal range of influence. The Adversary cannot occlude ground truth, influence underlying environment dynamics or reward signals, introduce non-stationarity, add stochasticity, see the Victim's actions, or access their parameters. Additionally, we present a simple meta-learning algorithm called Adversarial Cheap Talk (ACT) to train Adversaries in this setting. We demonstrate that an Adversary trained with ACT can still significantly influence the Victim's training and testing performance, despite the highly constrained setting. Affecting train-time performance reveals a new attack vector and provides insight into the success and failure modes of existing RL algorithms. More specifically, we show that an ACT Adversary is capable of harming performance by interfering with the learner's function approximation, or instead helping the Victim's performance by outputting useful features. Finally, we show that an ACT Adversary can manipulate messages during train-time to directly and arbitrarily control the Victim at test-time.



## **26. Towards Robust Neural Networks via Orthogonal Diversity**

cs.CV

**SubmitDate**: 2022-11-20    [abs](http://arxiv.org/abs/2010.12190v4) [paper-pdf](http://arxiv.org/pdf/2010.12190v4)

**Authors**: Kun Fang, Qinghua Tao, Yingwen Wu, Tao Li, Jia Cai, Feipeng Cai, Xiaolin Huang, Jie Yang

**Abstract**: Deep Neural Networks (DNNs) are vulnerable to invisible perturbations on the images generated by adversarial attacks, which raises researches on the adversarial robustness of DNNs. A series of methods represented by the adversarial training and its variants have proven as one of the most effective techniques in enhancing the DNN robustness. Generally, adversarial training focuses on enriching the training data by involving perturbed data. Despite of the efficiency in defending specific attacks, adversarial training is benefited from the data augmentation, which does not contribute to the robustness of DNN itself and usually suffers from accuracy drop on clean data as well as inefficiency in unknown attacks. Towards the robustness of DNN itself, we propose a novel defense that aims at augmenting the model in order to learn features adaptive to diverse inputs, including adversarial examples. Specifically, we introduce multiple paths to augment the network, and impose orthogonality constraints on these paths. In addition, a margin-maximization loss is designed to further boost DIversity via Orthogonality (DIO). Extensive empirical results on various data sets, architectures, and attacks demonstrate the adversarial robustness of the proposed DIO.



## **27. Invisible Backdoor Attack with Dynamic Triggers against Person Re-identification**

cs.CV

**SubmitDate**: 2022-11-20    [abs](http://arxiv.org/abs/2211.10933v1) [paper-pdf](http://arxiv.org/pdf/2211.10933v1)

**Authors**: Wenli Sun, Xinyang Jiang, Shuguang Dou, Dongsheng Li, Duoqian Miao, Cheng Deng, Cairong Zhao

**Abstract**: In recent years, person Re-identification (ReID) has rapidly progressed with wide real-world applications, but also poses significant risks of adversarial attacks. In this paper, we focus on the backdoor attack on deep ReID models. Existing backdoor attack methods follow an all-to-one/all attack scenario, where all the target classes in the test set have already been seen in the training set. However, ReID is a much more complex fine-grained open-set recognition problem, where the identities in the test set are not contained in the training set. Thus, previous backdoor attack methods for classification are not applicable for ReID. To ameliorate this issue, we propose a novel backdoor attack on deep ReID under a new all-to-unknown scenario, called Dynamic Triggers Invisible Backdoor Attack (DT-IBA). Instead of learning fixed triggers for the target classes from the training set, DT-IBA can dynamically generate new triggers for any unknown identities. Specifically, an identity hashing network is proposed to first extract target identity information from a reference image, which is then injected into the benign images by image steganography. We extensively validate the effectiveness and stealthiness of the proposed attack on benchmark datasets, and evaluate the effectiveness of several defense methods against our attack.



## **28. Spectral Adversarial Training for Robust Graph Neural Network**

cs.LG

Accepted by TKDE. Code availiable at  https://github.com/EdisonLeeeee/SAT

**SubmitDate**: 2022-11-20    [abs](http://arxiv.org/abs/2211.10896v1) [paper-pdf](http://arxiv.org/pdf/2211.10896v1)

**Authors**: Jintang Li, Jiaying Peng, Liang Chen, Zibin Zheng, Tingting Liang, Qing Ling

**Abstract**: Recent studies demonstrate that Graph Neural Networks (GNNs) are vulnerable to slight but adversarially designed perturbations, known as adversarial examples. To address this issue, robust training methods against adversarial examples have received considerable attention in the literature. \emph{Adversarial Training (AT)} is a successful approach to learning a robust model using adversarially perturbed training samples. Existing AT methods on GNNs typically construct adversarial perturbations in terms of graph structures or node features. However, they are less effective and fraught with challenges on graph data due to the discreteness of graph structure and the relationships between connected examples. In this work, we seek to address these challenges and propose Spectral Adversarial Training (SAT), a simple yet effective adversarial training approach for GNNs. SAT first adopts a low-rank approximation of the graph structure based on spectral decomposition, and then constructs adversarial perturbations in the spectral domain rather than directly manipulating the original graph structure. To investigate its effectiveness, we employ SAT on three widely used GNNs. Experimental results on four public graph datasets demonstrate that SAT significantly improves the robustness of GNNs against adversarial attacks without sacrificing classification accuracy and training efficiency.



## **29. Towards Compositional Adversarial Robustness: Generalizing Adversarial Training to Composite Semantic Perturbations**

cs.CV

**SubmitDate**: 2022-11-20    [abs](http://arxiv.org/abs/2202.04235v2) [paper-pdf](http://arxiv.org/pdf/2202.04235v2)

**Authors**: Lei Hsiung, Yun-Yun Tsai, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Model robustness against adversarial examples of single perturbation type such as the $\ell_{p}$-norm has been widely studied, yet its generalization to more realistic scenarios involving multiple semantic perturbations and their composition remains largely unexplored. In this paper, we first propose a novel method for generating composite adversarial examples. Our method can find the optimal attack composition by utilizing component-wise projected gradient descent and automatic attack-order scheduling. We then propose generalized adversarial training (GAT) to extend model robustness from $\ell_{p}$-ball to composite semantic perturbations, such as the combination of Hue, Saturation, Brightness, Contrast, and Rotation. Results obtained using ImageNet and CIFAR-10 datasets indicate that GAT can be robust not only to all the tested types of a single attack, but also to any combination of such attacks. GAT also outperforms baseline $\ell_{\infty}$-norm bounded adversarial training approaches by a significant margin.



## **30. Let Graph be the Go Board: Gradient-free Node Injection Attack for Graph Neural Networks via Reinforcement Learning**

cs.LG

AAAI 2023. arXiv admin note: substantial text overlap with  arXiv:2202.09389

**SubmitDate**: 2022-11-19    [abs](http://arxiv.org/abs/2211.10782v1) [paper-pdf](http://arxiv.org/pdf/2211.10782v1)

**Authors**: Mingxuan Ju, Yujie Fan, Chuxu Zhang, Yanfang Ye

**Abstract**: Graph Neural Networks (GNNs) have drawn significant attentions over the years and been broadly applied to essential applications requiring solid robustness or vigorous security standards, such as product recommendation and user behavior modeling. Under these scenarios, exploiting GNN's vulnerabilities and further downgrading its performance become extremely incentive for adversaries. Previous attackers mainly focus on structural perturbations or node injections to the existing graphs, guided by gradients from the surrogate models. Although they deliver promising results, several limitations still exist. For the structural perturbation attack, to launch a proposed attack, adversaries need to manipulate the existing graph topology, which is impractical in most circumstances. Whereas for the node injection attack, though being more practical, current approaches require training surrogate models to simulate a white-box setting, which results in significant performance downgrade when the surrogate architecture diverges from the actual victim model. To bridge these gaps, in this paper, we study the problem of black-box node injection attack, without training a potentially misleading surrogate model. Specifically, we model the node injection attack as a Markov decision process and propose Gradient-free Graph Advantage Actor Critic, namely G2A2C, a reinforcement learning framework in the fashion of advantage actor critic. By directly querying the victim model, G2A2C learns to inject highly malicious nodes with extremely limited attacking budgets, while maintaining a similar node feature distribution. Through our comprehensive experiments over eight acknowledged benchmark datasets with different characteristics, we demonstrate the superior performance of our proposed G2A2C over the existing state-of-the-art attackers. Source code is publicly available at: https://github.com/jumxglhf/G2A2C}.



## **31. Robust Smart Home Face Recognition under Starving Federated Data**

cs.LG

11 pages, 12 figures, 7 tables, accepted as a conference paper at  IEEE UV 2022, Boston, USA

**SubmitDate**: 2022-11-19    [abs](http://arxiv.org/abs/2211.05410v2) [paper-pdf](http://arxiv.org/pdf/2211.05410v2)

**Authors**: Jaechul Roh, Yajun Fang

**Abstract**: Over the past few years, the field of adversarial attack received numerous attention from various researchers with the help of successful attack success rate against well-known deep neural networks that were acknowledged to achieve high classification ability in various tasks. However, majority of the experiments were completed under a single model, which we believe it may not be an ideal case in a real-life situation. In this paper, we introduce a novel federated adversarial training method for smart home face recognition, named FLATS, where we observed some interesting findings that may not be easily noticed in a traditional adversarial attack to federated learning experiments. By applying different variations to the hyperparameters, we have spotted that our method can make the global model to be robust given a starving federated environment. Our code can be found on https://github.com/jcroh0508/FLATS.



## **32. A privacy-preserving data storage and service framework based on deep learning and blockchain for construction workers' wearable IoT sensors**

cs.CR

**SubmitDate**: 2022-11-19    [abs](http://arxiv.org/abs/2211.10713v1) [paper-pdf](http://arxiv.org/pdf/2211.10713v1)

**Authors**: Xiaoshan Zhou, Pin-Chao Liao

**Abstract**: Classifying brain signals collected by wearable Internet of Things (IoT) sensors, especially brain-computer interfaces (BCIs), is one of the fastest-growing areas of research. However, research has mostly ignored the secure storage and privacy protection issues of collected personal neurophysiological data. Therefore, in this article, we try to bridge this gap and propose a secure privacy-preserving protocol for implementing BCI applications. We first transformed brain signals into images and used generative adversarial network to generate synthetic signals to protect data privacy. Subsequently, we applied the paradigm of transfer learning for signal classification. The proposed method was evaluated by a case study and results indicate that real electroencephalogram data augmented with artificially generated samples provide superior classification performance. In addition, we proposed a blockchain-based scheme and developed a prototype on Ethereum, which aims to make storing, querying and sharing personal neurophysiological data and analysis reports secure and privacy-aware. The rights of three main transaction bodies - construction workers, BCI service providers and project managers - are described and the advantages of the proposed system are discussed. We believe this paper provides a well-rounded solution to safeguard private data against cyber-attacks, level the playing field for BCI application developers, and to the end improve professional well-being in the industry.



## **33. A Survey on Differential Privacy with Machine Learning and Future Outlook**

cs.LG

12 pages, 3 figures

**SubmitDate**: 2022-11-19    [abs](http://arxiv.org/abs/2211.10708v1) [paper-pdf](http://arxiv.org/pdf/2211.10708v1)

**Authors**: Samah Baraheem, Zhongmei Yao

**Abstract**: Nowadays, machine learning models and applications have become increasingly pervasive. With this rapid increase in the development and employment of machine learning models, a concern regarding privacy has risen. Thus, there is a legitimate need to protect the data from leaking and from any attacks. One of the strongest and most prevalent privacy models that can be used to protect machine learning models from any attacks and vulnerabilities is differential privacy (DP). DP is strict and rigid definition of privacy, where it can guarantee that an adversary is not capable to reliably predict if a specific participant is included in the dataset or not. It works by injecting a noise to the data whether to the inputs, the outputs, the ground truth labels, the objective functions, or even to the gradients to alleviate the privacy issue and protect the data. To this end, this survey paper presents different differentially private machine learning algorithms categorized into two main categories (traditional machine learning models vs. deep learning models). Moreover, future research directions for differential privacy with machine learning algorithms are outlined.



## **34. Phonemic Adversarial Attack against Audio Recognition in Real World**

cs.SD

**SubmitDate**: 2022-11-19    [abs](http://arxiv.org/abs/2211.10661v1) [paper-pdf](http://arxiv.org/pdf/2211.10661v1)

**Authors**: Jiakai Wang, Zhendong Chen, Zixin Yin, Qinghong Yang, Xianglong Liu

**Abstract**: Recently, adversarial attacks for audio recognition have attracted much attention. However, most of the existing studies mainly rely on the coarse-grain audio features at the instance level to generate adversarial noises, which leads to expensive generation time costs and weak universal attacking ability. Motivated by the observations that all audio speech consists of fundamental phonemes, this paper proposes a phonemic adversarial tack (PAT) paradigm, which attacks the fine-grain audio features at the phoneme level commonly shared across audio instances, to generate phonemic adversarial noises, enjoying the more general attacking ability with fast generation speed. Specifically, for accelerating the generation, a phoneme density balanced sampling strategy is introduced to sample quantity less but phonemic features abundant audio instances as the training data via estimating the phoneme density, which substantially alleviates the heavy dependency on the large training dataset. Moreover, for promoting universal attacking ability, the phonemic noise is optimized in an asynchronous way with a sliding window, which enhances the phoneme diversity and thus well captures the critical fundamental phonemic patterns. By conducting extensive experiments, we comprehensively investigate the proposed PAT framework and demonstrate that it outperforms the SOTA baselines by large margins (i.e., at least 11X speed up and 78% attacking ability improvement).



## **35. Scale-free and Task-agnostic Attack: Generating Photo-realistic Adversarial Patterns with Patch Quilting Generator**

cs.CV

**SubmitDate**: 2022-11-19    [abs](http://arxiv.org/abs/2208.06222v2) [paper-pdf](http://arxiv.org/pdf/2208.06222v2)

**Authors**: Xiangbo Gao, Cheng Luo, Qinliang Lin, Weicheng Xie, Minmin Liu, Linlin Shen, Keerthy Kusumam, Siyang Song

**Abstract**: \noindent Traditional L_p norm-restricted image attack algorithms suffer from poor transferability to black box scenarios and poor robustness to defense algorithms. Recent CNN generator-based attack approaches can synthesize unrestricted and semantically meaningful entities to the image, which is shown to be transferable and robust. However, such methods attack images by either synthesizing local adversarial entities, which are only suitable for attacking specific contents or performing global attacks, which are only applicable to a specific image scale. In this paper, we propose a novel Patch Quilting Generative Adversarial Networks (PQ-GAN) to learn the first scale-free CNN generator that can be applied to attack images with arbitrary scales for various computer vision tasks. The principal investigation on transferability of the generated adversarial examples, robustness to defense frameworks, and visual quality assessment show that the proposed PQG-based attack framework outperforms the other nine state-of-the-art adversarial attack approaches when attacking the neural networks trained on two standard evaluation datasets (i.e., ImageNet and CityScapes).



## **36. Investigating the Security of EV Charging Mobile Applications As an Attack Surface**

cs.CR

**SubmitDate**: 2022-11-19    [abs](http://arxiv.org/abs/2211.10603v1) [paper-pdf](http://arxiv.org/pdf/2211.10603v1)

**Authors**: K. Sarieddine, M. A. Sayed, S. Torabi, R. Atallah, C. Assi

**Abstract**: The adoption rate of EVs has witnessed a significant increase in recent years driven by multiple factors, chief among which is the increased flexibility and ease of access to charging infrastructure. To improve user experience, increase system flexibility and commercialize the charging process, mobile applications have been incorporated into the EV charging ecosystem. EV charging mobile applications allow consumers to remotely trigger actions on charging stations and use functionalities such as start/stop charging sessions, pay for usage, and locate charging stations, to name a few. In this paper, we study the security posture of the EV charging ecosystem against remote attacks, which exploit the insecurity of the EV charging mobile applications as an attack surface. We leverage a combination of static and dynamic analysis techniques to analyze the security of widely used EV charging mobile applications. Our analysis of 31 widely used mobile applications and their interactions with various components such as the cloud management systems indicate the lack of user/vehicle verification and improper authorization for critical functions, which lead to remote (dis)charging session hijacking and Denial of Service (DoS) attacks against the EV charging station. Indeed, we discuss specific remote attack scenarios and their impact on the EV users. More importantly, our analysis results demonstrate the feasibility of leveraging existing vulnerabilities across various EV charging mobile applications to perform wide-scale coordinated remote charging/discharging attacks against the connected critical infrastructure (e.g., power grid), with significant undesired economical and operational implications. Finally, we propose counter measures to secure the infrastructure and impede adversaries from performing reconnaissance and launching remote attacks using compromised accounts.



## **37. Person Text-Image Matching via Text-Feature Interpretability Embedding and External Attack Node Implantation**

cs.CV

**SubmitDate**: 2022-11-19    [abs](http://arxiv.org/abs/2211.08657v2) [paper-pdf](http://arxiv.org/pdf/2211.08657v2)

**Authors**: Fan Li, Hang Zhou, Huafeng Li, Yafei Zhang, Zhengtao Yu

**Abstract**: Person text-image matching, also known as text based person search, aims to retrieve images of specific pedestrians using text descriptions. Although person text-image matching has made great research progress, existing methods still face two challenges. First, the lack of interpretability of text features makes it challenging to effectively align them with their corresponding image features. Second, the same pedestrian image often corresponds to multiple different text descriptions, and a single text description can correspond to multiple different images of the same identity. The diversity of text descriptions and images makes it difficult for a network to extract robust features that match the two modalities. To address these problems, we propose a person text-image matching method by embedding text-feature interpretability and an external attack node. Specifically, we improve the interpretability of text features by providing them with consistent semantic information with image features to achieve the alignment of text and describe image region features.To address the challenges posed by the diversity of text and the corresponding person images, we treat the variation caused by diversity to features as caused by perturbation information and propose a novel adversarial attack and defense method to solve it. In the model design, graph convolution is used as the basic framework for feature representation and the adversarial attacks caused by text and image diversity on feature extraction is simulated by implanting an additional attack node in the graph convolution layer to improve the robustness of the model against text and image diversity. Extensive experiments demonstrate the effectiveness and superiority of text-pedestrian image matching over existing methods. The source code of the method is published at



## **38. Adversarial Detection by Approximation of Ensemble Boundary**

cs.LG

8 pages, 8 figures, 8 tables

**SubmitDate**: 2022-11-18    [abs](http://arxiv.org/abs/2211.10227v1) [paper-pdf](http://arxiv.org/pdf/2211.10227v1)

**Authors**: T. Windeatt

**Abstract**: A spectral approximation of a Boolean function is proposed for approximating the decision boundary of an ensemble of Deep Neural Networks (DNNs) solving two-class pattern recognition problems. The Walsh combination of relatively weak DNN classifiers is shown experimentally to be capable of detecting adversarial attacks. By observing the difference in Walsh coefficient approximation between clean and adversarial images, it appears that transferability of attack may be used for detection. Approximating the decision boundary may also aid in understanding the learning and transferability properties of DNNs. While the experiments here use images, the proposed approach of modelling two-class ensemble decision boundaries could in principle be applied to any application area.



## **39. Adv-Attribute: Inconspicuous and Transferable Adversarial Attack on Face Recognition**

cs.CV

Accepted by NeurIPS2022

**SubmitDate**: 2022-11-18    [abs](http://arxiv.org/abs/2210.06871v2) [paper-pdf](http://arxiv.org/pdf/2210.06871v2)

**Authors**: Shuai Jia, Bangjie Yin, Taiping Yao, Shouhong Ding, Chunhua Shen, Xiaokang Yang, Chao Ma

**Abstract**: Deep learning models have shown their vulnerability when dealing with adversarial attacks. Existing attacks almost perform on low-level instances, such as pixels and super-pixels, and rarely exploit semantic clues. For face recognition attacks, existing methods typically generate the l_p-norm perturbations on pixels, however, resulting in low attack transferability and high vulnerability to denoising defense models. In this work, instead of performing perturbations on the low-level pixels, we propose to generate attacks through perturbing on the high-level semantics to improve attack transferability. Specifically, a unified flexible framework, Adversarial Attributes (Adv-Attribute), is designed to generate inconspicuous and transferable attacks on face recognition, which crafts the adversarial noise and adds it into different attributes based on the guidance of the difference in face recognition features from the target. Moreover, the importance-aware attribute selection and the multi-objective optimization strategy are introduced to further ensure the balance of stealthiness and attacking strength. Extensive experiments on the FFHQ and CelebA-HQ datasets show that the proposed Adv-Attribute method achieves the state-of-the-art attacking success rates while maintaining better visual effects against recent attack methods.



## **40. Leveraging Algorithmic Fairness to Mitigate Blackbox Attribute Inference Attacks**

cs.LG

arXiv admin note: text overlap with arXiv:2202.02242

**SubmitDate**: 2022-11-18    [abs](http://arxiv.org/abs/2211.10209v1) [paper-pdf](http://arxiv.org/pdf/2211.10209v1)

**Authors**: Jan Aalmoes, Vasisht Duddu, Antoine Boutet

**Abstract**: Machine learning (ML) models have been deployed for high-stakes applications, e.g., healthcare and criminal justice. Prior work has shown that ML models are vulnerable to attribute inference attacks where an adversary, with some background knowledge, trains an ML attack model to infer sensitive attributes by exploiting distinguishable model predictions. However, some prior attribute inference attacks have strong assumptions about adversary's background knowledge (e.g., marginal distribution of sensitive attribute) and pose no more privacy risk than statistical inference. Moreover, none of the prior attacks account for class imbalance of sensitive attribute in datasets coming from real-world applications (e.g., Race and Sex). In this paper, we propose an practical and effective attribute inference attack that accounts for this imbalance using an adaptive threshold over the attack model's predictions. We exhaustively evaluate our proposed attack on multiple datasets and show that the adaptive threshold over the model's predictions drastically improves the attack accuracy over prior work. Finally, current literature lacks an effective defence against attribute inference attacks. We investigate the impact of fairness constraints (i.e., designed to mitigate unfairness in model predictions) during model training on our attribute inference attack. We show that constraint based fairness algorithms which enforces equalized odds acts as an effective defense against attribute inference attacks without impacting the model utility. Hence, the objective of algorithmic fairness and sensitive attribute privacy are aligned.



## **41. Cheating Automatic Short Answer Grading: On the Adversarial Usage of Adjectives and Adverbs**

cs.CL

**SubmitDate**: 2022-11-18    [abs](http://arxiv.org/abs/2201.08318v2) [paper-pdf](http://arxiv.org/pdf/2201.08318v2)

**Authors**: Anna Filighera, Sebastian Ochs, Tim Steuer, Thomas Tregel

**Abstract**: Automatic grading models are valued for the time and effort saved during the instruction of large student bodies. Especially with the increasing digitization of education and interest in large-scale standardized testing, the popularity of automatic grading has risen to the point where commercial solutions are widely available and used. However, for short answer formats, automatic grading is challenging due to natural language ambiguity and versatility. While automatic short answer grading models are beginning to compare to human performance on some datasets, their robustness, especially to adversarially manipulated data, is questionable. Exploitable vulnerabilities in grading models can have far-reaching consequences ranging from cheating students receiving undeserved credit to undermining automatic grading altogether - even when most predictions are valid. In this paper, we devise a black-box adversarial attack tailored to the educational short answer grading scenario to investigate the grading models' robustness. In our attack, we insert adjectives and adverbs into natural places of incorrect student answers, fooling the model into predicting them as correct. We observed a loss of prediction accuracy between 10 and 22 percentage points using the state-of-the-art models BERT and T5. While our attack made answers appear less natural to humans in our experiments, it did not significantly increase the graders' suspicions of cheating. Based on our experiments, we provide recommendations for utilizing automatic grading systems more safely in practice.



## **42. Data-Adaptive Discriminative Feature Localization with Statistically Guaranteed Interpretation**

stat.ML

27 pages, 11 figures

**SubmitDate**: 2022-11-18    [abs](http://arxiv.org/abs/2211.10061v1) [paper-pdf](http://arxiv.org/pdf/2211.10061v1)

**Authors**: Ben Dai, Xiaotong Shen, Lin Yee Chen, Chunlin Li, Wei Pan

**Abstract**: In explainable artificial intelligence, discriminative feature localization is critical to reveal a blackbox model's decision-making process from raw data to prediction. In this article, we use two real datasets, the MNIST handwritten digits and MIT-BIH Electrocardiogram (ECG) signals, to motivate key characteristics of discriminative features, namely adaptiveness, predictive importance and effectiveness. Then, we develop a localization framework based on adversarial attacks to effectively localize discriminative features. In contrast to existing heuristic methods, we also provide a statistically guaranteed interpretability of the localized features by measuring a generalized partial $R^2$. We apply the proposed method to the MNIST dataset and the MIT-BIH dataset with a convolutional auto-encoder. In the first, the compact image regions localized by the proposed method are visually appealing. Similarly, in the second, the identified ECG features are biologically plausible and consistent with cardiac electrophysiological principles while locating subtle anomalies in a QRS complex that may not be discernible by the naked eye. Overall, the proposed method compares favorably with state-of-the-art competitors. Accompanying this paper is a Python library dnn-locate (https://dnn-locate.readthedocs.io/en/latest/) that implements the proposed approach.



## **43. Adversarial Stimuli: Attacking Brain-Computer Interfaces via Perturbed Sensory Events**

cs.CR

**SubmitDate**: 2022-11-18    [abs](http://arxiv.org/abs/2211.10033v1) [paper-pdf](http://arxiv.org/pdf/2211.10033v1)

**Authors**: Bibek Upadhayay, Vahid Behzadan

**Abstract**: Machine learning models are known to be vulnerable to adversarial perturbations in the input domain, causing incorrect predictions. Inspired by this phenomenon, we explore the feasibility of manipulating EEG-based Motor Imagery (MI) Brain Computer Interfaces (BCIs) via perturbations in sensory stimuli. Similar to adversarial examples, these \emph{adversarial stimuli} aim to exploit the limitations of the integrated brain-sensor-processing components of the BCI system in handling shifts in participants' response to changes in sensory stimuli. This paper proposes adversarial stimuli as an attack vector against BCIs, and reports the findings of preliminary experiments on the impact of visual adversarial stimuli on the integrity of EEG-based MI BCIs. Our findings suggest that minor adversarial stimuli can significantly deteriorate the performance of MI BCIs across all participants (p=0.0003). Additionally, our results indicate that such attacks are more effective in conditions with induced stress.



## **44. Adaptive Test-Time Defense with the Manifold Hypothesis**

cs.LG

**SubmitDate**: 2022-11-17    [abs](http://arxiv.org/abs/2210.14404v3) [paper-pdf](http://arxiv.org/pdf/2210.14404v3)

**Authors**: Zhaoyuan Yang, Zhiwei Xu, Jing Zhang, Richard Hartley, Peter Tu

**Abstract**: In this work, we formulate a novel framework of adversarial robustness using the manifold hypothesis. Our framework provides sufficient conditions for defending against adversarial examples. We develop a test-time defense method with variational inference and our formulation. The developed approach combines manifold learning with variational inference to provide adversarial robustness without the need for adversarial training. We show that our approach can provide adversarial robustness even if attackers are aware of the existence of test-time defense. In addition, our approach can also serve as a test-time defense mechanism for variational autoencoders.



## **45. UPTON: Unattributable Authorship Text via Data Poisoning**

cs.CY

**SubmitDate**: 2022-11-17    [abs](http://arxiv.org/abs/2211.09717v1) [paper-pdf](http://arxiv.org/pdf/2211.09717v1)

**Authors**: Ziyao Wang, Thai Le, Dongwon Lee

**Abstract**: In online medium such as opinion column in Bloomberg, The Guardian and Western Journal, aspiring writers post their writings for various reasons with their names often proudly open. However, it may occur that such a writer wants to write in other venues anonymously or under a pseudonym (e.g., activist, whistle-blower). However, if an attacker has already built an accurate authorship attribution (AA) model based off of the writings from such platforms, attributing an anonymous writing to the known authorship is possible. Therefore, in this work, we ask a question "can one make the writings and texts, T, in the open spaces such as opinion sharing platforms unattributable so that AA models trained from T cannot attribute authorship well?" Toward this question, we present a novel solution, UPTON, that exploits textual data poisoning method to disturb the training process of AA models. UPTON uses data poisoning to destroy the authorship feature only in training samples by perturbing them, and try to make released textual data unlearnable on deep neuron networks. It is different from previous obfuscation works, that use adversarial attack to modify the test samples and mislead an AA model, and also the backdoor works, which use trigger words both in test and training samples and only change the model output when trigger words occur. Using four authorship datasets (e.g., IMDb10, IMDb64, Enron and WJO), then, we present empirical validation where: (1)UPTON is able to downgrade the test accuracy to about 30% with carefully designed target-selection methods. (2)UPTON poisoning is able to preserve most of the original semantics. The BERTSCORE between the clean and UPTON poisoned texts are higher than 0.95. The number is very closed to 1.00, which means no sematic change. (3)UPTON is also robust towards spelling correction systems.



## **46. An efficient combination of quantum error correction and authentication**

quant-ph

30 pages, 10 figures

**SubmitDate**: 2022-11-17    [abs](http://arxiv.org/abs/2211.09686v1) [paper-pdf](http://arxiv.org/pdf/2211.09686v1)

**Authors**: Yfke Dulek, Garazi Muguruza, Florian Speelman

**Abstract**: When sending quantum information over a channel, we want to ensure that the message remains intact. Quantum error correction and quantum authentication both aim to protect (quantum) information, but approach this task from two very different directions: error-correcting codes protect against probabilistic channel noise and are meant to be very robust against small errors, while authentication codes prevent adversarial attacks and are designed to be very sensitive against any error, including small ones.   In practice, when sending an authenticated state over a noisy channel, one would have to wrap it in an error-correcting code to counterbalance the sensitivity of the underlying authentication scheme. We study the question of whether this can be done more efficiently by combining the two functionalities in a single code. To illustrate the potential of such a combination, we design the threshold code, a modification of the trap authentication code which preserves that code's authentication properties, but which is naturally robust against depolarizing channel noise. We show that the threshold code needs polylogarithmically fewer qubits to achieve the same level of security and robustness, compared to the naive composition of the trap code with any concatenated CSS code. We believe our analysis opens the door to combining more general error-correction and authentication codes, which could improve the practicality of the resulting scheme.



## **47. Towards Good Practices in Evaluating Transfer Adversarial Attacks**

cs.CR

Our code and a list of categorized attacks are publicly available at  https://github.com/ZhengyuZhao/TransferAttackEval

**SubmitDate**: 2022-11-17    [abs](http://arxiv.org/abs/2211.09565v1) [paper-pdf](http://arxiv.org/pdf/2211.09565v1)

**Authors**: Zhengyu Zhao, Hanwei Zhang, Renjue Li, Ronan Sicre, Laurent Amsaleg, Michael Backes

**Abstract**: Transfer adversarial attacks raise critical security concerns in real-world, black-box scenarios. However, the actual progress of attack methods is difficult to assess due to two main limitations in existing evaluations. First, existing evaluations are unsystematic and sometimes unfair since new methods are often directly added to old ones without complete comparisons to similar methods. Second, existing evaluations mainly focus on transferability but overlook another key attack property: stealthiness. In this work, we design good practices to address these limitations. We first introduce a new attack categorization, which enables our systematic analyses of similar attacks in each specific category. Our analyses lead to new findings that complement or even challenge existing knowledge. Furthermore, we comprehensively evaluate 23 representative attacks against 9 defenses on ImageNet. We pay particular attention to stealthiness, by adopting diverse imperceptibility metrics and looking into new, finer-grained characteristics. Our evaluation reveals new important insights: 1) Transferability is highly contextual, and some white-box defenses may give a false sense of security since they are actually vulnerable to (black-box) transfer attacks; 2) All transfer attacks are less stealthy, and their stealthiness can vary dramatically under the same $L_{\infty}$ bound.



## **48. Ignore Previous Prompt: Attack Techniques For Language Models**

cs.CL

ML Safety Workshop NeurIPS 2022

**SubmitDate**: 2022-11-17    [abs](http://arxiv.org/abs/2211.09527v1) [paper-pdf](http://arxiv.org/pdf/2211.09527v1)

**Authors**: Fábio Perez, Ian Ribeiro

**Abstract**: Transformer-based large language models (LLMs) provide a powerful foundation for natural language tasks in large-scale customer-facing applications. However, studies that explore their vulnerabilities emerging from malicious user interaction are scarce. By proposing PromptInject, a prosaic alignment framework for mask-based iterative adversarial prompt composition, we examine how GPT-3, the most widely deployed language model in production, can be easily misaligned by simple handcrafted inputs. In particular, we investigate two types of attacks -- goal hijacking and prompt leaking -- and demonstrate that even low-aptitude, but sufficiently ill-intentioned agents, can easily exploit GPT-3's stochastic nature, creating long-tail risks. The code for PromptInject is available at https://github.com/agencyenterprise/PromptInject.



## **49. Look Closer to Your Enemy: Learning to Attack via Teacher-student Mimicking**

cs.CV

13 pages, 8 figures, NDSS

**SubmitDate**: 2022-11-17    [abs](http://arxiv.org/abs/2207.13381v3) [paper-pdf](http://arxiv.org/pdf/2207.13381v3)

**Authors**: Mingjie Wang, Zhiqing Tang, Sirui Li, Dingwen Xiao

**Abstract**: This paper aims to generate realistic attack samples of person re-identification, ReID, by reading the enemy's mind (VM). In this paper, we propose a novel inconspicuous and controllable ReID attack baseline, LCYE, to generate adversarial query images. Concretely, LCYE first distills VM's knowledge via teacher-student memory mimicking in the proxy task. Then this knowledge prior acts as an explicit cipher conveying what is essential and realistic, believed by VM, for accurate adversarial misleading. Besides, benefiting from the multiple opposing task framework of LCYE, we further investigate the interpretability and generalization of ReID models from the view of the adversarial attack, including cross-domain adaption, cross-model consensus, and online learning process. Extensive experiments on four ReID benchmarks show that our method outperforms other state-of-the-art attackers with a large margin in white-box, black-box, and target attacks. Our code is now available at https://gitfront.io/r/user-3704489/mKXusqDT4ffr/LCYE/.



## **50. Phantom Sponges: Exploiting Non-Maximum Suppression to Attack Deep Object Detectors**

cs.CV

**SubmitDate**: 2022-11-17    [abs](http://arxiv.org/abs/2205.13618v3) [paper-pdf](http://arxiv.org/pdf/2205.13618v3)

**Authors**: Avishag Shapira, Alon Zolfi, Luca Demetrio, Battista Biggio, Asaf Shabtai

**Abstract**: Adversarial attacks against deep learning-based object detectors have been studied extensively in the past few years. Most of the attacks proposed have targeted the model's integrity (i.e., caused the model to make incorrect predictions), while adversarial attacks targeting the model's availability, a critical aspect in safety-critical domains such as autonomous driving, have not yet been explored by the machine learning research community. In this paper, we propose a novel attack that negatively affects the decision latency of an end-to-end object detection pipeline. We craft a universal adversarial perturbation (UAP) that targets a widely used technique integrated in many object detector pipelines -- non-maximum suppression (NMS). Our experiments demonstrate the proposed UAP's ability to increase the processing time of individual frames by adding "phantom" objects that overload the NMS algorithm while preserving the detection of the original objects which allows the attack to go undetected for a longer period of time.



