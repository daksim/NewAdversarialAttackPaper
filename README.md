# Latest Adversarial Attack Papers
**update at 2023-06-29 15:30:22**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Mitigating the Accuracy-Robustness Trade-off via Multi-Teacher Adversarial Distillation**

cs.LG

**SubmitDate**: 2023-06-28    [abs](http://arxiv.org/abs/2306.16170v1) [paper-pdf](http://arxiv.org/pdf/2306.16170v1)

**Authors**: Shiji Zhao, Xizhe Wang, Xingxing Wei

**Abstract**: Adversarial training is a practical approach for improving the robustness of deep neural networks against adversarial attacks. Although bringing reliable robustness, the performance toward clean examples is negatively affected after adversarial training, which means a trade-off exists between accuracy and robustness. Recently, some studies have tried to use knowledge distillation methods in adversarial training, achieving competitive performance in improving the robustness but the accuracy for clean samples is still limited. In this paper, to mitigate the accuracy-robustness trade-off, we introduce the Multi-Teacher Adversarial Robustness Distillation (MTARD) to guide the model's adversarial training process by applying a strong clean teacher and a strong robust teacher to handle the clean examples and adversarial examples, respectively. During the optimization process, to ensure that different teachers show similar knowledge scales, we design the Entropy-Based Balance algorithm to adjust the teacher's temperature and keep the teachers' information entropy consistent. Besides, to ensure that the student has a relatively consistent learning speed from multiple teachers, we propose the Normalization Loss Balance algorithm to adjust the learning weights of different types of knowledge. A series of experiments conducted on public datasets demonstrate that MTARD outperforms the state-of-the-art adversarial training and distillation methods against various adversarial attacks.



## **2. Distributional Modeling for Location-Aware Adversarial Patches**

cs.CV

**SubmitDate**: 2023-06-28    [abs](http://arxiv.org/abs/2306.16131v1) [paper-pdf](http://arxiv.org/pdf/2306.16131v1)

**Authors**: Xingxing Wei, Shouwei Ruan, Yinpeng Dong, Hang Su

**Abstract**: Adversarial patch is one of the important forms of performing adversarial attacks in the physical world. To improve the naturalness and aggressiveness of existing adversarial patches, location-aware patches are proposed, where the patch's location on the target object is integrated into the optimization process to perform attacks. Although it is effective, efficiently finding the optimal location for placing the patches is challenging, especially under the black-box attack settings. In this paper, we propose the Distribution-Optimized Adversarial Patch (DOPatch), a novel method that optimizes a multimodal distribution of adversarial locations instead of individual ones. DOPatch has several benefits: Firstly, we find that the locations' distributions across different models are pretty similar, and thus we can achieve efficient query-based attacks to unseen models using a distributional prior optimized on a surrogate model. Secondly, DOPatch can generate diverse adversarial samples by characterizing the distribution of adversarial locations. Thus we can improve the model's robustness to location-aware patches via carefully designed Distributional-Modeling Adversarial Training (DOP-DMAT). We evaluate DOPatch on various face recognition and image recognition tasks and demonstrate its superiority and efficiency over existing methods. We also conduct extensive ablation studies and analyses to validate the effectiveness of our method and provide insights into the distribution of adversarial locations.



## **3. Evaluating Similitude and Robustness of Deep Image Denoising Models via Adversarial Attack**

cs.CV

12 pages, 15 figures

**SubmitDate**: 2023-06-28    [abs](http://arxiv.org/abs/2306.16050v1) [paper-pdf](http://arxiv.org/pdf/2306.16050v1)

**Authors**: Jie Ning, Yao Li, Zhichang Guo

**Abstract**: Deep neural networks (DNNs) have a wide range of applications in the field of image denoising, and they are superior to traditional image denoising. However, DNNs inevitably show vulnerability, which is the weak robustness in the face of adversarial attacks. In this paper, we find some similitudes between existing deep image denoising methods, as they are consistently fooled by adversarial attacks. First, denoising-PGD is proposed which is a denoising model full adversarial method. The current mainstream non-blind denoising models (DnCNN, FFDNet, ECNDNet, BRDNet), blind denoising models (DnCNN-B, Noise2Noise, RDDCNN-B, FAN), and plug-and-play (DPIR, CurvPnP) and unfolding denoising models (DeamNet) applied to grayscale and color images can be attacked by the same set of methods. Second, since the transferability of denoising-PGD is prominent in the image denoising task, we design experiments to explore the characteristic of the latent under the transferability. We correlate transferability with similitude and conclude that the deep image denoising models have high similitude. Third, we investigate the characteristic of the adversarial space and use adversarial training to complement the vulnerability of deep image denoising to adversarial attacks on image denoising. Finally, we constrain this adversarial attack method and propose the L2-denoising-PGD image denoising adversarial attack method that maintains the Gaussian distribution. Moreover, the model-driven image denoising BM3D shows some resistance in the face of adversarial attacks.



## **4. Enrollment-stage Backdoor Attacks on Speaker Recognition Systems via Adversarial Ultrasound**

cs.SD

**SubmitDate**: 2023-06-28    [abs](http://arxiv.org/abs/2306.16022v1) [paper-pdf](http://arxiv.org/pdf/2306.16022v1)

**Authors**: Xinfeng Li, Junning Ze, Chen Yan, Yushi Cheng, Xiaoyu Ji, Wenyuan Xu

**Abstract**: Automatic Speaker Recognition Systems (SRSs) have been widely used in voice applications for personal identification and access control. A typical SRS consists of three stages, i.e., training, enrollment, and recognition. Previous work has revealed that SRSs can be bypassed by backdoor attacks at the training stage or by adversarial example attacks at the recognition stage. In this paper, we propose TUNER, a new type of backdoor attack against the enrollment stage of SRS via adversarial ultrasound modulation, which is inaudible, synchronization-free, content-independent, and black-box. Our key idea is to first inject the backdoor into the SRS with modulated ultrasound when a legitimate user initiates the enrollment, and afterward, the polluted SRS will grant access to both the legitimate user and the adversary with high confidence. Our attack faces a major challenge of unpredictable user articulation at the enrollment stage. To overcome this challenge, we generate the ultrasonic backdoor by augmenting the optimization process with random speech content, vocalizing time, and volume of the user. Furthermore, to achieve real-world robustness, we improve the ultrasonic signal over traditional methods using sparse frequency points, pre-compensation, and single-sideband (SSB) modulation. We extensively evaluate TUNER on two common datasets and seven representative SRS models. Results show that our attack can successfully bypass speaker recognition systems while remaining robust to various speakers, speech content, et



## **5. What is the Solution for State-Adversarial Multi-Agent Reinforcement Learning?**

cs.AI

Workshop on New Frontiers in Learning, Control, and Dynamical Systems  at the International Conference on Machine Learning (ICML), Honolulu, Hawaii,  USA, 2023

**SubmitDate**: 2023-06-28    [abs](http://arxiv.org/abs/2212.02705v4) [paper-pdf](http://arxiv.org/pdf/2212.02705v4)

**Authors**: Songyang Han, Sanbao Su, Sihong He, Shuo Han, Haizhao Yang, Fei Miao

**Abstract**: Various methods for Multi-Agent Reinforcement Learning (MARL) have been developed with the assumption that agents' policies are based on accurate state information. However, policies learned through Deep Reinforcement Learning (DRL) are susceptible to adversarial state perturbation attacks. In this work, we propose a State-Adversarial Markov Game (SAMG) and make the first attempt to investigate the fundamental properties of MARL under state uncertainties. Our analysis shows that the commonly used solution concepts of optimal agent policy and robust Nash equilibrium do not always exist in SAMGs. To circumvent this difficulty, we consider a new solution concept called robust agent policy, where agents aim to maximize the worst-case expected state value. We prove the existence of robust agent policy for finite state and finite action SAMGs. Additionally, we propose a Robust Multi-Agent Adversarial Actor-Critic (RMA3C) algorithm to learn robust policies for MARL agents under state uncertainties. Our experiments demonstrate that our algorithm outperforms existing methods when faced with state perturbations and greatly improves the robustness of MARL policies. Our code is public on https://songyanghan.github.io/what_is_solution/.



## **6. Boosting Adversarial Transferability with Learnable Patch-wise Masks**

cs.CV

**SubmitDate**: 2023-06-28    [abs](http://arxiv.org/abs/2306.15931v1) [paper-pdf](http://arxiv.org/pdf/2306.15931v1)

**Authors**: Xingxing Wei, Shiji Zhao

**Abstract**: Adversarial examples have raised widespread attention in security-critical applications because of their transferability across different models. Although many methods have been proposed to boost adversarial transferability, a gap still exists in the practical demand. In this paper, we argue that the model-specific discriminative regions are a key factor to cause the over-fitting to the source model, and thus reduce the transferability to the target model. For that, a patch-wise mask is utilized to prune the model-specific regions when calculating adversarial perturbations. To accurately localize these regions, we present a learnable approach to optimize the mask automatically. Specifically, we simulate the target models in our framework, and adjust the patch-wise mask according to the feedback of simulated models. To improve the efficiency, Differential Evolutionary (DE) algorithm is utilized to search for patch-wise masks for a specific image. During iterative attacks, the learned masks are applied to the image to drop out the patches related to model-specific regions, thus making the gradients more generic and improving the adversarial transferability. The proposed approach is a pre-processing method and can be integrated with existing gradient-based methods to further boost the transfer attack success rate. Extensive experiments on the ImageNet dataset demonstrate the effectiveness of our method. We incorporate the proposed approach with existing methods in the ensemble attacks and achieve an average success rate of 93.01% against seven advanced defense methods, which can effectively enhance the state-of-the-art transfer-based attack performance.



## **7. A Diamond Model Analysis on Twitter's Biggest Hack**

cs.CR

8 pages, 3 figures, 2 tables

**SubmitDate**: 2023-06-28    [abs](http://arxiv.org/abs/2306.15878v1) [paper-pdf](http://arxiv.org/pdf/2306.15878v1)

**Authors**: Chaitanya Rahalkar

**Abstract**: Cyberattacks have prominently increased over the past few years now, and have targeted actors from a wide variety of domains. Understanding the motivation, infrastructure, attack vectors, etc. behind such attacks is vital to proactively work against preventing such attacks in the future and also to analyze the economic and social impact of such attacks. In this paper, we leverage the diamond model to perform an intrusion analysis case study of the 2020 Twitter account hijacking Cyberattack. We follow this standardized incident response model to map the adversary, capability, infrastructure, and victim and perform a comprehensive analysis of the attack, and the impact posed by the attack from a Cybersecurity policy standpoint.



## **8. Condorcet Attack Against Fair Transaction Ordering**

cs.CR

**SubmitDate**: 2023-06-27    [abs](http://arxiv.org/abs/2306.15743v1) [paper-pdf](http://arxiv.org/pdf/2306.15743v1)

**Authors**: Mohammad Amin Vafadar, Majid Khabbazian

**Abstract**: We introduce the Condorcet attack, a new threat to fair transaction ordering. Specifically, the attack undermines batch-order-fairness, the strongest notion of transaction fair ordering proposed to date. The batch-order-fairness guarantees that a transaction tx is ordered before tx' if a majority of nodes in the system receive tx before tx'; the only exception (due to an impossibility result) is when tx and tx' fall into a so-called "Condorcet cycle". When this happens, tx and tx' along with other transactions within the cycle are placed in a batch, and any unfairness inside a batch is ignored. In the Condorcet attack, an adversary attempts to undermine the system's fairness by imposing Condorcet cycles to the system. In this work, we show that the adversary can indeed impose a Condorcet cycle by submitting as few as two otherwise legitimate transactions to the system. Remarkably, the adversary (e.g., a malicious client) can achieve this even when all the nodes in the system behave honestly. A notable feature of the attack is that it is capable of "trapping" transactions that do not naturally fall inside a cycle, i.e. those that are transmitted at significantly different times (with respect to the network latency). To mitigate the attack, we propose three methods based on three different complementary approaches. We show the effectiveness of the proposed mitigation methods through simulations, and explain their limitations.



## **9. Cooperation or Competition: Avoiding Player Domination for Multi-Target Robustness via Adaptive Budgets**

cs.AI

**SubmitDate**: 2023-06-27    [abs](http://arxiv.org/abs/2306.15482v1) [paper-pdf](http://arxiv.org/pdf/2306.15482v1)

**Authors**: Yimu Wang, Dinghuai Zhang, Yihan Wu, Heng Huang, Hongyang Zhang

**Abstract**: Despite incredible advances, deep learning has been shown to be susceptible to adversarial attacks. Numerous approaches have been proposed to train robust networks both empirically and certifiably. However, most of them defend against only a single type of attack, while recent work takes steps forward in defending against multiple attacks. In this paper, to understand multi-target robustness, we view this problem as a bargaining game in which different players (adversaries) negotiate to reach an agreement on a joint direction of parameter updating. We identify a phenomenon named player domination in the bargaining game, namely that the existing max-based approaches, such as MAX and MSD, do not converge. Based on our theoretical analysis, we design a novel framework that adjusts the budgets of different adversaries to avoid any player dominance. Experiments on standard benchmarks show that employing the proposed framework to the existing approaches significantly advances multi-target robustness.



## **10. Robust Proxy: Improving Adversarial Robustness by Robust Proxy Learning**

cs.CV

Accepted at IEEE Transactions on Information Forensics and Security  (TIFS)

**SubmitDate**: 2023-06-27    [abs](http://arxiv.org/abs/2306.15457v1) [paper-pdf](http://arxiv.org/pdf/2306.15457v1)

**Authors**: Hong Joo Lee, Yong Man Ro

**Abstract**: Recently, it has been widely known that deep neural networks are highly vulnerable and easily broken by adversarial attacks. To mitigate the adversarial vulnerability, many defense algorithms have been proposed. Recently, to improve adversarial robustness, many works try to enhance feature representation by imposing more direct supervision on the discriminative feature. However, existing approaches lack an understanding of learning adversarially robust feature representation. In this paper, we propose a novel training framework called Robust Proxy Learning. In the proposed method, the model explicitly learns robust feature representations with robust proxies. To this end, firstly, we demonstrate that we can generate class-representative robust features by adding class-wise robust perturbations. Then, we use the class representative features as robust proxies. With the class-wise robust features, the model explicitly learns adversarially robust features through the proposed robust proxy learning framework. Through extensive experiments, we verify that we can manually generate robust features, and our proposed learning framework could increase the robustness of the DNNs.



## **11. Advancing Adversarial Training by Injecting Booster Signal**

cs.CV

Accepted at IEEE Transactions on Neural Networks and Learning Systems

**SubmitDate**: 2023-06-27    [abs](http://arxiv.org/abs/2306.15451v1) [paper-pdf](http://arxiv.org/pdf/2306.15451v1)

**Authors**: Hong Joo Lee, Youngjoon Yu, Yong Man Ro

**Abstract**: Recent works have demonstrated that deep neural networks (DNNs) are highly vulnerable to adversarial attacks. To defend against adversarial attacks, many defense strategies have been proposed, among which adversarial training has been demonstrated to be the most effective strategy. However, it has been known that adversarial training sometimes hurts natural accuracy. Then, many works focus on optimizing model parameters to handle the problem. Different from the previous approaches, in this paper, we propose a new approach to improve the adversarial robustness by using an external signal rather than model parameters. In the proposed method, a well-optimized universal external signal called a booster signal is injected into the outside of the image which does not overlap with the original content. Then, it boosts both adversarial robustness and natural accuracy. The booster signal is optimized in parallel to model parameters step by step collaboratively. Experimental results show that the booster signal can improve both the natural and robust accuracies over the recent state-of-the-art adversarial training methods. Also, optimizing the booster signal is general and flexible enough to be adopted on any existing adversarial training methods.



## **12. Adversarial Training for Graph Neural Networks**

cs.LG

**SubmitDate**: 2023-06-27    [abs](http://arxiv.org/abs/2306.15427v1) [paper-pdf](http://arxiv.org/pdf/2306.15427v1)

**Authors**: Lukas Gosch, Simon Geisler, Daniel Sturm, Bertrand Charpentier, Daniel Zügner, Stephan Günnemann

**Abstract**: Despite its success in the image domain, adversarial training does not (yet) stand out as an effective defense for Graph Neural Networks (GNNs) against graph structure perturbations. In the pursuit of fixing adversarial training (1) we show and overcome fundamental theoretical as well as practical limitations of the adopted graph learning setting in prior work; (2) we reveal that more flexible GNNs based on learnable graph diffusion are able to adjust to adversarial perturbations, while the learned message passing scheme is naturally interpretable; (3) we introduce the first attack for structure perturbations that, while targeting multiple nodes at once, is capable of handling global (graph-level) as well as local (node-level) constraints. Including these contributions, we demonstrate that adversarial training is a state-of-the-art defense against adversarial structure perturbations.



## **13. Your Attack Is Too DUMB: Formalizing Attacker Scenarios for Adversarial Transferability**

cs.CR

Accepted at RAID 2023

**SubmitDate**: 2023-06-27    [abs](http://arxiv.org/abs/2306.15363v1) [paper-pdf](http://arxiv.org/pdf/2306.15363v1)

**Authors**: Marco Alecci, Mauro Conti, Francesco Marchiori, Luca Martinelli, Luca Pajola

**Abstract**: Evasion attacks are a threat to machine learning models, where adversaries attempt to affect classifiers by injecting malicious samples. An alarming side-effect of evasion attacks is their ability to transfer among different models: this property is called transferability. Therefore, an attacker can produce adversarial samples on a custom model (surrogate) to conduct the attack on a victim's organization later. Although literature widely discusses how adversaries can transfer their attacks, their experimental settings are limited and far from reality. For instance, many experiments consider both attacker and defender sharing the same dataset, balance level (i.e., how the ground truth is distributed), and model architecture.   In this work, we propose the DUMB attacker model. This framework allows analyzing if evasion attacks fail to transfer when the training conditions of surrogate and victim models differ. DUMB considers the following conditions: Dataset soUrces, Model architecture, and the Balance of the ground truth. We then propose a novel testbed to evaluate many state-of-the-art evasion attacks with DUMB; the testbed consists of three computer vision tasks with two distinct datasets each, four types of balance levels, and three model architectures. Our analysis, which generated 13K tests over 14 distinct attacks, led to numerous novel findings in the scope of transferable attacks with surrogate models. In particular, mismatches between attackers and victims in terms of dataset source, balance levels, and model architecture lead to non-negligible loss of attack performance.



## **14. GPS-Spoofing Attack Detection Mechanism for UAV Swarms**

cs.CR

8 pages, 3 figures

**SubmitDate**: 2023-06-27    [abs](http://arxiv.org/abs/2301.12766v2) [paper-pdf](http://arxiv.org/pdf/2301.12766v2)

**Authors**: Pavlo Mykytyn, Marcin Brzozowski, Zoya Dyka, Peter Langendoerfer

**Abstract**: Recently autonomous and semi-autonomous Unmanned Aerial Vehicle (UAV) swarms started to receive a lot of research interest and demand from various civil application fields. However, for successful mission execution, UAV swarms require Global navigation satellite system signals and in particular, Global Positioning System (GPS) signals for navigation. Unfortunately, civil GPS signals are unencrypted and unauthenticated, which facilitates the execution of GPS spoofing attacks. During these attacks, adversaries mimic the authentic GPS signal and broadcast it to the targeted UAV in order to change its course, and force it to land or crash. In this study, we propose a GPS spoofing detection mechanism capable of detecting single-transmitter and multi-transmitter GPS spoofing attacks to prevent the outcomes mentioned above. Our detection mechanism is based on comparing the distance between each two swarm members calculated from their GPS coordinates to the distance acquired from Impulse Radio Ultra-Wideband ranging between the same swarm members. If the difference in distances is larger than a chosen threshold the GPS spoofing attack is declared detected.



## **15. Feature Adversarial Distillation for Point Cloud Classification**

cs.CV

Accepted to ICIP2023

**SubmitDate**: 2023-06-27    [abs](http://arxiv.org/abs/2306.14221v2) [paper-pdf](http://arxiv.org/pdf/2306.14221v2)

**Authors**: YuXing Lee, Wei Wu

**Abstract**: Due to the point cloud's irregular and unordered geometry structure, conventional knowledge distillation technology lost a lot of information when directly used on point cloud tasks. In this paper, we propose Feature Adversarial Distillation (FAD) method, a generic adversarial loss function in point cloud distillation, to reduce loss during knowledge transfer. In the feature extraction stage, the features extracted by the teacher are used as the discriminator, and the students continuously generate new features in the training stage. The feature of the student is obtained by attacking the feedback from the teacher and getting a score to judge whether the student has learned the knowledge well or not. In experiments on standard point cloud classification on ModelNet40 and ScanObjectNN datasets, our method reduced the information loss of knowledge transfer in distillation in 40x model compression while maintaining competitive performance.



## **16. A Highly Accurate Query-Recovery Attack against Searchable Encryption using Non-Indexed Documents**

cs.CR

Published in USENIX 2021. Full version with extended appendices and  removed some typos

**SubmitDate**: 2023-06-27    [abs](http://arxiv.org/abs/2306.15302v1) [paper-pdf](http://arxiv.org/pdf/2306.15302v1)

**Authors**: Marc Damie, Florian Hahn, Andreas Peter

**Abstract**: Cloud data storage solutions offer customers cost-effective and reduced data management. While attractive, data security issues remain to be a core concern. Traditional encryption protects stored documents, but hinders simple functionalities such as keyword search. Therefore, searchable encryption schemes have been proposed to allow for the search on encrypted data. Efficient schemes leak at least the access pattern (the accessed documents per keyword search), which is known to be exploitable in query recovery attacks assuming the attacker has a significant amount of background knowledge on the stored documents. Existing attacks can only achieve decent results with strong adversary models (e.g. at least 20% of previously known documents or require additional knowledge such as on query frequencies) and they give no metric to evaluate the certainty of recovered queries. This hampers their practical utility and questions their relevance in the real-world.   We propose a refined score attack which achieves query recovery rates of around 85% without requiring exact background knowledge on stored documents; a distributionally similar, but otherwise different (i.e., non-indexed), dataset suffices. The attack starts with very few known queries (around 10 known queries in our experiments over different datasets of varying size) and then iteratively recovers further queries with confidence scores by adding previously recovered queries that had high confidence scores to the set of known queries. Additional to high recovery rates, our approach yields interpretable results in terms of confidence scores.



## **17. On the Universal Adversarial Perturbations for Efficient Data-free Adversarial Detection**

cs.CL

Accepted by ACL2023 (Short Paper)

**SubmitDate**: 2023-06-27    [abs](http://arxiv.org/abs/2306.15705v1) [paper-pdf](http://arxiv.org/pdf/2306.15705v1)

**Authors**: Songyang Gao, Shihan Dou, Qi Zhang, Xuanjing Huang, Jin Ma, Ying Shan

**Abstract**: Detecting adversarial samples that are carefully crafted to fool the model is a critical step to socially-secure applications. However, existing adversarial detection methods require access to sufficient training data, which brings noteworthy concerns regarding privacy leakage and generalizability. In this work, we validate that the adversarial sample generated by attack algorithms is strongly related to a specific vector in the high-dimensional inputs. Such vectors, namely UAPs (Universal Adversarial Perturbations), can be calculated without original training data. Based on this discovery, we propose a data-agnostic adversarial detection framework, which induces different responses between normal and adversarial samples to UAPs. Experimental results show that our method achieves competitive detection performance on various text classification tasks, and maintains an equivalent time consumption to normal inference.



## **18. DSRM: Boost Textual Adversarial Training with Distribution Shift Risk Minimization**

cs.CL

Accepted by ACL2023

**SubmitDate**: 2023-06-27    [abs](http://arxiv.org/abs/2306.15164v1) [paper-pdf](http://arxiv.org/pdf/2306.15164v1)

**Authors**: Songyang Gao, Shihan Dou, Yan Liu, Xiao Wang, Qi Zhang, Zhongyu Wei, Jin Ma, Ying Shan

**Abstract**: Adversarial training is one of the best-performing methods in improving the robustness of deep language models. However, robust models come at the cost of high time consumption, as they require multi-step gradient ascents or word substitutions to obtain adversarial samples. In addition, these generated samples are deficient in grammatical quality and semantic consistency, which impairs the effectiveness of adversarial training. To address these problems, we introduce a novel, effective procedure for instead adversarial training with only clean data. Our procedure, distribution shift risk minimization (DSRM), estimates the adversarial loss by perturbing the input data's probability distribution rather than their embeddings. This formulation results in a robust model that minimizes the expected global loss under adversarial attacks. Our approach requires zero adversarial samples for training and reduces time consumption by up to 70\% compared to current best-performing adversarial training methods. Experiments demonstrate that DSRM considerably improves BERT's resistance to textual adversarial attacks and achieves state-of-the-art robust accuracy on various benchmarks.



## **19. Towards Sybil Resilience in Decentralized Learning**

cs.DC

**SubmitDate**: 2023-06-26    [abs](http://arxiv.org/abs/2306.15044v1) [paper-pdf](http://arxiv.org/pdf/2306.15044v1)

**Authors**: Thomas Werthenbach, Johan Pouwelse

**Abstract**: Federated learning is a privacy-enforcing machine learning technology but suffers from limited scalability. This limitation mostly originates from the internet connection and memory capacity of the central parameter server, and the complexity of the model aggregation function. Decentralized learning has recently been emerging as a promising alternative to federated learning. This novel technology eliminates the need for a central parameter server by decentralizing the model aggregation across all participating nodes. Numerous studies have been conducted on improving the resilience of federated learning against poisoning and Sybil attacks, whereas the resilience of decentralized learning remains largely unstudied. This research gap serves as the main motivator for this study, in which our objective is to improve the Sybil poisoning resilience of decentralized learning.   We present SybilWall, an innovative algorithm focused on increasing the resilience of decentralized learning against targeted Sybil poisoning attacks. By combining a Sybil-resistant aggregation function based on similarity between Sybils with a novel probabilistic gossiping mechanism, we establish a new benchmark for scalable, Sybil-resilient decentralized learning.   A comprehensive empirical evaluation demonstrated that SybilWall outperforms existing state-of-the-art solutions designed for federated learning scenarios and is the only algorithm to obtain consistent accuracy over a range of adversarial attack scenarios. We also found SybilWall to diminish the utility of creating many Sybils, as our evaluations demonstrate a higher success rate among adversaries employing fewer Sybils. Finally, we suggest a number of possible improvements to SybilWall and highlight promising future research directions.



## **20. Are aligned neural networks adversarially aligned?**

cs.CL

**SubmitDate**: 2023-06-26    [abs](http://arxiv.org/abs/2306.15447v1) [paper-pdf](http://arxiv.org/pdf/2306.15447v1)

**Authors**: Nicholas Carlini, Milad Nasr, Christopher A. Choquette-Choo, Matthew Jagielski, Irena Gao, Anas Awadalla, Pang Wei Koh, Daphne Ippolito, Katherine Lee, Florian Tramer, Ludwig Schmidt

**Abstract**: Large language models are now tuned to align with the goals of their creators, namely to be "helpful and harmless." These models should respond helpfully to user questions, but refuse to answer requests that could cause harm. However, adversarial users can construct inputs which circumvent attempts at alignment. In this work, we study to what extent these models remain aligned, even when interacting with an adversarial user who constructs worst-case inputs (adversarial examples). These inputs are designed to cause the model to emit harmful content that would otherwise be prohibited. We show that existing NLP-based optimization attacks are insufficiently powerful to reliably attack aligned text models: even when current NLP-based attacks fail, we can find adversarial inputs with brute force. As a result, the failure of current attacks should not be seen as proof that aligned text models remain aligned under adversarial inputs.   However the recent trend in large-scale ML models is multimodal models that allow users to provide images that influence the text that is generated. We show these models can be easily attacked, i.e., induced to perform arbitrary un-aligned behavior through adversarial perturbation of the input image. We conjecture that improved NLP attacks may demonstrate this same level of adversarial control over text-only models.



## **21. On the Resilience of Machine Learning-Based IDS for Automotive Networks**

cs.CR

**SubmitDate**: 2023-06-26    [abs](http://arxiv.org/abs/2306.14782v1) [paper-pdf](http://arxiv.org/pdf/2306.14782v1)

**Authors**: Ivo Zenden, Han Wang, Alfonso Iacovazzi, Arash Vahidi, Rolf Blom, Shahid Raza

**Abstract**: Modern automotive functions are controlled by a large number of small computers called electronic control units (ECUs). These functions span from safety-critical autonomous driving to comfort and infotainment. ECUs communicate with one another over multiple internal networks using different technologies. Some, such as Controller Area Network (CAN), are very simple and provide minimal or no security services. Machine learning techniques can be used to detect anomalous activities in such networks. However, it is necessary that these machine learning techniques are not prone to adversarial attacks. In this paper, we investigate adversarial sample vulnerabilities in four different machine learning-based intrusion detection systems for automotive networks. We show that adversarial samples negatively impact three of the four studied solutions. Furthermore, we analyze transferability of adversarial samples between different systems. We also investigate detection performance and the attack success rate after using adversarial samples in the training. After analyzing these results, we discuss whether current solutions are mature enough for a use in modern vehicles.



## **22. No Need to Know Physics: Resilience of Process-based Model-free Anomaly Detection for Industrial Control Systems**

cs.CR

An updated version of the paper has been published at ACSAC'2022:  Assessing Model-free Anomaly Detection in Industrial Control Systems Against  Generic Concealment Attacks https://dl.acm.org/doi/10.1145/3564625.3564633

**SubmitDate**: 2023-06-26    [abs](http://arxiv.org/abs/2012.03586v2) [paper-pdf](http://arxiv.org/pdf/2012.03586v2)

**Authors**: Alessandro Erba, Nils Ole Tippenhauer

**Abstract**: In recent years, a number of process-based anomaly detection schemes for Industrial Control Systems were proposed. In this work, we provide the first systematic analysis of such schemes, and introduce a taxonomy of properties that are verified by those detection systems. We then present a novel general framework to generate adversarial spoofing signals that violate physical properties of the system, and use the framework to analyze four anomaly detectors published at top security conferences. We find that three of those detectors are susceptible to a number of adversarial manipulations (e.g., spoofing with precomputed patterns), which we call Synthetic Sensor Spoofing and one is resilient against our attacks. We investigate the root of its resilience and demonstrate that it comes from the properties that we introduced. Our attacks reduce the Recall (True Positive Rate) of the attacked schemes making them not able to correctly detect anomalies. Thus, the vulnerabilities we discovered in the anomaly detectors show that (despite an original good detection performance), those detectors are not able to reliably learn physical properties of the system. Even attacks that prior work was expected to be resilient against (based on verified properties) were found to be successful. We argue that our findings demonstrate the need for both more complete attacks in datasets, and more critical analysis of process-based anomaly detectors. We plan to release our implementation as open-source, together with an extension of two public datasets with a set of Synthetic Sensor Spoofing attacks as generated by our framework.



## **23. PWSHAP: A Path-Wise Explanation Model for Targeted Variables**

stat.ML

**SubmitDate**: 2023-06-26    [abs](http://arxiv.org/abs/2306.14672v1) [paper-pdf](http://arxiv.org/pdf/2306.14672v1)

**Authors**: Lucile Ter-Minassian, Oscar Clivio, Karla Diaz-Ordaz, Robin J. Evans, Chris Holmes

**Abstract**: Predictive black-box models can exhibit high accuracy but their opaque nature hinders their uptake in safety-critical deployment environments. Explanation methods (XAI) can provide confidence for decision-making through increased transparency. However, existing XAI methods are not tailored towards models in sensitive domains where one predictor is of special interest, such as a treatment effect in a clinical model, or ethnicity in policy models. We introduce Path-Wise Shapley effects (PWSHAP), a framework for assessing the targeted effect of a binary (e.g.~treatment) variable from a complex outcome model. Our approach augments the predictive model with a user-defined directed acyclic graph (DAG). The method then uses the graph alongside on-manifold Shapley values to identify effects along causal pathways whilst maintaining robustness to adversarial attacks. We establish error bounds for the identified path-wise Shapley effects and for Shapley values. We show PWSHAP can perform local bias and mediation analyses with faithfulness to the model. Further, if the targeted variable is randomised we can quantify local effect modification. We demonstrate the resolution, interpretability, and true locality of our approach on examples and a real-world experiment.



## **24. A Threat-Intelligence Driven Methodology to Incorporate Uncertainty in Cyber Risk Analysis and Enhance Decision Making**

cs.CR

**SubmitDate**: 2023-06-26    [abs](http://arxiv.org/abs/2302.13082v2) [paper-pdf](http://arxiv.org/pdf/2302.13082v2)

**Authors**: Martijn Dekker, Lampis Alevizos

**Abstract**: The challenge of decision-making under uncertainty in information security has become increasingly important, given the unpredictable probabilities and effects of events in the ever-changing cyber threat landscape. Cyber threat intelligence provides decision-makers with the necessary information and context to understand and anticipate potential threats, reducing uncertainty and improving the accuracy of risk analysis. The latter is a principal element of evidence-based decision-making, and it is essential to recognize that addressing uncertainty requires a new, threat-intelligence driven methodology and risk analysis approach. We propose a solution to this challenge by introducing a threat-intelligence based security assessment methodology and a decision-making strategy that considers both known unknowns and unknown unknowns. The proposed methodology aims to enhance the quality of decision-making by utilizing causal graphs, which offer an alternative to conventional methodologies that rely on attack trees, resulting in a reduction of uncertainty. Furthermore, we consider tactics, techniques, and procedures that are possible, probable, and plausible, improving the predictability of adversary behavior. Our proposed solution provides practical guidance for information security leaders to make informed decisions in uncertain situations. This paper offers a new perspective on addressing the challenge of decision-making under uncertainty in information security by introducing a methodology that can help decision-makers navigate the intricacies of the dynamic and continuously evolving landscape of cyber threats.



## **25. 3D-Aware Adversarial Makeup Generation for Facial Privacy Protection**

cs.CV

Accepted by TPAMI 2023

**SubmitDate**: 2023-06-26    [abs](http://arxiv.org/abs/2306.14640v1) [paper-pdf](http://arxiv.org/pdf/2306.14640v1)

**Authors**: Yueming Lyu, Yue Jiang, Ziwen He, Bo Peng, Yunfan Liu, Jing Dong

**Abstract**: The privacy and security of face data on social media are facing unprecedented challenges as it is vulnerable to unauthorized access and identification. A common practice for solving this problem is to modify the original data so that it could be protected from being recognized by malicious face recognition (FR) systems. However, such ``adversarial examples'' obtained by existing methods usually suffer from low transferability and poor image quality, which severely limits the application of these methods in real-world scenarios. In this paper, we propose a 3D-Aware Adversarial Makeup Generation GAN (3DAM-GAN). which aims to improve the quality and transferability of synthetic makeup for identity information concealing. Specifically, a UV-based generator consisting of a novel Makeup Adjustment Module (MAM) and Makeup Transfer Module (MTM) is designed to render realistic and robust makeup with the aid of symmetric characteristics of human faces. Moreover, a makeup attack mechanism with an ensemble training strategy is proposed to boost the transferability of black-box models. Extensive experiment results on several benchmark datasets demonstrate that 3DAM-GAN could effectively protect faces against various FR models, including both publicly available state-of-the-art models and commercial face verification APIs, such as Face++, Baidu and Aliyun.



## **26. The race to robustness: exploiting fragile models for urban camouflage and the imperative for machine learning security**

cs.LG

Accepted to IEEE TENSYMP 2023

**SubmitDate**: 2023-06-26    [abs](http://arxiv.org/abs/2306.14609v1) [paper-pdf](http://arxiv.org/pdf/2306.14609v1)

**Authors**: Harriet Farlow, Matthew Garratt, Gavin Mount, Tim Lynar

**Abstract**: Adversarial Machine Learning (AML) represents the ability to disrupt Machine Learning (ML) algorithms through a range of methods that broadly exploit the architecture of deep learning optimisation. This paper presents Distributed Adversarial Regions (DAR), a novel method that implements distributed instantiations of computer vision-based AML attack methods that may be used to disguise objects from image recognition in both white and black box settings. We consider the context of object detection models used in urban environments, and benchmark the MobileNetV2, NasNetMobile and DenseNet169 models against a subset of relevant images from the ImageNet dataset. We evaluate optimal parameters (size, number and perturbation method), and compare to state-of-the-art AML techniques that perturb the entire image. We find that DARs can cause a reduction in confidence of 40.4% on average, but with the benefit of not requiring the entire image, or the focal object, to be perturbed. The DAR method is a deliberately simple approach where the intention is to highlight how an adversary with very little skill could attack models that may already be productionised, and to emphasise the fragility of foundational object detection models. We present this as a contribution to the field of ML security as well as AML. This paper contributes a novel adversarial method, an original comparison between DARs and other AML methods, and frames it in a new context - that of urban camouflage and the necessity for ML security and model robustness.



## **27. Towards Out-of-Distribution Adversarial Robustness**

cs.LG

Version of NeurIPS 2023 submission

**SubmitDate**: 2023-06-26    [abs](http://arxiv.org/abs/2210.03150v4) [paper-pdf](http://arxiv.org/pdf/2210.03150v4)

**Authors**: Adam Ibrahim, Charles Guille-Escuret, Ioannis Mitliagkas, Irina Rish, David Krueger, Pouya Bashivan

**Abstract**: Adversarial robustness continues to be a major challenge for deep learning. A core issue is that robustness to one type of attack often fails to transfer to other attacks. While prior work establishes a theoretical trade-off in robustness against different $L_p$ norms, we show that there is potential for improvement against many commonly used attacks by adopting a domain generalisation approach. Concretely, we treat each type of attack as a domain, and apply the Risk Extrapolation method (REx), which promotes similar levels of robustness against all training attacks. Compared to existing methods, we obtain similar or superior worst-case adversarial robustness on attacks seen during training. Moreover, we achieve superior performance on families or tunings of attacks only encountered at test time. On ensembles of attacks, our approach improves the accuracy from 3.4% with the best existing baseline to 25.9% on MNIST, and from 16.9% to 23.5% on CIFAR10.



## **28. Computational Asymmetries in Robust Classification**

cs.LG

**SubmitDate**: 2023-06-25    [abs](http://arxiv.org/abs/2306.14326v1) [paper-pdf](http://arxiv.org/pdf/2306.14326v1)

**Authors**: Samuele Marro, Michele Lombardi

**Abstract**: In the context of adversarial robustness, we make three strongly related contributions. First, we prove that while attacking ReLU classifiers is $\mathit{NP}$-hard, ensuring their robustness at training time is $\Sigma^2_P$-hard (even on a single example). This asymmetry provides a rationale for the fact that robust classifications approaches are frequently fooled in the literature. Second, we show that inference-time robustness certificates are not affected by this asymmetry, by introducing a proof-of-concept approach named Counter-Attack (CA). Indeed, CA displays a reversed asymmetry: running the defense is $\mathit{NP}$-hard, while attacking it is $\Sigma_2^P$-hard. Finally, motivated by our previous result, we argue that adversarial attacks can be used in the context of robustness certification, and provide an empirical evaluation of their effectiveness. As a byproduct of this process, we also release UG100, a benchmark dataset for adversarial attacks.



## **29. Enhancing Adversarial Training via Reweighting Optimization Trajectory**

cs.LG

Accepted by ECML 2023

**SubmitDate**: 2023-06-25    [abs](http://arxiv.org/abs/2306.14275v1) [paper-pdf](http://arxiv.org/pdf/2306.14275v1)

**Authors**: Tianjin Huang, Shiwei Liu, Tianlong Chen, Meng Fang, Li Shen, Vlaod Menkovski, Lu Yin, Yulong Pei, Mykola Pechenizkiy

**Abstract**: Despite the fact that adversarial training has become the de facto method for improving the robustness of deep neural networks, it is well-known that vanilla adversarial training suffers from daunting robust overfitting, resulting in unsatisfactory robust generalization. A number of approaches have been proposed to address these drawbacks such as extra regularization, adversarial weights perturbation, and training with more data over the last few years. However, the robust generalization improvement is yet far from satisfactory. In this paper, we approach this challenge with a brand new perspective -- refining historical optimization trajectories. We propose a new method named \textbf{Weighted Optimization Trajectories (WOT)} that leverages the optimization trajectories of adversarial training in time. We have conducted extensive experiments to demonstrate the effectiveness of WOT under various state-of-the-art adversarial attacks. Our results show that WOT integrates seamlessly with the existing adversarial training methods and consistently overcomes the robust overfitting issue, resulting in better adversarial robustness. For example, WOT boosts the robust accuracy of AT-PGD under AA-$L_{\infty}$ attack by 1.53\% $\sim$ 6.11\% and meanwhile increases the clean accuracy by 0.55\%$\sim$5.47\% across SVHN, CIFAR-10, CIFAR-100, and Tiny-ImageNet datasets.



## **30. A Spectral Perspective towards Understanding and Improving Adversarial Robustness**

cs.CV

**SubmitDate**: 2023-06-25    [abs](http://arxiv.org/abs/2306.14262v1) [paper-pdf](http://arxiv.org/pdf/2306.14262v1)

**Authors**: Binxiao Huang, Rui Lin, Chaofan Tao, Ngai Wong

**Abstract**: Deep neural networks (DNNs) are incredibly vulnerable to crafted, imperceptible adversarial perturbations. While adversarial training (AT) has proven to be an effective defense approach, the AT mechanism for robustness improvement is not fully understood. This work investigates AT from a spectral perspective, adding new insights to the design of effective defenses. In particular, we show that AT induces the deep model to focus more on the low-frequency region, which retains the shape-biased representations, to gain robustness. Further, we find that the spectrum of a white-box attack is primarily distributed in regions the model focuses on, and the perturbation attacks the spectral bands where the model is vulnerable. Based on this observation, to train a model tolerant to frequency-varying perturbation, we propose a spectral alignment regularization (SAR) such that the spectral output inferred by an attacked adversarial input stays as close as possible to its natural input counterpart. Experiments demonstrate that SAR and its weight averaging (WA) extension could significantly improve the robust accuracy by 1.14% ~ 3.87% relative to the standard AT, across multiple datasets (CIFAR-10, CIFAR-100 and Tiny ImageNet), and various attacks (PGD, C&W and Autoattack), without any extra data.



## **31. Backdoor Attacks in Peer-to-Peer Federated Learning**

cs.LG

**SubmitDate**: 2023-06-25    [abs](http://arxiv.org/abs/2301.09732v3) [paper-pdf](http://arxiv.org/pdf/2301.09732v3)

**Authors**: Gokberk Yar, Simona Boboila, Cristina Nita-Rotaru, Alina Oprea

**Abstract**: Most machine learning applications rely on centralized learning processes, opening up the risk of exposure of their training datasets. While federated learning (FL) mitigates to some extent these privacy risks, it relies on a trusted aggregation server for training a shared global model. Recently, new distributed learning architectures based on Peer-to-Peer Federated Learning (P2PFL) offer advantages in terms of both privacy and reliability. Still, their resilience to poisoning attacks during training has not been investigated. In this paper, we propose new backdoor attacks for P2PFL that leverage structural graph properties to select the malicious nodes, and achieve high attack success, while remaining stealthy. We evaluate our attacks under various realistic conditions, including multiple graph topologies, limited adversarial visibility of the network, and clients with non-IID data. Finally, we show the limitations of existing defenses adapted from FL and design a new defense that successfully mitigates the backdoor attacks, without an impact on model accuracy.



## **32. On Evaluating the Adversarial Robustness of Semantic Segmentation Models**

cs.CV

**SubmitDate**: 2023-06-25    [abs](http://arxiv.org/abs/2306.14217v1) [paper-pdf](http://arxiv.org/pdf/2306.14217v1)

**Authors**: Levente Halmosi, Mark Jelasity

**Abstract**: Achieving robustness against adversarial input perturbation is an important and intriguing problem in machine learning. In the area of semantic image segmentation, a number of adversarial training approaches have been proposed as a defense against adversarial perturbation, but the methodology of evaluating the robustness of the models is still lacking, compared to image classification. Here, we demonstrate that, just like in image classification, it is important to evaluate the models over several different and hard attacks. We propose a set of gradient based iterative attacks and show that it is essential to perform a large number of iterations. We include attacks against the internal representations of the models as well. We apply two types of attacks: maximizing the error with a bounded perturbation, and minimizing the perturbation for a given level of error. Using this set of attacks, we show for the first time that a number of models in previous work that are claimed to be robust are in fact not robust at all. We then evaluate simple adversarial training algorithms that produce reasonably robust models even under our set of strong attacks. Our results indicate that a key design decision to achieve any robustness is to use only adversarial examples during training. However, this introduces a trade-off between robustness and accuracy.



## **33. The defender's perspective on automatic speaker verification: An overview**

cs.SD

Accepted to IJCAI 2023 Workshop

**SubmitDate**: 2023-06-25    [abs](http://arxiv.org/abs/2305.12804v2) [paper-pdf](http://arxiv.org/pdf/2305.12804v2)

**Authors**: Haibin Wu, Jiawen Kang, Lingwei Meng, Helen Meng, Hung-yi Lee

**Abstract**: Automatic speaker verification (ASV) plays a critical role in security-sensitive environments. Regrettably, the reliability of ASV has been undermined by the emergence of spoofing attacks, such as replay and synthetic speech, as well as adversarial attacks and the relatively new partially fake speech. While there are several review papers that cover replay and synthetic speech, and adversarial attacks, there is a notable gap in a comprehensive review that addresses defense against adversarial attacks and the recently emerged partially fake speech. Thus, the aim of this paper is to provide a thorough and systematic overview of the defense methods used against these types of attacks.



## **34. Robust Spatiotemporal Traffic Forecasting with Reinforced Dynamic Adversarial Training**

cs.LG

Accepted by KDD 2023

**SubmitDate**: 2023-06-25    [abs](http://arxiv.org/abs/2306.14126v1) [paper-pdf](http://arxiv.org/pdf/2306.14126v1)

**Authors**: Fan Liu, Weijia Zhang, Hao Liu

**Abstract**: Machine learning-based forecasting models are commonly used in Intelligent Transportation Systems (ITS) to predict traffic patterns and provide city-wide services. However, most of the existing models are susceptible to adversarial attacks, which can lead to inaccurate predictions and negative consequences such as congestion and delays. Therefore, improving the adversarial robustness of these models is crucial for ITS. In this paper, we propose a novel framework for incorporating adversarial training into spatiotemporal traffic forecasting tasks. We demonstrate that traditional adversarial training methods designated for static domains cannot be directly applied to traffic forecasting tasks, as they fail to effectively defend against dynamic adversarial attacks. Then, we propose a reinforcement learning-based method to learn the optimal node selection strategy for adversarial examples, which simultaneously strengthens the dynamic attack defense capability and reduces the model overfitting. Additionally, we introduce a self-knowledge distillation regularization module to overcome the "forgetting issue" caused by continuously changing adversarial nodes during training. We evaluate our approach on two real-world traffic datasets and demonstrate its superiority over other baselines. Our method effectively enhances the adversarial robustness of spatiotemporal traffic forecasting models. The source code for our framework is available at https://github.com/usail-hkust/RDAT.



## **35. Identifying Adversarially Attackable and Robust Samples**

cs.LG

**SubmitDate**: 2023-06-25    [abs](http://arxiv.org/abs/2301.12896v3) [paper-pdf](http://arxiv.org/pdf/2301.12896v3)

**Authors**: Vyas Raina, Mark Gales

**Abstract**: Adversarial attacks insert small, imperceptible perturbations to input samples that cause large, undesired changes to the output of deep learning models. Despite extensive research on generating adversarial attacks and building defense systems, there has been limited research on understanding adversarial attacks from an input-data perspective. This work introduces the notion of sample attackability, where we aim to identify samples that are most susceptible to adversarial attacks (attackable samples) and conversely also identify the least susceptible samples (robust samples). We propose a deep-learning-based detector to identify the adversarially attackable and robust samples in an unseen dataset for an unseen target model. Experiments on standard image classification datasets enables us to assess the portability of the deep attackability detector across a range of architectures. We find that the deep attackability detector performs better than simple model uncertainty-based measures for identifying the attackable/robust samples. This suggests that uncertainty is an inadequate proxy for measuring sample distance to a decision boundary. In addition to better understanding adversarial attack theory, it is found that the ability to identify the adversarially attackable and robust samples has implications for improving the efficiency of sample-selection tasks.



## **36. Sentiment Perception Adversarial Attacks on Neural Machine Translation Systems**

cs.CL

**SubmitDate**: 2023-06-25    [abs](http://arxiv.org/abs/2305.01437v2) [paper-pdf](http://arxiv.org/pdf/2305.01437v2)

**Authors**: Vyas Raina, Mark Gales

**Abstract**: With the advent of deep learning methods, Neural Machine Translation (NMT) systems have become increasingly powerful. However, deep learning based systems are susceptible to adversarial attacks, where imperceptible changes to the input can cause undesirable changes at the output of the system. To date there has been little work investigating adversarial attacks on sequence-to-sequence systems, such as NMT models. Previous work in NMT has examined attacks with the aim of introducing target phrases in the output sequence. In this work, adversarial attacks for NMT systems are explored from an output perception perspective. Thus the aim of an attack is to change the perception of the output sequence, without altering the perception of the input sequence. For example, an adversary may distort the sentiment of translated reviews to have an exaggerated positive sentiment. In practice it is challenging to run extensive human perception experiments, so a proxy deep-learning classifier applied to the NMT output is used to measure perception changes. Experiments demonstrate that the sentiment perception of NMT systems' output sequences can be changed significantly with small imperceptible changes to input sequences.



## **37. Machine Learning needs its own Randomness Standard: Randomised Smoothing and PRNG-based attacks**

cs.LG

**SubmitDate**: 2023-06-24    [abs](http://arxiv.org/abs/2306.14043v1) [paper-pdf](http://arxiv.org/pdf/2306.14043v1)

**Authors**: Pranav Dahiya, Ilia Shumailov, Ross Anderson

**Abstract**: Randomness supports many critical functions in the field of machine learning (ML) including optimisation, data selection, privacy, and security. ML systems outsource the task of generating or harvesting randomness to the compiler, the cloud service provider or elsewhere in the toolchain. Yet there is a long history of attackers exploiting poor randomness, or even creating it -- as when the NSA put backdoors in random number generators to break cryptography. In this paper we consider whether attackers can compromise an ML system using only the randomness on which they commonly rely. We focus our effort on Randomised Smoothing, a popular approach to train certifiably robust models, and to certify specific input datapoints of an arbitrary model. We choose Randomised Smoothing since it is used for both security and safety -- to counteract adversarial examples and quantify uncertainty respectively. Under the hood, it relies on sampling Gaussian noise to explore the volume around a data point to certify that a model is not vulnerable to adversarial examples. We demonstrate an entirely novel attack against it, where an attacker backdoors the supplied randomness to falsely certify either an overestimate or an underestimate of robustness. We demonstrate that such attacks are possible, that they require very small changes to randomness to succeed, and that they can be hard to detect. As an example, we hide an attack in the random number generator and show that the randomness tests suggested by NIST fail to detect it. We advocate updating the NIST guidelines on random number testing to make them more appropriate for safety-critical and security-critical machine-learning applications.



## **38. Boosting Model Inversion Attacks with Adversarial Examples**

cs.CR

18 pages, 13 figures

**SubmitDate**: 2023-06-24    [abs](http://arxiv.org/abs/2306.13965v1) [paper-pdf](http://arxiv.org/pdf/2306.13965v1)

**Authors**: Shuai Zhou, Tianqing Zhu, Dayong Ye, Xin Yu, Wanlei Zhou

**Abstract**: Model inversion attacks involve reconstructing the training data of a target model, which raises serious privacy concerns for machine learning models. However, these attacks, especially learning-based methods, are likely to suffer from low attack accuracy, i.e., low classification accuracy of these reconstructed data by machine learning classifiers. Recent studies showed an alternative strategy of model inversion attacks, GAN-based optimization, can improve the attack accuracy effectively. However, these series of GAN-based attacks reconstruct only class-representative training data for a class, whereas learning-based attacks can reconstruct diverse data for different training data in each class. Hence, in this paper, we propose a new training paradigm for a learning-based model inversion attack that can achieve higher attack accuracy in a black-box setting. First, we regularize the training process of the attack model with an added semantic loss function and, second, we inject adversarial examples into the training data to increase the diversity of the class-related parts (i.e., the essential features for classification tasks) in training data. This scheme guides the attack model to pay more attention to the class-related parts of the original data during the data reconstruction process. The experimental results show that our method greatly boosts the performance of existing learning-based model inversion attacks. Even when no extra queries to the target model are allowed, the approach can still improve the attack accuracy of reconstructed data. This new attack shows that the severity of the threat from learning-based model inversion adversaries is underestimated and more robust defenses are required.



## **39. Similarity Preserving Adversarial Graph Contrastive Learning**

cs.LG

9 pages; KDD'23

**SubmitDate**: 2023-06-24    [abs](http://arxiv.org/abs/2306.13854v1) [paper-pdf](http://arxiv.org/pdf/2306.13854v1)

**Authors**: Yeonjun In, Kanghoon Yoon, Chanyoung Park

**Abstract**: Recent works demonstrate that GNN models are vulnerable to adversarial attacks, which refer to imperceptible perturbation on the graph structure and node features. Among various GNN models, graph contrastive learning (GCL) based methods specifically suffer from adversarial attacks due to their inherent design that highly depends on the self-supervision signals derived from the original graph, which however already contains noise when the graph is attacked. To achieve adversarial robustness against such attacks, existing methods adopt adversarial training (AT) to the GCL framework, which considers the attacked graph as an augmentation under the GCL framework. However, we find that existing adversarially trained GCL methods achieve robustness at the expense of not being able to preserve the node feature similarity. In this paper, we propose a similarity-preserving adversarial graph contrastive learning (SP-AGCL) framework that contrasts the clean graph with two auxiliary views of different properties (i.e., the node similarity-preserving view and the adversarial view). Extensive experiments demonstrate that SP-AGCL achieves a competitive performance on several downstream tasks, and shows its effectiveness in various scenarios, e.g., a network with adversarial attacks, noisy labels, and heterophilous neighbors. Our code is available at https://github.com/yeonjun-in/torch-SP-AGCL.



## **40. A First Order Meta Stackelberg Method for Robust Federated Learning**

cs.LG

Accepted to ICML 2023 Workshop on The 2nd New Frontiers In  Adversarial Machine Learning. arXiv admin note: substantial text overlap with  arXiv:2306.13273

**SubmitDate**: 2023-06-23    [abs](http://arxiv.org/abs/2306.13800v1) [paper-pdf](http://arxiv.org/pdf/2306.13800v1)

**Authors**: Yunian Pan, Tao Li, Henger Li, Tianyi Xu, Zizhan Zheng, Quanyan Zhu

**Abstract**: Previous research has shown that federated learning (FL) systems are exposed to an array of security risks. Despite the proposal of several defensive strategies, they tend to be non-adaptive and specific to certain types of attacks, rendering them ineffective against unpredictable or adaptive threats. This work models adversarial federated learning as a Bayesian Stackelberg Markov game (BSMG) to capture the defender's incomplete information of various attack types. We propose meta-Stackelberg learning (meta-SL), a provably efficient meta-learning algorithm, to solve the equilibrium strategy in BSMG, leading to an adaptable FL defense. We demonstrate that meta-SL converges to the first-order $\varepsilon$-equilibrium point in $O(\varepsilon^{-2})$ gradient iterations, with $O(\varepsilon^{-4})$ samples needed per iteration, matching the state of the art. Empirical evidence indicates that our meta-Stackelberg framework performs exceptionally well against potent model poisoning and backdoor attacks of an uncertain nature.



## **41. Creating Valid Adversarial Examples of Malware**

cs.CR

19 pages, 4 figures

**SubmitDate**: 2023-06-23    [abs](http://arxiv.org/abs/2306.13587v1) [paper-pdf](http://arxiv.org/pdf/2306.13587v1)

**Authors**: Matouš Kozák, Martin Jureček, Mark Stamp, Fabio Di Troia

**Abstract**: Machine learning is becoming increasingly popular as a go-to approach for many tasks due to its world-class results. As a result, antivirus developers are incorporating machine learning models into their products. While these models improve malware detection capabilities, they also carry the disadvantage of being susceptible to adversarial attacks. Although this vulnerability has been demonstrated for many models in white-box settings, a black-box attack is more applicable in practice for the domain of malware detection. We present a generator of adversarial malware examples using reinforcement learning algorithms. The reinforcement learning agents utilize a set of functionality-preserving modifications, thus creating valid adversarial examples. Using the proximal policy optimization (PPO) algorithm, we achieved an evasion rate of 53.84% against the gradient-boosted decision tree (GBDT) model. The PPO agent previously trained against the GBDT classifier scored an evasion rate of 11.41% against the neural network-based classifier MalConv and an average evasion rate of 2.31% against top antivirus programs. Furthermore, we discovered that random application of our functionality-preserving portable executable modifications successfully evades leading antivirus engines, with an average evasion rate of 11.65%. These findings indicate that machine learning-based models used in malware detection systems are vulnerable to adversarial attacks and that better safeguards need to be taken to protect these systems.



## **42. A First Order Meta Stackelberg Method for Robust Federated Learning (Technical Report)**

cs.CR

**SubmitDate**: 2023-06-23    [abs](http://arxiv.org/abs/2306.13273v1) [paper-pdf](http://arxiv.org/pdf/2306.13273v1)

**Authors**: Henger Li, Tianyi Xu, Tao Li, Yunian Pan, Quanyan Zhu, Zizhan Zheng

**Abstract**: Recent research efforts indicate that federated learning (FL) systems are vulnerable to a variety of security breaches. While numerous defense strategies have been suggested, they are mainly designed to counter specific attack patterns and lack adaptability, rendering them less effective when facing uncertain or adaptive threats. This work models adversarial FL as a Bayesian Stackelberg Markov game (BSMG) between the defender and the attacker to address the lack of adaptability to uncertain adaptive attacks. We further devise an effective meta-learning technique to solve for the Stackelberg equilibrium, leading to a resilient and adaptable defense. The experiment results suggest that our meta-Stackelberg learning approach excels in combating intense model poisoning and backdoor attacks of indeterminate types.



## **43. Document Image Cleaning using Budget-Aware Black-Box Approximation**

cs.CV

**SubmitDate**: 2023-06-22    [abs](http://arxiv.org/abs/2306.13236v1) [paper-pdf](http://arxiv.org/pdf/2306.13236v1)

**Authors**: Ganesh Tata, Katyani Singh, Eric Van Oeveren, Nilanjan Ray

**Abstract**: Recent work has shown that by approximating the behaviour of a non-differentiable black-box function using a neural network, the black-box can be integrated into a differentiable training pipeline for end-to-end training. This methodology is termed "differentiable bypass,'' and a successful application of this method involves training a document preprocessor to improve the performance of a black-box OCR engine. However, a good approximation of an OCR engine requires querying it for all samples throughout the training process, which can be computationally and financially expensive. Several zeroth-order optimization (ZO) algorithms have been proposed in black-box attack literature to find adversarial examples for a black-box model by computing its gradient in a query-efficient manner. However, the query complexity and convergence rate of such algorithms makes them infeasible for our problem. In this work, we propose two sample selection algorithms to train an OCR preprocessor with less than 10% of the original system's OCR engine queries, resulting in more than 60% reduction of the total training time without significant loss of accuracy. We also show an improvement of 4% in the word-level accuracy of a commercial OCR engine with only 2.5% of the total queries and a 32x reduction in monetary cost. Further, we propose a simple ranking technique to prune 30% of the document images from the training dataset without affecting the system's performance.



## **44. Visual Adversarial Examples Jailbreak Large Language Models**

cs.CR

**SubmitDate**: 2023-06-22    [abs](http://arxiv.org/abs/2306.13213v1) [paper-pdf](http://arxiv.org/pdf/2306.13213v1)

**Authors**: Xiangyu Qi, Kaixuan Huang, Ashwinee Panda, Mengdi Wang, Prateek Mittal

**Abstract**: Recently, there has been a surge of interest in introducing vision into Large Language Models (LLMs). The proliferation of large Visual Language Models (VLMs), such as Flamingo, BLIP-2, and GPT-4, signifies an exciting convergence of advancements in both visual and language foundation models. Yet, the risks associated with this integrative approach are largely unexamined. In this paper, we shed light on the security and safety implications of this trend. First, we underscore that the continuous and high-dimensional nature of the additional visual input space intrinsically makes it a fertile ground for adversarial attacks. This unavoidably expands the attack surfaces of LLMs. Second, we highlight that the broad functionality of LLMs also presents visual attackers with a wider array of achievable adversarial objectives, extending the implications of security failures beyond mere misclassification. To elucidate these risks, we study adversarial examples in the visual input space of a VLM. Specifically, against MiniGPT-4, which incorporates safety mechanisms that can refuse harmful instructions, we present visual adversarial examples that can circumvent the safety mechanisms and provoke harmful behaviors of the model. Remarkably, we discover that adversarial examples, even if optimized on a narrow, manually curated derogatory corpus against specific social groups, can universally jailbreak the model's safety mechanisms. A single such adversarial example can generally undermine MiniGPT-4's safety, enabling it to heed a wide range of harmful instructions and produce harmful content far beyond simply imitating the derogatory corpus used in optimization. Unveiling these risks, we accentuate the urgent need for comprehensive risk assessments, robust defense strategies, and the implementation of responsible practices for the secure and safe utilization of VLMs.



## **45. Evading Forensic Classifiers with Attribute-Conditioned Adversarial Faces**

cs.CV

Accepted in CVPR 2023. Project page:  https://koushiksrivats.github.io/face_attribute_attack/

**SubmitDate**: 2023-06-22    [abs](http://arxiv.org/abs/2306.13091v1) [paper-pdf](http://arxiv.org/pdf/2306.13091v1)

**Authors**: Fahad Shamshad, Koushik Srivatsan, Karthik Nandakumar

**Abstract**: The ability of generative models to produce highly realistic synthetic face images has raised security and ethical concerns. As a first line of defense against such fake faces, deep learning based forensic classifiers have been developed. While these forensic models can detect whether a face image is synthetic or real with high accuracy, they are also vulnerable to adversarial attacks. Although such attacks can be highly successful in evading detection by forensic classifiers, they introduce visible noise patterns that are detectable through careful human scrutiny. Additionally, these attacks assume access to the target model(s) which may not always be true. Attempts have been made to directly perturb the latent space of GANs to produce adversarial fake faces that can circumvent forensic classifiers. In this work, we go one step further and show that it is possible to successfully generate adversarial fake faces with a specified set of attributes (e.g., hair color, eye size, race, gender, etc.). To achieve this goal, we leverage the state-of-the-art generative model StyleGAN with disentangled representations, which enables a range of modifications without leaving the manifold of natural images. We propose a framework to search for adversarial latent codes within the feature space of StyleGAN, where the search can be guided either by a text prompt or a reference image. We also propose a meta-learning based optimization strategy to achieve transferable performance on unknown target models. Extensive experiments demonstrate that the proposed approach can produce semantically manipulated adversarial fake faces, which are true to the specified attribute set and can successfully fool forensic face classifiers, while remaining undetectable by humans. Code: https://github.com/koushiksrivats/face_attribute_attack.



## **46. Impacts and Risk of Generative AI Technology on Cyber Defense**

cs.CR

**SubmitDate**: 2023-06-22    [abs](http://arxiv.org/abs/2306.13033v1) [paper-pdf](http://arxiv.org/pdf/2306.13033v1)

**Authors**: Subash Neupane, Ivan A. Fernandez, Sudip Mittal, Shahram Rahimi

**Abstract**: Generative Artificial Intelligence (GenAI) has emerged as a powerful technology capable of autonomously producing highly realistic content in various domains, such as text, images, audio, and videos. With its potential for positive applications in creative arts, content generation, virtual assistants, and data synthesis, GenAI has garnered significant attention and adoption. However, the increasing adoption of GenAI raises concerns about its potential misuse for crafting convincing phishing emails, generating disinformation through deepfake videos, and spreading misinformation via authentic-looking social media posts, posing a new set of challenges and risks in the realm of cybersecurity. To combat the threats posed by GenAI, we propose leveraging the Cyber Kill Chain (CKC) to understand the lifecycle of cyberattacks, as a foundational model for cyber defense. This paper aims to provide a comprehensive analysis of the risk areas introduced by the offensive use of GenAI techniques in each phase of the CKC framework. We also analyze the strategies employed by threat actors and examine their utilization throughout different phases of the CKC, highlighting the implications for cyber defense. Additionally, we propose GenAI-enabled defense strategies that are both attack-aware and adaptive. These strategies encompass various techniques such as detection, deception, and adversarial training, among others, aiming to effectively mitigate the risks posed by GenAI-induced cyber threats.



## **47. AI Security for Geoscience and Remote Sensing: Challenges and Future Trends**

cs.CV

**SubmitDate**: 2023-06-22    [abs](http://arxiv.org/abs/2212.09360v2) [paper-pdf](http://arxiv.org/pdf/2212.09360v2)

**Authors**: Yonghao Xu, Tao Bai, Weikang Yu, Shizhen Chang, Peter M. Atkinson, Pedram Ghamisi

**Abstract**: Recent advances in artificial intelligence (AI) have significantly intensified research in the geoscience and remote sensing (RS) field. AI algorithms, especially deep learning-based ones, have been developed and applied widely to RS data analysis. The successful application of AI covers almost all aspects of Earth observation (EO) missions, from low-level vision tasks like super-resolution, denoising and inpainting, to high-level vision tasks like scene classification, object detection and semantic segmentation. While AI techniques enable researchers to observe and understand the Earth more accurately, the vulnerability and uncertainty of AI models deserve further attention, considering that many geoscience and RS tasks are highly safety-critical. This paper reviews the current development of AI security in the geoscience and RS field, covering the following five important aspects: adversarial attack, backdoor attack, federated learning, uncertainty and explainability. Moreover, the potential opportunities and trends are discussed to provide insights for future research. To the best of the authors' knowledge, this paper is the first attempt to provide a systematic review of AI security-related research in the geoscience and RS community. Available code and datasets are also listed in the paper to move this vibrant field of research forward.



## **48. Robust Semantic Segmentation: Strong Adversarial Attacks and Fast Training of Robust Models**

cs.CV

**SubmitDate**: 2023-06-22    [abs](http://arxiv.org/abs/2306.12941v1) [paper-pdf](http://arxiv.org/pdf/2306.12941v1)

**Authors**: Francesco Croce, Naman D Singh, Matthias Hein

**Abstract**: While a large amount of work has focused on designing adversarial attacks against image classifiers, only a few methods exist to attack semantic segmentation models. We show that attacking segmentation models presents task-specific challenges, for which we propose novel solutions. Our final evaluation protocol outperforms existing methods, and shows that those can overestimate the robustness of the models. Additionally, so far adversarial training, the most successful way for obtaining robust image classifiers, could not be successfully applied to semantic segmentation. We argue that this is because the task to be learned is more challenging, and requires significantly higher computational effort than for image classification. As a remedy, we show that by taking advantage of recent advances in robust ImageNet classifiers, one can train adversarially robust segmentation models at limited computational cost by fine-tuning robust backbones.



## **49. Cross-lingual Cross-temporal Summarization: Dataset, Models, Evaluation**

cs.CL

Work in progress

**SubmitDate**: 2023-06-22    [abs](http://arxiv.org/abs/2306.12916v1) [paper-pdf](http://arxiv.org/pdf/2306.12916v1)

**Authors**: Ran Zhang, Jihed Ouni, Steffen Eger

**Abstract**: While summarization has been extensively researched in natural language processing (NLP), cross-lingual cross-temporal summarization (CLCTS) is a largely unexplored area that has the potential to improve cross-cultural accessibility, information sharing, and understanding. This paper comprehensively addresses the CLCTS task, including dataset creation, modeling, and evaluation. We build the first CLCTS corpus, leveraging historical fictive texts and Wikipedia summaries in English and German, and examine the effectiveness of popular transformer end-to-end models with different intermediate task finetuning tasks. Additionally, we explore the potential of ChatGPT for CLCTS as a summarizer and an evaluator. Overall, we report evaluations from humans, ChatGPT, and several recent automatic evaluation metrics where we find our intermediate task finetuned end-to-end models generate bad to moderate quality summaries; ChatGPT as a summarizer (without any finetuning) provides moderate to good quality outputs and as an evaluator correlates moderately with human evaluations though it is prone to giving lower scores. ChatGPT also seems to be very adept at normalizing historical text. We finally test ChatGPT in a scenario with adversarially attacked and unseen source documents and find that ChatGPT is better at omission and entity swap than negating against its prior knowledge.



## **50. On the explainable properties of 1-Lipschitz Neural Networks: An Optimal Transport Perspective**

cs.AI

**SubmitDate**: 2023-06-22    [abs](http://arxiv.org/abs/2206.06854v2) [paper-pdf](http://arxiv.org/pdf/2206.06854v2)

**Authors**: Mathieu Serrurier, Franck Mamalet, Thomas Fel, Louis Béthune, Thibaut Boissin

**Abstract**: Input gradients have a pivotal role in a variety of applications, including adversarial attack algorithms for evaluating model robustness, explainable AI techniques for generating Saliency Maps, and counterfactual explanations. However, Saliency Maps generated by traditional neural networks are often noisy and provide limited insights. In this paper, we demonstrate that, on the contrary, the Saliency Maps of 1-Lipschitz neural networks, learnt with the dual loss of an optimal transportation problem, exhibit desirable XAI properties: They are highly concentrated on the essential parts of the image with low noise, significantly outperforming state-of-the-art explanation approaches across various models and metrics. We also prove that these maps align unprecedentedly well with human explanations on ImageNet. To explain the particularly beneficial properties of the Saliency Map for such models, we prove this gradient encodes both the direction of the transportation plan and the direction towards the nearest adversarial attack. Following the gradient down to the decision boundary is no longer considered an adversarial attack, but rather a counterfactual explanation that explicitly transports the input from one class to another. Thus, Learning with such a loss jointly optimizes the classification objective and the alignment of the gradient , i.e. the Saliency Map, to the transportation plan direction. These networks were previously known to be certifiably robust by design, and we demonstrate that they scale well for large problems and models, and are tailored for explainability using a fast and straightforward method.



