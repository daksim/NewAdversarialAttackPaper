# Latest Adversarial Attack Papers
**update at 2022-02-20 22:21:28**

[中文版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Alexa versus Alexa: Controlling Smart Speakers by Self-Issuing Voice Commands**

cs.CR

15 pages, 5 figures, published in Proceedings of the 2022 ACM Asia  Conference on Computer and Communications Security (ASIA CCS '22)

**SubmitDate**: 2022-02-17    [paper-pdf](http://arxiv.org/pdf/2202.08619v1)

**Authors**: Sergio Esposito, Daniele Sgandurra, Giampaolo Bella

**Abstracts**: We present Alexa versus Alexa (AvA), a novel attack that leverages audio files containing voice commands and audio reproduction methods in an offensive fashion, to gain control of Amazon Echo devices for a prolonged amount of time. AvA leverages the fact that Alexa running on an Echo device correctly interprets voice commands originated from audio files even when they are played by the device itself -- i.e., it leverages a command self-issue vulnerability. Hence, AvA removes the necessity of having a rogue speaker in proximity of the victim's Echo, a constraint that many attacks share. With AvA, an attacker can self-issue any permissible command to Echo, controlling it on behalf of the legitimate user. We have verified that, via AvA, attackers can control smart appliances within the household, buy unwanted items, tamper linked calendars and eavesdrop on the user. We also discovered two additional Echo vulnerabilities, which we call Full Volume and Break Tag Chain. The Full Volume increases the self-issue command recognition rate, by doubling it on average, hence allowing attackers to perform additional self-issue commands. Break Tag Chain increases the time a skill can run without user interaction, from eight seconds to more than one hour, hence enabling attackers to setup realistic social engineering scenarios. By exploiting these vulnerabilities, the adversary can self-issue commands that are correctly executed 99% of the times and can keep control of the device for a prolonged amount of time. We reported these vulnerabilities to Amazon via their vulnerability research program, who rated them with a Medium severity score. Finally, to assess limitations of AvA on a larger scale, we provide the results of a survey performed on a study group of 18 users, and we show that most of the limitations against AvA are hardly used in practice.



## **2. Fingerprinting Deep Neural Networks Globally via Universal Adversarial Perturbations**

cs.CR

**SubmitDate**: 2022-02-17    [paper-pdf](http://arxiv.org/pdf/2202.08602v1)

**Authors**: Zirui Peng, Shaofeng Li, Guoxing Chen, Cheng Zhang, Haojin Zhu, Minhui Xue

**Abstracts**: In this paper, we propose a novel and practical mechanism which enables the service provider to verify whether a suspect model is stolen from the victim model via model extraction attacks. Our key insight is that the profile of a DNN model's decision boundary can be uniquely characterized by its \textit{Universal Adversarial Perturbations (UAPs)}. UAPs belong to a low-dimensional subspace and piracy models' subspaces are more consistent with victim model's subspace compared with non-piracy model. Based on this, we propose a UAP fingerprinting method for DNN models and train an encoder via \textit{contrastive learning} that takes fingerprint as inputs, outputs a similarity score. Extensive studies show that our framework can detect model IP breaches with confidence $> 99.99 \%$ within only $20$ fingerprints of the suspect model. It has good generalizability across different model architectures and is robust against post-modifications on stolen models.



## **3. Improving Robustness of Deep Reinforcement Learning Agents: Environment Attack based on the Critic Network**

cs.LG

**SubmitDate**: 2022-02-17    [paper-pdf](http://arxiv.org/pdf/2104.03154v2)

**Authors**: Lucas Schott, Hatem Hajri, Sylvain Lamprier

**Abstracts**: To improve policy robustness of deep reinforcement learning agents, a line of recent works focus on producing disturbances of the environment. Existing approaches of the literature to generate meaningful disturbances of the environment are adversarial reinforcement learning methods. These methods set the problem as a two-player game between the protagonist agent, which learns to perform a task in an environment, and the adversary agent, which learns to disturb the protagonist via modifications of the considered environment. Both protagonist and adversary are trained with deep reinforcement learning algorithms. Alternatively, we propose in this paper to build on gradient-based adversarial attacks, usually used for classification tasks for instance, that we apply on the critic network of the protagonist to identify efficient disturbances of the environment. Rather than learning an attacker policy, which usually reveals as very complex and unstable, we leverage the knowledge of the critic network of the protagonist, to dynamically complexify the task at each step of the learning process. We show that our method, while being faster and lighter, leads to significantly better improvements in policy robustness than existing methods of the literature.



## **4. GasHis-Transformer: A Multi-scale Visual Transformer Approach for Gastric Histopathology Image Classification**

cs.CV

**SubmitDate**: 2022-02-17    [paper-pdf](http://arxiv.org/pdf/2104.14528v6)

**Authors**: Haoyuan Chen, Chen Li, Ge Wang, Xiaoyan Li, Md Rahaman, Hongzan Sun, Weiming Hu, Yixin Li, Wanli Liu, Changhao Sun, Shiliang Ai, Marcin Grzegorzek

**Abstracts**: Existing deep learning methods for diagnosis of gastric cancer commonly use convolutional neural network. Recently, the Visual Transformer has attracted great attention because of its performance and efficiency, but its applications are mostly in the field of computer vision. In this paper, a multi-scale visual transformer model, referred to as GasHis-Transformer, is proposed for Gastric Histopathological Image Classification (GHIC), which enables the automatic classification of microscopic gastric images into abnormal and normal cases. The GasHis-Transformer model consists of two key modules: A global information module and a local information module to extract histopathological features effectively. In our experiments, a public hematoxylin and eosin (H&E) stained gastric histopathological dataset with 280 abnormal and normal images are divided into training, validation and test sets by a ratio of 1 : 1 : 2. The GasHis-Transformer model is applied to estimate precision, recall, F1-score and accuracy on the test set of gastric histopathological dataset as 98.0%, 100.0%, 96.0% and 98.0%, respectively. Furthermore, a critical study is conducted to evaluate the robustness of GasHis-Transformer, where ten different noises including four adversarial attack and six conventional image noises are added. In addition, a clinically meaningful study is executed to test the gastrointestinal cancer identification performance of GasHis-Transformer with 620 abnormal images and achieves 96.8% accuracy. Finally, a comparative study is performed to test the generalizability with both H&E and immunohistochemical stained images on a lymphoma image dataset and a breast cancer dataset, producing comparable F1-scores (85.6% and 82.8%) and accuracies (83.9% and 89.4%), respectively. In conclusion, GasHisTransformer demonstrates high classification performance and shows its significant potential in the GHIC task.



## **5. Measuring the Transferability of $\ell_\infty$ Attacks by the $\ell_2$ Norm**

cs.LG

**SubmitDate**: 2022-02-17    [paper-pdf](http://arxiv.org/pdf/2102.10343v3)

**Authors**: Sizhe Chen, Qinghua Tao, Zhixing Ye, Xiaolin Huang

**Abstracts**: Deep neural networks could be fooled by adversarial examples with trivial differences to original samples. To keep the difference imperceptible in human eyes, researchers bound the adversarial perturbations by the $\ell_\infty$ norm, which is now commonly served as the standard to align the strength of different attacks for a fair comparison. However, we propose that using the $\ell_\infty$ norm alone is not sufficient in measuring the attack strength, because even with a fixed $\ell_\infty$ distance, the $\ell_2$ distance also greatly affects the attack transferability between models. Through the discovery, we reach more in-depth understandings towards the attack mechanism, i.e., several existing methods attack black-box models better partly because they craft perturbations with 70\% to 130\% larger $\ell_2$ distances. Since larger perturbations naturally lead to better transferability, we thereby advocate that the strength of attacks should be simultaneously measured by both the $\ell_\infty$ and $\ell_2$ norm. Our proposal is firmly supported by extensive experiments on ImageNet dataset from 7 attacks, 4 white-box models, and 9 black-box models.



## **6. Towards Evaluating the Robustness of Neural Networks Learned by Transduction**

cs.LG

Paper published at ICLR 2022. arXiv admin note: text overlap with  arXiv:2106.08387

**SubmitDate**: 2022-02-17    [paper-pdf](http://arxiv.org/pdf/2110.14735v2)

**Authors**: Jiefeng Chen, Xi Wu, Yang Guo, Yingyu Liang, Somesh Jha

**Abstracts**: There has been emerging interest in using transductive learning for adversarial robustness (Goldwasser et al., NeurIPS 2020; Wu et al., ICML 2020; Wang et al., ArXiv 2021). Compared to traditional defenses, these defense mechanisms "dynamically learn" the model based on test-time input; and theoretically, attacking these defenses reduces to solving a bilevel optimization problem, which poses difficulty in crafting adaptive attacks. In this paper, we examine these defense mechanisms from a principled threat analysis perspective. We formulate and analyze threat models for transductive-learning based defenses, and point out important subtleties. We propose the principle of attacking model space for solving bilevel attack objectives, and present Greedy Model Space Attack (GMSA), an attack framework that can serve as a new baseline for evaluating transductive-learning based defenses. Through systematic evaluation, we show that GMSA, even with weak instantiations, can break previous transductive-learning based defenses, which were resilient to previous attacks, such as AutoAttack. On the positive side, we report a somewhat surprising empirical result of "transductive adversarial training": Adversarially retraining the model using fresh randomness at the test time gives a significant increase in robustness against attacks we consider.



## **7. Generalizable Information Theoretic Causal Representation**

cs.LG

**SubmitDate**: 2022-02-17    [paper-pdf](http://arxiv.org/pdf/2202.08388v1)

**Authors**: Mengyue Yang, Xinyu Cai, Furui Liu, Xu Chen, Zhitang Chen, Jianye Hao, Jun Wang

**Abstracts**: It is evidence that representation learning can improve model's performance over multiple downstream tasks in many real-world scenarios, such as image classification and recommender systems. Existing learning approaches rely on establishing the correlation (or its proxy) between features and the downstream task (labels), which typically results in a representation containing cause, effect and spurious correlated variables of the label. Its generalizability may deteriorate because of the unstability of the non-causal parts. In this paper, we propose to learn causal representation from observational data by regularizing the learning procedure with mutual information measures according to our hypothetical causal graph. The optimization involves a counterfactual loss, based on which we deduce a theoretical guarantee that the causality-inspired learning is with reduced sample complexity and better generalization ability. Extensive experiments show that the models trained on causal representations learned by our approach is robust under adversarial attacks and distribution shift.



## **8. Characterizing Attacks on Deep Reinforcement Learning**

cs.LG

AAMAS 2022, 13 pages, 6 figures

**SubmitDate**: 2022-02-16    [paper-pdf](http://arxiv.org/pdf/1907.09470v3)

**Authors**: Xinlei Pan, Chaowei Xiao, Warren He, Shuang Yang, Jian Peng, Mingjie Sun, Jinfeng Yi, Zijiang Yang, Mingyan Liu, Bo Li, Dawn Song

**Abstracts**: Recent studies show that Deep Reinforcement Learning (DRL) models are vulnerable to adversarial attacks, which attack DRL models by adding small perturbations to the observations. However, some attacks assume full availability of the victim model, and some require a huge amount of computation, making them less feasible for real world applications. In this work, we make further explorations of the vulnerabilities of DRL by studying other aspects of attacks on DRL using realistic and efficient attacks. First, we adapt and propose efficient black-box attacks when we do not have access to DRL model parameters. Second, to address the high computational demands of existing attacks, we introduce efficient online sequential attacks that exploit temporal consistency across consecutive steps. Third, we explore the possibility of an attacker perturbing other aspects in the DRL setting, such as the environment dynamics. Finally, to account for imperfections in how an attacker would inject perturbations in the physical world, we devise a method for generating a robust physical perturbations to be printed. The attack is evaluated on a real-world robot under various conditions. We conduct extensive experiments both in simulation such as Atari games, robotics and autonomous driving, and on real-world robotics, to compare the effectiveness of the proposed attacks with baseline approaches. To the best of our knowledge, we are the first to apply adversarial attacks on DRL systems to physical robots.



## **9. Real-Time Neural Voice Camouflage**

cs.SD

14 pages

**SubmitDate**: 2022-02-16    [paper-pdf](http://arxiv.org/pdf/2112.07076v2)

**Authors**: Mia Chiquier, Chengzhi Mao, Carl Vondrick

**Abstracts**: Automatic speech recognition systems have created exciting possibilities for applications, however they also enable opportunities for systematic eavesdropping. We propose a method to camouflage a person's voice over-the-air from these systems without inconveniencing the conversation between people in the room. Standard adversarial attacks are not effective in real-time streaming situations because the characteristics of the signal will have changed by the time the attack is executed. We introduce predictive attacks, which achieve real-time performance by forecasting the attack that will be the most effective in the future. Under real-time constraints, our method jams the established speech recognition system DeepSpeech 3.9x more than baselines as measured through word error rate, and 6.6x more as measured through character error rate. We furthermore demonstrate our approach is practically effective in realistic environments over physical distances.



## **10. Ideal Tightly Couple (t,m,n) Secret Sharing**

cs.CR

few errors in the articles within the proposed scheme, and also  grammatical errors, so its our request pls withdraw our articles as soon as  possible

**SubmitDate**: 2022-02-16    [paper-pdf](http://arxiv.org/pdf/1905.02004v2)

**Authors**: Fuyou Miao, Keju Meng, Wenchao Huang, Yan Xiong, Xingfu Wang

**Abstracts**: As a fundamental cryptographic tool, (t,n)-threshold secret sharing ((t,n)-SS) divides a secret among n shareholders and requires at least t, (t<=n), of them to reconstruct the secret. Ideal (t,n)-SSs are most desirable in security and efficiency among basic (t,n)-SSs. However, an adversary, even without any valid share, may mount Illegal Participant (IP) attack or t/2-Private Channel Cracking (t/2-PCC) attack to obtain the secret in most (t,n)-SSs.To secure ideal (t,n)-SSs against the 2 attacks, 1) the paper introduces the notion of Ideal Tightly cOupled (t,m,n) Secret Sharing (or (t,m,n)-ITOSS ) to thwart IP attack without Verifiable SS; (t,m,n)-ITOSS binds all m, (m>=t), participants into a tightly coupled group and requires all participants to be legal shareholders before recovering the secret. 2) As an example, the paper presents a polynomial-based (t,m,n)-ITOSS scheme, in which the proposed k-round Random Number Selection (RNS) guarantees that adversaries have to crack at least symmetrical private channels among participants before obtaining the secret. Therefore, k-round RNS enhances the robustness of (t,m,n)-ITOSS against t/2-PCC attack to the utmost. 3) The paper finally presents a generalized method of converting an ideal (t,n)-SS into a (t,m,n)-ITOSS, which helps an ideal (t,n)-SS substantially improve the robustness against the above 2 attacks.



## **11. Deduplicating Training Data Mitigates Privacy Risks in Language Models**

cs.CR

**SubmitDate**: 2022-02-16    [paper-pdf](http://arxiv.org/pdf/2202.06539v2)

**Authors**: Nikhil Kandpal, Eric Wallace, Colin Raffel

**Abstracts**: Past work has shown that large language models are susceptible to privacy attacks, where adversaries generate sequences from a trained model and detect which sequences are memorized from the training set. In this work, we show that the success of these attacks is largely due to duplication in commonly used web-scraped training sets. We first show that the rate at which language models regenerate training sequences is superlinearly related to a sequence's count in the training set. For instance, a sequence that is present 10 times in the training data is on average generated ~1000 times more often than a sequence that is present only once. We next show that existing methods for detecting memorized sequences have near-chance accuracy on non-duplicated training sequences. Finally, we find that after applying methods to deduplicate training data, language models are considerably more secure against these types of privacy attacks. Taken together, our results motivate an increased focus on deduplication in privacy-sensitive applications and a reevaluation of the practicality of existing privacy attacks.



## **12. The Adversarial Security Mitigations of mmWave Beamforming Prediction Models using Defensive Distillation and Adversarial Retraining**

cs.CR

26 pages, under review

**SubmitDate**: 2022-02-16    [paper-pdf](http://arxiv.org/pdf/2202.08185v1)

**Authors**: Murat Kuzlu, Ferhat Ozgur Catak, Umit Cali, Evren Catak, Ozgur Guler

**Abstracts**: The design of a security scheme for beamforming prediction is critical for next-generation wireless networks (5G, 6G, and beyond). However, there is no consensus about protecting the beamforming prediction using deep learning algorithms in these networks. This paper presents the security vulnerabilities in deep learning for beamforming prediction using deep neural networks (DNNs) in 6G wireless networks, which treats the beamforming prediction as a multi-output regression problem. It is indicated that the initial DNN model is vulnerable against adversarial attacks, such as Fast Gradient Sign Method (FGSM), Basic Iterative Method (BIM), Projected Gradient Descent (PGD), and Momentum Iterative Method (MIM), because the initial DNN model is sensitive to the perturbations of the adversarial samples of the training data. This study also offers two mitigation methods, such as adversarial training and defensive distillation, for adversarial attacks against artificial intelligence (AI)-based models used in the millimeter-wave (mmWave) beamforming prediction. Furthermore, the proposed scheme can be used in situations where the data are corrupted due to the adversarial examples in the training data. Experimental results show that the proposed methods effectively defend the DNN models against adversarial attacks in next-generation wireless networks.



## **13. Finding Dynamics Preserving Adversarial Winning Tickets**

cs.LG

Accepted by AISTATS2022

**SubmitDate**: 2022-02-16    [paper-pdf](http://arxiv.org/pdf/2202.06488v2)

**Authors**: Xupeng Shi, Pengfei Zheng, A. Adam Ding, Yuan Gao, Weizhong Zhang

**Abstracts**: Modern deep neural networks (DNNs) are vulnerable to adversarial attacks and adversarial training has been shown to be a promising method for improving the adversarial robustness of DNNs. Pruning methods have been considered in adversarial context to reduce model capacity and improve adversarial robustness simultaneously in training. Existing adversarial pruning methods generally mimic the classical pruning methods for natural training, which follow the three-stage 'training-pruning-fine-tuning' pipelines. We observe that such pruning methods do not necessarily preserve the dynamics of dense networks, making it potentially hard to be fine-tuned to compensate the accuracy degradation in pruning. Based on recent works of \textit{Neural Tangent Kernel} (NTK), we systematically study the dynamics of adversarial training and prove the existence of trainable sparse sub-network at initialization which can be trained to be adversarial robust from scratch. This theoretically verifies the \textit{lottery ticket hypothesis} in adversarial context and we refer such sub-network structure as \textit{Adversarial Winning Ticket} (AWT). We also show empirical evidences that AWT preserves the dynamics of adversarial training and achieve equal performance as dense adversarial training.



## **14. Neural Network Trojans Analysis and Mitigation from the Input Domain**

cs.LG

**SubmitDate**: 2022-02-16    [paper-pdf](http://arxiv.org/pdf/2202.06382v2)

**Authors**: Zhenting Wang, Hailun Ding, Juan Zhai, Shiqing Ma

**Abstracts**: Deep Neural Networks (DNNs) can learn Trojans (or backdoors) from benign or poisoned data, which raises security concerns of using them. By exploiting such Trojans, the adversary can add a fixed input space perturbation to any given input to mislead the model predicting certain outputs (i.e., target labels). In this paper, we analyze such input space Trojans in DNNs, and propose a theory to explain the relationship of a model's decision regions and Trojans: a complete and accurate Trojan corresponds to a hyperplane decision region in the input domain. We provide a formal proof of this theory, and provide empirical evidence to support the theory and its relaxations. Based on our analysis, we design a novel training method that removes Trojans during training even on poisoned datasets, and evaluate our prototype on five datasets and five different attacks. Results show that our method outperforms existing solutions. Code: \url{https://anonymous.4open.science/r/NOLE-84C3}.



## **15. Understanding and Improving Graph Injection Attack by Promoting Unnoticeability**

cs.LG

ICLR2022

**SubmitDate**: 2022-02-16    [paper-pdf](http://arxiv.org/pdf/2202.08057v1)

**Authors**: Yongqiang Chen, Han Yang, Yonggang Zhang, Kaili Ma, Tongliang Liu, Bo Han, James Cheng

**Abstracts**: Recently Graph Injection Attack (GIA) emerges as a practical attack scenario on Graph Neural Networks (GNNs), where the adversary can merely inject few malicious nodes instead of modifying existing nodes or edges, i.e., Graph Modification Attack (GMA). Although GIA has achieved promising results, little is known about why it is successful and whether there is any pitfall behind the success. To understand the power of GIA, we compare it with GMA and find that GIA can be provably more harmful than GMA due to its relatively high flexibility. However, the high flexibility will also lead to great damage to the homophily distribution of the original graph, i.e., similarity among neighbors. Consequently, the threats of GIA can be easily alleviated or even prevented by homophily-based defenses designed to recover the original homophily. To mitigate the issue, we introduce a novel constraint -- homophily unnoticeability that enforces GIA to preserve the homophily, and propose Harmonious Adversarial Objective (HAO) to instantiate it. Extensive experiments verify that GIA with HAO can break homophily-based defenses and outperform previous GIA attacks by a significant margin. We believe our methods can serve for a more reliable evaluation of the robustness of GNNs.



## **16. Increasing-Margin Adversarial (IMA) Training to Improve Adversarial Robustness of Neural Networks**

cs.CV

45 pages, 15 figures, 31 tables

**SubmitDate**: 2022-02-16    [paper-pdf](http://arxiv.org/pdf/2005.09147v7)

**Authors**: Linhai Ma, Liang Liang

**Abstracts**: Convolutional neural network (CNN) has surpassed traditional methods for medical image classification. However, CNN is vulnerable to adversarial attacks which may lead to disastrous consequences in medical applications. Although adversarial noises are usually generated by attack algorithms, white-noise-induced adversarial samples can exist, and therefore the threats are real. In this study, we propose a novel training method, named IMA, to improve the robust-ness of CNN against adversarial noises. During training, the IMA method increases the margins of training samples in the input space, i.e., moving CNN decision boundaries far away from the training samples to improve robustness. The IMA method is evaluated on publicly available datasets under strong 100-PGD white-box adversarial attacks, and the results show that the proposed method significantly improved CNN classification and segmentation accuracy on noisy data while keeping a high accuracy on clean data. We hope our approach may facilitate the development of robust applications in medical field.



## **17. Backdoor Learning: A Survey**

cs.CR

17 pages. A curated list of backdoor learning resources in this paper  is presented in the Github Repo  (https://github.com/THUYimingLi/backdoor-learning-resources). We will try our  best to continuously maintain this Github Repo

**SubmitDate**: 2022-02-16    [paper-pdf](http://arxiv.org/pdf/2007.08745v5)

**Authors**: Yiming Li, Yong Jiang, Zhifeng Li, Shu-Tao Xia

**Abstracts**: Backdoor attack intends to embed hidden backdoor into deep neural networks (DNNs), so that the attacked models perform well on benign samples, whereas their predictions will be maliciously changed if the hidden backdoor is activated by attacker-specified triggers. This threat could happen when the training process is not fully controlled, such as training on third-party datasets or adopting third-party models, which poses a new and realistic threat. Although backdoor learning is an emerging and rapidly growing research area, its systematic review, however, remains blank. In this paper, we present the first comprehensive survey of this realm. We summarize and categorize existing backdoor attacks and defenses based on their characteristics, and provide a unified framework for analyzing poisoning-based backdoor attacks. Besides, we also analyze the relation between backdoor attacks and relevant fields ($i.e.,$ adversarial attacks and data poisoning), and summarize widely adopted benchmark datasets. Finally, we briefly outline certain future research directions relying upon reviewed works. A curated list of backdoor-related resources is also available at \url{https://github.com/THUYimingLi/backdoor-learning-resources}.



## **18. FedCG: Leverage Conditional GAN for Protecting Privacy and Maintaining Competitive Performance in Federated Learning**

cs.LG

**SubmitDate**: 2022-02-16    [paper-pdf](http://arxiv.org/pdf/2111.08211v2)

**Authors**: Yuezhou Wu, Yan Kang, Jiahuan Luo, Yuanqin He, Qiang Yang

**Abstracts**: Federated learning (FL) aims to protect data privacy by enabling clients to build machine learning models collaboratively without sharing their private data. Recent works demonstrate that information exchanged during FL is subject to gradient-based privacy attacks and, consequently, a variety of privacy-preserving methods have been adopted to thwart such attacks. However, these defensive methods either introduce orders of magnitudes more computational and communication overheads (e.g., with homomorphic encryption) or incur substantial model performance losses in terms of prediction accuracy (e.g., with differential privacy). In this work, we propose $\textsc{FedCG}$, a novel federated learning method that leverages conditional generative adversarial networks to achieve high-level privacy protection while still maintaining competitive model performance. $\textsc{FedCG}$ decomposes each client's local network into a private extractor and a public classifier and keeps the extractor local to protect privacy. Instead of exposing extractors, $\textsc{FedCG}$ shares clients' generators with the server for aggregating clients' shared knowledge aiming to enhance the performance of each client's local networks. Extensive experiments demonstrate that $\textsc{FedCG}$ can achieve competitive model performance compared with FL baselines, and privacy analysis shows that $\textsc{FedCG}$ has a high-level privacy-preserving capability.



## **19. Generative Adversarial Network-Driven Detection of Adversarial Tasks in Mobile Crowdsensing**

cs.CR

This paper contains pages, 4 figures which is accepted by IEEE ICC  2022

**SubmitDate**: 2022-02-16    [paper-pdf](http://arxiv.org/pdf/2202.07802v1)

**Authors**: Zhiyan Chen, Burak Kantarci

**Abstracts**: Mobile Crowdsensing systems are vulnerable to various attacks as they build on non-dedicated and ubiquitous properties. Machine learning (ML)-based approaches are widely investigated to build attack detection systems and ensure MCS systems security. However, adversaries that aim to clog the sensing front-end and MCS back-end leverage intelligent techniques, which are challenging for MCS platform and service providers to develop appropriate detection frameworks against these attacks. Generative Adversarial Networks (GANs) have been applied to generate synthetic samples, that are extremely similar to the real ones, deceiving classifiers such that the synthetic samples are indistinguishable from the originals. Previous works suggest that GAN-based attacks exhibit more crucial devastation than empirically designed attack samples, and result in low detection rate at the MCS platform. With this in mind, this paper aims to detect intelligently designed illegitimate sensing service requests by integrating a GAN-based model. To this end, we propose a two-level cascading classifier that combines the GAN discriminator with a binary classifier to prevent adversarial fake tasks. Through simulations, we compare our results to a single-level binary classifier, and the numeric results show that proposed approach raises Adversarial Attack Detection Rate (AADR), from $0\%$ to $97.5\%$ by KNN/NB, from $45.9\%$ to $100\%$ by Decision Tree. Meanwhile, with two-levels classifiers, Original Attack Detection Rate (OADR) improves for the three binary classifiers, with comparison, such as NB from $26.1\%$ to $61.5\%$.



## **20. Vulnerability-Aware Poisoning Mechanism for Online RL with Unknown Dynamics**

cs.LG

**SubmitDate**: 2022-02-15    [paper-pdf](http://arxiv.org/pdf/2009.00774v5)

**Authors**: Yanchao Sun, Da Huo, Furong Huang

**Abstracts**: Poisoning attacks on Reinforcement Learning (RL) systems could take advantage of RL algorithm's vulnerabilities and cause failure of the learning. However, prior works on poisoning RL usually either unrealistically assume the attacker knows the underlying Markov Decision Process (MDP), or directly apply the poisoning methods in supervised learning to RL. In this work, we build a generic poisoning framework for online RL via a comprehensive investigation of heterogeneous poisoning models in RL. Without any prior knowledge of the MDP, we propose a strategic poisoning algorithm called Vulnerability-Aware Adversarial Critic Poison (VA2C-P), which works for most policy-based deep RL agents, closing the gap that no poisoning method exists for policy-based RL agents. VA2C-P uses a novel metric, stability radius in RL, that measures the vulnerability of RL algorithms. Experiments on multiple deep RL agents and multiple environments show that our poisoning algorithm successfully prevents agents from learning a good policy or teaches the agents to converge to a target policy, with a limited attacking budget.



## **21. Defending against Reconstruction Attacks with Rényi Differential Privacy**

cs.LG

**SubmitDate**: 2022-02-15    [paper-pdf](http://arxiv.org/pdf/2202.07623v1)

**Authors**: Pierre Stock, Igor Shilov, Ilya Mironov, Alexandre Sablayrolles

**Abstracts**: Reconstruction attacks allow an adversary to regenerate data samples of the training set using access to only a trained model. It has been recently shown that simple heuristics can reconstruct data samples from language models, making this threat scenario an important aspect of model release. Differential privacy is a known solution to such attacks, but is often used with a relatively large privacy budget (epsilon > 8) which does not translate to meaningful guarantees. In this paper we show that, for a same mechanism, we can derive privacy guarantees for reconstruction attacks that are better than the traditional ones from the literature. In particular, we show that larger privacy budgets do not protect against membership inference, but can still protect extraction of rare secrets. We show experimentally that our guarantees hold against various language models, including GPT-2 finetuned on Wikitext-103.



## **22. StratDef: a strategic defense against adversarial attacks in malware detection**

cs.LG

**SubmitDate**: 2022-02-15    [paper-pdf](http://arxiv.org/pdf/2202.07568v1)

**Authors**: Aqib Rashid, Jose Such

**Abstracts**: Over the years, most research towards defenses against adversarial attacks on machine learning models has been in the image processing domain. The malware detection domain has received less attention despite its importance. Moreover, most work exploring defenses focuses on feature-based, gradient-based or randomized methods but with no strategy when applying them. In this paper, we introduce StratDef, which is a strategic defense system tailored for the malware detection domain based on a Moving Target Defense and Game Theory approach. We overcome challenges related to the systematic construction, selection and strategic use of models to maximize adversarial robustness. StratDef dynamically and strategically chooses the best models to increase the uncertainty for the attacker, whilst minimizing critical aspects in the adversarial ML domain like attack transferability. We provide the first comprehensive evaluation of defenses against adversarial attacks on machine learning for malware detection, where our threat model explores different levels of threat, attacker knowledge, capabilities, and attack intensities. We show that StratDef performs better than other defenses even when facing the peak adversarial threat. We also show that, from the existing defenses, only a few adversarially-trained models provide substantially better protection than just using vanilla models but are still outperformed by StratDef.



## **23. Random Walks for Adversarial Meshes**

cs.CV

**SubmitDate**: 2022-02-15    [paper-pdf](http://arxiv.org/pdf/2202.07453v1)

**Authors**: Amir Belder, Gal Yefet, Ran Ben Izhak, Ayellet Tal

**Abstracts**: A polygonal mesh is the most-commonly used representation of surfaces in computer graphics; thus, a variety of classification networks have been recently proposed. However, while adversarial attacks are wildly researched in 2D, almost no works on adversarial meshes exist. This paper proposes a novel, unified, and general adversarial attack, which leads to misclassification of numerous state-of-the-art mesh classification neural networks. Our attack approach is black-box, i.e. it has access only to the network's predictions, but not to the network's full architecture or gradients. The key idea is to train a network to imitate a given classification network. This is done by utilizing random walks along the mesh surface, which gather geometric information. These walks provide insight onto the regions of the mesh that are important for the correct prediction of the given classification network. These mesh regions are then modified more than other regions in order to attack the network in a manner that is barely visible to the naked eye.



## **24. Unreasonable Effectiveness of Last Hidden Layer Activations**

cs.LG

22 pages, Under review

**SubmitDate**: 2022-02-15    [paper-pdf](http://arxiv.org/pdf/2202.07342v1)

**Authors**: Omer Faruk Tuna, Ferhat Ozgur Catak, M. Taner Eskil

**Abstracts**: In standard Deep Neural Network (DNN) based classifiers, the general convention is to omit the activation function in the last (output) layer and directly apply the softmax function on the logits to get the probability scores of each class. In this type of architectures, the loss value of the classifier against any output class is directly proportional to the difference between the final probability score and the label value of the associated class. Standard White-box adversarial evasion attacks, whether targeted or untargeted, mainly try to exploit the gradient of the model loss function to craft adversarial samples and fool the model. In this study, we show both mathematically and experimentally that using some widely known activation functions in the output layer of the model with high temperature values has the effect of zeroing out the gradients for both targeted and untargeted attack cases, preventing attackers from exploiting the model's loss function to craft adversarial samples. We've experimentally verified the efficacy of our approach on MNIST (Digit), CIFAR10 datasets. Detailed experiments confirmed that our approach substantially improves robustness against gradient-based targeted and untargeted attack threats. And, we showed that the increased non-linearity at the output layer has some additional benefits against some other attack methods like Deepfool attack.



## **25. Unity is strength: Improving the Detection of Adversarial Examples with Ensemble Approaches**

cs.CV

Code is available at https://github.com/BIMIB-DISCo/ENAD-experiments

**SubmitDate**: 2022-02-15    [paper-pdf](http://arxiv.org/pdf/2111.12631v3)

**Authors**: Francesco Craighero, Fabrizio Angaroni, Fabio Stella, Chiara Damiani, Marco Antoniotti, Alex Graudenzi

**Abstracts**: A key challenge in computer vision and deep learning is the definition of robust strategies for the detection of adversarial examples. Here, we propose the adoption of ensemble approaches to leverage the effectiveness of multiple detectors in exploiting distinct properties of the input data. To this end, the ENsemble Adversarial Detector (ENAD) framework integrates scoring functions from state-of-the-art detectors based on Mahalanobis distance, Local Intrinsic Dimensionality, and One-Class Support Vector Machines, which process the hidden features of deep neural networks. ENAD is designed to ensure high standardization and reproducibility to the computational workflow. Importantly, extensive tests on benchmark datasets, models and adversarial attacks show that ENAD outperforms all competing methods in the large majority of settings. The improvement over the state-of-the-art and the intrinsic generality of the framework, which allows one to easily extend ENAD to include any set of detectors, set the foundations for the new area of ensemble adversarial detection.



## **26. Layer-wise Regularized Adversarial Training using Layers Sustainability Analysis (LSA) framework**

cs.CV

Layers Sustainability Analysis (LSA) framework

**SubmitDate**: 2022-02-15    [paper-pdf](http://arxiv.org/pdf/2202.02626v3)

**Authors**: Mohammad Khalooei, Mohammad Mehdi Homayounpour, Maryam Amirmazlaghani

**Abstracts**: Deep neural network models are used today in various applications of artificial intelligence, the strengthening of which, in the face of adversarial attacks is of particular importance. An appropriate solution to adversarial attacks is adversarial training, which reaches a trade-off between robustness and generalization. This paper introduces a novel framework (Layer Sustainability Analysis (LSA)) for the analysis of layer vulnerability in an arbitrary neural network in the scenario of adversarial attacks. LSA can be a helpful toolkit to assess deep neural networks and to extend the adversarial training approaches towards improving the sustainability of model layers via layer monitoring and analysis. The LSA framework identifies a list of Most Vulnerable Layers (MVL list) of the given network. The relative error, as a comparison measure, is used to evaluate representation sustainability of each layer against adversarial inputs. The proposed approach for obtaining robust neural networks to fend off adversarial attacks is based on a layer-wise regularization (LR) over LSA proposal(s) for adversarial training (AT); i.e. the AT-LR procedure. AT-LR could be used with any benchmark adversarial attack to reduce the vulnerability of network layers and to improve conventional adversarial training approaches. The proposed idea performs well theoretically and experimentally for state-of-the-art multilayer perceptron and convolutional neural network architectures. Compared with the AT-LR and its corresponding base adversarial training, the classification accuracy of more significant perturbations increased by 16.35%, 21.79%, and 10.730% on Moon, MNIST, and CIFAR-10 benchmark datasets, respectively. The LSA framework is available and published at https://github.com/khalooei/LSA.



## **27. Holistic Adversarial Robustness of Deep Learning Models**

cs.LG

survey paper on holistic adversarial robustness for deep learning

**SubmitDate**: 2022-02-15    [paper-pdf](http://arxiv.org/pdf/2202.07201v1)

**Authors**: Pin-Yu Chen, Sijia Liu

**Abstracts**: Adversarial robustness studies the worst-case performance of a machine learning model to ensure safety and reliability. With the proliferation of deep-learning based technology, the potential risks associated with model development and deployment can be amplified and become dreadful vulnerabilities. This paper provides a comprehensive overview of research topics and foundational principles of research methods for adversarial robustness of deep learning models, including attacks, defenses, verification, and novel applications.



## **28. Resilience from Diversity: Population-based approach to harden models against adversarial attacks**

cs.LG

12 pages, 6 figures, 5 tables

**SubmitDate**: 2022-02-15    [paper-pdf](http://arxiv.org/pdf/2111.10272v2)

**Authors**: Jasser Jasser, Ivan Garibay

**Abstracts**: Traditional deep learning networks (DNN) exhibit intriguing vulnerabilities that allow an attacker to force them to fail at their task. Notorious attacks such as the Fast Gradient Sign Method (FGSM) and the more powerful Projected Gradient Descent (PGD) generate adversarial samples by adding a magnitude of perturbation $\epsilon$ to the input's computed gradient, resulting in a deterioration of the effectiveness of the model's classification. This work introduces a model that is resilient to adversarial attacks. Our model leverages an established mechanism of defense which utilizes randomness and a population of DNNs. More precisely, our model consists of a population of $n$ diverse submodels, each one of them trained to individually obtain a high accuracy for the task at hand, while forced to maintain meaningful differences in their weights. Each time our model receives a classification query, it selects a submodel from its population at random to answer the query. To counter the attack transferability, diversity is introduced and maintained in the population of submodels. Thus introducing the concept of counter linking weights. A Counter-Linked Model (CLM) consists of a population of DNNs of the same architecture where a periodic random similarity examination is conducted during the simultaneous training to guarantee diversity while maintaining accuracy. Though the randomization technique proved to be resilient against adversarial attacks, we show that by retraining the DNNs ensemble or training them from the start with counter linking would enhance the robustness by around 20\% when tested on the MNIST dataset and at least 15\% when tested on the CIFAR-10 dataset. When CLM is coupled with adversarial training, this defense mechanism achieves state-of-the-art robustness.



## **29. Recent Advances in Reliable Deep Graph Learning: Adversarial Attack, Inherent Noise, and Distribution Shift**

cs.LG

**SubmitDate**: 2022-02-15    [paper-pdf](http://arxiv.org/pdf/2202.07114v1)

**Authors**: Bingzhe Wu, Jintang Li, Chengbin Hou, Guoji Fu, Yatao Bian, Liang Chen, Junzhou Huang

**Abstracts**: Deep graph learning (DGL) has achieved remarkable progress in both business and scientific areas ranging from finance and e-commerce to drug and advanced material discovery. Despite the progress, applying DGL to real-world applications faces a series of reliability threats including adversarial attacks, inherent noise, and distribution shift. This survey aims to provide a comprehensive review of recent advances for improving the reliability of DGL algorithms against the above threats. In contrast to prior related surveys which mainly focus on adversarial attacks and defense, our survey covers more reliability-related aspects of DGL, i.e., inherent noise and distribution shift. Additionally, we discuss the relationships among above aspects and highlight some important issues to be explored in future research.



## **30. Universal Adversarial Examples in Remote Sensing: Methodology and Benchmark**

cs.CV

**SubmitDate**: 2022-02-14    [paper-pdf](http://arxiv.org/pdf/2202.07054v1)

**Authors**: Yonghao Xu, Pedram Ghamisi

**Abstracts**: Deep neural networks have achieved great success in many important remote sensing tasks. Nevertheless, their vulnerability to adversarial examples should not be neglected. In this study, we systematically analyze the universal adversarial examples in remote sensing data for the first time, without any knowledge from the victim model. Specifically, we propose a novel black-box adversarial attack method, namely Mixup-Attack, and its simple variant Mixcut-Attack, for remote sensing data. The key idea of the proposed methods is to find common vulnerabilities among different networks by attacking the features in the shallow layer of a given surrogate model. Despite their simplicity, the proposed methods can generate transferable adversarial examples that deceive most of the state-of-the-art deep neural networks in both scene classification and semantic segmentation tasks with high success rates. We further provide the generated universal adversarial examples in the dataset named UAE-RS, which is the first dataset that provides black-box adversarial samples in the remote sensing field. We hope UAE-RS may serve as a benchmark that helps researchers to design deep neural networks with strong resistance toward adversarial attacks in the remote sensing field. Codes and the UAE-RS dataset will be available online.



## **31. White-Box Attacks on Hate-speech BERT Classifiers in German with Explicit and Implicit Character Level Defense**

cs.CL

**SubmitDate**: 2022-02-14    [paper-pdf](http://arxiv.org/pdf/2202.05778v2)

**Authors**: Shahrukh Khan, Mahnoor Shahid, Navdeeppal Singh

**Abstracts**: In this work, we evaluate the adversarial robustness of BERT models trained on German Hate Speech datasets. We also complement our evaluation with two novel white-box character and word level attacks thereby contributing to the range of attacks available. Furthermore, we also perform a comparison of two novel character-level defense strategies and evaluate their robustness with one another.



## **32. Robust and Information-theoretically Safe Bias Classifier against Adversarial Attacks**

cs.LG

**SubmitDate**: 2022-02-14    [paper-pdf](http://arxiv.org/pdf/2111.04404v2)

**Authors**: Lijia Yu, Xiao-Shan Gao

**Abstracts**: In this paper, the bias classifier is introduced, that is, the bias part of a DNN with Relu as the activation function is used as a classifier. The work is motivated by the fact that the bias part is a piecewise constant function with zero gradient and hence cannot be directly attacked by gradient-based methods to generate adversaries, such as FGSM. The existence of the bias classifier is proved and an effective training method for the bias classifier is given. It is proved that by adding a proper random first-degree part to the bias classifier, an information-theoretically safe classifier against the original-model gradient attack is obtained in the sense that the attack will generate a totally random attacking direction. This seems to be the first time that the concept of information-theoretically safe classifier is proposed. Several attack methods for the bias classifier are proposed and numerical experiments are used to show that the bias classifier is more robust than DNNs with similar size against these attacks in most cases.



## **33. Robustness against Adversarial Attacks in Neural Networks using Incremental Dissipativity**

cs.LG

**SubmitDate**: 2022-02-14    [paper-pdf](http://arxiv.org/pdf/2111.12906v2)

**Authors**: Bernardo Aquino, Arash Rahnama, Peter Seiler, Lizhen Lin, Vijay Gupta

**Abstracts**: Adversarial examples can easily degrade the classification performance in neural networks. Empirical methods for promoting robustness to such examples have been proposed, but often lack both analytical insights and formal guarantees. Recently, some robustness certificates have appeared in the literature based on system theoretic notions. This work proposes an incremental dissipativity-based robustness certificate for neural networks in the form of a linear matrix inequality for each layer. We also propose an equivalent spectral norm bound for this certificate which is scalable to neural networks with multiple layers. We demonstrate the improved performance against adversarial attacks on a feed-forward neural network trained on MNIST and an Alexnet trained using CIFAR-10.



## **34. Adversarial Fine-tuning for Backdoor Defense: Connect Adversarial Examples to Triggered Samples**

cs.CV

**SubmitDate**: 2022-02-13    [paper-pdf](http://arxiv.org/pdf/2202.06312v1)

**Authors**: Bingxu Mu, Le Wang, Zhenxing Niu

**Abstracts**: Deep neural networks (DNNs) are known to be vulnerable to backdoor attacks, i.e., a backdoor trigger planted at training time, the infected DNN model would misclassify any testing sample embedded with the trigger as target label. Due to the stealthiness of backdoor attacks, it is hard either to detect or erase the backdoor from infected models. In this paper, we propose a new Adversarial Fine-Tuning (AFT) approach to erase backdoor triggers by leveraging adversarial examples of the infected model. For an infected model, we observe that its adversarial examples have similar behaviors as its triggered samples. Based on such observation, we design the AFT to break the foundation of the backdoor attack (i.e., the strong correlation between a trigger and a target label). We empirically show that, against 5 state-of-the-art backdoor attacks, AFT can effectively erase the backdoor triggers without obvious performance degradation on clean samples, which significantly outperforms existing defense methods.



## **35. Local Differential Privacy for Federated Learning in Industrial Settings**

cs.CR

14 pages

**SubmitDate**: 2022-02-12    [paper-pdf](http://arxiv.org/pdf/2202.06053v1)

**Authors**: M. A. P. Chamikara, Dongxi Liu, Seyit Camtepe, Surya Nepal, Marthie Grobler, Peter Bertok, Ibrahim Khalil

**Abstracts**: Federated learning (FL) is a collaborative learning approach that has gained much attention due to its inherent privacy preservation capabilities. However, advanced adversarial attacks such as membership inference and model memorization can still make FL vulnerable and potentially leak sensitive private data. Literature shows a few attempts to alleviate this problem by using global (GDP) and local differential privacy (LDP). Compared to GDP, LDP approaches are gaining more popularity due to stronger privacy notions and native support for data distribution. However, DP approaches assume that the server that aggregates the models, to be honest (run the FL protocol honestly) or semi-honest (run the FL protocol honestly while also trying to learn as much information possible), making such approaches unreliable for real-world settings. In real-world industrial environments (e.g. healthcare), the distributed entities (e.g. hospitals) are already composed of locally running machine learning models (e.g. high-performing deep neural networks on local health records). Existing approaches do not provide a scalable mechanism to utilize such settings for privacy-preserving FL. This paper proposes a new local differentially private FL (named LDPFL) protocol for industrial settings. LDPFL avoids the requirement of an honest or a semi-honest server and provides better performance while enforcing stronger privacy levels compared to existing approaches. Our experimental evaluation of LDPFL shows high FL model performance (up to ~98%) under a small privacy budget (e.g. epsilon = 0.5) in comparison to existing methods.



## **36. RoPGen: Towards Robust Code Authorship Attribution via Automatic Coding Style Transformation**

cs.CR

ICSE 2022

**SubmitDate**: 2022-02-12    [paper-pdf](http://arxiv.org/pdf/2202.06043v1)

**Authors**: Zhen Li, Guenevere, Chen, Chen Chen, Yayi Zou, Shouhuai Xu

**Abstracts**: Source code authorship attribution is an important problem often encountered in applications such as software forensics, bug fixing, and software quality analysis. Recent studies show that current source code authorship attribution methods can be compromised by attackers exploiting adversarial examples and coding style manipulation. This calls for robust solutions to the problem of code authorship attribution. In this paper, we initiate the study on making Deep Learning (DL)-based code authorship attribution robust. We propose an innovative framework called Robust coding style Patterns Generation (RoPGen), which essentially learns authors' unique coding style patterns that are hard for attackers to manipulate or imitate. The key idea is to combine data augmentation and gradient augmentation at the adversarial training phase. This effectively increases the diversity of training examples, generates meaningful perturbations to gradients of deep neural networks, and learns diversified representations of coding styles. We evaluate the effectiveness of RoPGen using four datasets of programs written in C, C++, and Java. Experimental results show that RoPGen can significantly improve the robustness of DL-based code authorship attribution, by respectively reducing 22.8% and 41.0% of the success rate of targeted and untargeted attacks on average.



## **37. Robust Deep Semi-Supervised Learning: A Brief Introduction**

cs.LG

**SubmitDate**: 2022-02-12    [paper-pdf](http://arxiv.org/pdf/2202.05975v1)

**Authors**: Lan-Zhe Guo, Zhi Zhou, Yu-Feng Li

**Abstracts**: Semi-supervised learning (SSL) is the branch of machine learning that aims to improve learning performance by leveraging unlabeled data when labels are insufficient. Recently, SSL with deep models has proven to be successful on standard benchmark tasks. However, they are still vulnerable to various robustness threats in real-world applications as these benchmarks provide perfect unlabeled data, while in realistic scenarios, unlabeled data could be corrupted. Many researchers have pointed out that after exploiting corrupted unlabeled data, SSL suffers severe performance degradation problems. Thus, there is an urgent need to develop SSL algorithms that could work robustly with corrupted unlabeled data. To fully understand robust SSL, we conduct a survey study. We first clarify a formal definition of robust SSL from the perspective of machine learning. Then, we classify the robustness threats into three categories: i) distribution corruption, i.e., unlabeled data distribution is mismatched with labeled data; ii) feature corruption, i.e., the features of unlabeled examples are adversarially attacked; and iii) label corruption, i.e., the label distribution of unlabeled data is imbalanced. Under this unified taxonomy, we provide a thorough review and discussion of recent works that focus on these issues. Finally, we propose possible promising directions within robust SSL to provide insights for future research.



## **38. Measuring the Contribution of Multiple Model Representations in Detecting Adversarial Instances**

cs.LG

Correction: replaced "model-wise" with "unit-wise" in the first  sentence of Section 3.2

**SubmitDate**: 2022-02-12    [paper-pdf](http://arxiv.org/pdf/2111.07035v2)

**Authors**: Daniel Steinberg, Paul Munro

**Abstracts**: Deep learning models have been used for a wide variety of tasks. They are prevalent in computer vision, natural language processing, speech recognition, and other areas. While these models have worked well under many scenarios, it has been shown that they are vulnerable to adversarial attacks. This has led to a proliferation of research into ways that such attacks could be identified and/or defended against. Our goal is to explore the contribution that can be attributed to using multiple underlying models for the purpose of adversarial instance detection. Our paper describes two approaches that incorporate representations from multiple models for detecting adversarial examples. We devise controlled experiments for measuring the detection impact of incrementally utilizing additional models. For many of the scenarios we consider, the results show that performance increases with the number of underlying models used for extracting representations.



## **39. Adversarial Attacks and Defense Methods for Power Quality Recognition**

cs.CR

Technical report

**SubmitDate**: 2022-02-11    [paper-pdf](http://arxiv.org/pdf/2202.07421v1)

**Authors**: Jiwei Tian, Buhong Wang, Jing Li, Zhen Wang, Mete Ozay

**Abstracts**: Vulnerability of various machine learning methods to adversarial examples has been recently explored in the literature. Power systems which use these vulnerable methods face a huge threat against adversarial examples. To this end, we first propose a signal-specific method and a universal signal-agnostic method to attack power systems using generated adversarial examples. Black-box attacks based on transferable characteristics and the above two methods are also proposed and evaluated. We then adopt adversarial training to defend systems against adversarial attacks. Experimental analyses demonstrate that our signal-specific attack method provides less perturbation compared to the FGSM (Fast Gradient Sign Method), and our signal-agnostic attack method can generate perturbations fooling most natural signals with high probability. What's more, the attack method based on the universal signal-agnostic algorithm has a higher transfer rate of black-box attacks than the attack method based on the signal-specific algorithm. In addition, the results show that the proposed adversarial training improves robustness of power systems to adversarial examples.



## **40. Are socially-aware trajectory prediction models really socially-aware?**

cs.CV

**SubmitDate**: 2022-02-11    [paper-pdf](http://arxiv.org/pdf/2108.10879v2)

**Authors**: Saeed Saadatnejad, Mohammadhossein Bahari, Pedram Khorsandi, Mohammad Saneian, Seyed-Mohsen Moosavi-Dezfooli, Alexandre Alahi

**Abstracts**: Our field has recently witnessed an arms race of neural network-based trajectory predictors. While these predictors are at the core of many applications such as autonomous navigation or pedestrian flow simulations, their adversarial robustness has not been carefully studied. In this paper, we introduce a socially-attended attack to assess the social understanding of prediction models in terms of collision avoidance. An attack is a small yet carefully-crafted perturbations to fail predictors. Technically, we define collision as a failure mode of the output, and propose hard- and soft-attention mechanisms to guide our attack. Thanks to our attack, we shed light on the limitations of the current models in terms of their social understanding. We demonstrate the strengths of our method on the recent trajectory prediction models. Finally, we show that our attack can be employed to increase the social understanding of state-of-the-art models. The code is available online: https://s-attack.github.io/



## **41. Using Random Perturbations to Mitigate Adversarial Attacks on Sentiment Analysis Models**

cs.CL

To be published in the proceedings for the 18th International  Conference on Natural Language Processing (ICON 2021)

**SubmitDate**: 2022-02-11    [paper-pdf](http://arxiv.org/pdf/2202.05758v1)

**Authors**: Abigail Swenor, Jugal Kalita

**Abstracts**: Attacks on deep learning models are often difficult to identify and therefore are difficult to protect against. This problem is exacerbated by the use of public datasets that typically are not manually inspected before use. In this paper, we offer a solution to this vulnerability by using, during testing, random perturbations such as spelling correction if necessary, substitution by random synonym, or simply dropping the word. These perturbations are applied to random words in random sentences to defend NLP models against adversarial attacks. Our Random Perturbations Defense and Increased Randomness Defense methods are successful in returning attacked models to similar accuracy of models before attacks. The original accuracy of the model used in this work is 80% for sentiment classification. After undergoing attacks, the accuracy drops to accuracy between 0% and 44%. After applying our defense methods, the accuracy of the model is returned to the original accuracy within statistical significance.



## **42. On the Detection of Adaptive Adversarial Attacks in Speaker Verification Systems**

cs.CR

**SubmitDate**: 2022-02-11    [paper-pdf](http://arxiv.org/pdf/2202.05725v1)

**Authors**: Zesheng Chen

**Abstracts**: Speaker verification systems have been widely used in smart phones and Internet of things devices to identify a legitimate user. In recent work, it has been shown that adversarial attacks, such as FAKEBOB, can work effectively against speaker verification systems. The goal of this paper is to design a detector that can distinguish an original audio from an audio contaminated by adversarial attacks. Specifically, our designed detector, called MEH-FEST, calculates the minimum energy in high frequencies from the short-time Fourier transform of an audio and uses it as a detection metric. Through both analysis and experiments, we show that our proposed detector is easy to implement, fast to process an input audio, and effective in determining whether an audio is corrupted by FAKEBOB attacks. The experimental results indicate that the detector is extremely effective: with near zero false positive and false negative rates for detecting FAKEBOB attacks in Gaussian mixture model (GMM) and i-vector speaker verification systems. Moreover, adaptive adversarial attacks against our proposed detector and their countermeasures are discussed and studied, showing the game between attackers and defenders.



## **43. Towards Adversarially Robust Deepfake Detection: An Ensemble Approach**

cs.LG

**SubmitDate**: 2022-02-11    [paper-pdf](http://arxiv.org/pdf/2202.05687v1)

**Authors**: Ashish Hooda, Neal Mangaokar, Ryan Feng, Kassem Fawaz, Somesh Jha, Atul Prakash

**Abstracts**: Detecting deepfakes is an important problem, but recent work has shown that DNN-based deepfake detectors are brittle against adversarial deepfakes, in which an adversary adds imperceptible perturbations to a deepfake to evade detection. In this work, we show that a modification to the detection strategy in which we replace a single classifier with a carefully chosen ensemble, in which input transformations for each model in the ensemble induces pairwise orthogonal gradients, can significantly improve robustness beyond the de facto solution of adversarial training. We present theoretical results to show that such orthogonal gradients can help thwart a first-order adversary by reducing the dimensionality of the input subspace in which adversarial deepfakes lie. We validate the results empirically by instantiating and evaluating a randomized version of such "orthogonal" ensembles for adversarial deepfake detection and find that these randomized ensembles exhibit significantly higher robustness as deepfake detectors compared to state-of-the-art deepfake detectors against adversarial deepfakes, even those created using strong PGD-500 attacks.



## **44. FAAG: Fast Adversarial Audio Generation through Interactive Attack Optimisation**

cs.SD

**SubmitDate**: 2022-02-11    [paper-pdf](http://arxiv.org/pdf/2202.05416v1)

**Authors**: Yuantian Miao, Chao Chen, Lei Pan, Jun Zhang, Yang Xiang

**Abstracts**: Automatic Speech Recognition services (ASRs) inherit deep neural networks' vulnerabilities like crafted adversarial examples. Existing methods often suffer from low efficiency because the target phases are added to the entire audio sample, resulting in high demand for computational resources. This paper proposes a novel scheme named FAAG as an iterative optimization-based method to generate targeted adversarial examples quickly. By injecting the noise over the beginning part of the audio, FAAG generates adversarial audio in high quality with a high success rate timely. Specifically, we use audio's logits output to map each character in the transcription to an approximate position of the audio's frame. Thus, an adversarial example can be generated by FAAG in approximately two minutes using CPUs only and around ten seconds with one GPU while maintaining an average success rate over 85%. Specifically, the FAAG method can speed up around 60% compared with the baseline method during the adversarial example generation process. Furthermore, we found that appending benign audio to any suspicious examples can effectively defend against the targeted adversarial attack. We hope that this work paves the way for inventing new adversarial attacks against speech recognition with computational constraints.



## **45. SoK: Certified Robustness for Deep Neural Networks**

cs.LG

14 pages for the main text; recent advances (till Feb 2022) included

**SubmitDate**: 2022-02-10    [paper-pdf](http://arxiv.org/pdf/2009.04131v6)

**Authors**: Linyi Li, Tao Xie, Bo Li

**Abstracts**: Great advances in deep neural networks (DNNs) have led to state-of-the-art performance on a wide range of tasks. However, recent studies have shown that DNNs are vulnerable to adversarial attacks, which have brought great concerns when deploying these models to safety-critical applications such as autonomous driving. Different defense approaches have been proposed against adversarial attacks, including: a) empirical defenses, which usually can be adaptively attacked again without providing robustness certification; and b) certifiably robust approaches which consist of robustness verification providing the lower bound of robust accuracy against any attacks under certain conditions and corresponding robust training approaches. In this paper, we systematize the certifiably robust approaches and related practical and theoretical implications and findings. We also provide the first comprehensive benchmark on existing robustness verification and training approaches on different datasets. In particular, we 1) provide a taxonomy for the robustness verification and training approaches, as well as summarize the methodologies for representative algorithms, 2) reveal the characteristics, strengths, limitations, and fundamental connections among these approaches, 3) discuss current research progresses, theoretical barriers, main challenges, and future directions for certifiably robust approaches for DNNs, and 4) provide an open-sourced unified platform to evaluate over 20 representative certifiably robust approaches for a wide range of DNNs.



## **46. Towards Assessing and Characterizing the Semantic Robustness of Face Recognition**

cs.CV

26 pages, 18 figures

**SubmitDate**: 2022-02-10    [paper-pdf](http://arxiv.org/pdf/2202.04978v1)

**Authors**: Juan C. Pérez, Motasem Alfarra, Ali Thabet, Pablo Arbeláez, Bernard Ghanem

**Abstracts**: Deep Neural Networks (DNNs) lack robustness against imperceptible perturbations to their input. Face Recognition Models (FRMs) based on DNNs inherit this vulnerability. We propose a methodology for assessing and characterizing the robustness of FRMs against semantic perturbations to their input. Our methodology causes FRMs to malfunction by designing adversarial attacks that search for identity-preserving modifications to faces. In particular, given a face, our attacks find identity-preserving variants of the face such that an FRM fails to recognize the images belonging to the same identity. We model these identity-preserving semantic modifications via direction- and magnitude-constrained perturbations in the latent space of StyleGAN. We further propose to characterize the semantic robustness of an FRM by statistically describing the perturbations that induce the FRM to malfunction. Finally, we combine our methodology with a certification technique, thus providing (i) theoretical guarantees on the performance of an FRM, and (ii) a formal description of how an FRM may model the notion of face identity.



## **47. Beyond ImageNet Attack: Towards Crafting Adversarial Examples for Black-box Domains**

cs.CV

Accepted by ICLR 2022

**SubmitDate**: 2022-02-10    [paper-pdf](http://arxiv.org/pdf/2201.11528v3)

**Authors**: Qilong Zhang, Xiaodan Li, Yuefeng Chen, Jingkuan Song, Lianli Gao, Yuan He, Hui Xue

**Abstracts**: Adversarial examples have posed a severe threat to deep neural networks due to their transferable nature. Currently, various works have paid great efforts to enhance the cross-model transferability, which mostly assume the substitute model is trained in the same domain as the target model. However, in reality, the relevant information of the deployed model is unlikely to leak. Hence, it is vital to build a more practical black-box threat model to overcome this limitation and evaluate the vulnerability of deployed models. In this paper, with only the knowledge of the ImageNet domain, we propose a Beyond ImageNet Attack (BIA) to investigate the transferability towards black-box domains (unknown classification tasks). Specifically, we leverage a generative model to learn the adversarial function for disrupting low-level features of input images. Based on this framework, we further propose two variants to narrow the gap between the source and target domains from the data and model perspectives, respectively. Extensive experiments on coarse-grained and fine-grained domains demonstrate the effectiveness of our proposed methods. Notably, our methods outperform state-of-the-art approaches by up to 7.71\% (towards coarse-grained domains) and 25.91\% (towards fine-grained domains) on average. Our code is available at \url{https://github.com/qilong-zhang/Beyond-ImageNet-Attack}.



## **48. Adversarial Attack and Defense of YOLO Detectors in Autonomous Driving Scenarios**

cs.CV

7 pages, 3 figures

**SubmitDate**: 2022-02-10    [paper-pdf](http://arxiv.org/pdf/2202.04781v1)

**Authors**: Jung Im Choi, Qing Tian

**Abstracts**: Visual detection is a key task in autonomous driving, and it serves as one foundation for self-driving planning and control. Deep neural networks have achieved promising results in various computer vision tasks, but they are known to be vulnerable to adversarial attacks. A comprehensive understanding of deep visual detectors' vulnerability is required before people can improve their robustness. However, only a few adversarial attack/defense works have focused on object detection, and most of them employed only classification and/or localization losses, ignoring the objectness aspect. In this paper, we identify a serious objectness-related adversarial vulnerability in YOLO detectors and present an effective attack strategy aiming the objectness aspect of visual detection in autonomous vehicles. Furthermore, to address such vulnerability, we propose a new objectness-aware adversarial training approach for visual detection. Experiments show that the proposed attack targeting the objectness aspect is 45.17% and 43.50% more effective than those generated from classification and/or localization losses on the KITTI and COCO_traffic datasets, respectively. Also, the proposed adversarial defense approach can improve the detectors' robustness against objectness-oriented attacks by up to 21% and 12% mAP on KITTI and COCO_traffic, respectively.



## **49. IoTMonitor: A Hidden Markov Model-based Security System to Identify Crucial Attack Nodes in Trigger-action IoT Platforms**

cs.CR

This paper appears in the 2022 IEEE Wireless Communications and  Networking Conference (WCNC 2022). Personal use of this material is  permitted. Permission from IEEE must be obtained for all other uses

**SubmitDate**: 2022-02-09    [paper-pdf](http://arxiv.org/pdf/2202.04620v1)

**Authors**: Md Morshed Alam, Md Sajidul Islam Sajid, Weichao Wang, Jinpeng Wei

**Abstracts**: With the emergence and fast development of trigger-action platforms in IoT settings, security vulnerabilities caused by the interactions among IoT devices become more prevalent. The event occurrence at one device triggers an action in another device, which may eventually contribute to the creation of a chain of events in a network. Adversaries exploit the chain effect to compromise IoT devices and trigger actions of interest remotely just by injecting malicious events into the chain. To address security vulnerabilities caused by trigger-action scenarios, existing research efforts focus on the validation of the security properties of devices or verification of the occurrence of certain events based on their physical fingerprints on a device. We propose IoTMonitor, a security analysis system that discerns the underlying chain of event occurrences with the highest probability by observing a chain of physical evidence collected by sensors. We use the Baum-Welch algorithm to estimate transition and emission probabilities and the Viterbi algorithm to discern the event sequence. We can then identify the crucial nodes in the trigger-action sequence whose compromise allows attackers to reach their final goals. The experiment results of our designed system upon the PEEVES datasets show that we can rebuild the event occurrence sequence with high accuracy from the observations and identify the crucial nodes on the attack paths.



## **50. False Memory Formation in Continual Learners Through Imperceptible Backdoor Trigger**

cs.LG

**SubmitDate**: 2022-02-09    [paper-pdf](http://arxiv.org/pdf/2202.04479v1)

**Authors**: Muhammad Umer, Robi Polikar

**Abstracts**: In this brief, we show that sequentially learning new information presented to a continual (incremental) learning model introduces new security risks: an intelligent adversary can introduce small amount of misinformation to the model during training to cause deliberate forgetting of a specific task or class at test time, thus creating "false memory" about that task. We demonstrate such an adversary's ability to assume control of the model by injecting "backdoor" attack samples to commonly used generative replay and regularization based continual learning approaches using continual learning benchmark variants of MNIST, as well as the more challenging SVHN and CIFAR 10 datasets. Perhaps most damaging, we show this vulnerability to be very acute and exceptionally effective: the backdoor pattern in our attack model can be imperceptible to human eye, can be provided at any point in time, can be added into the training data of even a single possibly unrelated task and can be achieved with as few as just 1\% of total training dataset of a single task.



