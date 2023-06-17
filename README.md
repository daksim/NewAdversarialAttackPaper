# Latest Adversarial Attack Papers
**update at 2023-06-17 15:38:19**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Inroads into Autonomous Network Defence using Explained Reinforcement Learning**

cs.CR

**SubmitDate**: 2023-06-15    [abs](http://arxiv.org/abs/2306.09318v1) [paper-pdf](http://arxiv.org/pdf/2306.09318v1)

**Authors**: Myles Foley, Mia Wang, Zoe M, Chris Hicks, Vasilios Mavroudis

**Abstract**: Computer network defence is a complicated task that has necessitated a high degree of human involvement. However, with recent advancements in machine learning, fully autonomous network defence is becoming increasingly plausible. This paper introduces an end-to-end methodology for studying attack strategies, designing defence agents and explaining their operation. First, using state diagrams, we visualise adversarial behaviour to gain insight about potential points of intervention and inform the design of our defensive models. We opt to use a set of deep reinforcement learning agents trained on different parts of the task and organised in a shallow hierarchy. Our evaluation shows that the resulting design achieves a substantial performance improvement compared to prior work. Finally, to better investigate the decision-making process of our agents, we complete our analysis with a feature ablation and importance study.



## **2. Adversarial Cheap Talk**

cs.LG

To be published at ICML 2023. Project video and code are available at  https://sites.google.com/view/adversarial-cheap-talk

**SubmitDate**: 2023-06-15    [abs](http://arxiv.org/abs/2211.11030v2) [paper-pdf](http://arxiv.org/pdf/2211.11030v2)

**Authors**: Chris Lu, Timon Willi, Alistair Letcher, Jakob Foerster

**Abstract**: Adversarial attacks in reinforcement learning (RL) often assume highly-privileged access to the victim's parameters, environment, or data. Instead, this paper proposes a novel adversarial setting called a Cheap Talk MDP in which an Adversary can merely append deterministic messages to the Victim's observation, resulting in a minimal range of influence. The Adversary cannot occlude ground truth, influence underlying environment dynamics or reward signals, introduce non-stationarity, add stochasticity, see the Victim's actions, or access their parameters. Additionally, we present a simple meta-learning algorithm called Adversarial Cheap Talk (ACT) to train Adversaries in this setting. We demonstrate that an Adversary trained with ACT still significantly influences the Victim's training and testing performance, despite the highly constrained setting. Affecting train-time performance reveals a new attack vector and provides insight into the success and failure modes of existing RL algorithms. More specifically, we show that an ACT Adversary is capable of harming performance by interfering with the learner's function approximation, or instead helping the Victim's performance by outputting useful features. Finally, we show that an ACT Adversary can manipulate messages during train-time to directly and arbitrarily control the Victim at test-time. Project video and code are available at https://sites.google.com/view/adversarial-cheap-talk



## **3. DIFFender: Diffusion-Based Adversarial Defense against Patch Attacks in the Physical World**

cs.CV

**SubmitDate**: 2023-06-15    [abs](http://arxiv.org/abs/2306.09124v1) [paper-pdf](http://arxiv.org/pdf/2306.09124v1)

**Authors**: Caixin Kang, Yinpeng Dong, Zhengyi Wang, Shouwei Ruan, Hang Su, Xingxing Wei

**Abstract**: Adversarial attacks in the physical world, particularly patch attacks, pose significant threats to the robustness and reliability of deep learning models. Developing reliable defenses against patch attacks is crucial for real-world applications, yet current research in this area is severely lacking. In this paper, we propose DIFFender, a novel defense method that leverages the pre-trained diffusion model to perform both localization and defense against potential adversarial patch attacks. DIFFender is designed as a pipeline consisting of two main stages: patch localization and restoration. In the localization stage, we exploit the intriguing properties of a diffusion model to effectively identify the locations of adversarial patches. In the restoration stage, we employ a text-guided diffusion model to eliminate adversarial regions in the image while preserving the integrity of the visual content. Additionally, we design a few-shot prompt-tuning algorithm to facilitate simple and efficient tuning, enabling the learned representations to easily transfer to downstream tasks, which optimize two stages jointly. We conduct extensive experiments on image classification and face recognition to demonstrate that DIFFender exhibits superior robustness under strong adaptive attacks and generalizes well across various scenarios, diverse classifiers, and multiple attack methods.



## **4. The Effect of Length on Key Fingerprint Verification Security and Usability**

cs.CR

Accepted to International Conference on Availability, Reliability and  Security (ARES 2023)

**SubmitDate**: 2023-06-15    [abs](http://arxiv.org/abs/2306.04574v2) [paper-pdf](http://arxiv.org/pdf/2306.04574v2)

**Authors**: Dan Turner, Siamak F. Shahandashti, Helen Petrie

**Abstract**: In applications such as end-to-end encrypted instant messaging, secure email, and device pairing, users need to compare key fingerprints to detect impersonation and adversary-in-the-middle attacks. Key fingerprints are usually computed as truncated hashes of each party's view of the channel keys, encoded as an alphanumeric or numeric string, and compared out-of-band, e.g. manually, to detect any inconsistencies. Previous work has extensively studied the usability of various verification strategies and encoding formats, however, the exact effect of key fingerprint length on the security and usability of key fingerprint verification has not been rigorously investigated. We present a 162-participant study on the effect of numeric key fingerprint length on comparison time and error rate. While the results confirm some widely-held intuitions such as general comparison times and errors increasing significantly with length, a closer look reveals interesting nuances. The significant rise in comparison time only occurs when highly similar fingerprints are compared, and comparison time remains relatively constant otherwise. On errors, our results clearly distinguish between security non-critical errors that remain low irrespective of length and security critical errors that significantly rise, especially at higher fingerprint lengths. A noteworthy implication of this latter result is that Signal/WhatsApp key fingerprints provide a considerably lower level of security than usually assumed.



## **5. Community Detection Attack against Collaborative Learning-based Recommender Systems**

cs.IR

**SubmitDate**: 2023-06-15    [abs](http://arxiv.org/abs/2306.08929v1) [paper-pdf](http://arxiv.org/pdf/2306.08929v1)

**Authors**: Yacine Belal, Sonia Ben Mokhtar, Mohamed Maouche, Anthony Simonet-Boulogne

**Abstract**: Collaborative-learning based recommender systems emerged following the success of collaborative learning techniques such as Federated Learning (FL) and Gossip Learning (GL). In these systems, users participate in the training of a recommender system while keeping their history of consumed items on their devices. While these solutions seemed appealing for preserving the privacy of the participants at a first glance, recent studies have shown that collaborative learning can be vulnerable to a variety of privacy attacks. In this paper we propose a novel privacy attack called Community Detection Attack (CDA), which allows an adversary to discover the members of a community based on a set of items of her choice (e.g., discovering users interested in LGBT content). Through experiments on three real recommendation datasets and by using two state-of-the-art recommendation models, we assess the sensitivity of an FL-based recommender system as well as two flavors of Gossip Learning-based recommender systems to CDA. Results show that on all models and all datasets, the FL setting is more vulnerable to CDA than Gossip settings. We further evaluated two off-the-shelf mitigation strategies, namely differential privacy (DP) and a share less policy, which consists in sharing a subset of model parameters. Results show a better privacy-utility trade-off for the share less policy compared to DP especially in the Gossip setting.



## **6. MalProtect: Stateful Defense Against Adversarial Query Attacks in ML-based Malware Detection**

cs.LG

**SubmitDate**: 2023-06-14    [abs](http://arxiv.org/abs/2302.10739v2) [paper-pdf](http://arxiv.org/pdf/2302.10739v2)

**Authors**: Aqib Rashid, Jose Such

**Abstract**: ML models are known to be vulnerable to adversarial query attacks. In these attacks, queries are iteratively perturbed towards a particular class without any knowledge of the target model besides its output. The prevalence of remotely-hosted ML classification models and Machine-Learning-as-a-Service platforms means that query attacks pose a real threat to the security of these systems. To deal with this, stateful defenses have been proposed to detect query attacks and prevent the generation of adversarial examples by monitoring and analyzing the sequence of queries received by the system. Several stateful defenses have been proposed in recent years. However, these defenses rely solely on similarity or out-of-distribution detection methods that may be effective in other domains. In the malware detection domain, the methods to generate adversarial examples are inherently different, and therefore we find that such detection mechanisms are significantly less effective. Hence, in this paper, we present MalProtect, which is a stateful defense against query attacks in the malware detection domain. MalProtect uses several threat indicators to detect attacks. Our results show that it reduces the evasion rate of adversarial query attacks by 80+\% in Android and Windows malware, across a range of attacker scenarios. In the first evaluation of its kind, we show that MalProtect outperforms prior stateful defenses, especially under the peak adversarial threat.



## **7. Augment then Smooth: Reconciling Differential Privacy with Certified Robustness**

cs.LG

25 pages, 19 figures

**SubmitDate**: 2023-06-14    [abs](http://arxiv.org/abs/2306.08656v1) [paper-pdf](http://arxiv.org/pdf/2306.08656v1)

**Authors**: Jiapeng Wu, Atiyeh Ashari Ghomi, David Glukhov, Jesse C. Cresswell, Franziska Boenisch, Nicolas Papernot

**Abstract**: Machine learning models are susceptible to a variety of attacks that can erode trust in their deployment. These threats include attacks against the privacy of training data and adversarial examples that jeopardize model accuracy. Differential privacy and randomized smoothing are effective defenses that provide certifiable guarantees for each of these threats, however, it is not well understood how implementing either defense impacts the other. In this work, we argue that it is possible to achieve both privacy guarantees and certified robustness simultaneously. We provide a framework called DP-CERT for integrating certified robustness through randomized smoothing into differentially private model training. For instance, compared to differentially private stochastic gradient descent on CIFAR10, DP-CERT leads to a 12-fold increase in certified accuracy and a 10-fold increase in the average certified radius at the expense of a drop in accuracy of 1.2%. Through in-depth per-sample metric analysis, we show that the certified radius correlates with the local Lipschitz constant and smoothness of the loss surface. This provides a new way to diagnose when private models will fail to be robust.



## **8. A Unified Framework of Graph Information Bottleneck for Robustness and Membership Privacy**

cs.LG

**SubmitDate**: 2023-06-14    [abs](http://arxiv.org/abs/2306.08604v1) [paper-pdf](http://arxiv.org/pdf/2306.08604v1)

**Authors**: Enyan Dai, Limeng Cui, Zhengyang Wang, Xianfeng Tang, Yinghan Wang, Monica Cheng, Bing Yin, Suhang Wang

**Abstract**: Graph Neural Networks (GNNs) have achieved great success in modeling graph-structured data. However, recent works show that GNNs are vulnerable to adversarial attacks which can fool the GNN model to make desired predictions of the attacker. In addition, training data of GNNs can be leaked under membership inference attacks. This largely hinders the adoption of GNNs in high-stake domains such as e-commerce, finance and bioinformatics. Though investigations have been made in conducting robust predictions and protecting membership privacy, they generally fail to simultaneously consider the robustness and membership privacy. Therefore, in this work, we study a novel problem of developing robust and membership privacy-preserving GNNs. Our analysis shows that Information Bottleneck (IB) can help filter out noisy information and regularize the predictions on labeled samples, which can benefit robustness and membership privacy. However, structural noises and lack of labels in node classification challenge the deployment of IB on graph-structured data. To mitigate these issues, we propose a novel graph information bottleneck framework that can alleviate structural noises with neighbor bottleneck. Pseudo labels are also incorporated in the optimization to minimize the gap between the predictions on the labeled set and unlabeled set for membership privacy. Extensive experiments on real-world datasets demonstrate that our method can give robust predictions and simultaneously preserve membership privacy.



## **9. Tight Certification of Adversarially Trained Neural Networks via Nonconvex Low-Rank Semidefinite Relaxations**

cs.LG

ICML 2023

**SubmitDate**: 2023-06-14    [abs](http://arxiv.org/abs/2211.17244v3) [paper-pdf](http://arxiv.org/pdf/2211.17244v3)

**Authors**: Hong-Ming Chiu, Richard Y. Zhang

**Abstract**: Adversarial training is well-known to produce high-quality neural network models that are empirically robust against adversarial perturbations. Nevertheless, once a model has been adversarially trained, one often desires a certification that the model is truly robust against all future attacks. Unfortunately, when faced with adversarially trained models, all existing approaches have significant trouble making certifications that are strong enough to be practically useful. Linear programming (LP) techniques in particular face a "convex relaxation barrier" that prevent them from making high-quality certifications, even after refinement with mixed-integer linear programming (MILP) and branch-and-bound (BnB) techniques. In this paper, we propose a nonconvex certification technique, based on a low-rank restriction of a semidefinite programming (SDP) relaxation. The nonconvex relaxation makes strong certifications comparable to much more expensive SDP methods, while optimizing over dramatically fewer variables comparable to much weaker LP methods. Despite nonconvexity, we show how off-the-shelf local optimization algorithms can be used to achieve and to certify global optimality in polynomial time. Our experiments find that the nonconvex relaxation almost completely closes the gap towards exact certification of adversarially trained models.



## **10. Reliable Evaluation of Adversarial Transferability**

cs.CV

**SubmitDate**: 2023-06-14    [abs](http://arxiv.org/abs/2306.08565v1) [paper-pdf](http://arxiv.org/pdf/2306.08565v1)

**Authors**: Wenqian Yu, Jindong Gu, Zhijiang Li, Philip Torr

**Abstract**: Adversarial examples (AEs) with small adversarial perturbations can mislead deep neural networks (DNNs) into wrong predictions. The AEs created on one DNN can also fool another DNN. Over the last few years, the transferability of AEs has garnered significant attention as it is a crucial property for facilitating black-box attacks. Many approaches have been proposed to improve adversarial transferability. However, they are mainly verified across different convolutional neural network (CNN) architectures, which is not a reliable evaluation since all CNNs share some similar architectural biases. In this work, we re-evaluate 12 representative transferability-enhancing attack methods where we test on 18 popular models from 4 types of neural networks. Our reevaluation revealed that the adversarial transferability is often overestimated, and there is no single AE that can be transferred to all popular models. The transferability rank of previous attacking methods changes when under our comprehensive evaluation. Based on our analysis, we propose a reliable benchmark including three evaluation protocols. Adversarial transferability on our new benchmark is extremely low, which further confirms the overestimation of adversarial transferability. We release our benchmark at https://adv-trans-eval.github.io to facilitate future research, which includes code, model checkpoints, and evaluation protocols.



## **11. LMD: A Learnable Mask Network to Detect Adversarial Examples for Speaker Verification**

eess.AS

13 pages, 9 figures

**SubmitDate**: 2023-06-14    [abs](http://arxiv.org/abs/2211.00825v2) [paper-pdf](http://arxiv.org/pdf/2211.00825v2)

**Authors**: Xing Chen, Jie Wang, Xiao-Lei Zhang, Wei-Qiang Zhang, Kunde Yang

**Abstract**: Although the security of automatic speaker verification (ASV) is seriously threatened by recently emerged adversarial attacks, there have been some countermeasures to alleviate the threat. However, many defense approaches not only require the prior knowledge of the attackers but also possess weak interpretability. To address this issue, in this paper, we propose an attacker-independent and interpretable method, named learnable mask detector (LMD), to separate adversarial examples from the genuine ones. It utilizes score variation as an indicator to detect adversarial examples, where the score variation is the absolute discrepancy between the ASV scores of an original audio recording and its transformed audio synthesized from its masked complex spectrogram. A core component of the score variation detector is to generate the masked spectrogram by a neural network. The neural network needs only genuine examples for training, which makes it an attacker-independent approach. Its interpretability lies that the neural network is trained to minimize the score variation of the targeted ASV, and maximize the number of the masked spectrogram bins of the genuine training examples. Its foundation is based on the observation that, masking out the vast majority of the spectrogram bins with little speaker information will inevitably introduce a large score variation to the adversarial example, and a small score variation to the genuine example. Experimental results with 12 attackers and two representative ASV systems show that our proposed method outperforms five state-of-the-art baselines. The extensive experimental results can also be a benchmark for the detection-based ASV defenses.



## **12. COVER: A Heuristic Greedy Adversarial Attack on Prompt-based Learning in Language Models**

cs.CL

**SubmitDate**: 2023-06-14    [abs](http://arxiv.org/abs/2306.05659v2) [paper-pdf](http://arxiv.org/pdf/2306.05659v2)

**Authors**: Zihao Tan, Qingliang Chen, Wenbin Zhu, Yongjian Huang

**Abstract**: Prompt-based learning has been proved to be an effective way in pre-trained language models (PLMs), especially in low-resource scenarios like few-shot settings. However, the trustworthiness of PLMs is of paramount significance and potential vulnerabilities have been shown in prompt-based templates that could mislead the predictions of language models, causing serious security concerns. In this paper, we will shed light on some vulnerabilities of PLMs, by proposing a prompt-based adversarial attack on manual templates in black box scenarios. First of all, we design character-level and word-level heuristic approaches to break manual templates separately. Then we present a greedy algorithm for the attack based on the above heuristic destructive approaches. Finally, we evaluate our approach with the classification tasks on three variants of BERT series models and eight datasets. And comprehensive experimental results justify the effectiveness of our approach in terms of attack success rate and attack speed. Further experimental studies indicate that our proposed method also displays good capabilities in scenarios with varying shot counts, template lengths and query counts, exhibiting good generalizability.



## **13. A Relaxed Optimization Approach for Adversarial Attacks against Neural Machine Translation Models**

cs.CL

**SubmitDate**: 2023-06-14    [abs](http://arxiv.org/abs/2306.08492v1) [paper-pdf](http://arxiv.org/pdf/2306.08492v1)

**Authors**: Sahar Sadrizadeh, Clément Barbier, Ljiljana Dolamic, Pascal Frossard

**Abstract**: In this paper, we propose an optimization-based adversarial attack against Neural Machine Translation (NMT) models. First, we propose an optimization problem to generate adversarial examples that are semantically similar to the original sentences but destroy the translation generated by the target NMT model. This optimization problem is discrete, and we propose a continuous relaxation to solve it. With this relaxation, we find a probability distribution for each token in the adversarial example, and then we can generate multiple adversarial examples by sampling from these distributions. Experimental results show that our attack significantly degrades the translation quality of multiple NMT models while maintaining the semantic similarity between the original and adversarial sentences. Furthermore, our attack outperforms the baselines in terms of success rate, similarity preservation, effect on translation quality, and token error rate. Finally, we propose a black-box extension of our attack by sampling from an optimized probability distribution for a reference model whose gradients are accessible.



## **14. X-Detect: Explainable Adversarial Patch Detection for Object Detectors in Retail**

cs.CV

**SubmitDate**: 2023-06-14    [abs](http://arxiv.org/abs/2306.08422v1) [paper-pdf](http://arxiv.org/pdf/2306.08422v1)

**Authors**: Omer Hofman, Amit Giloni, Yarin Hayun, Ikuya Morikawa, Toshiya Shimizu, Yuval Elovici, Asaf Shabtai

**Abstract**: Object detection models, which are widely used in various domains (such as retail), have been shown to be vulnerable to adversarial attacks. Existing methods for detecting adversarial attacks on object detectors have had difficulty detecting new real-life attacks. We present X-Detect, a novel adversarial patch detector that can: i) detect adversarial samples in real time, allowing the defender to take preventive action; ii) provide explanations for the alerts raised to support the defender's decision-making process, and iii) handle unfamiliar threats in the form of new attacks. Given a new scene, X-Detect uses an ensemble of explainable-by-design detectors that utilize object extraction, scene manipulation, and feature transformation techniques to determine whether an alert needs to be raised. X-Detect was evaluated in both the physical and digital space using five different attack scenarios (including adaptive attacks) and the COCO dataset and our new Superstore dataset. The physical evaluation was performed using a smart shopping cart setup in real-world settings and included 17 adversarial patch attacks recorded in 1,700 adversarial videos. The results showed that X-Detect outperforms the state-of-the-art methods in distinguishing between benign and adversarial scenes for all attack scenarios while maintaining a 0% FPR (no false alarms) and providing actionable explanations for the alerts raised. A demo is available.



## **15. Global-Local Processing in Convolutional Neural Networks**

cs.CV

**SubmitDate**: 2023-06-14    [abs](http://arxiv.org/abs/2306.08336v1) [paper-pdf](http://arxiv.org/pdf/2306.08336v1)

**Authors**: Zahra Rezvani, Soroor Shekarizeh, Mohammad Sabokrou

**Abstract**: Convolutional Neural Networks (CNNs) have achieved outstanding performance on image processing challenges. Actually, CNNs imitate the typically developed human brain structures at the micro-level (Artificial neurons). At the same time, they distance themselves from imitating natural visual perception in humans at the macro architectures (high-level cognition). Recently it has been investigated that CNNs are highly biased toward local features and fail to detect the global aspects of their input. Nevertheless, the literature offers limited clues on this problem. To this end, we propose a simple yet effective solution inspired by the unconscious behavior of the human pupil. We devise a simple module called Global Advantage Stream (GAS) to learn and capture the holistic features of input samples (i.e., the global features). Then, the GAS features were combined with a CNN network as a plug-and-play component called the Global/Local Processing (GLP) model. The experimental results confirm that this stream improves the accuracy with an insignificant additional computational/temporal load and makes the network more robust to adversarial attacks. Furthermore, investigating the interpretation of the model shows that it learns a more holistic representation similar to the perceptual system of healthy humans



## **16. On the Robustness of Latent Diffusion Models**

cs.CV

**SubmitDate**: 2023-06-14    [abs](http://arxiv.org/abs/2306.08257v1) [paper-pdf](http://arxiv.org/pdf/2306.08257v1)

**Authors**: Jianping Zhang, Zhuoer Xu, Shiwen Cui, Changhua Meng, Weibin Wu, Michael R. Lyu

**Abstract**: Latent diffusion models achieve state-of-the-art performance on a variety of generative tasks, such as image synthesis and image editing. However, the robustness of latent diffusion models is not well studied. Previous works only focus on the adversarial attacks against the encoder or the output image under white-box settings, regardless of the denoising process. Therefore, in this paper, we aim to analyze the robustness of latent diffusion models more thoroughly. We first study the influence of the components inside latent diffusion models on their white-box robustness. In addition to white-box scenarios, we evaluate the black-box robustness of latent diffusion models via transfer attacks, where we consider both prompt-transfer and model-transfer settings and possible defense mechanisms. However, all these explorations need a comprehensive benchmark dataset, which is missing in the literature. Therefore, to facilitate the research of the robustness of latent diffusion models, we propose two automatic dataset construction pipelines for two kinds of image editing models and release the whole dataset. Our code and dataset are available at \url{https://github.com/jpzhang1810/LDM-Robustness}.



## **17. CARSO: Counter-Adversarial Recall of Synthetic Observations**

cs.CV

20 pages, 5 figures, 10 tables; Update: removed visual artifacts from  some figures, fixed typos/capitalisation, typographic/pagination improvements

**SubmitDate**: 2023-06-14    [abs](http://arxiv.org/abs/2306.06081v2) [paper-pdf](http://arxiv.org/pdf/2306.06081v2)

**Authors**: Emanuele Ballarin, Alessio Ansuini, Luca Bortolussi

**Abstract**: In this paper, we propose a novel adversarial defence mechanism for image classification -- CARSO -- inspired by cues from cognitive neuroscience. The method is synergistically complementary to adversarial training and relies on knowledge of the internal representation of the attacked classifier. Exploiting a generative model for adversarial purification, conditioned on such representation, it samples reconstructions of inputs to be finally classified. Experimental evaluation by a well-established benchmark of varied, strong adaptive attacks, across diverse image datasets and classifier architectures, shows that CARSO is able to defend the classifier significantly better than state-of-the-art adversarial training alone -- with a tolerable clean accuracy toll. Furthermore, the defensive architecture succeeds in effectively shielding itself from unforeseen threats, and end-to-end attacks adapted to fool stochastic defences. Code and pre-trained models are available at https://github.com/emaballarin/CARSO .



## **18. White-Box Adversarial Policies in Deep Reinforcement Learning**

cs.AI

Code is available at  https://github.com/thestephencasper/white_box_rarl

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2209.02167v2) [paper-pdf](http://arxiv.org/pdf/2209.02167v2)

**Authors**: Stephen Casper, Taylor Killian, Gabriel Kreiman, Dylan Hadfield-Menell

**Abstract**: In reinforcement learning (RL), adversarial policies can be developed by training an adversarial agent to minimize a target agent's rewards. Prior work has studied black-box versions of these attacks where the adversary only observes the world state and treats the target agent as any other part of the environment. However, this does not take into account additional structure in the problem. In this work, we take inspiration from the literature on white-box attacks to train more effective adversarial policies. We study white-box adversarial policies and show that having access to a target agent's internal state can be useful for identifying its vulnerabilities. We make two contributions. (1) We introduce white-box adversarial policies where an attacker observes both a target's internal state and the world state at each timestep. We formulate ways of using these policies to attack agents in 2-player games and text-generating language models. (2) We demonstrate that these policies can achieve higher initial and asymptotic performance against a target agent than black-box controls. Code is available at https://github.com/thestephencasper/lm_white_box_attacks



## **19. Class Attribute Inference Attacks: Inferring Sensitive Class Information by Diffusion-Based Attribute Manipulations**

cs.LG

46 pages, 37 figures, 5 tables

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2303.09289v2) [paper-pdf](http://arxiv.org/pdf/2303.09289v2)

**Authors**: Lukas Struppek, Dominik Hintersdorf, Felix Friedrich, Manuel Brack, Patrick Schramowski, Kristian Kersting

**Abstract**: Neural network-based image classifiers are powerful tools for computer vision tasks, but they inadvertently reveal sensitive attribute information about their classes, raising concerns about their privacy. To investigate this privacy leakage, we introduce the first Class Attribute Inference Attack (CAIA), which leverages recent advances in text-to-image synthesis to infer sensitive attributes of individual classes in a black-box setting, while remaining competitive with related white-box attacks. Our extensive experiments in the face recognition domain show that CAIA can accurately infer undisclosed sensitive attributes, such as an individual's hair color, gender, and racial appearance, which are not part of the training labels. Interestingly, we demonstrate that adversarial robust models are even more vulnerable to such privacy leakage than standard models, indicating that a trade-off between robustness and privacy exists.



## **20. Finite Gaussian Neurons: Defending against adversarial attacks by making neural networks say "I don't know"**

cs.LG

PhD thesis

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2306.07796v1) [paper-pdf](http://arxiv.org/pdf/2306.07796v1)

**Authors**: Felix Grezes

**Abstract**: Since 2014, artificial neural networks have been known to be vulnerable to adversarial attacks, which can fool the network into producing wrong or nonsensical outputs by making humanly imperceptible alterations to inputs. While defenses against adversarial attacks have been proposed, they usually involve retraining a new neural network from scratch, a costly task. In this work, I introduce the Finite Gaussian Neuron (FGN), a novel neuron architecture for artificial neural networks. My works aims to: - easily convert existing models to Finite Gaussian Neuron architecture, - while preserving the existing model's behavior on real data, - and offering resistance against adversarial attacks. I show that converted and retrained Finite Gaussian Neural Networks (FGNN) always have lower confidence (i.e., are not overconfident) in their predictions over randomized and Fast Gradient Sign Method adversarial images when compared to classical neural networks, while maintaining high accuracy and confidence over real MNIST images. To further validate the capacity of Finite Gaussian Neurons to protect from adversarial attacks, I compare the behavior of FGNs to that of Bayesian Neural Networks against both randomized and adversarial images, and show how the behavior of the two architectures differs. Finally I show some limitations of the FGN models by testing them on the more complex SPEECHCOMMANDS task, against the stronger Carlini-Wagner and Projected Gradient Descent adversarial attacks.



## **21. Area is all you need: repeatable elements make stronger adversarial attacks**

cs.CV

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2306.07768v1) [paper-pdf](http://arxiv.org/pdf/2306.07768v1)

**Authors**: Dillon Niederhut

**Abstract**: Over the last decade, deep neural networks have achieved state of the art in computer vision tasks. These models, however, are susceptible to unusual inputs, known as adversarial examples, that cause them to misclassify or otherwise fail to detect objects. Here, we provide evidence that the increasing success of adversarial attacks is primarily due to increasing their size. We then demonstrate a method for generating the largest possible adversarial patch by building a adversarial pattern out of repeatable elements. This approach achieves a new state of the art in evading detection by YOLOv2 and YOLOv3. Finally, we present an experiment that fails to replicate the prior success of several attacks published in this field, and end with some comments on testing and reproducibility.



## **22. PromptBench: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts**

cs.CL

Technical report; 23 pages; code is at:  https://github.com/microsoft/promptbench

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2306.04528v2) [paper-pdf](http://arxiv.org/pdf/2306.04528v2)

**Authors**: Kaijie Zhu, Jindong Wang, Jiaheng Zhou, Zichen Wang, Hao Chen, Yidong Wang, Linyi Yang, Wei Ye, Neil Zhenqiang Gong, Yue Zhang, Xing Xie

**Abstract**: The increasing reliance on Large Language Models (LLMs) across academia and industry necessitates a comprehensive understanding of their robustness to prompts. In response to this vital need, we introduce PromptBench, a robustness benchmark designed to measure LLMs' resilience to adversarial prompts. This study uses a plethora of adversarial textual attacks targeting prompts across multiple levels: character, word, sentence, and semantic. These prompts are then employed in diverse tasks, such as sentiment analysis, natural language inference, reading comprehension, machine translation, and math problem-solving. Our study generates 4,032 adversarial prompts, meticulously evaluated over 8 tasks and 13 datasets, with 567,084 test samples in total. Our findings demonstrate that contemporary LLMs are vulnerable to adversarial prompts. Furthermore, we present comprehensive analysis to understand the mystery behind prompt robustness and its transferability. We then offer insightful robustness analysis and pragmatic recommendations for prompt composition, beneficial to both researchers and everyday users. We make our code, prompts, and methodologies to generate adversarial prompts publicly accessible, thereby enabling and encouraging collaborative exploration in this pivotal field: https://github.com/microsoft/promptbench.



## **23. Privacy Inference-Empowered Stealthy Backdoor Attack on Federated Learning under Non-IID Scenarios**

cs.LG

It can be accepted IJCNN

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2306.08011v1) [paper-pdf](http://arxiv.org/pdf/2306.08011v1)

**Authors**: Haochen Mei, Gaolei Li, Jun Wu, Longfei Zheng

**Abstract**: Federated learning (FL) naturally faces the problem of data heterogeneity in real-world scenarios, but this is often overlooked by studies on FL security and privacy. On the one hand, the effectiveness of backdoor attacks on FL may drop significantly under non-IID scenarios. On the other hand, malicious clients may steal private data through privacy inference attacks. Therefore, it is necessary to have a comprehensive perspective of data heterogeneity, backdoor, and privacy inference. In this paper, we propose a novel privacy inference-empowered stealthy backdoor attack (PI-SBA) scheme for FL under non-IID scenarios. Firstly, a diverse data reconstruction mechanism based on generative adversarial networks (GANs) is proposed to produce a supplementary dataset, which can improve the attacker's local data distribution and support more sophisticated strategies for backdoor attacks. Based on this, we design a source-specified backdoor learning (SSBL) strategy as a demonstration, allowing the adversary to arbitrarily specify which classes are susceptible to the backdoor trigger. Since the PI-SBA has an independent poisoned data synthesis process, it can be integrated into existing backdoor attacks to improve their effectiveness and stealthiness in non-IID scenarios. Extensive experiments based on MNIST, CIFAR10 and Youtube Aligned Face datasets demonstrate that the proposed PI-SBA scheme is effective in non-IID FL and stealthy against state-of-the-art defense methods.



## **24. Malafide: a novel adversarial convolutive noise attack against deepfake and spoofing detection systems**

eess.AS

Accepted at INTERSPEECH 2023

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2306.07655v1) [paper-pdf](http://arxiv.org/pdf/2306.07655v1)

**Authors**: Michele Panariello, Wanying Ge, Hemlata Tak, Massimiliano Todisco, Nicholas Evans

**Abstract**: We present Malafide, a universal adversarial attack against automatic speaker verification (ASV) spoofing countermeasures (CMs). By introducing convolutional noise using an optimised linear time-invariant filter, Malafide attacks can be used to compromise CM reliability while preserving other speech attributes such as quality and the speaker's voice. In contrast to other adversarial attacks proposed recently, Malafide filters are optimised independently of the input utterance and duration, are tuned instead to the underlying spoofing attack, and require the optimisation of only a small number of filter coefficients. Even so, they degrade CM performance estimates by an order of magnitude, even in black-box settings, and can also be configured to overcome integrated CM and ASV subsystems. Integrated solutions that use self-supervised learning CMs, however, are more robust, under both black-box and white-box settings.



## **25. DHBE: Data-free Holistic Backdoor Erasing in Deep Neural Networks via Restricted Adversarial Distillation**

cs.LG

It has been accepted by asiaccs

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2306.08009v1) [paper-pdf](http://arxiv.org/pdf/2306.08009v1)

**Authors**: Zhicong Yan, Shenghong Li, Ruijie Zhao, Yuan Tian, Yuanyuan Zhao

**Abstract**: Backdoor attacks have emerged as an urgent threat to Deep Neural Networks (DNNs), where victim DNNs are furtively implanted with malicious neurons that could be triggered by the adversary. To defend against backdoor attacks, many works establish a staged pipeline to remove backdoors from victim DNNs: inspecting, locating, and erasing. However, in a scenario where a few clean data can be accessible, such pipeline is fragile and cannot erase backdoors completely without sacrificing model accuracy. To address this issue, in this paper, we propose a novel data-free holistic backdoor erasing (DHBE) framework. Instead of the staged pipeline, the DHBE treats the backdoor erasing task as a unified adversarial procedure, which seeks equilibrium between two different competing processes: distillation and backdoor regularization. In distillation, the backdoored DNN is distilled into a proxy model, transferring its knowledge about clean data, yet backdoors are simultaneously transferred. In backdoor regularization, the proxy model is holistically regularized to prevent from infecting any possible backdoor transferred from distillation. These two processes jointly proceed with data-free adversarial optimization until a clean, high-accuracy proxy model is obtained. With the novel adversarial design, our framework demonstrates its superiority in three aspects: 1) minimal detriment to model accuracy, 2) high tolerance for hyperparameters, and 3) no demand for clean data. Extensive experiments on various backdoor attacks and datasets are performed to verify the effectiveness of the proposed framework. Code is available at \url{https://github.com/yanzhicong/DHBE}



## **26. A Hypergraph-Based Machine Learning Ensemble Network Intrusion Detection System**

cs.CR

This work has been submitted to the IEEE for possible publication.  Copyright may be transferred without notice, after which this version may no  longer be accessible

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2211.03933v2) [paper-pdf](http://arxiv.org/pdf/2211.03933v2)

**Authors**: Zong-Zhi Lin, Thomas D. Pike, Mark M. Bailey, Nathaniel D. Bastian

**Abstract**: Network intrusion detection systems (NIDS) to detect malicious attacks continue to meet challenges. NIDS are often developed offline while they face auto-generated port scan infiltration attempts, resulting in a significant time lag from adversarial adaption to NIDS response. To address these challenges, we use hypergraphs focused on internet protocol addresses and destination ports to capture evolving patterns of port scan attacks. The derived set of hypergraph-based metrics are then used to train an ensemble machine learning (ML) based NIDS that allows for real-time adaption in monitoring and detecting port scanning activities, other types of attacks, and adversarial intrusions at high accuracy, precision and recall performances. This ML adapting NIDS was developed through the combination of (1) intrusion examples, (2) NIDS update rules, (3) attack threshold choices to trigger NIDS retraining requests, and (4) a production environment with no prior knowledge of the nature of network traffic. 40 scenarios were auto-generated to evaluate the ML ensemble NIDS comprising three tree-based models. The resulting ML Ensemble NIDS was extended and evaluated with the CIC-IDS2017 dataset. Results show that under the model settings of an Update-ALL-NIDS rule (specifically retrain and update all the three models upon the same NIDS retraining request) the proposed ML ensemble NIDS evolved intelligently and produced the best results with nearly 100% detection performance throughout the simulation.



## **27. Extracting Cloud-based Model with Prior Knowledge**

cs.CR

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2306.04192v4) [paper-pdf](http://arxiv.org/pdf/2306.04192v4)

**Authors**: Shiqian Zhao, Kangjie Chen, Meng Hao, Jian Zhang, Guowen Xu, Hongwei Li, Tianwei Zhang

**Abstract**: Machine Learning-as-a-Service, a pay-as-you-go business pattern, is widely accepted by third-party users and developers. However, the open inference APIs may be utilized by malicious customers to conduct model extraction attacks, i.e., attackers can replicate a cloud-based black-box model merely via querying malicious examples. Existing model extraction attacks mainly depend on the posterior knowledge (i.e., predictions of query samples) from Oracle. Thus, they either require high query overhead to simulate the decision boundary, or suffer from generalization errors and overfitting problems due to query budget limitations. To mitigate it, this work proposes an efficient model extraction attack based on prior knowledge for the first time. The insight is that prior knowledge of unlabeled proxy datasets is conducive to the search for the decision boundary (e.g., informative samples). Specifically, we leverage self-supervised learning including autoencoder and contrastive learning to pre-compile the prior knowledge of the proxy dataset into the feature extractor of the substitute model. Then we adopt entropy to measure and sample the most informative examples to query the target model. Our design leverages both prior and posterior knowledge to extract the model and thus eliminates generalizability errors and overfitting problems. We conduct extensive experiments on open APIs like Traffic Recognition, Flower Recognition, Moderation Recognition, and NSFW Recognition from real-world platforms, Azure and Clarifai. The experimental results demonstrate the effectiveness and efficiency of our attack. For example, our attack achieves 95.1% fidelity with merely 1.8K queries (cost 2.16$) on the NSFW Recognition API. Also, the adversarial examples generated with our substitute model have better transferability than others, which reveals that our scheme is more conducive to downstream attacks.



## **28. I See Dead People: Gray-Box Adversarial Attack on Image-To-Text Models**

cs.CV

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2306.07591v1) [paper-pdf](http://arxiv.org/pdf/2306.07591v1)

**Authors**: Raz Lapid, Moshe Sipper

**Abstract**: Modern image-to-text systems typically adopt the encoder-decoder framework, which comprises two main components: an image encoder, responsible for extracting image features, and a transformer-based decoder, used for generating captions. Taking inspiration from the analysis of neural networks' robustness against adversarial perturbations, we propose a novel gray-box algorithm for creating adversarial examples in image-to-text models. Unlike image classification tasks that have a finite set of class labels, finding visually similar adversarial examples in an image-to-text task poses greater challenges because the captioning system allows for a virtually infinite space of possible captions. In this paper, we present a gray-box adversarial attack on image-to-text, both untargeted and targeted. We formulate the process of discovering adversarial perturbations as an optimization problem that uses only the image-encoder component, meaning the proposed attack is language-model agnostic. Through experiments conducted on the ViT-GPT2 model, which is the most-used image-to-text model in Hugging Face, and the Flickr30k dataset, we demonstrate that our proposed attack successfully generates visually similar adversarial examples, both with untargeted and targeted captions. Notably, our attack operates in a gray-box manner, requiring no knowledge about the decoder module. We also show that our attacks fool the popular open-source platform Hugging Face.



## **29. How Secure is Your Website? A Comprehensive Investigation on CAPTCHA Providers and Solving Services**

cs.CR

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2306.07543v1) [paper-pdf](http://arxiv.org/pdf/2306.07543v1)

**Authors**: Rui Jin, Lin Huang, Jikang Duan, Wei Zhao, Yong Liao, Pengyuan Zhou

**Abstract**: Completely Automated Public Turing Test To Tell Computers and Humans Apart (CAPTCHA) has been implemented on many websites to identify between harmful automated bots and legitimate users. However, the revenue generated by the bots has turned circumventing CAPTCHAs into a lucrative business. Although earlier studies provided information about text-based CAPTCHAs and the associated CAPTCHA-solving services, a lot has changed in the past decade regarding content, suppliers, and solvers of CAPTCHA. We have conducted a comprehensive investigation of the latest third-party CAPTCHA providers and CAPTCHA-solving services' attacks. We dug into the details of CAPTCHA-As-a-Service and the latest CAPTCHA-solving services and carried out adversarial experiments on CAPTCHAs and CAPTCHA solvers. The experiment results show a worrying fact: most latest CAPTCHAs are vulnerable to both human solvers and automated solvers. New CAPTCHAs based on hard AI problems and behavior analysis are needed to stop CAPTCHA solvers.



## **30. Adversarial Attacks on the Interpretation of Neuron Activation Maximization**

cs.LG

**SubmitDate**: 2023-06-12    [abs](http://arxiv.org/abs/2306.07397v1) [paper-pdf](http://arxiv.org/pdf/2306.07397v1)

**Authors**: Geraldin Nanfack, Alexander Fulleringer, Jonathan Marty, Michael Eickenberg, Eugene Belilovsky

**Abstract**: The internal functional behavior of trained Deep Neural Networks is notoriously difficult to interpret. Activation-maximization approaches are one set of techniques used to interpret and analyze trained deep-learning models. These consist in finding inputs that maximally activate a given neuron or feature map. These inputs can be selected from a data set or obtained by optimization. However, interpretability methods may be subject to being deceived. In this work, we consider the concept of an adversary manipulating a model for the purpose of deceiving the interpretation. We propose an optimization framework for performing this manipulation and demonstrate a number of ways that popular activation-maximization interpretation techniques associated with CNNs can be manipulated to change the interpretations, shedding light on the reliability of these methods.



## **31. Gaussian Membership Inference Privacy**

cs.LG

The first two authors contributed equally

**SubmitDate**: 2023-06-12    [abs](http://arxiv.org/abs/2306.07273v1) [paper-pdf](http://arxiv.org/pdf/2306.07273v1)

**Authors**: Tobias Leemann, Martin Pawelczyk, Gjergji Kasneci

**Abstract**: We propose a new privacy notion called $f$-Membership Inference Privacy ($f$-MIP), which explicitly considers the capabilities of realistic adversaries under the membership inference attack threat model. By doing so $f$-MIP offers interpretable privacy guarantees and improved utility (e.g., better classification accuracy). Our novel theoretical analysis of likelihood ratio-based membership inference attacks on noisy stochastic gradient descent (SGD) results in a parametric family of $f$-MIP guarantees that we refer to as $\mu$-Gaussian Membership Inference Privacy ($\mu$-GMIP). Our analysis additionally yields an analytical membership inference attack that offers distinct advantages over previous approaches. First, unlike existing methods, our attack does not require training hundreds of shadow models to approximate the likelihood ratio. Second, our analytical attack enables straightforward auditing of our privacy notion $f$-MIP. Finally, our analysis emphasizes the importance of various factors, such as hyperparameters (e.g., batch size, number of model parameters) and data specific characteristics in controlling an attacker's success in reliably inferring a given point's membership to the training set. We demonstrate the effectiveness of our method on models trained across vision and tabular datasets.



## **32. When Vision Fails: Text Attacks Against ViT and OCR**

cs.CR

**SubmitDate**: 2023-06-12    [abs](http://arxiv.org/abs/2306.07033v1) [paper-pdf](http://arxiv.org/pdf/2306.07033v1)

**Authors**: Nicholas Boucher, Jenny Blessing, Ilia Shumailov, Ross Anderson, Nicolas Papernot

**Abstract**: While text-based machine learning models that operate on visual inputs of rendered text have become robust against a wide range of existing attacks, we show that they are still vulnerable to visual adversarial examples encoded as text. We use the Unicode functionality of combining diacritical marks to manipulate encoded text so that small visual perturbations appear when the text is rendered. We show how a genetic algorithm can be used to generate visual adversarial examples in a black-box setting, and conduct a user study to establish that the model-fooling adversarial examples do not affect human comprehension. We demonstrate the effectiveness of these attacks in the real world by creating adversarial examples against production models published by Facebook, Microsoft, IBM, and Google.



## **33. A Linear Reconstruction Approach for Attribute Inference Attacks against Synthetic Data**

cs.LG

**SubmitDate**: 2023-06-12    [abs](http://arxiv.org/abs/2301.10053v2) [paper-pdf](http://arxiv.org/pdf/2301.10053v2)

**Authors**: Meenatchi Sundaram Muthu Selva Annamalai, Andrea Gadotti, Luc Rocher

**Abstract**: Recent advances in synthetic data generation (SDG) have been hailed as a solution to the difficult problem of sharing sensitive data while protecting privacy. SDG aims to learn statistical properties of real data in order to generate "artificial" data that are structurally and statistically similar to sensitive data. However, prior research suggests that inference attacks on synthetic data can undermine privacy, but only for specific outlier records. In this work, we introduce a new attribute inference attack against synthetic data. The attack is based on linear reconstruction methods for aggregate statistics, which target all records in the dataset, not only outliers. We evaluate our attack on state-of-the-art SDG algorithms, including Probabilistic Graphical Models, Generative Adversarial Networks, and recent differentially private SDG mechanisms. By defining a formal privacy game, we show that our attack can be highly accurate even on arbitrary records, and that this is the result of individual information leakage (as opposed to population-level inference). We then systematically evaluate the tradeoff between protecting privacy and preserving statistical utility. Our findings suggest that current SDG methods cannot consistently provide sufficient privacy protection against inference attacks while retaining reasonable utility. The best method evaluated, a differentially private SDG mechanism, can provide both protection against inference attacks and reasonable utility, but only in very specific settings. Lastly, we show that releasing a larger number of synthetic records can improve utility but at the cost of making attacks far more effective.



## **34. How robust accuracy suffers from certified training with convex relaxations**

cs.LG

**SubmitDate**: 2023-06-12    [abs](http://arxiv.org/abs/2306.06995v1) [paper-pdf](http://arxiv.org/pdf/2306.06995v1)

**Authors**: Piersilvio De Bartolomeis, Jacob Clarysse, Amartya Sanyal, Fanny Yang

**Abstract**: Adversarial attacks pose significant threats to deploying state-of-the-art classifiers in safety-critical applications. Two classes of methods have emerged to address this issue: empirical defences and certified defences. Although certified defences come with robustness guarantees, empirical defences such as adversarial training enjoy much higher popularity among practitioners. In this paper, we systematically compare the standard and robust error of these two robust training paradigms across multiple computer vision tasks. We show that in most tasks and for both $\mathscr{l}_\infty$-ball and $\mathscr{l}_2$-ball threat models, certified training with convex relaxations suffers from worse standard and robust error than adversarial training. We further explore how the error gap between certified and adversarial training depends on the threat model and the data distribution. In particular, besides the perturbation budget, we identify as important factors the shape of the perturbation set and the implicit margin of the data distribution. We support our arguments with extensive ablations on both synthetic and image datasets.



## **35. Backdooring Neural Code Search**

cs.SE

Accepted to the 61st Annual Meeting of the Association for  Computational Linguistics (ACL 2023)

**SubmitDate**: 2023-06-12    [abs](http://arxiv.org/abs/2305.17506v2) [paper-pdf](http://arxiv.org/pdf/2305.17506v2)

**Authors**: Weisong Sun, Yuchen Chen, Guanhong Tao, Chunrong Fang, Xiangyu Zhang, Quanjun Zhang, Bin Luo

**Abstract**: Reusing off-the-shelf code snippets from online repositories is a common practice, which significantly enhances the productivity of software developers. To find desired code snippets, developers resort to code search engines through natural language queries. Neural code search models are hence behind many such engines. These models are based on deep learning and gain substantial attention due to their impressive performance. However, the security aspect of these models is rarely studied. Particularly, an adversary can inject a backdoor in neural code search models, which return buggy or even vulnerable code with security/privacy issues. This may impact the downstream software (e.g., stock trading systems and autonomous driving) and cause financial loss and/or life-threatening incidents. In this paper, we demonstrate such attacks are feasible and can be quite stealthy. By simply modifying one variable/function name, the attacker can make buggy/vulnerable code rank in the top 11%. Our attack BADCODE features a special trigger generation and injection procedure, making the attack more effective and stealthy. The evaluation is conducted on two neural code search models and the results show our attack outperforms baselines by 60%. Our user study demonstrates that our attack is more stealthy than the baseline by two times based on the F1 score.



## **36. Graph Agent Network: Empowering Nodes with Decentralized Communications Capabilities for Adversarial Resilience**

cs.LG

**SubmitDate**: 2023-06-12    [abs](http://arxiv.org/abs/2306.06909v1) [paper-pdf](http://arxiv.org/pdf/2306.06909v1)

**Authors**: Ao Liu, Wenshan Li, Tao Li, Beibei Li, Hanyuan Huang, Guangquan Xu, Pan Zhou

**Abstract**: End-to-end training with global optimization have popularized graph neural networks (GNNs) for node classification, yet inadvertently introduced vulnerabilities to adversarial edge-perturbing attacks. Adversaries can exploit the inherent opened interfaces of GNNs' input and output, perturbing critical edges and thus manipulating the classification results. Current defenses, due to their persistent utilization of global-optimization-based end-to-end training schemes, inherently encapsulate the vulnerabilities of GNNs. This is specifically evidenced in their inability to defend against targeted secondary attacks. In this paper, we propose the Graph Agent Network (GAgN) to address the aforementioned vulnerabilities of GNNs. GAgN is a graph-structured agent network in which each node is designed as an 1-hop-view agent. Through the decentralized interactions between agents, they can learn to infer global perceptions to perform tasks including inferring embeddings, degrees and neighbor relationships for given nodes. This empowers nodes to filtering adversarial edges while carrying out classification tasks. Furthermore, agents' limited view prevents malicious messages from propagating globally in GAgN, thereby resisting global-optimization-based secondary attacks. We prove that single-hidden-layer multilayer perceptrons (MLPs) are theoretically sufficient to achieve these functionalities. Experimental results show that GAgN effectively implements all its intended capabilities and, compared to state-of-the-art defenses, achieves optimal classification accuracy on the perturbed datasets.



## **37. GAN-CAN: A Novel Attack to Behavior-Based Driver Authentication Systems**

cs.CR

16 pages, 6 figures

**SubmitDate**: 2023-06-12    [abs](http://arxiv.org/abs/2306.05923v2) [paper-pdf](http://arxiv.org/pdf/2306.05923v2)

**Authors**: Emad Efatinasab, Francesco Marchiori, Denis Donadel, Alessandro Brighente, Mauro Conti

**Abstract**: For many years, car keys have been the sole mean of authentication in vehicles. Whether the access control process is physical or wireless, entrusting the ownership of a vehicle to a single token is prone to stealing attempts. For this reason, many researchers started developing behavior-based authentication systems. By collecting data in a moving vehicle, Deep Learning (DL) models can recognize patterns in the data and identify drivers based on their driving behavior. This can be used as an anti-theft system, as a thief would exhibit a different driving style compared to the vehicle owner's. However, the assumption that an attacker cannot replicate the legitimate driver behavior falls under certain conditions.   In this paper, we propose GAN-CAN, the first attack capable of fooling state-of-the-art behavior-based driver authentication systems in a vehicle. Based on the adversary's knowledge, we propose different GAN-CAN implementations. Our attack leverages the lack of security in the Controller Area Network (CAN) to inject suitably designed time-series data to mimic the legitimate driver. Our design of the malicious time series results from the combination of different Generative Adversarial Networks (GANs) and our study on the safety importance of the injected values during the attack. We tested GAN-CAN in an improved version of the most efficient driver behavior-based authentication model in the literature. We prove that our attack can fool it with an attack success rate of up to 0.99. We show how an attacker, without prior knowledge of the authentication system, can steal a car by deploying GAN-CAN in an off-the-shelf system in under 22 minutes.



## **38. Trustworthy Artificial Intelligence Framework for Proactive Detection and Risk Explanation of Cyber Attacks in Smart Grid**

cs.CR

Submitted for peer review

**SubmitDate**: 2023-06-12    [abs](http://arxiv.org/abs/2306.07993v1) [paper-pdf](http://arxiv.org/pdf/2306.07993v1)

**Authors**: Md. Shirajum Munir, Sachin Shetty, Danda B. Rawat

**Abstract**: The rapid growth of distributed energy resources (DERs), such as renewable energy sources, generators, consumers, and prosumers in the smart grid infrastructure, poses significant cybersecurity and trust challenges to the grid controller. Consequently, it is crucial to identify adversarial tactics and measure the strength of the attacker's DER. To enable a trustworthy smart grid controller, this work investigates a trustworthy artificial intelligence (AI) mechanism for proactive identification and explanation of the cyber risk caused by the control/status message of DERs. Thus, proposing and developing a trustworthy AI framework to facilitate the deployment of any AI algorithms for detecting potential cyber threats and analyzing root causes based on Shapley value interpretation while dynamically quantifying the risk of an attack based on Ward's minimum variance formula. The experiment with a state-of-the-art dataset establishes the proposed framework as a trustworthy AI by fulfilling the capabilities of reliability, fairness, explainability, transparency, reproducibility, and accountability.



## **39. Asymptotically Optimal Adversarial Strategies for the Probability Estimation Framework**

quant-ph

54 pages

**SubmitDate**: 2023-06-11    [abs](http://arxiv.org/abs/2306.06802v1) [paper-pdf](http://arxiv.org/pdf/2306.06802v1)

**Authors**: Soumyadip Patra, Peter Bierhorst

**Abstract**: The Probability Estimation Framework involves direct estimation of the probability of occurrences of outcomes conditioned on measurement settings and side information. It is a powerful tool for certifying randomness in quantum non-locality experiments. In this paper, we present a self-contained proof of the asymptotic optimality of the method. Our approach refines earlier results to allow a better characterisation of optimal adversarial attacks on the protocol. We apply these results to the (2,2,2) Bell scenario, obtaining an analytic characterisation of the optimal adversarial attacks bound by no-signalling principles, while also demonstrating the asymptotic robustness of the PEF method to deviations from expected experimental behaviour. We also study extensions of the analysis to quantum-limited adversaries in the (2,2,2) Bell scenario and no-signalling adversaries in higher $(n,m,k)$ Bell scenarios.



## **40. Adversarial Reconnaissance Mitigation and Modeling**

cs.CR

**SubmitDate**: 2023-06-11    [abs](http://arxiv.org/abs/2306.06769v1) [paper-pdf](http://arxiv.org/pdf/2306.06769v1)

**Authors**: Shanto Roy, Nazia Sharmin, Mohammad Sujan Miah, Jaime C Acosta, Christopher Kiekintveld, Aron Laszka

**Abstract**: Adversarial reconnaissance is a crucial step in sophisticated cyber-attacks as it enables threat actors to find the weakest points of otherwise well-defended systems. To thwart reconnaissance, defenders can employ cyber deception techniques, such as deploying honeypots. In recent years, researchers have made great strides in developing game-theoretic models to find optimal deception strategies. However, most of these game-theoretic models build on relatively simple models of adversarial reconnaissance -- even though reconnaissance should be a focus point as the very purpose of deception is to thwart reconnaissance. In this paper, we first discuss effective cyber reconnaissance mitigation techniques including deception strategies and beyond. Then we provide a review of the literature on deception games from the perspective of modeling adversarial reconnaissance, highlighting key aspects of reconnaissance that have not been adequately captured in prior work. We then describe a probability-theory based model of the adversaries' belief formation and illustrate using numerical examples that this model can capture key aspects of adversarial reconnaissance. We believe that our review and belief model can serve as a stepping stone for developing more realistic and practical deception games.



## **41. Securing Visually-Aware Recommender Systems: An Adversarial Image Reconstruction and Detection Framework**

cs.CV

**SubmitDate**: 2023-06-11    [abs](http://arxiv.org/abs/2306.07992v1) [paper-pdf](http://arxiv.org/pdf/2306.07992v1)

**Authors**: Minglei Yin, Bin Liu, Neil Zhenqiang Gong, Xin Li

**Abstract**: With rich visual data, such as images, becoming readily associated with items, visually-aware recommendation systems (VARS) have been widely used in different applications. Recent studies have shown that VARS are vulnerable to item-image adversarial attacks, which add human-imperceptible perturbations to the clean images associated with those items. Attacks on VARS pose new security challenges to a wide range of applications such as e-Commerce and social networks where VARS are widely used. How to secure VARS from such adversarial attacks becomes a critical problem. Currently, there is still a lack of systematic study on how to design secure defense strategies against visual attacks on VARS. In this paper, we attempt to fill this gap by proposing an adversarial image reconstruction and detection framework to secure VARS. Our proposed method can simultaneously (1) secure VARS from adversarial attacks characterized by local perturbations by image reconstruction based on global vision transformers; and (2) accurately detect adversarial examples using a novel contrastive learning approach. Meanwhile, our framework is designed to be used as both a filter and a detector so that they can be jointly trained to improve the flexibility of our defense strategy to a variety of attacks and VARS models. We have conducted extensive experimental studies with two popular attack methods (FGSM and PGD). Our experimental results on two real-world datasets show that our defense strategy against visual attacks is effective and outperforms existing methods on different attacks. Moreover, our method can detect adversarial examples with high accuracy.



## **42. Neural Architecture Design and Robustness: A Dataset**

cs.LG

ICLR 2023; project page: http://robustness.vision/

**SubmitDate**: 2023-06-11    [abs](http://arxiv.org/abs/2306.06712v1) [paper-pdf](http://arxiv.org/pdf/2306.06712v1)

**Authors**: Steffen Jung, Jovita Lukasik, Margret Keuper

**Abstract**: Deep learning models have proven to be successful in a wide range of machine learning tasks. Yet, they are often highly sensitive to perturbations on the input data which can lead to incorrect decisions with high confidence, hampering their deployment for practical use-cases. Thus, finding architectures that are (more) robust against perturbations has received much attention in recent years. Just like the search for well-performing architectures in terms of clean accuracy, this usually involves a tedious trial-and-error process with one additional challenge: the evaluation of a network's robustness is significantly more expensive than its evaluation for clean accuracy. Thus, the aim of this paper is to facilitate better streamlined research on architectural design choices with respect to their impact on robustness as well as, for example, the evaluation of surrogate measures for robustness. We therefore borrow one of the most commonly considered search spaces for neural architecture search for image classification, NAS-Bench-201, which contains a manageable size of 6466 non-isomorphic network designs. We evaluate all these networks on a range of common adversarial attacks and corruption types and introduce a database on neural architecture design and robustness evaluations. We further present three exemplary use cases of this dataset, in which we (i) benchmark robustness measurements based on Jacobian and Hessian matrices for their robustness predictability, (ii) perform neural architecture search on robust accuracies, and (iii) provide an initial analysis of how architectural design choices affect robustness. We find that carefully crafting the topology of a network can have substantial impact on its robustness, where networks with the same parameter count range in mean adversarial robust accuracy from 20%-41%. Code and data is available at http://robustness.vision/.



## **43. EvadeDroid: A Practical Evasion Attack on Machine Learning for Black-box Android Malware Detection**

cs.LG

**SubmitDate**: 2023-06-11    [abs](http://arxiv.org/abs/2110.03301v3) [paper-pdf](http://arxiv.org/pdf/2110.03301v3)

**Authors**: Hamid Bostani, Veelasha Moonsamy

**Abstract**: Over the last decade, researchers have extensively explored the vulnerabilities of Android malware detectors to adversarial examples through the development of evasion attacks; however, the practicality of these attacks in real-world scenarios remains arguable. The majority of studies have assumed attackers know the details of the target classifiers used for malware detection, while in reality, malicious actors have limited access to the target classifiers. This paper introduces EvadeDroid, a practical decision-based adversarial attack designed to effectively evade black-box Android malware detectors in real-world scenarios. In addition to generating real-world adversarial malware, the proposed evasion attack can also preserve the functionality of the original malware applications (apps). EvadeDroid constructs a collection of functionality-preserving transformations derived from benign donors that share opcode-level similarity with malware apps by leveraging an n-gram-based approach. These transformations are then used to morph malware instances into benign ones via an iterative and incremental manipulation strategy. The proposed manipulation technique is a novel, query-efficient optimization algorithm that can find and inject optimal sequences of transformations into malware apps. Our empirical evaluation demonstrates the efficacy of EvadeDroid under soft- and hard-label attacks. Furthermore, EvadeDroid exhibits the capability to generate real-world adversarial examples that can effectively evade a wide range of black-box ML-based malware detectors with minimal query requirements. Finally, we show that the proposed problem-space adversarial attack is able to preserve its stealthiness against five popular commercial antiviruses, thus demonstrating its feasibility in the real world.



## **44. Level Up with RealAEs: Leveraging Domain Constraints in Feature Space to Strengthen Robustness of Android Malware Detection**

cs.LG

**SubmitDate**: 2023-06-11    [abs](http://arxiv.org/abs/2205.15128v3) [paper-pdf](http://arxiv.org/pdf/2205.15128v3)

**Authors**: Hamid Bostani, Zhengyu Zhao, Zhuoran Liu, Veelasha Moonsamy

**Abstract**: The vulnerability to adversarial examples remains one major obstacle for Machine Learning (ML)-based Android malware detection. Realistic attacks in the Android malware domain create Realizable Adversarial Examples (RealAEs), i.e., AEs that satisfy the domain constraints of Android malware. Recent studies have shown that using such RealAEs in Adversarial Training (AT) is more effective in defending against realistic attacks than using unrealizable AEs (unRealAEs). This is because RealAEs allow defenders to explore certain pockets in the feature space that are vulnerable to realistic attacks. However, existing defenses commonly generate RealAEs in the problem space, which is known to be time-consuming and impractical for AT. In this paper, we propose to generate RealAEs in the feature space, leading to a simpler and more efficient solution. Our approach is driven by a novel interpretation of Android domain constraints in the feature space. More concretely, our defense first learns feature-space domain constraints by extracting meaningful feature dependencies from data and then applies them to generating feature-space RealAEs during AT. Extensive experiments on DREBIN, a well-known Android malware detector, demonstrate that our new defense outperforms not only unRealAE-based AT but also the state-of-the-art defense that relies on non-uniform perturbations. We further validate the ability of our learned feature-space domain constraints in representing Android malware properties by showing that our feature-space domain constraints can help distinguish RealAEs from unRealAEs.



## **45. Attacking Cooperative Multi-Agent Reinforcement Learning by Adversarial Minority Influence**

cs.LG

**SubmitDate**: 2023-06-11    [abs](http://arxiv.org/abs/2302.03322v2) [paper-pdf](http://arxiv.org/pdf/2302.03322v2)

**Authors**: Simin Li, Jun Guo, Jingqiao Xiu, Pu Feng, Xin Yu, Aishan Liu, Wenjun Wu, Xianglong Liu

**Abstract**: This study probes the vulnerabilities of cooperative multi-agent reinforcement learning (c-MARL) under adversarial attacks, a critical determinant of c-MARL's worst-case performance prior to real-world implementation. Current observation-based attacks, constrained by white-box assumptions, overlook c-MARL's complex multi-agent interactions and cooperative objectives, resulting in impractical and limited attack capabilities. To address these shortcomes, we propose Adversarial Minority Influence (AMI), a practical and strong for c-MARL. AMI is a practical black-box attack and can be launched without knowing victim parameters. AMI is also strong by considering the complex multi-agent interaction and the cooperative goal of agents, enabling a single adversarial agent to unilaterally misleads majority victims to form targeted worst-case cooperation. This mirrors minority influence phenomena in social psychology. To achieve maximum deviation in victim policies under complex agent-wise interactions, our unilateral attack aims to characterize and maximize the impact of the adversary on the victims. This is achieved by adapting a unilateral agent-wise relation metric derived from mutual information, thereby mitigating the adverse effects of victim influence on the adversary. To lead the victims into a jointly detrimental scenario, our targeted attack deceives victims into a long-term, cooperatively harmful situation by guiding each victim towards a specific target, determined through a trial-and-error process executed by a reinforcement learning agent. Through AMI, we achieve the first successful attack against real-world robot swarms and effectively fool agents in simulated environments into collectively worst-case scenarios, including Starcraft II and Multi-agent Mujoco. The source code and demonstrations can be found at: https://github.com/DIG-Beihang/AMI.



## **46. Defense Against Adversarial Attacks on Audio DeepFake Detection**

cs.SD

Accepted to INTERSPEECH 2023

**SubmitDate**: 2023-06-10    [abs](http://arxiv.org/abs/2212.14597v2) [paper-pdf](http://arxiv.org/pdf/2212.14597v2)

**Authors**: Piotr Kawa, Marcin Plata, Piotr Syga

**Abstract**: Audio DeepFakes (DF) are artificially generated utterances created using deep learning, with the primary aim of fooling the listeners in a highly convincing manner. Their quality is sufficient to pose a severe threat in terms of security and privacy, including the reliability of news or defamation. Multiple neural network-based methods to detect generated speech have been proposed to prevent the threats. In this work, we cover the topic of adversarial attacks, which decrease the performance of detectors by adding superficial (difficult to spot by a human) changes to input data. Our contribution contains evaluating the robustness of 3 detection architectures against adversarial attacks in two scenarios (white-box and using transferability) and enhancing it later by using adversarial training performed by our novel adaptive training. Moreover, one of the investigated architectures is RawNet3, which, to the best of our knowledge, we adapted for the first time to DeepFake detection.



## **47. The Defense of Networked Targets in General Lotto games**

cs.GT

**SubmitDate**: 2023-06-10    [abs](http://arxiv.org/abs/2306.06485v1) [paper-pdf](http://arxiv.org/pdf/2306.06485v1)

**Authors**: Adel Aghajan, Keith Paarporn, Jason R. Marden

**Abstract**: Ensuring the security of networked systems is a significant problem, considering the susceptibility of modern infrastructures and technologies to adversarial interference. A central component of this problem is how defensive resources should be allocated to mitigate the severity of potential attacks on the system. In this paper, we consider this in the context of a General Lotto game, where a defender and attacker deploys resources on the nodes of a network, and the objective is to secure as many links as possible. The defender secures a link only if it out-competes the attacker on both of its associated nodes. For bipartite networks, we completely characterize equilibrium payoffs and strategies for both the defender and attacker. Surprisingly, the resulting payoffs are the same for any bipartite graph. On arbitrary network structures, we provide lower and upper bounds on the defender's max-min value. Notably, the equilibrium payoff from bipartite networks serves as the lower bound. These results suggest that more connected networks are easier to defend against attacks. We confirm these findings with simulations that compute deterministic allocation strategies on large random networks. This also highlights the importance of randomization in the equilibrium strategies.



## **48. NeRFool: Uncovering the Vulnerability of Generalizable Neural Radiance Fields against Adversarial Perturbations**

cs.CV

Accepted by ICML 2023

**SubmitDate**: 2023-06-10    [abs](http://arxiv.org/abs/2306.06359v1) [paper-pdf](http://arxiv.org/pdf/2306.06359v1)

**Authors**: Yonggan Fu, Ye Yuan, Souvik Kundu, Shang Wu, Shunyao Zhang, Yingyan Lin

**Abstract**: Generalizable Neural Radiance Fields (GNeRF) are one of the most promising real-world solutions for novel view synthesis, thanks to their cross-scene generalization capability and thus the possibility of instant rendering on new scenes. While adversarial robustness is essential for real-world applications, little study has been devoted to understanding its implication on GNeRF. We hypothesize that because GNeRF is implemented by conditioning on the source views from new scenes, which are often acquired from the Internet or third-party providers, there are potential new security concerns regarding its real-world applications. Meanwhile, existing understanding and solutions for neural networks' adversarial robustness may not be applicable to GNeRF, due to its 3D nature and uniquely diverse operations. To this end, we present NeRFool, which to the best of our knowledge is the first work that sets out to understand the adversarial robustness of GNeRF. Specifically, NeRFool unveils the vulnerability patterns and important insights regarding GNeRF's adversarial robustness. Built upon the above insights gained from NeRFool, we further develop NeRFool+, which integrates two techniques capable of effectively attacking GNeRF across a wide range of target views, and provide guidelines for defending against our proposed attacks. We believe that our NeRFool/NeRFool+ lays the initial foundation for future innovations in developing robust real-world GNeRF solutions. Our codes are available at: https://github.com/GATECH-EIC/NeRFool.



## **49. Differentially private sliced inverse regression in the federated paradigm**

stat.ME

**SubmitDate**: 2023-06-10    [abs](http://arxiv.org/abs/2306.06324v1) [paper-pdf](http://arxiv.org/pdf/2306.06324v1)

**Authors**: Shuaida He, Jiarui Zhang, Xin Chen

**Abstract**: We extend the celebrated sliced inverse regression to address the challenges of decentralized data, prioritizing privacy and communication efficiency. Our approach, federated sliced inverse regression (FSIR), facilitates collaborative estimation of the sufficient dimension reduction subspace among multiple clients, solely sharing local estimates to protect sensitive datasets from exposure. To guard against potential adversary attacks, FSIR further employs diverse perturbation strategies, including a novel multivariate Gaussian mechanism that guarantees differential privacy at a low cost of statistical accuracy. Additionally, FSIR naturally incorporates a collaborative variable screening step, enabling effective handling of high-dimensional client data. Theoretical properties of FSIR are established for both low-dimensional and high-dimensional settings, supported by extensive numerical experiments and real data analysis.



## **50. The Certification Paradox: Certifications Admit Better Attacks**

cs.LG

16 pages, 6 figures

**SubmitDate**: 2023-06-09    [abs](http://arxiv.org/abs/2302.04379v2) [paper-pdf](http://arxiv.org/pdf/2302.04379v2)

**Authors**: Andrew C. Cullen, Shijie Liu, Paul Montague, Sarah M. Erfani, Benjamin I. P. Rubinstein

**Abstract**: In guaranteeing that no adversarial examples exist within a bounded region, certification mechanisms play an important role in demonstrating the robustness of neural networks. In this work we ask: Could certifications have any unintended consequences, through exposing additional information about certified models? We answer this question in the affirmative, demonstrating that certifications not only measure model robustness but also present a new attack surface. We propose \emph{Certification Aware Attacks}, that produce smaller adversarial perturbations more than twice as frequently as any prior approach, when launched against certified models. Our attacks achieve an up to $34\%$ reduction in the median perturbation norm (comparing target and attack instances), while requiring $90 \%$ less computational time than approaches like PGD. That our attacks achieve such significant reductions in perturbation size and computational cost highlights an apparent paradox in deploying certification mechanisms. We end the paper with a discussion of how these risks could potentially be mitigated.



