# Latest Adversarial Attack Papers
**update at 2022-04-30 06:31:27**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. UNBUS: Uncertainty-aware Deep Botnet Detection System in Presence of Perturbed Samples**

cs.CR

8 pages, 5 figures, 5 Tables

**SubmitDate**: 2022-04-28    [paper-pdf](http://arxiv.org/pdf/2204.09502v2)

**Authors**: Rahim Taheri

**Abstracts**: A rising number of botnet families have been successfully detected using deep learning architectures. While the variety of attacks increases, these architectures should become more robust against attacks. They have been proven to be very sensitive to small but well constructed perturbations in the input. Botnet detection requires extremely low false-positive rates (FPR), which are not commonly attainable in contemporary deep learning. Attackers try to increase the FPRs by making poisoned samples. The majority of recent research has focused on the use of model loss functions to build adversarial examples and robust models. In this paper, two LSTM-based classification algorithms for botnet classification with an accuracy higher than 98% are presented. Then, the adversarial attack is proposed, which reduces the accuracy to about 30%. Then, by examining the methods for computing the uncertainty, the defense method is proposed to increase the accuracy to about 70%. By using the deep ensemble and stochastic weight averaging quantification methods it has been investigated the uncertainty of the accuracy in the proposed methods.



## **2. Deepfake Forensics via An Adversarial Game**

cs.CV

Accepted by IEEE Transactions on Image Processing; 13 pages, 4  figures

**SubmitDate**: 2022-04-28    [paper-pdf](http://arxiv.org/pdf/2103.13567v2)

**Authors**: Zhi Wang, Yiwen Guo, Wangmeng Zuo

**Abstracts**: With the progress in AI-based facial forgery (i.e., deepfake), people are increasingly concerned about its abuse. Albeit effort has been made for training classification (also known as deepfake detection) models to recognize such forgeries, existing models suffer from poor generalization to unseen forgery technologies and high sensitivity to changes in image/video quality. In this paper, we advocate adversarial training for improving the generalization ability to both unseen facial forgeries and unseen image/video qualities. We believe training with samples that are adversarially crafted to attack the classification models improves the generalization ability considerably. Considering that AI-based face manipulation often leads to high-frequency artifacts that can be easily spotted by models yet difficult to generalize, we further propose a new adversarial training method that attempts to blur out these specific artifacts, by introducing pixel-wise Gaussian blurring models. With adversarial training, the classification models are forced to learn more discriminative and generalizable features, and the effectiveness of our method can be verified by plenty of empirical evidence. Our code will be made publicly available.



## **3. Adversarial Fine-tune with Dynamically Regulated Adversary**

cs.LG

**SubmitDate**: 2022-04-28    [paper-pdf](http://arxiv.org/pdf/2204.13232v1)

**Authors**: Pengyue Hou, Ming Zhou, Jie Han, Petr Musilek, Xingyu Li

**Abstracts**: Adversarial training is an effective method to boost model robustness to malicious, adversarial attacks. However, such improvement in model robustness often leads to a significant sacrifice of standard performance on clean images. In many real-world applications such as health diagnosis and autonomous surgical robotics, the standard performance is more valued over model robustness against such extremely malicious attacks. This leads to the question: To what extent we can boost model robustness without sacrificing standard performance? This work tackles this problem and proposes a simple yet effective transfer learning-based adversarial training strategy that disentangles the negative effects of adversarial samples on model's standard performance. In addition, we introduce a training-friendly adversarial attack algorithm, which facilitates the boost of adversarial robustness without introducing significant training complexity. Extensive experimentation indicates that the proposed method outperforms previous adversarial training algorithms towards the target: to improve model robustness while preserving model's standard performance on clean data.



## **4. An Adversarial Attack Analysis on Malicious Advertisement URL Detection Framework**

cs.LG

13

**SubmitDate**: 2022-04-27    [paper-pdf](http://arxiv.org/pdf/2204.13172v1)

**Authors**: Ehsan Nowroozi, Abhishek, Mohammadreza Mohammadi, Mauro Conti

**Abstracts**: Malicious advertisement URLs pose a security risk since they are the source of cyber-attacks, and the need to address this issue is growing in both industry and academia. Generally, the attacker delivers an attack vector to the user by means of an email, an advertisement link or any other means of communication and directs them to a malicious website to steal sensitive information and to defraud them. Existing malicious URL detection techniques are limited and to handle unseen features as well as generalize to test data. In this study, we extract a novel set of lexical and web-scrapped features and employ machine learning technique to set up system for fraudulent advertisement URLs detection. The combination set of six different kinds of features precisely overcome the obfuscation in fraudulent URL classification. Based on different statistical properties, we use twelve different formatted datasets for detection, prediction and classification task. We extend our prediction analysis for mismatched and unlabelled datasets. For this framework, we analyze the performance of four machine learning techniques: Random Forest, Gradient Boost, XGBoost and AdaBoost in the detection part. With our proposed method, we can achieve a false negative rate as low as 0.0037 while maintaining high accuracy of 99.63%. Moreover, we devise a novel unsupervised technique for data clustering using K- Means algorithm for the visual analysis. This paper analyses the vulnerability of decision tree-based models using the limited knowledge attack scenario. We considered the exploratory attack and implemented Zeroth Order Optimization adversarial attack on the detection models.



## **5. SSR-GNNs: Stroke-based Sketch Representation with Graph Neural Networks**

cs.CV

**SubmitDate**: 2022-04-27    [paper-pdf](http://arxiv.org/pdf/2204.13153v1)

**Authors**: Sheng Cheng, Yi Ren, Yezhou Yang

**Abstracts**: This paper follows cognitive studies to investigate a graph representation for sketches, where the information of strokes, i.e., parts of a sketch, are encoded on vertices and information of inter-stroke on edges. The resultant graph representation facilitates the training of a Graph Neural Networks for classification tasks, and achieves accuracy and robustness comparable to the state-of-the-art against translation and rotation attacks, as well as stronger attacks on graph vertices and topologies, i.e., modifications and addition of strokes, all without resorting to adversarial training. Prior studies on sketches, e.g., graph transformers, encode control points of stroke on vertices, which are not invariant to spatial transformations. In contrary, we encode vertices and edges using pairwise distances among control points to achieve invariance. Compared with existing generative sketch model for one-shot classification, our method does not rely on run-time statistical inference. Lastly, the proposed representation enables generation of novel sketches that are structurally similar to while separable from the existing dataset.



## **6. Defending Against Person Hiding Adversarial Patch Attack with a Universal White Frame**

cs.CV

Submitted by NeurIPS 2021 with response letter to the anonymous  reviewers' comments

**SubmitDate**: 2022-04-27    [paper-pdf](http://arxiv.org/pdf/2204.13004v1)

**Authors**: Youngjoon Yu, Hong Joo Lee, Hakmin Lee, Yong Man Ro

**Abstracts**: Object detection has attracted great attention in the computer vision area and has emerged as an indispensable component in many vision systems. In the era of deep learning, many high-performance object detection networks have been proposed. Although these detection networks show high performance, they are vulnerable to adversarial patch attacks. Changing the pixels in a restricted region can easily fool the detection network in the physical world. In particular, person-hiding attacks are emerging as a serious problem in many safety-critical applications such as autonomous driving and surveillance systems. Although it is necessary to defend against an adversarial patch attack, very few efforts have been dedicated to defending against person-hiding attacks. To tackle the problem, in this paper, we propose a novel defense strategy that mitigates a person-hiding attack by optimizing defense patterns, while previous methods optimize the model. In the proposed method, a frame-shaped pattern called a 'universal white frame' (UWF) is optimized and placed on the outside of the image. To defend against adversarial patch attacks, UWF should have three properties (i) suppressing the effect of the adversarial patch, (ii) maintaining its original prediction, and (iii) applicable regardless of images. To satisfy the aforementioned properties, we propose a novel pattern optimization algorithm that can defend against the adversarial patch. Through comprehensive experiments, we demonstrate that the proposed method effectively defends against the adversarial patch attack.



## **7. The MeVer DeepFake Detection Service: Lessons Learnt from Developing and Deploying in the Wild**

cs.CV

10 pages, 6 figures

**SubmitDate**: 2022-04-27    [paper-pdf](http://arxiv.org/pdf/2204.12816v1)

**Authors**: Spyridon Baxevanakis, Giorgos Kordopatis-Zilos, Panagiotis Galopoulos, Lazaros Apostolidis, Killian Levacher, Ipek B. Schlicht, Denis Teyssou, Ioannis Kompatsiaris, Symeon Papadopoulos

**Abstracts**: Enabled by recent improvements in generation methodologies, DeepFakes have become mainstream due to their increasingly better visual quality, the increase in easy-to-use generation tools and the rapid dissemination through social media. This fact poses a severe threat to our societies with the potential to erode social cohesion and influence our democracies. To mitigate the threat, numerous DeepFake detection schemes have been introduced in the literature but very few provide a web service that can be used in the wild. In this paper, we introduce the MeVer DeepFake detection service, a web service detecting deep learning manipulations in images and video. We present the design and implementation of the proposed processing pipeline that involves a model ensemble scheme, and we endow the service with a model card for transparency. Experimental results show that our service performs robustly on the three benchmark datasets while being vulnerable to Adversarial Attacks. Finally, we outline our experience and lessons learned when deploying a research system into production in the hopes that it will be useful to other academic and industry teams.



## **8. Improving the Transferability of Adversarial Examples with Restructure Embedded Patches**

cs.CV

**SubmitDate**: 2022-04-27    [paper-pdf](http://arxiv.org/pdf/2204.12680v1)

**Authors**: Huipeng Zhou, Yu-an Tan, Yajie Wang, Haoran Lyu, Shangbo Wu, Yuanzhang Li

**Abstracts**: Vision transformers (ViTs) have demonstrated impressive performance in various computer vision tasks. However, the adversarial examples generated by ViTs are challenging to transfer to other networks with different structures. Recent attack methods do not consider the specificity of ViTs architecture and self-attention mechanism, which leads to poor transferability of the generated adversarial samples by ViTs. We attack the unique self-attention mechanism in ViTs by restructuring the embedded patches of the input. The restructured embedded patches enable the self-attention mechanism to obtain more diverse patches connections and help ViTs keep regions of interest on the object. Therefore, we propose an attack method against the unique self-attention mechanism in ViTs, called Self-Attention Patches Restructure (SAPR). Our method is simple to implement yet efficient and applicable to any self-attention based network and gradient transferability-based attack methods. We evaluate attack transferability on black-box models with different structures. The result show that our method generates adversarial examples on white-box ViTs with higher transferability and higher image quality. Our research advances the development of black-box transfer attacks on ViTs and demonstrates the feasibility of using white-box ViTs to attack other black-box models.



## **9. Data Bootstrapping Approaches to Improve Low Resource Abusive Language Detection for Indic Languages**

cs.CL

Accepted at HT '22: 33rd ACM Conference on Hypertext and Social Media

**SubmitDate**: 2022-04-26    [paper-pdf](http://arxiv.org/pdf/2204.12543v1)

**Authors**: Mithun Das, Somnath Banerjee, Animesh Mukherjee

**Abstracts**: Abusive language is a growing concern in many social media platforms. Repeated exposure to abusive speech has created physiological effects on the target users. Thus, the problem of abusive language should be addressed in all forms for online peace and safety. While extensive research exists in abusive speech detection, most studies focus on English. Recently, many smearing incidents have occurred in India, which provoked diverse forms of abusive speech in online space in various languages based on the geographic location. Therefore it is essential to deal with such malicious content. In this paper, to bridge the gap, we demonstrate a large-scale analysis of multilingual abusive speech in Indic languages. We examine different interlingual transfer mechanisms and observe the performance of various multilingual models for abusive speech detection for eight different Indic languages. We also experiment to show how robust these models are on adversarial attacks. Finally, we conduct an in-depth error analysis by looking into the models' misclassified posts across various settings. We have made our code and models public for other researchers.



## **10. Restricted Black-box Adversarial Attack Against DeepFake Face Swapping**

cs.CV

**SubmitDate**: 2022-04-26    [paper-pdf](http://arxiv.org/pdf/2204.12347v1)

**Authors**: Junhao Dong, Yuan Wang, Jianhuang Lai, Xiaohua Xie

**Abstracts**: DeepFake face swapping presents a significant threat to online security and social media, which can replace the source face in an arbitrary photo/video with the target face of an entirely different person. In order to prevent this fraud, some researchers have begun to study the adversarial methods against DeepFake or face manipulation. However, existing works focus on the white-box setting or the black-box setting driven by abundant queries, which severely limits the practical application of these methods. To tackle this problem, we introduce a practical adversarial attack that does not require any queries to the facial image forgery model. Our method is built on a substitute model persuing for face reconstruction and then transfers adversarial examples from the substitute model directly to inaccessible black-box DeepFake models. Specially, we propose the Transferable Cycle Adversary Generative Adversarial Network (TCA-GAN) to construct the adversarial perturbation for disrupting unknown DeepFake systems. We also present a novel post-regularization module for enhancing the transferability of generated adversarial examples. To comprehensively measure the effectiveness of our approaches, we construct a challenging benchmark of DeepFake adversarial attacks for future development. Extensive experiments impressively show that the proposed adversarial attack method makes the visual quality of DeepFake face images plummet so that they are easier to be detected by humans and algorithms. Moreover, we demonstrate that the proposed algorithm can be generalized to offer face image protection against various face translation methods.



## **11. Boosting Adversarial Transferability of MLP-Mixer**

cs.CV

**SubmitDate**: 2022-04-26    [paper-pdf](http://arxiv.org/pdf/2204.12204v1)

**Authors**: Haoran Lyu, Yajie Wang, Yu-an Tan, Huipeng Zhou, Yuhang Zhao, Quanxin Zhang

**Abstracts**: The security of models based on new architectures such as MLP-Mixer and ViTs needs to be studied urgently. However, most of the current researches are mainly aimed at the adversarial attack against ViTs, and there is still relatively little adversarial work on MLP-mixer. We propose an adversarial attack method against MLP-Mixer called Maxwell's demon Attack (MA). MA breaks the channel-mixing and token-mixing mechanism of MLP-Mixer by controlling the part input of MLP-Mixer's each Mixer layer, and disturbs MLP-Mixer to obtain the main information of images. Our method can mask the part input of the Mixer layer, avoid overfitting of the adversarial examples to the source model, and improve the transferability of cross-architecture. Extensive experimental evaluation demonstrates the effectiveness and superior performance of the proposed MA. Our method can be easily combined with existing methods and can improve the transferability by up to 38.0% on MLP-based ResMLP. Adversarial examples produced by our method on MLP-Mixer are able to exceed the transferability of adversarial examples produced using DenseNet against CNNs. To the best of our knowledge, we are the first work to study adversarial transferability of MLP-Mixer.



## **12. Mixed Strategies for Security Games with General Defending Requirements**

cs.GT

Accepted by IJCAI-2022

**SubmitDate**: 2022-04-26    [paper-pdf](http://arxiv.org/pdf/2204.12158v1)

**Authors**: Rufan Bai, Haoxing Lin, Xinyu Yang, Xiaowei Wu, Minming Li, Weijia Jia

**Abstracts**: The Stackelberg security game is played between a defender and an attacker, where the defender needs to allocate a limited amount of resources to multiple targets in order to minimize the loss due to adversarial attack by the attacker. While allowing targets to have different values, classic settings often assume uniform requirements to defend the targets. This enables existing results that study mixed strategies (randomized allocation algorithms) to adopt a compact representation of the mixed strategies.   In this work, we initiate the study of mixed strategies for the security games in which the targets can have different defending requirements. In contrast to the case of uniform defending requirement, for which an optimal mixed strategy can be computed efficiently, we show that computing the optimal mixed strategy is NP-hard for the general defending requirements setting. However, we show that strong upper and lower bounds for the optimal mixed strategy defending result can be derived. We propose an efficient close-to-optimal Patching algorithm that computes mixed strategies that use only few pure strategies. We also study the setting when the game is played on a network and resource sharing is enabled between neighboring targets. Our experimental results demonstrate the effectiveness of our algorithm in several large real-world datasets.



## **13. Source-independent quantum random number generator against detector blinding attacks**

quant-ph

14 pages, 7 figures, 6 tables, comments are welcome

**SubmitDate**: 2022-04-26    [paper-pdf](http://arxiv.org/pdf/2204.12156v1)

**Authors**: Wen-Bo Liu, Yu-Shuo Lu, Yao Fu, Si-Cheng Huang, Ze-Jie Yin, Kun Jiang, Hua-Lei Yin, Zeng-Bing Chen

**Abstracts**: Randomness, mainly in the form of random numbers, is the fundamental prerequisite for the security of many cryptographic tasks. Quantum randomness can be extracted even if adversaries are fully aware of the protocol and even control the randomness source. However, an adversary can further manipulate the randomness via detector blinding attacks, which are a hacking attack suffered by protocols with trusted detectors. Here, by treating no-click events as valid error events, we propose a quantum random number generation protocol that can simultaneously address source vulnerability and ferocious detector blinding attacks. The method can be extended to high-dimensional random number generation. We experimentally demonstrate the ability of our protocol to generate random numbers for two-dimensional measurement with a generation speed of 0.515 Mbps, which is two orders of magnitude higher than that of device-independent protocols that can address both issues of imperfect sources and imperfect detectors.



## **14. Self-recoverable Adversarial Examples: A New Effective Protection Mechanism in Social Networks**

cs.CV

13 pages, 11 figures

**SubmitDate**: 2022-04-26    [paper-pdf](http://arxiv.org/pdf/2204.12050v1)

**Authors**: Jiawei Zhang, Jinwei Wang, Hao Wang, Xiangyang Luo

**Abstracts**: Malicious intelligent algorithms greatly threaten the security of social users' privacy by detecting and analyzing the uploaded photos to social network platforms. The destruction to DNNs brought by the adversarial attack sparks the potential that adversarial examples serve as a new protection mechanism for privacy security in social networks. However, the existing adversarial example does not have recoverability for serving as an effective protection mechanism. To address this issue, we propose a recoverable generative adversarial network to generate self-recoverable adversarial examples. By modeling the adversarial attack and recovery as a united task, our method can minimize the error of the recovered examples while maximizing the attack ability, resulting in better recoverability of adversarial examples. To further boost the recoverability of these examples, we exploit a dimension reducer to optimize the distribution of adversarial perturbation. The experimental results prove that the adversarial examples generated by the proposed method present superior recoverability, attack ability, and robustness on different datasets and network architectures, which ensure its effectiveness as a protection mechanism in social networks.



## **15. Can Rationalization Improve Robustness?**

cs.CL

Accepted to NAACL 2022

**SubmitDate**: 2022-04-25    [paper-pdf](http://arxiv.org/pdf/2204.11790v1)

**Authors**: Howard Chen, Jacqueline He, Karthik Narasimhan, Danqi Chen

**Abstracts**: A growing line of work has investigated the development of neural NLP models that can produce rationales--subsets of input that can explain their model predictions. In this paper, we ask whether such rationale models can also provide robustness to adversarial attacks in addition to their interpretable nature. Since these models need to first generate rationales ("rationalizer") before making predictions ("predictor"), they have the potential to ignore noise or adversarially added text by simply masking it out of the generated rationale. To this end, we systematically generate various types of 'AddText' attacks for both token and sentence-level rationalization tasks, and perform an extensive empirical evaluation of state-of-the-art rationale models across five different tasks. Our experiments reveal that the rationale models show the promise to improve robustness, while they struggle in certain scenarios--when the rationalizer is sensitive to positional bias or lexical choices of attack text. Further, leveraging human rationale as supervision does not always translate to better performance. Our study is a first step towards exploring the interplay between interpretability and robustness in the rationalize-then-predict framework.



## **16. Discovering Exfiltration Paths Using Reinforcement Learning with Attack Graphs**

cs.CR

The 5th IEEE Conference on Dependable and Secure Computing (IEEE DSC  2022)

**SubmitDate**: 2022-04-25    [paper-pdf](http://arxiv.org/pdf/2201.12416v2)

**Authors**: Tyler Cody, Abdul Rahman, Christopher Redino, Lanxiao Huang, Ryan Clark, Akshay Kakkar, Deepak Kushwaha, Paul Park, Peter Beling, Edward Bowen

**Abstracts**: Reinforcement learning (RL), in conjunction with attack graphs and cyber terrain, are used to develop reward and state associated with determination of optimal paths for exfiltration of data in enterprise networks. This work builds on previous crown jewels (CJ) identification that focused on the target goal of computing optimal paths that adversaries may traverse toward compromising CJs or hosts within their proximity. This work inverts the previous CJ approach based on the assumption that data has been stolen and now must be quietly exfiltrated from the network. RL is utilized to support the development of a reward function based on the identification of those paths where adversaries desire reduced detection. Results demonstrate promising performance for a sizable network environment.



## **17. Reconstructing Training Data with Informed Adversaries**

cs.CR

Published at "2022 IEEE Symposium on Security and Privacy (SP)"

**SubmitDate**: 2022-04-25    [paper-pdf](http://arxiv.org/pdf/2201.04845v2)

**Authors**: Borja Balle, Giovanni Cherubin, Jamie Hayes

**Abstracts**: Given access to a machine learning model, can an adversary reconstruct the model's training data? This work studies this question from the lens of a powerful informed adversary who knows all the training data points except one. By instantiating concrete attacks, we show it is feasible to reconstruct the remaining data point in this stringent threat model. For convex models (e.g. logistic regression), reconstruction attacks are simple and can be derived in closed-form. For more general models (e.g. neural networks), we propose an attack strategy based on training a reconstructor network that receives as input the weights of the model under attack and produces as output the target data point. We demonstrate the effectiveness of our attack on image classifiers trained on MNIST and CIFAR-10, and systematically investigate which factors of standard machine learning pipelines affect reconstruction success. Finally, we theoretically investigate what amount of differential privacy suffices to mitigate reconstruction attacks by informed adversaries. Our work provides an effective reconstruction attack that model developers can use to assess memorization of individual points in general settings beyond those considered in previous works (e.g. generative language models or access to training gradients); it shows that standard models have the capacity to store enough information to enable high-fidelity reconstruction of training data points; and it demonstrates that differential privacy can successfully mitigate such attacks in a parameter regime where utility degradation is minimal.



## **18. A Simple Structure For Building A Robust Model**

cs.CV

10 pages, 3 figures, 4 tables

**SubmitDate**: 2022-04-25    [paper-pdf](http://arxiv.org/pdf/2204.11596v1)

**Authors**: Xiao Tan, JingBo Gao, Ruolin Li

**Abstracts**: As deep learning applications, especially programs of computer vision, are increasingly deployed in our lives, we have to think more urgently about the security of these applications.One effective way to improve the security of deep learning models is to perform adversarial training, which allows the model to be compatible with samples that are deliberately created for use in attacking the model.Based on this, we propose a simple architecture to build a model with a certain degree of robustness, which improves the robustness of the trained network by adding an adversarial sample detection network for cooperative training.At the same time, we design a new data sampling strategy that incorporates multiple existing attacks, allowing the model to adapt to many different adversarial attacks with a single training.We conducted some experiments to test the effectiveness of this design based on Cifar10 dataset, and the results indicate that it has some degree of positive effect on the robustness of the model.Our code could be found at https://github.com/dowdyboy/simple_structure_for_robust_model.



## **19. Dominating Vertical Collaborative Learning Systems**

cs.CR

**SubmitDate**: 2022-04-25    [paper-pdf](http://arxiv.org/pdf/2201.02775v2)

**Authors**: Qi Pang, Yuanyuan Yuan, Shuai Wang

**Abstracts**: Vertical collaborative learning system also known as vertical federated learning (VFL) system has recently become prominent as a concept to process data distributed across many individual sources without the need to centralize it. Multiple participants collaboratively train models based on their local data in a privacy-preserving manner. To date, VFL has become a de facto solution to securely learn a model among organizations, allowing knowledge to be shared without compromising privacy of any individual organizations.   Despite the prosperous development of VFL systems, we find that certain inputs of a participant, named adversarial dominating inputs (ADIs), can dominate the joint inference towards the direction of the adversary's will and force other (victim) participants to make negligible contributions, losing rewards that are usually offered regarding the importance of their contributions in collaborative learning scenarios.   We conduct a systematic study on ADIs by first proving their existence in typical VFL systems. We then propose gradient-based methods to synthesize ADIs of various formats and exploit common VFL systems. We further launch greybox fuzz testing, guided by the resiliency score of "victim" participants, to perturb adversary-controlled inputs and systematically explore the VFL attack surface in a privacy-preserving manner. We conduct an in-depth study on the influence of critical parameters and settings in synthesizing ADIs. Our study reveals new VFL attack opportunities, promoting the identification of unknown threats before breaches and building more secure VFL systems.



## **20. Real or Virtual: A Video Conferencing Background Manipulation-Detection System**

cs.CV

34 pages. arXiv admin note: text overlap with arXiv:2106.15130

**SubmitDate**: 2022-04-25    [paper-pdf](http://arxiv.org/pdf/2204.11853v1)

**Authors**: Ehsan Nowroozi, Yassine Mekdad, Mauro Conti, Simone Milani, Selcuk Uluagac, Berrin Yanikoglu

**Abstracts**: Recently, the popularity and wide use of the last-generation video conferencing technologies created an exponential growth in its market size. Such technology allows participants in different geographic regions to have a virtual face-to-face meeting. Additionally, it enables users to employ a virtual background to conceal their own environment due to privacy concerns or to reduce distractions, particularly in professional settings. Nevertheless, in scenarios where the users should not hide their actual locations, they may mislead other participants by claiming their virtual background as a real one. Therefore, it is crucial to develop tools and strategies to detect the authenticity of the considered virtual background. In this paper, we present a detection strategy to distinguish between real and virtual video conferencing user backgrounds. We demonstrate that our detector is robust against two attack scenarios. The first scenario considers the case where the detector is unaware about the attacks and inn the second scenario, we make the detector aware of the adversarial attacks, which we refer to Adversarial Multimedia Forensics (i.e, the forensically-edited frames are included in the training set). Given the lack of publicly available dataset of virtual and real backgrounds for video conferencing, we created our own dataset and made them publicly available [1]. Then, we demonstrate the robustness of our detector against different adversarial attacks that the adversary considers. Ultimately, our detector's performance is significant against the CRSPAM1372 [2] features, and post-processing operations such as geometric transformations with different quality factors that the attacker may choose. Moreover, our performance results shows that we can perfectly identify a real from a virtual background with an accuracy of 99.80%.



## **21. Improving Deep Learning Model Robustness Against Adversarial Attack by Increasing the Network Capacity**

cs.LG

**SubmitDate**: 2022-04-24    [paper-pdf](http://arxiv.org/pdf/2204.11357v1)

**Authors**: Marco Marchetti, Edmond S. L. Ho

**Abstracts**: Nowadays, we are more and more reliant on Deep Learning (DL) models and thus it is essential to safeguard the security of these systems. This paper explores the security issues in Deep Learning and analyses, through the use of experiments, the way forward to build more resilient models. Experiments are conducted to identify the strengths and weaknesses of a new approach to improve the robustness of DL models against adversarial attacks. The results show improvements and new ideas that can be used as recommendations for researchers and practitioners to create increasingly better DL algorithms.



## **22. A Comprehensive Test Pattern Generation Approach Exploiting SAT Attack for Logic Locking**

cs.CR

10 pages, 8 figures

**SubmitDate**: 2022-04-24    [paper-pdf](http://arxiv.org/pdf/2204.11307v1)

**Authors**: Yadi Zhong, Ujjwal Guin

**Abstracts**: The need for reducing manufacturing defect escape in today's safety-critical applications requires increased fault coverage. However, generating a test set using commercial automatic test pattern generation (ATPG) tools that lead to zero-defect escape is still an open problem. It is challenging to detect all stuck-at faults to reach 100% fault coverage. In parallel, the hardware security community has been actively involved in developing solutions for logic locking to prevent IP piracy. Locks (e.g., XOR gates) are inserted in different locations of the netlist so that an adversary cannot determine the secret key. Unfortunately, the Boolean satisfiability (SAT) based attack, introduced in [1], can break different logic locking schemes in minutes. In this paper, we propose a novel test pattern generation approach using the powerful SAT attack on logic locking. A stuck-at fault is modeled as a locked gate with a secret key. Our modeling of stuck-at faults preserves the property of fault activation and propagation. We show that the input pattern that determines the key is a test for the stuck-at fault. We propose two different approaches for test pattern generation. First, a single stuck-at fault is targeted, and a corresponding locked circuit with one key bit is created. This approach generates one test pattern per fault. Second, we consider a group of faults and convert the circuit to its locked version with multiple key bits. The inputs obtained from the SAT tool are the test set for detecting this group of faults. Our approach is able to find test patterns for hard-to-detect faults that were previously failed in commercial ATPG tools. The proposed test pattern generation approach can efficiently detect redundant faults present in a circuit. We demonstrate the effectiveness of the approach on ITC'99 benchmarks. The results show that we can achieve a perfect fault coverage reaching 100%.



## **23. Dictionary Attacks on Speaker Verification**

cs.SD

Manuscript and supplement, currently under review

**SubmitDate**: 2022-04-24    [paper-pdf](http://arxiv.org/pdf/2204.11304v1)

**Authors**: Mirko Marras, Pawel Korus, Anubhav Jain, Nasir Memon

**Abstracts**: In this paper, we propose dictionary attacks against speaker verification - a novel attack vector that aims to match a large fraction of speaker population by chance. We introduce a generic formulation of the attack that can be used with various speech representations and threat models. The attacker uses adversarial optimization to maximize raw similarity of speaker embeddings between a seed speech sample and a proxy population. The resulting master voice successfully matches a non-trivial fraction of people in an unknown population. Adversarial waveforms obtained with our approach can match on average 69% of females and 38% of males enrolled in the target system at a strict decision threshold calibrated to yield false alarm rate of 1%. By using the attack with a black-box voice cloning system, we obtain master voices that are effective in the most challenging conditions and transferable between speaker encoders. We also show that, combined with multiple attempts, this attack opens even more to serious issues on the security of these systems.



## **24. Sampling-based Fast Gradient Rescaling Method for Highly Transferable Adversarial Attacks**

cs.CV

The writing and experiment of the article need to be further  strengthened

**SubmitDate**: 2022-04-24    [paper-pdf](http://arxiv.org/pdf/2204.02887v2)

**Authors**: Xu Han, Anmin Liu, Yifeng Xiong, Yanbo Fan, Kun He

**Abstracts**: Deep neural networks have shown to be very vulnerable to adversarial examples crafted by adding human-imperceptible perturbations to benign inputs. After achieving impressive attack success rates in the white-box setting, more focus is shifted to black-box attacks. In either case, the common gradient-based approaches generally use the $sign$ function to generate perturbations at the end of the process. However, only a few works pay attention to the limitation of the $sign$ function. Deviation between the original gradient and the generated noises may lead to inaccurate gradient update estimation and suboptimal solutions for adversarial transferability, which is crucial for black-box attacks. To address this issue, we propose a Sampling-based Fast Gradient Rescaling Method (S-FGRM) to improve the transferability of the crafted adversarial examples. Specifically, we use data rescaling to substitute the inefficient $sign$ function in gradient-based attacks without extra computational cost. We also propose a Depth First Sampling method to eliminate the fluctuation of rescaling and stabilize the gradient update. Our method can be used in any gradient-based optimizations and is extensible to be integrated with various input transformation or ensemble methods for further improving the adversarial transferability. Extensive experiments on the standard ImageNet dataset show that our S-FGRM could significantly boost the transferability of gradient-based attacks and outperform the state-of-the-art baselines.



## **25. Eliminating Backdoor Triggers for Deep Neural Networks Using Attention Relation Graph Distillation**

cs.LG

**SubmitDate**: 2022-04-24    [paper-pdf](http://arxiv.org/pdf/2204.09975v2)

**Authors**: Jun Xia, Ting Wang, Jiepin Ding, Xian Wei, Mingsong Chen

**Abstracts**: Due to the prosperity of Artificial Intelligence (AI) techniques, more and more backdoors are designed by adversaries to attack Deep Neural Networks (DNNs).Although the state-of-the-art method Neural Attention Distillation (NAD) can effectively erase backdoor triggers from DNNs, it still suffers from non-negligible Attack Success Rate (ASR) together with lowered classification ACCuracy (ACC), since NAD focuses on backdoor defense using attention features (i.e., attention maps) of the same order. In this paper, we introduce a novel backdoor defense framework named Attention Relation Graph Distillation (ARGD), which fully explores the correlation among attention features with different orders using our proposed Attention Relation Graphs (ARGs). Based on the alignment of ARGs between both teacher and student models during knowledge distillation, ARGD can eradicate more backdoor triggers than NAD. Comprehensive experimental results show that, against six latest backdoor attacks, ARGD outperforms NAD by up to 94.85% reduction in ASR, while ACC can be improved by up to 3.23%.



## **26. Stochastic Variance Reduced Ensemble Adversarial Attack for Boosting the Adversarial Transferability**

cs.LG

11 pages, 6 figures, accepted by CVPR 2022

**SubmitDate**: 2022-04-23    [paper-pdf](http://arxiv.org/pdf/2111.10752v2)

**Authors**: Yifeng Xiong, Jiadong Lin, Min Zhang, John E. Hopcroft, Kun He

**Abstracts**: The black-box adversarial attack has attracted impressive attention for its practical use in the field of deep learning security. Meanwhile, it is very challenging as there is no access to the network architecture or internal weights of the target model. Based on the hypothesis that if an example remains adversarial for multiple models, then it is more likely to transfer the attack capability to other models, the ensemble-based adversarial attack methods are efficient and widely used for black-box attacks. However, ways of ensemble attack are rather less investigated, and existing ensemble attacks simply fuse the outputs of all the models evenly. In this work, we treat the iterative ensemble attack as a stochastic gradient descent optimization process, in which the variance of the gradients on different models may lead to poor local optima. To this end, we propose a novel attack method called the stochastic variance reduced ensemble (SVRE) attack, which could reduce the gradient variance of the ensemble models and take full advantage of the ensemble attack. Empirical results on the standard ImageNet dataset demonstrate that the proposed method could boost the adversarial transferability and outperforms existing ensemble attacks significantly. Code is available at https://github.com/JHL-HUST/SVRE.



## **27. Certifiably Robust Variational Autoencoders**

stat.ML

12 pages and appendix

**SubmitDate**: 2022-04-23    [paper-pdf](http://arxiv.org/pdf/2102.07559v3)

**Authors**: Ben Barrett, Alexander Camuto, Matthew Willetts, Tom Rainforth

**Abstracts**: We introduce an approach for training Variational Autoencoders (VAEs) that are certifiably robust to adversarial attack. Specifically, we first derive actionable bounds on the minimal size of an input perturbation required to change a VAE's reconstruction by more than an allowed amount, with these bounds depending on certain key parameters such as the Lipschitz constants of the encoder and decoder. We then show how these parameters can be controlled, thereby providing a mechanism to ensure \textit{a priori} that a VAE will attain a desired level of robustness. Moreover, we extend this to a complete practical approach for training such VAEs to ensure our criteria are met. Critically, our method allows one to specify a desired level of robustness \emph{upfront} and then train a VAE that is guaranteed to achieve this robustness. We further demonstrate that these Lipschitz--constrained VAEs are more robust to attack than standard VAEs in practice.



## **28. Smart App Attack: Hacking Deep Learning Models in Android Apps**

cs.LG

Accepted to IEEE Transactions on Information Forensics and Security.  This is a preprint version, the copyright belongs to The Institute of  Electrical and Electronics Engineers

**SubmitDate**: 2022-04-23    [paper-pdf](http://arxiv.org/pdf/2204.11075v1)

**Authors**: Yujin Huang, Chunyang Chen

**Abstracts**: On-device deep learning is rapidly gaining popularity in mobile applications. Compared to offloading deep learning from smartphones to the cloud, on-device deep learning enables offline model inference while preserving user privacy. However, such mechanisms inevitably store models on users' smartphones and may invite adversarial attacks as they are accessible to attackers. Due to the characteristic of the on-device model, most existing adversarial attacks cannot be directly applied for on-device models. In this paper, we introduce a grey-box adversarial attack framework to hack on-device models by crafting highly similar binary classification models based on identified transfer learning approaches and pre-trained models from TensorFlow Hub. We evaluate the attack effectiveness and generality in terms of four different settings including pre-trained models, datasets, transfer learning approaches and adversarial attack algorithms. The results demonstrate that the proposed attacks remain effective regardless of different settings, and significantly outperform state-of-the-art baselines. We further conduct an empirical study on real-world deep learning mobile apps collected from Google Play. Among 53 apps adopting transfer learning, we find that 71.7\% of them can be successfully attacked, which includes popular ones in medicine, automation, and finance categories with critical usage scenarios. The results call for the awareness and actions of deep learning mobile app developers to secure the on-device models. The code of this work is available at https://github.com/Jinxhy/SmartAppAttack



## **29. Towards Data-Free Model Stealing in a Hard Label Setting**

cs.CR

CVPR 2022, Project Page: https://sites.google.com/view/dfms-hl

**SubmitDate**: 2022-04-23    [paper-pdf](http://arxiv.org/pdf/2204.11022v1)

**Authors**: Sunandini Sanyal, Sravanti Addepalli, R. Venkatesh Babu

**Abstracts**: Machine learning models deployed as a service (MLaaS) are susceptible to model stealing attacks, where an adversary attempts to steal the model within a restricted access framework. While existing attacks demonstrate near-perfect clone-model performance using softmax predictions of the classification network, most of the APIs allow access to only the top-1 labels. In this work, we show that it is indeed possible to steal Machine Learning models by accessing only top-1 predictions (Hard Label setting) as well, without access to model gradients (Black-Box setting) or even the training dataset (Data-Free setting) within a low query budget. We propose a novel GAN-based framework that trains the student and generator in tandem to steal the model effectively while overcoming the challenge of the hard label setting by utilizing gradients of the clone network as a proxy to the victim's gradients. We propose to overcome the large query costs associated with a typical Data-Free setting by utilizing publicly available (potentially unrelated) datasets as a weak image prior. We additionally show that even in the absence of such data, it is possible to achieve state-of-the-art results within a low query budget using synthetically crafted samples. We are the first to demonstrate the scalability of Model Stealing in a restricted access setting on a 100 class dataset as well.



## **30. GFCL: A GRU-based Federated Continual Learning Framework against Adversarial Attacks in IoV**

cs.LG

11 pages, 12 figures, 3 tables; This paper has been submitted to IEEE  Internet of Things Journal

**SubmitDate**: 2022-04-23    [paper-pdf](http://arxiv.org/pdf/2204.11010v1)

**Authors**: Anum Talpur, Mohan Gurusamy

**Abstracts**: The integration of ML in 5G-based Internet of Vehicles (IoV) networks has enabled intelligent transportation and smart traffic management. Nonetheless, the security against adversarial attacks is also increasingly becoming a challenging task. Specifically, Deep Reinforcement Learning (DRL) is one of the widely used ML designs in IoV applications. The standard ML security techniques are not effective in DRL where the algorithm learns to solve sequential decision-making through continuous interaction with the environment, and the environment is time-varying, dynamic, and mobile. In this paper, we propose a Gated Recurrent Unit (GRU)-based federated continual learning (GFCL) anomaly detection framework against adversarial attacks in IoV. The objective is to present a lightweight and scalable framework that learns and detects the illegitimate behavior without having a-priori training dataset consisting of attack samples. We use GRU to predict a future data sequence to analyze and detect illegitimate behavior from vehicles in a federated learning-based distributed manner. We investigate the performance of our framework using real-world vehicle mobility traces. The results demonstrate the effectiveness of our proposed solution for different performance metrics.



## **31. A Tale of Two Models: Constructing Evasive Attacks on Edge Models**

cs.CR

**SubmitDate**: 2022-04-22    [paper-pdf](http://arxiv.org/pdf/2204.10933v1)

**Authors**: Wei Hao, Aahil Awatramani, Jiayang Hu, Chengzhi Mao, Pin-Chun Chen, Eyal Cidon, Asaf Cidon, Junfeng Yang

**Abstracts**: Full-precision deep learning models are typically too large or costly to deploy on edge devices. To accommodate to the limited hardware resources, models are adapted to the edge using various edge-adaptation techniques, such as quantization and pruning. While such techniques may have a negligible impact on top-line accuracy, the adapted models exhibit subtle differences in output compared to the original model from which they are derived. In this paper, we introduce a new evasive attack, DIVA, that exploits these differences in edge adaptation, by adding adversarial noise to input data that maximizes the output difference between the original and adapted model. Such an attack is particularly dangerous, because the malicious input will trick the adapted model running on the edge, but will be virtually undetectable by the original model, which typically serves as the authoritative model version, used for validation, debugging and retraining. We compare DIVA to a state-of-the-art attack, PGD, and show that DIVA is only 1.7-3.6% worse on attacking the adapted model but 1.9-4.2 times more likely not to be detected by the the original model under a whitebox and semi-blackbox setting, compared to PGD.



## **32. How Sampling Impacts the Robustness of Stochastic Neural Networks**

cs.LG

**SubmitDate**: 2022-04-22    [paper-pdf](http://arxiv.org/pdf/2204.10839v1)

**Authors**: Sina Däubener, Asja Fischer

**Abstracts**: Stochastic neural networks (SNNs) are random functions and predictions are gained by averaging over multiple realizations of this random function. Consequently, an adversarial attack is calculated based on one set of samples and applied to the prediction defined by another set of samples. In this paper we analyze robustness in this setting by deriving a sufficient condition for the given prediction process to be robust against the calculated attack. This allows us to identify the factors that lead to an increased robustness of SNNs and helps to explain the impact of the variance and the amount of samples. Among other things, our theoretical analysis gives insights into (i) why increasing the amount of samples drawn for the estimation of adversarial examples increases the attack's strength, (ii) why decreasing sample size during inference hardly influences the robustness, and (iii) why a higher prediction variance between realizations relates to a higher robustness. We verify the validity of our theoretical findings by an extensive empirical analysis.



## **33. Defending Black-box Skeleton-based Human Activity Classifiers**

cs.CV

**SubmitDate**: 2022-04-22    [paper-pdf](http://arxiv.org/pdf/2203.04713v2)

**Authors**: He Wang, Yunfeng Diao, Zichang Tan, Guodong Guo

**Abstracts**: Deep learning has been regarded as the `go to' solution for many tasks today, but its intrinsic vulnerability to malicious attacks has become a major concern. The vulnerability is affected by a variety of factors including models, tasks, data, and attackers. Consequently, methods such as Adversarial Training and Randomized Smoothing have been proposed to tackle the problem in a wide range of applications. In this paper, we investigate skeleton-based Human Activity Recognition, which is an important type of time-series data but under-explored in defense against attacks. Our method is featured by (1) a new Bayesian Energy-based formulation of robust discriminative classifiers, (2) a new parameterization of the adversarial sample manifold of actions, and (3) a new post-train Bayesian treatment on both the adversarial samples and the classifier. We name our framework Bayesian Energy-based Adversarial Training or BEAT. BEAT is straightforward but elegant, which turns vulnerable black-box classifiers into robust ones without sacrificing accuracy. It demonstrates surprising and universal effectiveness across a wide range of action classifiers and datasets, under various attacks.



## **34. Enhancing the Transferability via Feature-Momentum Adversarial Attack**

cs.CV

**SubmitDate**: 2022-04-22    [paper-pdf](http://arxiv.org/pdf/2204.10606v1)

**Authors**: Xianglong, Yuezun Li, Haipeng Qu, Junyu Dong

**Abstracts**: Transferable adversarial attack has drawn increasing attention due to their practical threaten to real-world applications. In particular, the feature-level adversarial attack is one recent branch that can enhance the transferability via disturbing the intermediate features. The existing methods usually create a guidance map for features, where the value indicates the importance of the corresponding feature element and then employs an iterative algorithm to disrupt the features accordingly. However, the guidance map is fixed in existing methods, which can not consistently reflect the behavior of networks as the image is changed during iteration. In this paper, we describe a new method called Feature-Momentum Adversarial Attack (FMAA) to further improve transferability. The key idea of our method is that we estimate a guidance map dynamically at each iteration using momentum to effectively disturb the category-relevant features. Extensive experiments demonstrate that our method significantly outperforms other state-of-the-art methods by a large margin on different target models.



## **35. Data-Efficient Backdoor Attacks**

cs.CV

Accepted to IJCAI 2022 Long Oral

**SubmitDate**: 2022-04-22    [paper-pdf](http://arxiv.org/pdf/2204.12281v1)

**Authors**: Pengfei Xia, Ziqiang Li, Wei Zhang, Bin Li

**Abstracts**: Recent studies have proven that deep neural networks are vulnerable to backdoor attacks. Specifically, by mixing a small number of poisoned samples into the training set, the behavior of the trained model can be maliciously controlled. Existing attack methods construct such adversaries by randomly selecting some clean data from the benign set and then embedding a trigger into them. However, this selection strategy ignores the fact that each poisoned sample contributes inequally to the backdoor injection, which reduces the efficiency of poisoning. In this paper, we formulate improving the poisoned data efficiency by the selection as an optimization problem and propose a Filtering-and-Updating Strategy (FUS) to solve it. The experimental results on CIFAR-10 and ImageNet-10 indicate that the proposed method is effective: the same attack success rate can be achieved with only 47% to 75% of the poisoned sample volume compared to the random selection strategy. More importantly, the adversaries selected according to one setting can generalize well to other settings, exhibiting strong transferability.



## **36. Improving the Robustness of Adversarial Attacks Using an Affine-Invariant Gradient Estimator**

cs.CV

**SubmitDate**: 2022-04-22    [paper-pdf](http://arxiv.org/pdf/2109.05820v2)

**Authors**: Wenzhao Xiang, Hang Su, Chang Liu, Yandong Guo, Shibao Zheng

**Abstracts**: As designers of artificial intelligence try to outwit hackers, both sides continue to hone in on AI's inherent vulnerabilities. Designed and trained from certain statistical distributions of data, AI's deep neural networks (DNNs) remain vulnerable to deceptive inputs that violate a DNN's statistical, predictive assumptions. Before being fed into a neural network, however, most existing adversarial examples cannot maintain malicious functionality when applied to an affine transformation. For practical purposes, maintaining that malicious functionality serves as an important measure of the robustness of adversarial attacks. To help DNNs learn to defend themselves more thoroughly against attacks, we propose an affine-invariant adversarial attack, which can consistently produce more robust adversarial examples over affine transformations. For efficiency, we propose to disentangle current affine-transformation strategies from the Euclidean geometry coordinate plane with its geometric translations, rotations and dilations; we reformulate the latter two in polar coordinates. Afterwards, we construct an affine-invariant gradient estimator by convolving the gradient at the original image with derived kernels, which can be integrated with any gradient-based attack methods. Extensive experiments on ImageNet, including some experiments under physical condition, demonstrate that our method can significantly improve the affine invariance of adversarial examples and, as a byproduct, improve the transferability of adversarial examples, compared with alternative state-of-the-art methods.



## **37. Real-Time Detectors for Digital and Physical Adversarial Inputs to Perception Systems**

cs.CV

**SubmitDate**: 2022-04-21    [paper-pdf](http://arxiv.org/pdf/2002.09792v2)

**Authors**: Yiannis Kantaros, Taylor Carpenter, Kaustubh Sridhar, Yahan Yang, Insup Lee, James Weimer

**Abstracts**: Deep neural network (DNN) models have proven to be vulnerable to adversarial digital and physical attacks. In this paper, we propose a novel attack- and dataset-agnostic and real-time detector for both types of adversarial inputs to DNN-based perception systems. In particular, the proposed detector relies on the observation that adversarial images are sensitive to certain label-invariant transformations. Specifically, to determine if an image has been adversarially manipulated, the proposed detector checks if the output of the target classifier on a given input image changes significantly after feeding it a transformed version of the image under investigation. Moreover, we show that the proposed detector is computationally-light both at runtime and design-time which makes it suitable for real-time applications that may also involve large-scale image domains. To highlight this, we demonstrate the efficiency of the proposed detector on ImageNet, a task that is computationally challenging for the majority of relevant defenses, and on physically attacked traffic signs that may be encountered in real-time autonomy applications. Finally, we propose the first adversarial dataset, called AdvNet that includes both clean and physical traffic sign images. Our extensive comparative experiments on the MNIST, CIFAR10, ImageNet, and AdvNet datasets show that VisionGuard outperforms existing defenses in terms of scalability and detection performance. We have also evaluated the proposed detector on field test data obtained on a moving vehicle equipped with a perception-based DNN being under attack.



## **38. Adversarial Contrastive Learning by Permuting Cluster Assignments**

cs.LG

**SubmitDate**: 2022-04-21    [paper-pdf](http://arxiv.org/pdf/2204.10314v1)

**Authors**: Muntasir Wahed, Afrina Tabassum, Ismini Lourentzou

**Abstracts**: Contrastive learning has gained popularity as an effective self-supervised representation learning technique. Several research directions improve traditional contrastive approaches, e.g., prototypical contrastive methods better capture the semantic similarity among instances and reduce the computational burden by considering cluster prototypes or cluster assignments, while adversarial instance-wise contrastive methods improve robustness against a variety of attacks. To the best of our knowledge, no prior work jointly considers robustness, cluster-wise semantic similarity and computational efficiency. In this work, we propose SwARo, an adversarial contrastive framework that incorporates cluster assignment permutations to generate representative adversarial samples. We evaluate SwARo on multiple benchmark datasets and against various white-box and black-box attacks, obtaining consistent improvements over state-of-the-art baselines.



## **39. A Mask-Based Adversarial Defense Scheme**

cs.LG

7 pages

**SubmitDate**: 2022-04-21    [paper-pdf](http://arxiv.org/pdf/2204.11837v1)

**Authors**: Weizhen Xu, Chenyi Zhang, Fangzhen Zhao, Liangda Fang

**Abstracts**: Adversarial attacks hamper the functionality and accuracy of Deep Neural Networks (DNNs) by meddling with subtle perturbations to their inputs.In this work, we propose a new Mask-based Adversarial Defense scheme (MAD) for DNNs to mitigate the negative effect from adversarial attacks. To be precise, our method promotes the robustness of a DNN by randomly masking a portion of potential adversarial images, and as a result, the %classification result output of the DNN becomes more tolerant to minor input perturbations. Compared with existing adversarial defense techniques, our method does not need any additional denoising structure, nor any change to a DNN's design. We have tested this approach on a collection of DNN models for a variety of data sets, and the experimental results confirm that the proposed method can effectively improve the defense abilities of the DNNs against all of the tested adversarial attack methods. In certain scenarios, the DNN models trained with MAD have improved classification accuracy by as much as 20% to 90% compared to the original models that are given adversarial inputs.



## **40. Robustness of Machine Learning Models Beyond Adversarial Attacks**

cs.LG

25 pages, 7 figures

**SubmitDate**: 2022-04-21    [paper-pdf](http://arxiv.org/pdf/2204.10046v1)

**Authors**: Sebastian Scher, Andreas Trügler

**Abstracts**: Correctly quantifying the robustness of machine learning models is a central aspect in judging their suitability for specific tasks, and thus, ultimately, for generating trust in the models. We show that the widely used concept of adversarial robustness and closely related metrics based on counterfactuals are not necessarily valid metrics for determining the robustness of ML models against perturbations that occur "naturally", outside specific adversarial attack scenarios. Additionally, we argue that generic robustness metrics in principle are insufficient for determining real-world-robustness. Instead we propose a flexible approach that models possible perturbations in input data individually for each application. This is then combined with a probabilistic approach that computes the likelihood that a real-world perturbation will change a prediction, thus giving quantitative information of the robustness of the trained machine learning model. The method does not require access to the internals of the classifier and thus in principle works for any black-box model. It is, however, based on Monte-Carlo sampling and thus only suited for input spaces with small dimensions. We illustrate our approach on two dataset, as well as on analytically solvable cases. Finally, we discuss ideas on how real-world robustness could be computed or estimated in high-dimensional input spaces.



## **41. Is Neuron Coverage Needed to Make Person Detection More Robust?**

cs.CV

Accepted for publication at CVPR 2022 TCV workshop

**SubmitDate**: 2022-04-21    [paper-pdf](http://arxiv.org/pdf/2204.10027v1)

**Authors**: Svetlana Pavlitskaya, Şiyar Yıkmış, J. Marius Zöllner

**Abstracts**: The growing use of deep neural networks (DNNs) in safety- and security-critical areas like autonomous driving raises the need for their systematic testing. Coverage-guided testing (CGT) is an approach that applies mutation or fuzzing according to a predefined coverage metric to find inputs that cause misbehavior. With the introduction of a neuron coverage metric, CGT has also recently been applied to DNNs. In this work, we apply CGT to the task of person detection in crowded scenes. The proposed pipeline uses YOLOv3 for person detection and includes finding DNN bugs via sampling and mutation, and subsequent DNN retraining on the updated training set. To be a bug, we require a mutated image to cause a significant performance drop compared to a clean input. In accordance with the CGT, we also consider an additional requirement of increased coverage in the bug definition. In order to explore several types of robustness, our approach includes natural image transformations, corruptions, and adversarial examples generated with the Daedalus attack. The proposed framework has uncovered several thousand cases of incorrect DNN behavior. The relative change in mAP performance of the retrained models reached on average between 26.21\% and 64.24\% for different robustness types. However, we have found no evidence that the investigated coverage metrics can be advantageously used to improve robustness.



## **42. On the Certified Robustness for Ensemble Models and Beyond**

cs.LG

ICLR 2022. 51 pages, 10 pages for main text. Forum and code:  https://openreview.net/forum?id=tUa4REjGjTf

**SubmitDate**: 2022-04-21    [paper-pdf](http://arxiv.org/pdf/2107.10873v2)

**Authors**: Zhuolin Yang, Linyi Li, Xiaojun Xu, Bhavya Kailkhura, Tao Xie, Bo Li

**Abstracts**: Recent studies show that deep neural networks (DNN) are vulnerable to adversarial examples, which aim to mislead DNNs by adding perturbations with small magnitude. To defend against such attacks, both empirical and theoretical defense approaches have been extensively studied for a single ML model. In this work, we aim to analyze and provide the certified robustness for ensemble ML models, together with the sufficient and necessary conditions of robustness for different ensemble protocols. Although ensemble models are shown more robust than a single model empirically; surprisingly, we find that in terms of the certified robustness the standard ensemble models only achieve marginal improvement compared to a single model. Thus, to explore the conditions that guarantee to provide certifiably robust ensemble ML models, we first prove that diversified gradient and large confidence margin are sufficient and necessary conditions for certifiably robust ensemble models under the model-smoothness assumption. We then provide the bounded model-smoothness analysis based on the proposed Ensemble-before-Smoothing strategy. We also prove that an ensemble model can always achieve higher certified robustness than a single base model under mild conditions. Inspired by the theoretical findings, we propose the lightweight Diversity Regularized Training (DRT) to train certifiably robust ensemble ML models. Extensive experiments show that our DRT enhanced ensembles can consistently achieve higher certified robustness than existing single and ensemble ML models, demonstrating the state-of-the-art certified L2-robustness on MNIST, CIFAR-10, and ImageNet datasets.



## **43. Fast AdvProp**

cs.CV

ICLR 2022 camera ready version

**SubmitDate**: 2022-04-21    [paper-pdf](http://arxiv.org/pdf/2204.09838v1)

**Authors**: Jieru Mei, Yucheng Han, Yutong Bai, Yixiao Zhang, Yingwei Li, Xianhang Li, Alan Yuille, Cihang Xie

**Abstracts**: Adversarial Propagation (AdvProp) is an effective way to improve recognition models, leveraging adversarial examples. Nonetheless, AdvProp suffers from the extremely slow training speed, mainly because: a) extra forward and backward passes are required for generating adversarial examples; b) both original samples and their adversarial counterparts are used for training (i.e., 2$\times$ data). In this paper, we introduce Fast AdvProp, which aggressively revamps AdvProp's costly training components, rendering the method nearly as cheap as the vanilla training. Specifically, our modifications in Fast AdvProp are guided by the hypothesis that disentangled learning with adversarial examples is the key for performance improvements, while other training recipes (e.g., paired clean and adversarial training samples, multi-step adversarial attackers) could be largely simplified.   Our empirical results show that, compared to the vanilla training baseline, Fast AdvProp is able to further model performance on a spectrum of visual benchmarks, without incurring extra training cost. Additionally, our ablations find Fast AdvProp scales better if larger models are used, is compatible with existing data augmentation methods (i.e., Mixup and CutMix), and can be easily adapted to other recognition tasks like object detection. The code is available here: https://github.com/meijieru/fast_advprop.



## **44. GUARD: Graph Universal Adversarial Defense**

cs.LG

Code is publicly available at https://github.com/EdisonLeeeee/GUARD

**SubmitDate**: 2022-04-20    [paper-pdf](http://arxiv.org/pdf/2204.09803v1)

**Authors**: Jintang Li, Jie Liao, Ruofan Wu, Liang Chen, Changhua Meng, Zibin Zheng, Weiqiang Wang

**Abstracts**: Recently, graph convolutional networks (GCNs) have shown to be vulnerable to small adversarial perturbations, which becomes a severe threat and largely limits their applications in security-critical scenarios. To mitigate such a threat, considerable research efforts have been devoted to increasing the robustness of GCNs against adversarial attacks. However, current approaches for defense are typically designed for the whole graph and consider the global performance, posing challenges in protecting important local nodes from stronger adversarial targeted attacks. In this work, we present a simple yet effective method, named \textbf{\underline{G}}raph \textbf{\underline{U}}niversal \textbf{\underline{A}}dve\textbf{\underline{R}}sarial \textbf{\underline{D}}efense (GUARD). Unlike previous works, GUARD protects each individual node from attacks with a universal defensive patch, which is generated once and can be applied to any node (node-agnostic) in a graph. Extensive experiments on four benchmark datasets demonstrate that our method significantly improves robustness for several established GCNs against multiple adversarial attacks and outperforms existing adversarial defense methods by large margins. Our code is publicly available at https://github.com/EdisonLeeeee/GUARD.



## **45. Backdooring Explainable Machine Learning**

cs.CR

**SubmitDate**: 2022-04-20    [paper-pdf](http://arxiv.org/pdf/2204.09498v1)

**Authors**: Maximilian Noppel, Lukas Peter, Christian Wressnegger

**Abstracts**: Explainable machine learning holds great potential for analyzing and understanding learning-based systems. These methods can, however, be manipulated to present unfaithful explanations, giving rise to powerful and stealthy adversaries. In this paper, we demonstrate blinding attacks that can fully disguise an ongoing attack against the machine learning model. Similar to neural backdoors, we modify the model's prediction upon trigger presence but simultaneously also fool the provided explanation. This enables an adversary to hide the presence of the trigger or point the explanation to entirely different portions of the input, throwing a red herring. We analyze different manifestations of such attacks for different explanation types in the image domain, before we resume to conduct a red-herring attack against malware classification.



## **46. Adversarial Scratches: Deployable Attacks to CNN Classifiers**

cs.LG

This paper stems from 'Scratch that! An Evolution-based Adversarial  Attack against Neural Networks' for which an arXiv preprint is available at  arXiv:1912.02316. Further studies led to a complete overhaul of the work,  resulting in this paper. This work was submitted for review in Pattern  Recognition (Elsevier)

**SubmitDate**: 2022-04-20    [paper-pdf](http://arxiv.org/pdf/2204.09397v1)

**Authors**: Loris Giulivi, Malhar Jere, Loris Rossi, Farinaz Koushanfar, Gabriela Ciocarlie, Briland Hitaj, Giacomo Boracchi

**Abstracts**: A growing body of work has shown that deep neural networks are susceptible to adversarial examples. These take the form of small perturbations applied to the model's input which lead to incorrect predictions. Unfortunately, most literature focuses on visually imperceivable perturbations to be applied to digital images that often are, by design, impossible to be deployed to physical targets. We present Adversarial Scratches: a novel L0 black-box attack, which takes the form of scratches in images, and which possesses much greater deployability than other state-of-the-art attacks. Adversarial Scratches leverage B\'ezier Curves to reduce the dimension of the search space and possibly constrain the attack to a specific location. We test Adversarial Scratches in several scenarios, including a publicly available API and images of traffic signs. Results show that, often, our attack achieves higher fooling rate than other deployable state-of-the-art methods, while requiring significantly fewer queries and modifying very few pixels.



## **47. You Are What You Write: Preserving Privacy in the Era of Large Language Models**

cs.CL

**SubmitDate**: 2022-04-20    [paper-pdf](http://arxiv.org/pdf/2204.09391v1)

**Authors**: Richard Plant, Valerio Giuffrida, Dimitra Gkatzia

**Abstracts**: Large scale adoption of large language models has introduced a new era of convenient knowledge transfer for a slew of natural language processing tasks. However, these models also run the risk of undermining user trust by exposing unwanted information about the data subjects, which may be extracted by a malicious party, e.g. through adversarial attacks. We present an empirical investigation into the extent of the personal information encoded into pre-trained representations by a range of popular models, and we show a positive correlation between the complexity of a model, the amount of data used in pre-training, and data leakage. In this paper, we present the first wide coverage evaluation and comparison of some of the most popular privacy-preserving algorithms, on a large, multi-lingual dataset on sentiment analysis annotated with demographic information (location, age and gender). The results show since larger and more complex models are more prone to leaking private information, use of privacy-preserving methods is highly desirable. We also find that highly privacy-preserving technologies like differential privacy (DP) can have serious model utility effects, which can be ameliorated using hybrid or metric-DP techniques.



## **48. Identifying Near-Optimal Single-Shot Attacks on ICSs with Limited Process Knowledge**

cs.CR

This paper has been accepted at Applied Cryptography and Network  Security (ACNS) 2022

**SubmitDate**: 2022-04-19    [paper-pdf](http://arxiv.org/pdf/2204.09106v1)

**Authors**: Herson Esquivel-Vargas, John Henry Castellanos, Marco Caselli, Nils Ole Tippenhauer, Andreas Peter

**Abstracts**: Industrial Control Systems (ICSs) rely on insecure protocols and devices to monitor and operate critical infrastructure. Prior work has demonstrated that powerful attackers with detailed system knowledge can manipulate exchanged sensor data to deteriorate performance of the process, even leading to full shutdowns of plants. Identifying those attacks requires iterating over all possible sensor values, and running detailed system simulation or analysis to identify optimal attacks. That setup allows adversaries to identify attacks that are most impactful when applied on the system for the first time, before the system operators become aware of the manipulations.   In this work, we investigate if constrained attackers without detailed system knowledge and simulators can identify comparable attacks. In particular, the attacker only requires abstract knowledge on general information flow in the plant, instead of precise algorithms, operating parameters, process models, or simulators. We propose an approach that allows single-shot attacks, i.e., near-optimal attacks that are reliably shutting down a system on the first try. The approach is applied and validated on two use cases, and demonstrated to achieve comparable results to prior work, which relied on detailed system information and simulations.



## **49. Indiscriminate Data Poisoning Attacks on Neural Networks**

cs.LG

**SubmitDate**: 2022-04-19    [paper-pdf](http://arxiv.org/pdf/2204.09092v1)

**Authors**: Yiwei Lu, Gautam Kamath, Yaoliang Yu

**Abstracts**: Data poisoning attacks, in which a malicious adversary aims to influence a model by injecting "poisoned" data into the training process, have attracted significant recent attention. In this work, we take a closer look at existing poisoning attacks and connect them with old and new algorithms for solving sequential Stackelberg games. By choosing an appropriate loss function for the attacker and optimizing with algorithms that exploit second-order information, we design poisoning attacks that are effective on neural networks. We present efficient implementations that exploit modern auto-differentiation packages and allow simultaneous and coordinated generation of tens of thousands of poisoned points, in contrast to existing methods that generate poisoned points one by one. We further perform extensive experiments that empirically explore the effect of data poisoning attacks on deep neural networks.



## **50. A Brief Survey on Deep Learning Based Data Hiding**

cs.CR

v2: reorganize some sections and add several new papers published in  2021~2022

**SubmitDate**: 2022-04-19    [paper-pdf](http://arxiv.org/pdf/2103.01607v2)

**Authors**: Chaoning Zhang, Chenguo Lin, Philipp Benz, Kejiang Chen, Weiming Zhang, In So Kweon

**Abstracts**: Data hiding is the art of concealing messages with limited perceptual changes. Recently, deep learning has enriched it from various perspectives with significant progress. In this work, we conduct a brief yet comprehensive review of existing literature for deep learning based data hiding (deep hiding) by first classifying it according to three essential properties (i.e., capacity, security and robustness), and outline three commonly used architectures. Based on this, we summarize specific strategies for different applications of data hiding, including basic hiding, steganography, watermarking and light field messaging. Finally, further insight into deep hiding is provided by incorporating the perspective of adversarial attack.



