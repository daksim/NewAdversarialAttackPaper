# Latest Adversarial Attack Papers
**update at 2022-09-10 06:31:30**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. A Survey of Machine Unlearning**

cs.LG

arXiv admin note: text overlap with arXiv:2109.13398,  arXiv:2109.08266 by other authors. author note: fixed some overlaps

**SubmitDate**: 2022-09-08    [paper-pdf](http://arxiv.org/pdf/2209.02299v3)

**Authors**: Thanh Tam Nguyen, Thanh Trung Huynh, Phi Le Nguyen, Alan Wee-Chung Liew, Hongzhi Yin, Quoc Viet Hung Nguyen

**Abstracts**: Computer systems hold a large amount of personal data over decades. On the one hand, such data abundance allows breakthroughs in artificial intelligence (AI), especially machine learning (ML) models. On the other hand, it can threaten the privacy of users and weaken the trust between humans and AI. Recent regulations require that private information about a user can be removed from computer systems in general and from ML models in particular upon request (e.g. the "right to be forgotten"). While removing data from back-end databases should be straightforward, it is not sufficient in the AI context as ML models often "remember" the old data. Existing adversarial attacks proved that we can learn private membership or attributes of the training data from the trained models. This phenomenon calls for a new paradigm, namely machine unlearning, to make ML models forget about particular data. It turns out that recent works on machine unlearning have not been able to solve the problem completely due to the lack of common frameworks and resources. In this survey paper, we seek to provide a thorough investigation of machine unlearning in its definitions, scenarios, mechanisms, and applications. Specifically, as a categorical collection of state-of-the-art research, we hope to provide a broad reference for those seeking a primer on machine unlearning and its various formulations, design requirements, removal requests, algorithms, and uses in a variety of ML applications. Furthermore, we hope to outline key findings and trends in the paradigm as well as highlight new areas of research that have yet to see the application of machine unlearning, but could nonetheless benefit immensely. We hope this survey provides a valuable reference for ML researchers as well as those seeking to innovate privacy technologies. Our resources are at https://github.com/tamlhp/awesome-machine-unlearning.



## **2. SafeNet: The Unreasonable Effectiveness of Ensembles in Private Collaborative Learning**

cs.CR

**SubmitDate**: 2022-09-08    [paper-pdf](http://arxiv.org/pdf/2205.09986v2)

**Authors**: Harsh Chaudhari, Matthew Jagielski, Alina Oprea

**Abstracts**: Secure multiparty computation (MPC) has been proposed to allow multiple mutually distrustful data owners to jointly train machine learning (ML) models on their combined data. However, by design, MPC protocols faithfully compute the training functionality, which the adversarial ML community has shown to leak private information and can be tampered with in poisoning attacks. In this work, we argue that model ensembles, implemented in our framework called SafeNet, are a highly MPC-amenable way to avoid many adversarial ML attacks. The natural partitioning of data amongst owners in MPC training allows this approach to be highly scalable at training time, provide provable protection from poisoning attacks, and provably defense against a number of privacy attacks. We demonstrate SafeNet's efficiency, accuracy, and resilience to poisoning on several machine learning datasets and models trained in end-to-end and transfer learning scenarios. For instance, SafeNet reduces backdoor attack success significantly, while achieving $39\times$ faster training and $36 \times$ less communication than the four-party MPC framework of Dalskov et al. Our experiments show that ensembling retains these benefits even in many non-iid settings. The simplicity, cheap setup, and robustness properties of ensembling make it a strong first choice for training ML models privately in MPC.



## **3. Incorporating Locality of Images to Generate Targeted Transferable Adversarial Examples**

cs.CV

**SubmitDate**: 2022-09-08    [paper-pdf](http://arxiv.org/pdf/2209.03716v1)

**Authors**: Zhipeng Wei, Jingjing Chen, Zuxuan Wu, Yu-Gang Jiang

**Abstracts**: Despite that leveraging the transferability of adversarial examples can attain a fairly high attack success rate for non-targeted attacks, it does not work well in targeted attacks since the gradient directions from a source image to a targeted class are usually different in different DNNs. To increase the transferability of target attacks, recent studies make efforts in aligning the feature of the generated adversarial example with the feature distributions of the targeted class learned from an auxiliary network or a generative adversarial network. However, these works assume that the training dataset is available and require a lot of time to train networks, which makes it hard to apply to real-world scenarios. In this paper, we revisit adversarial examples with targeted transferability from the perspective of universality and find that highly universal adversarial perturbations tend to be more transferable. Based on this observation, we propose the Locality of Images (LI) attack to improve targeted transferability. Specifically, instead of using the classification loss only, LI introduces a feature similarity loss between intermediate features from adversarial perturbed original images and randomly cropped images, which makes the features from adversarial perturbations to be more dominant than that of benign images, hence improving targeted transferability. Through incorporating locality of images into optimizing perturbations, the LI attack emphasizes that targeted perturbations should be universal to diverse input patterns, even local image patches. Extensive experiments demonstrate that LI can achieve high success rates for transfer-based targeted attacks. On attacking the ImageNet-compatible dataset, LI yields an improvement of 12\% compared with existing state-of-the-art methods.



## **4. Exploring Adversarial Attacks and Defenses in Vision Transformers trained with DINO**

cs.CV

ICML 2022 Workshop paper accepted at AdvML Frontiers

**SubmitDate**: 2022-09-08    [paper-pdf](http://arxiv.org/pdf/2206.06761v4)

**Authors**: Javier Rando, Nasib Naimi, Thomas Baumann, Max Mathys

**Abstracts**: This work conducts the first analysis on the robustness against adversarial attacks on self-supervised Vision Transformers trained using DINO. First, we evaluate whether features learned through self-supervision are more robust to adversarial attacks than those emerging from supervised learning. Then, we present properties arising for attacks in the latent space. Finally, we evaluate whether three well-known defense strategies can increase adversarial robustness in downstream tasks by only fine-tuning the classification head to provide robustness even in view of limited compute resources. These defense strategies are: Adversarial Training, Ensemble Adversarial Training and Ensemble of Specialized Networks.



## **5. Feature Importance Guided Attack: A Model Agnostic Adversarial Attack**

cs.LG

**SubmitDate**: 2022-09-08    [paper-pdf](http://arxiv.org/pdf/2106.14815v2)

**Authors**: Gilad Gressel, Niranjan Hegde, Archana Sreekumar, Rishikumar Radhakrishnan, Kalyani Harikumar, Anjali S., Michael Darling

**Abstracts**: Research in adversarial learning has primarily focused on homogeneous unstructured datasets, which often map into the problem space naturally. Inverting a feature space attack on heterogeneous datasets into the problem space is much more challenging, particularly the task of finding the perturbation to perform. This work presents a formal search strategy: the `Feature Importance Guided Attack' (FIGA), which finds perturbations in the feature space of heterogeneous tabular datasets to produce evasion attacks. We first demonstrate FIGA in the feature space and then in the problem space. FIGA assumes no prior knowledge of the defending model's learning algorithm and does not require any gradient information. FIGA assumes knowledge of the feature representation and the mean feature values of defending model's dataset. FIGA leverages feature importance rankings by perturbing the most important features of the input in the direction of the target class. While FIGA is conceptually similar to other work which uses feature selection processes (e.g., mimicry attacks), we formalize an attack algorithm with three tunable parameters and investigate the strength of FIGA on tabular datasets. We demonstrate the effectiveness of FIGA by evading phishing detection models trained on four different tabular phishing datasets and one financial dataset with an average success rate of 94%. We extend FIGA to the phishing problem space by limiting the possible perturbations to be valid and feasible in the phishing domain. We generate valid adversarial phishing sites that are visually identical to their unperturbed counterpart and use them to attack six tabular ML models achieving a 13.05% average success rate.



## **6. AdaptOver: Adaptive Overshadowing Attacks in Cellular Networks**

cs.CR

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2106.05039v3)

**Authors**: Simon Erni, Martin Kotuliak, Patrick Leu, Marc Roeschlin, Srdjan Capkun

**Abstracts**: In cellular networks, attacks on the communication link between a mobile device and the core network significantly impact privacy and availability. Up until now, fake base stations have been required to execute such attacks. Since they require a continuously high output power to attract victims, they are limited in range and can be easily detected both by operators and dedicated apps on users' smartphones.   This paper introduces AdaptOver - a MITM attack system designed for cellular networks, specifically for LTE and 5G-NSA. AdaptOver allows an adversary to decode, overshadow (replace) and inject arbitrary messages over the air in either direction between the network and the mobile device. Using overshadowing, AdaptOver can cause a persistent ($\geq$ 12h) DoS or a privacy leak by triggering a UE to transmit its persistent identifier (IMSI) in plain text. These attacks can be launched against all users within a cell or specifically target a victim based on its phone number.   We implement AdaptOver using a software-defined radio and a low-cost amplification setup. We demonstrate the effects and practicality of the attacks on a live operational LTE and 5G-NSA network with a wide range of smartphones. Our experiments show that AdaptOver can launch an attack on a victim more than 3.8km away from the attacker. Given its practicability and efficiency, AdaptOver shows that existing countermeasures that are focused on fake base stations are no longer sufficient, marking a paradigm shift for designing security mechanisms in cellular networks.



## **7. Combing for Credentials: Active Pattern Extraction from Smart Reply**

cs.CR

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2207.10802v2)

**Authors**: Bargav Jayaraman, Esha Ghosh, Melissa Chase, Sambuddha Roy, Huseyin Inan, Wei Dai, David Evans

**Abstracts**: With the wide availability of large pre-trained language models such as GPT-2 and BERT, the recent trend has been to fine-tune a pre-trained model to achieve state-of-the-art performance on a downstream task. One natural example is the "Smart Reply" application where a pre-trained model is tuned to provide suggested responses for a given query message. Since these models are often tuned using sensitive data such as emails or chat transcripts, it is important to understand and mitigate the risk that the model leaks its tuning data. We investigate potential information leakage vulnerabilities in a typical Smart Reply pipeline and introduce a new type of active extraction attack that exploits canonical patterns in text containing sensitive data. We show experimentally that it is possible for an adversary to extract sensitive user information present in the training data. We explore potential mitigation strategies and demonstrate empirically how differential privacy appears to be an effective defense mechanism to such pattern extraction attacks.



## **8. Inferring Sensitive Attributes from Model Explanations**

cs.CR

ACM CIKM 2022

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2208.09967v2)

**Authors**: Vasisht Duddu, Antoine Boutet

**Abstracts**: Model explanations provide transparency into a trained machine learning model's blackbox behavior to a model builder. They indicate the influence of different input attributes to its corresponding model prediction. The dependency of explanations on input raises privacy concerns for sensitive user data. However, current literature has limited discussion on privacy risks of model explanations.   We focus on the specific privacy risk of attribute inference attack wherein an adversary infers sensitive attributes of an input (e.g., race and sex) given its model explanations. We design the first attribute inference attack against model explanations in two threat models where model builder either (a) includes the sensitive attributes in training data and input or (b) censors the sensitive attributes by not including them in the training data and input.   We evaluate our proposed attack on four benchmark datasets and four state-of-the-art algorithms. We show that an adversary can successfully infer the value of sensitive attributes from explanations in both the threat models accurately. Moreover, the attack is successful even by exploiting only the explanations corresponding to sensitive attributes. These suggest that our attack is effective against explanations and poses a practical threat to data privacy.   On combining the model predictions (an attack surface exploited by prior attacks) with explanations, we note that the attack success does not improve. Additionally, the attack success on exploiting model explanations is better compared to exploiting only model predictions. These suggest that model explanations are a strong attack surface to exploit for an adversary.



## **9. Securing the Spike: On the Transferabilty and Security of Spiking Neural Networks to Adversarial Examples**

cs.NE

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2209.03358v1)

**Authors**: Nuo Xu, Kaleel Mahmood, Haowen Fang, Ethan Rathbun, Caiwen Ding, Wujie Wen

**Abstracts**: Spiking neural networks (SNNs) have attracted much attention for their high energy efficiency and for recent advances in their classification performance. However, unlike traditional deep learning approaches, the analysis and study of the robustness of SNNs to adversarial examples remains relatively underdeveloped. In this work we advance the field of adversarial machine learning through experimentation and analyses of three important SNN security attributes. First, we show that successful white-box adversarial attacks on SNNs are highly dependent on the underlying surrogate gradient technique. Second, we analyze the transferability of adversarial examples generated by SNNs and other state-of-the-art architectures like Vision Transformers and Big Transfer CNNs. We demonstrate that SNNs are not often deceived by adversarial examples generated by Vision Transformers and certain types of CNNs. Lastly, we develop a novel white-box attack that generates adversarial examples capable of fooling both SNN models and non-SNN models simultaneously. Our experiments and analyses are broad and rigorous covering two datasets (CIFAR-10 and CIFAR-100), five different white-box attacks and twelve different classifier models.



## **10. Minotaur: Multi-Resource Blockchain Consensus**

cs.CR

To appear in ACM CCS 2022

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2201.11780v2)

**Authors**: Matthias Fitzi, Xuechao Wang, Sreeram Kannan, Aggelos Kiayias, Nikos Leonardos, Pramod Viswanath, Gerui Wang

**Abstracts**: Resource-based consensus is the backbone of permissionless distributed ledger systems. The security of such protocols relies fundamentally on the level of resources actively engaged in the system. The variety of different resources (and related proof protocols, some times referred to as PoX in the literature) raises the fundamental question whether it is possible to utilize many of them in tandem and build multi-resource consensus protocols. The challenge in combining different resources is to achieve fungibility between them, in the sense that security would hold as long as the cumulative adversarial power across all resources is bounded.   In this work, we put forth Minotaur, a multi-resource blockchain consensus protocol that combines proof-of-work (PoW) and proof-of-stake (PoS), and we prove it optimally fungible. At the core of our design, Minotaur operates in epochs while continuously sampling the active computational power to provide a fair exchange between the two resources, work and stake. Further, we demonstrate the ability of Minotaur to handle a higher degree of work fluctuation as compared to the Bitcoin blockchain; we also generalize Minotaur to any number of resources.   We demonstrate the simplicity of Minotaur via implementing a full stack client in Rust (available open source). We use the client to test the robustness of Minotaur to variable mining power and combined work/stake attacks and demonstrate concrete empirical evidence towards the suitability of Minotaur to serve as the consensus layer of a real-world blockchain.



## **11. Distributed Adversarial Training to Robustify Deep Neural Networks at Scale**

cs.LG

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2206.06257v2)

**Authors**: Gaoyuan Zhang, Songtao Lu, Yihua Zhang, Xiangyi Chen, Pin-Yu Chen, Quanfu Fan, Lee Martie, Lior Horesh, Mingyi Hong, Sijia Liu

**Abstracts**: Current deep neural networks (DNNs) are vulnerable to adversarial attacks, where adversarial perturbations to the inputs can change or manipulate classification. To defend against such attacks, an effective and popular approach, known as adversarial training (AT), has been shown to mitigate the negative impact of adversarial attacks by virtue of a min-max robust training method. While effective, it remains unclear whether it can successfully be adapted to the distributed learning context. The power of distributed optimization over multiple machines enables us to scale up robust training over large models and datasets. Spurred by that, we propose distributed adversarial training (DAT), a large-batch adversarial training framework implemented over multiple machines. We show that DAT is general, which supports training over labeled and unlabeled data, multiple types of attack generation methods, and gradient compression operations favored for distributed optimization. Theoretically, we provide, under standard conditions in the optimization theory, the convergence rate of DAT to the first-order stationary points in general non-convex settings. Empirically, we demonstrate that DAT either matches or outperforms state-of-the-art robust accuracies and achieves a graceful training speedup (e.g., on ResNet-50 under ImageNet). Codes are available at https://github.com/dat-2022/dat.



## **12. Privacy Against Inference Attacks in Vertical Federated Learning**

cs.LG

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2207.11788v3)

**Authors**: Borzoo Rassouli, Morteza Varasteh, Deniz Gunduz

**Abstracts**: Vertical federated learning is considered, where an active party, having access to true class labels, wishes to build a classification model by utilizing more features from a passive party, which has no access to the labels, to improve the model accuracy. In the prediction phase, with logistic regression as the classification model, several inference attack techniques are proposed that the adversary, i.e., the active party, can employ to reconstruct the passive party's features, regarded as sensitive information. These attacks, which are mainly based on a classical notion of the center of a set, i.e., the Chebyshev center, are shown to be superior to those proposed in the literature. Moreover, several theoretical performance guarantees are provided for the aforementioned attacks. Subsequently, we consider the minimum amount of information that the adversary needs to fully reconstruct the passive party's features. In particular, it is shown that when the passive party holds one feature, and the adversary is only aware of the signs of the parameters involved, it can perfectly reconstruct that feature when the number of predictions is large enough. Next, as a defense mechanism, a privacy-preserving scheme is proposed that worsen the adversary's reconstruction attacks, while preserving the full benefits that VFL brings to the active party. Finally, experimental results demonstrate the effectiveness of the proposed attacks and the privacy-preserving scheme.



## **13. Fact-Saboteurs: A Taxonomy of Evidence Manipulation Attacks against Fact-Verification Systems**

cs.CR

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2209.03755v1)

**Authors**: Sahar Abdelnabi, Mario Fritz

**Abstracts**: Mis- and disinformation are now a substantial global threat to our security and safety. To cope with the scale of online misinformation, one viable solution is to automate the fact-checking of claims by retrieving and verifying against relevant evidence. While major recent advances have been achieved in pushing forward the automatic fact-verification, a comprehensive evaluation of the possible attack vectors against such systems is still lacking. Particularly, the automated fact-verification process might be vulnerable to the exact disinformation campaigns it is trying to combat. In this work, we assume an adversary that automatically tampers with the online evidence in order to disrupt the fact-checking model via camouflaging the relevant evidence, or planting a misleading one. We first propose an exploratory taxonomy that spans these two targets and the different threat model dimensions. Guided by this, we design and propose several potential attack methods. We show that it is possible to subtly modify claim-salient snippets in the evidence, in addition to generating diverse and claim-aligned evidence. As a result, we highly degrade the fact-checking performance under many different permutations of the taxonomy's dimensions. The attacks are also robust against post-hoc modifications of the claim. Our analysis further hints at potential limitations in models' inference when faced with contradicting evidence. We emphasize that these attacks can have harmful implications on the inspectable and human-in-the-loop usage scenarios of such models, and we conclude by discussing challenges and directions for future defenses.



## **14. State of Security Awareness in the AM Industry: 2020 Survey**

cs.CR

The material was presented at ASTM ICAM 2021 and a publication was  accepted for publication as a Selected Technical Papers (STP)

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2209.03073v1)

**Authors**: Mark Yampolskiy, Paul Bates, Mohsen Seifi, Nima Shamsaei

**Abstracts**: Security of Additive Manufacturing (AM) gets increased attention due to the growing proliferation and adoption of AM in a variety of applications and business models. However, there is a significant disconnect between AM community focused on manufacturing and AM Security community focused on securing this highly computerized manufacturing technology. To bridge this gap, we surveyed the America Makes AM community, asking in total eleven AM security-related questions aiming to discover the existing concerns, posture, and expectations. The first set of questions aimed to discover how many of these organizations use AM, outsource AM, or provide AM as a service. Then we asked about biggest security concerns as well as about assessment of who the potential adversaries might be and their motivation for attack. We then proceeded with questions on any experienced security incidents, if any security risk assessment was conducted, and if the participants' organizations were partnering with external experts to secure AM. Lastly, we asked whether security measures are implemented at all and, if yes, whether they fall under the general cyber-security category. Out of 69 participants affiliated with commercial industry, agencies, and academia, 53 have completed the entire survey. This paper presents the results of this survey, as well as provides our assessment of the AM Security posture. The answers are a mixture of what we could label as expected, "shocking but not surprising," and completely unexpected. Assuming that the provided answers are somewhat representative to the current state of the AM industry, we conclude that the industry is not ready to prevent or detect AM-specific attacks that have been demonstrated in the research literature.



## **15. On the Transferability of Adversarial Examples between Encrypted Models**

cs.CV

to be appear in ISPACS 2022

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2209.02997v1)

**Authors**: Miki Tanaka, Isao Echizen, Hitoshi Kiya

**Abstracts**: Deep neural networks (DNNs) are well known to be vulnerable to adversarial examples (AEs). In addition, AEs have adversarial transferability, namely, AEs generated for a source model fool other (target) models. In this paper, we investigate the transferability of models encrypted for adversarially robust defense for the first time. To objectively verify the property of transferability, the robustness of models is evaluated by using a benchmark attack method, called AutoAttack. In an image-classification experiment, the use of encrypted models is confirmed not only to be robust against AEs but to also reduce the influence of AEs in terms of the transferability of models.



## **16. Adversarial Mask: Real-World Universal Adversarial Attack on Face Recognition Model**

cs.CV

16 pages, 9 figures

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2111.10759v3)

**Authors**: Alon Zolfi, Shai Avidan, Yuval Elovici, Asaf Shabtai

**Abstracts**: Deep learning-based facial recognition (FR) models have demonstrated state-of-the-art performance in the past few years, even when wearing protective medical face masks became commonplace during the COVID-19 pandemic. Given the outstanding performance of these models, the machine learning research community has shown increasing interest in challenging their robustness. Initially, researchers presented adversarial attacks in the digital domain, and later the attacks were transferred to the physical domain. However, in many cases, attacks in the physical domain are conspicuous, and thus may raise suspicion in real-world environments (e.g., airports). In this paper, we propose Adversarial Mask, a physical universal adversarial perturbation (UAP) against state-of-the-art FR models that is applied on face masks in the form of a carefully crafted pattern. In our experiments, we examined the transferability of our adversarial mask to a wide range of FR model architectures and datasets. In addition, we validated our adversarial mask's effectiveness in real-world experiments (CCTV use case) by printing the adversarial pattern on a fabric face mask. In these experiments, the FR system was only able to identify 3.34% of the participants wearing the mask (compared to a minimum of 83.34% with other evaluated masks). A demo of our experiments can be found at: https://youtu.be/_TXkDO5z11w.



## **17. Facial De-morphing: Extracting Component Faces from a Single Morph**

cs.CV

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2209.02933v1)

**Authors**: Sudipta Banerjee, Prateek Jaiswal, Arun Ross

**Abstracts**: A face morph is created by strategically combining two or more face images corresponding to multiple identities. The intention is for the morphed image to match with multiple identities. Current morph attack detection strategies can detect morphs but cannot recover the images or identities used in creating them. The task of deducing the individual face images from a morphed face image is known as \textit{de-morphing}. Existing work in de-morphing assume the availability of a reference image pertaining to one identity in order to recover the image of the accomplice - i.e., the other identity. In this work, we propose a novel de-morphing method that can recover images of both identities simultaneously from a single morphed face image without needing a reference image or prior information about the morphing process. We propose a generative adversarial network that achieves single image-based de-morphing with a surprisingly high degree of visual realism and biometric similarity with the original face images. We demonstrate the performance of our method on landmark-based morphs and generative model-based morphs with promising results.



## **18. Annealing Optimization for Progressive Learning with Stochastic Approximation**

eess.SY

arXiv admin note: text overlap with arXiv:2102.05836

**SubmitDate**: 2022-09-06    [paper-pdf](http://arxiv.org/pdf/2209.02826v1)

**Authors**: Christos Mavridis, John Baras

**Abstracts**: In this work, we introduce a learning model designed to meet the needs of applications in which computational resources are limited, and robustness and interpretability are prioritized. Learning problems can be formulated as constrained stochastic optimization problems, with the constraints originating mainly from model assumptions that define a trade-off between complexity and performance. This trade-off is closely related to over-fitting, generalization capacity, and robustness to noise and adversarial attacks, and depends on both the structure and complexity of the model, as well as the properties of the optimization methods used. We develop an online prototype-based learning algorithm based on annealing optimization that is formulated as an online gradient-free stochastic approximation algorithm. The learning model can be viewed as an interpretable and progressively growing competitive-learning neural network model to be used for supervised, unsupervised, and reinforcement learning. The annealing nature of the algorithm contributes to minimal hyper-parameter tuning requirements, poor local minima prevention, and robustness with respect to the initial conditions. At the same time, it provides online control over the performance-complexity trade-off by progressively increasing the complexity of the learning model as needed, through an intuitive bifurcation phenomenon. Finally, the use of stochastic approximation enables the study of the convergence of the learning algorithm through mathematical tools from dynamical systems and control, and allows for its integration with reinforcement learning algorithms, constructing an adaptive state-action aggregation scheme.



## **19. Bag of Tricks for FGSM Adversarial Training**

cs.CV

**SubmitDate**: 2022-09-06    [paper-pdf](http://arxiv.org/pdf/2209.02684v1)

**Authors**: Zichao Li, Li Liu, Zeyu Wang, Yuyin Zhou, Cihang Xie

**Abstracts**: Adversarial training (AT) with samples generated by Fast Gradient Sign Method (FGSM), also known as FGSM-AT, is a computationally simple method to train robust networks. However, during its training procedure, an unstable mode of "catastrophic overfitting" has been identified in arXiv:2001.03994 [cs.LG], where the robust accuracy abruptly drops to zero within a single training step. Existing methods use gradient regularizers or random initialization tricks to attenuate this issue, whereas they either take high computational cost or lead to lower robust accuracy. In this work, we provide the first study, which thoroughly examines a collection of tricks from three perspectives: Data Initialization, Network Structure, and Optimization, to overcome the catastrophic overfitting in FGSM-AT.   Surprisingly, we find that simple tricks, i.e., a) masking partial pixels (even without randomness), b) setting a large convolution stride and smooth activation functions, or c) regularizing the weights of the first convolutional layer, can effectively tackle the overfitting issue. Extensive results on a range of network architectures validate the effectiveness of each proposed trick, and the combinations of tricks are also investigated. For example, trained with PreActResNet-18 on CIFAR-10, our method attains 49.8% accuracy against PGD-50 attacker and 46.4% accuracy against AutoAttack, demonstrating that pure FGSM-AT is capable of enabling robust learners. The code and models are publicly available at https://github.com/UCSC-VLAA/Bag-of-Tricks-for-FGSM-AT.



## **20. Improving the Accuracy and Robustness of CNNs Using a Deep CCA Neural Data Regularizer**

cs.CV

**SubmitDate**: 2022-09-06    [paper-pdf](http://arxiv.org/pdf/2209.02582v1)

**Authors**: Cassidy Pirlot, Richard C. Gerum, Cory Efird, Joel Zylberberg, Alona Fyshe

**Abstracts**: As convolutional neural networks (CNNs) become more accurate at object recognition, their representations become more similar to the primate visual system. This finding has inspired us and other researchers to ask if the implication also runs the other way: If CNN representations become more brain-like, does the network become more accurate? Previous attempts to address this question showed very modest gains in accuracy, owing in part to limitations of the regularization method. To overcome these limitations, we developed a new neural data regularizer for CNNs that uses Deep Canonical Correlation Analysis (DCCA) to optimize the resemblance of the CNN's image representations to that of the monkey visual cortex. Using this new neural data regularizer, we see much larger performance gains in both classification accuracy and within-super-class accuracy, as compared to the previous state-of-the-art neural data regularizers. These networks are also more robust to adversarial attacks than their unregularized counterparts. Together, these results confirm that neural data regularization can push CNN performance higher, and introduces a new method that obtains a larger performance boost.



## **21. Instance Attack:An Explanation-based Vulnerability Analysis Framework Against DNNs for Malware Detection**

cs.CR

**SubmitDate**: 2022-09-06    [paper-pdf](http://arxiv.org/pdf/2209.02453v1)

**Authors**: Sun RuiJin, Guo ShiZe, Guo JinHong, Xing ChangYou, Yang LuMing, Guo Xi, Pan ZhiSong

**Abstracts**: Deep neural networks (DNNs) are increasingly being applied in malware detection and their robustness has been widely debated. Traditionally an adversarial example generation scheme relies on either detailed model information (gradient-based methods) or lots of samples to train a surrogate model, neither of which are available in most scenarios.   We propose the notion of the instance-based attack. Our scheme is interpretable and can work in a black-box environment. Given a specific binary example and a malware classifier, we use the data augmentation strategies to produce enough data from which we can train a simple interpretable model. We explain the detection model by displaying the weight of different parts of the specific binary. By analyzing the explanations, we found that the data subsections play an important role in Windows PE malware detection. We proposed a new function preserving transformation algorithm that can be applied to data subsections. By employing the binary-diversification techniques that we proposed, we eliminated the influence of the most weighted part to generate adversarial examples. Our algorithm can fool the DNNs in certain cases with a success rate of nearly 100\%. Our method outperforms the state-of-the-art method . The most important aspect is that our method operates in black-box settings and the results can be validated with domain knowledge. Our analysis model can assist people in improving the robustness of malware detectors.



## **22. MACAB: Model-Agnostic Clean-Annotation Backdoor to Object Detection with Natural Trigger in Real-World**

cs.CV

**SubmitDate**: 2022-09-06    [paper-pdf](http://arxiv.org/pdf/2209.02339v1)

**Authors**: Hua Ma, Yinshan Li, Yansong Gao, Zhi Zhang, Alsharif Abuadbba, Anmin Fu, Said F. Al-Sarawi, Nepal Surya, Derek Abbott

**Abstracts**: Object detection is the foundation of various critical computer-vision tasks such as segmentation, object tracking, and event detection. To train an object detector with satisfactory accuracy, a large amount of data is required. However, due to the intensive workforce involved with annotating large datasets, such a data curation task is often outsourced to a third party or relied on volunteers. This work reveals severe vulnerabilities of such data curation pipeline. We propose MACAB that crafts clean-annotated images to stealthily implant the backdoor into the object detectors trained on them even when the data curator can manually audit the images. We observe that the backdoor effect of both misclassification and the cloaking are robustly achieved in the wild when the backdoor is activated with inconspicuously natural physical triggers. Backdooring non-classification object detection with clean-annotation is challenging compared to backdooring existing image classification tasks with clean-label, owing to the complexity of having multiple objects within each frame, including victim and non-victim objects. The efficacy of the MACAB is ensured by constructively i abusing the image-scaling function used by the deep learning framework, ii incorporating the proposed adversarial clean image replica technique, and iii combining poison data selection criteria given constrained attacking budget. Extensive experiments demonstrate that MACAB exhibits more than 90% attack success rate under various real-world scenes. This includes both cloaking and misclassification backdoor effect even restricted with a small attack budget. The poisoned samples cannot be effectively identified by state-of-the-art detection techniques.The comprehensive video demo is at https://youtu.be/MA7L_LpXkp4, which is based on a poison rate of 0.14% for YOLOv4 cloaking backdoor and Faster R-CNN misclassification backdoor.



## **23. White-Box Adversarial Policies in Deep Reinforcement Learning**

cs.AI

Code is available at  https://github.com/thestephencasper/white_box_rarl

**SubmitDate**: 2022-09-05    [paper-pdf](http://arxiv.org/pdf/2209.02167v1)

**Authors**: Stephen Casper, Dylan Hadfield-Menell, Gabriel Kreiman

**Abstracts**: Adversarial examples against AI systems pose both risks via malicious attacks and opportunities for improving robustness via adversarial training. In multiagent settings, adversarial policies can be developed by training an adversarial agent to minimize a victim agent's rewards. Prior work has studied black-box attacks where the adversary only sees the state observations and effectively treats the victim as any other part of the environment. In this work, we experiment with white-box adversarial policies to study whether an agent's internal state can offer useful information for other agents. We make three contributions. First, we introduce white-box adversarial policies in which an attacker can observe a victim's internal state at each timestep. Second, we demonstrate that white-box access to a victim makes for better attacks in two-agent environments, resulting in both faster initial learning and higher asymptotic performance against the victim. Third, we show that training against white-box adversarial policies can be used to make learners in single-agent environments more robust to domain shifts.



## **24. Reinforcement learning-based optimised control for tracking of nonlinear systems with adversarial attacks**

eess.SY

Submitted for The 10th RSI International Conference on Robotics and  Mechatronics (ICRoM 2022)

**SubmitDate**: 2022-09-05    [paper-pdf](http://arxiv.org/pdf/2209.02165v1)

**Authors**: Farshad Rahimi, Sepideh Ziaei

**Abstracts**: This paper introduces a reinforcement learning-based tracking control approach for a class of nonlinear systems using neural networks. In this approach, adversarial attacks were considered both in the actuator and on the outputs. This approach incorporates a simultaneous tracking and optimization process. It is necessary to be able to solve the Hamilton-Jacobi-Bellman equation (HJB) in order to obtain optimal control input, but this is difficult due to the strong nonlinearity terms in the equation. In order to find the solution to the HJB equation, we used a reinforcement learning approach. In this online adaptive learning approach, three neural networks are simultaneously adapted: the critic neural network, the actor neural network, and the adversary neural network. Ultimately, simulation results are presented to demonstrate the effectiveness of the introduced method on a manipulator.



## **25. On the Anonymity of Peer-To-Peer Network Anonymity Schemes Used by Cryptocurrencies**

cs.CR

**SubmitDate**: 2022-09-05    [paper-pdf](http://arxiv.org/pdf/2201.11860v3)

**Authors**: Piyush Kumar Sharma, Devashish Gosain, Claudia Diaz

**Abstracts**: Cryptocurrency systems can be subject to deanonimization attacks by exploiting the network-level communication on their peer-to-peer network. Adversaries who control a set of colluding node(s) within the peer-to-peer network can observe transactions being exchanged and infer the parties involved. Thus, various network anonymity schemes have been proposed to mitigate this problem, with some solutions providing theoretical anonymity guarantees.   In this work, we model such peer-to-peer network anonymity solutions and evaluate their anonymity guarantees. To do so, we propose a novel framework that uses Bayesian inference to obtain the probability distributions linking transactions to their possible originators. We characterize transaction anonymity with those distributions, using entropy as metric of adversarial uncertainty on the originator's identity. In particular, we model Dandelion, Dandelion++ and Lightning Network. We study different configurations and demonstrate that none of them offers acceptable anonymity to their users. For instance, our analysis reveals that in the widely deployed Lightning Network, with 1% strategically chosen colluding nodes the adversary can uniquely determine the originator for about 50% of the total transactions in the network. In Dandelion, an adversary that controls 15% of the nodes has on average uncertainty among only 8 possible originators. Moreover, we observe that due to the way Dandelion and Dandelion++ are designed, increasing the network size does not correspond to an increase in the anonymity set of potential originators. Alarmingly, our longitudinal analysis of Lightning Network reveals rather an inverse trend -- with the growth of the network the overall anonymity decreases.



## **26. Evaluating the Susceptibility of Pre-Trained Language Models via Handcrafted Adversarial Examples**

cs.CL

10 pages, 1 figure, 3 tables

**SubmitDate**: 2022-09-05    [paper-pdf](http://arxiv.org/pdf/2209.02128v1)

**Authors**: Hezekiah J. Branch, Jonathan Rodriguez Cefalu, Jeremy McHugh, Leyla Hujer, Aditya Bahl, Daniel del Castillo Iglesias, Ron Heichman, Ramesh Darwishi

**Abstracts**: Recent advances in the development of large language models have resulted in public access to state-of-the-art pre-trained language models (PLMs), including Generative Pre-trained Transformer 3 (GPT-3) and Bidirectional Encoder Representations from Transformers (BERT). However, evaluations of PLMs, in practice, have shown their susceptibility to adversarial attacks during the training and fine-tuning stages of development. Such attacks can result in erroneous outputs, model-generated hate speech, and the exposure of users' sensitive information. While existing research has focused on adversarial attacks during either the training or the fine-tuning of PLMs, there is a deficit of information on attacks made between these two development phases. In this work, we highlight a major security vulnerability in the public release of GPT-3 and further investigate this vulnerability in other state-of-the-art PLMs. We restrict our work to pre-trained models that have not undergone fine-tuning. Further, we underscore token distance-minimized perturbations as an effective adversarial approach, bypassing both supervised and unsupervised quality measures. Following this approach, we observe a significant decrease in text classification quality when evaluating for semantic similarity.



## **27. PatchZero: Defending against Adversarial Patch Attacks by Detecting and Zeroing the Patch**

cs.CV

Accepted to WACV 2023

**SubmitDate**: 2022-09-05    [paper-pdf](http://arxiv.org/pdf/2207.01795v3)

**Authors**: Ke Xu, Yao Xiao, Zhaoheng Zheng, Kaijie Cai, Ram Nevatia

**Abstracts**: Adversarial patch attacks mislead neural networks by injecting adversarial pixels within a local region. Patch attacks can be highly effective in a variety of tasks and physically realizable via attachment (e.g. a sticker) to the real-world objects. Despite the diversity in attack patterns, adversarial patches tend to be highly textured and different in appearance from natural images. We exploit this property and present PatchZero, a general defense pipeline against white-box adversarial patches without retraining the downstream classifier or detector. Specifically, our defense detects adversaries at the pixel-level and "zeros out" the patch region by repainting with mean pixel values. We further design a two-stage adversarial training scheme to defend against the stronger adaptive attacks. PatchZero achieves SOTA defense performance on the image classification (ImageNet, RESISC45), object detection (PASCAL VOC), and video classification (UCF101) tasks with little degradation in benign performance. In addition, PatchZero transfers to different patch shapes and attack types.



## **28. Improving Out-of-Distribution Detection via Epistemic Uncertainty Adversarial Training**

cs.LG

8 pages, 5 figures

**SubmitDate**: 2022-09-05    [paper-pdf](http://arxiv.org/pdf/2209.03148v1)

**Authors**: Derek Everett, Andre T. Nguyen, Luke E. Richards, Edward Raff

**Abstracts**: The quantification of uncertainty is important for the adoption of machine learning, especially to reject out-of-distribution (OOD) data back to human experts for review. Yet progress has been slow, as a balance must be struck between computational efficiency and the quality of uncertainty estimates. For this reason many use deep ensembles of neural networks or Monte Carlo dropout for reasonable uncertainty estimates at relatively minimal compute and memory. Surprisingly, when we focus on the real-world applicable constraint of $\leq 1\%$ false positive rate (FPR), prior methods fail to reliably detect OOD samples as such. Notably, even Gaussian random noise fails to trigger these popular OOD techniques. We help to alleviate this problem by devising a simple adversarial training scheme that incorporates an attack of the epistemic uncertainty predicted by the dropout ensemble. We demonstrate this method improves OOD detection performance on standard data (i.e., not adversarially crafted), and improves the standardized partial AUC from near-random guessing performance to $\geq 0.75$.



## **29. Adversarial Detection: Attacking Object Detection in Real Time**

cs.AI

7 pages, 10 figures

**SubmitDate**: 2022-09-05    [paper-pdf](http://arxiv.org/pdf/2209.01962v1)

**Authors**: Han Wu, Syed Yunas, Sareh Rowlands, Wenjie Ruan, Johan Wahlstrom

**Abstracts**: Intelligent robots hinge on accurate object detection models to perceive the environment. Advances in deep learning security unveil that object detection models are vulnerable to adversarial attacks. However, prior research primarily focuses on attacking static images or offline videos. It is still unclear if such attacks could jeopardize real-world robotic applications in dynamic environments. There is still a gap between theoretical discoveries and real-world applications. We bridge the gap by proposing the first real-time online attack against object detection models. We devised three attacks that fabricate bounding boxes for nonexistent objects at desired locations.



## **30. Jamming Modulation: An Active Anti-Jamming Scheme**

cs.IT

**SubmitDate**: 2022-09-05    [paper-pdf](http://arxiv.org/pdf/2209.01943v1)

**Authors**: Jianhui Ma, Qiang Li, Zilong Liu, Linsong Du, Hongyang Chen, Nirwan Ansari

**Abstracts**: Providing quality communications under adversarial electronic attacks, e.g., broadband jamming attacks, is a challenging task. Unlike state-of-the-art approaches which treat jamming signals as destructive interference, this paper presents a novel active anti-jamming (AAJ) scheme for a jammed channel to enhance the communication quality between a transmitter node (TN) and receiver node (RN), where the TN actively exploits the jamming signal as a carrier to send messages. Specifically, the TN is equipped with a programmable-gain amplifier, which is capable of re-modulating the jamming signals for jamming modulation. Considering four typical jamming types, we derive both the bit error rates (BER) and the corresponding optimal detection thresholds of the AAJ scheme. The asymptotic performances of the AAJ scheme are discussed under the high jamming-to-noise ratio (JNR) and sampling rate cases. Our analysis shows that there exists a BER floor for sufficiently large JNR. Simulation results indicate that the proposed AAJ scheme allows the TN to communicate with the RN reliably even under extremely strong and/or broadband jamming. Additionally, we investigate the channel capacity of the proposed AAJ scheme and show that the channel capacity of the AAJ scheme outperforms that of the direct transmission when the JNR is relatively high.



## **31. Identifying a Training-Set Attack's Target Using Renormalized Influence Estimation**

cs.LG

Accepted at CCS'2022 -- Extended version including the supplementary  material

**SubmitDate**: 2022-09-05    [paper-pdf](http://arxiv.org/pdf/2201.10055v2)

**Authors**: Zayd Hammoudeh, Daniel Lowd

**Abstracts**: Targeted training-set attacks inject malicious instances into the training set to cause a trained model to mislabel one or more specific test instances. This work proposes the task of target identification, which determines whether a specific test instance is the target of a training-set attack. Target identification can be combined with adversarial-instance identification to find (and remove) the attack instances, mitigating the attack with minimal impact on other predictions. Rather than focusing on a single attack method or data modality, we build on influence estimation, which quantifies each training instance's contribution to a model's prediction. We show that existing influence estimators' poor practical performance often derives from their over-reliance on training instances and iterations with large losses. Our renormalized influence estimators fix this weakness; they far outperform the original estimators at identifying influential groups of training examples in both adversarial and non-adversarial settings, even finding up to 100% of adversarial training instances with no clean-data false positives. Target identification then simplifies to detecting test instances with anomalous influence values. We demonstrate our method's effectiveness on backdoor and poisoning attacks across various data domains, including text, vision, and speech, as well as against a gray-box, adaptive attacker that specifically optimizes the adversarial instances to evade our method. Our source code is available at https://github.com/ZaydH/target_identification.



## **32. "Is your explanation stable?": A Robustness Evaluation Framework for Feature Attribution**

cs.AI

Accepted by ACM CCS 2022

**SubmitDate**: 2022-09-05    [paper-pdf](http://arxiv.org/pdf/2209.01782v1)

**Authors**: Yuyou Gan, Yuhao Mao, Xuhong Zhang, Shouling Ji, Yuwen Pu, Meng Han, Jianwei Yin, Ting Wang

**Abstracts**: Understanding the decision process of neural networks is hard. One vital method for explanation is to attribute its decision to pivotal features. Although many algorithms are proposed, most of them solely improve the faithfulness to the model. However, the real environment contains many random noises, which may leads to great fluctuations in the explanations. More seriously, recent works show that explanation algorithms are vulnerable to adversarial attacks. All of these make the explanation hard to trust in real scenarios.   To bridge this gap, we propose a model-agnostic method \emph{Median Test for Feature Attribution} (MeTFA) to quantify the uncertainty and increase the stability of explanation algorithms with theoretical guarantees. MeTFA has the following two functions: (1) examine whether one feature is significantly important or unimportant and generate a MeTFA-significant map to visualize the results; (2) compute the confidence interval of a feature attribution score and generate a MeTFA-smoothed map to increase the stability of the explanation. Experiments show that MeTFA improves the visual quality of explanations and significantly reduces the instability while maintaining the faithfulness. To quantitatively evaluate the faithfulness of an explanation under different noise settings, we further propose several robust faithfulness metrics. Experiment results show that the MeTFA-smoothed explanation can significantly increase the robust faithfulness. In addition, we use two scenarios to show MeTFA's potential in the applications. First, when applied to the SOTA explanation method to locate context bias for semantic segmentation models, MeTFA-significant explanations use far smaller regions to maintain 99\%+ faithfulness. Second, when tested with different explanation-oriented attacks, MeTFA can help defend vanilla, as well as adaptive, adversarial attacks against explanations.



## **33. An Adaptive Black-box Defense against Trojan Attacks (TrojDef)**

cs.CR

**SubmitDate**: 2022-09-05    [paper-pdf](http://arxiv.org/pdf/2209.01721v1)

**Authors**: Guanxiong Liu, Abdallah Khreishah, Fatima Sharadgah, Issa Khalil

**Abstracts**: Trojan backdoor is a poisoning attack against Neural Network (NN) classifiers in which adversaries try to exploit the (highly desirable) model reuse property to implant Trojans into model parameters for backdoor breaches through a poisoned training process. Most of the proposed defenses against Trojan attacks assume a white-box setup, in which the defender either has access to the inner state of NN or is able to run back-propagation through it. In this work, we propose a more practical black-box defense, dubbed TrojDef, which can only run forward-pass of the NN. TrojDef tries to identify and filter out Trojan inputs (i.e., inputs augmented with the Trojan trigger) by monitoring the changes in the prediction confidence when the input is repeatedly perturbed by random noise. We derive a function based on the prediction outputs which is called the prediction confidence bound to decide whether the input example is Trojan or not. The intuition is that Trojan inputs are more stable as the misclassification only depends on the trigger, while benign inputs will suffer when augmented with noise due to the perturbation of the classification features.   Through mathematical analysis, we show that if the attacker is perfect in injecting the backdoor, the Trojan infected model will be trained to learn the appropriate prediction confidence bound, which is used to distinguish Trojan and benign inputs under arbitrary perturbations. However, because the attacker might not be perfect in injecting the backdoor, we introduce a nonlinear transform to the prediction confidence bound to improve the detection accuracy in practical settings. Extensive empirical evaluations show that TrojDef significantly outperforms the-state-of-the-art defenses and is highly stable under different settings, even when the classifier architecture, the training process, or the hyper-parameters change.



## **34. Autonomous Cross Domain Adaptation under Extreme Label Scarcity**

cs.LG

**SubmitDate**: 2022-09-04    [paper-pdf](http://arxiv.org/pdf/2209.01548v1)

**Authors**: Weiwei Weng, Mahardhika Pratama, Choiru Za'in, Marcus De Carvalho, Rakaraddi Appan, Andri Ashfahani, Edward Yapp Kien Yee

**Abstracts**: A cross domain multistream classification is a challenging problem calling for fast domain adaptations to handle different but related streams in never-ending and rapidly changing environments. Notwithstanding that existing multistream classifiers assume no labelled samples in the target stream, they still incur expensive labelling cost since they require fully labelled samples of the source stream. This paper aims to attack the problem of extreme label shortage in the cross domain multistream classification problems where only very few labelled samples of the source stream are provided before process runs. Our solution, namely Learning Streaming Process from Partial Ground Truth (LEOPARD), is built upon a flexible deep clustering network where its hidden nodes, layers and clusters are added and removed dynamically in respect to varying data distributions. A deep clustering strategy is underpinned by a simultaneous feature learning and clustering technique leading to clustering-friendly latent spaces. A domain adaptation strategy relies on the adversarial domain adaptation technique where a feature extractor is trained to fool a domain classifier classifying source and target streams. Our numerical study demonstrates the efficacy of LEOPARD where it delivers improved performances compared to prominent algorithms in 15 of 24 cases. Source codes of LEOPARD are shared in \url{https://github.com/wengweng001/LEOPARD.git} to enable further study.



## **35. Are Attribute Inference Attacks Just Imputation?**

cs.CR

13 (main body) + 4 (references and appendix) pages. To appear in  CCS'22

**SubmitDate**: 2022-09-02    [paper-pdf](http://arxiv.org/pdf/2209.01292v1)

**Authors**: Bargav Jayaraman, David Evans

**Abstracts**: Models can expose sensitive information about their training data. In an attribute inference attack, an adversary has partial knowledge of some training records and access to a model trained on those records, and infers the unknown values of a sensitive feature of those records. We study a fine-grained variant of attribute inference we call \emph{sensitive value inference}, where the adversary's goal is to identify with high confidence some records from a candidate set where the unknown attribute has a particular sensitive value. We explicitly compare attribute inference with data imputation that captures the training distribution statistics, under various assumptions about the training data available to the adversary. Our main conclusions are: (1) previous attribute inference methods do not reveal more about the training data from the model than can be inferred by an adversary without access to the trained model, but with the same knowledge of the underlying distribution as needed to train the attribute inference attack; (2) black-box attribute inference attacks rarely learn anything that cannot be learned without the model; but (3) white-box attacks, which we introduce and evaluate in the paper, can reliably identify some records with the sensitive value attribute that would not be predicted without having access to the model. Furthermore, we show that proposed defenses such as differentially private training and removing vulnerable records from training do not mitigate this privacy risk. The code for our experiments is available at \url{https://github.com/bargavj/EvaluatingDPML}.



## **36. Semi-supervised Conditional GAN for Simultaneous Generation and Detection of Phishing URLs: A Game theoretic Perspective**

cs.CR

Accepted to ICMLA 2022

**SubmitDate**: 2022-09-02    [paper-pdf](http://arxiv.org/pdf/2108.01852v2)

**Authors**: Sharif Amit Kamran, Shamik Sengupta, Alireza Tavakkoli

**Abstracts**: Spear Phishing is a type of cyber-attack where the attacker sends hyperlinks through email on well-researched targets. The objective is to obtain sensitive information by imitating oneself as a trustworthy website. In recent times, deep learning has become the standard for defending against such attacks. However, these architectures were designed with only defense in mind. Moreover, the attacker's perspective and motivation are absent while creating such models. To address this, we need a game-theoretic approach to understand the perspective of the attacker (Hacker) and the defender (Phishing URL detector). We propose a Conditional Generative Adversarial Network with novel training strategy for real-time phishing URL detection. Additionally, we train our architecture in a semi-supervised manner to distinguish between adversarial and real examples, along with detecting malicious and benign URLs. We also design two games between the attacker and defender in training and deployment settings by utilizing the game-theoretic perspective. Our experiments confirm that the proposed architecture surpasses recent state-of-the-art architectures for phishing URLs detection.



## **37. Bayesian Pseudo Labels: Expectation Maximization for Robust and Efficient Semi-Supervised Segmentation**

cs.CV

MICCAI 2022 (Early accept, Student Travel Award)

**SubmitDate**: 2022-09-02    [paper-pdf](http://arxiv.org/pdf/2208.04435v2)

**Authors**: Mou-Cheng Xu, Yukun Zhou, Chen Jin, Marius de Groot, Daniel C. Alexander, Neil P. Oxtoby, Yipeng Hu, Joseph Jacob

**Abstracts**: This paper concerns pseudo labelling in segmentation. Our contribution is fourfold. Firstly, we present a new formulation of pseudo-labelling as an Expectation-Maximization (EM) algorithm for clear statistical interpretation. Secondly, we propose a semi-supervised medical image segmentation method purely based on the original pseudo labelling, namely SegPL. We demonstrate SegPL is a competitive approach against state-of-the-art consistency regularisation based methods on semi-supervised segmentation on a 2D multi-class MRI brain tumour segmentation task and a 3D binary CT lung vessel segmentation task. The simplicity of SegPL allows less computational cost comparing to prior methods. Thirdly, we demonstrate that the effectiveness of SegPL may originate from its robustness against out-of-distribution noises and adversarial attacks. Lastly, under the EM framework, we introduce a probabilistic generalisation of SegPL via variational inference, which learns a dynamic threshold for pseudo labelling during the training. We show that SegPL with variational inference can perform uncertainty estimation on par with the gold-standard method Deep Ensemble.



## **38. Subject Membership Inference Attacks in Federated Learning**

cs.LG

**SubmitDate**: 2022-09-02    [paper-pdf](http://arxiv.org/pdf/2206.03317v2)

**Authors**: Anshuman Suri, Pallika Kanani, Virendra J. Marathe, Daniel W. Peterson

**Abstracts**: Privacy attacks on Machine Learning (ML) models often focus on inferring the existence of particular data points in the training data. However, what the adversary really wants to know is if a particular \emph{individual}'s (\emph{subject}'s) data was included during training. In such scenarios, the adversary is more likely to have access to the distribution of a particular subject, than actual records. Furthermore, in settings like cross-silo Federated Learning (FL), a subject's data can be embodied by multiple data records that are spread across multiple organizations. Nearly all of the existing private FL literature is dedicated to studying privacy at two granularities -- item-level (individual data records), and user-level (participating user in the federation), neither of which apply to data subjects in cross-silo FL. This insight motivates us to shift our attention from the privacy of data records to the privacy of \emph{data subjects}, also known as subject-level privacy. We propose two black-box attacks for \emph{subject membership inference}, of which one assumes access to a model after each training round. Using these attacks, we estimate subject membership inference risk on real-world data for single-party models as well as FL scenarios. We find our attacks to be extremely potent, even without access to exact training records, and using the knowledge of membership for a handful of subjects. To better understand the various factors that may influence subject privacy risk in cross-silo FL settings, we systematically generate several hundred synthetic federation configurations, varying properties of the data, model design and training, and the federation itself. Finally, we investigate the effectiveness of Differential Privacy in mitigating this threat.



## **39. Group Property Inference Attacks Against Graph Neural Networks**

cs.LG

Full version of the ACM CCS'22 paper

**SubmitDate**: 2022-09-02    [paper-pdf](http://arxiv.org/pdf/2209.01100v1)

**Authors**: Xiuling Wang, Wendy Hui Wang

**Abstracts**: With the fast adoption of machine learning (ML) techniques, sharing of ML models is becoming popular. However, ML models are vulnerable to privacy attacks that leak information about the training data. In this work, we focus on a particular type of privacy attacks named property inference attack (PIA) which infers the sensitive properties of the training data through the access to the target ML model. In particular, we consider Graph Neural Networks (GNNs) as the target model, and distribution of particular groups of nodes and links in the training graph as the target property. While the existing work has investigated PIAs that target at graph-level properties, no prior works have studied the inference of node and link properties at group level yet.   In this work, we perform the first systematic study of group property inference attacks (GPIA) against GNNs. First, we consider a taxonomy of threat models under both black-box and white-box settings with various types of adversary knowledge, and design six different attacks for these settings. We evaluate the effectiveness of these attacks through extensive experiments on three representative GNN models and three real-world graphs. Our results demonstrate the effectiveness of these attacks whose accuracy outperforms the baseline approaches. Second, we analyze the underlying factors that contribute to GPIA's success, and show that the target model trained on the graphs with or without the target property represents some dissimilarity in model parameters and/or model outputs, which enables the adversary to infer the existence of the property. Further, we design a set of defense mechanisms against the GPIA attacks, and demonstrate that these mechanisms can reduce attack accuracy effectively with small loss on GNN model accuracy.



## **40. Scalable Adversarial Attack Algorithms on Influence Maximization**

cs.SI

11 pages, 2 figures

**SubmitDate**: 2022-09-02    [paper-pdf](http://arxiv.org/pdf/2209.00892v1)

**Authors**: Lichao Sun, Xiaobin Rui, Wei Chen

**Abstracts**: In this paper, we study the adversarial attacks on influence maximization under dynamic influence propagation models in social networks. In particular, given a known seed set S, the problem is to minimize the influence spread from S by deleting a limited number of nodes and edges. This problem reflects many application scenarios, such as blocking virus (e.g. COVID-19) propagation in social networks by quarantine and vaccination, blocking rumor spread by freezing fake accounts, or attacking competitor's influence by incentivizing some users to ignore the information from the competitor. In this paper, under the linear threshold model, we adapt the reverse influence sampling approach and provide efficient algorithms of sampling valid reverse reachable paths to solve the problem.



## **41. Adversarial Color Film: Effective Physical-World Attack to DNNs**

cs.CV

**SubmitDate**: 2022-09-02    [paper-pdf](http://arxiv.org/pdf/2209.02430v1)

**Authors**: Chengyin Hu, Weiwen Shi

**Abstracts**: It is well known that the performance of deep neural networks (DNNs) is susceptible to subtle interference. So far, camera-based physical adversarial attacks haven't gotten much attention, but it is the vacancy of physical attack. In this paper, we propose a simple and efficient camera-based physical attack called Adversarial Color Film (AdvCF), which manipulates the physical parameters of color film to perform attacks. Carefully designed experiments show the effectiveness of the proposed method in both digital and physical environments. In addition, experimental results show that the adversarial samples generated by AdvCF have excellent performance in attack transferability, which enables AdvCF effective black-box attacks. At the same time, we give the guidance of defense against AdvCF by means of adversarial training. Finally, we look into AdvCF's threat to future vision-based systems and propose some promising mentality for camera-based physical attacks.



## **42. Impact of Colour Variation on Robustness of Deep Neural Networks**

cs.CV

arXiv admin note: substantial text overlap with arXiv:2209.02132

**SubmitDate**: 2022-09-02    [paper-pdf](http://arxiv.org/pdf/2209.02832v1)

**Authors**: Chengyin Hu, Weiwen Shi

**Abstracts**: Deep neural networks (DNNs) have have shown state-of-the-art performance for computer vision applications like image classification, segmentation and object detection. Whereas recent advances have shown their vulnerability to manual digital perturbations in the input data, namely adversarial attacks. The accuracy of the networks is significantly affected by the data distribution of their training dataset. Distortions or perturbations on color space of input images generates out-of-distribution data, which make networks more likely to misclassify them. In this work, we propose a color-variation dataset by distorting their RGB color on a subset of the ImageNet with 27 different combinations. The aim of our work is to study the impact of color variation on the performance of DNNs. We perform experiments on several state-of-the-art DNN architectures on the proposed dataset, and the result shows a significant correlation between color variation and loss of accuracy. Furthermore, based on the ResNet50 architecture, we demonstrate some experiments of the performance of recently proposed robust training techniques and strategies, such as Augmix, revisit, and free normalizer, on our proposed dataset. Experimental results indicate that these robust training techniques can improve the robustness of deep networks to color variation.



## **43. Impact of Scaled Image on Robustness of Deep Neural Networks**

cs.CV

**SubmitDate**: 2022-09-02    [paper-pdf](http://arxiv.org/pdf/2209.02132v1)

**Authors**: Chengyin Hu, Weiwen Shi

**Abstracts**: Deep neural networks (DNNs) have been widely used in computer vision tasks like image classification, object detection and segmentation. Whereas recent studies have shown their vulnerability to manual digital perturbations or distortion in the input images. The accuracy of the networks is remarkably influenced by the data distribution of their training dataset. Scaling the raw images creates out-of-distribution data, which makes it a possible adversarial attack to fool the networks. In this work, we propose a Scaling-distortion dataset ImageNet-CS by Scaling a subset of the ImageNet Challenge dataset by different multiples. The aim of our work is to study the impact of scaled images on the performance of advanced DNNs. We perform experiments on several state-of-the-art deep neural network architectures on the proposed ImageNet-CS, and the results show a significant positive correlation between scaling size and accuracy decline. Moreover, based on ResNet50 architecture, we demonstrate some tests on the performance of recent proposed robust training techniques and strategies like Augmix, Revisiting and Normalizer Free on our proposed ImageNet-CS. Experiment results have shown that these robust training techniques can improve networks' robustness to scaling transformation.



## **44. Reliable Representations Make A Stronger Defender: Unsupervised Structure Refinement for Robust GNN**

cs.LG

Accepted in KDD2022

**SubmitDate**: 2022-09-02    [paper-pdf](http://arxiv.org/pdf/2207.00012v2)

**Authors**: Kuan Li, Yang Liu, Xiang Ao, Jianfeng Chi, Jinghua Feng, Hao Yang, Qing He

**Abstracts**: Benefiting from the message passing mechanism, Graph Neural Networks (GNNs) have been successful on flourish tasks over graph data. However, recent studies have shown that attackers can catastrophically degrade the performance of GNNs by maliciously modifying the graph structure. A straightforward solution to remedy this issue is to model the edge weights by learning a metric function between pairwise representations of two end nodes, which attempts to assign low weights to adversarial edges. The existing methods use either raw features or representations learned by supervised GNNs to model the edge weights. However, both strategies are faced with some immediate problems: raw features cannot represent various properties of nodes (e.g., structure information), and representations learned by supervised GNN may suffer from the poor performance of the classifier on the poisoned graph. We need representations that carry both feature information and as mush correct structure information as possible and are insensitive to structural perturbations. To this end, we propose an unsupervised pipeline, named STABLE, to optimize the graph structure. Finally, we input the well-refined graph into a downstream classifier. For this part, we design an advanced GCN that significantly enhances the robustness of vanilla GCN without increasing the time complexity. Extensive experiments on four real-world graph benchmarks demonstrate that STABLE outperforms the state-of-the-art methods and successfully defends against various attacks.



## **45. Universal Fourier Attack for Time Series**

cs.CR

**SubmitDate**: 2022-09-02    [paper-pdf](http://arxiv.org/pdf/2209.00757v1)

**Authors**: Elizabeth Coda, Brad Clymer, Chance DeSmet, Yijing Watkins, Michael Girard

**Abstracts**: A wide variety of adversarial attacks have been proposed and explored using image and audio data. These attacks are notoriously easy to generate digitally when the attacker can directly manipulate the input to a model, but are much more difficult to implement in the real-world. In this paper we present a universal, time invariant attack for general time series data such that the attack has a frequency spectrum primarily composed of the frequencies present in the original data. The universality of the attack makes it fast and easy to implement as no computation is required to add it to an input, while time invariance is useful for real-world deployment. Additionally, the frequency constraint ensures the attack can withstand filtering. We demonstrate the effectiveness of the attack in two different domains, speech recognition and unintended radiated emission, and show that the attack is robust against common transform-and-compare defense pipelines.



## **46. Adversarial for Social Privacy: A Poisoning Strategy to Degrade User Identity Linkage**

cs.SI

**SubmitDate**: 2022-09-01    [paper-pdf](http://arxiv.org/pdf/2209.00269v1)

**Authors**: Jiangli Shao, Yongqing Wang, Boshen Shi, Hao Gao, Huawei Shen, Xueqi Cheng

**Abstracts**: Privacy issues on social networks have been extensively discussed in recent years. The user identity linkage (UIL) task, aiming at finding corresponding users across different social networks, would be a threat to privacy if unethically applied. The sensitive user information might be detected through connected identities. A promising and novel solution to this issue is to design an adversarial strategy to degrade the matching performance of UIL models. However, most existing adversarial attacks on graphs are designed for models working in a single network, while UIL is a cross-network learning task. Meanwhile, privacy protection against UIL works unilaterally in real-world scenarios, i.e., the service provider can only add perturbations to its own network to protect its users from being linked. To tackle these challenges, this paper proposes a novel adversarial attack strategy that poisons one target network to prevent its nodes from being linked to other networks by UIL algorithms. Specifically, we reformalize the UIL problem in the perspective of kernelized topology consistency and convert the attack objective to maximizing the structural changes within the target network before and after attacks. A novel graph kernel is then defined with Earth mover's distance (EMD) on the edge-embedding space. In terms of efficiency, a fast attack strategy is proposed by greedy searching and replacing EMD with its lower bound. Results on three real-world datasets indicate that the proposed attacks can best fool a wide range of UIL models and reach a balance between attack effectiveness and imperceptibility.



## **47. FDB: Fraud Dataset Benchmark**

cs.LG

**SubmitDate**: 2022-08-31    [paper-pdf](http://arxiv.org/pdf/2208.14417v2)

**Authors**: Prince Grover, Zheng Li, Jianbo Liu, Jakub Zablocki, Hao Zhou, Julia Xu, Anqi Cheng

**Abstracts**: Standardized datasets and benchmarks have spurred innovations in computer vision, natural language processing, multi-modal and tabular settings. We note that, as compared to other well researched fields fraud detection has numerous differences. The differences include a high class imbalance, diverse feature types, frequently changing fraud patterns, and adversarial nature of the problem. Due to these differences, the modeling approaches that are designed for other classification tasks may not work well for the fraud detection. We introduce Fraud Dataset Benchmark (FDB), a compilation of publicly available datasets catered to fraud detection. FDB comprises variety of fraud related tasks, ranging from identifying fraudulent card-not-present transactions, detecting bot attacks, classifying malicious URLs, predicting risk of loan to content moderation. The Python based library from FDB provides consistent API for data loading with standardized training and testing splits. For reference, we also provide baseline evaluations of different modeling approaches on FDB. Considering the increasing popularity of Automated Machine Learning (AutoML) for various research and business problems, we used AutoML frameworks for our baseline evaluations. For fraud prevention, the organizations that operate with limited resources and lack ML expertise often hire a team of investigators, use blocklists and manual rules, all of which are inefficient and do not scale well. Such organizations can benefit from AutoML solutions that are easy to deploy in production and pass the bar of fraud prevention requirements. We hope that FDB helps in the development of customized fraud detection techniques catered to different fraud modus operandi (MOs) as well as in the improvement of AutoML systems that can work well for all datasets in the benchmark.



## **48. Wiggle: Physical Challenge-Response Verification of Vehicle Platooning**

cs.CR

10 pages, 13 figures

**SubmitDate**: 2022-08-31    [paper-pdf](http://arxiv.org/pdf/2209.00080v1)

**Authors**: Connor Dickey, Christopher Smith, Quentin Johnson, Jingcheng Li, Ziqi Xu, Loukas Lazos, Ming Li

**Abstracts**: Autonomous vehicle platooning promises many benefits such as fuel efficiency, road safety, reduced traffic congestion, and passenger comfort. Platooning vehicles travel in a single file, in close distance, and at the same velocity. The platoon formation is autonomously maintained by a Cooperative Adaptive Cruise Control (CACC) system which relies on sensory data and vehicle-to-vehicle (V2V) communications. In fact, V2V messages play a critical role in shortening the platooning distance while maintaining safety. Whereas V2V message integrity and source authentication can be verified via cryptographic methods, establishing the truthfulness of the message contents is a much harder task.   This work establishes a physical access control mechanism to restrict V2V messages to platooning members. Specifically, we aim at tying the digital identity of a candidate requesting to join a platoon to its physical trajectory relative to the platoon. We propose the {\em Wiggle} protocol that employs a physical challenge-response exchange to prove that a candidate requesting to be admitted into a platoon actually follows it. The protocol name is inspired by the random longitudinal movements that the candidate is challenged to execute. {\em Wiggle} prevents any remote adversary from joining the platoon and injecting fake CACC messages. Compared to prior works, {\em Wiggle} is resistant to pre-recording attacks and can verify that the candidate is directly behind the verifier at the same lane.



## **49. Membership Inference Attacks by Exploiting Loss Trajectory**

cs.CR

Accepted by CCS 2022

**SubmitDate**: 2022-08-31    [paper-pdf](http://arxiv.org/pdf/2208.14933v1)

**Authors**: Yiyong Liu, Zhengyu Zhao, Michael Backes, Yang Zhang

**Abstracts**: Machine learning models are vulnerable to membership inference attacks in which an adversary aims to predict whether or not a particular sample was contained in the target model's training dataset. Existing attack methods have commonly exploited the output information (mostly, losses) solely from the given target model. As a result, in practical scenarios where both the member and non-member samples yield similarly small losses, these methods are naturally unable to differentiate between them. To address this limitation, in this paper, we propose a new attack method, called \system, which can exploit the membership information from the whole training process of the target model for improving the attack performance. To mount the attack in the common black-box setting, we leverage knowledge distillation, and represent the membership information by the losses evaluated on a sequence of intermediate models at different distillation epochs, namely \emph{distilled loss trajectory}, together with the loss from the given target model. Experimental results over different datasets and model architectures demonstrate the great advantage of our attack in terms of different metrics. For example, on CINIC-10, our attack achieves at least 6$\times$ higher true-positive rate at a low false-positive rate of 0.1\% than existing methods. Further analysis demonstrates the general effectiveness of our attack in more strict scenarios.



## **50. Be Your Own Neighborhood: Detecting Adversarial Example by the Neighborhood Relations Built on Self-Supervised Learning**

cs.LG

co-first author

**SubmitDate**: 2022-08-31    [paper-pdf](http://arxiv.org/pdf/2209.00005v1)

**Authors**: Zhiyuan He, Yijun Yang, Pin-Yu Chen, Qiang Xu, Tsung-Yi Ho

**Abstracts**: Deep Neural Networks (DNNs) have achieved excellent performance in various fields. However, DNNs' vulnerability to Adversarial Examples (AE) hinders their deployments to safety-critical applications. This paper presents a novel AE detection framework, named BEYOND, for trustworthy predictions. BEYOND performs the detection by distinguishing the AE's abnormal relation with its augmented versions, i.e. neighbors, from two prospects: representation similarity and label consistency. An off-the-shelf Self-Supervised Learning (SSL) model is used to extract the representation and predict the label for its highly informative representation capacity compared to supervised learning models. For clean samples, their representations and predictions are closely consistent with their neighbors, whereas those of AEs differ greatly. Furthermore, we explain this observation and show that by leveraging this discrepancy BEYOND can effectively detect AEs. We develop a rigorous justification for the effectiveness of BEYOND. Furthermore, as a plug-and-play model, BEYOND can easily cooperate with the Adversarial Trained Classifier (ATC), achieving the state-of-the-art (SOTA) robustness accuracy. Experimental results show that BEYOND outperforms baselines by a large margin, especially under adaptive attacks. Empowered by the robust relation net built on SSL, we found that BEYOND outperforms baselines in terms of both detection ability and speed. Our code will be publicly available.



