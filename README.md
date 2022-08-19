# Latest Adversarial Attack Papers
**update at 2022-08-20 06:31:38**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. How many perturbations break this model? Evaluating robustness beyond adversarial accuracy**

cs.LG

**SubmitDate**: 2022-08-18    [paper-pdf](http://arxiv.org/pdf/2207.04129v2)

**Authors**: Raphael Olivier, Bhiksha Raj

**Abstracts**: Robustness to adversarial attack is typically evaluated with adversarial accuracy. This metric quantifies the number of points for which, given a threat model, successful adversarial perturbations cannot be found. While essential, this metric does not capture all aspects of robustness and in particular leaves out the question of how many perturbations can be found for each point. In this work we introduce an alternative approach, adversarial sparsity, which quantifies how difficult it is to find a successful perturbation given both an input point and a constraint on the direction of the perturbation. This constraint may be angular (L2 perturbations), or based on the number of pixels (Linf perturbations).   We show that sparsity provides valuable insight on neural networks in multiple ways. analyzing the sparsity of existing robust models illustrates important differences between them that accuracy analysis does not, and suggests approaches for improving their robustness. When applying broken defenses effective against weak attacks but not strong ones, sparsity can discriminate between the totally ineffective and the partially effective defenses. Finally, with sparsity we can measure increases in robustness that do not affect accuracy: we show for example that data augmentation can by itself increase adversarial robustness, without using adversarial training.



## **2. Resisting Adversarial Attacks in Deep Neural Networks using Diverse Decision Boundaries**

cs.LG

**SubmitDate**: 2022-08-18    [paper-pdf](http://arxiv.org/pdf/2208.08697v1)

**Authors**: Manaar Alam, Shubhajit Datta, Debdeep Mukhopadhyay, Arijit Mondal, Partha Pratim Chakrabarti

**Abstracts**: The security of deep learning (DL) systems is an extremely important field of study as they are being deployed in several applications due to their ever-improving performance to solve challenging tasks. Despite overwhelming promises, the deep learning systems are vulnerable to crafted adversarial examples, which may be imperceptible to the human eye, but can lead the model to misclassify. Protections against adversarial perturbations on ensemble-based techniques have either been shown to be vulnerable to stronger adversaries or shown to lack an end-to-end evaluation. In this paper, we attempt to develop a new ensemble-based solution that constructs defender models with diverse decision boundaries with respect to the original model. The ensemble of classifiers constructed by (1) transformation of the input by a method called Split-and-Shuffle, and (2) restricting the significant features by a method called Contrast-Significant-Features are shown to result in diverse gradients with respect to adversarial attacks, which reduces the chance of transferring adversarial examples from the original to the defender model targeting the same class. We present extensive experimentations using standard image classification datasets, namely MNIST, CIFAR-10 and CIFAR-100 against state-of-the-art adversarial attacks to demonstrate the robustness of the proposed ensemble-based defense. We also evaluate the robustness in the presence of a stronger adversary targeting all the models within the ensemble simultaneously. Results for the overall false positives and false negatives have been furnished to estimate the overall performance of the proposed methodology.



## **3. Reverse Engineering of Integrated Circuits: Tools and Techniques**

cs.CR

**SubmitDate**: 2022-08-18    [paper-pdf](http://arxiv.org/pdf/2208.08689v1)

**Authors**: Abhijitt Dhavlle

**Abstracts**: Consumer and defense systems demanded design and manufacturing of electronics with increased performance, compared to their predecessors. As such systems became ubiquitous in a plethora of domains, their application surface increased, thus making them a target for adversaries. Hence, with improved performance the aspect of security demanded even more attention of the designers. The research community is rife with extensive details of attacks that target the confidential design details by exploiting vulnerabilities. The adversary could target the physical design of a semiconductor chip or break a cryptographic algorithm by extracting the secret keys, using attacks that will be discussed in this thesis. This thesis focuses on presenting a brief overview of IC reverse engineering attack and attacks targeting cryptographic systems. Further, the thesis presents my contributions to the defenses for the discussed attacks. The globalization of the Integrated Circuit (IC) supply chain has rendered the advantage of low-cost and high-performance ICs in the market for the end users. But this has also made the design vulnerable to over production, IP Piracy, reverse engineering attacks and hardware malware during the manufacturing and post manufacturing process. Logic locking schemes have been proposed in the past to overcome the design trust issues but the new state-of-the-art attacks such as SAT has proven a larger threat. This work highlights the reverse engineering attack and a proposed hardened platform along with its framework.



## **4. Enhancing Targeted Attack Transferability via Diversified Weight Pruning**

cs.CV

8 pages + 2 pages of references

**SubmitDate**: 2022-08-18    [paper-pdf](http://arxiv.org/pdf/2208.08677v1)

**Authors**: Hung-Jui Wang, Yu-Yu Wu, Shang-Tse Chen

**Abstracts**: Malicious attackers can generate targeted adversarial examples by imposing human-imperceptible noise on images, forcing neural network models to produce specific incorrect outputs. With cross-model transferable adversarial examples, the vulnerability of neural networks remains even if the model information is kept secret from the attacker. Recent studies have shown the effectiveness of ensemble-based methods in generating transferable adversarial examples. However, existing methods fall short under the more challenging scenario of creating targeted attacks transferable among distinct models. In this work, we propose Diversified Weight Pruning (DWP) to further enhance the ensemble-based methods by leveraging the weight pruning method commonly used in model compression. Specifically, we obtain multiple diverse models by a random weight pruning method. These models preserve similar accuracies and can serve as additional models for ensemble-based methods, yielding stronger transferable targeted attacks. Experiments on ImageNet-Compatible Dataset under the more challenging scenarios are provided: transferring to distinct architectures and to adversarially trained models. The results show that our proposed DWP improves the targeted attack success rates with up to 4.1% and 8.0% on the combination of state-of-the-art methods, respectively



## **5. Enhancing Diffusion-Based Image Synthesis with Robust Classifier Guidance**

cs.CV

**SubmitDate**: 2022-08-18    [paper-pdf](http://arxiv.org/pdf/2208.08664v1)

**Authors**: Bahjat Kawar, Roy Ganz, Michael Elad

**Abstracts**: Denoising diffusion probabilistic models (DDPMs) are a recent family of generative models that achieve state-of-the-art results. In order to obtain class-conditional generation, it was suggested to guide the diffusion process by gradients from a time-dependent classifier. While the idea is theoretically sound, deep learning-based classifiers are infamously susceptible to gradient-based adversarial attacks. Therefore, while traditional classifiers may achieve good accuracy scores, their gradients are possibly unreliable and might hinder the improvement of the generation results. Recent work discovered that adversarially robust classifiers exhibit gradients that are aligned with human perception, and these could better guide a generative process towards semantically meaningful images. We utilize this observation by defining and training a time-dependent adversarially robust classifier and use it as guidance for a generative diffusion model. In experiments on the highly challenging and diverse ImageNet dataset, our scheme introduces significantly more intelligible intermediate gradients, better alignment with theoretical findings, as well as improved generation results under several evaluation metrics. Furthermore, we conduct an opinion survey whose findings indicate that human raters prefer our method's results.



## **6. Efficient Detection and Filtering Systems for Distributed Training**

cs.LG

18 pages, 14 figures, 6 tables. arXiv admin note: substantial text  overlap with arXiv:2108.02416

**SubmitDate**: 2022-08-18    [paper-pdf](http://arxiv.org/pdf/2208.08085v2)

**Authors**: Konstantinos Konstantinidis, Aditya Ramamoorthy

**Abstracts**: A plethora of modern machine learning tasks require the utilization of large-scale distributed clusters as a critical component of the training pipeline. However, abnormal Byzantine behavior of the worker nodes can derail the training and compromise the quality of the inference. Such behavior can be attributed to unintentional system malfunctions or orchestrated attacks; as a result, some nodes may return arbitrary results to the parameter server (PS) that coordinates the training. Recent work considers a wide range of attack models and has explored robust aggregation and/or computational redundancy to correct the distorted gradients. In this work, we consider attack models ranging from strong ones: $q$ omniscient adversaries with full knowledge of the defense protocol that can change from iteration to iteration to weak ones: $q$ randomly chosen adversaries with limited collusion abilities which only change every few iterations at a time. Our algorithms rely on redundant task assignments coupled with detection of adversarial behavior. For strong attacks, we demonstrate a reduction in the fraction of distorted gradients ranging from 16%-99% as compared to the prior state-of-the-art. Our top-1 classification accuracy results on the CIFAR-10 data set demonstrate 25% advantage in accuracy (averaged over strong and weak scenarios) under the most sophisticated attacks compared to state-of-the-art methods.



## **7. ObfuNAS: A Neural Architecture Search-based DNN Obfuscation Approach**

cs.CR

9 pages

**SubmitDate**: 2022-08-17    [paper-pdf](http://arxiv.org/pdf/2208.08569v1)

**Authors**: Tong Zhou, Shaolei Ren, Xiaolin Xu

**Abstracts**: Malicious architecture extraction has been emerging as a crucial concern for deep neural network (DNN) security. As a defense, architecture obfuscation is proposed to remap the victim DNN to a different architecture. Nonetheless, we observe that, with only extracting an obfuscated DNN architecture, the adversary can still retrain a substitute model with high performance (e.g., accuracy), rendering the obfuscation techniques ineffective. To mitigate this under-explored vulnerability, we propose ObfuNAS, which converts the DNN architecture obfuscation into a neural architecture search (NAS) problem. Using a combination of function-preserving obfuscation strategies, ObfuNAS ensures that the obfuscated DNN architecture can only achieve lower accuracy than the victim. We validate the performance of ObfuNAS with open-source architecture datasets like NAS-Bench-101 and NAS-Bench-301. The experimental results demonstrate that ObfuNAS can successfully find the optimal mask for a victim model within a given FLOPs constraint, leading up to 2.6% inference accuracy degradation for attackers with only 0.14x FLOPs overhead. The code is available at: https://github.com/Tongzhou0101/ObfuNAS.



## **8. Learning to Generate Image Source-Agnostic Universal Adversarial Perturbations**

cs.LG

**SubmitDate**: 2022-08-17    [paper-pdf](http://arxiv.org/pdf/2009.13714v4)

**Authors**: Pu Zhao, Parikshit Ram, Songtao Lu, Yuguang Yao, Djallel Bouneffouf, Xue Lin, Sijia Liu

**Abstracts**: Adversarial perturbations are critical for certifying the robustness of deep learning models. A universal adversarial perturbation (UAP) can simultaneously attack multiple images, and thus offers a more unified threat model, obviating an image-wise attack algorithm. However, the existing UAP generator is underdeveloped when images are drawn from different image sources (e.g., with different image resolutions). Towards an authentic universality across image sources, we take a novel view of UAP generation as a customized instance of few-shot learning, which leverages bilevel optimization and learning-to-optimize (L2O) techniques for UAP generation with improved attack success rate (ASR). We begin by considering the popular model agnostic meta-learning (MAML) framework to meta-learn a UAP generator. However, we see that the MAML framework does not directly offer the universal attack across image sources, requiring us to integrate it with another meta-learning framework of L2O. The resulting scheme for meta-learning a UAP generator (i) has better performance (50% higher ASR) than baselines such as Projected Gradient Descent, (ii) has better performance (37% faster) than the vanilla L2O and MAML frameworks (when applicable), and (iii) is able to simultaneously handle UAP generation for different victim models and image data sources.



## **9. I-GWAS: Privacy-Preserving Interdependent Genome-Wide Association Studies**

q-bio.GN

**SubmitDate**: 2022-08-17    [paper-pdf](http://arxiv.org/pdf/2208.08361v1)

**Authors**: Túlio Pascoal, Jérémie Decouchant, Antoine Boutet, Marcus Völp

**Abstracts**: Genome-wide Association Studies (GWASes) identify genomic variations that are statistically associated with a trait, such as a disease, in a group of individuals. Unfortunately, careless sharing of GWAS statistics might give rise to privacy attacks. Several works attempted to reconcile secure processing with privacy-preserving releases of GWASes. However, we highlight that these approaches remain vulnerable if GWASes utilize overlapping sets of individuals and genomic variations. In such conditions, we show that even when relying on state-of-the-art techniques for protecting releases, an adversary could reconstruct the genomic variations of up to 28.6% of participants, and that the released statistics of up to 92.3% of the genomic variations would enable membership inference attacks. We introduce I-GWAS, a novel framework that securely computes and releases the results of multiple possibly interdependent GWASes. I-GWAScontinuously releases privacy-preserving and noise-free GWAS results as new genomes become available.



## **10. Attackar: Attack of the Evolutionary Adversary**

cs.CV

**SubmitDate**: 2022-08-17    [paper-pdf](http://arxiv.org/pdf/2208.08297v1)

**Authors**: Raz Lapid, Zvika Haramaty, Moshe Sipper

**Abstracts**: Deep neural networks (DNNs) are sensitive to adversarial data in a variety of scenarios, including the black-box scenario, where the attacker is only allowed to query the trained model and receive an output. Existing black-box methods for creating adversarial instances are costly, often using gradient estimation or training a replacement network. This paper introduces \textit{Attackar}, an evolutionary, score-based, black-box attack. Attackar is based on a novel objective function that can be used in gradient-free optimization problems. The attack only requires access to the output logits of the classifier and is thus not affected by gradient masking. No additional information is needed, rendering our method more suitable to real-life situations. We test its performance with three different state-of-the-art models -- Inception-v3, ResNet-50, and VGG-16-BN -- against three benchmark datasets: MNIST, CIFAR10 and ImageNet. Furthermore, we evaluate Attackar's performance on non-differential transformation defenses and state-of-the-art robust models. Our results demonstrate the superior performance of Attackar, both in terms of accuracy score and query efficiency.



## **11. On the Privacy Effect of Data Enhancement via the Lens of Memorization**

cs.LG

**SubmitDate**: 2022-08-17    [paper-pdf](http://arxiv.org/pdf/2208.08270v1)

**Authors**: Xiao Li, Qiongxiu Li, Zhanhao Hu, Xiaolin Hu

**Abstracts**: Machine learning poses severe privacy concerns as it is shown that the learned models can reveal sensitive information about their training data. Many works have investigated the effect of widely-adopted data augmentation (DA) and adversarial training (AT) techniques, termed data enhancement in the paper, on the privacy leakage of machine learning models. Such privacy effects are often measured by membership inference attacks (MIAs), which aim to identify whether a particular example belongs to the training set or not. We propose to investigate privacy from a new perspective called memorization. Through the lens of memorization, we find that previously deployed MIAs produce misleading results as they are less likely to identify samples with higher privacy risks as members compared to samples with low privacy risks. To solve this problem, we deploy a recent attack that can capture the memorization degrees of individual samples for evaluation. Through extensive experiments, we unveil non-trivial findings about the connections between three important properties of machine learning models, including privacy, generalization gap, and adversarial robustness. We demonstrate that, unlike existing results, the generalization gap is shown not highly correlated with privacy leakage. Moreover, stronger adversarial robustness does not necessarily imply that the model is more susceptible to privacy attacks.



## **12. Robustness of the Tangle 2.0 Consensus**

cs.DC

**SubmitDate**: 2022-08-17    [paper-pdf](http://arxiv.org/pdf/2208.08254v1)

**Authors**: Bing-Yang Lin, Daria Dziubałtowska, Piotr Macek, Andreas Penzkofer, Sebastian Müller

**Abstracts**: In this paper, we investigate the performance of the Tangle 2.0 consensus protocol in a Byzantine environment. We use an agent-based simulation model that incorporates the main features of the Tangle 2.0 consensus protocol. Our experimental results demonstrate that the Tangle 2.0 protocol is robust to the bait-and-switch attack up to the theoretical upper bound of the adversary's 33% voting weight. We further show that the common coin mechanism in Tangle 2.0 is necessary for robustness against powerful adversaries. Moreover, the experimental results confirm that the protocol can achieve around 1s confirmation time in typical scenarios and that the confirmation times of non-conflicting transactions are not affected by the presence of conflicts.



## **13. Two Heads are Better than One: Robust Learning Meets Multi-branch Models**

cs.CV

10 pages, 5 Figures

**SubmitDate**: 2022-08-17    [paper-pdf](http://arxiv.org/pdf/2208.08083v1)

**Authors**: Dong Huang, Qingwen Bu, Yuhao Qing, Haowen Pi, Sen Wang, Heming Cui

**Abstracts**: Deep neural networks (DNNs) are vulnerable to adversarial examples, in which DNNs are misled to false outputs due to inputs containing imperceptible perturbations. Adversarial training, a reliable and effective method of defense, may significantly reduce the vulnerability of neural networks and becomes the de facto standard for robust learning. While many recent works practice the data-centric philosophy, such as how to generate better adversarial examples or use generative models to produce additional training data, we look back to the models themselves and revisit the adversarial robustness from the perspective of deep feature distribution as an insightful complementarity. In this paper, we propose Branch Orthogonality adveRsarial Training (BORT) to obtain state-of-the-art performance with solely the original dataset for adversarial training. To practice our design idea of integrating multiple orthogonal solution spaces, we leverage a simple and straightforward multi-branch neural network that eclipses adversarial attacks with no increase in inference time. We heuristically propose a corresponding loss function, branch-orthogonal loss, to make each solution space of the multi-branch model orthogonal. We evaluate our approach on CIFAR-10, CIFAR-100, and SVHN against \ell_{\infty} norm-bounded perturbations of size \epsilon = 8/255, respectively. Exhaustive experiments are conducted to show that our method goes beyond all state-of-the-art methods without any tricks. Compared to all methods that do not use additional data for training, our models achieve 67.3% and 41.5% robust accuracy on CIFAR-10 and CIFAR-100 (improving upon the state-of-the-art by +7.23% and +9.07%). We also outperform methods using a training set with a far larger scale than ours. All our models and codes are available online at https://github.com/huangd1999/BORT.



## **14. An Efficient Multi-Step Framework for Malware Packing Identification**

cs.CR

**SubmitDate**: 2022-08-17    [paper-pdf](http://arxiv.org/pdf/2208.08071v1)

**Authors**: Jong-Wouk Kim, Yang-Sae Moon, Mi-Jung Choi

**Abstracts**: Malware developers use combinations of techniques such as compression, encryption, and obfuscation to bypass anti-virus software. Malware with anti-analysis technologies can bypass AI-based anti-virus software and malware analysis tools. Therefore, classifying pack files is one of the big challenges. Problems arise if the malware classifiers learn packers' features, not those of malware. Training the models with unintended erroneous data turn into poisoning attacks, adversarial attacks, and evasion attacks. Therefore, researchers should consider packing to build appropriate malware classifier models. In this paper, we propose a multi-step framework for classifying and identifying packed samples which consists of pseudo-optimal feature selection, machine learning-based classifiers, and packer identification steps. In the first step, we use the CART algorithm and the permutation importance to preselect important 20 features. In the second step, each model learns 20 preselected features for classifying the packed files with the highest performance. As a result, the XGBoost, which learned the features preselected by XGBoost with the permutation importance, showed the highest performance of any other experiment scenarios with an accuracy of 99.67%, an F1-Score of 99.46%, and an area under the curve (AUC) of 99.98%. In the third step, we propose a new approach that can identify packers only for samples classified as Well-Known Packed.



## **15. Adversarial Prefetch: New Cross-Core Cache Side Channel Attacks**

cs.CR

camera-ready for IEEE S&P 2022

**SubmitDate**: 2022-08-17    [paper-pdf](http://arxiv.org/pdf/2110.12340v3)

**Authors**: Yanan Guo, Andrew Zigerelli, Youtao Zhang, Jun Yang

**Abstracts**: Modern x86 processors have many prefetch instructions that can be used by programmers to boost performance. However, these instructions may also cause security problems. In particular, we found that on Intel processors, there are two security flaws in the implementation of PREFETCHW, an instruction for accelerating future writes. First, this instruction can execute on data with read-only permission. Second, the execution time of this instruction leaks the current coherence state of the target data.   Based on these two design issues, we build two cross-core private cache attacks that work with both inclusive and non-inclusive LLCs, named Prefetch+Reload and Prefetch+Prefetch. We demonstrate the significance of our attacks in different scenarios. First, in the covert channel case, Prefetch+Reload and Prefetch+Prefetch achieve 782 KB/s and 822 KB/s channel capacities, when using only one shared cache line between the sender and receiver, the largest-to-date single-line capacities for CPU cache covert channels. Further, in the side channel case, our attacks can monitor the access pattern of the victim on the same processor, with almost zero error rate. We show that they can be used to leak private information of real-world applications such as cryptographic keys. Finally, our attacks can be used in transient execution attacks in order to leak more secrets within the transient window than prior work. From the experimental results, our attacks allow leaking about 2 times as many secret bytes, compared to Flush+Reload, which is widely used in transient execution attacks.



## **16. A Context-Aware Approach for Textual Adversarial Attack through Probability Difference Guided Beam Search**

cs.CL

**SubmitDate**: 2022-08-17    [paper-pdf](http://arxiv.org/pdf/2208.08029v1)

**Authors**: Huijun Liu, Jie Yu, Shasha Li, Jun Ma, Bin Ji

**Abstracts**: Textual adversarial attacks expose the vulnerabilities of text classifiers and can be used to improve their robustness. Existing context-aware methods solely consider the gold label probability and use the greedy search when searching an attack path, often limiting the attack efficiency. To tackle these issues, we propose PDBS, a context-aware textual adversarial attack model using Probability Difference guided Beam Search. The probability difference is an overall consideration of all class label probabilities, and PDBS uses it to guide the selection of attack paths. In addition, PDBS uses the beam search to find a successful attack path, thus avoiding suffering from limited search space. Extensive experiments and human evaluation demonstrate that PDBS outperforms previous best models in a series of evaluation metrics, especially bringing up to a +19.5% attack success rate. Ablation studies and qualitative analyses further confirm the efficiency of PDBS.



## **17. FRL: Federated Rank Learning**

cs.LG

**SubmitDate**: 2022-08-16    [paper-pdf](http://arxiv.org/pdf/2110.04350v3)

**Authors**: Hamid Mozaffari, Virat Shejwalkar, Amir Houmansadr

**Abstracts**: Federated learning (FL) allows mutually untrusted clients to collaboratively train a common machine learning model without sharing their private/proprietary training data among each other. FL is unfortunately susceptible to poisoning by malicious clients who aim to hamper the accuracy of the commonly trained model through sending malicious model updates during FL's training process.   We argue that the key factor to the success of poisoning attacks against existing FL systems is the large space of model updates available to the clients, allowing malicious clients to search for the most poisonous model updates, e.g., by solving an optimization problem. To address this, we propose Federated Rank Learning (FRL). FRL reduces the space of client updates from model parameter updates (a continuous space of float numbers) in standard FL to the space of parameter rankings (a discrete space of integer values). To be able to train the global model using parameter ranks (instead of parameter weights), FRL leverage ideas from recent supermasks training mechanisms. Specifically, FRL clients rank the parameters of a randomly initialized neural network (provided by the server) based on their local training data. The FRL server uses a voting mechanism to aggregate the parameter rankings submitted by clients in each training epoch to generate the global ranking of the next training epoch.   Intuitively, our voting-based aggregation mechanism prevents poisoning clients from making significant adversarial modifications to the global model, as each client will have a single vote! We demonstrate the robustness of FRL to poisoning through analytical proofs and experimentation. We also show FRL's high communication efficiency. Our experiments demonstrate the superiority of FRL in real-world FL settings.



## **18. Adversarial Image Color Transformations in Explicit Color Filter Space**

cs.CV

Code is available at  https://github.com/ZhengyuZhao/ACE/tree/master/Journal_version. This work has  been submitted to the IEEE for possible publication. Copyright may be  transferred without notice, after which this version may no longer be  accessible

**SubmitDate**: 2022-08-16    [paper-pdf](http://arxiv.org/pdf/2011.06690v2)

**Authors**: Zhengyu Zhao, Zhuoran Liu, Martha Larson

**Abstracts**: Deep Neural Networks have been shown to be vulnerable to adversarial images. Conventional attacks strive for indistinguishable adversarial images with strictly restricted perturbations. Recently, researchers have moved to explore distinguishable yet non-suspicious adversarial images and demonstrated that color transformation attacks are effective. In this work, we propose Adversarial Color Filter (AdvCF), a novel color transformation attack that is optimized with gradient information in the parameter space of a simple color filter. In particular, our color filter space is explicitly specified so that we are able to provide a systematic analysis of model robustness against adversarial color transformations, from both the attack and defense perspectives. In contrast, existing color transformation attacks do not offer the opportunity for systematic analysis due to the lack of such an explicit space. We further conduct extensive comparisons between different color transformation attacks on both the success rate and image acceptability, through a user study. Additional results provide interesting new insights into model robustness against AdvCF in another three visual tasks. We also highlight the human-interpretability of AdvCF, which is promising in practical use scenarios, and show its superiority over the state-of-the-art human-interpretable color transformation attack on both the image acceptability and efficiency.



## **19. FedPerm: Private and Robust Federated Learning by Parameter Permutation**

cs.LG

**SubmitDate**: 2022-08-16    [paper-pdf](http://arxiv.org/pdf/2208.07922v1)

**Authors**: Hamid Mozaffari, Virendra J. Marathe, Dave Dice

**Abstracts**: Federated Learning (FL) is a distributed learning paradigm that enables mutually untrusting clients to collaboratively train a common machine learning model. Client data privacy is paramount in FL. At the same time, the model must be protected from poisoning attacks from adversarial clients. Existing solutions address these two problems in isolation. We present FedPerm, a new FL algorithm that addresses both these problems by combining a novel intra-model parameter shuffling technique that amplifies data privacy, with Private Information Retrieval (PIR) based techniques that permit cryptographic aggregation of clients' model updates. The combination of these techniques further helps the federation server constrain parameter updates from clients so as to curtail effects of model poisoning attacks by adversarial clients. We further present FedPerm's unique hyperparameters that can be used effectively to trade off computation overheads with model utility. Our empirical evaluation on the MNIST dataset demonstrates FedPerm's effectiveness over existing Differential Privacy (DP) enforcement solutions in FL.



## **20. Adversarial Relighting Against Face Recognition**

cs.CV

**SubmitDate**: 2022-08-16    [paper-pdf](http://arxiv.org/pdf/2108.07920v3)

**Authors**: Ruijun Gao, Qing Guo, Qian Zhang, Felix Juefei-Xu, Hongkai Yu, Wei Feng

**Abstracts**: Deep face recognition (FR) has achieved significantly high accuracy on several challenging datasets and fosters successful real-world applications, even showing high robustness to the illumination variation that is usually regarded as a main threat to the FR system. However, in the real world, illumination variation caused by diverse lighting conditions cannot be fully covered by the limited face dataset. In this paper, we study the threat of lighting against FR from a new angle, i.e., adversarial attack, and identify a new task, i.e., adversarial relighting. Given a face image, adversarial relighting aims to produce a naturally relighted counterpart while fooling the state-of-the-art deep FR methods. To this end, we first propose the physical model-based adversarial relighting attack (ARA) denoted as albedo-quotient-based adversarial relighting attack (AQ-ARA). It generates natural adversarial light under the physical lighting model and guidance of FR systems and synthesizes adversarially relighted face images. Moreover, we propose the auto-predictive adversarial relighting attack (AP-ARA) by training an adversarial relighting network (ARNet) to automatically predict the adversarial light in a one-step manner according to different input faces, allowing efficiency-sensitive applications. More importantly, we propose to transfer the above digital attacks to physical ARA (Phy-ARA) through a precise relighting device, making the estimated adversarial lighting condition reproducible in the real world. We validate our methods on three state-of-the-art deep FR methods, i.e., FaceNet, ArcFace, and CosFace, on two public datasets. The extensive and insightful results demonstrate our work can generate realistic adversarial relighted face images fooling FR easily, revealing the threat of specific light directions and strengths.



## **21. StratDef: a strategic defense against adversarial attacks in malware detection**

cs.LG

**SubmitDate**: 2022-08-16    [paper-pdf](http://arxiv.org/pdf/2202.07568v3)

**Authors**: Aqib Rashid, Jose Such

**Abstracts**: Over the years, most research towards defenses against adversarial attacks on machine learning models has been in the image recognition domain. The malware detection domain has received less attention despite its importance. Moreover, most work exploring these defenses has focused on several methods but with no strategy when applying them. In this paper, we introduce StratDef, which is a strategic defense system tailored for the malware detection domain based on a moving target defense approach. We overcome challenges related to the systematic construction, selection and strategic use of models to maximize adversarial robustness. StratDef dynamically and strategically chooses the best models to increase the uncertainty for the attacker, whilst minimizing critical aspects in the adversarial ML domain like attack transferability. We provide the first comprehensive evaluation of defenses against adversarial attacks on machine learning for malware detection, where our threat model explores different levels of threat, attacker knowledge, capabilities, and attack intensities. We show that StratDef performs better than other defenses even when facing the peak adversarial threat. We also show that, from the existing defenses, only a few adversarially-trained models provide substantially better protection than just using vanilla models but are still outperformed by StratDef.



## **22. A Physical-World Adversarial Attack for 3D Face Recognition**

cs.CV

7 pages, 5 figures, Submit to AAAI 2023

**SubmitDate**: 2022-08-16    [paper-pdf](http://arxiv.org/pdf/2205.13412v2)

**Authors**: Yanjie Li, Yiquan Li, Xuelong Dai, Songtao Guo, Bin Xiao

**Abstracts**: The 3D face recognition has long been considered secure for its resistance to current physical adversarial attacks, like adversarial patches. However, this paper shows that a 3D face recognition system can be easily attacked, leading to evading and impersonation attacks. We are the first to propose a physically realizable attack for the 3D face recognition system, named structured light imaging attack (SLIA), which exploits the weakness of structured-light-based 3D scanning devices. SLIA utilizes the projector in the structured light imaging system to create adversarial illuminations to contaminate the reconstructed point cloud. Firstly, we propose a 3D transform-invariant loss function (3D-TI) to generate adversarial perturbations that are more robust to head movements. Then we integrate the 3D imaging process into the attack optimization, which minimizes the total pixel shifting of fringe patterns. We realize both dodging and impersonation attacks on a real-world 3D face recognition system. Our methods need fewer modifications on projected patterns compared with Chamfer and Chamfer+kNN-based methods and achieve average attack success rates of 0.47 (impersonation) and 0.89 (dodging). This paper exposes the insecurity of present structured light imaging technology and sheds light on designing secure 3D face recognition authentication systems.



## **23. Near Optimal Adversarial Attack on UCB Bandits**

cs.LG

**SubmitDate**: 2022-08-16    [paper-pdf](http://arxiv.org/pdf/2008.09312v2)

**Authors**: Shiliang Zuo

**Abstracts**: We consider a stochastic multi-arm bandit problem where rewards are subject to adversarial corruption. We propose a novel attack strategy that manipulates a UCB principle into pulling some non-optimal target arm $T - o(T)$ times with a cumulative cost that scales as $\sqrt{\log T}$, where $T$ is the number of rounds. We also prove the first lower bound on the cumulative attack cost. Our lower bound matches our upper bound up to $\log \log T$ factors, showing our attack to be near optimal.



## **24. Defending Black-box Skeleton-based Human Activity Classifiers**

cs.CV

**SubmitDate**: 2022-08-16    [paper-pdf](http://arxiv.org/pdf/2203.04713v3)

**Authors**: He Wang, Yunfeng Diao, Zichang Tan, Guodong Guo

**Abstracts**: Skeletal motions have been heavily replied upon for human activity recognition (HAR). Recently, a universal vulnerability of skeleton-based HAR has been identified across a variety of classifiers and data, calling for mitigation. To this end, we propose the first black-box defense method for skeleton-based HAR to our best knowledge. Our method is featured by full Bayesian treatments of the clean data, the adversaries and the classifier, leading to (1) a new Bayesian Energy-based formulation of robust discriminative classifiers, (2) a new adversary sampling scheme based on natural motion manifolds, and (3) a new post-train Bayesian strategy for black-box defense. We name our framework Bayesian Energy-based Adversarial Training or BEAT. BEAT is straightforward but elegant, which turns vulnerable black-box classifiers into robust ones without sacrificing accuracy. It demonstrates surprising and universal effectiveness across a wide range of skeletal HAR classifiers and datasets, under various attacks.



## **25. CTI4AI: Threat Intelligence Generation and Sharing after Red Teaming AI Models**

cs.CR

**SubmitDate**: 2022-08-16    [paper-pdf](http://arxiv.org/pdf/2208.07476v1)

**Authors**: Chuyen Nguyen, Caleb Morgan, Sudip Mittal

**Abstracts**: As the practicality of Artificial Intelligence (AI) and Machine Learning (ML) based techniques grow, there is an ever increasing threat of adversarial attacks. There is a need to red team this ecosystem to identify system vulnerabilities, potential threats, characterize properties that will enhance system robustness, and encourage the creation of effective defenses. A secondary need is to share this AI security threat intelligence between different stakeholders like, model developers, users, and AI/ML security professionals. In this paper, we create and describe a prototype system CTI4AI, to overcome the need to methodically identify and share AI/ML specific vulnerabilities and threat intelligence.



## **26. MENLI: Robust Evaluation Metrics from Natural Language Inference**

cs.CL

**SubmitDate**: 2022-08-15    [paper-pdf](http://arxiv.org/pdf/2208.07316v1)

**Authors**: Yanran Chen, Steffen Eger

**Abstracts**: Recently proposed BERT-based evaluation metrics perform well on standard evaluation benchmarks but are vulnerable to adversarial attacks, e.g., relating to factuality errors. We argue that this stems (in part) from the fact that they are models of semantic similarity. In contrast, we develop evaluation metrics based on Natural Language Inference (NLI), which we deem a more appropriate modeling. We design a preference-based adversarial attack framework and show that our NLI based metrics are much more robust to the attacks than the recent BERT-based metrics. On standard benchmarks, our NLI based metrics outperform existing summarization metrics, but perform below SOTA MT metrics. However, when we combine existing metrics with our NLI metrics, we obtain both higher adversarial robustness (+20% to +30%) and higher quality metrics as measured on standard benchmarks (+5% to +25%).



## **27. Bounding Membership Inference**

cs.LG

**SubmitDate**: 2022-08-15    [paper-pdf](http://arxiv.org/pdf/2202.12232v3)

**Authors**: Anvith Thudi, Ilia Shumailov, Franziska Boenisch, Nicolas Papernot

**Abstracts**: Differential Privacy (DP) is the de facto standard for reasoning about the privacy guarantees of a training algorithm. Despite the empirical observation that DP reduces the vulnerability of models to existing membership inference (MI) attacks, a theoretical underpinning as to why this is the case is largely missing in the literature. In practice, this means that models need to be trained with DP guarantees that greatly decrease their accuracy. In this paper, we provide a tighter bound on the positive accuracy (i.e., attack precision) of any MI adversary when a training algorithm provides $\epsilon$-DP or $(\epsilon, \delta)$-DP. Our bound informs the design of a novel privacy amplification scheme, where an effective training set is sub-sampled from a larger set prior to the beginning of training, to greatly reduce the bound on MI accuracy. As a result, our scheme enables DP users to employ looser DP guarantees when training their model to limit the success of any MI adversary; this ensures that the model's accuracy is less impacted by the privacy guarantee. Finally, we discuss implications of our MI bound on the field of machine unlearning.



## **28. Man-in-the-Middle Attack against Object Detection Systems**

cs.RO

7 pages, 8 figures

**SubmitDate**: 2022-08-15    [paper-pdf](http://arxiv.org/pdf/2208.07174v1)

**Authors**: Han Wu, Sareh Rowlands, Johan Wahlstrom

**Abstracts**: Is deep learning secure for robots? As embedded systems have access to more powerful CPUs and GPUs, deep-learning-enabled object detection systems become pervasive in robotic applications. Meanwhile, prior research unveils that deep learning models are vulnerable to adversarial attacks. Does this put real-world robots at threat? Our research borrows the idea of the Main-in-the-Middle attack from Cryptography to attack an object detection system. Our experimental results prove that we can generate a strong Universal Adversarial Perturbation (UAP) within one minute and then use the perturbation to attack a detection system via the Man-in-the-Middle attack. Our findings raise a serious concern over the applications of deep learning models in safety-critical systems such as autonomous driving.



## **29. GUARD: Graph Universal Adversarial Defense**

cs.LG

Preprint. Code is publicly available at  https://github.com/EdisonLeeeee/GUARD

**SubmitDate**: 2022-08-15    [paper-pdf](http://arxiv.org/pdf/2204.09803v3)

**Authors**: Jintang Li, Jie Liao, Ruofan Wu, Liang Chen, Jiawang Dan, Changhua Meng, Zibin Zheng, Weiqiang Wang

**Abstracts**: Graph convolutional networks (GCNs) have shown to be vulnerable to small adversarial perturbations, which becomes a severe threat and largely limits their applications in security-critical scenarios. To mitigate such a threat, considerable research efforts have been devoted to increasing the robustness of GCNs against adversarial attacks. However, current approaches for defense are typically designed for the whole graph and consider the global performance, posing challenges in protecting important local nodes from stronger adversarial targeted attacks. In this work, we present a simple yet effective method, named Graph Universal Adversarial Defense (GUARD). Unlike previous works, GUARD protects each individual node from attacks with a universal defensive patch, which is generated once and can be applied to any node (node-agnostic) in a graph. Extensive experiments on four benchmark datasets demonstrate that our method significantly improves robustness for several established GCNs against multiple adversarial attacks and outperforms state-of-the-art defense methods by large margins. Our code is publicly available at https://github.com/EdisonLeeeee/GUARD.



## **30. A Multi-objective Memetic Algorithm for Auto Adversarial Attack Optimization Design**

cs.CV

**SubmitDate**: 2022-08-15    [paper-pdf](http://arxiv.org/pdf/2208.06984v1)

**Authors**: Jialiang Sun, Wen Yao, Tingsong Jiang, Xiaoqian Chen

**Abstracts**: The phenomenon of adversarial examples has been revealed in variant scenarios. Recent studies show that well-designed adversarial defense strategies can improve the robustness of deep learning models against adversarial examples. However, with the rapid development of defense technologies, it also tends to be more difficult to evaluate the robustness of the defensed model due to the weak performance of existing manually designed adversarial attacks. To address the challenge, given the defensed model, the efficient adversarial attack with less computational burden and lower robust accuracy is needed to be further exploited. Therefore, we propose a multi-objective memetic algorithm for auto adversarial attack optimization design, which realizes the automatical search for the near-optimal adversarial attack towards defensed models. Firstly, the more general mathematical model of auto adversarial attack optimization design is constructed, where the search space includes not only the attacker operations, magnitude, iteration number, and loss functions but also the connection ways of multiple adversarial attacks. In addition, we develop a multi-objective memetic algorithm combining NSGA-II and local search to solve the optimization problem. Finally, to decrease the evaluation cost during the search, we propose a representative data selection strategy based on the sorting of cross entropy loss values of each images output by models. Experiments on CIFAR10, CIFAR100, and ImageNet datasets show the effectiveness of our proposed method.



## **31. InvisibiliTee: Angle-agnostic Cloaking from Person-Tracking Systems with a Tee**

cs.CV

12 pages, 10 figures and the ICANN 2022 accpeted paper

**SubmitDate**: 2022-08-15    [paper-pdf](http://arxiv.org/pdf/2208.06962v1)

**Authors**: Yaxian Li, Bingqing Zhang, Guoping Zhao, Mingyu Zhang, Jiajun Liu, Ziwei Wang, Jirong Wen

**Abstracts**: After a survey for person-tracking system-induced privacy concerns, we propose a black-box adversarial attack method on state-of-the-art human detection models called InvisibiliTee. The method learns printable adversarial patterns for T-shirts that cloak wearers in the physical world in front of person-tracking systems. We design an angle-agnostic learning scheme which utilizes segmentation of the fashion dataset and a geometric warping process so the adversarial patterns generated are effective in fooling person detectors from all camera angles and for unseen black-box detection models. Empirical results in both digital and physical environments show that with the InvisibiliTee on, person-tracking systems' ability to detect the wearer drops significantly.



## **32. ARIEL: Adversarial Graph Contrastive Learning**

cs.LG

**SubmitDate**: 2022-08-15    [paper-pdf](http://arxiv.org/pdf/2208.06956v1)

**Authors**: Shengyu Feng, Baoyu Jing, Yada Zhu, Hanghang Tong

**Abstracts**: Contrastive learning is an effective unsupervised method in graph representation learning, and the key component of contrastive learning lies in the construction of positive and negative samples. Previous methods usually utilize the proximity of nodes in the graph as the principle. Recently, the data augmentation based contrastive learning method has advanced to show great power in the visual domain, and some works extended this method from images to graphs. However, unlike the data augmentation on images, the data augmentation on graphs is far less intuitive and much harder to provide high-quality contrastive samples, which leaves much space for improvement. In this work, by introducing an adversarial graph view for data augmentation, we propose a simple but effective method, Adversarial Graph Contrastive Learning (ARIEL), to extract informative contrastive samples within reasonable constraints. We develop a new technique called information regularization for stable training and use subgraph sampling for scalability. We generalize our method from node-level contrastive learning to the graph-level by treating each graph instance as a supernode. ARIEL consistently outperforms the current graph contrastive learning methods for both node-level and graph-level classification tasks on real-world datasets. We further demonstrate that ARIEL is more robust in face of adversarial attacks.



## **33. GNPassGAN: Improved Generative Adversarial Networks For Trawling Offline Password Guessing**

cs.CR

9 pages, 8 tables, 3 figures

**SubmitDate**: 2022-08-14    [paper-pdf](http://arxiv.org/pdf/2208.06943v1)

**Authors**: Fangyi Yu, Miguel Vargas Martin

**Abstracts**: The security of passwords depends on a thorough understanding of the strategies used by attackers. Unfortunately, real-world adversaries use pragmatic guessing tactics like dictionary attacks, which are difficult to simulate in password security research. Dictionary attacks must be carefully configured and modified to represent an actual threat. This approach, however, needs domain-specific knowledge and expertise that are difficult to duplicate. This paper reviews various deep learning-based password guessing approaches that do not require domain knowledge or assumptions about users' password structures and combinations. It also introduces GNPassGAN, a password guessing tool built on generative adversarial networks for trawling offline attacks. In comparison to the state-of-the-art PassGAN model, GNPassGAN is capable of guessing 88.03\% more passwords and generating 31.69\% fewer duplicates.



## **34. Gradient Mask: Lateral Inhibition Mechanism Improves Performance in Artificial Neural Networks**

cs.CV

**SubmitDate**: 2022-08-14    [paper-pdf](http://arxiv.org/pdf/2208.06918v1)

**Authors**: Lei Jiang, Yongqing Liu, Shihai Xiao, Yansong Chua

**Abstracts**: Lateral inhibitory connections have been observed in the cortex of the biological brain, and has been extensively studied in terms of its role in cognitive functions. However, in the vanilla version of backpropagation in deep learning, all gradients (which can be understood to comprise of both signal and noise gradients) flow through the network during weight updates. This may lead to overfitting. In this work, inspired by biological lateral inhibition, we propose Gradient Mask, which effectively filters out noise gradients in the process of backpropagation. This allows the learned feature information to be more intensively stored in the network while filtering out noisy or unimportant features. Furthermore, we demonstrate analytically how lateral inhibition in artificial neural networks improves the quality of propagated gradients. A new criterion for gradient quality is proposed which can be used as a measure during training of various convolutional neural networks (CNNs). Finally, we conduct several different experiments to study how Gradient Mask improves the performance of the network both quantitatively and qualitatively. Quantitatively, accuracy in the original CNN architecture, accuracy after pruning, and accuracy after adversarial attacks have shown improvements. Qualitatively, the CNN trained using Gradient Mask has developed saliency maps that focus primarily on the object of interest, which is useful for data augmentation and network interpretability.



## **35. IPvSeeYou: Exploiting Leaked Identifiers in IPv6 for Street-Level Geolocation**

cs.NI

Accepted to S&P '23

**SubmitDate**: 2022-08-14    [paper-pdf](http://arxiv.org/pdf/2208.06767v1)

**Authors**: Erik Rye, Robert Beverly

**Abstracts**: We present IPvSeeYou, a privacy attack that permits a remote and unprivileged adversary to physically geolocate many residential IPv6 hosts and networks with street-level precision. The crux of our method involves: 1) remotely discovering wide area (WAN) hardware MAC addresses from home routers; 2) correlating these MAC addresses with their WiFi BSSID counterparts of known location; and 3) extending coverage by associating devices connected to a common penultimate provider router.   We first obtain a large corpus of MACs embedded in IPv6 addresses via high-speed network probing. These MAC addresses are effectively leaked up the protocol stack and largely represent WAN interfaces of residential routers, many of which are all-in-one devices that also provide WiFi. We develop a technique to statistically infer the mapping between a router's WAN and WiFi MAC addresses across manufacturers and devices, and mount a large-scale data fusion attack that correlates WAN MACs with WiFi BSSIDs available in wardriving (geolocation) databases. Using these correlations, we geolocate the IPv6 prefixes of $>$12M routers in the wild across 146 countries and territories. Selected validation confirms a median geolocation error of 39 meters. We then exploit technology and deployment constraints to extend the attack to a larger set of IPv6 residential routers by clustering and associating devices with a common penultimate provider router. While we responsibly disclosed our results to several manufacturers and providers, the ossified ecosystem of deployed residential cable and DSL routers suggests that our attack will remain a privacy threat into the foreseeable future.



## **36. Adversarial Texture for Fooling Person Detectors in the Physical World**

cs.CV

Accepted by CVPR 2022

**SubmitDate**: 2022-08-13    [paper-pdf](http://arxiv.org/pdf/2203.03373v4)

**Authors**: Zhanhao Hu, Siyuan Huang, Xiaopei Zhu, Fuchun Sun, Bo Zhang, Xiaolin Hu

**Abstracts**: Nowadays, cameras equipped with AI systems can capture and analyze images to detect people automatically. However, the AI system can make mistakes when receiving deliberately designed patterns in the real world, i.e., physical adversarial examples. Prior works have shown that it is possible to print adversarial patches on clothes to evade DNN-based person detectors. However, these adversarial examples could have catastrophic drops in the attack success rate when the viewing angle (i.e., the camera's angle towards the object) changes. To perform a multi-angle attack, we propose Adversarial Texture (AdvTexture). AdvTexture can cover clothes with arbitrary shapes so that people wearing such clothes can hide from person detectors from different viewing angles. We propose a generative method, named Toroidal-Cropping-based Expandable Generative Attack (TC-EGA), to craft AdvTexture with repetitive structures. We printed several pieces of cloth with AdvTexure and then made T-shirts, skirts, and dresses in the physical world. Experiments showed that these clothes could fool person detectors in the physical world.



## **37. An Analytic Framework for Robust Training of Artificial Neural Networks**

cs.LG

**SubmitDate**: 2022-08-13    [paper-pdf](http://arxiv.org/pdf/2205.13502v2)

**Authors**: Ramin Barati, Reza Safabakhsh, Mohammad Rahmati

**Abstracts**: The reliability of a learning model is key to the successful deployment of machine learning in various industries. Creating a robust model, particularly one unaffected by adversarial attacks, requires a comprehensive understanding of the adversarial examples phenomenon. However, it is difficult to describe the phenomenon due to the complicated nature of the problems in machine learning. Consequently, many studies investigate the phenomenon by proposing a simplified model of how adversarial examples occur and validate it by predicting some aspect of the phenomenon. While these studies cover many different characteristics of the adversarial examples, they have not reached a holistic approach to the geometric and analytic modeling of the phenomenon. This paper propose a formal framework to study the phenomenon in learning theory and make use of complex analysis and holomorphicity to offer a robust learning rule for artificial neural networks. With the help of complex analysis, we can effortlessly move between geometric and analytic perspectives of the phenomenon and offer further insights on the phenomenon by revealing its connection with harmonic functions. Using our model, we can explain some of the most intriguing characteristics of adversarial examples, including transferability of adversarial examples, and pave the way for novel approaches to mitigate the effects of the phenomenon.



## **38. Revisiting Adversarial Attacks on Graph Neural Networks for Graph Classification**

cs.SI

**SubmitDate**: 2022-08-13    [paper-pdf](http://arxiv.org/pdf/2208.06651v1)

**Authors**: Beini Xie, Heng Chang, Xin Wang, Tian Bian, Shiji Zhou, Daixin Wang, Zhiqiang Zhang, Wenwu Zhu

**Abstracts**: Graph neural networks (GNNs) have achieved tremendous success in the task of graph classification and diverse downstream real-world applications. Despite their success, existing approaches are either limited to structure attacks or restricted to local information. This calls for a more general attack framework on graph classification, which faces significant challenges due to the complexity of generating local-node-level adversarial examples using the global-graph-level information. To address this "global-to-local" problem, we present a general framework CAMA to generate adversarial examples by manipulating graph structure and node features in a hierarchical style. Specifically, we make use of Graph Class Activation Mapping and its variant to produce node-level importance corresponding to the graph classification task. Then through a heuristic design of algorithms, we can perform both feature and structure attacks under unnoticeable perturbation budgets with the help of both node-level and subgraph-level importance. Experiments towards attacking four state-of-the-art graph classification models on six real-world benchmarks verify the flexibility and effectiveness of our framework.



## **39. Poison Ink: Robust and Invisible Backdoor Attack**

cs.CR

IEEE Transactions on Image Processing (TIP)

**SubmitDate**: 2022-08-13    [paper-pdf](http://arxiv.org/pdf/2108.02488v3)

**Authors**: Jie Zhang, Dongdong Chen, Qidong Huang, Jing Liao, Weiming Zhang, Huamin Feng, Gang Hua, Nenghai Yu

**Abstracts**: Recent research shows deep neural networks are vulnerable to different types of attacks, such as adversarial attack, data poisoning attack and backdoor attack. Among them, backdoor attack is the most cunning one and can occur in almost every stage of deep learning pipeline. Therefore, backdoor attack has attracted lots of interests from both academia and industry. However, most existing backdoor attack methods are either visible or fragile to some effortless pre-processing such as common data transformations. To address these limitations, we propose a robust and invisible backdoor attack called "Poison Ink". Concretely, we first leverage the image structures as target poisoning areas, and fill them with poison ink (information) to generate the trigger pattern. As the image structure can keep its semantic meaning during the data transformation, such trigger pattern is inherently robust to data transformations. Then we leverage a deep injection network to embed such trigger pattern into the cover image to achieve stealthiness. Compared to existing popular backdoor attack methods, Poison Ink outperforms both in stealthiness and robustness. Through extensive experiments, we demonstrate Poison Ink is not only general to different datasets and network architectures, but also flexible for different attack scenarios. Besides, it also has very strong resistance against many state-of-the-art defense techniques.



## **40. MaskBlock: Transferable Adversarial Examples with Bayes Approach**

cs.LG

Under Review

**SubmitDate**: 2022-08-13    [paper-pdf](http://arxiv.org/pdf/2208.06538v1)

**Authors**: Mingyuan Fan, Cen Chen, Ximeng Liu, Wenzhong Guo

**Abstracts**: The transferability of adversarial examples (AEs) across diverse models is of critical importance for black-box adversarial attacks, where attackers cannot access the information about black-box models. However, crafted AEs always present poor transferability. In this paper, by regarding the transferability of AEs as generalization ability of the model, we reveal that vanilla black-box attacks craft AEs via solving a maximum likelihood estimation (MLE) problem. For MLE, the results probably are model-specific local optimum when available data is small, i.e., limiting the transferability of AEs. By contrast, we re-formulate crafting transferable AEs as the maximizing a posteriori probability estimation problem, which is an effective approach to boost the generalization of results with limited available data. Because Bayes posterior inference is commonly intractable, a simple yet effective method called MaskBlock is developed to approximately estimate. Moreover, we show that the formulated framework is a generalization version for various attack methods. Extensive experiments illustrate MaskBlock can significantly improve the transferability of crafted adversarial examples by up to about 20%.



## **41. Hide and Seek: on the Stealthiness of Attacks against Deep Learning Systems**

cs.CR

To appear in European Symposium on Research in Computer Security  (ESORICS) 2022

**SubmitDate**: 2022-08-12    [paper-pdf](http://arxiv.org/pdf/2205.15944v2)

**Authors**: Zeyan Liu, Fengjun Li, Jingqiang Lin, Zhu Li, Bo Luo

**Abstracts**: With the growing popularity of artificial intelligence and machine learning, a wide spectrum of attacks against deep learning models have been proposed in the literature. Both the evasion attacks and the poisoning attacks attempt to utilize adversarially altered samples to fool the victim model to misclassify the adversarial sample. While such attacks claim to be or are expected to be stealthy, i.e., imperceptible to human eyes, such claims are rarely evaluated. In this paper, we present the first large-scale study on the stealthiness of adversarial samples used in the attacks against deep learning. We have implemented 20 representative adversarial ML attacks on six popular benchmarking datasets. We evaluate the stealthiness of the attack samples using two complementary approaches: (1) a numerical study that adopts 24 metrics for image similarity or quality assessment; and (2) a user study of 3 sets of questionnaires that has collected 20,000+ annotations from 1,000+ responses. Our results show that the majority of the existing attacks introduce nonnegligible perturbations that are not stealthy to human eyes. We further analyze the factors that contribute to attack stealthiness. We further examine the correlation between the numerical analysis and the user studies, and demonstrate that some image quality metrics may provide useful guidance in attack designs, while there is still a significant gap between assessed image quality and visual stealthiness of attacks.



## **42. PRIVEE: A Visual Analytic Workflow for Proactive Privacy Risk Inspection of Open Data**

cs.CR

Accepted for IEEE Symposium on Visualization in Cyber Security, 2022

**SubmitDate**: 2022-08-12    [paper-pdf](http://arxiv.org/pdf/2208.06481v1)

**Authors**: Kaustav Bhattacharjee, Akm Islam, Jaideep Vaidya, Aritra Dasgupta

**Abstracts**: Open data sets that contain personal information are susceptible to adversarial attacks even when anonymized. By performing low-cost joins on multiple datasets with shared attributes, malicious users of open data portals might get access to information that violates individuals' privacy. However, open data sets are primarily published using a release-and-forget model, whereby data owners and custodians have little to no cognizance of these privacy risks. We address this critical gap by developing a visual analytic solution that enables data defenders to gain awareness about the disclosure risks in local, joinable data neighborhoods. The solution is derived through a design study with data privacy researchers, where we initially play the role of a red team and engage in an ethical data hacking exercise based on privacy attack scenarios. We use this problem and domain characterization to develop a set of visual analytic interventions as a defense mechanism and realize them in PRIVEE, a visual risk inspection workflow that acts as a proactive monitor for data defenders. PRIVEE uses a combination of risk scores and associated interactive visualizations to let data defenders explore vulnerable joins and interpret risks at multiple levels of data granularity. We demonstrate how PRIVEE can help emulate the attack strategies and diagnose disclosure risks through two case studies with data privacy experts.



## **43. UniNet: A Unified Scene Understanding Network and Exploring Multi-Task Relationships through the Lens of Adversarial Attacks**

cs.CV

Accepted at DeepMTL workshop, ICCV 2021

**SubmitDate**: 2022-08-12    [paper-pdf](http://arxiv.org/pdf/2108.04584v2)

**Authors**: Naresh Kumar Gurulingan, Elahe Arani, Bahram Zonooz

**Abstracts**: Scene understanding is crucial for autonomous systems which intend to operate in the real world. Single task vision networks extract information only based on some aspects of the scene. In multi-task learning (MTL), on the other hand, these single tasks are jointly learned, thereby providing an opportunity for tasks to share information and obtain a more comprehensive understanding. To this end, we develop UniNet, a unified scene understanding network that accurately and efficiently infers vital vision tasks including object detection, semantic segmentation, instance segmentation, monocular depth estimation, and monocular instance depth prediction. As these tasks look at different semantic and geometric information, they can either complement or conflict with each other. Therefore, understanding inter-task relationships can provide useful cues to enable complementary information sharing. We evaluate the task relationships in UniNet through the lens of adversarial attacks based on the notion that they can exploit learned biases and task interactions in the neural network. Extensive experiments on the Cityscapes dataset, using untargeted and targeted attacks reveal that semantic tasks strongly interact amongst themselves, and the same holds for geometric tasks. Additionally, we show that the relationship between semantic and geometric tasks is asymmetric and their interaction becomes weaker as we move towards higher-level representations.



## **44. Unifying Gradients to Improve Real-world Robustness for Deep Networks**

stat.ML

**SubmitDate**: 2022-08-12    [paper-pdf](http://arxiv.org/pdf/2208.06228v1)

**Authors**: Yingwen Wu, Sizhe Chen, Kun Fang, Xiaolin Huang

**Abstracts**: The wide application of deep neural networks (DNNs) demands an increasing amount of attention to their real-world robustness, i.e., whether a DNN resists black-box adversarial attacks, among them score-based query attacks (SQAs) are the most threatening ones because of their practicalities and effectiveness: the attackers only need dozens of queries on model outputs to seriously hurt a victim network. Defending against SQAs requires a slight but artful variation of outputs due to the service purpose for users, who share the same output information with attackers. In this paper, we propose a real-world defense, called Unifying Gradients (UniG), to unify gradients of different data so that attackers could only probe a much weaker attack direction that is similar for different samples. Since such universal attack perturbations have been validated as less aggressive than the input-specific perturbations, UniG protects real-world DNNs by indicating attackers a twisted and less informative attack direction. To enhance UniG's practical significance in real-world applications, we implement it as a Hadamard product module that is computationally-efficient and readily plugged into any model. According to extensive experiments on 5 SQAs and 4 defense baselines, UniG significantly improves real-world robustness without hurting clean accuracy on CIFAR10 and ImageNet. For instance, UniG maintains a CIFAR-10 model of 77.80% accuracy under 2500-query Square attack while the state-of-the-art adversarially-trained model only has 67.34% on CIFAR10. Simultaneously, UniG greatly surpasses all compared baselines in clean accuracy and the modification degree of outputs. The code would be released.



## **45. Scale-free Photo-realistic Adversarial Pattern Attack**

cs.CV

**SubmitDate**: 2022-08-12    [paper-pdf](http://arxiv.org/pdf/2208.06222v1)

**Authors**: Xiangbo Gao, Weicheng Xie, Minmin Liu, Cheng Luo, Qinliang Lin, Linlin Shen, Keerthy Kusumam, Siyang Song

**Abstracts**: Traditional pixel-wise image attack algorithms suffer from poor robustness to defense algorithms, i.e., the attack strength degrades dramatically when defense algorithms are applied. Although Generative Adversarial Networks (GAN) can partially address this problem by synthesizing a more semantically meaningful texture pattern, the main limitation is that existing generators can only generate images of a specific scale. In this paper, we propose a scale-free generation-based attack algorithm that synthesizes semantically meaningful adversarial patterns globally to images with arbitrary scales. Our generative attack approach consistently outperforms the state-of-the-art methods on a wide range of attack settings, i.e. the proposed approach largely degraded the performance of various image classification, object detection, and instance segmentation algorithms under different advanced defense methods.



## **46. A Knowledge Distillation-Based Backdoor Attack in Federated Learning**

cs.LG

**SubmitDate**: 2022-08-12    [paper-pdf](http://arxiv.org/pdf/2208.06176v1)

**Authors**: Yifan Wang, Wei Fan, Keke Yang, Naji Alhusaini, Jing Li

**Abstracts**: Federated Learning (FL) is a novel framework of decentralized machine learning. Due to the decentralized feature of FL, it is vulnerable to adversarial attacks in the training procedure, e.g. , backdoor attacks. A backdoor attack aims to inject a backdoor into the machine learning model such that the model will make arbitrarily incorrect behavior on the test sample with some specific backdoor trigger. Even though a range of backdoor attack methods of FL has been introduced, there are also methods defending against them. Many of the defending methods utilize the abnormal characteristics of the models with backdoor or the difference between the models with backdoor and the regular models. To bypass these defenses, we need to reduce the difference and the abnormal characteristics. We find a source of such abnormality is that backdoor attack would directly flip the label of data when poisoning the data. However, current studies of the backdoor attack in FL are not mainly focus on reducing the difference between the models with backdoor and the regular models. In this paper, we propose Adversarial Knowledge Distillation(ADVKD), a method combine knowledge distillation with backdoor attack in FL. With knowledge distillation, we can reduce the abnormal characteristics in model result from the label flipping, thus the model can bypass the defenses. Compared to current methods, we show that ADVKD can not only reach a higher attack success rate, but also successfully bypass the defenses when other methods fails. To further explore the performance of ADVKD, we test how the parameters affect the performance of ADVKD under different scenarios. According to the experiment result, we summarize how to adjust the parameter for better performance under different scenarios. We also use several methods to visualize the effect of different attack and explain the effectiveness of ADVKD.



## **47. A Survey of MulVAL Extensions and Their Attack Scenarios Coverage**

cs.CR

**SubmitDate**: 2022-08-11    [paper-pdf](http://arxiv.org/pdf/2208.05750v1)

**Authors**: David Tayouri, Nick Baum, Asaf Shabtai, Rami Puzis

**Abstracts**: Organizations employ various adversary models in order to assess the risk and potential impact of attacks on their networks. Attack graphs represent vulnerabilities and actions an attacker can take to identify and compromise an organization's assets. Attack graphs facilitate both visual presentation and algorithmic analysis of attack scenarios in the form of attack paths. MulVAL is a generic open-source framework for constructing logical attack graphs, which has been widely used by researchers and practitioners and extended by them with additional attack scenarios. This paper surveys all of the existing MulVAL extensions, and maps all MulVAL interaction rules to MITRE ATT&CK Techniques to estimate their attack scenarios coverage. This survey aligns current MulVAL extensions along unified ontological concepts and highlights the existing gaps. It paves the way for methodical improvement of MulVAL and the comprehensive modeling of the entire landscape of adversarial behaviors captured in MITRE ATT&CK.



## **48. Diverse Generative Adversarial Perturbations on Attention Space for Transferable Adversarial Attacks**

cs.CV

ICIP 2022

**SubmitDate**: 2022-08-11    [paper-pdf](http://arxiv.org/pdf/2208.05650v1)

**Authors**: Woo Jae Kim, Seunghoon Hong, Sung-Eui Yoon

**Abstracts**: Adversarial attacks with improved transferability - the ability of an adversarial example crafted on a known model to also fool unknown models - have recently received much attention due to their practicality. Nevertheless, existing transferable attacks craft perturbations in a deterministic manner and often fail to fully explore the loss surface, thus falling into a poor local optimum and suffering from low transferability. To solve this problem, we propose Attentive-Diversity Attack (ADA), which disrupts diverse salient features in a stochastic manner to improve transferability. Primarily, we perturb the image attention to disrupt universal features shared by different models. Then, to effectively avoid poor local optima, we disrupt these features in a stochastic manner and explore the search space of transferable perturbations more exhaustively. More specifically, we use a generator to produce adversarial perturbations that each disturbs features in different ways depending on an input latent code. Extensive experimental evaluations demonstrate the effectiveness of our method, outperforming the transferability of state-of-the-art methods. Codes are available at https://github.com/wkim97/ADA.



## **49. Controlled Quantum Teleportation in the Presence of an Adversary**

quant-ph

**SubmitDate**: 2022-08-10    [paper-pdf](http://arxiv.org/pdf/2208.05554v1)

**Authors**: Sayan Gangopadhyay, Tiejun Wang, Atefeh Mashatan, Shohini Ghose

**Abstracts**: We present a device independent analysis of controlled quantum teleportation where the receiver is not trusted. We show that the notion of genuine tripartite nonlocality allows us to certify control power in such a scenario. By considering a specific adversarial attack strategy on a device characterized by depolarizing noise, we find that control power is a monotonically increasing function of genuine tripartite nonlocality. These results are relevant for building practical quantum communication networks and also shed light on the role of nonlocality in multipartite quantum information processing.



## **50. Pikachu: Securing PoS Blockchains from Long-Range Attacks by Checkpointing into Bitcoin PoW using Taproot**

cs.CR

To appear at ConsensusDay 22 (ACM CCS 2022 Workshop)

**SubmitDate**: 2022-08-10    [paper-pdf](http://arxiv.org/pdf/2208.05408v1)

**Authors**: Sarah Azouvi, Marko Vukolić

**Abstracts**: Blockchain systems based on a reusable resource, such as proof-of-stake (PoS), provide weaker security guarantees than those based on proof-of-work. Specifically, they are vulnerable to long-range attacks, where an adversary can corrupt prior participants in order to rewrite the full history of the chain. To prevent this attack on a PoS chain, we propose a protocol that checkpoints the state of the PoS chain to a proof-of-work blockchain such as Bitcoin. Our checkpointing protocol hence does not rely on any central authority. Our work uses Schnorr signatures and leverages Bitcoin recent Taproot upgrade, allowing us to create a checkpointing transaction of constant size. We argue for the security of our protocol and present an open-source implementation that was tested on the Bitcoin testnet.



