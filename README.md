# Latest Adversarial Attack Papers
**update at 2022-10-08 06:31:24**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Truth Serum: Poisoning Machine Learning Models to Reveal Their Secrets**

cs.CR

ACM CCS 2022

**SubmitDate**: 2022-10-06    [abs](http://arxiv.org/abs/2204.00032v2) [paper-pdf](http://arxiv.org/pdf/2204.00032v2)

**Authors**: Florian Tramèr, Reza Shokri, Ayrton San Joaquin, Hoang Le, Matthew Jagielski, Sanghyun Hong, Nicholas Carlini

**Abstract**: We introduce a new class of attacks on machine learning models. We show that an adversary who can poison a training dataset can cause models trained on this dataset to leak significant private details of training points belonging to other parties. Our active inference attacks connect two independent lines of work targeting the integrity and privacy of machine learning training data.   Our attacks are effective across membership inference, attribute inference, and data extraction. For example, our targeted attacks can poison <0.1% of the training dataset to boost the performance of inference attacks by 1 to 2 orders of magnitude. Further, an adversary who controls a significant fraction of the training data (e.g., 50%) can launch untargeted attacks that enable 8x more precise inference on all other users' otherwise-private data points.   Our results cast doubts on the relevance of cryptographic privacy guarantees in multiparty computation protocols for machine learning, if parties can arbitrarily select their share of training data.



## **2. EvilScreen Attack: Smart TV Hijacking via Multi-channel Remote Control Mimicry**

cs.CR

**SubmitDate**: 2022-10-06    [abs](http://arxiv.org/abs/2210.03014v1) [paper-pdf](http://arxiv.org/pdf/2210.03014v1)

**Authors**: Yiwei Zhang, Siqi Ma, Tiancheng Chen, Juanru Li, Robert H. Deng, Elisa Bertino

**Abstract**: Modern smart TVs often communicate with their remote controls (including those smart phone simulated ones) using multiple wireless channels (e.g., Infrared, Bluetooth, and Wi-Fi). However, this multi-channel remote control communication introduces a new attack surface. An inherent security flaw is that remote controls of most smart TVs are designed to work in a benign environment rather than an adversarial one, and thus wireless communications between a smart TV and its remote controls are not strongly protected. Attackers could leverage such flaw to abuse the remote control communication and compromise smart TV systems. In this paper, we propose EvilScreen, a novel attack that exploits ill-protected remote control communications to access protected resources of a smart TV or even control the screen. EvilScreen exploits a multi-channel remote control mimicry vulnerability present in today smart TVs. Unlike other attacks, which compromise the TV system by exploiting code vulnerabilities or malicious third-party apps, EvilScreen directly reuses commands of different remote controls, combines them together to circumvent deployed authentication and isolation policies, and finally accesses or controls TV resources remotely. We evaluated eight mainstream smart TVs and found that they are all vulnerable to EvilScreen attacks, including a Samsung product adopting the ISO/IEC security specification.



## **3. CANIFE: Crafting Canaries for Empirical Privacy Measurement in Federated Learning**

cs.LG

**SubmitDate**: 2022-10-06    [abs](http://arxiv.org/abs/2210.02912v1) [paper-pdf](http://arxiv.org/pdf/2210.02912v1)

**Authors**: Samuel Maddock, Alexandre Sablayrolles, Pierre Stock

**Abstract**: Federated Learning (FL) is a setting for training machine learning models in distributed environments where the clients do not share their raw data but instead send model updates to a server. However, model updates can be subject to attacks and leak private information. Differential Privacy (DP) is a leading mitigation strategy which involves adding noise to clipped model updates, trading off performance for strong theoretical privacy guarantees. Previous work has shown that the threat model of DP is conservative and that the obtained guarantees may be vacuous or may not directly translate to information leakage in practice. In this paper, we aim to achieve a tighter measurement of the model exposure by considering a realistic threat model. We propose a novel method, CANIFE, that uses canaries - carefully crafted samples by a strong adversary to evaluate the empirical privacy of a training round. We apply this attack to vision models trained on CIFAR-10 and CelebA and to language models trained on Sent140 and Shakespeare. In particular, in realistic FL scenarios, we demonstrate that the empirical epsilon obtained with CANIFE is 2-7x lower than the theoretical bound.



## **4. How Far Are We from Real Synonym Substitution Attacks?**

cs.CL

Preprint, work in progress

**SubmitDate**: 2022-10-06    [abs](http://arxiv.org/abs/2210.02844v1) [paper-pdf](http://arxiv.org/pdf/2210.02844v1)

**Authors**: Cheng-Han Chiang, Hung-yi Lee

**Abstract**: In this paper, we explore the following question: how far are we from real synonym substitution attacks (SSAs). We approach this question by examining how SSAs replace words in the original sentence and show that there are still unresolved obstacles that make current SSAs generate invalid adversarial samples. We reveal that four widely used word substitution methods generate a large fraction of invalid substitution words that are ungrammatical or do not preserve the original sentence's semantics. Next, we show that the semantic and grammatical constraints used in SSAs for detecting invalid word replacements are highly insufficient in detecting invalid adversarial samples. Our work is an important stepping stone to constructing better SSAs in the future.



## **5. Deep Reinforcement Learning based Evasion Generative Adversarial Network for Botnet Detection**

cs.CR

**SubmitDate**: 2022-10-06    [abs](http://arxiv.org/abs/2210.02840v1) [paper-pdf](http://arxiv.org/pdf/2210.02840v1)

**Authors**: Rizwan Hamid Randhawa, Nauman Aslam, Mohammad Alauthman, Muhammad Khalid, Husnain Rafiq

**Abstract**: Botnet detectors based on machine learning are potential targets for adversarial evasion attacks. Several research works employ adversarial training with samples generated from generative adversarial nets (GANs) to make the botnet detectors adept at recognising adversarial evasions. However, the synthetic evasions may not follow the original semantics of the input samples. This paper proposes a novel GAN model leveraged with deep reinforcement learning (DRL) to explore semantic aware samples and simultaneously harden its detection. A DRL agent is used to attack the discriminator of the GAN that acts as a botnet detector. The discriminator is trained on the crafted perturbations by the agent during the GAN training, which helps the GAN generator converge earlier than the case without DRL. We name this model RELEVAGAN, i.e. ["relive a GAN" or deep REinforcement Learning-based Evasion Generative Adversarial Network] because, with the help of DRL, it minimises the GAN's job by letting its generator explore the evasion samples within the semantic limits. During the GAN training, the attacks are conducted to adjust the discriminator weights for learning crafted perturbations by the agent. RELEVAGAN does not require adversarial training for the ML classifiers since it can act as an adversarial semantic-aware botnet detection model. Code will be available at https://github.com/rhr407/RELEVAGAN.



## **6. Adversarial Attack and Defense on Graph Data: A Survey**

cs.CR

Published on IEEE TKDE. For more open-source and up-to-date  information, please check our Github repository:  https://github.com/YingtongDou/graph-adversarial-learning-literature

**SubmitDate**: 2022-10-06    [abs](http://arxiv.org/abs/1812.10528v4) [paper-pdf](http://arxiv.org/pdf/1812.10528v4)

**Authors**: Lichao Sun, Yingtong Dou, Carl Yang, Ji Wang, Yixin Liu, Philip S. Yu, Lifang He, Bo Li

**Abstract**: Deep neural networks (DNNs) have been widely applied to various applications, including image classification, text generation, audio recognition, and graph data analysis. However, recent studies have shown that DNNs are vulnerable to adversarial attacks. Though there are several works about adversarial attack and defense strategies on domains such as images and natural language processing, it is still difficult to directly transfer the learned knowledge to graph data due to its representation structure. Given the importance of graph analysis, an increasing number of studies over the past few years have attempted to analyze the robustness of machine learning models on graph data. Nevertheless, existing research considering adversarial behaviors on graph data often focuses on specific types of attacks with certain assumptions. In addition, each work proposes its own mathematical formulation, which makes the comparison among different methods difficult. Therefore, this review is intended to provide an overall landscape of more than 100 papers on adversarial attack and defense strategies for graph data, and establish a unified formulation encompassing most graph adversarial learning models. Moreover, we also compare different graph attacks and defenses along with their contributions and limitations, as well as summarize the evaluation metrics, datasets and future trends. We hope this survey can help fill the gap in the literature and facilitate further development of this promising new field.



## **7. Who's Afraid of Adversarial Transferability?**

cs.LG

**SubmitDate**: 2022-10-06    [abs](http://arxiv.org/abs/2105.00433v3) [paper-pdf](http://arxiv.org/pdf/2105.00433v3)

**Authors**: Ziv Katzir, Yuval Elovici

**Abstract**: Adversarial transferability, namely the ability of adversarial perturbations to simultaneously fool multiple learning models, has long been the "big bad wolf" of adversarial machine learning. Successful transferability-based attacks requiring no prior knowledge of the attacked model's parameters or training data have been demonstrated numerous times in the past, implying that machine learning models pose an inherent security threat to real-life systems. However, all of the research performed in this area regarded transferability as a probabilistic property and attempted to estimate the percentage of adversarial examples that are likely to mislead a target model given some predefined evaluation set. As a result, those studies ignored the fact that real-life adversaries are often highly sensitive to the cost of a failed attack. We argue that overlooking this sensitivity has led to an exaggerated perception of the transferability threat, when in fact real-life transferability-based attacks are quite unlikely. By combining theoretical reasoning with a series of empirical results, we show that it is practically impossible to predict whether a given adversarial example is transferable to a specific target model in a black-box setting, hence questioning the validity of adversarial transferability as a real-life attack tool for adversaries that are sensitive to the cost of a failed attack.



## **8. On Optimal Learning Under Targeted Data Poisoning**

cs.LG

**SubmitDate**: 2022-10-06    [abs](http://arxiv.org/abs/2210.02713v1) [paper-pdf](http://arxiv.org/pdf/2210.02713v1)

**Authors**: Steve Hanneke, Amin Karbasi, Mohammad Mahmoody, Idan Mehalel, Shay Moran

**Abstract**: Consider the task of learning a hypothesis class $\mathcal{H}$ in the presence of an adversary that can replace up to an $\eta$ fraction of the examples in the training set with arbitrary adversarial examples. The adversary aims to fail the learner on a particular target test point $x$ which is known to the adversary but not to the learner. In this work we aim to characterize the smallest achievable error $\epsilon=\epsilon(\eta)$ by the learner in the presence of such an adversary in both realizable and agnostic settings. We fully achieve this in the realizable setting, proving that $\epsilon=\Theta(\mathtt{VC}(\mathcal{H})\cdot \eta)$, where $\mathtt{VC}(\mathcal{H})$ is the VC dimension of $\mathcal{H}$. Remarkably, we show that the upper bound can be attained by a deterministic learner. In the agnostic setting we reveal a more elaborate landscape: we devise a deterministic learner with a multiplicative regret guarantee of $\epsilon \leq C\cdot\mathtt{OPT} + O(\mathtt{VC}(\mathcal{H})\cdot \eta)$, where $C > 1$ is a universal numerical constant. We complement this by showing that for any deterministic learner there is an attack which worsens its error to at least $2\cdot \mathtt{OPT}$. This implies that a multiplicative deterioration in the regret is unavoidable in this case. Finally, the algorithms we develop for achieving the optimal rates are inherently improper. Nevertheless, we show that for a variety of natural concept classes, such as linear classifiers, it is possible to retain the dependence $\epsilon=\Theta_{\mathcal{H}}(\eta)$ by a proper algorithm in the realizable setting. Here $\Theta_{\mathcal{H}}$ conceals a polynomial dependence on $\mathtt{VC}(\mathcal{H})$.



## **9. Circumventing Backdoor Defenses That Are Based on Latent Separability**

cs.LG

**SubmitDate**: 2022-10-06    [abs](http://arxiv.org/abs/2205.13613v2) [paper-pdf](http://arxiv.org/pdf/2205.13613v2)

**Authors**: Xiangyu Qi, Tinghao Xie, Yiming Li, Saeed Mahloujifar, Prateek Mittal

**Abstract**: Recent studies revealed that deep learning is susceptible to backdoor poisoning attacks. An adversary can embed a hidden backdoor into a model to manipulate its predictions by only modifying a few training data, without controlling the training process. Currently, a tangible signature has been widely observed across a diverse set of backdoor poisoning attacks -- models trained on a poisoned dataset tend to learn separable latent representations for poison and clean samples. This latent separation is so pervasive that a family of backdoor defenses directly take it as a default assumption (dubbed latent separability assumption), based on which to identify poison samples via cluster analysis in the latent space. An intriguing question consequently follows: is the latent separation unavoidable for backdoor poisoning attacks? This question is central to understanding whether the assumption of latent separability provides a reliable foundation for defending against backdoor poisoning attacks. In this paper, we design adaptive backdoor poisoning attacks to present counter-examples against this assumption. Our methods include two key components: (1) a set of trigger-planted samples correctly labeled to their semantic classes (other than the target class) that can regularize backdoor learning; (2) asymmetric trigger planting strategies that help to boost attack success rate (ASR) as well as to diversify latent representations of poison samples. Extensive experiments on benchmark datasets verify the effectiveness of our adaptive attacks in bypassing existing latent separation based backdoor defenses. Moreover, our attacks still maintain a high attack success rate with negligible clean accuracy drop. Our studies call for defense designers to take caution when leveraging latent separation as an assumption in their defenses.



## **10. Dynamic Stochastic Ensemble with Adversarial Robust Lottery Ticket Subnetworks**

cs.CV

**SubmitDate**: 2022-10-06    [abs](http://arxiv.org/abs/2210.02618v1) [paper-pdf](http://arxiv.org/pdf/2210.02618v1)

**Authors**: Qi Peng, Wenlin Liu, Ruoxi Qin, Libin Hou, Bin Yan, Linyuan Wang

**Abstract**: Adversarial attacks are considered the intrinsic vulnerability of CNNs. Defense strategies designed for attacks have been stuck in the adversarial attack-defense arms race, reflecting the imbalance between attack and defense. Dynamic Defense Framework (DDF) recently changed the passive safety status quo based on the stochastic ensemble model. The diversity of subnetworks, an essential concern in the DDF, can be effectively evaluated by the adversarial transferability between different networks. Inspired by the poor adversarial transferability between subnetworks of scratch tickets with various remaining ratios, we propose a method to realize the dynamic stochastic ensemble defense strategy. We discover the adversarial transferable diversity between robust lottery ticket subnetworks drawn from different basic structures and sparsity. The experimental results suggest that our method achieves better robust and clean recognition accuracy by adversarial transferable diversity, which would decrease the reliability of attacks.



## **11. A Closer Look at Robustness to L-infinity and Spatial Perturbations and their Composition**

cs.LG

16 pages, 5 figures, and 3 tables

**SubmitDate**: 2022-10-05    [abs](http://arxiv.org/abs/2210.02577v1) [paper-pdf](http://arxiv.org/pdf/2210.02577v1)

**Authors**: Luke Rowe, Benjamin Thérien, Krzysztof Czarnecki, Hongyang Zhang

**Abstract**: In adversarial machine learning, the popular $\ell_\infty$ threat model has been the focus of much previous work. While this mathematical definition of imperceptibility successfully captures an infinite set of additive image transformations that a model should be robust to, this is only a subset of all transformations which leave the semantic label of an image unchanged. Indeed, previous work also considered robustness to spatial attacks as well as other semantic transformations; however, designing defense methods against the composition of spatial and $\ell_{\infty}$ perturbations remains relatively underexplored. In the following, we improve the understanding of this seldom investigated compositional setting. We prove theoretically that no linear classifier can achieve more than trivial accuracy against a composite adversary in a simple statistical setting, illustrating its difficulty. We then investigate how state-of-the-art $\ell_{\infty}$ defenses can be adapted to this novel threat model and study their performance against compositional attacks. We find that our newly proposed TRADES$_{\text{All}}$ strategy performs the strongest of all. Analyzing its logit's Lipschitz constant for RT transformations of different sizes, we find that TRADES$_{\text{All}}$ remains stable over a wide range of RT transformations with and without $\ell_\infty$ perturbations.



## **12. Learning of Dynamical Systems under Adversarial Attacks -- Null Space Property Perspective**

eess.SY

8 pages, 2 figures

**SubmitDate**: 2022-10-05    [abs](http://arxiv.org/abs/2210.01421v2) [paper-pdf](http://arxiv.org/pdf/2210.01421v2)

**Authors**: Han Feng, Baturalp Yalcin, Javad Lavaei

**Abstract**: We study the identification of a linear time-invariant dynamical system affected by large-and-sparse disturbances modeling adversarial attacks or faults. Under the assumption that the states are measurable, we develop necessary and sufficient conditions for the recovery of the system matrices by solving a constrained lasso-type optimization problem. In addition, we provide an upper bound on the estimation error whenever the disturbance sequence is a combination of small noise values and large adversarial values. Our results depend on the null space property that has been widely used in the lasso literature, and we investigate under what conditions this property holds for linear time-invariant dynamical systems. Lastly, we further study the conditions for a specific probabilistic model and support the results with numerical experiments.



## **13. On Adversarial Robustness of Deep Image Deblurring**

cs.CV

ICIP 2022

**SubmitDate**: 2022-10-05    [abs](http://arxiv.org/abs/2210.02502v1) [paper-pdf](http://arxiv.org/pdf/2210.02502v1)

**Authors**: Kanchana Vaishnavi Gandikota, Paramanand Chandramouli, Michael Moeller

**Abstract**: Recent approaches employ deep learning-based solutions for the recovery of a sharp image from its blurry observation. This paper introduces adversarial attacks against deep learning-based image deblurring methods and evaluates the robustness of these neural networks to untargeted and targeted attacks. We demonstrate that imperceptible distortion can significantly degrade the performance of state-of-the-art deblurring networks, even producing drastically different content in the output, indicating the strong need to include adversarially robust training not only in classification but also for image recovery.



## **14. Dynamical systems' based neural networks**

cs.LG

25 pages with 2 appendices

**SubmitDate**: 2022-10-05    [abs](http://arxiv.org/abs/2210.02373v1) [paper-pdf](http://arxiv.org/pdf/2210.02373v1)

**Authors**: Elena Celledoni, Davide Murari, Brynjulf Owren, Carola-Bibiane Schönlieb, Ferdia Sherry

**Abstract**: Neural networks have gained much interest because of their effectiveness in many applications. However, their mathematical properties are generally not well understood. If there is some underlying geometric structure inherent to the data or to the function to approximate, it is often desirable to take this into account in the design of the neural network. In this work, we start with a non-autonomous ODE and build neural networks using a suitable, structure-preserving, numerical time-discretisation. The structure of the neural network is then inferred from the properties of the ODE vector field. Besides injecting more structure into the network architectures, this modelling procedure allows a better theoretical understanding of their behaviour. We present two universal approximation results and demonstrate how to impose some particular properties on the neural networks. A particular focus is on 1-Lipschitz architectures including layers that are not 1-Lipschitz. These networks are expressive and robust against adversarial attacks, as shown for the CIFAR-10 dataset.



## **15. Can Adversarial Training Be Manipulated By Non-Robust Features?**

cs.LG

NeurIPS 2022

**SubmitDate**: 2022-10-05    [abs](http://arxiv.org/abs/2201.13329v3) [paper-pdf](http://arxiv.org/pdf/2201.13329v3)

**Authors**: Lue Tao, Lei Feng, Hongxin Wei, Jinfeng Yi, Sheng-Jun Huang, Songcan Chen

**Abstract**: Adversarial training, originally designed to resist test-time adversarial examples, has shown to be promising in mitigating training-time availability attacks. This defense ability, however, is challenged in this paper. We identify a novel threat model named stability attacks, which aims to hinder robust availability by slightly manipulating the training data. Under this threat, we show that adversarial training using a conventional defense budget $\epsilon$ provably fails to provide test robustness in a simple statistical setting, where the non-robust features of the training data can be reinforced by $\epsilon$-bounded perturbation. Further, we analyze the necessity of enlarging the defense budget to counter stability attacks. Finally, comprehensive experiments demonstrate that stability attacks are harmful on benchmark datasets, and thus the adaptive defense is necessary to maintain robustness. Our code is available at https://github.com/TLMichael/Hypocritical-Perturbation.



## **16. Image Masking for Robust Self-Supervised Monocular Depth Estimation**

cs.CV

**SubmitDate**: 2022-10-05    [abs](http://arxiv.org/abs/2210.02357v1) [paper-pdf](http://arxiv.org/pdf/2210.02357v1)

**Authors**: Hemang Chawla, Kishaan Jeeveswaran, Elahe Arani, Bahram Zonooz

**Abstract**: Self-supervised monocular depth estimation is a salient task for 3D scene understanding. Learned jointly with monocular ego-motion estimation, several methods have been proposed to predict accurate pixel-wise depth without using labeled data. Nevertheless, these methods focus on improving performance under ideal conditions without natural or digital corruptions. A general absence of occlusions is assumed even for object-specific depth estimation. These methods are also vulnerable to adversarial attacks, which is a pertinent concern for their reliable deployment on robots and autonomous driving systems. We propose MIMDepth, a method that adapts masked image modeling (MIM) for self-supervised monocular depth estimation. While MIM has been used to learn generalizable features during pre-training, we show how it could be adapted for direct training of monocular depth estimation. Our experiments show that MIMDepth is more robust to noise, blur, weather conditions, digital artifacts, occlusions, as well as untargeted and targeted adversarial attacks.



## **17. Jitter Does Matter: Adapting Gaze Estimation to New Domains**

cs.CV

7 pages, 5 figures

**SubmitDate**: 2022-10-05    [abs](http://arxiv.org/abs/2210.02082v1) [paper-pdf](http://arxiv.org/pdf/2210.02082v1)

**Authors**: Ruicong Liu, Yiwei Bao, Mingjie Xu, Haofei Wang, Yunfei Liu, Feng Lu

**Abstract**: Deep neural networks have demonstrated superior performance on appearance-based gaze estimation tasks. However, due to variations in person, illuminations, and background, performance degrades dramatically when applying the model to a new domain. In this paper, we discover an interesting gaze jitter phenomenon in cross-domain gaze estimation, i.e., the gaze predictions of two similar images can be severely deviated in target domain. This is closely related to cross-domain gaze estimation tasks, but surprisingly, it has not been noticed yet previously. Therefore, we innovatively propose to utilize the gaze jitter to analyze and optimize the gaze domain adaptation task. We find that the high-frequency component (HFC) is an important factor that leads to jitter. Based on this discovery, we add high-frequency components to input images using the adversarial attack and employ contrastive learning to encourage the model to obtain similar representations between original and perturbed data, which reduces the impacts of HFC. We evaluate the proposed method on four cross-domain gaze estimation tasks, and experimental results demonstrate that it significantly reduces the gaze jitter and improves the gaze estimation performance in target domains.



## **18. Natural Color Fool: Towards Boosting Black-box Unrestricted Attacks**

cs.CV

NeurIPS 2022

**SubmitDate**: 2022-10-05    [abs](http://arxiv.org/abs/2210.02041v1) [paper-pdf](http://arxiv.org/pdf/2210.02041v1)

**Authors**: Shengming Yuan, Qilong Zhang, Lianli Gao, Yaya Cheng, Jingkuan Song

**Abstract**: Unrestricted color attacks, which manipulate semantically meaningful color of an image, have shown their stealthiness and success in fooling both human eyes and deep neural networks. However, current works usually sacrifice the flexibility of the uncontrolled setting to ensure the naturalness of adversarial examples. As a result, the black-box attack performance of these methods is limited. To boost transferability of adversarial examples without damaging image quality, we propose a novel Natural Color Fool (NCF) which is guided by realistic color distributions sampled from a publicly available dataset and optimized by our neighborhood search and initialization reset. By conducting extensive experiments and visualizations, we convincingly demonstrate the effectiveness of our proposed method. Notably, on average, results show that our NCF can outperform state-of-the-art approaches by 15.0%$\sim$32.9% for fooling normally trained models and 10.0%$\sim$25.3% for evading defense methods. Our code is available at https://github.com/ylhz/Natural-Color-Fool.



## **19. Towards Adversarially Robust Deepfake Detection: An Ensemble Approach**

cs.LG

**SubmitDate**: 2022-10-05    [abs](http://arxiv.org/abs/2202.05687v2) [paper-pdf](http://arxiv.org/pdf/2202.05687v2)

**Authors**: Ashish Hooda, Neal Mangaokar, Ryan Feng, Kassem Fawaz, Somesh Jha, Atul Prakash

**Abstract**: Detecting deepfakes remains an open problem. Current detection methods fail against an adversary who adds imperceptible adversarial perturbations to the deepfake to evade detection. We propose Disjoint Deepfake Detection (D3), a deepfake detector designed to improve adversarial robustness beyond de facto solutions such as adversarial training. D3 uses an ensemble of models over disjoint subsets of the frequency spectrum to significantly improve robustness. Our key insight is to leverage a redundancy in the frequency domain and apply a saliency partitioning technique to disjointly distribute frequency components across multiple models. We formally prove that these disjoint ensembles lead to a reduction in the dimensionality of the input subspace where adversarial deepfakes lie. We then empirically validate the D3 method against white-box attacks and black-box attacks and find that D3 significantly outperforms existing state-of-the-art defenses applied to deepfake detection.



## **20. Practical Adversarial Attacks on Spatiotemporal Traffic Forecasting Models**

cs.LG

Accepted by NeurIPS 2022

**SubmitDate**: 2022-10-05    [abs](http://arxiv.org/abs/2210.02447v1) [paper-pdf](http://arxiv.org/pdf/2210.02447v1)

**Authors**: Fan Liu, Hao Liu, Wenzhao Jiang

**Abstract**: Machine learning based traffic forecasting models leverage sophisticated spatiotemporal auto-correlations to provide accurate predictions of city-wide traffic states. However, existing methods assume a reliable and unbiased forecasting environment, which is not always available in the wild. In this work, we investigate the vulnerability of spatiotemporal traffic forecasting models and propose a practical adversarial spatiotemporal attack framework. Specifically, instead of simultaneously attacking all geo-distributed data sources, an iterative gradient-guided node saliency method is proposed to identify the time-dependent set of victim nodes. Furthermore, we devise a spatiotemporal gradient descent based scheme to generate real-valued adversarial traffic states under a perturbation constraint. Meanwhile, we theoretically demonstrate the worst performance bound of adversarial traffic forecasting attacks. Extensive experiments on two real-world datasets show that the proposed two-step framework achieves up to $67.8\%$ performance degradation on various advanced spatiotemporal forecasting models. Remarkably, we also show that adversarial training with our proposed attacks can significantly improve the robustness of spatiotemporal traffic forecasting models. Our code is available in \url{https://github.com/luckyfan-cs/ASTFA}.



## **21. Robust Fair Clustering: A Novel Fairness Attack and Defense Framework**

cs.LG

**SubmitDate**: 2022-10-04    [abs](http://arxiv.org/abs/2210.01953v1) [paper-pdf](http://arxiv.org/pdf/2210.01953v1)

**Authors**: Anshuman Chhabra, Peizhao Li, Prasant Mohapatra, Hongfu Liu

**Abstract**: Clustering algorithms are widely used in many societal resource allocation applications, such as loan approvals and candidate recruitment, among others, and hence, biased or unfair model outputs can adversely impact individuals that rely on these applications. To this end, many fair clustering approaches have been recently proposed to counteract this issue. Due to the potential for significant harm, it is essential to ensure that fair clustering algorithms provide consistently fair outputs even under adversarial influence. However, fair clustering algorithms have not been studied from an adversarial attack perspective. In contrast to previous research, we seek to bridge this gap and conduct a robustness analysis against fair clustering by proposing a novel black-box fairness attack. Through comprehensive experiments, we find that state-of-the-art models are highly susceptible to our attack as it can reduce their fairness performance significantly. Finally, we propose Consensus Fair Clustering (CFC), the first robust fair clustering approach that transforms consensus clustering into a fair graph partitioning problem, and iteratively learns to generate fair cluster outputs. Experimentally, we observe that CFC is highly robust to the proposed attack and is thus a truly robust fair clustering alternative.



## **22. On the Robustness of Deep Clustering Models: Adversarial Attacks and Defenses**

cs.LG

Accepted to the 36th Conference on Neural Information Processing  Systems (NeurIPS 2022)

**SubmitDate**: 2022-10-04    [abs](http://arxiv.org/abs/2210.01940v1) [paper-pdf](http://arxiv.org/pdf/2210.01940v1)

**Authors**: Anshuman Chhabra, Ashwin Sekhari, Prasant Mohapatra

**Abstract**: Clustering models constitute a class of unsupervised machine learning methods which are used in a number of application pipelines, and play a vital role in modern data science. With recent advancements in deep learning -- deep clustering models have emerged as the current state-of-the-art over traditional clustering approaches, especially for high-dimensional image datasets. While traditional clustering approaches have been analyzed from a robustness perspective, no prior work has investigated adversarial attacks and robustness for deep clustering models in a principled manner. To bridge this gap, we propose a blackbox attack using Generative Adversarial Networks (GANs) where the adversary does not know which deep clustering model is being used, but can query it for outputs. We analyze our attack against multiple state-of-the-art deep clustering models and real-world datasets, and find that it is highly successful. We then employ some natural unsupervised defense approaches, but find that these are unable to mitigate our attack. Finally, we attack Face++, a production-level face clustering API service, and find that we can significantly reduce its performance as well. Through this work, we thus aim to motivate the need for truly robust deep clustering models.



## **23. Revisiting and Advancing Fast Adversarial Training Through The Lens of Bi-Level Optimization**

cs.LG

**SubmitDate**: 2022-10-04    [abs](http://arxiv.org/abs/2112.12376v6) [paper-pdf](http://arxiv.org/pdf/2112.12376v6)

**Authors**: Yihua Zhang, Guanhua Zhang, Prashant Khanduri, Mingyi Hong, Shiyu Chang, Sijia Liu

**Abstract**: Adversarial training (AT) is a widely recognized defense mechanism to gain the robustness of deep neural networks against adversarial attacks. It is built on min-max optimization (MMO), where the minimizer (i.e., defender) seeks a robust model to minimize the worst-case training loss in the presence of adversarial examples crafted by the maximizer (i.e., attacker). However, the conventional MMO method makes AT hard to scale. Thus, Fast-AT (Wong et al., 2020) and other recent algorithms attempt to simplify MMO by replacing its maximization step with the single gradient sign-based attack generation step. Although easy to implement, Fast-AT lacks theoretical guarantees, and its empirical performance is unsatisfactory due to the issue of robust catastrophic overfitting when training with strong adversaries. In this paper, we advance Fast-AT from the fresh perspective of bi-level optimization (BLO). We first show that the commonly-used Fast-AT is equivalent to using a stochastic gradient algorithm to solve a linearized BLO problem involving a sign operation. However, the discrete nature of the sign operation makes it difficult to understand the algorithm performance. Inspired by BLO, we design and analyze a new set of robust training algorithms termed Fast Bi-level AT (Fast-BAT), which effectively defends sign-based projected gradient descent (PGD) attacks without using any gradient sign method or explicit robust regularization. In practice, we show our method yields substantial robustness improvements over baselines across multiple models and datasets. Codes are available at https://github.com/OPTML-Group/Fast-BAT.



## **24. Invariant Aggregator for Defending Federated Backdoor Attacks**

cs.LG

**SubmitDate**: 2022-10-04    [abs](http://arxiv.org/abs/2210.01834v1) [paper-pdf](http://arxiv.org/pdf/2210.01834v1)

**Authors**: Xiaoyang Wang, Dimitrios Dimitriadis, Sanmi Koyejo, Shruti Tople

**Abstract**: Federated learning is gaining popularity as it enables training of high-utility models across several clients without directly sharing their private data. As a downside, the federated setting makes the model vulnerable to various adversarial attacks in the presence of malicious clients. Specifically, an adversary can perform backdoor attacks to control model predictions via poisoning the training dataset with a trigger. In this work, we propose a mitigation for backdoor attacks in a federated learning setup. Our solution forces the model optimization trajectory to focus on the invariant directions that are generally useful for utility and avoid selecting directions that favor few and possibly malicious clients. Concretely, we consider the sign consistency of the pseudo-gradient (the client update) as an estimation of the invariance. Following this, our approach performs dimension-wise filtering to remove pseudo-gradient elements with low sign consistency. Then, a robust mean estimator eliminates outliers among the remaining dimensions. Our theoretical analysis further shows the necessity of the defense combination and illustrates how our proposed solution defends the federated learning model. Empirical results on three datasets with different modalities and varying number of clients show that our approach mitigates backdoor attacks with a negligible cost on the model utility.



## **25. Data Leakage in Tabular Federated Learning**

cs.LG

**SubmitDate**: 2022-10-04    [abs](http://arxiv.org/abs/2210.01785v1) [paper-pdf](http://arxiv.org/pdf/2210.01785v1)

**Authors**: Mark Vero, Mislav Balunović, Dimitar I. Dimitrov, Martin Vechev

**Abstract**: While federated learning (FL) promises to preserve privacy in distributed training of deep learning models, recent work in the image and NLP domains showed that training updates leak private data of participating clients. At the same time, most high-stakes applications of FL (e.g., legal and financial) use tabular data. Compared to the NLP and image domains, reconstruction of tabular data poses several unique challenges: (i) categorical features introduce a significantly more difficult mixed discrete-continuous optimization problem, (ii) the mix of categorical and continuous features causes high variance in the final reconstructions, and (iii) structured data makes it difficult for the adversary to judge reconstruction quality. In this work, we tackle these challenges and propose the first comprehensive reconstruction attack on tabular data, called TabLeak. TabLeak is based on three key ingredients: (i) a softmax structural prior, implicitly converting the mixed discrete-continuous optimization problem into an easier fully continuous one, (ii) a way to reduce the variance of our reconstructions through a pooled ensembling scheme exploiting the structure of tabular data, and (iii) an entropy measure which can successfully assess reconstruction quality. Our experimental evaluation demonstrates the effectiveness of TabLeak, reaching a state-of-the-art on four popular tabular datasets. For instance, on the Adult dataset, we improve attack accuracy by 10% compared to the baseline on the practically relevant batch size of 32 and further obtain non-trivial reconstructions for batch sizes as large as 128. Our findings are important as they show that performing FL on tabular data, which often poses high privacy risks, is highly vulnerable.



## **26. Adversarial Attack on Attackers: Post-Process to Mitigate Black-Box Score-Based Query Attacks**

cs.LG

accepted by NeurIPS 2022

**SubmitDate**: 2022-10-04    [abs](http://arxiv.org/abs/2205.12134v2) [paper-pdf](http://arxiv.org/pdf/2205.12134v2)

**Authors**: Sizhe Chen, Zhehao Huang, Qinghua Tao, Yingwen Wu, Cihang Xie, Xiaolin Huang

**Abstract**: The score-based query attacks (SQAs) pose practical threats to deep neural networks by crafting adversarial perturbations within dozens of queries, only using the model's output scores. Nonetheless, we note that if the loss trend of the outputs is slightly perturbed, SQAs could be easily misled and thereby become much less effective. Following this idea, we propose a novel defense, namely Adversarial Attack on Attackers (AAA), to confound SQAs towards incorrect attack directions by slightly modifying the output logits. In this way, (1) SQAs are prevented regardless of the model's worst-case robustness; (2) the original model predictions are hardly changed, i.e., no degradation on clean accuracy; (3) the calibration of confidence scores can be improved simultaneously. Extensive experiments are provided to verify the above advantages. For example, by setting $\ell_\infty=8/255$ on CIFAR-10, our proposed AAA helps WideResNet-28 secure 80.59% accuracy under Square attack (2500 queries), while the best prior defense (i.e., adversarial training) only attains 67.44%. Since AAA attacks SQA's general greedy strategy, such advantages of AAA over 8 defenses can be consistently observed on 8 CIFAR-10/ImageNet models under 6 SQAs, using different attack targets, bounds, norms, losses, and strategies. Moreover, AAA calibrates better without hurting the accuracy. Our code is available at https://github.com/Sizhe-Chen/AAA.



## **27. Causal Intervention-based Prompt Debiasing for Event Argument Extraction**

cs.CL

**SubmitDate**: 2022-10-04    [abs](http://arxiv.org/abs/2210.01561v1) [paper-pdf](http://arxiv.org/pdf/2210.01561v1)

**Authors**: Jiaju Lin, Jie Zhou, Qin Chen

**Abstract**: Prompt-based methods have become increasingly popular among information extraction tasks, especially in low-data scenarios. By formatting a finetune task into a pre-training objective, prompt-based methods resolve the data scarce problem effectively. However, seldom do previous research investigate the discrepancy among different prompt formulating strategies. In this work, we compare two kinds of prompts, name-based prompt and ontology-base prompt, and reveal how ontology-base prompt methods exceed its counterpart in zero-shot event argument extraction (EAE) . Furthermore, we analyse the potential risk in ontology-base prompts via a causal view and propose a debias method by causal intervention. Experiments on two benchmarks demonstrate that modified by our debias method, the baseline model becomes both more effective and robust, with significant improvement in the resistance to adversarial attacks.



## **28. Decompiling x86 Deep Neural Network Executables**

cs.CR

The extended version of a paper to appear in the Proceedings of the  32nd USENIX Security Symposium, 2023, (USENIX Security '23), 25 pages

**SubmitDate**: 2022-10-04    [abs](http://arxiv.org/abs/2210.01075v2) [paper-pdf](http://arxiv.org/pdf/2210.01075v2)

**Authors**: Zhibo Liu, Yuanyuan Yuan, Shuai Wang, Xiaofei Xie, Lei Ma

**Abstract**: Due to their widespread use on heterogeneous hardware devices, deep learning (DL) models are compiled into executables by DL compilers to fully leverage low-level hardware primitives. This approach allows DL computations to be undertaken at low cost across a variety of computing platforms, including CPUs, GPUs, and various hardware accelerators.   We present BTD (Bin to DNN), a decompiler for deep neural network (DNN) executables. BTD takes DNN executables and outputs full model specifications, including types of DNN operators, network topology, dimensions, and parameters that are (nearly) identical to those of the input models. BTD delivers a practical framework to process DNN executables compiled by different DL compilers and with full optimizations enabled on x86 platforms. It employs learning-based techniques to infer DNN operators, dynamic analysis to reveal network architectures, and symbolic execution to facilitate inferring dimensions and parameters of DNN operators.   Our evaluation reveals that BTD enables accurate recovery of full specifications of complex DNNs with millions of parameters (e.g., ResNet). The recovered DNN specifications can be re-compiled into a new DNN executable exhibiting identical behavior to the input executable. We show that BTD can boost two representative attacks, adversarial example generation and knowledge stealing, against DNN executables. We also demonstrate cross-architecture legacy code reuse using BTD, and envision BTD being used for other critical downstream tasks like DNN security hardening and patching.



## **29. Physical Passive Patch Adversarial Attacks on Visual Odometry Systems**

cs.CV

Accepted to ACCV 2022

**SubmitDate**: 2022-10-04    [abs](http://arxiv.org/abs/2207.05729v2) [paper-pdf](http://arxiv.org/pdf/2207.05729v2)

**Authors**: Yaniv Nemcovsky, Matan Jacoby, Alex M. Bronstein, Chaim Baskin

**Abstract**: Deep neural networks are known to be susceptible to adversarial perturbations -- small perturbations that alter the output of the network and exist under strict norm limitations. While such perturbations are usually discussed as tailored to a specific input, a universal perturbation can be constructed to alter the model's output on a set of inputs. Universal perturbations present a more realistic case of adversarial attacks, as awareness of the model's exact input is not required. In addition, the universal attack setting raises the subject of generalization to unseen data, where given a set of inputs, the universal perturbations aim to alter the model's output on out-of-sample data. In this work, we study physical passive patch adversarial attacks on visual odometry-based autonomous navigation systems. A visual odometry system aims to infer the relative camera motion between two corresponding viewpoints, and is frequently used by vision-based autonomous navigation systems to estimate their state. For such navigation systems, a patch adversarial perturbation poses a severe security issue, as it can be used to mislead a system onto some collision course. To the best of our knowledge, we show for the first time that the error margin of a visual odometry model can be significantly increased by deploying patch adversarial attacks in the scene. We provide evaluation on synthetic closed-loop drone navigation data and demonstrate that a comparable vulnerability exists in real data. A reference implementation of the proposed method and the reported experiments is provided at https://github.com/patchadversarialattacks/patchadversarialattacks.



## **30. Exploring Adversarially Robust Training for Unsupervised Domain Adaptation**

cs.CV

Accepted at Asian Conference on Computer Vision (ACCV) 2022

**SubmitDate**: 2022-10-04    [abs](http://arxiv.org/abs/2202.09300v2) [paper-pdf](http://arxiv.org/pdf/2202.09300v2)

**Authors**: Shao-Yuan Lo, Vishal M. Patel

**Abstract**: Unsupervised Domain Adaptation (UDA) methods aim to transfer knowledge from a labeled source domain to an unlabeled target domain. UDA has been extensively studied in the computer vision literature. Deep networks have been shown to be vulnerable to adversarial attacks. However, very little focus is devoted to improving the adversarial robustness of deep UDA models, causing serious concerns about model reliability. Adversarial Training (AT) has been considered to be the most successful adversarial defense approach. Nevertheless, conventional AT requires ground-truth labels to generate adversarial examples and train models, which limits its effectiveness in the unlabeled target domain. In this paper, we aim to explore AT to robustify UDA models: How to enhance the unlabeled data robustness via AT while learning domain-invariant features for UDA? To answer this question, we provide a systematic study into multiple AT variants that can potentially be applied to UDA. Moreover, we propose a novel Adversarially Robust Training method for UDA accordingly, referred to as ARTUDA. Extensive experiments on multiple adversarial attacks and UDA benchmarks show that ARTUDA consistently improves the adversarial robustness of UDA models. Code is available at https://github.com/shaoyuanlo/ARTUDA



## **31. A Study on the Efficiency and Generalization of Light Hybrid Retrievers**

cs.IR

**SubmitDate**: 2022-10-04    [abs](http://arxiv.org/abs/2210.01371v1) [paper-pdf](http://arxiv.org/pdf/2210.01371v1)

**Authors**: Man Luo, Shashank Jain, Anchit Gupta, Arash Einolghozati, Barlas Oguz, Debojeet Chatterjee, Xilun Chen, Chitta Baral, Peyman Heidari

**Abstract**: Existing hybrid retrievers which integrate sparse and dense retrievers, are indexing-heavy, limiting their applicability in real-world on-devices settings. We ask the question "Is it possible to reduce the indexing memory of hybrid retrievers without sacrificing performance?" Driven by this question, we leverage an indexing-efficient dense retriever (i.e. DrBoost) to obtain a light hybrid retriever. Moreover, to further reduce the memory, we introduce a lighter dense retriever (LITE) which is jointly trained on contrastive learning and knowledge distillation from DrBoost. Compared to previous heavy hybrid retrievers, our Hybrid-LITE retriever saves 13 memory while maintaining 98.0 performance.   In addition, we study the generalization of light hybrid retrievers along two dimensions, out-of-domain (OOD) generalization and robustness against adversarial attacks. We evaluate models on two existing OOD benchmarks and create six adversarial attack sets for robustness evaluation. Experiments show that our light hybrid retrievers achieve better robustness performance than both sparse and dense retrievers. Nevertheless there is a large room to improve the robustness of retrievers, and our datasets can aid future research.



## **32. Strength-Adaptive Adversarial Training**

cs.LG

**SubmitDate**: 2022-10-04    [abs](http://arxiv.org/abs/2210.01288v1) [paper-pdf](http://arxiv.org/pdf/2210.01288v1)

**Authors**: Chaojian Yu, Dawei Zhou, Li Shen, Jun Yu, Bo Han, Mingming Gong, Nannan Wang, Tongliang Liu

**Abstract**: Adversarial training (AT) is proved to reliably improve network's robustness against adversarial data. However, current AT with a pre-specified perturbation budget has limitations in learning a robust network. Firstly, applying a pre-specified perturbation budget on networks of various model capacities will yield divergent degree of robustness disparity between natural and robust accuracies, which deviates from robust network's desideratum. Secondly, the attack strength of adversarial training data constrained by the pre-specified perturbation budget fails to upgrade as the growth of network robustness, which leads to robust overfitting and further degrades the adversarial robustness. To overcome these limitations, we propose \emph{Strength-Adaptive Adversarial Training} (SAAT). Specifically, the adversary employs an adversarial loss constraint to generate adversarial training data. Under this constraint, the perturbation budget will be adaptively adjusted according to the training state of adversarial data, which can effectively avoid robust overfitting. Besides, SAAT explicitly constrains the attack strength of training data through the adversarial loss, which manipulates model capacity scheduling during training, and thereby can flexibly control the degree of robustness disparity and adjust the tradeoff between natural accuracy and robustness. Extensive experiments show that our proposal boosts the robustness of adversarial training.



## **33. On Attacking Out-Domain Uncertainty Estimation in Deep Neural Networks**

cs.LG

**SubmitDate**: 2022-10-03    [abs](http://arxiv.org/abs/2210.02191v1) [paper-pdf](http://arxiv.org/pdf/2210.02191v1)

**Authors**: Huimin Zeng, Zhenrui Yue, Yang Zhang, Ziyi Kou, Lanyu Shang, Dong Wang

**Abstract**: In many applications with real-world consequences, it is crucial to develop reliable uncertainty estimation for the predictions made by the AI decision systems. Targeting at the goal of estimating uncertainty, various deep neural network (DNN) based uncertainty estimation algorithms have been proposed. However, the robustness of the uncertainty returned by these algorithms has not been systematically explored. In this work, to raise the awareness of the research community on robust uncertainty estimation, we show that state-of-the-art uncertainty estimation algorithms could fail catastrophically under our proposed adversarial attack despite their impressive performance on uncertainty estimation. In particular, we aim at attacking the out-domain uncertainty estimation: under our attack, the uncertainty model would be fooled to make high-confident predictions for the out-domain data, which they originally would have rejected. Extensive experimental results on various benchmark image datasets show that the uncertainty estimated by state-of-the-art methods could be easily corrupted by our attack.



## **34. Adversarially Robust One-class Novelty Detection**

cs.CV

Accepted in IEEE Transactions on Pattern Analysis and Machine  Intelligence (T-PAMI), 2022

**SubmitDate**: 2022-10-03    [abs](http://arxiv.org/abs/2108.11168v2) [paper-pdf](http://arxiv.org/pdf/2108.11168v2)

**Authors**: Shao-Yuan Lo, Poojan Oza, Vishal M. Patel

**Abstract**: One-class novelty detectors are trained with examples of a particular class and are tasked with identifying whether a query example belongs to the same known class. Most recent advances adopt a deep auto-encoder style architecture to compute novelty scores for detecting novel class data. Deep networks have shown to be vulnerable to adversarial attacks, yet little focus is devoted to studying the adversarial robustness of deep novelty detectors. In this paper, we first show that existing novelty detectors are susceptible to adversarial examples. We further demonstrate that commonly-used defense approaches for classification tasks have limited effectiveness in one-class novelty detection. Hence, we need a defense specifically designed for novelty detection. To this end, we propose a defense strategy that manipulates the latent space of novelty detectors to improve the robustness against adversarial examples. The proposed method, referred to as Principal Latent Space (PrincipaLS), learns the incrementally-trained cascade principal components in the latent space to robustify novelty detectors. PrincipaLS can purify latent space against adversarial examples and constrain latent space to exclusively model the known class distribution. We conduct extensive experiments on eight attacks, five datasets and seven novelty detectors, showing that PrincipaLS consistently enhances the adversarial robustness of novelty detection models. Code is available at https://github.com/shaoyuanlo/PrincipaLS



## **35. AutoJoin: Efficient Adversarial Training for Robust Maneuvering via Denoising Autoencoder and Joint Learning**

cs.LG

**SubmitDate**: 2022-10-03    [abs](http://arxiv.org/abs/2205.10933v2) [paper-pdf](http://arxiv.org/pdf/2205.10933v2)

**Authors**: Michael Villarreal, Bibek Poudel, Ryan Wickman, Yu Shen, Weizi Li

**Abstract**: As a result of increasingly adopted machine learning algorithms and ubiquitous sensors, many 'perception-to-control' systems are developed and deployed. For these systems to be trustworthy, we need to improve their robustness with adversarial training being one approach. We propose a gradient-free adversarial training technique, called AutoJoin, which is a very simple yet effective and efficient approach to produce robust models for imaged-based maneuvering. Compared to other SOTA methods with testing on over 5M perturbed and clean images, AutoJoin achieves significant performance increases up to the 40% range under gradient-free perturbations while improving on clean performance up to 300%. Regarding efficiency, AutoJoin demonstrates strong advantages over other SOTA techniques by saving up to 83% time per training epoch and 90% training data. Although not the focus of AutoJoin, it even demonstrates superb ability in defending gradient-based attacks. The core idea of AutoJoin is to use a decoder attachment to the original regression model creating a denoising autoencoder within the architecture. This architecture allows the tasks 'maneuvering' and 'denoising sensor input' to be jointly learnt and reinforce each other's performance.



## **36. Leveraging Local Patch Differences in Multi-Object Scenes for Generative Adversarial Attacks**

cs.CV

Accepted at WACV 2023 (Round 1), camera-ready version

**SubmitDate**: 2022-10-03    [abs](http://arxiv.org/abs/2209.09883v2) [paper-pdf](http://arxiv.org/pdf/2209.09883v2)

**Authors**: Abhishek Aich, Shasha Li, Chengyu Song, M. Salman Asif, Srikanth V. Krishnamurthy, Amit K. Roy-Chowdhury

**Abstract**: State-of-the-art generative model-based attacks against image classifiers overwhelmingly focus on single-object (i.e., single dominant object) images. Different from such settings, we tackle a more practical problem of generating adversarial perturbations using multi-object (i.e., multiple dominant objects) images as they are representative of most real-world scenes. Our goal is to design an attack strategy that can learn from such natural scenes by leveraging the local patch differences that occur inherently in such images (e.g. difference between the local patch on the object `person' and the object `bike' in a traffic scene). Our key idea is to misclassify an adversarial multi-object image by confusing the victim classifier for each local patch in the image. Based on this, we propose a novel generative attack (called Local Patch Difference or LPD-Attack) where a novel contrastive loss function uses the aforesaid local differences in feature space of multi-object scenes to optimize the perturbation generator. Through various experiments across diverse victim convolutional neural networks, we show that our approach outperforms baseline generative attacks with highly transferable perturbations when evaluated under different white-box and black-box settings.



## **37. Bridging the Performance Gap between FGSM and PGD Adversarial Training**

cs.CR

**SubmitDate**: 2022-10-03    [abs](http://arxiv.org/abs/2011.05157v2) [paper-pdf](http://arxiv.org/pdf/2011.05157v2)

**Authors**: Tianjin Huang, Vlado Menkovski, Yulong Pei, Mykola Pechenizkiy

**Abstract**: Deep learning achieves state-of-the-art performance in many tasks but exposes to the underlying vulnerability against adversarial examples. Across existing defense techniques, adversarial training with the projected gradient decent attack (adv.PGD) is considered as one of the most effective ways to achieve moderate adversarial robustness. However, adv.PGD requires too much training time since the projected gradient attack (PGD) takes multiple iterations to generate perturbations. On the other hand, adversarial training with the fast gradient sign method (adv.FGSM) takes much less training time since the fast gradient sign method (FGSM) takes one step to generate perturbations but fails to increase adversarial robustness. In this work, we extend adv.FGSM to make it achieve the adversarial robustness of adv.PGD. We demonstrate that the large curvature along FGSM perturbed direction leads to a large difference in performance of adversarial robustness between adv.FGSM and adv.PGD, and therefore propose combining adv.FGSM with a curvature regularization (adv.FGSMR) in order to bridge the performance gap between adv.FGSM and adv.PGD. The experiments show that adv.FGSMR has higher training efficiency than adv.PGD. In addition, it achieves comparable performance of adversarial robustness on MNIST dataset under white-box attack, and it achieves better performance than adv.PGD under white-box attack and effectively defends the transferable adversarial attack on CIFAR-10 dataset.



## **38. MultiGuard: Provably Robust Multi-label Classification against Adversarial Examples**

cs.CR

Accepted by NeurIPS 2022

**SubmitDate**: 2022-10-03    [abs](http://arxiv.org/abs/2210.01111v1) [paper-pdf](http://arxiv.org/pdf/2210.01111v1)

**Authors**: Jinyuan Jia, Wenjie Qu, Neil Zhenqiang Gong

**Abstract**: Multi-label classification, which predicts a set of labels for an input, has many applications. However, multiple recent studies showed that multi-label classification is vulnerable to adversarial examples. In particular, an attacker can manipulate the labels predicted by a multi-label classifier for an input via adding carefully crafted, human-imperceptible perturbation to it. Existing provable defenses for multi-class classification achieve sub-optimal provable robustness guarantees when generalized to multi-label classification. In this work, we propose MultiGuard, the first provably robust defense against adversarial examples to multi-label classification. Our MultiGuard leverages randomized smoothing, which is the state-of-the-art technique to build provably robust classifiers. Specifically, given an arbitrary multi-label classifier, our MultiGuard builds a smoothed multi-label classifier via adding random noise to the input. We consider isotropic Gaussian noise in this work. Our major theoretical contribution is that we show a certain number of ground truth labels of an input are provably in the set of labels predicted by our MultiGuard when the $\ell_2$-norm of the adversarial perturbation added to the input is bounded. Moreover, we design an algorithm to compute our provable robustness guarantees. Empirically, we evaluate our MultiGuard on VOC 2007, MS-COCO, and NUS-WIDE benchmark datasets. Our code is available at: \url{https://github.com/quwenjie/MultiGuard}



## **39. ASGNN: Graph Neural Networks with Adaptive Structure**

cs.LG

**SubmitDate**: 2022-10-03    [abs](http://arxiv.org/abs/2210.01002v1) [paper-pdf](http://arxiv.org/pdf/2210.01002v1)

**Authors**: Zepeng Zhang, Songtao Lu, Zengfeng Huang, Ziping Zhao

**Abstract**: The graph neural network (GNN) models have presented impressive achievements in numerous machine learning tasks. However, many existing GNN models are shown to be vulnerable to adversarial attacks, which creates a stringent need to build robust GNN architectures. In this work, we propose a novel interpretable message passing scheme with adaptive structure (ASMP) to defend against adversarial attacks on graph structure. Layers in ASMP are derived based on optimization steps that minimize an objective function that learns the node feature and the graph structure simultaneously. ASMP is adaptive in the sense that the message passing process in different layers is able to be carried out over dynamically adjusted graphs. Such property allows more fine-grained handling of the noisy (or perturbed) graph structure and hence improves the robustness. Convergence properties of the ASMP scheme are theoretically established. Integrating ASMP with neural networks can lead to a new family of GNN models with adaptive structure (ASGNN). Extensive experiments on semi-supervised node classification tasks demonstrate that the proposed ASGNN outperforms the state-of-the-art GNN architectures in terms of classification performance under various adversarial attacks.



## **40. Improving Robustness of Deep Reinforcement Learning Agents: Environment Attack based on the Critic Network**

cs.LG

8 pages, 8 figures

**SubmitDate**: 2022-10-03    [abs](http://arxiv.org/abs/2104.03154v3) [paper-pdf](http://arxiv.org/pdf/2104.03154v3)

**Authors**: Lucas Schott, Hatem Hajri, Sylvain Lamprier

**Abstract**: To improve policy robustness of deep reinforcement learning agents, a line of recent works focus on producing disturbances of the environment. Existing approaches of the literature to generate meaningful disturbances of the environment are adversarial reinforcement learning methods. These methods set the problem as a two-player game between the protagonist agent, which learns to perform a task in an environment, and the adversary agent, which learns to disturb the protagonist via modifications of the considered environment. Both protagonist and adversary are trained with deep reinforcement learning algorithms. Alternatively, we propose in this paper to build on gradient-based adversarial attacks, usually used for classification tasks for instance, that we apply on the critic network of the protagonist to identify efficient disturbances of the environment. Rather than learning an attacker policy, which usually reveals as very complex and unstable, we leverage the knowledge of the critic network of the protagonist, to dynamically complexify the task at each step of the learning process. We show that our method, while being faster and lighter, leads to significantly better improvements in policy robustness than existing methods of the literature.



## **41. Push-Pull: Characterizing the Adversarial Robustness for Audio-Visual Active Speaker Detection**

cs.SD

Accepted by SLT 2022

**SubmitDate**: 2022-10-03    [abs](http://arxiv.org/abs/2210.00753v1) [paper-pdf](http://arxiv.org/pdf/2210.00753v1)

**Authors**: Xuanjun Chen, Haibin Wu, Helen Meng, Hung-yi Lee, Jyh-Shing Roger Jang

**Abstract**: Audio-visual active speaker detection (AVASD) is well-developed, and now is an indispensable front-end for several multi-modal applications. However, to the best of our knowledge, the adversarial robustness of AVASD models hasn't been investigated, not to mention the effective defense against such attacks. In this paper, we are the first to reveal the vulnerability of AVASD models under audio-only, visual-only, and audio-visual adversarial attacks through extensive experiments. What's more, we also propose a novel audio-visual interaction loss (AVIL) for making attackers difficult to find feasible adversarial examples under an allocated attack budget. The loss aims at pushing the inter-class embeddings to be dispersed, namely non-speech and speech clusters, sufficiently disentangled, and pulling the intra-class embeddings as close as possible to keep them compact. Experimental results show the AVIL outperforms the adversarial training by 33.14 mAP (%) under multi-modal attacks.



## **42. On the Robustness of Safe Reinforcement Learning under Observational Perturbations**

cs.LG

30 pages, 4 figures, 8 tables

**SubmitDate**: 2022-10-03    [abs](http://arxiv.org/abs/2205.14691v2) [paper-pdf](http://arxiv.org/pdf/2205.14691v2)

**Authors**: Zuxin Liu, Zijian Guo, Zhepeng Cen, Huan Zhang, Jie Tan, Bo Li, Ding Zhao

**Abstract**: Safe reinforcement learning (RL) trains a policy to maximize the task reward while satisfying safety constraints. While prior works focus on the performance optimality, we find that the optimal solutions of many safe RL problems are not robust and safe against carefully designed observational perturbations. We formally analyze the unique properties of designing effective state adversarial attackers in the safe RL setting. We show that baseline adversarial attack techniques for standard RL tasks are not always effective for safe RL and proposed two new approaches - one maximizes the cost and the other maximizes the reward. One interesting and counter-intuitive finding is that the maximum reward attack is strong, as it can both induce unsafe behaviors and make the attack stealthy by maintaining the reward. We further propose a more effective adversarial training framework for safe RL and evaluate it via comprehensive experiments. This paper provides a pioneer work to investigate the safety and robustness of RL under observational attacks for future safe RL studies.



## **43. Improving Model Robustness with Latent Distribution Locally and Globally**

cs.LG

**SubmitDate**: 2022-10-03    [abs](http://arxiv.org/abs/2107.04401v2) [paper-pdf](http://arxiv.org/pdf/2107.04401v2)

**Authors**: Zhuang Qian, Shufei Zhang, Kaizhu Huang, Qiufeng Wang, Rui Zhang, Xinping Yi

**Abstract**: In this work, we consider model robustness of deep neural networks against adversarial attacks from a global manifold perspective. Leveraging both the local and global latent information, we propose a novel adversarial training method through robust optimization, and a tractable way to generate Latent Manifold Adversarial Examples (LMAEs) via an adversarial game between a discriminator and a classifier. The proposed adversarial training with latent distribution (ATLD) method defends against adversarial attacks by crafting LMAEs with the latent manifold in an unsupervised manner. ATLD preserves the local and global information of latent manifold and promises improved robustness against adversarial attacks. To verify the effectiveness of our proposed method, we conduct extensive experiments over different datasets (e.g., CIFAR-10, CIFAR-100, SVHN) with different adversarial attacks (e.g., PGD, CW), and show that our method substantially outperforms the state-of-the-art (e.g., Feature Scattering) in adversarial robustness by a large accuracy margin. The source codes are available at https://github.com/LitterQ/ATLD-pytorch.



## **44. Automated Security Analysis of Exposure Notification Systems**

cs.CR

23 pages, Full version of the corresponding USENIX Security '23 paper

**SubmitDate**: 2022-10-02    [abs](http://arxiv.org/abs/2210.00649v1) [paper-pdf](http://arxiv.org/pdf/2210.00649v1)

**Authors**: Kevin Morio, Ilkan Esiyok, Dennis Jackson, Robert Künnemann

**Abstract**: We present the first formal analysis and comparison of the security of the two most widely deployed exposure notification systems, ROBERT and the Google and Apple Exposure Notification (GAEN) framework. ROBERT is the most popular instalment of the centralised approach to exposure notification, in which the risk score is computed by a central server. GAEN, in contrast, follows the decentralised approach, where the user's phone calculates the risk. The relative merits of centralised and decentralised systems have proven to be a controversial question. The majority of the previous analyses have focused on the privacy implications of these systems, ours is the first formal analysis to evaluate the security of the deployed systems -- the absence of false risk alerts. We model the French deployment of ROBERT and the most widely deployed GAEN variant, Germany's Corona-Warn-App. We isolate the precise conditions under which these systems prevent false alerts. We determine exactly how an adversary can subvert the system via network and Bluetooth sniffing, database leakage or the compromise of phones, back-end systems and health authorities. We also investigate the security of the original specification of the DP3T protocol, in order to identify gaps between the proposed scheme and its ultimate deployment. We find a total of 27 attack patterns, including many that distinguish the centralised from the decentralised approach, as well as attacks on the authorisation procedure that differentiate all three protocols. Our results suggest that ROBERT's centralised design is more vulnerable against both opportunistic and highly resourced attackers trying to perform mass-notification attacks.



## **45. Spectral Augmentation for Self-Supervised Learning on Graphs**

cs.LG

26 pages, 5 figures, 12 tables

**SubmitDate**: 2022-10-02    [abs](http://arxiv.org/abs/2210.00643v1) [paper-pdf](http://arxiv.org/pdf/2210.00643v1)

**Authors**: Lu Lin, Jinghui Chen, Hongning Wang

**Abstract**: Graph contrastive learning (GCL), as an emerging self-supervised learning technique on graphs, aims to learn representations via instance discrimination. Its performance heavily relies on graph augmentation to reflect invariant patterns that are robust to small perturbations; yet it still remains unclear about what graph invariance GCL should capture. Recent studies mainly perform topology augmentations in a uniformly random manner in the spatial domain, ignoring its influence on the intrinsic structural properties embedded in the spectral domain. In this work, we aim to find a principled way for topology augmentations by exploring the invariance of graphs from the spectral perspective. We develop spectral augmentation which guides topology augmentations by maximizing the spectral change. Extensive experiments on both graph and node classification tasks demonstrate the effectiveness of our method in self-supervised representation learning. The proposed method also brings promising generalization capability in transfer learning, and is equipped with intriguing robustness property under adversarial attacks. Our study sheds light on a general principle for graph topology augmentation.



## **46. Graph Structural Attack by Perturbing Spectral Distance**

cs.LG

Proceedings of the 28th ACM SIGKDD international conference on  knowledge discovery & data mining (KDD'22)

**SubmitDate**: 2022-10-02    [abs](http://arxiv.org/abs/2111.00684v3) [paper-pdf](http://arxiv.org/pdf/2111.00684v3)

**Authors**: Lu Lin, Ethan Blaser, Hongning Wang

**Abstract**: Graph Convolutional Networks (GCNs) have fueled a surge of research interest due to their encouraging performance on graph learning tasks, but they are also shown vulnerability to adversarial attacks. In this paper, an effective graph structural attack is investigated to disrupt graph spectral filters in the Fourier domain, which are the theoretical foundation of GCNs. We define the notion of spectral distance based on the eigenvalues of graph Laplacian to measure the disruption of spectral filters. We realize the attack by maximizing the spectral distance and propose an efficient approximation to reduce the time complexity brought by eigen-decomposition. The experiments demonstrate the remarkable effectiveness of the proposed attack in both black-box and white-box settings for both test-time evasion attacks and training-time poisoning attacks. Our qualitative analysis suggests the connection between the imposed spectral changes in the Fourier domain and the attack behavior in the spatial domain, which provides empirical evidence that maximizing spectral distance is an effective way to change the graph structural property and thus disturb the frequency components for graph filters to affect the learning of GCNs.



## **47. GANTouch: An Attack-Resilient Framework for Touch-based Continuous Authentication System**

cs.CR

11 pages, 7 figures, 2 tables, 3 algorithms, in IEEE TBIOM 2022

**SubmitDate**: 2022-10-02    [abs](http://arxiv.org/abs/2210.01594v1) [paper-pdf](http://arxiv.org/pdf/2210.01594v1)

**Authors**: Mohit Agrawal, Pragyan Mehrotra, Rajesh Kumar, Rajiv Ratn Shah

**Abstract**: Previous studies have shown that commonly studied (vanilla) implementations of touch-based continuous authentication systems (V-TCAS) are susceptible to active adversarial attempts. This study presents a novel Generative Adversarial Network assisted TCAS (G-TCAS) framework and compares it to the V-TCAS under three active adversarial environments viz. Zero-effort, Population, and Random-vector. The Zero-effort environment was implemented in two variations viz. Zero-effort (same-dataset) and Zero-effort (cross-dataset). The first involved a Zero-effort attack from the same dataset, while the second used three different datasets. G-TCAS showed more resilience than V-TCAS under the Population and Random-vector, the more damaging adversarial scenarios than the Zero-effort. On average, the increase in the false accept rates (FARs) for V-TCAS was much higher (27.5% and 21.5%) than for G-TCAS (14% and 12.5%) for Population and Random-vector attacks, respectively. Moreover, we performed a fairness analysis of TCAS for different genders and found TCAS to be fair across genders. The findings suggest that we should evaluate TCAS under active adversarial environments and affirm the usefulness of GANs in the TCAS pipeline.



## **48. Optimization for Robustness Evaluation beyond $\ell_p$ Metrics**

cs.LG

5 pages, 1 figure, 3 tables, submitted to the 2023 IEEE International  Conference on Acoustics, Speech, and Signal Processing (ICASSP 2023) and the  14th International OPT Workshop on Optimization for Machine Learning

**SubmitDate**: 2022-10-02    [abs](http://arxiv.org/abs/2210.00621v1) [paper-pdf](http://arxiv.org/pdf/2210.00621v1)

**Authors**: Hengyue Liang, Buyun Liang, Ying Cui, Tim Mitchell, Ju Sun

**Abstract**: Empirical evaluation of deep learning models against adversarial attacks entails solving nontrivial constrained optimization problems. Popular algorithms for solving these constrained problems rely on projected gradient descent (PGD) and require careful tuning of multiple hyperparameters. Moreover, PGD can only handle $\ell_1$, $\ell_2$, and $\ell_\infty$ attack models due to the use of analytical projectors. In this paper, we introduce a novel algorithmic framework that blends a general-purpose constrained-optimization solver PyGRANSO, With Constraint-Folding (PWCF), to add reliability and generality to robustness evaluation. PWCF 1) finds good-quality solutions without the need of delicate hyperparameter tuning, and 2) can handle general attack models, e.g., general $\ell_p$ ($p \geq 0$) and perceptual attacks, which are inaccessible to PGD-based algorithms.



## **49. iCTGAN--An Attack Mitigation Technique for Random-vector Attack on Accelerometer-based Gait Authentication Systems**

cs.CR

9 pages, 5 figures, IEEE International Joint Conference on Biometrics  (IJCB 2022)

**SubmitDate**: 2022-10-02    [abs](http://arxiv.org/abs/2210.00615v1) [paper-pdf](http://arxiv.org/pdf/2210.00615v1)

**Authors**: Jun Hyung Mo, Rajesh Kumar

**Abstract**: A recent study showed that commonly (vanilla) studied implementations of accelerometer-based gait authentication systems ($v$ABGait) are susceptible to random-vector attack. The same study proposed a beta noise-assisted implementation ($\beta$ABGait) to mitigate the attack. In this paper, we assess the effectiveness of the random-vector attack on both $v$ABGait and $\beta$ABGait using three accelerometer-based gait datasets. In addition, we propose $i$ABGait, an alternative implementation of ABGait, which uses a Conditional Tabular Generative Adversarial Network. Then we evaluate $i$ABGait's resilience against the traditional zero-effort and random-vector attacks. The results show that $i$ABGait mitigates the impact of the random-vector attack to a reasonable extent and outperforms $\beta$ABGait in most experimental settings.



## **50. FoveaTer: Foveated Transformer for Image Classification**

cs.CV

**SubmitDate**: 2022-10-02    [abs](http://arxiv.org/abs/2105.14173v3) [paper-pdf](http://arxiv.org/pdf/2105.14173v3)

**Authors**: Aditya Jonnalagadda, William Yang Wang, B. S. Manjunath, Miguel P. Eckstein

**Abstract**: Many animals and humans process the visual field with a varying spatial resolution (foveated vision) and use peripheral processing to make eye movements and point the fovea to acquire high-resolution information about objects of interest. This architecture results in computationally efficient rapid scene exploration. Recent progress in self-attention-based Vision Transformers, an alternative to the traditionally convolution-reliant computer vision systems. However, the Transformer models do not explicitly model the foveated properties of the visual system nor the interaction between eye movements and the classification task. We propose Foveated Transformer (FoveaTer) model, which uses pooling regions and eye movements to perform object classification tasks using a Vision Transformer architecture. Using square pooling regions or biologically-inspired radial-polar pooling regions, our proposed model pools the image features from the convolution backbone and uses the pooled features as an input to transformer layers. It decides on subsequent fixation location based on the attention assigned by the Transformer to various locations from past and present fixations. It dynamically allocates more fixation/computational resources to more challenging images before making the final image category decision. Using five ablation studies, we evaluate the contribution of different components of the Foveated model. We perform a psychophysics scene categorization task and use the experimental data to find a suitable radial-polar pooling region combination. We also show that the Foveated model better explains the human decisions in a scene categorization task than a Baseline model. We demonstrate our model's robustness against PGD adversarial attacks with both types of pooling regions, where we see the Foveated model outperform the Baseline model.



