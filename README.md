# Latest Adversarial Attack Papers
**update at 2022-08-01 06:31:21**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Look Closer to Your Enemy: Learning to Attack via Teacher-student Mimicking**

cs.CV

13 pages, 8 figures, NDSS

**SubmitDate**: 2022-07-28    [paper-pdf](http://arxiv.org/pdf/2207.13381v2)

**Authors**: Mingejie Wang, Zhiqing Tang, Sirui Li, Dingwen Xiao

**Abstracts**: This paper aims to generate realistic attack samples of person re-identification, ReID, by reading the enemy's mind (VM). In this paper, we propose a novel inconspicuous and controllable ReID attack baseline, LCYE, to generate adversarial query images. Concretely, LCYE first distills VM's knowledge via teacher-student memory mimicking in the proxy task. Then this knowledge prior acts as an explicit cipher conveying what is essential and realistic, believed by VM, for accurate adversarial misleading. Besides, benefiting from the multiple opposing task framework of LCYE, we further investigate the interpretability and generalization of ReID models from the view of the adversarial attack, including cross-domain adaption, cross-model consensus, and online learning process. Extensive experiments on four ReID benchmarks show that our method outperforms other state-of-the-art attackers with a large margin in white-box, black-box, and target attacks. Our code is now available at https://gitfront.io/r/user-3704489/mKXusqDT4ffr/LCYE/.



## **2. Privacy-Preserving Federated Recurrent Neural Networks**

cs.CR

**SubmitDate**: 2022-07-28    [paper-pdf](http://arxiv.org/pdf/2207.13947v1)

**Authors**: Sinem Sav, Abdulrahman Diaa, Apostolos Pyrgelis, Jean-Philippe Bossuat, Jean-Pierre Hubaux

**Abstracts**: We present RHODE, a novel system that enables privacy-preserving training of and prediction on Recurrent Neural Networks (RNNs) in a federated learning setting by relying on multiparty homomorphic encryption (MHE). RHODE preserves the confidentiality of the training data, the model, and the prediction data; and it mitigates the federated learning attacks that target the gradients under a passive-adversary threat model. We propose a novel packing scheme, multi-dimensional packing, for a better utilization of Single Instruction, Multiple Data (SIMD) operations under encryption. With multi-dimensional packing, RHODE enables the efficient processing, in parallel, of a batch of samples. To avoid the exploding gradients problem, we also provide several clip-by-value approximations for enabling gradient clipping under encryption. We experimentally show that the model performance with RHODE remains similar to non-secure solutions both for homogeneous and heterogeneous data distribution among the data holders. Our experimental evaluation shows that RHODE scales linearly with the number of data holders and the number of timesteps, sub-linearly and sub-quadratically with the number of features and the number of hidden units of RNNs, respectively. To the best of our knowledge, RHODE is the first system that provides the building blocks for the training of RNNs and its variants, under encryption in a federated learning setting.



## **3. Label-Only Membership Inference Attack against Node-Level Graph Neural Networks**

cs.CR

**SubmitDate**: 2022-07-27    [paper-pdf](http://arxiv.org/pdf/2207.13766v1)

**Authors**: Mauro Conti, Jiaxin Li, Stjepan Picek, Jing Xu

**Abstracts**: Graph Neural Networks (GNNs), inspired by Convolutional Neural Networks (CNNs), aggregate the message of nodes' neighbors and structure information to acquire expressive representations of nodes for node classification, graph classification, and link prediction. Previous studies have indicated that GNNs are vulnerable to Membership Inference Attacks (MIAs), which infer whether a node is in the training data of GNNs and leak the node's private information, like the patient's disease history. The implementation of previous MIAs takes advantage of the models' probability output, which is infeasible if GNNs only provide the prediction label (label-only) for the input.   In this paper, we propose a label-only MIA against GNNs for node classification with the help of GNNs' flexible prediction mechanism, e.g., obtaining the prediction label of one node even when neighbors' information is unavailable. Our attacking method achieves around 60\% accuracy, precision, and Area Under the Curve (AUC) for most datasets and GNN models, some of which are competitive or even better than state-of-the-art probability-based MIAs implemented under our environment and settings. Additionally, we analyze the influence of the sampling method, model selection approach, and overfitting level on the attack performance of our label-only MIA. Both of those factors have an impact on the attack performance. Then, we consider scenarios where assumptions about the adversary's additional dataset (shadow dataset) and extra information about the target model are relaxed. Even in those scenarios, our label-only MIA achieves a better attack performance in most cases. Finally, we explore the effectiveness of possible defenses, including Dropout, Regularization, Normalization, and Jumping knowledge. None of those four defenses prevent our attack completely.



## **4. SAC-AP: Soft Actor Critic based Deep Reinforcement Learning for Alert Prioritization**

cs.CR

8 pages, 8 figures, IEEE WORLD CONGRESS ON COMPUTATIONAL INTELLIGENCE  2022

**SubmitDate**: 2022-07-27    [paper-pdf](http://arxiv.org/pdf/2207.13666v1)

**Authors**: Lalitha Chavali, Tanay Gupta, Paresh Saxena

**Abstracts**: Intrusion detection systems (IDS) generate a large number of false alerts which makes it difficult to inspect true positives. Hence, alert prioritization plays a crucial role in deciding which alerts to investigate from an enormous number of alerts that are generated by IDS. Recently, deep reinforcement learning (DRL) based deep deterministic policy gradient (DDPG) off-policy method has shown to achieve better results for alert prioritization as compared to other state-of-the-art methods. However, DDPG is prone to the problem of overfitting. Additionally, it also has a poor exploration capability and hence it is not suitable for problems with a stochastic environment. To address these limitations, we present a soft actor-critic based DRL algorithm for alert prioritization (SAC-AP), an off-policy method, based on the maximum entropy reinforcement learning framework that aims to maximize the expected reward while also maximizing the entropy. Further, the interaction between an adversary and a defender is modeled as a zero-sum game and a double oracle framework is utilized to obtain the approximate mixed strategy Nash equilibrium (MSNE). SAC-AP finds robust alert investigation policies and computes pure strategy best response against opponent's mixed strategy. We present the overall design of SAC-AP and evaluate its performance as compared to other state-of-the art alert prioritization methods. We consider defender's loss, i.e., the defender's inability to investigate the alerts that are triggered due to attacks, as the performance metric. Our results show that SAC-AP achieves up to 30% decrease in defender's loss as compared to the DDPG based alert prioritization method and hence provides better protection against intrusions. Moreover, the benefits are even higher when SAC-AP is compared to other traditional alert prioritization methods including Uniform, GAIN, RIO and Suricata.



## **5. Membership Inference Attacks via Adversarial Examples**

cs.LG

**SubmitDate**: 2022-07-27    [paper-pdf](http://arxiv.org/pdf/2207.13572v1)

**Authors**: Hamid Jalalzai, Elie Kadoche, Rémi Leluc, Vincent Plassier

**Abstracts**: The raise of machine learning and deep learning led to significant improvement in several domains. This change is supported by both the dramatic rise in computation power and the collection of large datasets. Such massive datasets often include personal data which can represent a threat to privacy. Membership inference attacks are a novel direction of research which aims at recovering training data used by a learning algorithm. In this paper, we develop a mean to measure the leakage of training data leveraging a quantity appearing as a proxy of the total variation of a trained model near its training samples. We extend our work by providing a novel defense mechanism. Our contributions are supported by empirical evidence through convincing numerical experiments.



## **6. Robust Textual Embedding against Word-level Adversarial Attacks**

cs.CL

Accepted by UAI 2022, code is available at  https://github.com/JHL-HUST/FTML

**SubmitDate**: 2022-07-27    [paper-pdf](http://arxiv.org/pdf/2202.13817v2)

**Authors**: Yichen Yang, Xiaosen Wang, Kun He

**Abstracts**: We attribute the vulnerability of natural language processing models to the fact that similar inputs are converted to dissimilar representations in the embedding space, leading to inconsistent outputs, and we propose a novel robust training method, termed Fast Triplet Metric Learning (FTML). Specifically, we argue that the original sample should have similar representation with its adversarial counterparts and distinguish its representation from other samples for better robustness. To this end, we adopt the triplet metric learning into the standard training to pull words closer to their positive samples (i.e., synonyms) and push away their negative samples (i.e., non-synonyms) in the embedding space. Extensive experiments demonstrate that FTML can significantly promote the model robustness against various advanced adversarial attacks while keeping competitive classification accuracy on original samples. Besides, our method is efficient as it only needs to adjust the embedding and introduces very little overhead on the standard training. Our work shows great potential of improving the textual robustness through robust word embedding.



## **7. Improved and Interpretable Defense to Transferred Adversarial Examples by Jacobian Norm with Selective Input Gradient Regularization**

cs.LG

Under review

**SubmitDate**: 2022-07-27    [paper-pdf](http://arxiv.org/pdf/2207.13036v2)

**Authors**: Deyin Liu, Lin Wu, Farid Boussaid, Mohammed Bennamoun

**Abstracts**: Deep neural networks (DNNs) are known to be vulnerable to adversarial examples that are crafted with imperceptible perturbations, i.e., a small change in an input image can induce a mis-classification, and thus threatens the reliability of deep learning based deployment systems. Adversarial training (AT) is often adopted to improve the robustness of DNNs through training a mixture of corrupted and clean data. However, most of AT based methods are ineffective in dealing with \textit{transferred adversarial examples} which are generated to fool a wide spectrum of defense models, and thus cannot satisfy the generalization requirement raised in real-world scenarios. Moreover, adversarially training a defense model in general cannot produce interpretable predictions towards the inputs with perturbations, whilst a highly interpretable robust model is required by different domain experts to understand the behaviour of a DNN. In this work, we propose an approach based on Jacobian norm and Selective Input Gradient Regularization (J-SIGR), which suggests the linearized robustness through Jacobian normalization and also regularizes the perturbation-based saliency maps to imitate the model's interpretable predictions. As such, we achieve both the improved defense and high interpretability of DNNs. Finally, we evaluate our method across different architectures against powerful adversarial attacks. Experiments demonstrate that the proposed J-SIGR confers improved robustness against transferred adversarial attacks, and we also show that the predictions from the neural network are easy to interpret.



## **8. Point Cloud Attacks in Graph Spectral Domain: When 3D Geometry Meets Graph Signal Processing**

cs.CV

arXiv admin note: substantial text overlap with arXiv:2202.07261

**SubmitDate**: 2022-07-27    [paper-pdf](http://arxiv.org/pdf/2207.13326v1)

**Authors**: Daizong Liu, Wei Hu, Xin Li

**Abstracts**: With the increasing attention in various 3D safety-critical applications, point cloud learning models have been shown to be vulnerable to adversarial attacks. Although existing 3D attack methods achieve high success rates, they delve into the data space with point-wise perturbation, which may neglect the geometric characteristics. Instead, we propose point cloud attacks from a new perspective -- the graph spectral domain attack, aiming to perturb graph transform coefficients in the spectral domain that corresponds to varying certain geometric structure. Specifically, leveraging on graph signal processing, we first adaptively transform the coordinates of points onto the spectral domain via graph Fourier transform (GFT) for compact representation. Then, we analyze the influence of different spectral bands on the geometric structure, based on which we propose to perturb the GFT coefficients via a learnable graph spectral filter. Considering the low-frequency components mainly contribute to the rough shape of the 3D object, we further introduce a low-frequency constraint to limit perturbations within imperceptible high-frequency components. Finally, the adversarial point cloud is generated by transforming the perturbed spectral representation back to the data domain via the inverse GFT. Experimental results demonstrate the effectiveness of the proposed attack in terms of both the imperceptibility and attack success rates.



## **9. Perception-Aware Attack: Creating Adversarial Music via Reverse-Engineering Human Perception**

cs.SD

ACM CCS 2022

**SubmitDate**: 2022-07-26    [paper-pdf](http://arxiv.org/pdf/2207.13192v1)

**Authors**: Rui Duan, Zhe Qu, Shangqing Zhao, Leah Ding, Yao Liu, Zhuo Lu

**Abstracts**: Recently, adversarial machine learning attacks have posed serious security threats against practical audio signal classification systems, including speech recognition, speaker recognition, and music copyright detection. Previous studies have mainly focused on ensuring the effectiveness of attacking an audio signal classifier via creating a small noise-like perturbation on the original signal. It is still unclear if an attacker is able to create audio signal perturbations that can be well perceived by human beings in addition to its attack effectiveness. This is particularly important for music signals as they are carefully crafted with human-enjoyable audio characteristics.   In this work, we formulate the adversarial attack against music signals as a new perception-aware attack framework, which integrates human study into adversarial attack design. Specifically, we conduct a human study to quantify the human perception with respect to a change of a music signal. We invite human participants to rate their perceived deviation based on pairs of original and perturbed music signals, and reverse-engineer the human perception process by regression analysis to predict the human-perceived deviation given a perturbed signal. The perception-aware attack is then formulated as an optimization problem that finds an optimal perturbation signal to minimize the prediction of perceived deviation from the regressed human perception model. We use the perception-aware framework to design a realistic adversarial music attack against YouTube's copyright detector. Experiments show that the perception-aware attack produces adversarial music with significantly better perceptual quality than prior work.



## **10. FlashSyn: Flash Loan Attack Synthesis via Counter Example Driven Approximation**

cs.PL

29 pages, 8 figures, technical report

**SubmitDate**: 2022-07-26    [paper-pdf](http://arxiv.org/pdf/2206.10708v2)

**Authors**: Zhiyang Chen, Sidi Mohamed Beillahi, Fan Long

**Abstracts**: In decentralized finance (DeFi) ecosystem, lenders can offer flash loans to borrowers, i.e., loans that are only valid within a blockchain transaction and must be repaid with some fees by the end of that transaction. Unlike normal loans, flash loans allow borrowers to borrow a large amount of assets without upfront collaterals deposits. Malicious adversaries can use flash loans to gather large amount of assets to launch costly exploitations targeting DeFi protocols. In this paper, we introduce a new framework for automated synthesis of adversarial contracts that exploit DeFi protocols using flash loans. To bypass the complexity of a DeFi protocol, we propose a new technique to approximate the DeFi protocol functional behaviors using numerical methods (polynomial linear regression and nearest-neighbor interpolation). We then construct an optimization query using the approximated functions of the DeFi protocol to find an adversarial attack constituted of a sequence of functions invocations with optimal parameters that gives the maximum profit. To improve the accuracy of the approximation, we propose a new counterexamples-driven approximation refinement technique. We implement our framework in a tool called FlashSyn. We evaluate FlashSyn on 12 DeFi protocols that were victims to flash loan attacks and DeFi protocols from Damn Vulnerable DeFi challenges. FlashSyn automatically synthesizes an adversarial attack for each one of them.



## **11. Exploring the Unprecedented Privacy Risks of the Metaverse**

cs.CR

**SubmitDate**: 2022-07-26    [paper-pdf](http://arxiv.org/pdf/2207.13176v1)

**Authors**: Vivek Nair, Gonzalo Munilla Garrido, Dawn Song

**Abstracts**: Thirty study participants playtested an innocent-looking "escape room" game in virtual reality (VR). Behind the scenes, an adversarial program had accurately inferred over 25 personal data attributes, from anthropometrics like height and wingspan to demographics like age and gender, within just a few minutes of gameplay. As notoriously data-hungry companies become increasingly involved in VR development, this experimental scenario may soon represent a typical VR user experience. While virtual telepresence applications (and the so-called "metaverse") have recently received increased attention and investment from major tech firms, these environments remain relatively under-studied from a security and privacy standpoint. In this work, we illustrate how VR attackers can covertly ascertain dozens of personal data attributes from seemingly-anonymous users of popular metaverse applications like VRChat. These attackers can be as simple as other VR users without special privilege, and the potential scale and scope of this data collection far exceed what is feasible within traditional mobile and web applications. We aim to shed light on the unique privacy risks of the metaverse, and provide the first holistic framework for understanding intrusive data harvesting attacks in these emerging VR ecosystems.



## **12. LGV: Boosting Adversarial Example Transferability from Large Geometric Vicinity**

cs.LG

Accepted at ECCV 2022

**SubmitDate**: 2022-07-26    [paper-pdf](http://arxiv.org/pdf/2207.13129v1)

**Authors**: Martin Gubri, Maxime Cordy, Mike Papadakis, Yves Le Traon, Koushik Sen

**Abstracts**: We propose transferability from Large Geometric Vicinity (LGV), a new technique to increase the transferability of black-box adversarial attacks. LGV starts from a pretrained surrogate model and collects multiple weight sets from a few additional training epochs with a constant and high learning rate. LGV exploits two geometric properties that we relate to transferability. First, models that belong to a wider weight optimum are better surrogates. Second, we identify a subspace able to generate an effective surrogate ensemble among this wider optimum. Through extensive experiments, we show that LGV alone outperforms all (combinations of) four established test-time transformations by 1.8 to 59.9 percentage points. Our findings shed new light on the importance of the geometry of the weight space to explain the transferability of adversarial examples.



## **13. Making Corgis Important for Honeycomb Classification: Adversarial Attacks on Concept-based Explainability Tools**

cs.LG

AdvML Frontiers 2022 @ ICML 2022 workshop

**SubmitDate**: 2022-07-26    [paper-pdf](http://arxiv.org/pdf/2110.07120v2)

**Authors**: Davis Brown, Henry Kvinge

**Abstracts**: Methods for model explainability have become increasingly critical for testing the fairness and soundness of deep learning. Concept-based interpretability techniques, which use a small set of human-interpretable concept exemplars in order to measure the influence of a concept on a model's internal representation of input, are an important thread in this line of research. In this work we show that these explainability methods can suffer the same vulnerability to adversarial attacks as the models they are meant to analyze. We demonstrate this phenomenon on two well-known concept-based interpretability methods: TCAV and faceted feature visualization. We show that by carefully perturbing the examples of the concept that is being investigated, we can radically change the output of the interpretability method. The attacks that we propose can either induce positive interpretations (polka dots are an important concept for a model when classifying zebras) or negative interpretations (stripes are not an important factor in identifying images of a zebra). Our work highlights the fact that in safety-critical applications, there is need for security around not only the machine learning pipeline but also the model interpretation process.



## **14. TnT Attacks! Universal Naturalistic Adversarial Patches Against Deep Neural Network Systems**

cs.CV

Accepted for publication in the IEEE Transactions on Information  Forensics & Security (TIFS)

**SubmitDate**: 2022-07-26    [paper-pdf](http://arxiv.org/pdf/2111.09999v2)

**Authors**: Bao Gia Doan, Minhui Xue, Shiqing Ma, Ehsan Abbasnejad, Damith C. Ranasinghe

**Abstracts**: Deep neural networks are vulnerable to attacks from adversarial inputs and, more recently, Trojans to misguide or hijack the model's decision. We expose the existence of an intriguing class of spatially bounded, physically realizable, adversarial examples -- Universal NaTuralistic adversarial paTches -- we call TnTs, by exploring the superset of the spatially bounded adversarial example space and the natural input space within generative adversarial networks. Now, an adversary can arm themselves with a patch that is naturalistic, less malicious-looking, physically realizable, highly effective achieving high attack success rates, and universal. A TnT is universal because any input image captured with a TnT in the scene will: i) misguide a network (untargeted attack); or ii) force the network to make a malicious decision (targeted attack). Interestingly, now, an adversarial patch attacker has the potential to exert a greater level of control -- the ability to choose a location-independent, natural-looking patch as a trigger in contrast to being constrained to noisy perturbations -- an ability is thus far shown to be only possible with Trojan attack methods needing to interfere with the model building processes to embed a backdoor at the risk discovery; but, still realize a patch deployable in the physical world. Through extensive experiments on the large-scale visual classification task, ImageNet with evaluations across its entire validation set of 50,000 images, we demonstrate the realistic threat from TnTs and the robustness of the attack. We show a generalization of the attack to create patches achieving higher attack success rates than existing state-of-the-art methods. Our results show the generalizability of the attack to different visual classification tasks (CIFAR-10, GTSRB, PubFig) and multiple state-of-the-art deep neural networks such as WideResnet50, Inception-V3 and VGG-16.



## **15. Verification-Aided Deep Ensemble Selection**

cs.LG

To appear in FMCAD 2022

**SubmitDate**: 2022-07-25    [paper-pdf](http://arxiv.org/pdf/2202.03898v2)

**Authors**: Guy Amir, Tom Zelazny, Guy Katz, Michael Schapira

**Abstracts**: Deep neural networks (DNNs) have become the technology of choice for realizing a variety of complex tasks. However, as highlighted by many recent studies, even an imperceptible perturbation to a correctly classified input can lead to misclassification by a DNN. This renders DNNs vulnerable to strategic input manipulations by attackers, and also oversensitive to environmental noise.   To mitigate this phenomenon, practitioners apply joint classification by an *ensemble* of DNNs. By aggregating the classification outputs of different individual DNNs for the same input, ensemble-based classification reduces the risk of misclassifications due to the specific realization of the stochastic training process of any single DNN. However, the effectiveness of a DNN ensemble is highly dependent on its members *not simultaneously erring* on many different inputs.   In this case study, we harness recent advances in DNN verification to devise a methodology for identifying ensemble compositions that are less prone to simultaneous errors, even when the input is adversarially perturbed -- resulting in more robustly-accurate ensemble-based classification.   Our proposed framework uses a DNN verifier as a backend, and includes heuristics that help reduce the high complexity of directly verifying ensembles. More broadly, our work puts forth a novel universal objective for formal verification that can potentially improve the robustness of real-world, deep-learning-based systems across a variety of application domains.



## **16. $p$-DkNN: Out-of-Distribution Detection Through Statistical Testing of Deep Representations**

cs.LG

**SubmitDate**: 2022-07-25    [paper-pdf](http://arxiv.org/pdf/2207.12545v1)

**Authors**: Adam Dziedzic, Stephan Rabanser, Mohammad Yaghini, Armin Ale, Murat A. Erdogdu, Nicolas Papernot

**Abstracts**: The lack of well-calibrated confidence estimates makes neural networks inadequate in safety-critical domains such as autonomous driving or healthcare. In these settings, having the ability to abstain from making a prediction on out-of-distribution (OOD) data can be as important as correctly classifying in-distribution data. We introduce $p$-DkNN, a novel inference procedure that takes a trained deep neural network and analyzes the similarity structures of its intermediate hidden representations to compute $p$-values associated with the end-to-end model prediction. The intuition is that statistical tests performed on latent representations can serve not only as a classifier, but also offer a statistically well-founded estimation of uncertainty. $p$-DkNN is scalable and leverages the composition of representations learned by hidden layers, which makes deep representation learning successful. Our theoretical analysis builds on Neyman-Pearson classification and connects it to recent advances in selective classification (reject option). We demonstrate advantageous trade-offs between abstaining from predicting on OOD inputs and maintaining high accuracy on in-distribution inputs. We find that $p$-DkNN forces adaptive attackers crafting adversarial examples, a form of worst-case OOD inputs, to introduce semantically meaningful changes to the inputs.



## **17. TAFIM: Targeted Adversarial Attacks against Facial Image Manipulations**

cs.CV

(ECCV 2022 Paper) Video: https://youtu.be/11VMOJI7tKg Project Page:  https://shivangi-aneja.github.io/projects/tafim/

**SubmitDate**: 2022-07-25    [paper-pdf](http://arxiv.org/pdf/2112.09151v2)

**Authors**: Shivangi Aneja, Lev Markhasin, Matthias Niessner

**Abstracts**: Face manipulation methods can be misused to affect an individual's privacy or to spread disinformation. To this end, we introduce a novel data-driven approach that produces image-specific perturbations which are embedded in the original images. The key idea is that these protected images prevent face manipulation by causing the manipulation model to produce a predefined manipulation target (uniformly colored output image in our case) instead of the actual manipulation. In addition, we propose to leverage differentiable compression approximation, hence making generated perturbations robust to common image compression. In order to prevent against multiple manipulation methods simultaneously, we further propose a novel attention-based fusion of manipulation-specific perturbations. Compared to traditional adversarial attacks that optimize noise patterns for each image individually, our generalized model only needs a single forward pass, thus running orders of magnitude faster and allowing for easy integration in image processing stacks, even on resource-constrained devices like smartphones.



## **18. SegPGD: An Effective and Efficient Adversarial Attack for Evaluating and Boosting Segmentation Robustness**

cs.CV

**SubmitDate**: 2022-07-25    [paper-pdf](http://arxiv.org/pdf/2207.12391v1)

**Authors**: Jindong Gu, Hengshuang Zhao, Volker Tresp, Philip Torr

**Abstracts**: Deep neural network-based image classifications are vulnerable to adversarial perturbations. The image classifications can be easily fooled by adding artificial small and imperceptible perturbations to input images. As one of the most effective defense strategies, adversarial training was proposed to address the vulnerability of classification models, where the adversarial examples are created and injected into training data during training. The attack and defense of classification models have been intensively studied in past years. Semantic segmentation, as an extension of classifications, has also received great attention recently. Recent work shows a large number of attack iterations are required to create effective adversarial examples to fool segmentation models. The observation makes both robustness evaluation and adversarial training on segmentation models challenging. In this work, we propose an effective and efficient segmentation attack method, dubbed SegPGD. Besides, we provide a convergence analysis to show the proposed SegPGD can create more effective adversarial examples than PGD under the same number of attack iterations. Furthermore, we propose to apply our SegPGD as the underlying attack method for segmentation adversarial training. Since SegPGD can create more effective adversarial examples, the adversarial training with our SegPGD can boost the robustness of segmentation models. Our proposals are also verified with experiments on popular Segmentation model architectures and standard segmentation datasets.



## **19. Adversarial Attack across Datasets**

cs.CV

**SubmitDate**: 2022-07-25    [paper-pdf](http://arxiv.org/pdf/2110.07718v2)

**Authors**: Yunxiao Qin, Yuanhao Xiong, Jinfeng Yi, Lihong Cao, Cho-Jui Hsieh

**Abstracts**: Existing transfer attack methods commonly assume that the attacker knows the training set (e.g., the label set, the input size) of the black-box victim models, which is usually unrealistic because in some cases the attacker cannot know this information. In this paper, we define a Generalized Transferable Attack (GTA) problem where the attacker doesn't know this information and is acquired to attack any randomly encountered images that may come from unknown datasets. To solve the GTA problem, we propose a novel Image Classification Eraser (ICE) that trains a particular attacker to erase classification information of any images from arbitrary datasets. Experiments on several datasets demonstrate that ICE greatly outperforms existing transfer attacks on GTA, and show that ICE uses similar texture-like noises to perturb different images from different datasets. Moreover, fast fourier transformation analysis indicates that the main components in each ICE noise are three sine waves for the R, G, and B image channels. Inspired by this interesting finding, we then design a novel Sine Attack (SA) method to optimize the three sine waves. Experiments show that SA performs comparably to ICE, indicating that the three sine waves are effective and enough to break DNNs under the GTA setting.



## **20. Improving Adversarial Robustness via Mutual Information Estimation**

cs.LG

This version has modified Eq.2 and its proof in the published version

**SubmitDate**: 2022-07-25    [paper-pdf](http://arxiv.org/pdf/2207.12203v1)

**Authors**: Dawei Zhou, Nannan Wang, Xinbo Gao, Bo Han, Xiaoyu Wang, Yibing Zhan, Tongliang Liu

**Abstracts**: Deep neural networks (DNNs) are found to be vulnerable to adversarial noise. They are typically misled by adversarial samples to make wrong predictions. To alleviate this negative effect, in this paper, we investigate the dependence between outputs of the target model and input adversarial samples from the perspective of information theory, and propose an adversarial defense method. Specifically, we first measure the dependence by estimating the mutual information (MI) between outputs and the natural patterns of inputs (called natural MI) and MI between outputs and the adversarial patterns of inputs (called adversarial MI), respectively. We find that adversarial samples usually have larger adversarial MI and smaller natural MI compared with those w.r.t. natural samples. Motivated by this observation, we propose to enhance the adversarial robustness by maximizing the natural MI and minimizing the adversarial MI during the training process. In this way, the target model is expected to pay more attention to the natural pattern that contains objective semantics. Empirical evaluations demonstrate that our method could effectively improve the adversarial accuracy against multiple attacks.



## **21. Versatile Weight Attack via Flipping Limited Bits**

cs.CR

Extension of our ICLR 2021 work: arXiv:2102.10496

**SubmitDate**: 2022-07-25    [paper-pdf](http://arxiv.org/pdf/2207.12405v1)

**Authors**: Jiawang Bai, Baoyuan Wu, Zhifeng Li, Shu-tao Xia

**Abstracts**: To explore the vulnerability of deep neural networks (DNNs), many attack paradigms have been well studied, such as the poisoning-based backdoor attack in the training stage and the adversarial attack in the inference stage. In this paper, we study a novel attack paradigm, which modifies model parameters in the deployment stage. Considering the effectiveness and stealthiness goals, we provide a general formulation to perform the bit-flip based weight attack, where the effectiveness term could be customized depending on the attacker's purpose. Furthermore, we present two cases of the general formulation with different malicious purposes, i.e., single sample attack (SSA) and triggered samples attack (TSA). To this end, we formulate this problem as a mixed integer programming (MIP) to jointly determine the state of the binary bits (0 or 1) in the memory and learn the sample modification. Utilizing the latest technique in integer programming, we equivalently reformulate this MIP problem as a continuous optimization problem, which can be effectively and efficiently solved using the alternating direction method of multipliers (ADMM) method. Consequently, the flipped critical bits can be easily determined through optimization, rather than using a heuristic strategy. Extensive experiments demonstrate the superiority of SSA and TSA in attacking DNNs.



## **22. Privacy Against Inference Attacks in Vertical Federated Learning**

cs.LG

**SubmitDate**: 2022-07-24    [paper-pdf](http://arxiv.org/pdf/2207.11788v1)

**Authors**: Borzoo Rassouli, Morteza Varasteh, Deniz Gunduz

**Abstracts**: Vertical federated learning is considered, where an active party, having access to true class labels, wishes to build a classification model by utilizing more features from a passive party, which has no access to the labels, to improve the model accuracy. In the prediction phase, with logistic regression as the classification model, several inference attack techniques are proposed that the adversary, i.e., the active party, can employ to reconstruct the passive party's features, regarded as sensitive information. These attacks, which are mainly based on a classical notion of the center of a set, i.e., the Chebyshev center, are shown to be superior to those proposed in the literature. Moreover, several theoretical performance guarantees are provided for the aforementioned attacks. Subsequently, we consider the minimum amount of information that the adversary needs to fully reconstruct the passive party's features. In particular, it is shown that when the passive party holds one feature, and the adversary is only aware of the signs of the parameters involved, it can perfectly reconstruct that feature when the number of predictions is large enough. Next, as a defense mechanism, two privacy-preserving schemes are proposed that worsen the adversary's reconstruction attacks, while preserving the full benefits that VFL brings to the active party. Finally, experimental results demonstrate the effectiveness of the proposed attacks and the privacy-preserving schemes.



## **23. Can we achieve robustness from data alone?**

cs.LG

**SubmitDate**: 2022-07-24    [paper-pdf](http://arxiv.org/pdf/2207.11727v1)

**Authors**: Nikolaos Tsilivis, Jingtong Su, Julia Kempe

**Abstracts**: Adversarial training and its variants have come to be the prevailing methods to achieve adversarially robust classification using neural networks. However, its increased computational cost together with the significant gap between standard and robust performance hinder progress and beg the question of whether we can do better. In this work, we take a step back and ask: Can models achieve robustness via standard training on a suitably optimized set? To this end, we devise a meta-learning method for robust classification, that optimizes the dataset prior to its deployment in a principled way, and aims to effectively remove the non-robust parts of the data. We cast our optimization method as a multi-step PGD procedure on kernel regression, with a class of kernels that describe infinitely wide neural nets (Neural Tangent Kernels - NTKs). Experiments on MNIST and CIFAR-10 demonstrate that the datasets we produce enjoy very high robustness against PGD attacks, when deployed in both kernel regression classifiers and neural networks. However, this robustness is somewhat fallacious, as alternative attacks manage to fool the models, which we find to be the case for previous similar works in the literature as well. We discuss potential reasons for this and outline further avenues of research.



## **24. Proving Common Mechanisms Shared by Twelve Methods of Boosting Adversarial Transferability**

cs.LG

**SubmitDate**: 2022-07-24    [paper-pdf](http://arxiv.org/pdf/2207.11694v1)

**Authors**: Quanshi Zhang, Xin Wang, Jie Ren, Xu Cheng, Shuyun Lin, Yisen Wang, Xiangming Zhu

**Abstracts**: Although many methods have been proposed to enhance the transferability of adversarial perturbations, these methods are designed in a heuristic manner, and the essential mechanism for improving adversarial transferability is still unclear. This paper summarizes the common mechanism shared by twelve previous transferability-boosting methods in a unified view, i.e., these methods all reduce game-theoretic interactions between regional adversarial perturbations. To this end, we focus on the attacking utility of all interactions between regional adversarial perturbations, and we first discover and prove the negative correlation between the adversarial transferability and the attacking utility of interactions. Based on this discovery, we theoretically prove and empirically verify that twelve previous transferability-boosting methods all reduce interactions between regional adversarial perturbations. More crucially, we consider the reduction of interactions as the essential reason for the enhancement of adversarial transferability. Furthermore, we design the interaction loss to directly penalize interactions between regional adversarial perturbations during attacking. Experimental results show that the interaction loss significantly improves the transferability of adversarial perturbations.



## **25. Testing the Robustness of Learned Index Structures**

cs.DB

**SubmitDate**: 2022-07-23    [paper-pdf](http://arxiv.org/pdf/2207.11575v1)

**Authors**: Matthias Bachfischer, Renata Borovica-Gajic, Benjamin I. P. Rubinstein

**Abstracts**: While early empirical evidence has supported the case for learned index structures as having favourable average-case performance, little is known about their worst-case performance. By contrast, classical structures are known to achieve optimal worst-case behaviour. This work evaluates the robustness of learned index structures in the presence of adversarial workloads. To simulate adversarial workloads, we carry out a data poisoning attack on linear regression models that manipulates the cumulative distribution function (CDF) on which the learned index model is trained. The attack deteriorates the fit of the underlying ML model by injecting a set of poisoning keys into the training dataset, which leads to an increase in the prediction error of the model and thus deteriorates the overall performance of the learned index structure. We assess the performance of various regression methods and the learned index implementations ALEX and PGM-Index. We show that learned index structures can suffer from a significant performance deterioration of up to 20% when evaluated on poisoned vs. non-poisoned datasets.



## **26. How does Heterophily Impact the Robustness of Graph Neural Networks? Theoretical Connections and Practical Implications**

cs.LG

KDD 2022 camera ready version + full appendix; 20 pages, 2 figures

**SubmitDate**: 2022-07-23    [paper-pdf](http://arxiv.org/pdf/2106.07767v4)

**Authors**: Jiong Zhu, Junchen Jin, Donald Loveland, Michael T. Schaub, Danai Koutra

**Abstracts**: We bridge two research directions on graph neural networks (GNNs), by formalizing the relation between heterophily of node labels (i.e., connected nodes tend to have dissimilar labels) and the robustness of GNNs to adversarial attacks. Our theoretical and empirical analyses show that for homophilous graph data, impactful structural attacks always lead to reduced homophily, while for heterophilous graph data the change in the homophily level depends on the node degrees. These insights have practical implications for defending against attacks on real-world graphs: we deduce that separate aggregators for ego- and neighbor-embeddings, a design principle which has been identified to significantly improve prediction for heterophilous graph data, can also offer increased robustness to GNNs. Our comprehensive experiments show that GNNs merely adopting this design achieve improved empirical and certifiable robustness compared to the best-performing unvaccinated model. Additionally, combining this design with explicit defense mechanisms against adversarial attacks leads to an improved robustness with up to 18.33% performance increase under attacks compared to the best-performing vaccinated model.



## **27. Do Perceptually Aligned Gradients Imply Adversarial Robustness?**

cs.CV

**SubmitDate**: 2022-07-22    [paper-pdf](http://arxiv.org/pdf/2207.11378v1)

**Authors**: Roy Ganz, Bahjat Kawar, Michael Elad

**Abstracts**: In the past decade, deep learning-based networks have achieved unprecedented success in numerous tasks, including image classification. Despite this remarkable achievement, recent studies have demonstrated that such networks are easily fooled by small malicious perturbations, also known as adversarial examples. This security weakness led to extensive research aimed at obtaining robust models. Beyond the clear robustness benefits of such models, it was also observed that their gradients with respect to the input align with human perception. Several works have identified Perceptually Aligned Gradients (PAG) as a byproduct of robust training, but none have considered it as a standalone phenomenon nor studied its own implications. In this work, we focus on this trait and test whether Perceptually Aligned Gradients imply Robustness. To this end, we develop a novel objective to directly promote PAG in training classifiers and examine whether models with such gradients are more robust to adversarial attacks. Extensive experiments on CIFAR-10 and STL validate that such models have improved robust performance, exposing the surprising bidirectional connection between PAG and robustness.



## **28. Practical Privacy Attacks on Vertical Federated Learning**

cs.CR

**SubmitDate**: 2022-07-22    [paper-pdf](http://arxiv.org/pdf/2011.09290v3)

**Authors**: Haiqin Weng, Juntao Zhang, Xingjun Ma, Feng Xue, Tao Wei, Shouling Ji, Zhiyuan Zong

**Abstracts**: Federated learning (FL) is a privacy-preserving learning paradigm that allows multiple parities to jointly train a powerful machine learning model without sharing their private data. According to the form of collaboration, FL can be further divided into horizontal federated learning (HFL) and vertical federated learning (VFL). In HFL, participants share the same feature space and collaborate on data samples, while in VFL, participants share the same sample IDs and collaborate on features. VFL has a broader scope of applications and is arguably more suitable for joint model training between large enterprises.   In this paper, we focus on VFL and investigate potential privacy leakage in real-world VFL frameworks. We design and implement two practical privacy attacks: reverse multiplication attack for the logistic regression VFL protocol; and reverse sum attack for the XGBoost VFL protocol. We empirically show that the two attacks are (1) effective - the adversary can successfully steal the private training data, even when the intermediate outputs are encrypted to protect data privacy; (2) evasive - the attacks do not deviate from the protocol specification nor deteriorate the accuracy of the target model; and (3) easy - the adversary needs little prior knowledge about the data distribution of the target participant. We also show the leaked information is as effective as the raw training data in training an alternative classifier. We further discuss potential countermeasures and their challenges, which we hope can lead to several promising research directions.



## **29. On Higher Adversarial Susceptibility of Contrastive Self-Supervised Learning**

cs.CV

**SubmitDate**: 2022-07-22    [paper-pdf](http://arxiv.org/pdf/2207.10862v1)

**Authors**: Rohit Gupta, Naveed Akhtar, Ajmal Mian, Mubarak Shah

**Abstracts**: Contrastive self-supervised learning (CSL) has managed to match or surpass the performance of supervised learning in image and video classification. However, it is still largely unknown if the nature of the representation induced by the two learning paradigms is similar. We investigate this under the lens of adversarial robustness. Our analytical treatment of the problem reveals intrinsic higher sensitivity of CSL over supervised learning. It identifies the uniform distribution of data representation over a unit hypersphere in the CSL representation space as the key contributor to this phenomenon. We establish that this increases model sensitivity to input perturbations in the presence of false negatives in the training data. Our finding is supported by extensive experiments for image and video classification using adversarial perturbations and other input corruptions. Building on the insights, we devise strategies that are simple, yet effective in improving model robustness with CSL training. We demonstrate up to 68% reduction in the performance gap between adversarially attacked CSL and its supervised counterpart. Finally, we contribute to robust CSL paradigm by incorporating our findings in adversarial self-supervised learning. We demonstrate an average gain of about 5% over two different state-of-the-art methods in this domain.



## **30. Adversarially-Aware Robust Object Detector**

cs.CV

ECCV2022 oral paper

**SubmitDate**: 2022-07-22    [paper-pdf](http://arxiv.org/pdf/2207.06202v3)

**Authors**: Ziyi Dong, Pengxu Wei, Liang Lin

**Abstracts**: Object detection, as a fundamental computer vision task, has achieved a remarkable progress with the emergence of deep neural networks. Nevertheless, few works explore the adversarial robustness of object detectors to resist adversarial attacks for practical applications in various real-world scenarios. Detectors have been greatly challenged by unnoticeable perturbation, with sharp performance drop on clean images and extremely poor performance on adversarial images. In this work, we empirically explore the model training for adversarial robustness in object detection, which greatly attributes to the conflict between learning clean images and adversarial images. To mitigate this issue, we propose a Robust Detector (RobustDet) based on adversarially-aware convolution to disentangle gradients for model learning on clean and adversarial images. RobustDet also employs the Adversarial Image Discriminator (AID) and Consistent Features with Reconstruction (CFR) to ensure a reliable robustness. Extensive experiments on PASCAL VOC and MS-COCO demonstrate that our model effectively disentangles gradients and significantly enhances the detection robustness with maintaining the detection ability on clean images.



## **31. Boosting Transferability of Targeted Adversarial Examples via Hierarchical Generative Networks**

cs.LG

**SubmitDate**: 2022-07-22    [paper-pdf](http://arxiv.org/pdf/2107.01809v2)

**Authors**: Xiao Yang, Yinpeng Dong, Tianyu Pang, Hang Su, Jun Zhu

**Abstracts**: Transfer-based adversarial attacks can evaluate model robustness in the black-box setting. Several methods have demonstrated impressive untargeted transferability, however, it is still challenging to efficiently produce targeted transferability. To this end, we develop a simple yet effective framework to craft targeted transfer-based adversarial examples, applying a hierarchical generative network. In particular, we contribute to amortized designs that well adapt to multi-class targeted attacks. Extensive experiments on ImageNet show that our method improves the success rates of targeted black-box attacks by a significant margin over the existing methods -- it reaches an average success rate of 29.1\% against six diverse models based only on one substitute white-box model, which significantly outperforms the state-of-the-art gradient-based attack methods. Moreover, the proposed method is also more efficient beyond an order of magnitude than gradient-based methods.



## **32. Synthetic Dataset Generation for Adversarial Machine Learning Research**

cs.CV

**SubmitDate**: 2022-07-21    [paper-pdf](http://arxiv.org/pdf/2207.10719v1)

**Authors**: Xiruo Liu, Shibani Singh, Cory Cornelius, Colin Busho, Mike Tan, Anindya Paul, Jason Martin

**Abstracts**: Existing adversarial example research focuses on digitally inserted perturbations on top of existing natural image datasets. This construction of adversarial examples is not realistic because it may be difficult, or even impossible, for an attacker to deploy such an attack in the real-world due to sensing and environmental effects. To better understand adversarial examples against cyber-physical systems, we propose approximating the real-world through simulation. In this paper we describe our synthetic dataset generation tool that enables scalable collection of such a synthetic dataset with realistic adversarial examples. We use the CARLA simulator to collect such a dataset and demonstrate simulated attacks that undergo the same environmental transforms and processing as real-world images. Our tools have been used to collect datasets to help evaluate the efficacy of adversarial examples, and can be found at https://github.com/carla-simulator/carla/pull/4992.



## **33. Careful What You Wish For: on the Extraction of Adversarially Trained Models**

cs.LG

To be published in the proceedings of the 19th Annual International  Conference on Privacy, Security & Trust (PST 2022). The conference  proceedings will be included in IEEE Xplore as in previous editions of the  conference

**SubmitDate**: 2022-07-21    [paper-pdf](http://arxiv.org/pdf/2207.10561v1)

**Authors**: Kacem Khaled, Gabriela Nicolescu, Felipe Gohring de Magalhães

**Abstracts**: Recent attacks on Machine Learning (ML) models such as evasion attacks with adversarial examples and models stealing through extraction attacks pose several security and privacy threats. Prior work proposes to use adversarial training to secure models from adversarial examples that can evade the classification of a model and deteriorate its performance. However, this protection technique affects the model's decision boundary and its prediction probabilities, hence it might raise model privacy risks. In fact, a malicious user using only a query access to the prediction output of a model can extract it and obtain a high-accuracy and high-fidelity surrogate model. To have a greater extraction, these attacks leverage the prediction probabilities of the victim model. Indeed, all previous work on extraction attacks do not take into consideration the changes in the training process for security purposes. In this paper, we propose a framework to assess extraction attacks on adversarially trained models with vision datasets. To the best of our knowledge, our work is the first to perform such evaluation. Through an extensive empirical study, we demonstrate that adversarially trained models are more vulnerable to extraction attacks than models obtained under natural training circumstances. They can achieve up to $\times1.2$ higher accuracy and agreement with a fraction lower than $\times0.75$ of the queries. We additionally find that the adversarial robustness capability is transferable through extraction attacks, i.e., extracted Deep Neural Networks (DNNs) from robust models show an enhanced accuracy to adversarial examples compared to extracted DNNs from naturally trained (i.e. standard) models.



## **34. Triangle Attack: A Query-efficient Decision-based Adversarial Attack**

cs.CV

Accepted by ECCV 2022, code is available at  https://github.com/xiaosen-wang/TA

**SubmitDate**: 2022-07-21    [paper-pdf](http://arxiv.org/pdf/2112.06569v3)

**Authors**: Xiaosen Wang, Zeliang Zhang, Kangheng Tong, Dihong Gong, Kun He, Zhifeng Li, Wei Liu

**Abstracts**: Decision-based attack poses a severe threat to real-world applications since it regards the target model as a black box and only accesses the hard prediction label. Great efforts have been made recently to decrease the number of queries; however, existing decision-based attacks still require thousands of queries in order to generate good quality adversarial examples. In this work, we find that a benign sample, the current and the next adversarial examples can naturally construct a triangle in a subspace for any iterative attacks. Based on the law of sines, we propose a novel Triangle Attack (TA) to optimize the perturbation by utilizing the geometric information that the longer side is always opposite the larger angle in any triangle. However, directly applying such information on the input image is ineffective because it cannot thoroughly explore the neighborhood of the input sample in the high dimensional space. To address this issue, TA optimizes the perturbation in the low frequency space for effective dimensionality reduction owing to the generality of such geometric property. Extensive evaluations on ImageNet dataset show that TA achieves a much higher attack success rate within 1,000 queries and needs a much less number of queries to achieve the same attack success rate under various perturbation budgets than existing decision-based attacks. With such high efficiency, we further validate the applicability of TA on real-world API, i.e., Tencent Cloud API.



## **35. Knowledge-enhanced Black-box Attacks for Recommendations**

cs.LG

Accepted in the KDD'22

**SubmitDate**: 2022-07-21    [paper-pdf](http://arxiv.org/pdf/2207.10307v1)

**Authors**: Jingfan Chen, Wenqi Fan, Guanghui Zhu, Xiangyu Zhao, Chunfeng Yuan, Qing Li, Yihua Huang

**Abstracts**: Recent studies have shown that deep neural networks-based recommender systems are vulnerable to adversarial attacks, where attackers can inject carefully crafted fake user profiles (i.e., a set of items that fake users have interacted with) into a target recommender system to achieve malicious purposes, such as promote or demote a set of target items. Due to the security and privacy concerns, it is more practical to perform adversarial attacks under the black-box setting, where the architecture/parameters and training data of target systems cannot be easily accessed by attackers. However, generating high-quality fake user profiles under black-box setting is rather challenging with limited resources to target systems. To address this challenge, in this work, we introduce a novel strategy by leveraging items' attribute information (i.e., items' knowledge graph), which can be publicly accessible and provide rich auxiliary knowledge to enhance the generation of fake user profiles. More specifically, we propose a knowledge graph-enhanced black-box attacking framework (KGAttack) to effectively learn attacking policies through deep reinforcement learning techniques, in which knowledge graph is seamlessly integrated into hierarchical policy networks to generate fake user profiles for performing adversarial black-box attacks. Comprehensive experiments on various real-world datasets demonstrate the effectiveness of the proposed attacking framework under the black-box setting.



## **36. Image Generation Network for Covert Transmission in Online Social Network**

cs.CV

ACMMM2022 Poster

**SubmitDate**: 2022-07-21    [paper-pdf](http://arxiv.org/pdf/2207.10292v1)

**Authors**: Zhengxin You, Qichao Ying, Sheng Li, Zhenxing Qian, Xinpeng Zhang

**Abstracts**: Online social networks have stimulated communications over the Internet more than ever, making it possible for secret message transmission over such noisy channels. In this paper, we propose a Coverless Image Steganography Network, called CIS-Net, that synthesizes a high-quality image directly conditioned on the secret message to transfer. CIS-Net is composed of four modules, namely, the Generation, Adversarial, Extraction, and Noise Module. The receiver can extract the hidden message without any loss even the images have been distorted by JPEG compression attacks. To disguise the behaviour of steganography, we collected images in the context of profile photos and stickers and train our network accordingly. As such, the generated images are more inclined to escape from malicious detection and attack. The distinctions from previous image steganography methods are majorly the robustness and losslessness against diverse attacks. Experiments over diverse public datasets have manifested the superior ability of anti-steganalysis.



## **37. Switching One-Versus-the-Rest Loss to Increase the Margin of Logits for Adversarial Robustness**

cs.LG

20 pages, 16 figures

**SubmitDate**: 2022-07-21    [paper-pdf](http://arxiv.org/pdf/2207.10283v1)

**Authors**: Sekitoshi Kanai, Shin'ya Yamaguchi, Masanori Yamada, Hiroshi Takahashi, Yasutoshi Ida

**Abstracts**: Defending deep neural networks against adversarial examples is a key challenge for AI safety. To improve the robustness effectively, recent methods focus on important data points near the decision boundary in adversarial training. However, these methods are vulnerable to Auto-Attack, which is an ensemble of parameter-free attacks for reliable evaluation. In this paper, we experimentally investigate the causes of their vulnerability and find that existing methods reduce margins between logits for the true label and the other labels while keeping their gradient norms non-small values. Reduced margins and non-small gradient norms cause their vulnerability since the largest logit can be easily flipped by the perturbation. Our experiments also show that the histogram of the logit margins has two peaks, i.e., small and large logit margins. From the observations, we propose switching one-versus-the-rest loss (SOVR), which uses one-versus-the-rest loss when data have small logit margins so that it increases the margins. We find that SOVR increases logit margins more than existing methods while keeping gradient norms small and outperforms them in terms of the robustness against Auto-Attack.



## **38. FOCUS: Fairness via Agent-Awareness for Federated Learning on Heterogeneous Data**

cs.LG

**SubmitDate**: 2022-07-21    [paper-pdf](http://arxiv.org/pdf/2207.10265v1)

**Authors**: Wenda Chu, Chulin Xie, Boxin Wang, Linyi Li, Lang Yin, Han Zhao, Bo Li

**Abstracts**: Federated learning (FL) provides an effective paradigm to train machine learning models over distributed data with privacy protection. However, recent studies show that FL is subject to various security, privacy, and fairness threats due to the potentially malicious and heterogeneous local agents. For instance, it is vulnerable to local adversarial agents who only contribute low-quality data, with the goal of harming the performance of those with high-quality data. This kind of attack hence breaks existing definitions of fairness in FL that mainly focus on a certain notion of performance parity. In this work, we aim to address this limitation and propose a formal definition of fairness via agent-awareness for FL (FAA), which takes the heterogeneous data contributions of local agents into account. In addition, we propose a fair FL training algorithm based on agent clustering (FOCUS) to achieve FAA. Theoretically, we prove the convergence and optimality of FOCUS under mild conditions for linear models and general convex loss functions with bounded smoothness. We also prove that FOCUS always achieves higher fairness measured by FAA compared with standard FedAvg protocol under both linear models and general convex loss functions. Empirically, we evaluate FOCUS on four datasets, including synthetic data, images, and texts under different settings, and we show that FOCUS achieves significantly higher fairness based on FAA while maintaining similar or even higher prediction accuracy compared with FedAvg.



## **39. Illusionary Attacks on Sequential Decision Makers and Countermeasures**

cs.AI

**SubmitDate**: 2022-07-20    [paper-pdf](http://arxiv.org/pdf/2207.10170v1)

**Authors**: Tim Franzmeyer, João F. Henriques, Jakob N. Foerster, Philip H. S. Torr, Adel Bibi, Christian Schroeder de Witt

**Abstracts**: Autonomous intelligent agents deployed to the real-world need to be robust against adversarial attacks on sensory inputs. Existing work in reinforcement learning focuses on minimum-norm perturbation attacks, which were originally introduced to mimic a notion of perceptual invariance in computer vision. In this paper, we note that such minimum-norm perturbation attacks can be trivially detected by victim agents, as these result in observation sequences that are not consistent with the victim agent's actions. Furthermore, many real-world agents, such as physical robots, commonly operate under human supervisors, which are not susceptible to such perturbation attacks. As a result, we propose to instead focus on illusionary attacks, a novel form of attack that is consistent with the world model of the victim agent. We provide a formal definition of this novel attack framework, explore its characteristics under a variety of conditions, and conclude that agents must seek realism feedback to be robust to illusionary attacks.



## **40. PFMC: a parallel symbolic model checker for security protocol verification**

cs.LO

**SubmitDate**: 2022-07-20    [paper-pdf](http://arxiv.org/pdf/2207.09895v1)

**Authors**: Alex James, Alwen Tiu, Nisansala Yatapanage

**Abstracts**: We present an investigation into the design and implementation of a parallel model checker for security protocol verification that is based on a symbolic model of the adversary, where instantiations of concrete terms and messages are avoided until needed to resolve a particular assertion. We propose to build on this naturally lazy approach to parallelise this symbolic state exploration and evaluation. We utilise the concept of strategies in Haskell, which abstracts away from the low-level details of thread management and modularly adds parallel evaluation strategies (encapsulated as a monad in Haskell). We build on an existing symbolic model checker, OFMC, which is already implemented in Haskell. We show that there is a very significant speed up of around 3-5 times improvement when moving from the original single-threaded implementation of OFMC to our multi-threaded version, for both the Dolev-Yao attacker model and more general algebraic attacker models. We identify several issues in parallelising the model checker: among others, controlling growth of memory consumption, balancing lazy vs strict evaluation, and achieving an optimal granularity of parallelism.



## **41. Adaptive Image Transformations for Transfer-based Adversarial Attack**

cs.CV

34 pages, 7 figures, 11 tables. Accepted by ECCV2022

**SubmitDate**: 2022-07-20    [paper-pdf](http://arxiv.org/pdf/2111.13844v3)

**Authors**: Zheng Yuan, Jie Zhang, Shiguang Shan

**Abstracts**: Adversarial attacks provide a good way to study the robustness of deep learning models. One category of methods in transfer-based black-box attack utilizes several image transformation operations to improve the transferability of adversarial examples, which is effective, but fails to take the specific characteristic of the input image into consideration. In this work, we propose a novel architecture, called Adaptive Image Transformation Learner (AITL), which incorporates different image transformation operations into a unified framework to further improve the transferability of adversarial examples. Unlike the fixed combinational transformations used in existing works, our elaborately designed transformation learner adaptively selects the most effective combination of image transformations specific to the input image. Extensive experiments on ImageNet demonstrate that our method significantly improves the attack success rates on both normally trained models and defense models under various settings.



## **42. On the Robustness of Quality Measures for GANs**

cs.LG

Accepted at the European Conference in Computer Vision (ECCV 2022)

**SubmitDate**: 2022-07-20    [paper-pdf](http://arxiv.org/pdf/2201.13019v2)

**Authors**: Motasem Alfarra, Juan C. Pérez, Anna Frühstück, Philip H. S. Torr, Peter Wonka, Bernard Ghanem

**Abstracts**: This work evaluates the robustness of quality measures of generative models such as Inception Score (IS) and Fr\'echet Inception Distance (FID). Analogous to the vulnerability of deep models against a variety of adversarial attacks, we show that such metrics can also be manipulated by additive pixel perturbations. Our experiments indicate that one can generate a distribution of images with very high scores but low perceptual quality. Conversely, one can optimize for small imperceptible perturbations that, when added to real world images, deteriorate their scores. We further extend our evaluation to generative models themselves, including the state of the art network StyleGANv2. We show the vulnerability of both the generative model and the FID against additive perturbations in the latent space. Finally, we show that the FID can be robustified by simply replacing the standard Inception with a robust Inception. We validate the effectiveness of the robustified metric through extensive experiments, showing it is more robust against manipulation.



## **43. On the Versatile Uses of Partial Distance Correlation in Deep Learning**

cs.CV

**SubmitDate**: 2022-07-20    [paper-pdf](http://arxiv.org/pdf/2207.09684v1)

**Authors**: Xingjian Zhen, Zihang Meng, Rudrasis Chakraborty, Vikas Singh

**Abstracts**: Comparing the functional behavior of neural network models, whether it is a single network over time or two (or more networks) during or post-training, is an essential step in understanding what they are learning (and what they are not), and for identifying strategies for regularization or efficiency improvements. Despite recent progress, e.g., comparing vision transformers to CNNs, systematic comparison of function, especially across different networks, remains difficult and is often carried out layer by layer. Approaches such as canonical correlation analysis (CCA) are applicable in principle, but have been sparingly used so far. In this paper, we revisit a (less widely known) from statistics, called distance correlation (and its partial variant), designed to evaluate correlation between feature spaces of different dimensions. We describe the steps necessary to carry out its deployment for large scale models -- this opens the door to a surprising array of applications ranging from conditioning one deep model w.r.t. another, learning disentangled representations as well as optimizing diverse models that would directly be more robust to adversarial attacks. Our experiments suggest a versatile regularizer (or constraint) with many advantages, which avoids some of the common difficulties one faces in such analyses. Code is at https://github.com/zhenxingjian/Partial_Distance_Correlation.



## **44. Detecting Textual Adversarial Examples through Randomized Substitution and Vote**

cs.CL

Accepted by UAI 2022, code is avaliable at  https://github.com/JHL-HUST/RSV

**SubmitDate**: 2022-07-20    [paper-pdf](http://arxiv.org/pdf/2109.05698v2)

**Authors**: Xiaosen Wang, Yifeng Xiong, Kun He

**Abstracts**: A line of work has shown that natural text processing models are vulnerable to adversarial examples. Correspondingly, various defense methods are proposed to mitigate the threat of textual adversarial examples, eg, adversarial training, input transformations, detection, etc. In this work, we treat the optimization process for synonym substitution based textual adversarial attacks as a specific sequence of word replacement, in which each word mutually influences other words. We identify that we could destroy such mutual interaction and eliminate the adversarial perturbation by randomly substituting a word with its synonyms. Based on this observation, we propose a novel textual adversarial example detection method, termed Randomized Substitution and Vote (RS&V), which votes the prediction label by accumulating the logits of k samples generated by randomly substituting the words in the input text with synonyms. The proposed RS&V is generally applicable to any existing neural networks without modification on the architecture or extra training, and it is orthogonal to prior work on making the classification network itself more robust. Empirical evaluations on three benchmark datasets demonstrate that our RS&V could detect the textual adversarial examples more successfully than the existing detection methods while maintaining the high classification accuracy on benign samples.



## **45. Diversified Adversarial Attacks based on Conjugate Gradient Method**

cs.LG

Proceedings of the 39th International Conference on Machine Learning  (ICML 2022)

**SubmitDate**: 2022-07-19    [paper-pdf](http://arxiv.org/pdf/2206.09628v2)

**Authors**: Keiichiro Yamamura, Haruki Sato, Nariaki Tateiwa, Nozomi Hata, Toru Mitsutake, Issa Oe, Hiroki Ishikura, Katsuki Fujisawa

**Abstracts**: Deep learning models are vulnerable to adversarial examples, and adversarial attacks used to generate such examples have attracted considerable research interest. Although existing methods based on the steepest descent have achieved high attack success rates, ill-conditioned problems occasionally reduce their performance. To address this limitation, we utilize the conjugate gradient (CG) method, which is effective for this type of problem, and propose a novel attack algorithm inspired by the CG method, named the Auto Conjugate Gradient (ACG) attack. The results of large-scale evaluation experiments conducted on the latest robust models show that, for most models, ACG was able to find more adversarial examples with fewer iterations than the existing SOTA algorithm Auto-PGD (APGD). We investigated the difference in search performance between ACG and APGD in terms of diversification and intensification, and define a measure called Diversity Index (DI) to quantify the degree of diversity. From the analysis of the diversity using this index, we show that the more diverse search of the proposed method remarkably improves its attack success rate.



## **46. Towards Robust Multivariate Time-Series Forecasting: Adversarial Attacks and Defense Mechanisms**

cs.LG

**SubmitDate**: 2022-07-19    [paper-pdf](http://arxiv.org/pdf/2207.09572v1)

**Authors**: Linbo Liu, Youngsuk Park, Trong Nghia Hoang, Hilaf Hasson, Jun Huan

**Abstracts**: As deep learning models have gradually become the main workhorse of time series forecasting, the potential vulnerability under adversarial attacks to forecasting and decision system accordingly has emerged as a main issue in recent years. Albeit such behaviors and defense mechanisms started to be investigated for the univariate time series forecasting, there are still few studies regarding the multivariate forecasting which is often preferred due to its capacity to encode correlations between different time series. In this work, we study and design adversarial attack on multivariate probabilistic forecasting models, taking into consideration attack budget constraints and the correlation architecture between multiple time series. Specifically, we investigate a sparse indirect attack that hurts the prediction of an item (time series) by only attacking the history of a small number of other items to save attacking cost. In order to combat these attacks, we also develop two defense strategies. First, we adopt randomized smoothing to multivariate time series scenario and verify its effectiveness via empirical experiments. Second, we leverage a sparse attacker to enable end-to-end adversarial training that delivers robust probabilistic forecasters. Extensive experiments on real dataset confirm that our attack schemes are powerful and our defend algorithms are more effective compared with other baseline defense mechanisms.



## **47. Increasing the Cost of Model Extraction with Calibrated Proof of Work**

cs.CR

Published as a conference paper at ICLR 2022 (Spotlight - 5% of  submitted papers)

**SubmitDate**: 2022-07-19    [paper-pdf](http://arxiv.org/pdf/2201.09243v2)

**Authors**: Adam Dziedzic, Muhammad Ahmad Kaleem, Yu Shen Lu, Nicolas Papernot

**Abstracts**: In model extraction attacks, adversaries can steal a machine learning model exposed via a public API by repeatedly querying it and adjusting their own model based on obtained predictions. To prevent model stealing, existing defenses focus on detecting malicious queries, truncating, or distorting outputs, thus necessarily introducing a tradeoff between robustness and model utility for legitimate users. Instead, we propose to impede model extraction by requiring users to complete a proof-of-work before they can read the model's predictions. This deters attackers by greatly increasing (even up to 100x) the computational effort needed to leverage query access for model extraction. Since we calibrate the effort required to complete the proof-of-work to each query, this only introduces a slight overhead for regular users (up to 2x). To achieve this, our calibration applies tools from differential privacy to measure the information revealed by a query. Our method requires no modification of the victim model and can be applied by machine learning practitioners to guard their publicly exposed models against being easily stolen.



## **48. Assaying Out-Of-Distribution Generalization in Transfer Learning**

cs.LG

**SubmitDate**: 2022-07-19    [paper-pdf](http://arxiv.org/pdf/2207.09239v1)

**Authors**: Florian Wenzel, Andrea Dittadi, Peter Vincent Gehler, Carl-Johann Simon-Gabriel, Max Horn, Dominik Zietlow, David Kernert, Chris Russell, Thomas Brox, Bernt Schiele, Bernhard Schölkopf, Francesco Locatello

**Abstracts**: Since out-of-distribution generalization is a generally ill-posed problem, various proxy targets (e.g., calibration, adversarial robustness, algorithmic corruptions, invariance across shifts) were studied across different research programs resulting in different recommendations. While sharing the same aspirational goal, these approaches have never been tested under the same experimental conditions on real data. In this paper, we take a unified view of previous work, highlighting message discrepancies that we address empirically, and providing recommendations on how to measure the robustness of a model and how to improve it. To this end, we collect 172 publicly available dataset pairs for training and out-of-distribution evaluation of accuracy, calibration error, adversarial attacks, environment invariance, and synthetic corruptions. We fine-tune over 31k networks, from nine different architectures in the many- and few-shot setting. Our findings confirm that in- and out-of-distribution accuracies tend to increase jointly, but show that their relation is largely dataset-dependent, and in general more nuanced and more complex than posited by previous, smaller scale studies.



## **49. MUD-PQFed: Towards Malicious User Detection in Privacy-Preserving Quantized Federated Learning**

cs.CR

13 pages,13 figures

**SubmitDate**: 2022-07-19    [paper-pdf](http://arxiv.org/pdf/2207.09080v1)

**Authors**: Hua Ma, Qun Li, Yifeng Zheng, Zhi Zhang, Xiaoning Liu, Yansong Gao, Said F. Al-Sarawi, Derek Abbott

**Abstracts**: Federated Learning (FL), a distributed machine learning paradigm, has been adapted to mitigate privacy concerns for customers. Despite their appeal, there are various inference attacks that can exploit shared-plaintext model updates to embed traces of customer private information, leading to serious privacy concerns. To alleviate this privacy issue, cryptographic techniques such as Secure Multi-Party Computation and Homomorphic Encryption have been used for privacy-preserving FL. However, such security issues in privacy-preserving FL are poorly elucidated and underexplored. This work is the first attempt to elucidate the triviality of performing model corruption attacks on privacy-preserving FL based on lightweight secret sharing. We consider scenarios in which model updates are quantized to reduce communication overhead in this case, where an adversary can simply provide local parameters outside the legal range to corrupt the model. We then propose the MUD-PQFed protocol, which can precisely detect malicious clients performing attacks and enforce fair penalties. By removing the contributions of detected malicious clients, the global model utility is preserved to be comparable to the baseline global model without the attack. Extensive experiments validate effectiveness in maintaining baseline accuracy and detecting malicious clients in a fine-grained manner



## **50. $\ell_\infty$-Robustness and Beyond: Unleashing Efficient Adversarial Training**

cs.LG

Accepted to the 17th European Conference on Computer Vision (ECCV  2022)

**SubmitDate**: 2022-07-19    [paper-pdf](http://arxiv.org/pdf/2112.00378v2)

**Authors**: Hadi M. Dolatabadi, Sarah Erfani, Christopher Leckie

**Abstracts**: Neural networks are vulnerable to adversarial attacks: adding well-crafted, imperceptible perturbations to their input can modify their output. Adversarial training is one of the most effective approaches in training robust models against such attacks. However, it is much slower than vanilla training of neural networks since it needs to construct adversarial examples for the entire training data at every iteration, hampering its effectiveness. Recently, Fast Adversarial Training (FAT) was proposed that can obtain robust models efficiently. However, the reasons behind its success are not fully understood, and more importantly, it can only train robust models for $\ell_\infty$-bounded attacks as it uses FGSM during training. In this paper, by leveraging the theory of coreset selection, we show how selecting a small subset of training data provides a general, more principled approach toward reducing the time complexity of robust training. Unlike existing methods, our approach can be adapted to a wide variety of training objectives, including TRADES, $\ell_p$-PGD, and Perceptual Adversarial Training (PAT). Our experimental results indicate that our approach speeds up adversarial training by 2-3 times while experiencing a slight reduction in the clean and robust accuracy.



